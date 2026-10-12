"""Order-book reconstruction from recorded `book` + `price_change` events.

Step 1 of the fill-model build order: replay the recorded stream into a
ladder per outcome token, and **validate** that replay rather than trust it.
Every queue-position and markout number downstream reads this ladder, so an
unverified replay would make every later result unverifiable too.

The venue gives us two independent checks, and both are used:

  * **Snapshot agreement.** A `book` event is a full ladder. When one arrives
    for an asset we have been replaying, the replayed ladder is compared to
    it level by level *before* re-syncing to it.
  * **Top-of-book agreement.** Every `price_change` entry carries the venue's
    own `best_bid`/`best_ask`. That is a check at every update, not just at
    snapshots. One subtlety, found by running this on recorded data: the
    venue stamps every event in one WS message with the best bid/ask after
    the *whole message*. A crossing order arrives as two events (the new
    ask, then the consumed bid removed), and the first already reports the
    final touch. So the check runs once per message -- consecutive frames
    sharing receive and venue timestamps -- not once per event.

State is only trusted between a snapshot and the next connection boundary.
After a reconnect (or any gap) an asset is *unsynced* until its next `book`
arrives: changes to unsynced assets are counted and dropped, never applied to
a guessed ladder. This is the same rule the fill model must follow -- book
state across a gap is unknown, not unchanged.

All prices and sizes are `Decimal`, parsed from the venue's strings, so
"agreement" means exact equality rather than float tolerance.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

Side = Literal["bid", "ask"]
Levels = dict[Decimal, Decimal]

_SIDES: dict[str, Side] = {"BUY": "bid", "SELL": "ask"}


# --- parsed events -----------------------------------------------------------


@dataclass(frozen=True)
class BookSnapshot:
    asset_id: str
    recv_ts: datetime
    venue_ts: int
    bids: Levels
    asks: Levels


@dataclass(frozen=True)
class LevelChange:
    """One level set to an absolute size (0 removes it)."""

    asset_id: str
    side: Side
    price: Decimal
    size: Decimal
    best_bid: Decimal | None
    best_ask: Decimal | None


@dataclass(frozen=True)
class ChangeFrame:
    """One `price_change` event: possibly several assets, applied together."""

    recv_ts: datetime
    venue_ts: int
    changes: tuple[LevelChange, ...]


@dataclass(frozen=True)
class Reset:
    """A connection boundary: every asset's state becomes unknown."""

    recv_ts: datetime
    reason: str


ReplayEvent = BookSnapshot | ChangeFrame | Reset


def _dec(raw: Any) -> Decimal | None:
    if raw is None or raw == "":
        return None
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        return None


def _levels(raw: Any) -> Levels:
    out: Levels = {}
    for level in raw or []:
        price, size = _dec(level.get("price")), _dec(level.get("size"))
        if price is not None and size is not None and size > 0:
            out[price] = size
    return out


def parse_event(
    event_type: str, payload: str, recv_ts: datetime, venue_ts: int
) -> BookSnapshot | ChangeFrame | None:
    """Turn one recorded row into a replay event (None for other types)."""
    if event_type not in ("book", "price_change"):
        return None
    data = json.loads(payload)
    if event_type == "book":
        return BookSnapshot(
            asset_id=str(data.get("asset_id") or ""),
            recv_ts=recv_ts,
            venue_ts=venue_ts,
            bids=_levels(data.get("bids")),
            asks=_levels(data.get("asks")),
        )
    changes: list[LevelChange] = []
    for c in data.get("price_changes") or []:
        side = _SIDES.get(str(c.get("side")))
        price, size = _dec(c.get("price")), _dec(c.get("size"))
        if side is None or price is None or size is None:
            continue
        changes.append(
            LevelChange(
                asset_id=str(c.get("asset_id") or ""),
                side=side,
                price=price,
                size=size,
                best_bid=_dec(c.get("best_bid")),
                best_ask=_dec(c.get("best_ask")),
            )
        )
    return ChangeFrame(recv_ts=recv_ts, venue_ts=venue_ts, changes=tuple(changes))


# --- the ladder --------------------------------------------------------------


@dataclass
class Ladder:
    bids: Levels = field(default_factory=dict)
    asks: Levels = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snap: BookSnapshot) -> Ladder:
        return cls(bids=dict(snap.bids), asks=dict(snap.asks))

    def set_level(self, side: Side, price: Decimal, size: Decimal) -> None:
        book = self.bids if side == "bid" else self.asks
        if size > 0:
            book[price] = size
        else:
            book.pop(price, None)

    @property
    def best_bid(self) -> Decimal | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return min(self.asks) if self.asks else None

    def diff(self, snap: BookSnapshot) -> list[tuple[Side, Decimal, Decimal, Decimal]]:
        """(side, price, replayed_size, snapshot_size) for every disagreeing level."""
        out: list[tuple[Side, Decimal, Decimal, Decimal]] = []
        zero = Decimal(0)
        for side, mine, theirs in (("bid", self.bids, snap.bids), ("ask", self.asks, snap.asks)):
            for price in sorted(mine.keys() | theirs.keys()):
                a, b = mine.get(price, zero), theirs.get(price, zero)
                if a != b:
                    out.append((side, price, a, b))  # type: ignore[arg-type]
        return out


# --- replay + validation -----------------------------------------------------


@dataclass
class Mismatch:
    asset_id: str
    recv_ts: datetime
    kind: Literal["snapshot", "top"]
    detail: str


@dataclass
class ReplayStats:
    resets: int = 0
    snapshots: int = 0
    frames: int = 0
    changes_applied: int = 0
    # Deltas for assets with no trusted state (before their first snapshot
    # after a reset). Dropped, never applied to a guessed ladder.
    changes_unsynced: int = 0

    snapshot_checks: int = 0
    snapshot_matches: int = 0
    # Of the snapshot mismatches, how many disagree at the touch -- the levels
    # a maker's fill depends on -- rather than only deep in the book.
    snapshot_top_mismatches: int = 0

    top_checks: int = 0
    top_matches: int = 0

    mismatches_by_asset: Counter[str] = field(default_factory=Counter)
    examples: list[Mismatch] = field(default_factory=list)

    @property
    def snapshot_agreement(self) -> float:
        return self.snapshot_matches / self.snapshot_checks if self.snapshot_checks else 1.0

    @property
    def top_agreement(self) -> float:
        return self.top_matches / self.top_checks if self.top_checks else 1.0

    def summary(self) -> dict[str, float | int]:
        return {
            "resets": self.resets,
            "snapshots": self.snapshots,
            "frames": self.frames,
            "changes_applied": self.changes_applied,
            "changes_unsynced": self.changes_unsynced,
            "snapshot_checks": self.snapshot_checks,
            "snapshot_agreement": round(self.snapshot_agreement, 6),
            "snapshot_top_mismatches": self.snapshot_top_mismatches,
            "top_checks": self.top_checks,
            "top_agreement": round(self.top_agreement, 6),
            "assets_with_mismatch": len(self.mismatches_by_asset),
        }


class Replayer:
    """Apply a stream of replay events, checking every claim the venue makes."""

    def __init__(self, *, max_examples: int = 20) -> None:
        self.ladders: dict[str, Ladder] = {}
        self.stats = ReplayStats()
        self.max_examples = max_examples
        # Top-of-book checks deferred to the end of the current WS message.
        self._message: tuple[datetime, int] | None = None
        self._pending: dict[str, LevelChange] = {}

    def _record(self, m: Mismatch) -> None:
        self.stats.mismatches_by_asset[m.asset_id] += 1
        if len(self.stats.examples) < self.max_examples:
            self.stats.examples.append(m)

    def apply(self, event: ReplayEvent) -> None:
        key = (event.recv_ts, event.venue_ts) if isinstance(event, ChangeFrame) else None
        if key is None or key != self._message:
            self.finish()
        if isinstance(event, Reset):
            self.ladders.clear()
            self.stats.resets += 1
        elif isinstance(event, BookSnapshot):
            self._on_book(event)
        else:
            self._message = key
            self._on_frame(event)

    def finish(self) -> None:
        """Run the top-of-book checks for the message just completed."""
        recv_ts = self._message[0] if self._message else None
        for asset_id, change in self._pending.items():
            ladder = self.ladders.get(asset_id)
            if ladder is None or recv_ts is None:
                continue
            self.stats.top_checks += 1
            ours = (ladder.best_bid, ladder.best_ask)
            theirs = (change.best_bid, change.best_ask)
            if _top_equal(ours, theirs):
                self.stats.top_matches += 1
            else:
                self._record(
                    Mismatch(
                        asset_id,
                        recv_ts,
                        "top",
                        f"replayed bid/ask {ours[0]}/{ours[1]} vs venue {theirs[0]}/{theirs[1]}",
                    )
                )
        self._pending = {}
        self._message = None

    def run(self, events: Iterable[ReplayEvent]) -> ReplayStats:
        for event in events:
            self.apply(event)
        self.finish()
        return self.stats

    def _on_book(self, snap: BookSnapshot) -> None:
        self.stats.snapshots += 1
        current = self.ladders.get(snap.asset_id)
        if current is not None:
            self.stats.snapshot_checks += 1
            diff = current.diff(snap)
            if not diff:
                self.stats.snapshot_matches += 1
            else:
                truth = Ladder.from_snapshot(snap)
                top_differs = (current.best_bid, current.best_ask) != (
                    truth.best_bid,
                    truth.best_ask,
                )
                if top_differs:
                    self.stats.snapshot_top_mismatches += 1
                shown = ", ".join(f"{s}@{p}: {a}->{b}" for s, p, a, b in diff[:4])
                self._record(
                    Mismatch(
                        snap.asset_id,
                        snap.recv_ts,
                        "snapshot",
                        f"{len(diff)} levels differ (top {'differs' if top_differs else 'ok'}):"
                        f" {shown}",
                    )
                )
        # Re-sync either way: a known error must not propagate.
        self.ladders[snap.asset_id] = Ladder.from_snapshot(snap)

    def _on_frame(self, frame: ChangeFrame) -> None:
        self.stats.frames += 1
        for change in frame.changes:
            ladder = self.ladders.get(change.asset_id)
            if ladder is None:
                self.stats.changes_unsynced += 1
                continue
            ladder.set_level(change.side, change.price, change.size)
            self.stats.changes_applied += 1
            self._pending[change.asset_id] = change


def _top_equal(
    ours: tuple[Decimal | None, Decimal | None], theirs: tuple[Decimal | None, Decimal | None]
) -> bool:
    """Venue reports an empty side as 0 (bid) or 1 (ask); treat those as None."""
    bid_t = None if theirs[0] in (None, Decimal(0)) else theirs[0]
    ask_t = None if theirs[1] in (None, Decimal(1)) else theirs[1]
    return ours[0] == bid_t and ours[1] == ask_t


def with_resets(
    events: Iterable[tuple[datetime, BookSnapshot | ChangeFrame]],
    *,
    boundaries: Iterable[datetime] = (),
    silence_s: float = 20.0,
) -> Iterator[ReplayEvent]:
    """Interleave `Reset`s into a time-ordered event stream.

    Two sources of boundary, because the coverage ledger only exists for
    recent sessions:

      * explicit `boundaries` -- coverage segment `connect_ts` values;
      * any global silence of `silence_s` or more. Across ~120 subscribed
        assets the stream is never quiet that long while connected, so a
        silence that long is a disconnect the ledger did not record.
    """
    pending = sorted(boundaries)
    i = 0
    prev: datetime | None = None
    for ts, event in events:
        while i < len(pending) and pending[i] <= ts:
            yield Reset(pending[i], "coverage")
            i += 1
        if prev is not None and (ts - prev).total_seconds() >= silence_s:
            yield Reset(ts, "silence")
        prev = ts
        yield event
