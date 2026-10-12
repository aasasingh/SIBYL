"""Queue simulator: would a hypothetical resting order have filled?

Step 3 of the fill-model build order. A simulated order rides along the
replay as a listener: it never touches the real book (the data is passive),
it only tracks how much visible size sits *ahead* of it at its level and
decides, from attributed trades, when the queue in front has been consumed.

Rules (research/fill-model-design.md), and where each is enforced:

  1. **Join the back.** Queue-ahead at activation = full visible size at the
     level. (`_activate`)
  2. **Trades consume from the front.** Traded volume at our level reduces
     queue-ahead first; only the excess fills us. (`_consume`)
  3. **Partial fills are partial.** (`_consume`)
  4. **A sweep with less volume than queue-ahead does not fill.** Trades at
     prices *through* ours (a taker who would have hit us first) go through
     the same front-first consumption. (`_on_change`)
  5. **Latency is non-zero** for placement *and* cancellation: an order can
     still be filled during its cancel latency. (`_advance`)
  6. **Fills are at our limit**, never better.

Two things are not knowable from the data, so they are parameters with the
pessimistic setting as default and the alternatives reported beside it:

  * `cancel_rule` -- when size at our level is cancelled, was it ahead of
    us or behind? "back" (pessimistic: behind us first, we don't advance),
    "pro_rata", or "front" (optimistic).
  * `ambiguous_as` -- a decrease attribution could not cleanly resolve
    (12.4% of traded notional in the first data) is treated as a "cancel"
    (pessimistic: no consumption) or a "trade".

And `touch=True` is the naive comparison the design doc warns about: fill
the whole order whenever anything trades at or through its price, no queue.
If a strategy only works under `touch`, that *is* the finding.

The local clock cannot measure latency on this machine: receive-minus-venue
drifts ~1 ms/min and stepped from ~0.4 s to ~1.0 s across a restart, i.e.
it is clock offset, not network delay. `latency_ms` is therefore an
assumption, to be swept.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from sibyl.fill.attribution import Attributed
from sibyl.fill.ladder import BookSnapshot, ChangeFrame, LevelChange, Replayer, Reset, Side

CancelRule = Literal["back", "pro_rata", "front"]
AmbiguousAs = Literal["cancel", "trade"]
FillKind = Literal["queue", "through", "cross", "touch"]
Status = Literal["pending", "live", "done", "voided", "unplaceable"]
ZERO = Decimal(0)

AttributionKey = tuple[str, Side, Decimal, int, Decimal]


@dataclass(frozen=True)
class Order:
    order_id: str
    asset_id: str
    side: Side
    size: Decimal
    place_ts: int  # venue ms: when the decision is made
    cancel_ts: int  # venue ms: when the cancel is sent
    price: Decimal | None = None  # None: join the touch on `side` at activation


@dataclass(frozen=True)
class Fill:
    order_id: str
    venue_ts: int
    size: Decimal
    price: Decimal
    kind: FillKind
    ambiguous: bool
    mid: Decimal | None  # after the update that filled us; step 4 marks out from it


@dataclass
class OrderState:
    order: Order
    status: Status = "pending"
    price: Decimal | None = None
    queue_ahead: Decimal = ZERO
    queue_at_join: Decimal = ZERO
    filled: Decimal = ZERO
    reason: str = ""
    fills: list[Fill] = field(default_factory=list)

    @property
    def remaining(self) -> Decimal:
        return self.order.size - self.filled


def attribution_index(attributed: Iterable[Attributed]) -> dict[AttributionKey, list[Attributed]]:
    """Look up a replayed decrease's attribution by its identity."""
    index: dict[AttributionKey, list[Attributed]] = defaultdict(list)
    for a in attributed:
        d = a.decrease
        index[(d.asset_id, d.side, d.price, d.venue_ts, d.amount)].append(a)
    return index


def _through(side: Side, level: Decimal, ours: Decimal) -> bool:
    """Is `level` strictly worse for a maker than our price (a taker went past us)?"""
    return level < ours if side == "bid" else level > ours


def _crosses(side: Side, opposite_price: Decimal, ours: Decimal) -> bool:
    return opposite_price <= ours if side == "bid" else opposite_price >= ours


class QueueSimulator:
    def __init__(
        self,
        replayer: Replayer,
        orders: Iterable[Order],
        attribution: Mapping[AttributionKey, list[Attributed]],
        *,
        latency_ms: int = 250,
        cancel_rule: CancelRule = "back",
        ambiguous_as: AmbiguousAs = "cancel",
        touch: bool = False,
    ) -> None:
        self.replayer = replayer
        self.states = [OrderState(o) for o in orders]
        self._pending = sorted(self.states, key=lambda s: s.order.place_ts, reverse=True)
        self._live: dict[str, list[OrderState]] = defaultdict(list)
        self._next_expiry: int | None = None
        # Each simulator consumes its own copy: lookups pop as they match.
        self._attribution = {k: list(v) for k, v in attribution.items()}
        self.latency_ms = latency_ms
        self.cancel_rule = cancel_rule
        self.ambiguous_as = ambiguous_as
        self.touch = touch
        replayer.listeners.append(self._on_change)
        replayer.snapshot_listeners.append(self._on_snapshot)
        replayer.reset_listeners.append(self._on_reset)

    # --- clock ---------------------------------------------------------------

    def _advance(self, now: int) -> None:
        while self._pending and self._pending[-1].order.place_ts + self.latency_ms <= now:
            self._activate(self._pending.pop())
        if self._next_expiry is None or now < self._next_expiry:
            return
        self._next_expiry = None
        for asset in list(self._live):
            keep = []
            for s in self._live[asset]:
                expiry = s.order.cancel_ts + self.latency_ms
                if s.status == "live" and expiry <= now:
                    s.status, s.reason = "done", "cancelled"
                if s.status == "live":
                    keep.append(s)
                    self._note_expiry(expiry)
            self._live[asset] = keep

    def _note_expiry(self, expiry: int) -> None:
        if self._next_expiry is None or expiry < self._next_expiry:
            self._next_expiry = expiry

    def _activate(self, s: OrderState) -> None:
        ladder = self.replayer.ladders.get(s.order.asset_id)
        if ladder is None:
            s.status, s.reason = "unplaceable", "book state unknown"
            return
        price = s.order.price
        if price is None:
            price = ladder.best_bid if s.order.side == "bid" else ladder.best_ask
        if price is None:
            s.status, s.reason = "unplaceable", "empty side"
            return
        opposite = ladder.best_ask if s.order.side == "bid" else ladder.best_bid
        if opposite is not None and _crosses(s.order.side, opposite, price):
            s.status, s.reason = "unplaceable", "would cross (taker, not maker)"
            return
        book = ladder.bids if s.order.side == "bid" else ladder.asks
        s.price = price
        s.queue_ahead = s.queue_at_join = book.get(price, ZERO)
        s.status = "live"
        self._live[s.order.asset_id].append(s)
        self._note_expiry(s.order.cancel_ts + self.latency_ms)

    def finish(self) -> None:
        for s in self.states:
            if s.status == "pending":
                s.status, s.reason = "unplaceable", "after end of data"
            elif s.status == "live":
                s.status, s.reason = "done", "end of data"

    # --- replay hooks --------------------------------------------------------

    def _on_reset(self, reset: Reset) -> None:
        # Book state across a gap is unknown: never simulate through it.
        for states in self._live.values():
            for s in states:
                s.status, s.reason = "voided", f"gap ({reset.reason})"
        self._live.clear()

    def _on_snapshot(self, snap: BookSnapshot) -> None:
        for s in self._live.get(snap.asset_id, []):
            book = snap.bids if s.order.side == "bid" else snap.asks
            assert s.price is not None
            s.queue_ahead = min(s.queue_ahead, book.get(s.price, ZERO))

    def _attributed(self, change: LevelChange, amount: Decimal, ts: int) -> tuple[Decimal, bool]:
        hits = self._attribution.get((change.asset_id, change.side, change.price, ts, amount))
        if not hits:
            return ZERO, False
        a = hits.pop(0)
        ambiguous = a.kind in ("mixed", "contested")
        if ambiguous and self.ambiguous_as == "cancel":
            return ZERO, True
        return a.traded, ambiguous

    def _on_change(self, change: LevelChange, before: Decimal, frame: ChangeFrame) -> None:
        self._advance(frame.venue_ts)
        live = self._live.get(change.asset_id)
        if not live:
            return
        decrease = before - change.size
        traded, ambiguous = (
            self._attributed(change, decrease, frame.venue_ts) if decrease > 0 else (ZERO, False)
        )
        for s in live:
            if s.status != "live":
                continue
            assert s.price is not None
            if change.side != s.order.side:
                # A new opposite order at or through our price would have
                # matched us as taker. Only a real increase counts.
                if change.size > before and _crosses(s.order.side, change.price, s.price):
                    kind: FillKind = "touch" if self.touch else "cross"
                    self._fill(s, min(s.remaining, change.size - before), frame, kind, False)
                continue
            at_level = change.price == s.price
            through = _through(s.order.side, change.price, s.price)
            if not (at_level or through) or decrease <= 0:
                continue
            if self.touch:
                if traded > 0:
                    self._fill(s, s.remaining, frame, "touch", ambiguous)
                continue
            if traded > 0:
                self._consume(s, traded, frame, "queue" if at_level else "through", ambiguous)
            if at_level:
                self._cancel(s, decrease - traded, level_after=change.size)

    # --- queue arithmetic ----------------------------------------------------

    def _consume(
        self, s: OrderState, traded: Decimal, frame: ChangeFrame, kind: FillKind, ambiguous: bool
    ) -> None:
        excess = traded - s.queue_ahead
        s.queue_ahead = max(ZERO, s.queue_ahead - traded)
        if excess > 0:
            self._fill(s, min(s.remaining, excess), frame, kind, ambiguous)

    def _cancel(self, s: OrderState, cancelled: Decimal, *, level_after: Decimal) -> None:
        if cancelled > 0:
            level_before = level_after + cancelled
            if self.cancel_rule == "front":
                s.queue_ahead -= min(cancelled, s.queue_ahead)
            elif self.cancel_rule == "pro_rata" and level_before > 0:
                s.queue_ahead -= cancelled * s.queue_ahead / level_before
        # Whatever the rule, we cannot have more ahead of us than is visible.
        s.queue_ahead = max(ZERO, min(s.queue_ahead, level_after))

    def _fill(
        self, s: OrderState, size: Decimal, frame: ChangeFrame, kind: FillKind, ambiguous: bool
    ) -> None:
        if size <= 0:
            return
        assert s.price is not None
        ladder = self.replayer.ladders.get(s.order.asset_id)
        mid = None
        if ladder and ladder.best_bid is not None and ladder.best_ask is not None:
            mid = (ladder.best_bid + ladder.best_ask) / 2
        s.fills.append(Fill(s.order.order_id, frame.venue_ts, size, s.price, kind, ambiguous, mid))
        s.filled += size
        if s.remaining <= 0:
            s.status, s.reason = "done", "filled"


@dataclass(frozen=True)
class SimSummary:
    orders: int
    placed: int
    unplaceable: int
    voided: int
    any_fill: int
    full_fill: int
    fill_rate: float  # orders with any fill / placed
    filled_share: float  # filled size / placed size
    ambiguous_fill_share: float  # of filled size
    median_queue_at_join: float

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def summarise(states: Iterable[OrderState]) -> SimSummary:
    states = list(states)
    placed = [s for s in states if s.status in ("done", "voided")]
    filled_size = sum((s.filled for s in placed), ZERO)
    placed_size = sum((s.order.size for s in placed), ZERO)
    ambiguous_size = sum((f.size for s in placed for f in s.fills if f.ambiguous), ZERO)
    queues = sorted(float(s.queue_at_join) for s in placed)
    return SimSummary(
        orders=len(states),
        placed=len(placed),
        unplaceable=sum(1 for s in states if s.status == "unplaceable"),
        voided=sum(1 for s in states if s.status == "voided"),
        any_fill=sum(1 for s in placed if s.filled > 0),
        full_fill=sum(1 for s in placed if s.remaining <= 0),
        fill_rate=round(sum(1 for s in placed if s.filled > 0) / len(placed), 6) if placed else 0.0,
        filled_share=round(float(filled_size / placed_size), 6) if placed_size else 0.0,
        ambiguous_fill_share=round(float(ambiguous_size / filled_size), 6) if filled_size else 0.0,
        median_queue_at_join=queues[len(queues) // 2] if queues else 0.0,
    )
