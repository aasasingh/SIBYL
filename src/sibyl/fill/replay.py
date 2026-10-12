"""Read recorded events off disk and replay them through the validator.

Ordering is by *receive* time, then file and row order -- the sequence a live
system would actually have seen. Rows within one WS frame share a receive
timestamp and sit consecutively in one part file, so the tiebreak preserves
frame order.

Duplicates are dropped by payload hash. Two recorders running at once (as
during a hand-over to new code) record the same venue events twice; a
duplicated book delta is harmless, but a duplicated trade would later be
counted as twice the consumption.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from sibyl.fill.attribution import (
    Attributed,
    AttributionStats,
    Decrease,
    Trade,
    attribute,
    parse_trade,
)
from sibyl.fill.ladder import (
    BookSnapshot,
    ChangeFrame,
    LevelChange,
    Replayer,
    ReplayStats,
    parse_event,
    with_resets,
)
from sibyl.record.coverage import _connect as _coverage_connect


def raw_rows(
    events_dir: Path,
    *,
    event_types: tuple[str, ...],
    start: datetime | None = None,
    end: datetime | None = None,
    batch: int = 20_000,
) -> Iterator[tuple[datetime, int, str, str]]:
    """(recv_ts, venue_ts, event_type, payload), deduped, in receive order."""
    if not any(events_dir.glob("part-*.parquet")):
        return
    pattern = (events_dir / "part-*.parquet").as_posix()
    where = [f"event_type IN ({', '.join('?' for _ in event_types)})"]
    params: list[object] = list(event_types)
    if start is not None:
        where.append("recv_ts >= ?")
        params.append(start)
    if end is not None:
        where.append("recv_ts < ?")
        params.append(end)
    with duckdb.connect() as con:
        con.execute(
            f"""
            SELECT recv_ts, venue_ts, event_type, payload
            FROM read_parquet('{pattern}', filename = true, file_row_number = true)
            WHERE {" AND ".join(where)}
            QUALIFY row_number() OVER (
                PARTITION BY md5(payload) ORDER BY recv_ts, filename, file_row_number
            ) = 1
            ORDER BY recv_ts, filename, file_row_number
            """,
            params,
        )
        while chunk := con.fetchmany(batch):
            for recv_ts, venue_ts, event_type, payload in chunk:
                yield recv_ts, int(venue_ts), event_type, payload


def _rows(
    events_dir: Path, *, start: datetime | None, end: datetime | None
) -> Iterator[tuple[datetime, BookSnapshot | ChangeFrame]]:
    for recv_ts, venue_ts, event_type, payload in raw_rows(
        events_dir, event_types=("book", "price_change"), start=start, end=end
    ):
        event = parse_event(event_type, payload, recv_ts, venue_ts)
        if event is not None:
            yield recv_ts, event


def coverage_boundaries(root: Path) -> list[datetime]:
    with _coverage_connect(root) as con:
        return [r[0] for r in con.execute("SELECT connect_ts FROM coverage").fetchall()]


def replay(
    root: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    silence_s: float = 20.0,
    max_examples: int = 20,
    replayer: Replayer | None = None,
) -> ReplayStats:
    """Replay everything recorded under `root` and report agreement.

    Pass a `replayer` with listeners attached to observe the level history.
    """
    stream = with_resets(
        _rows(root / "book_events", start=start, end=end),
        boundaries=coverage_boundaries(root),
        silence_s=silence_s,
    )
    return (replayer or Replayer(max_examples=max_examples)).run(stream)


def mirror_map(root: Path) -> dict[str, str]:
    """YES <-> NO token pairs, from every recorder manifest ever written."""
    pairs: dict[str, str] = {}
    for path in sorted((root / "book_events").glob("manifest-*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for market in manifest.get("markets", []):
            tokens = market.get("tokens") or []
            if len(tokens) == 2:
                a, b = str(tokens[0]), str(tokens[1])
                pairs[a], pairs[b] = b, a
    return pairs


def load_trades(
    root: Path, *, start: datetime | None = None, end: datetime | None = None
) -> list[Trade]:
    trades: list[Trade] = []
    for _, _, _, payload in raw_rows(
        root / "book_events", event_types=("last_trade_price",), start=start, end=end
    ):
        trade = parse_trade(payload)
        if trade is not None:
            trades.append(trade)
    return trades


def attribute_recorded(
    root: Path,
    *,
    window_ms: int = 500,
    fallback_ms: int = 1500,
    start: datetime | None = None,
    end: datetime | None = None,
    silence_s: float = 20.0,
) -> tuple[list[Attributed], AttributionStats, ReplayStats]:
    """Replay recorded books, collect every level decrease, attribute it.

    Holds the whole range in memory (~290k decreases/day); pass `start` and
    `end` to bound it on long recordings.
    """
    decreases: list[Decrease] = []

    def collect(change: LevelChange, before: Decimal, frame: ChangeFrame) -> None:
        if change.size < before:
            decreases.append(
                Decrease(
                    asset_id=change.asset_id,
                    side=change.side,
                    price=change.price,
                    amount=before - change.size,
                    venue_ts=frame.venue_ts,
                    recv_ts=frame.recv_ts,
                )
            )

    replayer = Replayer()
    replayer.listeners.append(collect)
    replay_stats = replay(root, start=start, end=end, silence_s=silence_s, replayer=replayer)
    attributed, stats = attribute(
        decreases,
        load_trades(root, start=start, end=end),
        mirror=mirror_map(root),
        window_ms=window_ms,
        fallback_ms=fallback_ms,
    )
    return attributed, stats, replay_stats
