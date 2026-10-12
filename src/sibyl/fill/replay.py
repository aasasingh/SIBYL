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

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import duckdb

from sibyl.fill.ladder import (
    BookSnapshot,
    ChangeFrame,
    Replayer,
    ReplayStats,
    parse_event,
    with_resets,
)
from sibyl.record.coverage import _connect as _coverage_connect


def _rows(
    events_dir: Path, *, start: datetime | None, end: datetime | None, batch: int = 20_000
) -> Iterator[tuple[datetime, BookSnapshot | ChangeFrame]]:
    if not any(events_dir.glob("part-*.parquet")):
        return
    pattern = (events_dir / "part-*.parquet").as_posix()
    where = ["event_type IN ('book', 'price_change')"]
    params: list[datetime] = []
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
                event = parse_event(event_type, payload, recv_ts, int(venue_ts))
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
) -> ReplayStats:
    """Replay everything recorded under `root` and report agreement."""
    stream = with_resets(
        _rows(root / "book_events", start=start, end=end),
        boundaries=coverage_boundaries(root),
        silence_s=silence_s,
    )
    return Replayer(max_examples=max_examples).run(stream)
