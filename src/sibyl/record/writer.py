"""Append-only raw event store for recorded market data.

Events are stored LOSSLESSLY: the venue's payload goes to disk as the exact
JSON string it arrived as, alongside our own receive clock. That is
deliberate. This data is time-locked -- a book update not captured tonight
can never be captured -- and we do not yet know precisely what the fill model
will need from it. Normalising on ingest would mean discovering a missing
field months of recording too late.

Derived tables (top-of-book series, reconstructed ladders) are built from
this on read, never the other way round.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EVENTS_SCHEMA = pa.schema(
    [
        # Our clock on receipt. Latency against venue_ts is itself a
        # measurement the fill model needs, so both are kept.
        ("recv_ts", pa.timestamp("us", tz="UTC")),
        ("venue_ts", pa.int64()),  # venue timestamp, milliseconds, as sent
        ("event_type", pa.string()),  # book | price_change | last_trade_price | ...
        ("asset_id", pa.string()),  # ERC-1155 token id (one outcome leg)
        ("market", pa.string()),  # condition id
        ("payload", pa.string()),  # raw JSON, verbatim
    ]
)


@dataclass
class EventRow:
    recv_ts: datetime
    venue_ts: int
    event_type: str
    asset_id: str
    market: str
    payload: str


class EventWriter:
    """Buffers events and flushes Parquet parts.

    A part is written per flush and never rewritten, so a killed recorder
    keeps everything it had already flushed.
    """

    def __init__(self, root: Path, *, flush_every: int = 2000) -> None:
        self.dir = root / "book_events"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer: list[EventRow] = []
        self.rows_written = 0
        self.parts_written = 0

    def add(self, row: EventRow) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> int:
        if not self._buffer:
            return 0
        table = pa.Table.from_pylist(
            [
                {
                    "recv_ts": r.recv_ts,
                    "venue_ts": r.venue_ts,
                    "event_type": r.event_type,
                    "asset_id": r.asset_id,
                    "market": r.market,
                    "payload": r.payload,
                }
                for r in self._buffer
            ],
            schema=EVENTS_SCHEMA,
        )
        name = f"part-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(table, self.dir / name, compression="zstd")
        n = len(self._buffer)
        self._buffer = []
        self.rows_written += n
        self.parts_written += 1
        return n

    @property
    def buffered(self) -> int:
        return len(self._buffer)
