"""Coverage ledger: when was book state actually known?

A fill model that replays recorded events must know where the holes are. If
the recorder was disconnected between 03:11 and 03:14, the book during that
window is *unknown*, not *unchanged* — and a simulator that silently
interpolates will happily fill orders across the gap, at prices that may
never have existed.

So every connection writes a segment: when it opened, when it closed, and
why. The gaps between consecutive segments are, by construction, the
intervals with no coverage. `find_gaps` reads them back.

One subtlety worth stating: a segment boundary is necessary but not
sufficient for "state known". On reconnect the venue re-sends a full `book`
snapshot per asset, but those arrive over some seconds, and a quiet asset may
not be re-synced until much later. So a segment's opening moments are
coverage for the *connection*, not yet for every asset. `asset_resync_times`
gives the per-asset truth.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

COVERAGE_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),  # one recorder process
        ("segment_id", pa.int64()),  # one WS connection within the session
        ("assets_version", pa.int64()),  # which subscription refresh
        ("connect_ts", pa.timestamp("us", tz="UTC")),
        ("disconnect_ts", pa.timestamp("us", tz="UTC")),
        ("duration_s", pa.float64()),
        ("end_reason", pa.string()),  # refresh | error | shutdown | deadline
        ("end_detail", pa.string()),
        ("n_assets", pa.int64()),
        ("n_events", pa.int64()),
    ]
)


@dataclass
class Segment:
    session_id: str
    segment_id: int
    assets_version: int
    connect_ts: datetime
    n_assets: int
    n_events: int = 0


class CoverageWriter:
    """One Parquet part per closed segment.

    Written on close rather than buffered, because the whole point is to
    survive the process dying: a buffered coverage record would be the first
    thing lost in exactly the crash it is meant to document.
    """

    def __init__(self, root: Path, session_id: str | None = None) -> None:
        self.dir = root / "coverage"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._next_segment = 0

    def open_segment(self, *, assets_version: int, n_assets: int) -> Segment:
        segment = Segment(
            session_id=self.session_id,
            segment_id=self._next_segment,
            assets_version=assets_version,
            connect_ts=datetime.now(UTC),
            n_assets=n_assets,
        )
        self._next_segment += 1
        return segment

    def close_segment(
        self, segment: Segment, *, reason: str, detail: str = "", n_events: int | None = None
    ) -> None:
        disconnect = datetime.now(UTC)
        if n_events is not None:
            segment.n_events = n_events
        table = pa.Table.from_pylist(
            [
                {
                    "session_id": segment.session_id,
                    "segment_id": segment.segment_id,
                    "assets_version": segment.assets_version,
                    "connect_ts": segment.connect_ts,
                    "disconnect_ts": disconnect,
                    "duration_s": (disconnect - segment.connect_ts).total_seconds(),
                    "end_reason": reason,
                    "end_detail": detail[:500],
                    "n_assets": segment.n_assets,
                    "n_events": segment.n_events,
                }
            ],
            schema=COVERAGE_SCHEMA,
        )
        name = f"seg-{segment.session_id}-{segment.segment_id:06d}.parquet"
        pq.write_table(table, self.dir / name)


def _connect(root: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    pattern = (root / "coverage" / "*.parquet").as_posix()
    if any((root / "coverage").glob("*.parquet")):
        con.execute(f"CREATE VIEW coverage AS SELECT * FROM read_parquet('{pattern}')")
    else:
        con.register("_empty", pa.Table.from_pylist([], schema=COVERAGE_SCHEMA))
        con.execute("CREATE VIEW coverage AS SELECT * FROM _empty")
    return con


def find_gaps(root: Path, *, min_seconds: float = 0.0) -> list[tuple[datetime, datetime, float]]:
    """(gap_start, gap_end, seconds) for every interval with no connection.

    Computed across ALL sessions ordered by time, so a recorder restart shows
    up as a gap exactly like a dropped connection — which from the fill
    model's point of view it is.
    """
    with _connect(root) as con:
        rows = con.execute(
            """
            WITH ordered AS (
                SELECT connect_ts, disconnect_ts,
                       lag(disconnect_ts) OVER (ORDER BY connect_ts) AS prev_end
                FROM coverage
            )
            SELECT prev_end, connect_ts,
                   epoch(connect_ts - prev_end) AS gap_s
            FROM ordered
            WHERE prev_end IS NOT NULL
              AND connect_ts > prev_end
            ORDER BY prev_end
            """
        ).fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows if float(r[2]) >= min_seconds]


def coverage_summary(root: Path) -> dict[str, float | int]:
    """Totals for a quick health check."""
    with _connect(root) as con:
        row = con.execute(
            """
            SELECT count(*) AS segments,
                   count(DISTINCT session_id) AS sessions,
                   coalesce(sum(duration_s), 0) AS connected_s,
                   coalesce(sum(n_events), 0) AS events,
                   min(connect_ts) AS first_ts,
                   max(disconnect_ts) AS last_ts
            FROM coverage
            """
        ).fetchone()
    assert row is not None
    segments, sessions, connected_s, events, first_ts, last_ts = row
    gaps = find_gaps(root)
    wall = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0.0
    return {
        "segments": int(segments),
        "sessions": int(sessions),
        "connected_s": float(connected_s),
        "events": int(events),
        "wall_s": float(wall),
        "gap_count": len(gaps),
        "gap_total_s": float(sum(g[2] for g in gaps)),
        # Fraction of the observed window during which we held a connection.
        "uptime": float(connected_s / wall) if wall > 0 else 1.0,
    }


def asset_resync_times(root: Path, events_dir: Path) -> list[tuple[str, datetime, datetime]]:
    """(asset_id, segment_connect, first_book_after_connect) per segment.

    The delay between a connection opening and an asset's first full `book`
    snapshot is the window where that asset's ladder was genuinely unknown,
    even though the connection was up.
    """
    with _connect(root) as con:
        pattern = (events_dir / "*.parquet").as_posix()
        if not any(events_dir.glob("*.parquet")):
            return []
        con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{pattern}')")
        rows = con.execute(
            """
            SELECT e.asset_id, c.connect_ts, min(e.recv_ts) AS first_book
            FROM coverage c
            JOIN events e
              ON e.recv_ts >= c.connect_ts AND e.recv_ts <= c.disconnect_ts
            WHERE e.event_type = 'book' AND e.asset_id <> ''
            GROUP BY e.asset_id, c.connect_ts
            ORDER BY c.connect_ts, e.asset_id
            """
        ).fetchall()
    return [(str(r[0]), r[1], r[2]) for r in rows]
