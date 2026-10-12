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

import os
import time
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
        ("end_reason", pa.string()),  # refresh | error | shutdown | deadline | open
        ("end_detail", pa.string()),
        ("n_assets", pa.int64()),
        ("n_events", pa.int64()),
    ]
)

# A segment whose connection is up (or whose process died without closing it).
OPEN = "open"


@dataclass
class Segment:
    session_id: str
    segment_id: int
    assets_version: int
    connect_ts: datetime
    n_assets: int
    n_events: int = 0


class CoverageWriter:
    """One Parquet file per segment, rewritten in place as the segment lives.

    A segment that is only written on close is lost in exactly the crash it
    is meant to document: a hard kill, power loss or laptop sleep runs no
    `finally`. So the file is written three ways:

      * `mark_open` on connect -- `end_reason="open"`, zero length;
      * `heartbeat` about every minute -- still "open", `disconnect_ts`
        advanced to now;
      * `close_segment` on a clean exit -- the real reason.

    A segment still reading "open" after its session has died therefore
    claims coverage only up to its last heartbeat. Callers flush event data
    *before* each heartbeat, so the claimed interval is always backed by
    rows on disk; the up-to-a-minute tail before the crash is reported as a
    gap, which is the conservative direction.

    Every write goes to a temp file then `os.replace`, so a crash mid-write
    leaves the previous version intact rather than a truncated Parquet file
    that would break every reader of the ledger.
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

    def mark_open(self, segment: Segment) -> None:
        """Record that the connection is up, before any data is claimed."""
        self._write(segment, end=segment.connect_ts, reason=OPEN)

    def heartbeat(self, segment: Segment, *, n_events: int | None = None) -> None:
        """Extend an open segment's claimed coverage to now.

        Flush event data first: the heartbeat asserts that state up to this
        instant is on disk.
        """
        if n_events is not None:
            segment.n_events = n_events
        self._write(segment, end=datetime.now(UTC), reason=OPEN)

    def close_segment(
        self, segment: Segment, *, reason: str, detail: str = "", n_events: int | None = None
    ) -> None:
        if n_events is not None:
            segment.n_events = n_events
        self._write(segment, end=datetime.now(UTC), reason=reason, detail=detail)

    def _write(self, segment: Segment, *, end: datetime, reason: str, detail: str = "") -> None:
        table = pa.Table.from_pylist(
            [
                {
                    "session_id": segment.session_id,
                    "segment_id": segment.segment_id,
                    "assets_version": segment.assets_version,
                    "connect_ts": segment.connect_ts,
                    "disconnect_ts": end,
                    "duration_s": (end - segment.connect_ts).total_seconds(),
                    "end_reason": reason,
                    "end_detail": detail[:500],
                    "n_assets": segment.n_assets,
                    "n_events": segment.n_events,
                }
            ],
            schema=COVERAGE_SCHEMA,
        )
        name = f"seg-{segment.session_id}-{segment.segment_id:06d}.parquet"
        # The temp suffix keeps it out of the `*.parquet` glob readers use.
        tmp = self.dir / f"{name}.tmp"
        pq.write_table(table, tmp)
        _replace(tmp, self.dir / name)


def _replace(src: Path, dst: Path, *, attempts: int = 5) -> None:
    """`os.replace`, retried: on Windows it fails while a reader (e.g. a
    concurrent `sibyl coverage`) holds the destination open."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.2 * (i + 1))


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
                   max(disconnect_ts) AS last_ts,
                   count(*) FILTER (WHERE end_reason = 'open') AS unclosed
            FROM coverage
            """
        ).fetchone()
    assert row is not None
    segments, sessions, connected_s, events, first_ts, last_ts, unclosed = row
    gaps = find_gaps(root)
    wall = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0.0
    return {
        "segments": int(segments),
        "sessions": int(sessions),
        # Segments never closed: the live one, plus one per hard crash. A
        # crashed segment's coverage ends at its last heartbeat.
        "unclosed": int(unclosed),
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
