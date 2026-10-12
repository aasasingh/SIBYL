"""Coverage-ledger tests.

The fill model's correctness depends on knowing where book state was
unknown. A gap that goes unrecorded becomes a gap the simulator interpolates
across, filling orders at prices that may never have existed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sibyl.record.coverage import (
    CoverageWriter,
    Segment,
    coverage_summary,
    find_gaps,
)


def _segment(
    writer: CoverageWriter,
    *,
    start: datetime,
    seconds: float,
    reason: str = "error",
    n_events: int = 10,
) -> None:
    """Write a closed segment with explicit timestamps."""
    seg = Segment(
        session_id=writer.session_id,
        segment_id=writer._next_segment,
        assets_version=1,
        connect_ts=start,
        n_assets=120,
    )
    writer._next_segment += 1
    # close_segment stamps "now", so write the row directly for determinism.
    import pyarrow as pa
    import pyarrow.parquet as pq

    from sibyl.record.coverage import COVERAGE_SCHEMA

    end = start + timedelta(seconds=seconds)
    table = pa.Table.from_pylist(
        [
            {
                "session_id": seg.session_id,
                "segment_id": seg.segment_id,
                "assets_version": 1,
                "connect_ts": start,
                "disconnect_ts": end,
                "duration_s": seconds,
                "end_reason": reason,
                "end_detail": "",
                "n_assets": 120,
                "n_events": n_events,
            }
        ],
        schema=COVERAGE_SCHEMA,
    )
    pq.write_table(table, writer.dir / f"seg-{seg.session_id}-{seg.segment_id:06d}.parquet")


BASE = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


class TestGapDetection:
    def test_contiguous_segments_have_no_gap(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        _segment(w, start=BASE, seconds=600)
        _segment(w, start=BASE + timedelta(seconds=600), seconds=600)
        assert find_gaps(tmp_path) == []

    def test_a_disconnect_is_reported_as_a_gap(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        _segment(w, start=BASE, seconds=600)
        # 90 seconds of darkness before reconnecting
        _segment(w, start=BASE + timedelta(seconds=690), seconds=600)
        gaps = find_gaps(tmp_path)
        assert len(gaps) == 1
        start, end, seconds = gaps[0]
        assert seconds == 90.0
        assert start == BASE + timedelta(seconds=600)
        assert end == BASE + timedelta(seconds=690)

    def test_a_process_restart_is_also_a_gap(self, tmp_path: Path) -> None:
        """Gaps span sessions: from the fill model's view a restart and a
        dropped connection are the same thing."""
        first = CoverageWriter(tmp_path, session_id="aaa")
        _segment(first, start=BASE, seconds=300, reason="shutdown")
        second = CoverageWriter(tmp_path, session_id="bbb")
        _segment(second, start=BASE + timedelta(hours=3), seconds=300)
        gaps = find_gaps(tmp_path)
        assert len(gaps) == 1
        assert gaps[0][2] == 3 * 3600 - 300

    def test_min_seconds_filters_trivial_gaps(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        _segment(w, start=BASE, seconds=600)
        _segment(w, start=BASE + timedelta(seconds=600.5), seconds=600)
        assert find_gaps(tmp_path, min_seconds=1.0) == []
        assert len(find_gaps(tmp_path, min_seconds=0.1)) == 1

    def test_no_coverage_yet_is_not_an_error(self, tmp_path: Path) -> None:
        assert find_gaps(tmp_path) == []
        summary = coverage_summary(tmp_path)
        assert summary["segments"] == 0


class TestSummary:
    def test_uptime_accounts_for_gaps(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        _segment(w, start=BASE, seconds=900, n_events=100)
        _segment(w, start=BASE + timedelta(seconds=1800), seconds=900, n_events=50)
        s = coverage_summary(tmp_path)
        assert s["segments"] == 2
        assert s["connected_s"] == 1800.0
        assert s["wall_s"] == 2700.0
        assert s["gap_count"] == 1
        assert s["gap_total_s"] == 900.0
        assert abs(float(s["uptime"]) - 2 / 3) < 1e-9
        assert s["events"] == 150

    def test_perfect_uptime(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        _segment(w, start=BASE, seconds=600)
        _segment(w, start=BASE + timedelta(seconds=600), seconds=600)
        s = coverage_summary(tmp_path)
        assert s["gap_count"] == 0
        assert abs(float(s["uptime"]) - 1.0) < 1e-9


class TestWriterRoundTrip:
    def test_open_and_close_writes_a_readable_segment(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=2, n_assets=42)
        w.close_segment(seg, reason="refresh", detail="scheduled", n_events=7)
        s = coverage_summary(tmp_path)
        assert s["segments"] == 1
        assert s["events"] == 7
        assert s["gap_count"] == 0

    def test_segment_ids_increment(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        a = w.open_segment(assets_version=1, n_assets=1)
        b = w.open_segment(assets_version=1, n_assets=1)
        assert b.segment_id == a.segment_id + 1

    def test_close_records_the_reason(self, tmp_path: Path) -> None:
        import duckdb

        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=1, n_assets=1)
        w.close_segment(seg, reason="error", detail="ConnectionClosed: 1006")
        pattern = (tmp_path / "coverage" / "*.parquet").as_posix()
        with duckdb.connect() as con:
            reason, detail = con.execute(
                f"SELECT end_reason, end_detail FROM read_parquet('{pattern}')"
            ).fetchone()  # type: ignore[misc]
        assert reason == "error"
        assert "ConnectionClosed" in detail
