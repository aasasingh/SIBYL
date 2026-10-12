"""Coverage-ledger tests.

The fill model's correctness depends on knowing where book state was
unknown. A gap that goes unrecorded becomes a gap the simulator interpolates
across, filling orders at prices that may never have existed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


def _rows(root: Path) -> list[tuple[object, ...]]:
    import duckdb

    pattern = (root / "coverage" / "*.parquet").as_posix()
    with duckdb.connect() as con:
        return con.execute(
            "SELECT session_id, segment_id, connect_ts, disconnect_ts, end_reason, n_events "
            f"FROM read_parquet('{pattern}') ORDER BY connect_ts"
        ).fetchall()


class TestCrashDurability:
    """A segment must exist on disk while it is live, not only after close.

    The motivating incident: a session that ran 22:46 -> 04:08 left no
    coverage record at all, because the process died without running its
    `finally` and the segment was only ever written on close.
    """

    def test_mark_open_writes_a_zero_length_open_segment(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=1, n_assets=120)
        w.mark_open(seg)
        [(_, _, connect, disconnect, reason, _)] = _rows(tmp_path)
        assert reason == "open"
        # Nothing is claimed until a heartbeat confirms data is on disk.
        assert connect == disconnect
        assert coverage_summary(tmp_path)["unclosed"] == 1

    def test_heartbeat_extends_the_same_segment(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=1, n_assets=120)
        w.mark_open(seg)
        w.heartbeat(seg, n_events=5)
        w.heartbeat(seg, n_events=12)
        rows = _rows(tmp_path)
        assert len(rows) == 1, "heartbeats rewrite the segment, never append"
        _, _, connect, disconnect, reason, n_events = rows[0]
        assert reason == "open"
        assert disconnect > connect  # type: ignore[operator]
        assert n_events == 12

    def test_close_finalises_the_open_segment(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=1, n_assets=120)
        w.mark_open(seg)
        w.heartbeat(seg, n_events=3)
        w.close_segment(seg, reason="refresh", n_events=9)
        [(_, _, _, _, reason, n_events)] = _rows(tmp_path)
        assert (reason, n_events) == ("refresh", 9)
        assert coverage_summary(tmp_path)["unclosed"] == 0

    def test_a_crash_keeps_coverage_up_to_the_last_heartbeat(self, tmp_path: Path) -> None:
        dead = CoverageWriter(tmp_path, session_id="dead")
        seg = dead.open_segment(assets_version=1, n_assets=120)
        dead.mark_open(seg)
        dead.heartbeat(seg, n_events=100)
        # Hard crash: no close_segment. A new process starts later.
        del dead
        alive = CoverageWriter(tmp_path, session_id="alive")
        alive.mark_open(alive.open_segment(assets_version=1, n_assets=120))

        rows = _rows(tmp_path)
        assert [r[0] for r in rows] == ["dead", "alive"]
        last_beat = rows[0][3]
        gaps = find_gaps(tmp_path)
        assert len(gaps) == 1
        # The gap starts at the last heartbeat -- not lost, not overclaimed.
        assert gaps[0][0] == last_beat
        assert gaps[0][1] == rows[1][2]
        s = coverage_summary(tmp_path)
        assert s["segments"] == 2
        assert s["events"] == 100

    def test_writes_leave_no_temp_files(self, tmp_path: Path) -> None:
        w = CoverageWriter(tmp_path)
        seg = w.open_segment(assets_version=1, n_assets=1)
        w.mark_open(seg)
        w.heartbeat(seg)
        w.close_segment(seg, reason="shutdown")
        assert sorted(p.name for p in w.dir.iterdir()) == [f"seg-{w.session_id}-000000.parquet"]


class TestRecorderHeartbeat:
    async def test_live_segment_is_on_disk_and_heartbeats_through_silence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive the real recorder loop against a fake socket that sends one
        event and then goes silent: the segment must be written while the
        connection is still up, and keep advancing with no traffic."""
        import asyncio
        import contextlib
        import json

        import sibyl.record.recorder as recorder

        class FakeWS:
            def __init__(self) -> None:
                self.sent = False

            async def __aenter__(self) -> FakeWS:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def send(self, msg: str) -> None:
                return None

            async def recv(self) -> str:
                if not self.sent:
                    self.sent = True
                    return json.dumps(
                        {"event_type": "book", "asset_id": "a1", "market": "m", "timestamp": "1"}
                    )
                await asyncio.Event().wait()  # silent forever
                raise AssertionError

        class FakeClient:
            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

        async def fake_select(*_: object, **__: object) -> tuple[list[str], list[dict[str, str]]]:
            return ["a1", "a2"], [{"market_id": "m"}]

        monkeypatch.setattr("websockets.connect", lambda *a, **k: FakeWS())
        monkeypatch.setattr(recorder, "PolymarketReadClient", FakeClient)
        monkeypatch.setattr(recorder, "select_assets", fake_select)

        task = asyncio.create_task(recorder.record(data_dir=tmp_path, heartbeat_seconds=0.1))
        try:
            await asyncio.sleep(0.25)
            first = _rows(tmp_path)
            await asyncio.sleep(0.3)
            second = _rows(tmp_path)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert len(first) == len(second) == 1
        assert first[0][4] == second[0][4] == "open"
        assert second[0][5] == 1
        # Advanced during pure silence.
        assert second[0][3] > first[0][3]  # type: ignore[operator]
        # The claimed coverage is backed by flushed rows.
        assert any((tmp_path / "book_events").glob("part-*.parquet"))
