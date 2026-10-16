"""Markout-harness tests: sign conventions, unknown horizons, clustering."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sibyl.cli import _markout
from sibyl.fill.ladder import BookSnapshot, ChangeFrame, LevelChange, Replayer, Reset
from sibyl.fill.markout import MidTracker, mark_out, markout_recorded, summarise_markouts
from sibyl.fill.queue import Fill, Order, OrderState
from sibyl.record.writer import EVENTS_SCHEMA

D = Decimal
T0 = datetime(2026, 10, 16, 12, tzinfo=UTC)


def _ts(ms: int) -> datetime:
    return T0 + timedelta(milliseconds=ms)


def _book(asset: str = "A", bid: str = "0.40", ask: str = "0.44", ts: int = 1) -> BookSnapshot:
    return BookSnapshot(asset, _ts(ts), ts, {D(bid): D(100)}, {D(ask): D(100)})


def _set(r: Replayer, ts: int, side: str, price: str, size: str, asset: str = "A") -> None:
    change = LevelChange(asset, side, D(price), D(size), None, None)  # type: ignore[arg-type]
    r.apply(ChangeFrame(_ts(ts), ts, (change,)))


def _filled(
    side: str, price: str, ts: int, mid: str, asset: str = "A", size: str = "10"
) -> OrderState:
    order = Order(f"{asset}-{side}-{ts}", asset, side, D(size), 0, 10**9)  # type: ignore[arg-type]
    s = OrderState(order, status="done", price=D(price), filled=D(size))
    s.fills.append(Fill(order.order_id, ts, D(size), D(price), "queue", False, D(mid)))
    return s


class TestMidTracker:
    def test_mid_follows_the_book(self) -> None:
        r = Replayer()
        t = MidTracker(r)
        r.apply(_book())
        _set(r, 1000, "bid", "0.42", "5")
        r.finish()
        assert t.mid_at("A", 500) == pytest.approx(0.42)
        assert t.mid_at("A", 1000) == pytest.approx(0.43)
        assert t.mid_at("A", -1) is None

    def test_one_sided_book_has_no_mid(self) -> None:
        r = Replayer()
        t = MidTracker(r)
        r.apply(_book())
        _set(r, 1000, "ask", "0.44", "0")
        assert t.mid_at("A", 2000) is None

    def test_reset_makes_mid_unknown_and_marks_a_gap(self) -> None:
        r = Replayer()
        t = MidTracker(r)
        r.apply(_book())
        _set(r, 1000, "bid", "0.41", "5")
        r.apply(Reset(_ts(1500), "coverage"))
        assert t.mid_at("A", 1200) is None
        assert t.gap_between(0, 2000)
        assert not t.gap_between(1001, 5000)


class TestMarkouts:
    def _tracker(self) -> MidTracker:
        r = Replayer()
        t = MidTracker(r)
        r.apply(_book())  # mid 0.42
        _set(r, 60_000, "bid", "0.30", "100")  # nothing at mid until now ...
        _set(r, 60_000, "bid", "0.40", "0")  # ... mid falls to 0.37
        _set(r, 400_000, "ask", "0.50", "1")  # clock past every 5-min horizon
        return t

    def test_a_bid_that_was_run_over_marks_out_negative(self) -> None:
        marks, _ = mark_out([_filled("bid", "0.40", 0, "0.42")], self._tracker(), horizons_s=[60])
        (m,) = marks
        assert m.spread_c == pytest.approx(2.0)  # bought 2c under mid
        assert m.drift_c == pytest.approx(-5.0)  # then mid fell 5c
        assert m.markout_c == pytest.approx(-3.0)

    def test_an_ask_gains_when_the_mid_falls(self) -> None:
        marks, _ = mark_out([_filled("ask", "0.44", 0, "0.42")], self._tracker(), horizons_s=[60])
        assert marks[0].markout_c == pytest.approx(7.0)

    def test_rebate_is_the_us_schedule(self) -> None:
        marks, _ = mark_out([_filled("bid", "0.50", 0, "0.51")], self._tracker(), horizons_s=[60])
        assert marks[0].rebate_c == pytest.approx(0.3125)

    def test_horizons_past_the_data_or_across_a_gap_are_unmarked(self) -> None:
        t = self._tracker()
        marks, unmarked = mark_out([_filled("bid", "0.40", 0, "0.42")], t, horizons_s=[60, 3600])
        assert [m.horizon_s for m in marks] == [60]
        assert unmarked == {3600: 1}
        t.replayer.apply(Reset(_ts(400_001), "silence"))
        t.replayer.apply(_book(ts=500_000))
        _, unmarked = mark_out([_filled("bid", "0.40", 350_000, "0.37")], t, horizons_s=[60])
        assert unmarked == {60: 1}


class TestSummary:
    def test_weighted_mean_and_clustered_se(self) -> None:
        r = Replayer()
        t = MidTracker(r)
        for asset in ("A", "B", "C"):
            r.apply(_book(asset))
        _set(r, 100_000, "ask", "0.50", "1")
        fills = [
            _filled("bid", "0.40", 0, "0.42", "A", "10"),  # +2c
            _filled("bid", "0.38", 0, "0.42", "B", "30"),  # +4c
            _filled("ask", "0.44", 0, "0.42", "C", "10"),  # +2c
        ]
        marks, _ = mark_out(fills, t, horizons_s=[60])
        s = summarise_markouts(marks, 60)
        assert s is not None
        assert (s.fills, s.markets, s.shares) == (3, 3, 50.0)
        assert s.markout_c == pytest.approx(3.2)  # (20 + 120 + 20) / 50
        assert s.drift_c == pytest.approx(0.0)
        assert s.markout_se_c > 0
        assert s.net_ci95_c[0] < s.net_c < s.net_ci95_c[1]

    def test_both_legs_of_a_market_are_one_cluster(self) -> None:
        r = Replayer()
        t = MidTracker(r)
        r.apply(_book("YES"))
        r.apply(_book("NO", "0.56", "0.60"))
        _set(r, 100_000, "ask", "0.50", "1", "YES")
        fills = [_filled("bid", "0.40", 0, "0.42", "YES"), _filled("bid", "0.56", 0, "0.58", "NO")]
        marks, _ = mark_out(fills, t, mirror={"YES": "NO", "NO": "YES"}, horizons_s=[60])
        s = summarise_markouts(marks, 60)
        assert s is not None and s.markets == 1

    def test_no_rows_no_summary(self) -> None:
        assert summarise_markouts([], 60) is None


def _recording(tmp_path: Path) -> int:
    """Three hours of one book, with one ask landing on the bid probe at 3000 s."""
    events = tmp_path / "book_events"
    events.mkdir()
    t0 = int(T0.timestamp() * 1000)
    (events / "manifest-s-v1.json").write_text(
        json.dumps(
            {"started": T0.isoformat(), "assets": ["A"], "markets": [{"tokens": ["A", "B"]}]}
        )
    )

    def row(ms: int, payload: dict[str, object]) -> dict[str, object]:
        return {
            "recv_ts": _ts(ms),
            "venue_ts": t0 + ms,
            "event_type": payload["event_type"],
            "asset_id": "A",
            "market": "m",
            "payload": json.dumps({**payload, "timestamp": str(t0 + ms)}),
        }

    book: dict[str, object] = {
        "event_type": "book",
        "asset_id": "A",
        "bids": [{"price": "0.40", "size": "100"}],
        "asks": [{"price": "0.45", "size": "100"}],
    }
    cross: dict[str, object] = {
        "event_type": "price_change",
        "price_changes": [
            {"asset_id": "A", "price": "0.40", "size": "5", "side": "SELL"},
        ],
    }
    rows = [row(s * 1000, book) for s in range(0, 3 * 3600, 10)]
    rows.append(row(3_010_000, cross))  # an ask lands on the probe bid placed at 3000 s
    rows.sort(key=lambda r: r["venue_ts"])  # type: ignore[arg-type, return-value]
    pq.write_table(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA), events / "part-a.parquet")
    return t0


class TestChunkedRun:
    def test_every_probe_is_simulated_once_and_marked(self, tmp_path: Path) -> None:
        t0 = _recording(tmp_path)
        results = markout_recorded(tmp_path, chunk_h=0.5, warmup_s=600, horizons_s=(60,))
        states = results["pessimistic"].states
        ids = [s.order.order_id for s in states]
        assert len(ids) == len(set(ids))  # chunks never overlap
        places = sorted({s.order.place_ts - t0 for s in states})
        assert places == list(range(600_000, places[-1] + 1, 300_000))
        (m,) = results["pessimistic"].marks
        assert m.fill.kind == "cross" and m.fill.venue_ts == t0 + 3_010_000
        # Bought at 0.40; the next snapshot puts the mid back at 0.425.
        assert m.markout_c == pytest.approx(2.5)

    def test_cli_writes_a_report(self, tmp_path: Path) -> None:
        _recording(tmp_path)
        out = tmp_path / "report.json"
        args = argparse.Namespace(
            data_dir=tmp_path,
            start=None,
            end=None,
            every_s=300.0,
            life_s=300.0,
            latency_ms=250,
            chunk_h=0.5,
            json_out=out,
        )
        assert _markout(args) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["variants"]["pessimistic"]["all_60s"]["fills"] == 1
