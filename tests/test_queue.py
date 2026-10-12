"""Queue-simulator tests.

Each test pins one rule from research/fill-model-design.md. The shape of a
test: a book is set up by snapshot, a hypothetical order joins it, and
replayed deltas plus their trade attribution decide whether it fills.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sibyl.fill.attribution import Attributed, Decrease
from sibyl.fill.ladder import BookSnapshot, ChangeFrame, LevelChange, Replayer, Reset
from sibyl.fill.queue import (
    AmbiguousAs,
    CancelRule,
    Order,
    OrderState,
    QueueSimulator,
    summarise,
)
from sibyl.fill.replay import probe_orders

D = Decimal
T0 = datetime(2026, 10, 12, 12, tzinfo=UTC)


def _ts(ms: int) -> datetime:
    return T0 + timedelta(milliseconds=ms)


class Book:
    """A replayer plus one simulator, driven by hand."""

    def __init__(
        self,
        order: Order,
        *,
        bids: dict[str, str] | None = None,
        asks: dict[str, str] | None = None,
        trades: dict[tuple[str, str, int], str] | None = None,
        ambiguous: set[tuple[str, str, int]] | None = None,
        latency_ms: int = 0,
        cancel_rule: CancelRule = "back",
        ambiguous_as: AmbiguousAs = "cancel",
        touch: bool = False,
    ) -> None:
        self.replayer = Replayer()
        self.trades = trades or {}
        self.ambiguous = ambiguous or set()
        self.replayer.apply(
            BookSnapshot(
                "A",
                _ts(0),
                0,
                {D(p): D(s) for p, s in (bids or {"0.40": "100"}).items()},
                {D(p): D(s) for p, s in (asks or {"0.45": "100"}).items()},
            )
        )
        self.sim = QueueSimulator(
            self.replayer,
            [order],
            {},
            latency_ms=latency_ms,
            cancel_rule=cancel_rule,
            ambiguous_as=ambiguous_as,
            touch=touch,
        )

    def set(self, ts: int, side: str, price: str, size: str) -> Book:
        """Apply one level update; label it from `trades` if it is a decrease."""
        ladder = self.replayer.ladders["A"]
        before = (ladder.bids if side == "bid" else ladder.asks).get(D(price), D(0))
        key = (side, price, ts)
        if D(size) < before and key in self.trades:
            d = Decrease("A", side, D(price), before - D(size), ts, _ts(ts))  # type: ignore[arg-type]
            a = Attributed(d, D(self.trades[key]), contested=key in self.ambiguous)
            self.sim._attribution[(d.asset_id, d.side, d.price, d.venue_ts, d.amount)] = [a]
        change = LevelChange("A", side, D(price), D(size), ladder.best_bid, ladder.best_ask)  # type: ignore[arg-type]
        self.replayer.apply(ChangeFrame(_ts(ts), ts, (change,)))
        return self

    def tick(self, ts: int) -> Book:
        """An unrelated update elsewhere in the book, to advance the clock."""
        return self.set(ts, "ask", "0.90", str(ts))

    @property
    def state(self) -> OrderState:
        self.sim.finish()
        return self.sim.states[0]


def bid(size: str = "10", place: int = 1, cancel: int = 10_000, price: str | None = None) -> Order:
    return Order("o", "A", "bid", D(size), place, cancel, D(price) if price else None)


class TestJoinAndConsume:
    def test_joins_the_back_of_the_visible_queue(self) -> None:
        b = Book(bid()).tick(5)
        assert b.state.queue_at_join == D(100)
        assert b.state.price == D("0.40")

    def test_trades_consume_the_queue_ahead_before_filling_us(self) -> None:
        b = Book(bid(), trades={("bid", "0.40", 20): "60", ("bid", "0.40", 30): "50"})
        b.tick(5).set(10, "bid", "0.40", "150")  # 50 joins behind us
        b.set(20, "bid", "0.40", "90")  # 60 traded: all from ahead of us
        assert b.sim.states[0].filled == 0
        assert b.sim.states[0].queue_ahead == D(40)
        b.set(30, "bid", "0.40", "40")  # 50 traded: 40 ahead, then us
        assert b.state.filled == D(10)
        assert b.state.reason == "filled"

    def test_a_taker_who_takes_only_what_is_ahead_leaves_us_unfilled(self) -> None:
        """With nobody behind us, the whole visible level is ahead of us: a
        trade can never exceed it, so we fill only once the price trades
        through. This is why the pessimistic rule fills rarely."""
        b = Book(bid(), trades={("bid", "0.40", 10): "100"})
        b.tick(5).set(10, "bid", "0.40", "0")
        assert b.state.filled == 0

    def test_excess_beyond_the_queue_fills_only_that_much(self) -> None:
        b = Book(bid("10"), bids={"0.40": "100"}, trades={("bid", "0.40", 10): "100"})
        b.tick(5)
        # Pretend the queue ahead had already shrunk to 96.
        b.sim.states[0].queue_ahead = D(96)
        b.set(10, "bid", "0.40", "0")
        assert b.state.filled == D(4)
        assert b.state.fills[0].price == D("0.40")  # rule 6: at our limit


class TestCancelRules:
    def _run(self, rule: CancelRule) -> Decimal:
        b = Book(bid(), cancel_rule=rule)
        b.tick(5)
        b.set(10, "bid", "0.40", "150")  # 50 joins behind us
        b.set(20, "bid", "0.40", "120")  # 30 cancelled: ahead or behind?
        return b.state.queue_ahead

    def test_pessimistic_assumes_cancels_were_behind_us(self) -> None:
        assert self._run("back") == D(100)

    def test_pro_rata(self) -> None:
        assert self._run("pro_rata") == D(80)

    def test_optimistic_assumes_cancels_were_ahead(self) -> None:
        assert self._run("front") == D(70)

    def test_queue_ahead_never_exceeds_visible_size(self) -> None:
        b = Book(bid(), cancel_rule="back")
        b.tick(5).set(20, "bid", "0.40", "30")
        assert b.state.queue_ahead == D(30)


class TestAmbiguity:
    def _run(self, ambiguous_as: AmbiguousAs) -> OrderState:
        key = ("bid", "0.40", 10)
        b = Book(bid(), trades={key: "110"}, ambiguous={key}, ambiguous_as=ambiguous_as)
        b.tick(5).set(10, "bid", "0.40", "0")
        return b.state

    def test_pessimistic_treats_ambiguous_volume_as_cancelled(self) -> None:
        assert self._run("cancel").filled == 0

    def test_alternative_counts_it_and_flags_the_fill(self) -> None:
        s = self._run("trade")
        assert s.filled == D(10)
        assert s.fills[0].ambiguous


class TestSweepsAndTouch:
    def test_a_sweep_through_us_with_less_than_queue_ahead_does_not_fill(self) -> None:
        b = Book(bid(), bids={"0.40": "100", "0.39": "50"}, trades={("bid", "0.39", 10): "50"})
        b.tick(5).set(10, "bid", "0.39", "0")
        assert b.state.filled == 0
        assert b.state.queue_ahead == D(50)

    def test_fill_on_touch_ignores_the_queue(self) -> None:
        """The naive rule, kept only as the labelled comparison."""
        b = Book(bid(), trades={("bid", "0.40", 10): "1"}, touch=True, ambiguous_as="trade")
        b.tick(5).set(10, "bid", "0.40", "99")
        assert b.state.filled == D(10)
        assert b.state.fills[0].kind == "touch"

    def test_an_opposite_order_at_our_price_crosses_into_us(self) -> None:
        b = Book(bid(price="0.42")).tick(5)
        b.set(10, "ask", "0.42", "7")
        assert b.state.filled == D(7)
        assert b.state.fills[0].kind == "cross"


class TestLatencyAndGaps:
    def test_an_order_is_not_live_until_placement_latency_passes(self) -> None:
        b = Book(bid(place=0), trades={("bid", "0.40", 100): "200"}, latency_ms=250)
        b.set(100, "bid", "0.40", "0")
        b.tick(300)
        # Activated after the sweep: the level is empty, queue-ahead 0, unfilled.
        assert b.state.filled == 0
        assert b.state.queue_at_join == 0

    def test_cancel_latency_still_exposes_the_order(self) -> None:
        # Alone at 0.41, so a taker trading through to 0.40 must hit us first.
        order = bid(place=0, cancel=100, price="0.41")
        b = Book(order, trades={("bid", "0.40", 300): "100"}, latency_ms=250)
        b.tick(260)  # live at 0 + 250
        b.set(300, "bid", "0.40", "0")  # cancel sent at 100, lands at 350
        assert b.state.filled == D(10)
        assert b.state.fills[0].kind == "through"

    def test_after_cancel_latency_the_order_is_gone(self) -> None:
        order = bid(place=0, cancel=100, price="0.41")
        b = Book(order, trades={("bid", "0.40", 400): "100"}, latency_ms=250)
        b.tick(260).set(400, "bid", "0.40", "0")
        assert b.state.filled == 0
        assert b.state.reason == "cancelled"

    def test_a_gap_voids_live_orders(self) -> None:
        b = Book(bid()).tick(5)
        b.replayer.apply(Reset(_ts(6), "coverage"))
        assert b.state.status == "voided"

    def test_orders_on_unknown_books_are_unplaceable(self) -> None:
        r = Replayer()
        sim = QueueSimulator(r, [bid(place=0)], {}, latency_ms=0)
        r.apply(BookSnapshot("OTHER", _ts(0), 0, {D("0.5"): D(1)}, {}))
        r.apply(ChangeFrame(_ts(1), 1, (LevelChange("OTHER", "bid", D("0.5"), D(2), None, None),)))
        sim.finish()
        assert sim.states[0].status == "unplaceable"

    def test_a_crossing_limit_is_a_taker_not_a_maker(self) -> None:
        b = Book(bid(price="0.45")).tick(5)
        assert b.state.status == "unplaceable"
        assert "cross" in b.state.reason


class TestSummaryAndProbes:
    def test_summary_counts(self) -> None:
        b = Book(bid(), trades={("bid", "0.40", 20): "110"})
        b.tick(5).set(10, "bid", "0.40", "150").set(20, "bid", "0.40", "40")
        s = summarise([b.state])
        assert (s.placed, s.any_fill, s.full_fill, s.fill_rate) == (1, 1, 1, 1.0)
        assert s.median_queue_at_join == 100.0

    def test_probes_follow_the_manifest_in_force(self, tmp_path: Path) -> None:
        events = tmp_path / "book_events"
        events.mkdir()
        for name, started, assets in (
            ("v1", T0, ["old"]),
            ("v2", T0 + timedelta(hours=1), ["new"]),
        ):
            (events / f"manifest-s-{name}.json").write_text(
                json.dumps({"started": started.isoformat(), "assets": assets})
            )
        t0 = int(T0.timestamp() * 1000)
        orders = probe_orders(
            tmp_path, start_ms=t0, end_ms=t0 + 2 * 3600_000, every_s=1800, life_s=600
        )
        assets_by_time = sorted({(o.place_ts - t0, o.asset_id) for o in orders})
        assert assets_by_time == [
            (1800_000, "old"),
            (3600_000, "new"),
            (5400_000, "new"),
        ]
        assert {o.side for o in orders} == {"bid", "ask"}
