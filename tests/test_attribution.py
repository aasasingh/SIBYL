"""Trade-attribution tests.

A decrease at a level is a trade or a cancel, and the two mean opposite
things for adverse selection. These pin down the measured rules: the taker
side decides which book was hit, trades hit both legs, and ambiguity is
labelled rather than silently resolved.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

from sibyl.fill.attribution import Decrease, Trade, attribute, parse_trade

D = Decimal
T0 = datetime(2026, 10, 12, tzinfo=UTC)
MIRROR = {"Y": "N", "N": "Y"}


def dec(asset: str, side: str, price: str, amount: str, ts: int) -> Decrease:
    return Decrease(asset, side, D(price), D(amount), ts, T0)  # type: ignore[arg-type]


def trade(asset: str, side: str, price: str, size: str, ts: int) -> Trade:
    return Trade(asset, D(price), D(size), side, ts, f"0x{ts}")  # type: ignore[arg-type]


def bot(window_ms: int) -> list[str]:
    """The measured bot: 5 @ 0.002 once a second, delta ~20 ms before trade."""
    decs = [dec("Y", "ask", "0.002", "5", t * 1000 - 20) for t in range(1, 6)]
    trades = [trade("Y", "BUY", "0.002", "5", t * 1000) for t in range(1, 6)]
    out, _ = attribute(decs, trades, mirror={}, window_ms=window_ms)
    assert all(a.traded == D(5) for a in out), "each trade lands on its own decrease"
    return [a.kind for a in out]


class TestMatching:
    def test_a_taker_buy_consumes_the_ask(self) -> None:
        out, s = attribute(
            [dec("Y", "ask", "0.60", "100", 980), dec("Y", "bid", "0.60", "100", 980)],
            [trade("Y", "BUY", "0.60", "100", 1000)],
            mirror={},
        )
        assert {a.decrease.side: a.kind for a in out} == {"ask": "trade", "bid": "cancel"}
        assert s.unmatched_trades == 0

    def test_a_trade_is_credited_on_the_mirror_leg_too(self) -> None:
        """A YES ask at 0.60 is a NO bid at 0.40. Without mirroring the NO-side
        decrease would be booked as a cancel."""
        out, s = attribute(
            [dec("Y", "ask", "0.60", "100", 980), dec("N", "bid", "0.40", "100", 980)],
            [trade("Y", "BUY", "0.60", "100", 1000)],
            mirror=MIRROR,
        )
        assert [a.kind for a in out] == ["trade", "trade"]
        assert s.unmatched_mirror == 0

    def test_without_a_trade_a_decrease_is_a_cancel(self) -> None:
        out, _ = attribute([dec("Y", "bid", "0.40", "50", 0)], [], mirror={})
        assert out[0].kind == "cancel"
        assert out[0].cancelled == D("50")

    def test_partial_explanation_is_mixed(self) -> None:
        out, s = attribute(
            [dec("Y", "ask", "0.60", "100", 980)],
            [trade("Y", "BUY", "0.60", "30", 1000)],
            mirror={},
        )
        assert out[0].kind == "mixed"
        assert (out[0].traded, out[0].cancelled) == (D("30"), D("70"))
        assert s.summary()["decreases_mixed"] == 1

    def test_trades_outside_every_window_are_reported_unmatched(self) -> None:
        _, s = attribute(
            [dec("Y", "ask", "0.60", "100", 0)],
            [trade("Y", "BUY", "0.60", "100", 10_000)],
            mirror={},
        )
        assert s.unmatched_trades == 1
        assert s.unmatched_direct == D("100")


class TestAmbiguity:
    def test_a_unique_exact_match_is_clean_despite_nearby_cancels(self) -> None:
        out, _ = attribute(
            [
                dec("Y", "ask", "0.60", "100", 980),  # the trade
                dec("Y", "ask", "0.60", "37", 1200),  # an unrelated cancel
            ],
            [trade("Y", "BUY", "0.60", "100", 1000)],
            mirror={},
        )
        assert [a.kind for a in out] == ["trade", "cancel"]

    def test_repeated_identical_trades_resolve_by_timing(self) -> None:
        assert set(bot(window_ms=500)) == {"trade"}

    def test_a_window_wider_than_the_trade_spacing_makes_them_ambiguous(self) -> None:
        # Documents the window's upper bound: the spacing of repeated trades.
        assert "contested" in bot(window_ms=1200)

    def test_two_equal_candidates_are_contested(self) -> None:
        out, _ = attribute(
            [dec("Y", "ask", "0.60", "100", 900), dec("Y", "ask", "0.60", "100", 1100)],
            [trade("Y", "BUY", "0.60", "100", 1000)],
            mirror={},
        )
        assert all(a.kind == "contested" for a in out)
        assert sum(a.traded for a in out) == D("100")

    def test_fallback_matches_are_low_confidence(self) -> None:
        out, s = attribute(
            [dec("Y", "ask", "0.60", "100", 0)],
            [trade("Y", "BUY", "0.60", "100", 1000)],  # outside 500, inside 1500
            mirror={},
            window_ms=500,
            fallback_ms=1500,
        )
        assert out[0].kind == "contested"
        assert out[0].traded == D("100")
        assert s.unmatched_trades == 0

    def test_ambiguity_is_also_weighted_by_notional(self) -> None:
        _, s = attribute(
            [
                dec("Y", "ask", "0.002", "5", 980),  # cent-sized, clean
                dec("Y", "ask", "0.50", "100", 1980),  # $30 traded, mixed
            ],
            [trade("Y", "BUY", "0.002", "5", 1000), trade("Y", "BUY", "0.50", "60", 2000)],
            mirror={},
        )
        summary = s.summary()
        assert summary["ambiguous_share_of_traded"] == 0.5
        assert summary["ambiguous_share_of_traded_notional"] > 0.99


class TestParsing:
    def test_parse_trade(self) -> None:
        payload = json.dumps(
            {
                "asset_id": "Y",
                "price": "0.988",
                "size": "296.58",
                "side": "SELL",
                "timestamp": "1791758381722",
                "transaction_hash": "0xabc",
            }
        )
        t = parse_trade(payload)
        assert t is not None
        assert (t.consumed, t.price, t.size) == ("bid", D("0.988"), D("296.58"))

    def test_unknown_side_is_rejected(self) -> None:
        assert parse_trade(json.dumps({"side": "?"})) is None
