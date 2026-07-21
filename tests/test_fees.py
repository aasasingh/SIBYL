"""Fee-curve tests (spec §2.2, §3.4).

The golden test loads tests/fixtures/polymarket_fee_table_100lot.json,
transcribed verbatim from https://docs.polymarket.us/fees on 2026-07-21.
Regenerating that fixture from the formula under test would be circular;
extend it only by copying rows from the published docs.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sibyl.costs.fees import (
    kalshi_maker_fee,
    kalshi_taker_fee,
    polymarket_effective_maker_cost,
    polymarket_effective_taker_cost,
    polymarket_maker_rebate,
    polymarket_taker_fee,
    polymarket_volume_discount,
)

FIXTURE = Path(__file__).parent / "fixtures" / "polymarket_fee_table_100lot.json"

cent_prices = st.integers(min_value=1, max_value=99).map(lambda c: Decimal(c) / 100)
lot_sizes = st.integers(min_value=1, max_value=10_000).map(Decimal)


class TestPolymarketTaker:
    def test_at_the_money_100_lot(self) -> None:
        # 0.06 * 100 * 0.5 * 0.5 = 1.50 exactly
        assert polymarket_taker_fee("0.50", 100) == Decimal("1.50")

    def test_tail_is_25x_cheaper_than_middle(self) -> None:
        at_money = polymarket_taker_fee("0.50", 100)
        tail = polymarket_taker_fee("0.99", 100)
        assert tail == Decimal("0.06")
        assert at_money / tail == 25

    def test_bankers_rounding_half_to_even(self) -> None:
        # raw = 0.06 * 250 * 0.9 * 0.1 = 1.35; need exact .xx5 cases instead:
        # 0.06 * C * p(1-p) with p=0.9 -> 0.0054/contract
        # C=375 -> 2.025 -> half-to-even at the cent -> 2.02 (2 is even)
        assert polymarket_taker_fee("0.90", 375) == Decimal("2.02")
        # C=625 -> 3.375 -> half-to-even -> 3.38 (7 is odd, rounds to even 8)
        assert polymarket_taker_fee("0.90", 625) == Decimal("3.38")

    def test_volume_discount_tiers(self) -> None:
        assert polymarket_volume_discount("0") == Decimal("0")
        assert polymarket_volume_discount("249999") == Decimal("0")
        assert polymarket_volume_discount("250000") == Decimal("0.10")
        assert polymarket_volume_discount("1000000") == Decimal("0.25")
        assert polymarket_volume_discount("10000000") == Decimal("0.50")

    def test_discount_applied_before_rounding(self) -> None:
        assert polymarket_taker_fee("0.50", 100, volume_discount=Decimal("0.10")) == Decimal("1.35")

    @given(price=cent_prices, contracts=lot_sizes)
    def test_symmetric_in_probability(self, price: Decimal, contracts: Decimal) -> None:
        assert polymarket_taker_fee(price, contracts) == polymarket_taker_fee(1 - price, contracts)

    @given(price=cent_prices, contracts=lot_sizes)
    def test_never_negative_and_peaks_at_half(self, price: Decimal, contracts: Decimal) -> None:
        fee = polymarket_taker_fee(price, contracts)
        assert fee >= 0
        assert fee <= polymarket_taker_fee("0.50", contracts)


class TestPolymarketMaker:
    def test_rebate_at_the_money(self) -> None:
        # 0.0125 * 100 * 0.25 = 0.3125 -> 0.31 (third decimal 2 rounds down)
        assert polymarket_maker_rebate("0.50", 100) == Decimal("0.31")

    def test_bankers_rounding(self) -> None:
        # 0.0125 * 200 * 0.09 = 0.225 -> half-to-even -> 0.22
        assert polymarket_maker_rebate("0.90", 200) == Decimal("0.22")
        # 0.0125 * 600 * 0.09 = 0.675 -> half-to-even -> 0.68
        assert polymarket_maker_rebate("0.90", 600) == Decimal("0.68")

    @given(price=cent_prices, contracts=lot_sizes)
    def test_rebate_below_taker_fee(self, price: Decimal, contracts: Decimal) -> None:
        assert polymarket_maker_rebate(price, contracts) <= polymarket_taker_fee(price, contracts)


class TestKalshi:
    def test_taker_ceils_to_cent(self) -> None:
        # 0.07 * 1 * 0.25 = 0.0175 -> ceil -> 0.02
        assert kalshi_taker_fee("0.50", 1) == Decimal("0.02")
        # exact cent stays: 0.07 * 100 * 0.25 = 1.75
        assert kalshi_taker_fee("0.50", 100) == Decimal("1.75")

    def test_maker_zero_by_default(self) -> None:
        assert kalshi_maker_fee("0.50", 100) == Decimal("0.00")

    def test_designated_market_maker_coefficient(self) -> None:
        # 0.0175 * 100 * 0.25 = 0.4375 -> ceil -> 0.44
        assert kalshi_maker_fee("0.50", 100, designated_market=True) == Decimal("0.44")

    def test_special_event_flat_fee_dominates_longshots(self) -> None:
        # 0.25% of $1 per contract: 100 contracts -> $0.25 regardless of price
        fee = kalshi_maker_fee("0.03", 100, special_event=True)
        assert fee == Decimal("0.25")
        # ~8% of the $3.00 the position costs — the structural trap from §2.2
        assert fee / (Decimal("0.03") * 100) > Decimal("0.08")

    @given(price=cent_prices, contracts=lot_sizes)
    def test_ceil_never_below_raw(self, price: Decimal, contracts: Decimal) -> None:
        raw = Decimal("0.07") * contracts * price * (1 - price)
        assert kalshi_taker_fee(price, contracts) >= raw


class TestEffectiveCost:
    def test_taker_cost_above_maker_cost(self) -> None:
        p = Decimal("0.50")
        assert polymarket_effective_taker_cost(p) == Decimal("0.515")
        assert polymarket_effective_maker_cost(p) == Decimal("0.496875")

    def test_cross_venue_taker_taker_breakeven_spread(self) -> None:
        # §2.2: at the money the round trip costs ~1.5¢ + 1.75¢ = 3.25¢
        pm = polymarket_effective_taker_cost("0.50") - Decimal("0.50")
        from sibyl.costs.fees import kalshi_effective_taker_cost

        k = kalshi_effective_taker_cost("0.50") - Decimal("0.50")
        assert pm + k == Decimal("0.0325")


class TestFloatRejection:
    def test_floats_rejected_everywhere(self) -> None:
        for fn in (polymarket_taker_fee, polymarket_maker_rebate, kalshi_taker_fee):
            with pytest.raises(TypeError):
                fn(0.5, 100)
            with pytest.raises(TypeError):
                fn("0.5", 100.0)

    def test_out_of_range_prices_rejected(self) -> None:
        for bad in ("0", "1", "-0.1", "1.5"):
            with pytest.raises(ValueError):
                polymarket_taker_fee(bad, 100)


def test_golden_polymarket_published_table() -> None:
    """Reproduce every officially published taker-pays / maker-receives row.

    Fixture transcribed verbatim from https://docs.polymarket.us/fees
    (schedule effective 2026-07-01). The $0.25 row (raw 1.125 -> $1.12) only
    passes under banker's rounding, so this test pins the rounding mode
    against the venue's own numbers, not our formula's.
    """
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["rows"]
    assert rows, "fixture has no rows"
    for row in rows:
        price = row["price"]
        assert polymarket_taker_fee(price, 100) == Decimal(row["taker_pays"]), price
        assert polymarket_maker_rebate(price, 100) == Decimal(row["maker_receives"]), price
