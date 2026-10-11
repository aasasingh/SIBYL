"""Polymarket International fee schedule tests.

Each coefficient is cross-checked against the venue's own published 100-share
peak fee, which is an independent confirmation: coef * 100 * 0.25 == peak.
Source: docs.polymarket.com/polymarket-learn/trading/fees (verified 2026-10-11).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from sibyl.costs.fees import (
    INTL_MAKER_REBATE_SHARE,
    INTL_TAKER_COEF,
    IntlFeeCategory,
    intl_fee_category,
    polymarket_intl_effective_taker_cost,
    polymarket_intl_maker_fee,
    polymarket_intl_taker_fee,
    polymarket_taker_fee,
)

# Published peak fee per 100 shares at p=0.50, straight from the docs table.
PUBLISHED_PEAKS: dict[IntlFeeCategory, str] = {
    IntlFeeCategory.CRYPTO: "1.75",
    IntlFeeCategory.SPORTS: "1.25",
    IntlFeeCategory.ECONOMICS: "1.25",
    IntlFeeCategory.CULTURE: "1.25",
    IntlFeeCategory.WEATHER: "1.25",
    IntlFeeCategory.OTHER: "1.25",
    IntlFeeCategory.FINANCE: "1.00",
    IntlFeeCategory.POLITICS: "1.00",
    IntlFeeCategory.MENTIONS: "1.00",
    IntlFeeCategory.TECH: "1.00",
    IntlFeeCategory.GEOPOLITICS: "0",
}


class TestPublishedPeaks:
    @pytest.mark.parametrize(("category", "peak"), PUBLISHED_PEAKS.items())
    def test_reproduces_published_peak(self, category: IntlFeeCategory, peak: str) -> None:
        assert polymarket_intl_taker_fee("0.50", 100, category=category) == Decimal(peak)

    def test_every_category_has_a_published_peak(self) -> None:
        assert set(PUBLISHED_PEAKS) == set(IntlFeeCategory)
        assert set(INTL_TAKER_COEF) == set(IntlFeeCategory)
        assert set(INTL_MAKER_REBATE_SHARE) == set(IntlFeeCategory)


class TestIntlCurve:
    def test_geopolitics_is_free(self) -> None:
        for price in ("0.05", "0.50", "0.95"):
            assert polymarket_intl_taker_fee(price, 1000, category=IntlFeeCategory.GEOPOLITICS) == 0

    def test_symmetric_in_probability(self) -> None:
        a = polymarket_intl_taker_fee("0.30", 100, category=IntlFeeCategory.SPORTS)
        b = polymarket_intl_taker_fee("0.70", 100, category=IntlFeeCategory.SPORTS)
        assert a == b

    def test_tails_are_far_cheaper_than_the_middle(self) -> None:
        mid = polymarket_intl_taker_fee("0.50", 100, category=IntlFeeCategory.CRYPTO)
        tail = polymarket_intl_taker_fee("0.99", 100, category=IntlFeeCategory.CRYPTO)
        # True ratio is p(1-p) at 0.50 vs 0.99 = 0.25/0.0099 = 25.25…; the
        # round "25x" quoted for the US curve is an artifact of cent rounding,
        # which INTL's 5-decimal rounding does not impose.
        assert mid / tail == pytest.approx(Decimal("25.2525"), abs=Decimal("0.001"))
        assert tail == Decimal("0.06930")

    def test_sub_minimum_fee_rounds_to_zero(self) -> None:
        # 0.04 * 1 * 0.01 * 0.99 = 0.000396 -> above the 0.00001 minimum
        assert polymarket_intl_taker_fee("0.01", 1, category=IntlFeeCategory.POLITICS) > 0
        # A fee below 0.00001 rounds away entirely.
        tiny = polymarket_intl_taker_fee(
            "0.0001", Decimal("0.01"), category=IntlFeeCategory.POLITICS
        )
        assert tiny == 0

    def test_five_decimal_rounding_not_cent_rounding(self) -> None:
        """INTL keeps 5 decimals; the US curve rounds to the cent."""
        fee = polymarket_intl_taker_fee("0.37", 1, category=IntlFeeCategory.CRYPTO)
        # 0.07 * 0.37 * 0.63 = 0.0163170 -> 0.01632 (5dp, half-to-even)
        assert fee == Decimal("0.01632")

    def test_makers_never_charged(self) -> None:
        assert polymarket_intl_maker_fee("0.50", 10_000) == 0

    def test_floats_rejected(self) -> None:
        with pytest.raises(TypeError):
            polymarket_intl_taker_fee(0.5, 100)


class TestUsVsIntlAreDistinct:
    def test_same_market_prices_differently_per_venue(self) -> None:
        """A politics market: US charges 0.06, INTL charges 0.04. Using one
        venue's curve on the other's book manufactures phantom edge."""
        us = polymarket_taker_fee("0.50", 100)
        intl = polymarket_intl_taker_fee("0.50", 100, category=IntlFeeCategory.POLITICS)
        assert us == Decimal("1.50")
        assert intl == Decimal("1.00")
        assert us != intl

    def test_intl_has_no_per_fill_maker_credit(self) -> None:
        """US maker cost is below price (per-fill rebate); INTL maker cost is
        exactly price, because the rebate is a daily pool share."""
        from sibyl.costs.fees import polymarket_effective_maker_cost

        assert polymarket_effective_maker_cost("0.50") < Decimal("0.50")
        assert polymarket_intl_maker_fee("0.50", 100) == 0

    def test_effective_cost_includes_fee(self) -> None:
        cost = polymarket_intl_effective_taker_cost("0.50", category=IntlFeeCategory.CRYPTO)
        assert cost == Decimal("0.50") + Decimal("0.07") * Decimal("0.25")


class TestFeeCategoryResolution:
    def test_geopolitics_beats_politics(self) -> None:
        """The fee difference is total (0 vs 0.04), so priority matters."""
        assert intl_fee_category(["Politics", "Geopolitics"]) is IntlFeeCategory.GEOPOLITICS
        assert intl_fee_category(["Ukraine", "Trump"]) is IntlFeeCategory.GEOPOLITICS

    def test_plain_politics(self) -> None:
        assert intl_fee_category(["Politics", "Senate"]) is IntlFeeCategory.POLITICS

    def test_unknown_tags_fall_back_to_other_not_free(self) -> None:
        """Defaulting to a cheaper category would understate costs."""
        category = intl_fee_category(["Totally Unknown Tag"])
        assert category is IntlFeeCategory.OTHER
        assert INTL_TAKER_COEF[category] == Decimal("0.05")
        assert intl_fee_category(None) is IntlFeeCategory.OTHER

    def test_known_categories(self) -> None:
        assert intl_fee_category(["Bitcoin"]) is IntlFeeCategory.CRYPTO
        assert intl_fee_category(["NFL"]) is IntlFeeCategory.SPORTS
        assert intl_fee_category(["Earnings"]) is IntlFeeCategory.FINANCE

    def test_token_launch_markets_priced_as_crypto(self) -> None:
        """FDV / pre-market token markets take the crypto coefficient (0.07),
        not finance (0.04): the likelier reading, and the conservative one."""
        assert intl_fee_category(["FDV", "Pre-Market"]) is IntlFeeCategory.CRYPTO
        assert INTL_TAKER_COEF[intl_fee_category(["Token Sales"])] == Decimal("0.07")
