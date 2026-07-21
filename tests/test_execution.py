"""Depth-walk execution tests: never price size at the top of book."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from sibyl.costs.execution import buy_no, buy_yes, walk_levels
from sibyl.costs.fees import kalshi_taker_fee, polymarket_taker_fee
from sibyl.venues.normalise import BinaryBook, Level, Venue

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _level(price: str, size: str) -> Level:
    return Level(price=Decimal(price), size=Decimal(size))


def _no_fee(price: Decimal, contracts: Decimal) -> Decimal:
    return Decimal("0")


BOOK = BinaryBook(
    venue=Venue.KALSHI,
    market_id="M",
    ts=NOW,
    yes_bids=(_level("0.40", "100"), _level("0.38", "200")),
    yes_asks=(_level("0.42", "50"), _level("0.45", "100"), _level("0.50", "500")),
)


class TestWalkLevels:
    def test_single_level_fill(self) -> None:
        est = walk_levels(BOOK.yes_asks, Decimal("50"), _no_fee)
        assert est.complete
        assert est.avg_price == Decimal("0.42")
        assert est.levels_consumed == 1

    def test_multi_level_fill_averages_up(self) -> None:
        est = walk_levels(BOOK.yes_asks, Decimal("100"), _no_fee)
        assert est.complete
        # 50 @ 0.42 + 50 @ 0.45 = 43.50 for 100
        assert est.notional == Decimal("43.50")
        assert est.avg_price == Decimal("0.435")
        assert est.avg_price is not None
        top = BOOK.best_ask
        assert top is not None and est.avg_price > top.price  # slippage is real

    def test_partial_fill_reported(self) -> None:
        est = walk_levels(BOOK.yes_asks, Decimal("1000"), _no_fee)
        assert not est.complete
        assert est.filled == Decimal("650")
        assert est.requested == Decimal("1000")

    def test_fee_accrues_per_level(self) -> None:
        est = walk_levels(BOOK.yes_asks, Decimal("100"), polymarket_taker_fee)
        # fee(0.42, 50) = .06*50*.42*.58 = 0.7308 -> 0.73
        # fee(0.45, 50) = .06*50*.45*.55 = 0.7425 -> 0.74
        assert est.fee == Decimal("1.47")
        assert est.total_cost == Decimal("43.50") + Decimal("1.47")

    def test_kalshi_per_level_ceil(self) -> None:
        est = walk_levels(BOOK.yes_asks, Decimal("1"), kalshi_taker_fee)
        # 0.07 * 1 * 0.42 * 0.58 = 0.017052 -> ceil -> 0.02
        assert est.fee == Decimal("0.02")

    def test_zero_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            walk_levels(BOOK.yes_asks, Decimal("0"), _no_fee)


class TestSides:
    def test_buy_yes_walks_asks(self) -> None:
        est = buy_yes(BOOK, Decimal("50"), _no_fee)
        assert est.avg_price == Decimal("0.42")

    def test_buy_no_walks_yes_bids_complemented(self) -> None:
        est = buy_no(BOOK, Decimal("100"), _no_fee)
        # NO asks derived from YES bids: 0.60 (from bid 0.40), then 0.62
        assert est.complete
        assert est.avg_price == Decimal("0.60")
        deeper = buy_no(BOOK, Decimal("150"), _no_fee)
        # 100 @ 0.60 + 50 @ 0.62 = 91.00 for 150
        assert deeper.notional == Decimal("91.00")

    def test_within_market_round_trip_costs_more_than_one(self) -> None:
        # taking both sides of an uncrossed book at size must cost >= 1/contract
        yes = buy_yes(BOOK, Decimal("50"), _no_fee)
        no = buy_no(BOOK, Decimal("50"), _no_fee)
        assert yes.effective_price is not None and no.effective_price is not None
        assert yes.effective_price + no.effective_price >= 1
