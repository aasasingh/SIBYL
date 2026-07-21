"""Capital lockup / APY tests (§2.3.5, §5.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from sibyl.costs.capital import (
    excess_apy,
    lockup_apy,
    lockup_days,
    pessimistic_settlement,
)
from sibyl.venues.normalise import Venue

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


class TestPessimisticSettlement:
    def test_buffers_extend_nominal_close(self) -> None:
        close = NOW + timedelta(days=30)
        assert pessimistic_settlement(close, Venue.POLYMARKET_US) == close + timedelta(days=2)
        assert pessimistic_settlement(close, Venue.KALSHI) == close + timedelta(days=1)

    def test_lockup_floored_at_one_day(self) -> None:
        past_close = NOW - timedelta(days=5)
        assert lockup_days(NOW, past_close, Venue.KALSHI) == Decimal("1")


class TestLockupApy:
    def test_spec_example_97c_contract_30_days(self) -> None:
        # §2.3.5: 97¢ resolving in 30 days ≈ 37% annualized/APY range
        apy = lockup_apy(Decimal("1"), Decimal("0.97"), Decimal("30"))
        assert Decimal("0.35") < apy < Decimal("0.50")

    def test_same_edge_shorter_lockup_higher_apy(self) -> None:
        fast = lockup_apy(Decimal("1"), Decimal("0.99"), Decimal("3"))
        slow = lockup_apy(Decimal("1"), Decimal("0.99"), Decimal("90"))
        assert fast > slow

    def test_negative_edge_gives_negative_apy(self) -> None:
        assert lockup_apy(Decimal("1"), Decimal("1.02"), Decimal("30")) < 0

    def test_dispute_buffer_reduces_apy(self) -> None:
        close = NOW + timedelta(days=10)
        nominal_days = Decimal("10")
        pessimistic = lockup_days(NOW, close, Venue.POLYMARKET_US)
        assert pessimistic == Decimal("12")
        assert lockup_apy(Decimal("1"), Decimal("0.97"), pessimistic) < lockup_apy(
            Decimal("1"), Decimal("0.97"), nominal_days
        )

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            lockup_apy(Decimal("1"), Decimal("0"), Decimal("30"))
        with pytest.raises(ValueError):
            lockup_apy(Decimal("1"), Decimal("0.97"), Decimal("0.5"))


class TestExcessApy:
    def test_kalshi_pays_a_cash_hurdle(self) -> None:
        gross = lockup_apy(Decimal("1"), Decimal("0.97"), Decimal("30"))
        assert excess_apy(Decimal("1"), Decimal("0.97"), Decimal("30"), Venue.KALSHI) == (
            gross - Decimal("0.04")
        )
        assert (
            excess_apy(Decimal("1"), Decimal("0.97"), Decimal("30"), Venue.POLYMARKET_US) == gross
        )
