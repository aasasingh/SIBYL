"""Hypothesis property tests for the §5.2 parity invariants — Phase 1 acceptance."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from sibyl.alpha.parity import (
    cross_venue,
    neg_risk_no_basket,
    neg_risk_yes_basket,
    subset,
    within_market,
)

# Effective per-contract costs: price plus friction, so slightly beyond [0,1].
costs = st.integers(min_value=1, max_value=1100).map(lambda m: Decimal(m) / 1000)
frictions = st.integers(min_value=0, max_value=50).map(lambda m: Decimal(m) / 1000)


class TestWithinMarket:
    def test_known_arb(self) -> None:
        assert within_market(Decimal("0.48"), Decimal("0.49")) == Decimal("0.03")

    def test_no_arb_at_or_above_one(self) -> None:
        assert within_market(Decimal("0.50"), Decimal("0.50")) is None
        assert within_market(Decimal("0.55"), Decimal("0.50")) is None

    @given(cost_yes=costs, cost_no=costs)
    def test_arb_iff_total_below_one(self, cost_yes: Decimal, cost_no: Decimal) -> None:
        profit = within_market(cost_yes, cost_no)
        if cost_yes + cost_no < 1:
            assert profit == 1 - (cost_yes + cost_no)
        else:
            assert profit is None

    @given(cost_yes=costs, cost_no=costs, friction=frictions)
    def test_friction_never_creates_an_arb(
        self, cost_yes: Decimal, cost_no: Decimal, friction: Decimal
    ) -> None:
        base = within_market(cost_yes, cost_no)
        worse = within_market(cost_yes + friction, cost_no)
        if base is None:
            assert worse is None
        elif worse is not None:
            assert worse <= base


class TestCrossVenue:
    @given(ya=costs, na=costs, yb=costs, nb=costs)
    def test_picks_cheaper_leg_combination(
        self, ya: Decimal, na: Decimal, yb: Decimal, nb: Decimal
    ) -> None:
        profit = cross_venue(ya, na, yb, nb)
        combos = [ya + nb, yb + na]
        if min(combos) < 1:
            assert profit == 1 - min(combos)
        else:
            assert profit is None

    @given(ya=costs, na=costs, yb=costs, nb=costs)
    def test_never_worse_than_either_within_market(
        self, ya: Decimal, na: Decimal, yb: Decimal, nb: Decimal
    ) -> None:
        cross = cross_venue(ya, na, yb, nb)
        for wm in (within_market(ya, nb), within_market(yb, na)):
            if wm is not None:
                assert cross is not None and cross >= wm


class TestNegRiskBaskets:
    @given(
        costs_yes=st.lists(costs, min_size=2, max_size=12),
    )
    def test_yes_basket_iff_sum_below_one(self, costs_yes: list[Decimal]) -> None:
        profit = neg_risk_yes_basket(costs_yes)
        total = sum(costs_yes, Decimal("0"))
        if total < 1:
            assert profit == 1 - total
        else:
            assert profit is None

    @given(costs_no=st.lists(costs, min_size=2, max_size=12))
    def test_no_basket_iff_sum_below_n_minus_one(self, costs_no: list[Decimal]) -> None:
        profit = neg_risk_no_basket(costs_no)
        total = sum(costs_no, Decimal("0"))
        n = len(costs_no)
        if total < n - 1:
            assert profit == (n - 1) - total
        else:
            assert profit is None

    @given(
        cents=st.lists(st.integers(min_value=1, max_value=50), min_size=2, max_size=8),
        friction=frictions,
    )
    def test_coherent_frictionful_partition_has_no_dutch_book(
        self, cents: list[int], friction: Decimal
    ) -> None:
        """If Σ p_YES = 1 exactly and every leg costs p + friction, neither
        basket can be an arb — retail's Σ=1.04 'free money' dies here.

        The partition is built from integer cents summing to exactly 100:
        normalising by Decimal division would produce non-terminating thirds
        whose sum falls 1E-28 short of 1 and manufacture a phantom arb —
        precisely the rounding-error trap from spec §3.2.
        """
        remainder = 100 - sum(cents)
        assume(remainder >= 1)
        coherent = [Decimal(c) / 100 for c in [*cents, remainder]]
        assert sum(coherent, Decimal("0")) == 1
        yes_costs = [p + friction for p in coherent]
        no_costs = [(1 - p) + friction for p in coherent]
        assert neg_risk_yes_basket(yes_costs) is None
        assert neg_risk_no_basket(no_costs) is None

    def test_partition_needs_two_outcomes(self) -> None:
        with pytest.raises(ValueError):
            neg_risk_yes_basket([Decimal("0.5")])


class TestSubset:
    def test_known_subset_arb(self) -> None:
        # 2024-election shape: YES(wins-election superset via inauguration
        # framing) cheap + NO(certified-by-network subset) cheap
        assert subset(Decimal("0.60"), Decimal("0.35")) == Decimal("0.05")

    @given(cost_yes_sup=costs, cost_no_sub=costs)
    def test_same_form_as_within_market(self, cost_yes_sup: Decimal, cost_no_sub: Decimal) -> None:
        assert subset(cost_yes_sup, cost_no_sub) == within_market(cost_yes_sup, cost_no_sub)
