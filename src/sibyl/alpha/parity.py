"""No-arbitrage parity conditions (§5.2).

Every function takes EFFECTIVE per-contract costs — fee, half-spread, walked
slippage, and gas already included — and returns the guaranteed per-contract
profit if the structure clears, else None. Feeding quoted mid prices into
these functions is exactly the retail mistake (§2.3.3) this module's types
try to make awkward: costs come from `costs.execution`, not from quotes.

Semantic preconditions are the caller's problem and they are not optional:
cross-venue conditions require a verified EQUIVALENT relation, subset requires
a verified SUBSET relation, and NegRisk baskets require a complete, mutually
exclusive, non-augmented partition (§4.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

ONE = Decimal("1")


def _positive_or_none(profit: Decimal) -> Decimal | None:
    return profit if profit > 0 else None


def within_market(cost_yes: Decimal, cost_no: Decimal) -> Decimal | None:
    """Buy YES and NO in the same market: payoff 1, profit 1 - total cost."""
    return _positive_or_none(ONE - (cost_yes + cost_no))


def cross_venue(
    cost_yes_a: Decimal,
    cost_no_a: Decimal,
    cost_yes_b: Decimal,
    cost_no_b: Decimal,
) -> Decimal | None:
    """Best of YES@A+NO@B vs YES@B+NO@A on verified-equivalent markets.

    Capital stays locked to resolution on both venues — rank the result by
    APY (costs.capital), never by this absolute number.
    """
    best_cost = min(cost_yes_a + cost_no_b, cost_yes_b + cost_no_a)
    return _positive_or_none(ONE - best_cost)


def neg_risk_yes_basket(costs_yes: Sequence[Decimal]) -> Decimal | None:
    """Buy every YES in a complete mutually-exclusive set: payoff exactly 1."""
    if len(costs_yes) < 2:
        raise ValueError("a partition has at least 2 outcomes")
    return _positive_or_none(ONE - sum(costs_yes, Decimal("0")))


def neg_risk_no_basket(costs_no: Sequence[Decimal]) -> Decimal | None:
    """Buy every NO in a complete mutually-exclusive set of n: payoff n - 1."""
    n = len(costs_no)
    if n < 2:
        raise ValueError("a partition has at least 2 outcomes")
    return _positive_or_none(Decimal(n - 1) - sum(costs_no, Decimal("0")))


def subset(cost_yes_superset: Decimal, cost_no_subset: Decimal) -> Decimal | None:
    """f(s) ⊂ f(S): YES(superset) + NO(subset) pays at least 1 in every state.

    Cases: s occurs → S occurs too, YES(S)=1, NO(s)=0 → 1. S occurs without
    s → YES(S)=1, NO(s)=1 → 2. Neither occurs → YES(S)=0, NO(s)=1 → 1.
    Guaranteed profit is against the worst case, 1 - total cost. (Direction
    matters: NO(superset) + YES(subset) pays 0 in the S-without-s state.)
    """
    return _positive_or_none(ONE - (cost_yes_superset + cost_no_subset))
