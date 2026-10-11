"""Apply a fitted slope to a live quote (spec §5.4).

    p* = p^b / (p^b + (1-p)^b)      equivalently   logit(p*) = b * logit(p)
    edge = p*_lower_bound - c_eff

Two rules this module exists to enforce:

  1. Edge is computed against the LOWER BOUND of the recalibrated
     probability, never the point estimate. The point estimate is the
     optimistic read of a noisy fit.
  2. A cell whose CI straddles b = 1 produces NO signal. `recalibrate`
     returns `tradeable=False` and an edge of zero rather than a small
     positive number that would survive into sizing.

This is the Decimal boundary: statistics arrive as floats, and everything
that touches an effective cost leaves as Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from sibyl.calibration.fit import CellFit, expit, logit

# Recalibrated probabilities are clamped to the tradeable price range: a model
# claiming 0.999 on a market quoted at 0.97 is extrapolating past any data.
_P_FLOOR = 0.01
_P_CEIL = 0.99


def recalibrated_probability(price: float, slope: float, intercept: float = 0.0) -> float:
    """p* from the fitted logistic map, clamped to [0.01, 0.99]."""
    value = float(expit(intercept + slope * logit(price)))
    return min(max(value, _P_FLOOR), _P_CEIL)


@dataclass(frozen=True)
class Signal:
    """A recalibration signal for one quote, with its own honesty flags."""

    price: Decimal
    fair_value: Decimal  # p* point estimate
    fair_value_low: Decimal  # p* at the slope CI's unfavourable end
    fair_value_high: Decimal
    effective_cost: Decimal
    edge: Decimal  # fair_value_low - effective_cost, floored at 0
    tradeable: bool
    reason: str

    @property
    def has_edge(self) -> bool:
        return self.tradeable and self.edge > 0


def _to_decimal(value: float, places: str = "0.000001") -> Decimal:
    return Decimal(str(value)).quantize(Decimal(places))


def recalibrate(
    price: Decimal,
    cell: CellFit,
    effective_cost: Decimal,
    *,
    use_pooled: bool = True,
) -> Signal:
    """Turn a quote plus a fitted cell into a sized-ready signal.

    `effective_cost` must already include fees, half-spread and walked
    slippage (see costs.execution) — this function does not model costs, it
    only compares against them.
    """
    price_f = float(price)
    no_signal = Signal(
        price=price,
        fair_value=price,
        fair_value_low=price,
        fair_value_high=price,
        effective_cost=effective_cost,
        edge=Decimal("0"),
        tradeable=False,
        reason="",
    )

    if not (0 < price_f < 1):
        return Signal(**{**no_signal.__dict__, "reason": f"price {price} outside (0,1)"})
    if not cell.tradeable:
        # Spec §6: a cell whose CI straddles 1.0 is not tradeable. Say so.
        detail = cell.note or (
            "CI straddles b=1" if cell.slope_ci is not None else "no clustered CI"
        )
        return Signal(**{**no_signal.__dict__, "reason": f"{cell.domain}/{cell.horizon}: {detail}"})

    slope = cell.effective_slope if use_pooled else cell.slope
    if not np.isfinite(slope):
        return Signal(**{**no_signal.__dict__, "reason": "non-finite slope"})

    point = recalibrated_probability(price_f, slope, cell.intercept)
    # Map the slope interval through the same transform. Which CI end is
    # unfavourable depends on the side of 0.5: above it a larger slope pushes
    # p* up, below it a larger slope pushes p* down.
    assert cell.slope_ci is not None  # guaranteed by cell.tradeable
    ends = [
        recalibrated_probability(price_f, cell.slope_ci[0], cell.intercept),
        recalibrated_probability(price_f, cell.slope_ci[1], cell.intercept),
    ]
    low, high = min(ends), max(ends)

    edge_f = low - float(effective_cost)
    return Signal(
        price=price,
        fair_value=_to_decimal(point),
        fair_value_low=_to_decimal(low),
        fair_value_high=_to_decimal(high),
        effective_cost=effective_cost,
        edge=_to_decimal(max(edge_f, 0.0)),
        tradeable=True,
        reason=f"b={slope:.3f} ({cell.domain}/{cell.horizon})",
    )
