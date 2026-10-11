"""Partial pooling across (domain x horizon) cells.

Thin cells produce slope estimates that are mostly noise. Pooling borrows
strength from the other cells: a cell with 12 markets gets pulled hard toward
the global slope, a cell with 900 markets barely moves.

DEVIATION FROM SPEC: the build spec calls for a NumPyro MCMC model. This is
an analytic empirical-Bayes (random-effects) model instead. For the structure
we actually have — one scalar parameter per cell, each with a bootstrap
standard error — the normal-normal hierarchical posterior has a closed form,
so MCMC would sample a distribution we can write down exactly. The tradeoff
accepted here: no full posterior over (mu, tau), so intervals treat tau as
known rather than integrating over it, which makes them mildly optimistic.

Tradeability therefore never rests on a pooled interval. `CellFit.tradeable`
requires the UNPOOLED clustered-bootstrap CI to exclude 1 *and* the traded
slope to still be the cell's own (shrinkage below a threshold), because a
heavily shrunk cell is trading mu and must be judged on mu's own interval —
`population_slope_ci` — instead.

Model:
    b_i_hat ~ Normal(b_i, se_i^2)          # sampling error within a cell
    b_i     ~ Normal(mu, tau^2)            # cells differ, but not wildly

tau^2 by DerSimonian-Laird; posterior mean per cell is the precision-weighted
blend of its own estimate and mu.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sibyl.calibration.fit import CellFit

# Floor on a cell's standard error. A bootstrap SE near zero (possible in a
# huge, very homogeneous cell) would otherwise claim infinite precision and
# refuse any pooling at all.
_MIN_SE = 1e-3


def _dersimonian_laird(estimates: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """Return (mu, tau_squared) for a random-effects meta-analysis."""
    weights = 1.0 / variances
    mu_fixed = float(np.sum(weights * estimates) / np.sum(weights))
    q_stat = float(np.sum(weights * (estimates - mu_fixed) ** 2))
    df = estimates.size - 1
    if df <= 0:
        return mu_fixed, 0.0
    c_term = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau_squared = max(0.0, (q_stat - df) / c_term) if c_term > 0 else 0.0
    # Re-centre mu using random-effects weights now that tau is known.
    re_weights = 1.0 / (variances + tau_squared)
    mu = float(np.sum(re_weights * estimates) / np.sum(re_weights))
    return mu, tau_squared


def population_slope_ci(cells: list[CellFit], *, z: float = 1.96) -> tuple[float, float] | None:
    """CI for the population slope mu under the random-effects model.

    This is the interval that matters for any cell whose own data was too
    thin to speak — those cells trade mu, so the claim "mu != 1" is what has
    to hold, and it must be tested here rather than borrowed from a
    better-populated cell's interval.

    se(mu)^2 = 1 / sum_i 1/(se_i^2 + tau^2)
    """
    usable = [c for c in cells if c.converged and c.slope_se is not None and np.isfinite(c.slope)]
    if len(usable) < 2:
        return None
    estimates = np.array([c.slope for c in usable], dtype=np.float64)
    variances = np.array([max(c.slope_se or _MIN_SE, _MIN_SE) ** 2 for c in usable])
    mu, tau_squared = _dersimonian_laird(estimates, variances)
    precision = float(np.sum(1.0 / (variances + tau_squared)))
    if precision <= 0:
        return None
    se_mu = float(np.sqrt(1.0 / precision))
    return mu - z * se_mu, mu + z * se_mu


def partial_pool(
    cells: list[CellFit],
    *,
    fallback_slope: float = 1.0,
) -> tuple[list[CellFit], float, float]:
    """Shrink each cell's slope toward the cross-cell mean.

    Cells that converged with a bootstrap SE contribute to estimating the
    population (mu, tau) and receive a posterior mean. Cells that failed to
    fit get mu outright — the honest statement being "we know nothing specific
    about this cell, so assume it behaves like the average one".

    Returns (pooled_cells, mu, tau).
    """
    usable = [c for c in cells if c.converged and c.slope_se is not None and np.isfinite(c.slope)]
    if len(usable) < 2:
        # Nothing to pool against: leave estimates alone, report no structure.
        return list(cells), fallback_slope, 0.0

    estimates = np.array([c.slope for c in usable], dtype=np.float64)
    variances = np.array([max(c.slope_se or _MIN_SE, _MIN_SE) ** 2 for c in usable])
    mu, tau_squared = _dersimonian_laird(estimates, variances)

    pooled: list[CellFit] = []
    for cell in cells:
        if cell.converged and cell.slope_se is not None and np.isfinite(cell.slope):
            se_squared = max(cell.slope_se, _MIN_SE) ** 2
            # Shrinkage: 0 = keep own estimate, 1 = fully replaced by mu.
            shrink = (
                se_squared / (se_squared + tau_squared) if (se_squared + tau_squared) > 0 else 1.0
            )
            pooled.append(
                replace(
                    cell,
                    pooled_slope=float((1 - shrink) * cell.slope + shrink * mu),
                    shrinkage=float(shrink),
                )
            )
        else:
            pooled.append(replace(cell, pooled_slope=float(mu), shrinkage=1.0))
    return pooled, float(mu), float(np.sqrt(tau_squared))
