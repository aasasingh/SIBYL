"""Per-(domain x horizon) logistic recalibration (spec §2.3.4, §6 Phase 3).

Model, per cell:  logit P(y=1) = a + b * logit(p)

b > 1  => quoted prices are compressed toward 0.5 (favourites underpriced)
b < 1  => quoted prices are over-extended (fade the extremes)
b ~ 1  => the market is already calibrated in this cell; no edge here

Everything in this module is float math on NumPy arrays. That is deliberate
and it is NOT a violation of the Decimal rule: these are statistical
estimates, not money. The Decimal boundary is re-entered in `apply.py`, where
a recalibrated probability meets an effective cost.

Two statistical commitments the spec insists on, both load-bearing:

  * Confidence intervals come from a MARKET-CLUSTERED bootstrap. Hourly rows
    inside one market are the same bet observed repeatedly — a row-level
    bootstrap treats ~130 correlated observations as independent evidence and
    reports intervals several times too narrow, turning noise into signal.
  * A cell whose CI straddles b = 1 is reported as NOT TRADEABLE, however
    pretty its point estimate looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

# Prices are clipped before the logit so that a 0 or 1 quote cannot produce an
# infinite feature. 1e-4 keeps 1c and 99c quotes intact while bounding the tail.
_EPS = 1e-4
MIN_ROWS = 50
MIN_MARKETS = 10
MIN_LABEL_BALANCE = 3  # need >= this many of each outcome to identify a slope

FloatArray = npt.NDArray[np.float64]


def logit(p: FloatArray | float) -> FloatArray:
    arr = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(arr / (1 - arr))


def expit(x: FloatArray | float) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass(frozen=True)
class CellFit:
    """A fitted cell, with everything needed to judge whether to trust it."""

    domain: str
    horizon: str
    n_rows: int
    n_markets: int
    slope: float
    intercept: float
    slope_ci: tuple[float, float] | None = None
    # Clustered-bootstrap standard error — the shrinkage weight for pooling.
    slope_se: float | None = None
    brier_model: float = float("nan")
    brier_baseline: float = float("nan")
    converged: bool = True
    note: str = ""
    # Set when partial pooling has shrunk this cell toward the global slope.
    pooled_slope: float | None = None
    shrinkage: float | None = None

    @property
    def effective_slope(self) -> float:
        """Slope to actually use: the pooled one when pooling was applied."""
        return self.pooled_slope if self.pooled_slope is not None else self.slope

    @property
    def tradeable(self) -> bool:
        """A cell is tradeable only if its CI excludes b = 1 (spec §6).

        No CI, failed fit, or an interval spanning 1.0 all mean the same
        thing operationally: we have no demonstrated edge in this cell.
        """
        if not self.converged or self.slope_ci is None:
            return False
        low, high = self.slope_ci
        return low > 1.0 or high < 1.0

    @property
    def brier_improvement(self) -> float:
        """Baseline minus model: positive means recalibration helped."""
        return self.brier_baseline - self.brier_model

    def describe(self) -> str:
        ci = (
            f"[{self.slope_ci[0]:.3f}, {self.slope_ci[1]:.3f}]"
            if self.slope_ci is not None
            else "no CI"
        )
        verdict = "TRADEABLE" if self.tradeable else "no signal"
        pooled = (
            f" pooled={self.pooled_slope:.3f} (shrink {self.shrinkage:.2f})"
            if self.pooled_slope is not None and self.shrinkage is not None
            else ""
        )
        return (
            f"{self.domain:<14} {self.horizon:<8} n={self.n_rows:<6} m={self.n_markets:<5} "
            f"b={self.slope:.3f} {ci}{pooled} dBrier={self.brier_improvement:+.5f} {verdict}"
            + (f" [{self.note}]" if self.note else "")
        )


def fit_logistic(
    x: FloatArray,
    y: FloatArray,
    *,
    weights: FloatArray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float, bool]:
    """Unpenalised logistic regression of y on [1, x] by Newton-IRLS.

    Returns (intercept, slope, converged). Hand-rolled rather than sklearn's
    LogisticRegression because that applies L2 regularisation by default,
    which biases the slope toward zero — i.e. it would systematically
    understate exactly the compression effect being measured.
    """
    n = x.shape[0]
    design = np.column_stack([np.ones(n), x])
    w_obs = np.ones(n) if weights is None else weights.astype(np.float64)
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = design @ beta
        mu = expit(eta)
        var = np.clip(mu * (1 - mu), 1e-10, None) * w_obs
        gradient = design.T @ (w_obs * (y - mu))
        hessian = design.T @ (design * var[:, None])
        # Ridge of last resort: keeps the solve stable in degenerate cells
        # (tiny value, orders of magnitude below any real signal).
        hessian += np.eye(2) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float(beta[0]), float(beta[1]), False
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return float("nan"), float("nan"), False
        if np.max(np.abs(step)) < tol:
            return float(beta[0]), float(beta[1]), True
    return float(beta[0]), float(beta[1]), False


def brier(predicted: FloatArray, y: FloatArray) -> float:
    return float(np.mean((predicted - y) ** 2))


def clustered_bootstrap_slope(
    x: FloatArray,
    y: FloatArray,
    market_ids: npt.NDArray[np.str_],
    *,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 20260721,
) -> tuple[tuple[float, float], float] | None:
    """Percentile CI and standard error for the slope, resampling MARKETS.

    Each draw picks whole markets with replacement and refits on their pooled
    rows, so the interval reflects how much independent evidence there really
    is: the number of distinct bets, not the number of hourly snapshots.
    Returns ((ci_low, ci_high), se) or None when the cell cannot support it.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(market_ids)
    if unique.size < MIN_MARKETS:
        return None
    # Row indices grouped by market, so a resample is a cheap concatenate.
    order = np.argsort(market_ids, kind="stable")
    sorted_ids = market_ids[order]
    boundaries = np.searchsorted(sorted_ids, unique)
    groups = np.split(order, boundaries[1:])

    slopes: list[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, unique.size, size=unique.size)
        idx = np.concatenate([groups[p] for p in picks])
        yb = y[idx]
        # A draw that happens to contain only one outcome cannot identify a
        # slope; skip it rather than feeding a garbage value into the interval.
        if yb.min() == yb.max():
            continue
        _, slope, converged = fit_logistic(x[idx], yb)
        if converged and np.isfinite(slope):
            slopes.append(slope)
    if len(slopes) < n_boot // 4:
        return None
    low, high = np.quantile(slopes, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high)), float(np.std(slopes, ddof=1))


@dataclass
class CellData:
    """Rows for one (domain, horizon) cell."""

    domain: str
    horizon: str
    price: FloatArray
    y: FloatArray
    market_id: npt.NDArray[np.str_]
    weight: FloatArray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.price.shape[0])

    @property
    def n_markets(self) -> int:
        return int(np.unique(self.market_id).size)


def fit_cell(
    data: CellData,
    *,
    n_boot: int = 400,
    seed: int = 20260721,
    min_rows: int = MIN_ROWS,
    min_markets: int = MIN_MARKETS,
) -> CellFit:
    """Fit one cell, with a clustered CI and an in-sample Brier comparison.

    Thin cells return a non-converged fit carrying the reason rather than a
    confident-looking number — `hierarchical.partial_pool` is what rescues
    them, by borrowing strength from the other cells.
    """
    base = CellFit(
        domain=data.domain,
        horizon=data.horizon,
        n_rows=data.n_rows,
        n_markets=data.n_markets,
        slope=float("nan"),
        intercept=float("nan"),
        converged=False,
    )
    if data.n_rows < min_rows:
        return CellFit(**{**base.__dict__, "note": f"only {data.n_rows} rows"})
    if data.n_markets < min_markets:
        return CellFit(**{**base.__dict__, "note": f"only {data.n_markets} markets"})
    positives = int(data.y.sum())
    negatives = data.n_rows - positives
    if min(positives, negatives) < MIN_LABEL_BALANCE:
        return CellFit(**{**base.__dict__, "note": f"degenerate labels ({positives}/{negatives})"})

    x = logit(data.price)
    intercept, slope, converged = fit_logistic(x, data.y, weights=data.weight)
    if not converged or not np.isfinite(slope):
        return CellFit(**{**base.__dict__, "note": "fit did not converge"})

    predicted = expit(intercept + slope * x)
    boot = clustered_bootstrap_slope(x, data.y, data.market_id, n_boot=n_boot, seed=seed)
    ci, se = boot if boot is not None else (None, None)
    return CellFit(
        domain=data.domain,
        horizon=data.horizon,
        n_rows=data.n_rows,
        n_markets=data.n_markets,
        slope=slope,
        intercept=intercept,
        slope_ci=ci,
        slope_se=se,
        brier_model=brier(predicted, data.y),
        brier_baseline=brier(np.clip(data.price, 0.0, 1.0), data.y),
        converged=True,
    )


@dataclass
class FitReport:
    """All cells plus the pooled global fit."""

    cells: list[CellFit] = field(default_factory=list)
    global_slope: float = float("nan")
    global_intercept: float = float("nan")
    global_ci: tuple[float, float] | None = None
    n_rows: int = 0
    n_markets: int = 0

    def tradeable_cells(self) -> list[CellFit]:
        return [c for c in self.cells if c.tradeable]

    def by_key(self) -> dict[tuple[str, str], CellFit]:
        return {(c.domain, c.horizon): c for c in self.cells}
