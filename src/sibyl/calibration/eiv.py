"""Errors-in-variables: how much slope is the tape hiding?

The Phase 3 write-up originally claimed that tape-vs-quote bias "cuts one
way" -- that a negative result on optimistic data is a stronger negative.
That is true for a P&L backtest and FALSE for a calibration fit, and the
distinction matters enough to measure.

The model being fitted is

    logit P(y=1) = a + b * logit(p)

with `logit(p)` as the regressor. Classical measurement error in a regressor
ATTENUATES its coefficient toward zero:

    plim b_hat = lambda * b_true,    lambda = Var(x_true) / Var(x_observed)

So noisy tape prices push the estimated slope toward 0, i.e. toward "no
compression". A real b > 1 could be measured as b ~ 1. The kill could be a
false negative, and saying otherwise was wrong.

Two things this module does:

1. **Estimates the noise** from the data instead of assuming it. For a series
   that is a random walk plus independent noise,

       Var(x_{t+k} - x_t) = k * sigma2_signal + 2 * sigma2_noise

   so regressing the mean squared k-step difference on k gives an intercept
   of 2*sigma2_noise. This is the standard variance-ratio identification and
   it works on the hourly bars already in the ledger.

2. **Corrects the slope** for the measured attenuation, giving the latent
   slope the tape is consistent with.

What this does NOT do is turn a corrected slope into a tradeable edge, and
the reason is the crux:

    For PREDICTION from an equally noisy input, the attenuated coefficient is
    the CORRECT one. If you will observe a noisy price at trade time, then
    b_hat -- not b_corrected -- is what maps it to a calibrated probability.
    EIV correction answers "is the market really compressed?" (inference),
    not "what should I multiply this quote by?" (prediction).

The problem is therefore not attenuation as such, but a MISMATCH: trained on
tape noise, deployed against quote noise. Those have different variances, so
neither b_hat nor b_corrected is the right deployment coefficient, and no
amount of arithmetic on tape data can produce it. Only recorded books can --
which is what `sibyl record` exists to collect.

Assumptions, stated because they are load-bearing: classical (independent,
additive) measurement error in logit space; a random-walk signal within a
market; and noise that is stationary over a market's life. Each is an
approximation, so treat the output as an order of magnitude, not a point
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sibyl.calibration.fit import CellData, logit

FloatArray = npt.NDArray[np.float64]

# Lags used for the variance-ratio regression. Short lags carry the noise
# signature; long lags are dominated by genuine drift and by nonstationarity.
DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
MIN_SERIES_LEN = 8


@dataclass(frozen=True)
class NoiseEstimate:
    """Measurement-error decomposition for one cell."""

    sigma2_noise: float  # variance of the per-observation noise in logit(p)
    sigma2_signal_step: float  # per-step variance of the latent walk
    var_observed: float  # cross-sectional variance of observed logit(p)
    n_series: int
    n_pairs: int
    r2: float  # fit quality of the variance-ratio regression

    @property
    def reliability(self) -> float:
        """lambda = Var(x_true)/Var(x_obs), the attenuation factor in (0, 1].

        Clamped: a noise estimate exceeding the total observed variance is a
        model failure, not a negative true variance.
        """
        if self.var_observed <= 0:
            return 1.0
        lam = (self.var_observed - self.sigma2_noise) / self.var_observed
        return float(min(max(lam, 1e-3), 1.0))

    @property
    def noise_share(self) -> float:
        """Fraction of observed regressor variance that is measurement error."""
        return 1.0 - self.reliability

    def corrected_slope(self, observed_slope: float) -> float:
        """b_true implied by an observed (attenuated) slope."""
        return observed_slope / self.reliability


def _series_by_group(values: FloatArray, groups: npt.NDArray[np.str_]) -> list[FloatArray]:
    """Split `values` into per-group sequences, preserving input order.

    Input order is time order: rows come from the ledger sorted by bar close
    within a market, so no re-sort is applied here.
    """
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_values = values[order]
    unique = np.unique(sorted_groups)
    boundaries = np.searchsorted(sorted_groups, unique)
    return [s for s in np.split(sorted_values, boundaries[1:]) if s.size >= MIN_SERIES_LEN]


def estimate_noise(
    price: FloatArray,
    market_id: npt.NDArray[np.str_],
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> NoiseEstimate | None:
    """Variance-ratio estimate of measurement-error variance in logit(price).

    Returns None when there are too few usable series to identify it.
    """
    x = logit(price)
    series = _series_by_group(x, market_id)
    if len(series) < 5:
        return None

    lag_values: list[float] = []
    mean_sq: list[float] = []
    total_pairs = 0
    for lag in lags:
        diffs: list[FloatArray] = []
        for s in series:
            if s.size > lag:
                diffs.append(s[lag:] - s[:-lag])
        if not diffs:
            continue
        pooled = np.concatenate(diffs)
        if pooled.size < 20:
            continue
        lag_values.append(float(lag))
        mean_sq.append(float(np.mean(pooled**2)))
        total_pairs += int(pooled.size)
    if len(lag_values) < 3:
        return None

    k = np.array(lag_values)
    v = np.array(mean_sq)
    # v(k) = 2*sigma2_noise + k*sigma2_signal_step
    design = np.column_stack([np.ones_like(k), k])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    intercept, slope_k = float(coef[0]), float(coef[1])
    predicted = design @ coef
    ss_res = float(np.sum((v - predicted) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return NoiseEstimate(
        # A negative intercept means no detectable noise, not negative variance.
        sigma2_noise=max(intercept / 2.0, 0.0),
        sigma2_signal_step=max(slope_k, 0.0),
        var_observed=float(np.var(x)),
        n_series=len(series),
        n_pairs=total_pairs,
        r2=r2,
    )


@dataclass(frozen=True)
class AttenuationResult:
    cell: str
    n_rows: int
    observed_slope: float
    reliability: float
    corrected_slope: float
    noise_sd: float
    r2: float

    @property
    def crosses_one(self) -> bool:
        """True when correction moves the slope from <=1 to >1.

        These are the cells where the kill could be a measurement artifact
        rather than an absence of compression.
        """
        return self.observed_slope <= 1.0 < self.corrected_slope


def attenuation_for_cell(data: CellData, observed_slope: float) -> AttenuationResult | None:
    noise = estimate_noise(data.price, data.market_id)
    if noise is None:
        return None
    return AttenuationResult(
        cell=f"{data.domain}/{data.horizon}",
        n_rows=data.n_rows,
        observed_slope=observed_slope,
        reliability=noise.reliability,
        corrected_slope=noise.corrected_slope(observed_slope),
        noise_sd=float(np.sqrt(noise.sigma2_noise)),
        r2=noise.r2,
    )
