"""Errors-in-variables tests.

Measurement error in logit(p) attenuates the fitted slope toward zero, so a
real effect can be measured as none. These tests pin that the estimator
recovers a KNOWN injected noise level and a KNOWN attenuation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sibyl.calibration.eiv import attenuation_for_cell, estimate_noise
from sibyl.calibration.fit import CellData, expit, fit_logistic, logit


def _series(
    *,
    n_markets: int,
    length: int,
    noise_sd: float,
    drift_sd: float = 0.05,
    seed: int = 11,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_]]:
    """Random-walk latent price plus iid logit-space measurement noise."""
    rng = np.random.default_rng(seed)
    prices: list[float] = []
    ids: list[str] = []
    for m in range(n_markets):
        walk = float(rng.uniform(-2.0, 2.0))
        for _ in range(length):
            walk += float(rng.normal(0, drift_sd))
            observed = walk + float(rng.normal(0, noise_sd))
            prices.append(float(np.clip(expit(observed), 0.01, 0.99)))
            ids.append(f"M{m:04d}")
    return np.array(prices), np.array(ids, dtype=np.str_)


class TestNoiseEstimation:
    @pytest.mark.parametrize("true_noise_sd", [0.2, 0.5])
    def test_recovers_injected_noise_level(self, true_noise_sd: float) -> None:
        price, ids = _series(n_markets=200, length=40, noise_sd=true_noise_sd)
        est = estimate_noise(price, ids)
        assert est is not None
        recovered = float(np.sqrt(est.sigma2_noise))
        # Clipping at the 0.01/0.99 price bounds biases this slightly; an
        # order-of-magnitude recovery is what the method claims.
        assert recovered == pytest.approx(true_noise_sd, rel=0.45), (
            f"recovered {recovered:.3f} for injected {true_noise_sd}"
        )

    def test_no_noise_gives_reliability_near_one(self) -> None:
        price, ids = _series(n_markets=150, length=40, noise_sd=0.0)
        est = estimate_noise(price, ids)
        assert est is not None
        assert est.reliability > 0.95
        assert est.noise_share < 0.05

    def test_more_noise_lowers_reliability(self) -> None:
        low = estimate_noise(*_series(n_markets=150, length=40, noise_sd=0.1))
        high = estimate_noise(*_series(n_markets=150, length=40, noise_sd=0.8))
        assert low is not None and high is not None
        assert high.reliability < low.reliability

    def test_too_few_series_returns_none(self) -> None:
        price, ids = _series(n_markets=2, length=40, noise_sd=0.3)
        assert estimate_noise(price, ids) is None

    def test_reliability_is_bounded(self) -> None:
        price, ids = _series(n_markets=100, length=30, noise_sd=3.0)
        est = estimate_noise(price, ids)
        assert est is not None
        assert 0.0 < est.reliability <= 1.0


class TestAttenuationCorrection:
    def test_correction_raises_the_slope(self) -> None:
        """Attenuation pulls toward 0, so the correction must push away."""
        price, ids = _series(n_markets=200, length=30, noise_sd=0.5)
        y = np.asarray(np.random.default_rng(1).random(price.size) < price, dtype=np.float64)
        data = CellData(domain="politics", horizon="2d-1w", price=price, y=y, market_id=ids)
        result = attenuation_for_cell(data, observed_slope=1.1)
        assert result is not None
        assert result.corrected_slope > result.observed_slope
        assert 0 < result.reliability <= 1

    def test_zero_noise_leaves_the_slope_alone(self) -> None:
        price, ids = _series(n_markets=150, length=30, noise_sd=0.0)
        data = CellData(
            domain="politics",
            horizon="2d-1w",
            price=price,
            y=np.zeros(price.size),
            market_id=ids,
        )
        result = attenuation_for_cell(data, observed_slope=1.2)
        assert result is not None
        assert result.corrected_slope == pytest.approx(1.2, rel=0.06)

    def test_crosses_one_flags_only_a_verdict_change(self) -> None:
        price, ids = _series(n_markets=150, length=30, noise_sd=0.6)
        data = CellData(
            domain="politics",
            horizon="2d-1w",
            price=price,
            y=np.zeros(price.size),
            market_id=ids,
        )
        below = attenuation_for_cell(data, observed_slope=0.99)
        above = attenuation_for_cell(data, observed_slope=1.40)
        assert below is not None and above is not None
        # A slope already above 1 cannot "cross" it.
        assert not above.crosses_one
        assert below.crosses_one == (below.corrected_slope > 1.0)


def test_attenuation_is_real_but_small_when_spread_dominates() -> None:
    """The quantitative reason the Phase 3 kill survives EIV.

    Attenuation depends on noise RELATIVE to the regressor's spread. Across
    markets, logit(p) spans roughly +/-4.6, so its variance dwarfs
    within-market noise and lambda stays near 1.
    """
    price, ids = _series(n_markets=300, length=30, noise_sd=0.43, drift_sd=0.05)
    est = estimate_noise(price, ids)
    assert est is not None
    assert est.var_observed > 1.0
    assert est.reliability > 0.8
    # A 1.20 observed slope cannot be a disguised 2.0.
    assert est.corrected_slope(1.20) < 1.6


def test_fitted_slope_is_attenuated_by_injected_noise() -> None:
    """End-to-end: inject noise into a known relationship, watch b fall."""
    rng = np.random.default_rng(7)
    true_b = 2.0
    base = rng.uniform(0.08, 0.92, size=6000)
    true_p = expit(true_b * logit(base))
    y = np.asarray(rng.random(base.size) < true_p, dtype=np.float64)

    _i, clean, _c = fit_logistic(logit(base), y)
    noisy_x = logit(base) + rng.normal(0, 0.8, size=base.size)
    _i2, noisy, _c2 = fit_logistic(noisy_x, y)
    assert clean > noisy, f"noise must attenuate: clean={clean:.3f} noisy={noisy:.3f}"
