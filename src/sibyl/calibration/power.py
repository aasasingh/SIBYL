"""Power analysis for the Phase 3 walk-forward test.

The first version of this experiment ran ONE simulation per condition and the
write-up then claimed "a true slope >= 1.3 would still have been detected".
That claim rested on a single random draw, and the experiment's own `b = 1.0`
row came out positive in 4 of 4 folds -- a false positive that showed exactly
how far one draw can mislead.

So: many seeds per condition, reporting

  * **detection rate** -- P(walk-forward says "improvement" | true slope > 1),
    i.e. statistical power; and
  * **false positive rate** -- P(walk-forward says "improvement" | b = 1),
    which is the row that calibrates how much to believe the rest.

The decision rule mirrors `report.render` exactly: a condition "detects" when
mean improvement > 0 AND more than half the folds are positive. Using a
different rule here would measure the power of a test we do not run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sibyl.calibration.fit import expit, fit_logistic, logit
from sibyl.calibration.walkforward import Rows, walk_forward


@dataclass(frozen=True)
class PowerResult:
    true_slope: float
    noise_sd: float
    n_seeds: int
    detections: int
    mean_fitted_slope: float
    mean_improvement: float
    improvement_sd: float

    @property
    def detection_rate(self) -> float:
        return self.detections / self.n_seeds if self.n_seeds else float("nan")

    @property
    def is_null(self) -> bool:
        """b = 1 means no compression, so detections here are false positives."""
        return abs(self.true_slope - 1.0) < 1e-9

    def describe(self) -> str:
        label = "FALSE POSITIVE rate" if self.is_null else "detection rate"
        return (
            f"b={self.true_slope:<4.1f} noise={self.noise_sd:<5.3f} "
            f"fitted={self.mean_fitted_slope:<6.3f} "
            f"improvement={self.mean_improvement:+.6f} (sd {self.improvement_sd:.6f})  "
            f"{label} {self.detection_rate:>5.1%} ({self.detections}/{self.n_seeds})"
        )


def synthetic_rows(
    *,
    true_slope: float,
    noise_sd: float,
    n_events: int = 400,
    markets_per_event: int = 3,
    rows_per_market: int = 12,
    drift_sd: float = 0.05,
    seed: int = 7,
    shared_event_outcome: bool = True,
) -> Rows:
    """Markets whose outcomes follow a known slope, observed through noise.

    With  (the default), one uniform draw per event
    decides every market in it, so outcomes within an event are dependent and
    the effective sample is the event count rather than the market count.
    That dependence is what makes the real data carry less evidence than its
    row count suggests, and leaving it out made an earlier version of this
    experiment report optimistic power.

    It is an APPROXIMATION, and in a specific direction: this makes outcomes
    within an event comonotonic, whereas a real NegRisk set is mutually
    exclusive (exactly one YES), i.e. negatively dependent. Both shrink the
    independent evidence below the market count; they are not the same
    structure. Treat the resulting power figures as the right order of
    magnitude, not as exact.
    """
    rng = np.random.default_rng(seed)
    domain: list[str] = []
    horizon: list[str] = []
    price: list[float] = []
    y: list[float] = []
    market: list[str] = []
    cluster: list[str] = []
    close: list[float] = []
    for e in range(n_events):
        event_close = 1.75e9 + e * 3600 * 18
        # One draw per event when outcomes are shared; the per-market
        # probability still follows the slope, so the fitted relationship is
        # unchanged while the independent-evidence count falls.
        event_draw = float(rng.random())
        for k in range(markets_per_event):
            base = float(rng.uniform(0.08, 0.92))
            true_p = float(expit(true_slope * logit(base)))
            draw = event_draw if shared_event_outcome else float(rng.random())
            outcome = float(draw < true_p)
            walk = float(logit(base))
            for _ in range(rows_per_market):
                walk += float(rng.normal(0, drift_sd))
                observed = walk + float(rng.normal(0, noise_sd))
                price.append(float(np.clip(expit(observed), 0.01, 0.99)))
                y.append(outcome)
                domain.append("politics")
                horizon.append("2d-1w")
                market.append(f"E{e:04d}M{k}")
                cluster.append(f"E{e:04d}")
                close.append(event_close)
    return Rows(
        domain=np.array(domain, dtype=np.str_),
        horizon=np.array(horizon, dtype=np.str_),
        price=np.array(price, dtype=np.float64),
        y=np.array(y, dtype=np.float64),
        market_id=np.array(market, dtype=np.str_),
        close_time=np.array(close, dtype=np.float64),
        cluster_id=np.array(cluster, dtype=np.str_),
    )


def detected(improvements: list[float]) -> bool:
    """The decision rule used by the real report, applied verbatim."""
    if not improvements:
        return False
    mean = float(np.mean(improvements))
    wins = sum(1 for i in improvements if i > 0)
    return mean > 0 and wins > len(improvements) / 2


def run_condition(
    *,
    true_slope: float,
    noise_sd: float,
    n_seeds: int = 60,
    n_folds: int = 4,
    n_boot: int = 40,
    base_seed: int = 20261011,
    n_events: int = 400,
    markets_per_event: int = 3,
    rows_per_market: int = 12,
    shared_event_outcome: bool = True,
) -> PowerResult:
    """Repeat the whole walk-forward procedure across seeds."""
    rng = np.random.default_rng(base_seed)
    detections = 0
    fitted: list[float] = []
    means: list[float] = []
    for _ in range(n_seeds):
        seed = int(rng.integers(1, 2**31 - 1))
        rows = synthetic_rows(
            true_slope=true_slope,
            noise_sd=noise_sd,
            seed=seed,
            n_events=n_events,
            markets_per_event=markets_per_event,
            rows_per_market=rows_per_market,
            shared_event_outcome=shared_event_outcome,
        )
        _intercept, slope, converged = fit_logistic(logit(rows.price), rows.y)
        if converged and np.isfinite(slope):
            fitted.append(slope)
        improvements = [f.improvement for f in walk_forward(rows, n_folds=n_folds, n_boot=n_boot)]
        if improvements:
            means.append(float(np.mean(improvements)))
        detections += bool(detected(improvements))
    return PowerResult(
        true_slope=true_slope,
        noise_sd=noise_sd,
        n_seeds=n_seeds,
        detections=detections,
        mean_fitted_slope=float(np.mean(fitted)) if fitted else float("nan"),
        mean_improvement=float(np.mean(means)) if means else float("nan"),
        improvement_sd=float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"),
    )


def power_curve(
    *,
    slopes: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0),
    noise_sd: float,
    n_seeds: int = 60,
    n_folds: int = 4,
    n_boot: int = 40,
    n_events: int = 400,
) -> list[PowerResult]:
    """Detection rate across true slopes at one noise level."""
    return [
        run_condition(
            true_slope=s,
            noise_sd=noise_sd,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_boot=n_boot,
            n_events=n_events,
        )
        for s in slopes
    ]


def smallest_detectable_slope(
    results: list[PowerResult], *, threshold: float = 0.80
) -> float | None:
    """Lowest true slope whose detection rate reaches `threshold`.

    None means no tested slope reached it -- the honest answer when the
    procedure simply lacks the power, rather than extrapolating a curve.
    """
    for r in sorted(results, key=lambda r: r.true_slope):
        if not r.is_null and r.detection_rate >= threshold:
            return r.true_slope
    return None


def expected_noise_floor(results: list[PowerResult]) -> float | None:
    """Mean improvement observed under the null, i.e. the noise floor."""
    for r in results:
        if r.is_null:
            return r.mean_improvement
    return None
