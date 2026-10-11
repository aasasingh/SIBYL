"""Calibration engine tests.

The load-bearing test here is `test_recovers_known_slope`: data is generated
with a known slope b, and the fitter must recover it. Without that, a fitting
bug produces plausible-looking numbers that are silently wrong.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from sibyl.calibration.apply import Signal, recalibrate, recalibrated_probability
from sibyl.calibration.fit import (
    CellData,
    CellFit,
    brier,
    clustered_bootstrap_slope,
    expit,
    fit_cell,
    fit_logistic,
    logit,
)
from sibyl.calibration.hierarchical import partial_pool, population_slope_ci
from sibyl.calibration.walkforward import (
    Rows,
    _cluster_level_split,
    reliability_table,
    walk_forward,
)


def _synthetic_cell(
    *,
    slope: float,
    n_markets: int = 200,
    rows_per_market: int = 8,
    seed: int = 7,
    domain: str = "politics",
    horizon: str = "2d-1w",
) -> CellData:
    """Markets whose outcomes follow logit(P) = slope * logit(price) exactly.

    Each market has ONE outcome shared by all its rows — matching the real
    data's structure, where hourly snapshots of one market share one label.
    """
    rng = np.random.default_rng(seed)
    prices: list[float] = []
    ys: list[float] = []
    ids: list[str] = []
    for m in range(n_markets):
        base = float(rng.uniform(0.08, 0.92))
        true_p = float(expit(slope * logit(base)))
        outcome = float(rng.random() < true_p)
        for _ in range(rows_per_market):
            jitter = float(np.clip(base + rng.normal(0, 0.01), 0.02, 0.98))
            prices.append(jitter)
            ys.append(outcome)
            ids.append(f"M{m:04d}")
    return CellData(
        domain=domain,
        horizon=horizon,
        price=np.array(prices),
        y=np.array(ys),
        market_id=np.array(ids, dtype=np.str_),
    )


class TestFitter:
    def test_logit_expit_round_trip(self) -> None:
        p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        assert np.allclose(expit(logit(p)), p, atol=1e-9)

    def test_logit_clips_degenerate_prices(self) -> None:
        assert np.isfinite(logit(0.0)) and np.isfinite(logit(1.0))

    @pytest.mark.parametrize("true_slope", [0.7, 1.0, 1.5])
    def test_recovers_known_slope(self, true_slope: float) -> None:
        """The fitter must recover the slope it was given, within noise."""
        data = _synthetic_cell(slope=true_slope, n_markets=600, seed=11)
        _intercept, slope, converged = fit_logistic(logit(data.price), data.y)
        assert converged
        assert abs(slope - true_slope) < 0.2, f"recovered {slope:.3f}, expected {true_slope}"

    def test_unpenalised_fit_does_not_shrink_slope(self) -> None:
        """A regularised fit would bias b toward 0 and hide compression."""
        data = _synthetic_cell(slope=1.8, n_markets=800, seed=3)
        _i, slope, _c = fit_logistic(logit(data.price), data.y)
        assert slope > 1.4

    def test_degenerate_labels_do_not_converge_silently(self) -> None:
        x = np.linspace(-2, 2, 100)
        y = np.ones(100)  # separable: no finite MLE exists
        _i, slope, converged = fit_logistic(x, y)
        assert not converged or abs(slope) > 5


class TestClusteredBootstrap:
    def test_clustered_ci_is_wider_than_row_level(self) -> None:
        """The whole statistical point: rows within a market are correlated,
        so pretending they're independent gives a falsely narrow interval."""
        data = _synthetic_cell(slope=1.4, n_markets=120, rows_per_market=20, seed=5)
        x = logit(data.price)
        clustered = clustered_bootstrap_slope(x, data.y, data.market_id, n_boot=200)
        assert clustered is not None
        (c_low, c_high), _se = clustered

        # Same data, but each row labelled as its own "market" -> row-level.
        fake_ids = np.array([f"R{i}" for i in range(data.n_rows)], dtype=np.str_)
        row_level = clustered_bootstrap_slope(x, data.y, fake_ids, n_boot=200)
        assert row_level is not None
        (r_low, r_high), _ = row_level

        assert (c_high - c_low) > (r_high - r_low), (
            "clustered CI must be wider than the row-level CI; "
            f"clustered={c_high - c_low:.3f} row={r_high - r_low:.3f}"
        )

    def test_returns_none_for_too_few_markets(self) -> None:
        data = _synthetic_cell(slope=1.2, n_markets=5)
        assert clustered_bootstrap_slope(logit(data.price), data.y, data.market_id) is None


class TestCellFitGating:
    def test_thin_cell_is_not_tradeable_and_says_why(self) -> None:
        fit = fit_cell(_synthetic_cell(slope=1.5, n_markets=4, rows_per_market=2))
        assert not fit.converged
        assert not fit.tradeable
        assert fit.note

    def test_calibrated_market_is_not_tradeable(self) -> None:
        """b = 1 data must NOT be reported as an opportunity."""
        fit = fit_cell(_synthetic_cell(slope=1.0, n_markets=300, seed=21), n_boot=200)
        assert fit.converged
        assert fit.slope_ci is not None
        assert not fit.tradeable, f"b=1 data reported tradeable with CI {fit.slope_ci}"

    def test_strongly_compressed_market_is_tradeable(self) -> None:
        fit = fit_cell(_synthetic_cell(slope=2.2, n_markets=600, seed=31), n_boot=200)
        assert fit.converged and fit.slope_ci is not None
        assert fit.tradeable
        assert fit.slope_ci[0] > 1.0

    def test_brier_improvement_positive_when_prices_compressed(self) -> None:
        fit = fit_cell(_synthetic_cell(slope=2.0, n_markets=400, seed=41), n_boot=100)
        assert fit.brier_improvement > 0


class TestPartialPooling:
    def test_thin_cells_shrink_more_than_thick_ones(self) -> None:
        thick = fit_cell(_synthetic_cell(slope=1.6, n_markets=500, seed=1), n_boot=150)
        thin = fit_cell(
            _synthetic_cell(slope=0.4, n_markets=15, rows_per_market=4, seed=2, domain="weather"),
            n_boot=150,
        )
        extra = fit_cell(
            _synthetic_cell(slope=1.5, n_markets=400, seed=3, domain="crypto"), n_boot=150
        )
        pooled, mu, tau = partial_pool([thick, thin, extra])
        by_domain = {c.domain: c for c in pooled}
        assert np.isfinite(mu) and tau >= 0
        thin_p, thick_p = by_domain["weather"], by_domain["politics"]
        if thin_p.shrinkage is not None and thick_p.shrinkage is not None:
            assert thin_p.shrinkage > thick_p.shrinkage

    def test_heavily_shrunk_cell_is_not_tradeable_on_a_borrowed_interval(self) -> None:
        """Regression guard for a real incoherence in the first full run.

        A cell can have a CI excluding 1 while shrinkage pulls the slope it
        would actually trade back to the population mean. The interval then
        describes an estimate the system is not using, so the cell must not
        count as a cell-specific edge.
        """
        wide = CellFit(
            domain="sports",
            horizon="6-12h",
            n_rows=383,
            n_markets=186,
            slope=2.252,
            intercept=0.0,
            slope_ci=(1.437, 9.246),  # excludes 1
            slope_se=2.0,
            converged=True,
        )
        assert wide.tradeable, "unpooled, a CI excluding 1 is tradeable"
        borrowed = replace(wide, pooled_slope=1.184, shrinkage=0.99)
        assert not borrowed.tradeable
        assert "shrinkage" in borrowed.gating_note

        own = replace(wide, pooled_slope=2.0, shrinkage=0.2)
        assert own.tradeable, "a lightly shrunk cell keeps its own signal"

    def test_pooling_across_one_is_not_tradeable(self) -> None:
        """If pooling moves the slope to the other side of 1.0, the direction
        of the interval no longer matches the trade."""
        cell = CellFit(
            domain="crypto",
            horizon="1w-1mo",
            n_rows=500,
            n_markets=60,
            slope=0.70,
            intercept=0.0,
            slope_ci=(0.45, 0.95),  # excludes 1, says over-extended
            slope_se=0.1,
            converged=True,
            pooled_slope=1.05,  # pooling flipped it to compressed
            shrinkage=0.5,
        )
        assert not cell.tradeable

    def test_population_ci_reported_for_global_claim(self) -> None:
        fits = [
            fit_cell(_synthetic_cell(slope=1.5, n_markets=300, seed=s, domain=d), n_boot=120)
            for s, d in ((1, "politics"), (2, "crypto"), (3, "finance"))
        ]
        ci = population_slope_ci(fits)
        assert ci is not None
        low, high = ci
        assert low < high
        # Data generated at b=1.5 should put the population interval above 1.
        assert low > 1.0

    def test_failed_cell_inherits_population_slope(self) -> None:
        good = fit_cell(_synthetic_cell(slope=1.5, n_markets=300, seed=9), n_boot=100)
        good2 = fit_cell(
            _synthetic_cell(slope=1.4, n_markets=300, seed=10, domain="crypto"), n_boot=100
        )
        broken = CellFit(
            domain="weather",
            horizon="0-1h",
            n_rows=3,
            n_markets=1,
            slope=float("nan"),
            intercept=float("nan"),
            converged=False,
            note="too thin",
        )
        pooled, mu, _tau = partial_pool([good, good2, broken])
        failed = next(c for c in pooled if c.domain == "weather")
        assert failed.pooled_slope == pytest.approx(mu)
        assert failed.shrinkage == 1.0
        assert not failed.tradeable  # pooling never makes a dead cell tradeable


class TestApply:
    def test_recalibration_formula_matches_spec(self) -> None:
        # p* = p^b / (p^b + (1-p)^b)
        p, b = 0.70, 1.83
        expected = p**b / (p**b + (1 - p) ** b)
        assert recalibrated_probability(p, b) == pytest.approx(expected, abs=1e-9)

    def test_spec_worked_example_political_week_out(self) -> None:
        """§2.3.4: a 70c political contract a week out at b=1.83 implies ~83%."""
        assert recalibrated_probability(0.70, 1.83) == pytest.approx(0.83, abs=0.01)

    def test_slope_one_is_identity(self) -> None:
        for p in (0.1, 0.4, 0.9):
            assert recalibrated_probability(p, 1.0) == pytest.approx(p, abs=1e-9)

    def test_untradeable_cell_yields_no_signal(self) -> None:
        cell = fit_cell(_synthetic_cell(slope=1.0, n_markets=300, seed=77), n_boot=150)
        signal = recalibrate(Decimal("0.70"), cell, Decimal("0.72"))
        assert not signal.tradeable
        assert signal.edge == 0
        assert not signal.has_edge
        assert signal.reason

    def test_edge_uses_lower_bound_not_point_estimate(self) -> None:
        cell = fit_cell(_synthetic_cell(slope=2.2, n_markets=600, seed=31), n_boot=200)
        assert cell.tradeable
        signal = recalibrate(Decimal("0.70"), cell, Decimal("0.70"))
        assert signal.fair_value_low <= signal.fair_value <= signal.fair_value_high
        # edge must be computed off the conservative end
        assert signal.edge == max(
            Decimal("0"), signal.fair_value_low - signal.effective_cost
        ).quantize(Decimal("0.000001"))

    def test_expensive_cost_kills_edge(self) -> None:
        cell = fit_cell(_synthetic_cell(slope=2.2, n_markets=600, seed=31), n_boot=200)
        signal = recalibrate(Decimal("0.70"), cell, Decimal("0.99"))
        assert signal.edge == 0 and not signal.has_edge

    def test_signal_is_decimal_typed(self) -> None:
        cell = fit_cell(_synthetic_cell(slope=2.2, n_markets=600, seed=31), n_boot=150)
        signal = recalibrate(Decimal("0.70"), cell, Decimal("0.70"))
        assert isinstance(signal, Signal)
        for value in (signal.fair_value, signal.fair_value_low, signal.edge):
            assert isinstance(value, Decimal)


def _synthetic_rows(*, slope: float, n_markets: int = 400, seed: int = 5) -> Rows:
    rng = np.random.default_rng(seed)
    cell = _synthetic_cell(slope=slope, n_markets=n_markets, seed=seed)
    n = cell.n_rows
    # Market close times spread over two years so folds can split on time.
    market_order = {m: i for i, m in enumerate(np.unique(cell.market_id))}
    close = np.array(
        [1.75e9 + market_order[m] * 3600 * 12 for m in cell.market_id], dtype=np.float64
    )
    horizons = rng.choice(np.array(["24-48h", "2d-1w", "1w-1mo"]), size=n)
    return Rows(
        domain=np.array(["politics"] * n, dtype=np.str_),
        horizon=horizons.astype(np.str_),
        price=cell.price,
        y=cell.y,
        market_id=cell.market_id,
        close_time=close,
    )


class TestWalkForward:
    def test_folds_do_not_share_markets(self) -> None:
        rows = _synthetic_rows(slope=1.8, n_markets=300)
        results = walk_forward(rows, n_folds=3, n_boot=60)
        assert results
        for fold in results:
            assert fold.train_markets > 0 and fold.test_markets > 0

    def test_detects_real_signal_out_of_sample(self) -> None:
        rows = _synthetic_rows(slope=2.0, n_markets=600, seed=13)
        results = walk_forward(rows, n_folds=3, n_boot=60)
        assert results
        assert float(np.mean([f.improvement for f in results])) > 0

    def test_reports_no_improvement_on_calibrated_data(self) -> None:
        """Honesty check: on b=1 data, walk-forward must not claim a win."""
        rows = _synthetic_rows(slope=1.0, n_markets=600, seed=17)
        results = walk_forward(rows, n_folds=3, n_boot=60)
        assert results
        mean_improvement = float(np.mean([f.improvement for f in results]))
        assert mean_improvement < 0.004, f"claimed {mean_improvement:+.5f} on calibrated data"

    def test_reliability_table_shape(self) -> None:
        rows = _synthetic_rows(slope=1.5, n_markets=200)
        table = reliability_table(rows.price, rows.y, rows.market_id, n_bins=5)
        assert table and all(0 <= b.observed <= 1 for b in table)
        assert sum(b.n_rows for b in table) == len(rows)

    def test_reliability_reports_market_counts_not_just_rows(self) -> None:
        """A bin of many rows may rest on few independent outcomes; the table
        must expose that so a row-level gap isn't mistaken for an edge."""
        rows = _synthetic_rows(slope=1.5, n_markets=200)
        for b in reliability_table(rows.price, rows.y, rows.market_id, n_bins=5):
            assert b.n_markets <= b.n_rows
            assert 0 <= b.observed_market_level <= 1


def test_brier_bounds() -> None:
    y = np.array([1.0, 0.0])
    assert brier(np.array([1.0, 0.0]), y) == 0.0
    assert brier(np.array([0.0, 1.0]), y) == 1.0


class TestEventLevelClustering:
    """Markets inside an event share an outcome (NegRisk: exactly one YES),
    so the event is the independent unit, not the market."""

    @staticmethod
    def _event_structured_cell(markets_per_event: int = 10, n_events: int = 40) -> CellData:
        """Events x markets, where every market in an event shares the
        event's single outcome — the NegRisk dependence structure."""
        rng = np.random.default_rng(99)
        prices, ys, mids, cids = [], [], [], []
        for e in range(n_events):
            outcome = float(rng.random() < 0.5)
            for k in range(markets_per_event):
                base = float(rng.uniform(0.1, 0.9))
                for _ in range(6):
                    prices.append(base)
                    ys.append(outcome)  # shared across the whole event
                    mids.append(f"E{e:03d}M{k:02d}")
                    cids.append(f"E{e:03d}")
        return CellData(
            domain="politics",
            horizon="2d-1w",
            price=np.array(prices),
            y=np.array(ys),
            market_id=np.array(mids, dtype=np.str_),
            cluster_id=np.array(cids, dtype=np.str_),
        )

    def test_clustering_direction_is_not_assumed(self) -> None:
        """Deliberately NOT asserting that event clustering widens the CI.

        Both directions were observed in synthetic designs: when price
        variation lives WITHIN an event and the outcome is shared, each event
        is an informative within-event comparison and bundling can tighten
        the interval rather than widen it. The defensible claims are that the
        evidence is counted in events (below), that an event never straddles
        a fold (TestWalkForwardEventSplit), and whatever the real data shows
        -- measured in research/phase3-findings.md, not assumed here.
        """
        data = self._event_structured_cell()
        by_event = clustered_bootstrap_slope(logit(data.price), data.y, data.clusters, n_boot=150)
        by_market = clustered_bootstrap_slope(logit(data.price), data.y, data.market_id, n_boot=150)
        assert by_event is not None and by_market is not None
        # Both are finite intervals; the ordering is an empirical question.
        for (low, high), _se in (by_event, by_market):
            assert np.isfinite(low) and np.isfinite(high) and high > low

    def test_cell_counts_events_separately_from_markets(self) -> None:
        data = self._event_structured_cell()
        assert data.n_markets == 400
        assert data.n_clusters == 40
        fit = fit_cell(data, n_boot=100)
        assert fit.n_clusters == 40
        assert fit.n_markets == 400

    def test_cell_with_many_markets_but_few_events_is_rejected(self) -> None:
        """300 markets drawn from 3 events is 3 observations, not 300."""
        rng = np.random.default_rng(5)
        prices, ys, mids, cids = [], [], [], []
        for e in range(3):
            outcome = float(rng.random() < 0.5)
            for k in range(100):
                for _ in range(6):
                    prices.append(float(rng.uniform(0.1, 0.9)))
                    ys.append(outcome)
                    mids.append(f"E{e}M{k:03d}")
                    cids.append(f"E{e}")
        data = CellData(
            domain="politics",
            horizon="2d-1w",
            price=np.array(prices),
            y=np.array(ys),
            market_id=np.array(mids, dtype=np.str_),
            cluster_id=np.array(cids, dtype=np.str_),
        )
        fit = fit_cell(data, n_boot=100)
        assert not fit.tradeable
        assert "events" in fit.note


class TestWalkForwardEventSplit:
    def test_events_are_never_split_across_folds(self) -> None:
        """A NegRisk set straddling the boundary would hand the test fold its
        own answer: train on 'A won', and 'did B win?' is already resolved."""
        rng = np.random.default_rng(3)
        n_events, per_event, rows_per = 60, 5, 4
        domain, horizon, price, y, mid, cid, close = [], [], [], [], [], [], []
        for e in range(n_events):
            outcome = float(rng.random() < 0.5)
            t = 1.75e9 + e * 3600 * 24
            for k in range(per_event):
                for _ in range(rows_per):
                    domain.append("politics")
                    horizon.append("2d-1w")
                    price.append(float(rng.uniform(0.1, 0.9)))
                    y.append(outcome)
                    mid.append(f"E{e:03d}M{k}")
                    cid.append(f"E{e:03d}")
                    close.append(t)
        rows = Rows(
            domain=np.array(domain, dtype=np.str_),
            horizon=np.array(horizon, dtype=np.str_),
            price=np.array(price),
            y=np.array(y),
            market_id=np.array(mid, dtype=np.str_),
            close_time=np.array(close),
            cluster_id=np.array(cid, dtype=np.str_),
        )
        splits = _cluster_level_split(rows, n_folds=3, min_train_frac=0.4)
        assert splits
        for train_mask, test_mask, _ in splits:
            train_events = set(rows.clusters[train_mask].tolist())
            test_events = set(rows.clusters[test_mask].tolist())
            assert not (train_events & test_events), (
                f"event(s) in both folds: {sorted(train_events & test_events)[:3]}"
            )
