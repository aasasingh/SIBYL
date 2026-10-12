"""Power-analysis tests.

These exist because the first version of the power experiment ran ONE seed
per condition and the write-up then made a claim ("b >= 1.3 would have been
detected") that 80 seeds flatly contradicted. The lesson is encoded here: the
decision rule must match the report's, and the null condition must be
measured rather than assumed to be well behaved.
"""

from __future__ import annotations

import numpy as np

from sibyl.calibration.fit import fit_logistic, logit
from sibyl.calibration.power import (
    PowerResult,
    detected,
    expected_noise_floor,
    run_condition,
    smallest_detectable_slope,
    synthetic_rows,
)


class TestDecisionRule:
    def test_matches_the_report_rule(self) -> None:
        """mean > 0 AND more than half the folds positive."""
        assert detected([0.01, 0.01, -0.001, 0.005])
        # positive mean, but only half the folds positive
        assert not detected([0.05, -0.01, 0.05, -0.01][:2] + [-0.01, -0.01])
        # majority positive but negative mean
        assert not detected([0.001, 0.001, 0.001, -0.10])
        assert not detected([])

    def test_a_single_draw_can_mislead(self) -> None:
        """The methodological point, as an executable statement."""
        assert detected([0.002, 0.001, 0.003, -0.001])  # looks like signal
        assert not detected([-0.002, -0.001, 0.003, -0.001])  # same effect, new draw


class TestSyntheticGenerator:
    def test_event_structure_is_reproduced(self) -> None:
        rows = synthetic_rows(
            true_slope=1.5, noise_sd=0.3, n_events=50, markets_per_event=4, rows_per_market=6
        )
        assert np.unique(rows.clusters).size == 50
        assert np.unique(rows.market_id).size == 200
        assert len(rows) == 50 * 4 * 6

    def test_outcomes_within_an_event_are_dependent(self) -> None:
        """One draw per event, so outcomes are COMONOTONIC in price.

        Not identical: markets in an event have different prices, hence
        different win probabilities, so a single event draw resolves the
        expensive ones YES and the cheap ones NO. The testable property is
        the threshold structure -- within an event, every market priced above
        the winner's price must also have won.
        """
        rows = synthetic_rows(
            true_slope=1.5, noise_sd=0.0, n_events=40, markets_per_event=6, rows_per_market=1
        )
        checked = 0
        for event in np.unique(rows.clusters):
            mask = rows.clusters == event
            prices, outcomes = rows.price[mask], rows.y[mask]
            order = np.argsort(prices)
            sorted_outcomes = outcomes[order]
            # A comonotonic draw makes the winners a suffix of the
            # price-sorted list: no 0 may appear above a 1.
            first_win = (
                np.argmax(sorted_outcomes) if sorted_outcomes.any() else len(sorted_outcomes)
            )
            assert np.all(sorted_outcomes[first_win:] == 1.0), (
                f"event {event} outcomes not monotone in price: {sorted_outcomes}"
            )
            checked += 1
        assert checked == 40

    def test_independent_mode_breaks_that_dependence(self) -> None:
        """The flag must actually change the generative structure."""
        shared = synthetic_rows(
            true_slope=1.0,
            noise_sd=0.0,
            n_events=300,
            markets_per_event=4,
            rows_per_market=1,
            seed=5,
            shared_event_outcome=True,
        )
        independent = synthetic_rows(
            true_slope=1.0,
            noise_sd=0.0,
            n_events=300,
            markets_per_event=4,
            rows_per_market=1,
            seed=5,
            shared_event_outcome=False,
        )

        def within_event_spread(rows: object) -> float:
            r = rows  # typed loosely for the helper
            per_event = [
                r.y[r.clusters == e].std()  # type: ignore[attr-defined]
                for e in np.unique(r.clusters)  # type: ignore[attr-defined]
            ]
            return float(np.mean(per_event))

        # Independent outcomes vary more within an event than shared ones.
        assert within_event_spread(independent) > within_event_spread(shared)

    def test_noise_attenuates_the_recoverable_slope(self) -> None:
        clean = synthetic_rows(true_slope=2.0, noise_sd=0.0, n_events=400, seed=3)
        noisy = synthetic_rows(true_slope=2.0, noise_sd=0.8, n_events=400, seed=3)
        _i, b_clean, _c = fit_logistic(logit(clean.price), clean.y)
        _i2, b_noisy, _c2 = fit_logistic(logit(noisy.price), noisy.y)
        assert b_clean > b_noisy

    def test_seed_is_deterministic(self) -> None:
        a = synthetic_rows(true_slope=1.4, noise_sd=0.3, n_events=30, seed=99)
        b = synthetic_rows(true_slope=1.4, noise_sd=0.3, n_events=30, seed=99)
        assert np.array_equal(a.price, b.price)
        assert np.array_equal(a.y, b.y)


class TestRunCondition:
    def test_aggregates_many_seeds(self) -> None:
        result = run_condition(true_slope=1.0, noise_sd=0.3, n_seeds=5, n_events=60, n_boot=20)
        assert result.n_seeds == 5
        assert 0 <= result.detections <= 5
        assert 0.0 <= result.detection_rate <= 1.0
        assert result.is_null  # b = 1 is the null

    def test_strong_signal_is_detected_more_often_than_the_null(self) -> None:
        """The only ordering the experiment must guarantee."""
        null = run_condition(true_slope=1.0, noise_sd=0.3, n_seeds=12, n_events=200, n_boot=20)
        strong = run_condition(true_slope=2.5, noise_sd=0.3, n_seeds=12, n_events=200, n_boot=20)
        assert strong.detection_rate > null.detection_rate

    def test_null_flag_only_for_slope_one(self) -> None:
        assert run_condition(
            true_slope=1.0, noise_sd=0.0, n_seeds=2, n_events=40, n_boot=20
        ).is_null
        assert not run_condition(
            true_slope=1.4, noise_sd=0.0, n_seeds=2, n_events=40, n_boot=20
        ).is_null


def _result(slope: float, rate: float, n: int = 100) -> PowerResult:
    return PowerResult(
        true_slope=slope,
        noise_sd=0.4,
        n_seeds=n,
        detections=int(rate * n),
        mean_fitted_slope=slope,
        mean_improvement=0.001,
        improvement_sd=0.001,
    )


class TestSummaryHelpers:
    def test_smallest_detectable_skips_the_null(self) -> None:
        results = [_result(1.0, 0.99), _result(1.3, 0.2), _result(2.0, 0.95)]
        assert smallest_detectable_slope(results, threshold=0.8) == 2.0

    def test_returns_none_when_nothing_reaches_the_threshold(self) -> None:
        """The honest answer when the procedure simply lacks power — no
        extrapolating a curve to invent a bound."""
        results = [_result(1.0, 0.4), _result(1.3, 0.2), _result(2.0, 0.5)]
        assert smallest_detectable_slope(results, threshold=0.8) is None

    def test_noise_floor_comes_from_the_null_row(self) -> None:
        results = [_result(1.0, 0.4), _result(2.0, 0.9)]
        assert expected_noise_floor(results) == 0.001
        assert expected_noise_floor([_result(2.0, 0.9)]) is None

    def test_describe_labels_the_null_as_false_positives(self) -> None:
        assert "FALSE POSITIVE" in _result(1.0, 0.44).describe()
        assert "detection rate" in _result(1.5, 0.62).describe()
