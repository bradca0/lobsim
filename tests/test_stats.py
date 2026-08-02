"""Statistics tests, checked against analytically known answers wherever one exists.

A bug in this module would not crash anything -- it would quietly produce confident-looking
intervals around the wrong number -- so every estimator is validated against a case whose correct
answer is known independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from lobsim.stats import (
    block_bootstrap_ci,
    bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_bonferroni,
    paired_comparison,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    stationary_bootstrap_sample,
)


class TestBootstrapCI:
    def test_interval_brackets_the_sample_mean(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(5.0, 2.0, 400)
        ci = bootstrap_ci(sample, n_boot=2000, rng=rng)
        assert ci.low < ci.point < ci.high
        assert ci.point == pytest.approx(sample.mean())

    def test_width_approximates_the_analytic_standard_error(self) -> None:
        """For a mean of i.i.d. normals the 95% CI half-width should be ~1.96 * sd / sqrt(n)."""
        rng = np.random.default_rng(1)
        sample = rng.normal(0.0, 3.0, 2000)
        ci = bootstrap_ci(sample, n_boot=4000, rng=rng)
        analytic = 1.96 * sample.std(ddof=1) / np.sqrt(sample.size)
        assert (ci.high - ci.low) / 2 == pytest.approx(analytic, rel=0.12)

    def test_coverage_is_close_to_nominal(self) -> None:
        """The property that matters: a 90% interval should contain the truth ~90% of the time."""
        rng = np.random.default_rng(2)
        covered = 0
        trials = 200
        for _ in range(trials):
            sample = rng.normal(1.0, 1.0, 120)
            ci = bootstrap_ci(sample, n_boot=400, level=0.90, rng=rng)
            covered += ci.low <= 1.0 <= ci.high
        assert 0.83 <= covered / trials <= 0.97

    def test_excludes_zero_flag(self) -> None:
        rng = np.random.default_rng(3)
        assert bootstrap_ci(rng.normal(10.0, 1.0, 200), n_boot=800, rng=rng).excludes_zero
        assert not bootstrap_ci(rng.normal(0.0, 1.0, 200), n_boot=800, rng=rng).excludes_zero

    def test_degenerate_input_is_nan_not_a_crash(self) -> None:
        ci = bootstrap_ci(np.array([1.0]))
        assert np.isnan(ci.point) and np.isnan(ci.low)

    def test_non_finite_values_are_dropped(self) -> None:
        sample = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        assert bootstrap_ci(sample, n_boot=200).point == pytest.approx(2.0)


class TestStationaryBootstrap:
    def test_resample_preserves_length_and_membership(self) -> None:
        rng = np.random.default_rng(4)
        series = np.arange(50.0)
        resampled = stationary_bootstrap_sample(series, 5.0, rng)
        assert resampled.size == series.size
        assert set(resampled).issubset(set(series))

    def test_blocks_preserve_autocorrelation(self) -> None:
        """The whole point: an i.i.d. bootstrap would destroy serial dependence, blocks keep it."""
        rng = np.random.default_rng(5)
        phi = 0.9
        series = np.zeros(4000)
        noise = rng.normal(size=series.size)
        for i in range(1, series.size):
            series[i] = phi * series[i - 1] + noise[i]

        def lag1(x: np.ndarray) -> float:
            centred = x - x.mean()
            return float((centred[:-1] * centred[1:]).sum() / (centred**2).sum())

        block = np.mean([lag1(stationary_bootstrap_sample(series, 50.0, rng)) for _ in range(30)])
        iid = np.mean([lag1(rng.permutation(series)) for _ in range(30)])
        assert block > 0.6
        assert abs(iid) < 0.1

    def test_block_ci_is_wider_than_the_iid_ci_for_dependent_data(self) -> None:
        """An i.i.d. bootstrap on autocorrelated data gives intervals that are too narrow."""
        rng = np.random.default_rng(6)
        series = np.zeros(2000)
        noise = rng.normal(size=series.size)
        for i in range(1, series.size):
            series[i] = 0.95 * series[i - 1] + noise[i]
        mean = lambda x: float(np.mean(x))  # noqa: E731
        block = block_bootstrap_ci(series, mean, mean_block=50.0, n_boot=300, rng=rng)
        iid = bootstrap_ci(series, mean, n_boot=300, rng=rng)
        assert (block.high - block.low) > 1.5 * (iid.high - iid.low)

    def test_short_series_is_nan(self) -> None:
        ci = block_bootstrap_ci(np.array([1.0, 2.0]), lambda x: float(np.mean(x)))
        assert np.isnan(ci.point)


class TestPairedComparison:
    def test_pairing_detects_a_shift_masked_by_common_noise(self) -> None:
        """The case pairing exists for: a small effect buried under large shared variation."""
        rng = np.random.default_rng(7)
        common = rng.normal(0.0, 50.0, 200)  # episode difficulty, shared by both policies
        baseline = common + rng.normal(0.0, 1.0, 200)
        policy = common + rng.normal(0.0, 1.0, 200) + 2.0
        result = paired_comparison(policy, baseline, "policy", "baseline", n_boot=2000, rng=rng)
        assert result.mean_difference == pytest.approx(2.0, abs=0.3)
        assert result.ci.excludes_zero
        assert result.p_value < 1e-10

    def test_no_effect_is_not_declared_significant(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(0.0, 1.0, 300)
        b = rng.normal(0.0, 1.0, 300)
        result = paired_comparison(a, b, "a", "b", n_boot=2000, rng=rng)
        assert not result.ci.excludes_zero
        assert result.p_value > 0.05

    def test_identical_inputs_are_reported_as_no_difference(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = paired_comparison(values, values, "a", "b", n_boot=200)
        assert result.mean_difference == 0.0
        assert result.wilcoxon_p == 1.0

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal shapes"):
            paired_comparison(np.zeros(4), np.zeros(5), "a", "b")

    def test_too_few_observations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            paired_comparison(np.zeros(1), np.zeros(1), "a", "b")


class TestHolmBonferroni:
    def test_matches_a_hand_computed_example(self) -> None:
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
        # Sorted: a=0.01 (x3), c=0.03 (x2), b=0.04 (x1) -> 0.03, 0.06, 0.06 after monotonicity.
        assert adjusted["a"]["p_adjusted"] == pytest.approx(0.03)
        assert adjusted["c"]["p_adjusted"] == pytest.approx(0.06)
        assert adjusted["b"]["p_adjusted"] == pytest.approx(0.06)

    def test_adjusted_values_are_monotone_in_rank(self) -> None:
        rng = np.random.default_rng(9)
        p_values = {f"t{i}": float(p) for i, p in enumerate(rng.random(20))}
        adjusted = holm_bonferroni(p_values)
        ordered = sorted(adjusted.values(), key=lambda d: d["rank"])
        series = [d["p_adjusted"] for d in ordered]
        assert series == sorted(series)

    def test_is_never_more_conservative_than_bonferroni(self) -> None:
        p_values = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        adjusted = holm_bonferroni(p_values)
        for key, p in p_values.items():
            assert adjusted[key]["p_adjusted"] <= min(1.0, len(p_values) * p) + 1e-12

    def test_significance_uses_the_adjusted_value(self) -> None:
        adjusted = holm_bonferroni({"a": 0.001, "b": 0.30}, alpha=0.05)
        assert adjusted["a"]["significant"] == 1.0
        assert adjusted["b"]["significant"] == 0.0

    def test_no_tests_is_empty(self) -> None:
        assert holm_bonferroni({}) == {}


class TestSharpeAndDeflation:
    def test_sharpe_matches_the_definition(self) -> None:
        returns = np.array([1.0, 2.0, 3.0, 4.0])
        assert sharpe_ratio(returns) == pytest.approx(returns.mean() / returns.std(ddof=1))

    def test_more_evidence_sharpens_the_verdict_in_both_directions(self) -> None:
        """More observations of the same Sharpe make the conclusion more confident.

        Deliberately checked in both directions. An earlier version of this test asserted only
        that PSR rises with sample size, and it failed because the drawn sample happened to have a
        *negative* Sharpe -- for which the correct behaviour is for PSR to fall toward 0. The
        estimator was right and the test was wrong, so the sign is now asserted explicitly rather
        than assumed.
        """
        rng = np.random.default_rng(10)

        positive = rng.normal(0.4, 1.0, 60)
        assert sharpe_ratio(positive) > 0, "precondition: this sample must have positive Sharpe"
        assert probabilistic_sharpe_ratio(np.tile(positive, 12)) > probabilistic_sharpe_ratio(
            positive
        )

        negative = -positive
        assert sharpe_ratio(negative) < 0
        assert probabilistic_sharpe_ratio(np.tile(negative, 12)) < probabilistic_sharpe_ratio(
            negative
        )

    def test_expected_max_sharpe_grows_with_the_number_of_trials(self) -> None:
        """More attempts means a higher bar, which is the entire point of the correction."""
        one = expected_max_sharpe(10, 0.01)
        many = expected_max_sharpe(1000, 0.01)
        assert 0.0 < one < many

    def test_a_single_trial_needs_no_deflation(self) -> None:
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_deflation_penalises_a_wide_search(self) -> None:
        rng = np.random.default_rng(11)
        returns = rng.normal(0.08, 1.0, 400)
        few = deflated_sharpe_ratio(returns, n_trials=2)
        many = deflated_sharpe_ratio(returns, n_trials=500)
        assert few["sharpe"] == many["sharpe"]  # the raw Sharpe is unchanged
        assert many["benchmark"] > few["benchmark"]
        assert many["deflated_sharpe"] < few["deflated_sharpe"]

    def test_pure_noise_does_not_survive_deflation(self) -> None:
        rng = np.random.default_rng(12)
        result = deflated_sharpe_ratio(rng.normal(0.0, 1.0, 300), n_trials=100)
        assert result["deflated_sharpe"] < 0.95

    def test_degenerate_inputs_are_nan(self) -> None:
        result = deflated_sharpe_ratio(np.array([1.0, 1.0]), n_trials=10)
        assert np.isnan(result["deflated_sharpe"])
        assert np.isnan(sharpe_ratio(np.ones(50)))
