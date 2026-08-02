"""Tests for the stylized-fact estimators.

Each estimator is checked against a process whose true value is known analytically, so a bug in
the estimator cannot be mistaken for a property of the simulator.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lobsim.validation import (
    StylizedFact,
    autocorrelation,
    depth_profile_hump,
    excess_kurtosis,
    hill_tail_index,
    log_returns,
    signature_plot,
    variance_ratio,
)


def gaussian_walk(n: int, sigma: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


class TestLogReturns:
    def test_returns_have_one_fewer_element(self) -> None:
        assert log_returns(np.array([1.0, 2.0, 4.0])).size == 2

    def test_doubling_gives_log_two(self) -> None:
        np.testing.assert_allclose(log_returns(np.array([1.0, 2.0])), [np.log(2.0)])

    def test_degenerate_input_is_empty_not_an_error(self) -> None:
        assert log_returns(np.array([5.0])).size == 0

    def test_non_positive_prices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            log_returns(np.array([1.0, 0.0, 2.0]))


class TestExcessKurtosis:
    def test_gaussian_sample_is_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        assert abs(excess_kurtosis(rng.normal(size=200_000))) < 0.1

    def test_student_t_is_strongly_positive(self) -> None:
        rng = np.random.default_rng(2)
        assert excess_kurtosis(rng.standard_t(df=4, size=200_000)) > 1.0

    def test_uniform_is_negative(self) -> None:
        rng = np.random.default_rng(3)
        assert excess_kurtosis(rng.uniform(size=200_000)) < -1.0

    def test_short_or_constant_input_is_nan(self) -> None:
        assert np.isnan(excess_kurtosis(np.array([1.0, 2.0])))
        assert np.isnan(excess_kurtosis(np.ones(50)))


class TestAutocorrelation:
    def test_white_noise_is_near_zero(self) -> None:
        rng = np.random.default_rng(4)
        assert abs(autocorrelation(rng.normal(size=100_000), lag=1)) < 0.02

    def test_ar1_recovers_its_coefficient(self) -> None:
        rng = np.random.default_rng(5)
        phi = 0.6
        x = np.zeros(200_000)
        noise = rng.normal(size=x.size)
        for i in range(1, x.size):
            x[i] = phi * x[i - 1] + noise[i]
        assert autocorrelation(x, lag=1) == pytest.approx(phi, abs=0.02)
        assert autocorrelation(x, lag=2) == pytest.approx(phi**2, abs=0.02)

    def test_rejects_non_positive_lag(self) -> None:
        with pytest.raises(ValueError, match="lag must be positive"):
            autocorrelation(np.arange(10.0), lag=0)

    def test_short_or_constant_series_is_nan(self) -> None:
        assert np.isnan(autocorrelation(np.arange(3.0), lag=5))
        assert np.isnan(autocorrelation(np.ones(100), lag=1))


class TestVarianceRatio:
    def test_random_walk_is_one(self) -> None:
        prices = gaussian_walk(200_000, seed=6)
        assert variance_ratio(prices, q=4) == pytest.approx(1.0, abs=0.05)
        assert variance_ratio(prices, q=16) == pytest.approx(1.0, abs=0.08)

    def test_mean_reverting_series_is_below_one(self) -> None:
        """Bid-ask bounce: alternating one-tick moves around a flat level."""
        rng = np.random.default_rng(7)
        bounce = 100.0 + 0.5 * rng.choice([-1.0, 1.0], size=100_000)
        assert variance_ratio(bounce, q=8) < 0.5

    def test_trending_series_is_above_one(self) -> None:
        rng = np.random.default_rng(8)
        momentum = np.zeros(100_000)
        step = rng.normal(size=momentum.size)
        for i in range(1, momentum.size):
            step[i] += 0.4 * step[i - 1]
            momentum[i] = momentum[i - 1] + step[i]
        assert variance_ratio(100.0 * np.exp(0.001 * momentum), q=8) > 1.3

    def test_rejects_degenerate_horizon(self) -> None:
        with pytest.raises(ValueError, match="q must be at least 2"):
            variance_ratio(gaussian_walk(100), q=1)

    def test_short_input_is_nan(self) -> None:
        assert np.isnan(variance_ratio(gaussian_walk(5), q=8))


class TestHillTailIndex:
    @pytest.mark.parametrize("alpha", [1.5, 2.0, 3.0])
    def test_recovers_a_known_pareto_exponent(self, alpha: float) -> None:
        rng = np.random.default_rng(9)
        sample = (1.0 - rng.random(400_000)) ** (-1.0 / alpha)
        assert hill_tail_index(sample) == pytest.approx(alpha, rel=0.1)

    def test_a_thin_tail_is_diagnosed_by_instability_in_the_tail_fraction(self) -> None:
        """An exponential has no power-law tail, and the estimator must reveal that.

        At any *fixed* tail fraction the Hill estimator returns a finite number for any sample --
        for an exponential at the top 5% it lands near 1/ln(20) ~ 3, indistinguishable from a
        genuine Pareto(3) if read naively. The distinguishing signature is stability: a true power
        law gives the same index at every tail depth, while a thin tail drifts upward without
        bound as the threshold rises. That is the property worth asserting.
        """
        rng = np.random.default_rng(10)
        exponential = rng.exponential(size=400_000)
        pareto = (1.0 - rng.random(400_000)) ** (-1.0 / 2.0)

        shallow_exp = hill_tail_index(exponential, tail_fraction=0.10)
        deep_exp = hill_tail_index(exponential, tail_fraction=0.005)
        assert deep_exp > 1.5 * shallow_exp, "an exponential tail must not look scale-free"

        shallow_par = hill_tail_index(pareto, tail_fraction=0.10)
        deep_par = hill_tail_index(pareto, tail_fraction=0.005)
        assert deep_par == pytest.approx(shallow_par, rel=0.15), "a Pareto tail must be stable"

    def test_small_samples_are_nan(self) -> None:
        assert np.isnan(hill_tail_index(np.arange(1.0, 50.0)))

    def test_degenerate_sample_is_nan(self) -> None:
        assert np.isnan(hill_tail_index(np.ones(1000)))


class TestDepthProfile:
    def test_finds_the_hump(self) -> None:
        assert depth_profile_hump(np.array([5.0, 9.0, 14.0, 11.0, 6.0])) == 2

    def test_monotone_decreasing_profile_peaks_at_the_touch(self) -> None:
        assert depth_profile_hump(np.array([20.0, 10.0, 5.0])) == 0

    def test_empty_profile_is_sentinel(self) -> None:
        assert depth_profile_hump(np.array([])) == -1


class TestSignaturePlot:
    def test_random_walk_is_flat_across_horizons(self) -> None:
        prices = gaussian_walk(200_000, sigma=0.01, seed=11)
        plot = signature_plot(prices, horizons=(1, 4, 16))
        values = [plot[h] for h in (1, 4, 16)]
        assert all(v == pytest.approx(1e-4, rel=0.15) for v in values)

    def test_microstructure_noise_slopes_downward(self) -> None:
        """A random walk observed with additive noise looks more volatile at short horizons."""
        rng = np.random.default_rng(12)
        efficient = np.cumsum(rng.normal(0.0, 0.01, 200_000))
        observed = 100.0 * np.exp(efficient + rng.normal(0.0, 0.02, efficient.size))
        plot = signature_plot(observed, horizons=(1, 32))
        assert plot[1] > 2 * plot[32]

    def test_insufficient_data_is_nan(self) -> None:
        assert np.isnan(signature_plot(gaussian_walk(10), horizons=(50,))[50])


class TestStylizedFactRecord:
    def _fact(self, value: float) -> StylizedFact:
        return StylizedFact(
            name="demo",
            value=value,
            target_low=1.0,
            target_high=2.0,
            units="ticks",
            description="d",
        )

    def test_pass_band_is_inclusive(self) -> None:
        assert self._fact(1.0).passes
        assert self._fact(2.0).passes
        assert self._fact(1.5).passes

    def test_values_outside_the_band_fail(self) -> None:
        assert not self._fact(0.99).passes
        assert not self._fact(2.01).passes

    def test_serialisation_round_trips_through_json(self) -> None:
        payload = json.loads(json.dumps(self._fact(1.5).to_dict()))
        assert payload["name"] == "demo"
        assert payload["value"] == 1.5
        assert payload["passes"] is True
