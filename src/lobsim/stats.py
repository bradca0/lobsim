"""Statistical machinery for comparing policies honestly.

Three separate problems are handled here, and conflating them is the usual way backtest statistics
go wrong.

**Dependence.** Per-episode results are independent by construction -- distinct seeds, no shared
state -- so an ordinary i.i.d. bootstrap is correct for them. Within an episode, the
mark-to-market series is strongly autocorrelated, so a *stationary block* bootstrap
(Politis & Romano 1994) is used instead. Using the i.i.d. bootstrap on an autocorrelated series
would produce confidence intervals that are far too narrow.

**Pairing.** Every policy sees the same seeds, so comparisons are paired. The paired difference has
much lower variance than the difference of means, because the shared component -- was this a
volatile episode or a quiet one -- cancels.

**Selection.** Trying many configurations and reporting the best inflates the apparent Sharpe. The
deflated Sharpe ratio (Bailey & Lopez de Prado 2014) corrects for the number of trials, and the
Holm-Bonferroni step-down controls the family-wise error rate across the policy-vs-baseline
comparisons. The number of trials fed to the deflation is the *actual* number of configurations
fitted, recorded by the training script rather than asserted after the fact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """A paired comparison of one policy against a baseline."""

    name: str
    baseline: str
    mean_difference: float
    ci: ConfidenceInterval
    t_statistic: float
    p_value: float
    wilcoxon_p: float
    n: int

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "baseline": self.baseline,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci.low,
            "ci_high": self.ci.high,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "wilcoxon_p": self.wilcoxon_p,
            "n": self.n,
        }


def bootstrap_ci(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    n_boot: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for i.i.d. observations (one value per independent episode)."""
    sample = np.asarray(sample, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), level)
    rng = rng or np.random.default_rng(0)
    draws = rng.integers(0, sample.size, size=(n_boot, sample.size))
    estimates = np.array([statistic(sample[row]) for row in draws])
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        point=statistic(sample),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
        level=level,
    )


def stationary_bootstrap_sample(
    series: np.ndarray, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """One stationary-bootstrap resample (Politis & Romano 1994).

    Blocks have geometrically distributed lengths with mean ``mean_block`` and wrap around the end
    of the series, which is what makes the resampled series stationary. Preserving blocks preserves
    short-range dependence; an i.i.d. bootstrap would destroy it and understate the variance.
    """
    n = series.size
    if n == 0:
        return series
    p = 1.0 / max(mean_block, 1.0)
    indices = np.empty(n, dtype=np.int64)
    current = int(rng.integers(0, n))
    for i in range(n):
        indices[i] = current
        # Start a new block with probability p, otherwise continue the current one (wrapping).
        current = int(rng.integers(0, n)) if rng.random() < p else (current + 1) % n
    return series[indices]


def block_bootstrap_ci(
    series: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    mean_block: float = 20.0,
    n_boot: int = 2_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> ConfidenceInterval:
    """Percentile CI for a statistic of an autocorrelated series."""
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.size < 3:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), level)
    rng = rng or np.random.default_rng(0)
    estimates = np.array(
        [statistic(stationary_bootstrap_sample(series, mean_block, rng)) for _ in range(n_boot)]
    )
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        point=statistic(series),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
        level=level,
    )


def paired_comparison(
    policy_values: np.ndarray,
    baseline_values: np.ndarray,
    name: str,
    baseline: str,
    n_boot: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> ComparisonResult:
    """Compare two policies on the same seeds, using the paired differences throughout."""
    policy_values = np.asarray(policy_values, dtype=np.float64)
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    if policy_values.shape != baseline_values.shape:
        raise ValueError(
            f"paired comparison needs equal shapes, got {policy_values.shape} and "
            f"{baseline_values.shape}"
        )
    differences = policy_values - baseline_values
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        raise ValueError("need at least two paired observations")

    ci = bootstrap_ci(differences, n_boot=n_boot, level=level, rng=rng)
    t_statistic, p_value = scipy_stats.ttest_1samp(differences, 0.0)
    # The Wilcoxon signed-rank test is reported alongside the t-test because episode PnL is
    # heavy-tailed; if the two disagree, the t-test is the one to distrust.
    if np.all(differences == 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(scipy_stats.wilcoxon(differences).pvalue)

    return ComparisonResult(
        name=name,
        baseline=baseline,
        mean_difference=float(differences.mean()),
        ci=ci,
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        wilcoxon_p=wilcoxon_p,
        n=int(differences.size),
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm step-down adjustment controlling the family-wise error rate.

    Uniformly more powerful than plain Bonferroni at the same FWER, and it needs no independence
    assumption between the tests -- which matters here, because the comparisons share a baseline
    and are therefore strongly correlated.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, dict[str, float]] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        # Enforce monotonicity: an adjusted p-value may never decrease down the sorted list.
        running_max = max(running_max, value)
        adjusted[key] = {
            "p_value": p,
            "p_adjusted": running_max,
            "significant": float(running_max <= alpha),
            "rank": float(rank + 1),
        }
    return adjusted


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-observation Sharpe. Not annualised: the episode clock has no calendar meaning."""
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return float("nan")
    sd = float(returns.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(returns.mean() / sd)


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), correcting for skew and kurtosis of the return sample."""
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = returns.size
    observed = sharpe_ratio(returns)
    if n < 3 or not np.isfinite(observed):
        return float("nan")
    skew = float(scipy_stats.skew(returns))
    kurtosis = float(scipy_stats.kurtosis(returns, fisher=False))
    denominator = 1.0 - skew * observed + 0.25 * (kurtosis - 1.0) * observed**2
    if denominator <= 0.0:
        return float("nan")
    z = (observed - benchmark) * np.sqrt(n - 1) / np.sqrt(denominator)
    return float(scipy_stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe achievable by chance across ``n_trials`` independent attempts.

    This is the benchmark a strategy must clear to be interesting: with enough attempts, some
    configuration will look good on noise alone.
    """
    if n_trials < 2 or sharpe_variance <= 0.0:
        return 0.0
    sd = float(np.sqrt(sharpe_variance))
    gamma = EULER_MASCHERONI
    term_one = scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    term_two = scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - gamma) * term_one + gamma * term_two))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sharpe_variance: float | None = None,
) -> dict[str, float]:
    """Bailey & Lopez de Prado's deflated Sharpe ratio.

    Returns the observed Sharpe, the selection-adjusted benchmark it must beat, and the
    probability that the true Sharpe exceeds that benchmark. A DSR below ~0.95 means the result is
    not distinguishable from the best of ``n_trials`` lucky draws.

    ``sharpe_variance`` is the variance of Sharpe estimates *across the trials that were run*. When
    it is unknown, the conservative default of ``1 / (n - 1)`` -- the asymptotic variance of a
    Sharpe estimate under the null -- is used instead, which is stated explicitly rather than
    silently assumed.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = returns.size
    observed = sharpe_ratio(returns)
    if n < 3 or not np.isfinite(observed):
        return {
            "sharpe": float("nan"),
            "benchmark": float("nan"),
            "deflated_sharpe": float("nan"),
            "n_trials": float(n_trials),
        }
    variance = sharpe_variance if sharpe_variance is not None else 1.0 / (n - 1)
    benchmark = expected_max_sharpe(n_trials, variance)
    return {
        "sharpe": observed,
        "benchmark": benchmark,
        "deflated_sharpe": probabilistic_sharpe_ratio(returns, benchmark),
        "n_trials": float(n_trials),
        "sharpe_variance_used": float(variance),
    }
