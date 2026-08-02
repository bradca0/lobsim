"""Estimators for the stylized facts a simulated market should reproduce.

A simulator's fidelity is a *measurement*, not a claim. Each function here estimates one property
that is well documented in the empirical microstructure literature; `scripts/run_validation.py`
runs them over many independent episodes, and the README reports the results including the ones
that fail. Nothing in this module knows about the market maker -- validation is done on an
agentless market so that the agent cannot flatter it.

The facts checked, and why each one matters for this project specifically:

* **Fat-tailed returns** -- excess kurtosis well above the Gaussian value of 0. If returns were
  Gaussian, inventory risk would be far easier to hedge than it is in reality.
* **Volatility clustering** -- positive, slowly decaying autocorrelation of |returns| even though
  raw returns are close to uncorrelated. This is what makes a fixed-spread quote sometimes badly
  mispriced for long stretches.
* **Near-martingale mid** -- a variance ratio near 1 at longer horizons. A mid that mean-reverts
  would hand a market maker free money and would invalidate every PnL number downstream, so this
  is the single most important check in the file.
* **Concave depth profile** -- resting volume is not maximal at the touch but a few ticks behind
  it, the well-known hump shape.
* **Power-law order sizes** -- a Hill tail-index estimate in the 1.5-3 range reported in equity
  and futures data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StylizedFact:
    """One measured property, with the empirical target it is being judged against."""

    name: str
    value: float
    target_low: float
    target_high: float
    units: str
    description: str

    @property
    def passes(self) -> bool:
        return self.target_low <= self.value <= self.target_high

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": float(self.value),
            "target_low": self.target_low,
            "target_high": self.target_high,
            "units": self.units,
            "description": self.description,
            "passes": self.passes,
        }


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns of a strictly positive price series."""
    prices = np.asarray(prices, dtype=np.float64)
    if prices.size < 2:
        return np.empty(0, dtype=np.float64)
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive to take log returns")
    return np.diff(np.log(prices))


def excess_kurtosis(returns: np.ndarray) -> float:
    """Fisher excess kurtosis. Zero for a Gaussian; real return series sit well above it."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 4:
        return float("nan")
    centred = returns - returns.mean()
    variance = float((centred**2).mean())
    if variance <= 0.0:
        return float("nan")
    return float((centred**4).mean() / variance**2 - 3.0)


def autocorrelation(series: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at a single lag."""
    series = np.asarray(series, dtype=np.float64)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if series.size <= lag + 1:
        return float("nan")
    centred = series - series.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0.0:
        return float("nan")
    return float((centred[:-lag] * centred[lag:]).sum() / denominator)


def variance_ratio(prices: np.ndarray, q: int) -> float:
    """Lo–MacKinlay variance ratio at horizon ``q``.

    ``VR(q) = Var(r_q) / (q * Var(r_1))``. A random walk gives 1; values below 1 indicate mean
    reversion (a market maker's dream and a red flag for a simulator), above 1 trending.
    """
    if q < 2:
        raise ValueError("q must be at least 2")
    returns = log_returns(prices)
    if returns.size < 2 * q:
        return float("nan")
    single = float(returns.var(ddof=1))
    if single <= 0.0:
        return float("nan")
    usable = (returns.size // q) * q
    aggregated = returns[:usable].reshape(-1, q).sum(axis=1)
    if aggregated.size < 2:
        return float("nan")
    return float(aggregated.var(ddof=1) / (q * single))


def hill_tail_index(sizes: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Hill estimator of the power-law tail index of a positive sample.

    The estimate uses the largest ``tail_fraction`` of observations. For a distribution with
    ``P(X > x) ~ x**-alpha`` the estimator returns ``alpha``.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    sizes = sizes[sizes > 0]
    if sizes.size < 100:
        return float("nan")
    k = max(int(sizes.size * tail_fraction), 10)
    ordered = np.sort(sizes)[::-1]
    top = ordered[:k]
    threshold = ordered[k]
    if threshold <= 0:
        return float("nan")
    logs = np.log(top / threshold)
    mean_log = float(logs.mean())
    if mean_log <= 0.0:
        return float("nan")
    return 1.0 / mean_log


def depth_profile_hump(depths: np.ndarray) -> int:
    """Index of the fullest price level in an average depth profile ordered from the touch.

    Real books are hump-shaped: the most volume rests a few ticks *behind* the touch, not at it.
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.size == 0:
        return -1
    return int(np.argmax(depths))


def signature_plot(prices: np.ndarray, horizons: tuple[int, ...]) -> dict[int, float]:
    """Realised variance per unit time at several sampling horizons.

    A flat signature plot means the price behaves like a random walk at every scale. A downward
    slope from short to long horizons is the classic microstructure-noise signature and is what
    real data shows.
    """
    out: dict[int, float] = {}
    returns = log_returns(prices)
    for h in horizons:
        if returns.size < 2 * h:
            out[h] = float("nan")
            continue
        usable = (returns.size // h) * h
        aggregated = returns[:usable].reshape(-1, h).sum(axis=1)
        out[h] = float(aggregated.var(ddof=1) / h)
    return out
