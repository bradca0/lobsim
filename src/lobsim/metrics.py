"""Performance and risk measures for a market-making episode.

The centrepiece is :func:`decompose_pnl`, which splits total PnL into the two economically distinct
things a market maker earns and loses:

* **Spread capture** -- the instantaneous edge on every fill, measured against the mid at the
  moment of execution. This is what the market maker is *paid* for providing liquidity.
* **Inventory PnL** -- the mark-to-market of carrying a position while the price moves. This is
  what the market maker is *risked on*.

The decomposition is exact, not approximate, and :mod:`tests.test_metrics` asserts the identity
against the engine's own cash accounting. It matters because two policies with identical total PnL
can be completely different businesses: one earning a wide edge and bleeding it back on inventory,
another earning a thin edge it actually keeps.

Markouts answer the complementary question -- how much of the captured spread survives contact
with the next few seconds of price movement. A market maker with a large positive spread capture
and a large negative markout is being adversely selected: it is filled precisely when the price is
about to move against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lobsim.engine import EpisodeResult
from lobsim.types import NS_PER_SECOND, Fill


@dataclass(frozen=True)
class PnLDecomposition:
    """Exact split of realised PnL into liquidity provision and inventory risk, in ticks."""

    total: float
    spread_capture: float
    inventory_pnl: float
    liquidation_cost: float

    @property
    def residual(self) -> float:
        """Should be ~0 up to floating-point error; asserted in tests."""
        return self.total - (self.spread_capture + self.inventory_pnl)


def decompose_pnl(result: EpisodeResult) -> PnLDecomposition:
    """Split episode PnL into spread capture and inventory PnL.

    Both components are accumulated by the engine event by event (see
    :meth:`lobsim.engine.Simulation._mark_inventory`), so this is an exact identity:

        PnL = sum_fills  sign * size * (mid_after_fill - price)   [spread capture]
            + sum_events inventory_before * (mid_after - mid_before)  [inventory PnL]

    :attr:`PnLDecomposition.residual` is the check, and it is asserted to be ~0 in the tests.
    """
    return PnLDecomposition(
        total=result.pnl,
        spread_capture=result.spread_capture,
        inventory_pnl=result.inventory_pnl,
        liquidation_cost=result.liquidation_cost,
    )


def markout(
    fills: list[Fill],
    ts: np.ndarray,
    mid: np.ndarray,
    horizon_seconds: float,
) -> float:
    """Mean per-lot PnL of the agent's fills marked at ``horizon_seconds`` after execution.

    Positive means the fills were, on average, on the right side of the subsequent price move.
    Negative is adverse selection: the market maker was picked off. Fills whose horizon extends
    past the end of the recorded series are dropped rather than clamped, because clamping would
    systematically mark late fills at a shorter -- and therefore less adverse -- horizon.
    """
    if not fills or ts.size == 0:
        return float("nan")
    horizon_ns = int(horizon_seconds * NS_PER_SECOND)
    total = 0.0
    volume = 0
    for fill in fills:
        target = fill.ts + horizon_ns
        if target > ts[-1]:
            continue
        index = int(np.searchsorted(ts, target, side="left"))
        if index >= mid.size:
            continue
        total += fill.side.sign * fill.size * (mid[index] - fill.price)
        volume += fill.size
    if volume == 0:
        return float("nan")
    return total / volume


def max_drawdown(series: np.ndarray) -> float:
    """Largest peak-to-trough decline of a mark-to-market series, as a positive number."""
    series = np.asarray(series, dtype=np.float64)
    if series.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(series)
    return float(np.max(running_peak - series))


def inventory_stats(inventory: np.ndarray) -> dict[str, float]:
    """Risk profile of the position path."""
    inventory = np.asarray(inventory, dtype=np.float64)
    if inventory.size == 0:
        return {"inventory_rms": 0.0, "inventory_max_abs": 0.0, "inventory_mean": 0.0}
    return {
        "inventory_rms": float(np.sqrt(np.mean(inventory**2))),
        "inventory_max_abs": float(np.max(np.abs(inventory))),
        "inventory_mean": float(np.mean(inventory)),
    }


def sharpe(returns: np.ndarray, periods_per_unit: float = 1.0) -> float:
    """Sharpe ratio of a return series, scaled by ``sqrt(periods_per_unit)``.

    Returns NaN for a degenerate (constant or empty) series rather than dividing by zero, so an
    inactive policy is reported as undefined rather than as infinitely good.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return float("nan")
    sd = float(returns.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_unit))


def episode_summary(result: EpisodeResult) -> dict[str, float]:
    """Every per-episode number the analysis stage consumes, in ticks and lots."""
    decomposition = decompose_pnl(result)
    stats = inventory_stats(result.inventory)
    mtm_changes = np.diff(result.mark_to_market) if result.mark_to_market.size > 1 else np.empty(0)
    filled = result.filled_volume

    return {
        "seed": float(result.seed),
        "pnl": result.pnl,
        "spread_capture": decomposition.spread_capture,
        "inventory_pnl": decomposition.inventory_pnl,
        "liquidation_cost": decomposition.liquidation_cost,
        "pnl_residual": decomposition.residual,
        "n_fills": float(result.n_fills),
        "filled_volume": float(filled),
        "pnl_per_lot": result.pnl / filled if filled else float("nan"),
        "edge_per_lot": decomposition.spread_capture / filled if filled else float("nan"),
        "markout_1s": markout(result.fills, result.ts, result.mid, 1.0),
        "markout_5s": markout(result.fills, result.ts, result.mid, 5.0),
        "markout_30s": markout(result.fills, result.ts, result.mid, 30.0),
        "max_drawdown": max_drawdown(result.mark_to_market),
        "intra_episode_sharpe": sharpe(mtm_changes),
        "post_only_rejections": float(result.post_only_rejections),
        "n_quotes": float(result.n_agent_quotes),
        **stats,
    }
