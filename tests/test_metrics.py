"""Metric tests.

The important one is :class:`TestDecompositionIdentity`. If spread capture plus inventory PnL does
not reproduce realised PnL, then the attribution shown in the README is fiction, so it is checked
as a property over many seeds and every policy rather than on a single hand-picked example.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lobsim.agents import AlwaysAtTouch, AvellanedaStoikov, FixedSpread, Inactive, InventorySkew
from lobsim.backtest import BacktestConfig, run_one
from lobsim.engine import SimConfig, run_episode
from lobsim.metrics import (
    decompose_pnl,
    episode_summary,
    inventory_stats,
    markout,
    max_drawdown,
    sharpe,
)
from lobsim.types import Fill, FillModel, Side

SHORT = SimConfig(horizon_seconds=90.0, burn_in_seconds=10.0)

POLICIES = [
    partial(AlwaysAtTouch, size=2),
    partial(InventorySkew, size=2),
    partial(FixedSpread, size=2, half_spread=2),
    partial(AvellanedaStoikov, size=2),
    partial(Inactive, size=1),
]


class TestDecompositionIdentity:
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        seed=st.integers(min_value=0, max_value=5_000),
        policy_index=st.integers(min_value=0, max_value=len(POLICIES) - 1),
    )
    def test_spread_capture_plus_inventory_pnl_equals_realised_pnl(
        self, seed: int, policy_index: int
    ) -> None:
        result = run_one(seed, POLICIES[policy_index], BacktestConfig(sim=SHORT))
        decomposition = decompose_pnl(result)
        assert decomposition.residual == pytest.approx(0.0, abs=1e-6)

    def test_identity_holds_under_the_optimistic_fill_model_too(self) -> None:
        config = BacktestConfig(sim=SHORT, fill_model=FillModel.OPTIMISTIC)
        for seed in range(6):
            result = run_one(seed, partial(AlwaysAtTouch, size=2), config)
            assert decompose_pnl(result).residual == pytest.approx(0.0, abs=1e-6)

    def test_a_policy_that_never_trades_decomposes_to_zero(self) -> None:
        result = run_episode(3, Inactive(), SHORT)
        decomposition = decompose_pnl(result)
        assert decomposition.total == 0.0
        assert decomposition.spread_capture == 0.0
        assert decomposition.inventory_pnl == 0.0

    def test_the_components_are_not_trivially_zero(self) -> None:
        """Guards against the identity holding because both sides are always 0."""
        result = run_episode(1, AlwaysAtTouch(size=2), SimConfig(horizon_seconds=300.0))
        decomposition = decompose_pnl(result)
        assert abs(decomposition.spread_capture) > 1.0
        assert abs(decomposition.inventory_pnl) > 1.0


class TestMarkout:
    def _series(self) -> tuple[np.ndarray, np.ndarray]:
        ts = np.arange(0, 60, dtype=np.int64) * 1_000_000_000
        mid = np.full(60, 100.0)
        return ts, mid

    def test_a_buy_filled_below_a_flat_mid_marks_out_positive(self) -> None:
        ts, mid = self._series()
        fills = [Fill(ts=0, side=Side.BUY, price=99, size=5, is_maker=True, mid_at_fill=100.0)]
        assert markout(fills, ts, mid, 5.0) == pytest.approx(1.0)

    def test_a_sell_filled_above_a_flat_mid_marks_out_positive(self) -> None:
        ts, mid = self._series()
        fills = [Fill(ts=0, side=Side.SELL, price=101, size=5, is_maker=True, mid_at_fill=100.0)]
        assert markout(fills, ts, mid, 5.0) == pytest.approx(1.0)

    def test_a_buy_ahead_of_a_falling_price_marks_out_negative(self) -> None:
        """Adverse selection: filled on the bid just before the market drops."""
        ts, _ = self._series()
        mid = np.linspace(100.0, 90.0, 60)
        fills = [Fill(ts=0, side=Side.BUY, price=100, size=1, is_maker=True, mid_at_fill=100.0)]
        assert markout(fills, ts, mid, 30.0) < 0.0

    def test_markout_is_volume_weighted(self) -> None:
        ts, mid = self._series()
        fills = [
            Fill(ts=0, side=Side.BUY, price=99, size=1, is_maker=True, mid_at_fill=100.0),
            Fill(ts=0, side=Side.BUY, price=90, size=9, is_maker=True, mid_at_fill=100.0),
        ]
        # (1*1 + 9*10) / 10 = 9.1, not the unweighted mean of 1 and 10.
        assert markout(fills, ts, mid, 5.0) == pytest.approx(9.1)

    def test_fills_without_a_full_horizon_are_dropped_not_clamped(self) -> None:
        """Clamping would mark late fills at a shorter, less adverse horizon."""
        ts, mid = self._series()
        late = [Fill(ts=ts[-1], side=Side.BUY, price=99, size=1, is_maker=True, mid_at_fill=100.0)]
        assert np.isnan(markout(late, ts, mid, 30.0))

    def test_no_fills_is_undefined_rather_than_zero(self) -> None:
        ts, mid = self._series()
        assert np.isnan(markout([], ts, mid, 5.0))


class TestDrawdown:
    def test_monotone_growth_has_no_drawdown(self) -> None:
        assert max_drawdown(np.array([0.0, 1.0, 2.0, 3.0])) == 0.0

    def test_peak_to_trough_is_measured_from_the_running_maximum(self) -> None:
        assert max_drawdown(np.array([0.0, 10.0, 4.0, 8.0, 1.0])) == 9.0

    def test_empty_series_is_zero(self) -> None:
        assert max_drawdown(np.array([])) == 0.0


class TestInventoryStats:
    def test_rms_and_extremes(self) -> None:
        stats = inventory_stats(np.array([3.0, -4.0, 0.0]))
        assert stats["inventory_rms"] == pytest.approx(np.sqrt(25.0 / 3.0))
        assert stats["inventory_max_abs"] == 4.0
        assert stats["inventory_mean"] == pytest.approx(-1.0 / 3.0)

    def test_empty_path_is_all_zero(self) -> None:
        assert inventory_stats(np.array([])) == {
            "inventory_rms": 0.0,
            "inventory_max_abs": 0.0,
            "inventory_mean": 0.0,
        }


class TestSharpe:
    def test_matches_the_definition(self) -> None:
        returns = np.array([1.0, 2.0, 3.0, 4.0])
        expected = returns.mean() / returns.std(ddof=1)
        assert sharpe(returns) == pytest.approx(expected)

    def test_scaling_multiplies_by_the_square_root_of_the_period_count(self) -> None:
        returns = np.array([1.0, -1.0, 2.0, 0.5])
        assert sharpe(returns, 4.0) == pytest.approx(sharpe(returns) * 2.0)

    def test_a_constant_series_is_undefined_not_infinite(self) -> None:
        """An inactive policy has zero variance; reporting infinite Sharpe would be absurd."""
        assert np.isnan(sharpe(np.zeros(50)))

    def test_too_short_a_series_is_undefined(self) -> None:
        assert np.isnan(sharpe(np.array([1.0])))


class TestEpisodeSummary:
    def test_summary_is_json_friendly_and_complete(self) -> None:
        result = run_episode(2, AlwaysAtTouch(size=2), SHORT)
        summary = episode_summary(result)
        for key in ("pnl", "spread_capture", "inventory_pnl", "markout_5s", "inventory_rms"):
            assert key in summary
        assert all(isinstance(v, float) for v in summary.values())

    def test_per_lot_metrics_are_undefined_when_nothing_traded(self) -> None:
        summary = episode_summary(run_episode(2, Inactive(), SHORT))
        assert np.isnan(summary["pnl_per_lot"])
        assert np.isnan(summary["edge_per_lot"])
        assert summary["filled_volume"] == 0.0


class TestOneSidedBook:
    """The identity must survive a book that goes one-sided, which is where it first broke."""

    def test_identity_holds_when_the_mid_becomes_undefined(self) -> None:
        result = run_one(0, partial(AlwaysAtTouch, size=2), BacktestConfig(sim=SHORT))
        assert result.one_sided_events > 0, "this seed no longer exercises the one-sided path"
        assert decompose_pnl(result).residual == pytest.approx(0.0, abs=1e-6)

    def test_a_swept_side_carries_the_last_reference_forward(self) -> None:
        from lobsim.engine import Simulation
        from lobsim.flow import FlowParams

        sim = Simulation(SHORT, FlowParams(), seed=0)
        sim.flow.seed_book(sim.book)
        reference = sim._reference_price()
        for order_id in list(sim.book.resting_order_ids):
            order = sim.book.get_order(order_id)
            if order is not None and order.side is Side.SELL:
                sim.book.cancel(order_id)
        assert sim.book.best_ask is None
        assert sim._reference_price() == reference  # frozen, not undefined
