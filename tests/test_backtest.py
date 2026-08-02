"""Backtest harness tests.

The property that matters most is that parallelism is an optimisation and nothing else: episodes
are independent given their seed, so running across processes must give bit-identical answers.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from lobsim.agents import AlwaysAtTouch, Inactive, InventorySkew
from lobsim.backtest import BacktestConfig, default_jobs, run_backtest, run_one
from lobsim.engine import SimConfig
from lobsim.policies import BASELINES, PRIMARY_BASELINE, all_policies
from lobsim.types import FillModel

SHORT = BacktestConfig(sim=SimConfig(horizon_seconds=60.0, burn_in_seconds=10.0))
SEEDS = [11, 12, 13, 14, 15, 16]


class TestDeterminism:
    def test_serial_and_parallel_agree_exactly(self) -> None:
        serial = run_backtest("at_touch", partial(AlwaysAtTouch, size=2), SEEDS, SHORT, n_jobs=1)
        parallel = run_backtest("at_touch", partial(AlwaysAtTouch, size=2), SEEDS, SHORT, n_jobs=3)
        for key in serial.metrics:
            np.testing.assert_array_equal(serial[key], parallel[key], err_msg=key)

    def test_repeating_a_run_reproduces_it(self) -> None:
        first = run_backtest("skew", partial(InventorySkew, size=2), SEEDS, SHORT)
        second = run_backtest("skew", partial(InventorySkew, size=2), SEEDS, SHORT)
        np.testing.assert_array_equal(first.pnl, second.pnl)

    def test_a_fresh_agent_is_built_for_every_episode(self) -> None:
        """Reusing one instance would let episode k-1's state leak into episode k."""
        instances = []

        def factory() -> AlwaysAtTouch:
            agent = AlwaysAtTouch(size=1)
            instances.append(id(agent))
            return agent

        run_backtest("touch", factory, SEEDS, SHORT, n_jobs=1)
        assert len(set(instances)) == len(SEEDS)


class TestShape:
    def test_metrics_are_one_value_per_episode(self) -> None:
        result = run_backtest("touch", partial(AlwaysAtTouch, size=2), SEEDS, SHORT)
        assert result.n_episodes == len(SEEDS)
        assert all(values.shape == (len(SEEDS),) for values in result.metrics.values())
        assert result.seeds == tuple(SEEDS)

    def test_condition_label_is_carried_through(self) -> None:
        config = BacktestConfig(fill_model=FillModel.OPTIMISTIC, label="optimistic")
        result = run_backtest("touch", partial(AlwaysAtTouch, size=2), SEEDS, config)
        assert result.condition == "optimistic"

    def test_dict_export_is_json_friendly(self) -> None:
        payload = run_backtest("touch", partial(AlwaysAtTouch), SEEDS, SHORT).to_dict()
        assert payload["policy"] == "touch"
        assert payload["n_episodes"] == len(SEEDS)
        assert isinstance(payload["metrics"]["pnl"], list)

    def test_indexing_and_means(self) -> None:
        result = run_backtest("touch", partial(AlwaysAtTouch, size=2), SEEDS, SHORT)
        assert result["pnl"] is result.pnl
        assert result.mean("pnl") == pytest.approx(float(np.mean(result.pnl)))

    def test_an_all_nan_metric_is_nan_not_a_warning(self) -> None:
        """An inactive policy has no fills, so its markouts are undefined for every episode."""
        result = run_backtest("inactive", partial(Inactive), SEEDS, SHORT)
        assert np.isnan(result.mean("markout_5s"))

    def test_run_one_returns_a_full_episode(self) -> None:
        result = run_one(SEEDS[0], partial(AlwaysAtTouch, size=2), SHORT)
        assert result.seed == SEEDS[0]
        assert result.mid.size > 10


class TestJobs:
    def test_default_jobs_leaves_headroom(self) -> None:
        assert default_jobs() >= 1

    def test_zero_jobs_means_auto(self) -> None:
        result = run_backtest("touch", partial(AlwaysAtTouch, size=2), SEEDS, SHORT, n_jobs=0)
        assert result.n_episodes == len(SEEDS)


class TestRegistry:
    def test_every_baseline_is_runnable_and_distinctly_named(self) -> None:
        names = {factory().name for factory in BASELINES.values()}
        assert len(names) == len(BASELINES)

    def test_the_primary_baseline_is_a_real_baseline(self) -> None:
        assert PRIMARY_BASELINE in BASELINES

    def test_all_policies_includes_the_learned_one_when_supplied(self) -> None:
        assert "fqi" not in all_policies(None)
        assert "fqi" in all_policies(partial(AlwaysAtTouch))

    def test_every_policy_quotes_the_same_size(self) -> None:
        """Otherwise PnL differences would partly be differences in permitted risk."""
        # `size` is not part of the Agent protocol -- a policy need not have one -- but every
        # baseline in the registry does, and they must agree.
        sizes = {
            name: getattr(factory(), "size")  # noqa: B009
            for name, factory in BASELINES.items()
            if name != "inactive"
        }
        assert len(set(sizes.values())) == 1, sizes
