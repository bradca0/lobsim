"""Episode runner: evaluates a policy across a seed set and collects per-episode metrics.

Policies are supplied as *factories* rather than instances. A learned policy carries state -- a
fitted model, smoothed features, an inventory view -- and reusing one instance across episodes
risks leaking information from episode *k-1* into episode *k*. Constructing a fresh agent per
episode makes that impossible by construction rather than by remembering to call ``reset``.

Episodes are independent given their seed, so they parallelise exactly: running across processes
changes wall-clock time and nothing else. That property is asserted in the tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from lobsim.engine import Agent, EpisodeResult, SimConfig, Simulation
from lobsim.flow import FlowParams
from lobsim.metrics import episode_summary
from lobsim.types import FillModel

AgentFactory = Callable[[], Agent]


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that defines an experimental condition apart from the policy itself."""

    sim: SimConfig = field(default_factory=SimConfig)
    flow: FlowParams = field(default_factory=FlowParams)
    fill_model: FillModel = FillModel.QUEUE_AWARE
    label: str = "base"


@dataclass
class BacktestResult:
    """Per-episode metrics for one policy under one condition."""

    policy: str
    condition: str
    seeds: tuple[int, ...]
    metrics: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.metrics[key]

    @property
    def pnl(self) -> np.ndarray:
        return self.metrics["pnl"]

    @property
    def n_episodes(self) -> int:
        return len(self.seeds)

    def mean(self, key: str) -> float:
        """Mean across episodes, ignoring NaNs. All-NaN (e.g. markouts of a policy that never
        traded) returns NaN rather than warning."""
        values = self.metrics[key]
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "condition": self.condition,
            "seeds": list(self.seeds),
            "n_episodes": self.n_episodes,
            "metrics": {k: v.tolist() for k, v in self.metrics.items()},
        }


def run_one(
    seed: int,
    factory: AgentFactory,
    config: BacktestConfig,
) -> EpisodeResult:
    """Run a single episode with a freshly constructed agent."""
    simulation = Simulation(
        config=config.sim,
        flow_params=config.flow,
        seed=seed,
        agent=factory(),
        fill_model=config.fill_model,
    )
    return simulation.run()


def _worker(args: tuple[int, AgentFactory, BacktestConfig]) -> dict[str, float]:
    seed, factory, config = args
    # Pin BLAS/OpenMP to a single thread inside each worker. A learned policy scores its actions
    # once per decision on a *single* row, and sklearn's thread-pool setup dominates that call:
    # measured at 12.4 ms per decision multi-threaded versus 0.8 ms pinned, a ~15x difference on
    # pure overhead. Parallelism belongs at the episode level, where there is real work to spread.
    with threadpool_limits(limits=1):
        return episode_summary(run_one(seed, factory, config))


def run_backtest(
    policy: str,
    factory: AgentFactory,
    seeds: Sequence[int],
    config: BacktestConfig | None = None,
    n_jobs: int = 1,
    progress: bool = False,
) -> BacktestResult:
    """Evaluate ``factory`` over ``seeds`` and collect per-episode metrics.

    ``n_jobs > 1`` spreads episodes across processes. Results are bit-identical to the serial path
    because each episode's randomness is fully determined by its seed.
    """
    config = config or BacktestConfig()
    tasks = [(seed, factory, config) for seed in seeds]

    summaries: list[dict[str, float]]
    if n_jobs == 1:
        summaries = []
        for index, task in enumerate(tasks):
            summaries.append(_worker(task))
            if progress:
                print(f"    {policy}: {index + 1}/{len(tasks)}", end="\r", flush=True)
        if progress:
            print()
    else:
        workers = min(n_jobs, len(tasks)) if n_jobs > 0 else default_jobs()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_worker, tasks, chunksize=4))
        if progress:
            print(f"    {policy}: {len(tasks)}/{len(tasks)} (parallel x{workers})")

    keys = summaries[0].keys()
    metrics = {key: np.array([s[key] for s in summaries], dtype=np.float64) for key in keys}
    return BacktestResult(
        policy=policy,
        condition=config.label,
        seeds=tuple(seeds),
        metrics=metrics,
    )


def default_jobs() -> int:
    """Leave two cores for the OS; this runs on a laptop."""
    return max(1, (os.cpu_count() or 2) - 2)
