"""Transition collection for batch reinforcement learning.

Rewards are computed *here*, outside the policy, from the engine's mark-to-market hooks. The agent
never sees its own PnL when deciding, so there is no path by which a policy could condition on
information a live market maker would not have. The base reward for a decision step is the change
in mark-to-market over the interval that decision was in force, and the final step is credited with
the post-liquidation PnL so the cost of unwinding is attributed to the position that caused it.

**The running inventory penalty is not optional dressing -- it is what makes the problem
well-posed.** Under a martingale mid, holding inventory has zero *expected* PnL: it is pure
variance. A risk-neutral value function therefore sees no reason to control position at all, and
the greedy policy degenerates to quoting both sides unconditionally, which is precisely what was
measured before this term existed (the learned policy collapsed onto the always-at-touch baseline
and inherited its losses). Charging ``phi * q^2`` per step is the discrete-time analogue of the
running penalty in Cartea-Jaimungal, and of the exponential utility that produces the
Avellaneda-Stoikov reservation price. ``phi`` is a risk-aversion parameter selected on validation
seeds; every number reported afterwards is *unpenalised* PnL, so the penalty shapes learning
without flattering the results.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from lobsim.agents.fqi import FQIAgent, QFunction, Transition
from lobsim.backtest import BacktestConfig, default_jobs
from lobsim.engine import Simulation


@dataclass
class _Recorder:
    """Pairs consecutive decisions into transitions as an episode runs."""

    policy: FQIAgent
    episode: int = -1
    inventory_penalty: float = 0.0
    transitions: list[Transition] = field(default_factory=list)
    _pending: tuple[np.ndarray, int, float, int] | None = None

    def _penalty(self, inventory: int) -> float:
        return self.inventory_penalty * float(inventory) ** 2

    def on_decision(self, mark: float, inventory: int) -> None:
        state, action = self.policy.last_state, self.policy.last_action
        if state is None or action is None:  # pragma: no cover - act() always sets both
            return
        if self._pending is not None:
            previous_state, previous_action, previous_mark, previous_inventory = self._pending
            self.transitions.append(
                Transition(
                    state=previous_state,
                    action=previous_action,
                    # The penalty is charged on the position carried *through* the interval.
                    reward=mark - previous_mark - self._penalty(previous_inventory),
                    next_state=state,
                    terminal=False,
                    episode=self.episode,
                )
            )
        self._pending = (state, action, mark, inventory)

    def on_episode_end(self, final_mark: float) -> None:
        """Credit the last decision with everything that happened after it, including the unwind."""
        if self._pending is None:
            return
        state, action, mark, inventory = self._pending
        self.transitions.append(
            Transition(
                state=state,
                action=action,
                reward=final_mark - mark - self._penalty(inventory),
                next_state=state,  # unused: terminal transitions get no continuation value
                terminal=True,
                episode=self.episode,
            )
        )
        self._pending = None


def collect_episode(
    seed: int,
    epsilon: float,
    config: BacktestConfig,
    model: QFunction | None = None,
    feature_groups: tuple[str, ...] = ("book", "flow", "queue", "position"),
    size: int = 2,
    inventory_penalty: float = 0.0,
    action_persistence: float = 10.0,
) -> list[Transition]:
    """Run one episode under an epsilon-greedy behaviour policy and return its transitions."""
    policy = FQIAgent(
        model=model,
        epsilon=epsilon,
        size=size,
        feature_groups=feature_groups,
        action_persistence=action_persistence,
    )
    recorder = _Recorder(policy, episode=seed, inventory_penalty=inventory_penalty)
    simulation = Simulation(
        config=config.sim,
        flow_params=config.flow,
        seed=seed,
        agent=policy,
        fill_model=config.fill_model,
        on_decision=recorder.on_decision,
        on_episode_end=recorder.on_episode_end,
    )
    simulation.run()
    return recorder.transitions


def _collect_worker(
    args: tuple[int, float, BacktestConfig, QFunction | None, tuple[str, ...], int, float, float],
) -> list[Transition]:
    # See the note in lobsim.backtest._worker: single-row inference is dominated by thread-pool
    # setup, so workers run single-threaded and parallelism lives at the episode level.
    with threadpool_limits(limits=1):
        return collect_episode(*args)


def collect_dataset(
    seeds: Sequence[int],
    epsilon: float,
    config: BacktestConfig,
    model: QFunction | None = None,
    feature_groups: tuple[str, ...] = ("book", "flow", "queue", "position"),
    size: int = 2,
    inventory_penalty: float = 0.0,
    action_persistence: float = 10.0,
    n_jobs: int | None = None,
    progress: bool = False,
) -> list[Transition]:
    """Collect transitions across many seeds, in parallel."""
    tasks = [
        (seed, epsilon, config, model, feature_groups, size, inventory_penalty, action_persistence)
        for seed in seeds
    ]
    workers = min(n_jobs or default_jobs(), len(tasks))
    out: list[Transition] = []
    if workers <= 1:
        for task in tasks:
            out.extend(_collect_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(_collect_worker, tasks, chunksize=2):
                out.extend(batch)
    if progress:
        print(f"    collected {len(out):,} transitions from {len(tasks)} episodes")
    return out


def reward_summary(transitions: Sequence[Transition]) -> dict[str, float]:
    """Sanity statistics on the reward signal; a degenerate reward makes FQI meaningless."""
    rewards = np.array([t.reward for t in transitions], dtype=np.float64)
    if rewards.size == 0:
        return {"n": 0.0}
    return {
        "n": float(rewards.size),
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "frac_nonzero": float((rewards != 0).mean()),
    }
