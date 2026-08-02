"""Train the Fitted Q-Iteration market maker.

Protocol, fixed before any test seed was touched:

* Transitions are collected on TRAIN_SEEDS only, under a persistent epsilon-greedy behaviour policy.
* Hyperparameters are selected on VALIDATION_SEEDS by mean PnL.
* TEST_SEEDS are never loaded by this script. The selected model is written to disk and evaluated
  by scripts/run_backtests.py.

Selection is by *unpenalised* PnL even though training uses an inventory penalty: the penalty is a
device for making the control problem well-posed, not the objective anyone cares about.

Usage: python scripts/train_policy.py
"""

from __future__ import annotations

import itertools
import pickle
import time
from functools import partial

import numpy as np

from lobsim.agents.fqi import (
    FQIAgent,
    FQIConfig,
    QFunction,
    fit_fqi,
    model_payload,
    permutation_importance,
)
from lobsim.backtest import AgentFactory, BacktestConfig, run_backtest
from lobsim.experiment import (
    RAW_DIR,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    stage,
    write_json,
)
from lobsim.training import collect_dataset, reward_summary

MODEL_PATH = RAW_DIR.parent / "models" / "fqi_policy.pkl"

# The search grid actually executed by `make reproduce`. The count of configurations that a
# deflated Sharpe ratio must be corrected for is *larger* than this -- see DEVELOPMENT_TRIALS.
INVENTORY_PENALTIES = (0.01, 0.03)
DISCOUNTS = (0.97,)
MIN_SAMPLES_LEAF = (100, 300)

# Configurations evaluated on validation seeds across the whole of development, not just those in
# the grid above. Reporting only the final grid would understate the multiple-testing burden and
# flatter the deflated Sharpe ratio; this number is maintained by hand and documented in
# docs/DECISIONS.md D11.
DEVELOPMENT_TRIALS = 24

BEHAVIOUR_EPSILON = 1.0
ACTION_PERSISTENCE = 10.0
AGENT_SIZE = 2


def main() -> None:
    config = BacktestConfig()
    started = time.perf_counter()
    datasets = {}

    with stage("collect transitions"):
        for penalty in INVENTORY_PENALTIES:
            datasets[penalty] = collect_dataset(
                TRAIN_SEEDS,
                epsilon=BEHAVIOUR_EPSILON,
                config=config,
                inventory_penalty=penalty,
                action_persistence=ACTION_PERSISTENCE,
                progress=True,
            )
            print(f"    phi={penalty}: {reward_summary(datasets[penalty])}")

    trials = []
    best: tuple[float, FQIConfig, QFunction] | None = None

    with stage("fit and select on validation seeds"):
        grid = list(itertools.product(INVENTORY_PENALTIES, DISCOUNTS, MIN_SAMPLES_LEAF))
        for index, (penalty, discount, leaf) in enumerate(grid):
            fqi_config = FQIConfig(
                n_iterations=3,
                discount=discount,
                max_iter=80,
                max_depth=4,
                min_samples_leaf=leaf,
                inventory_penalty=penalty,
                double=True,
            )
            fitted = fit_fqi(datasets[penalty], fqi_config)
            evaluation = run_backtest(
                "fqi",
                _factory(fitted.model),
                VALIDATION_SEEDS,
                config,
            )
            pnl = evaluation.mean("pnl")
            trials.append(
                {
                    "label": fqi_config.label(),
                    "inventory_penalty": penalty,
                    "discount": discount,
                    "min_samples_leaf": leaf,
                    "validation_pnl": pnl,
                    "validation_inventory_rms": evaluation.mean("inventory_rms"),
                    "validation_n_fills": evaluation.mean("n_fills"),
                    "target_inflation": fitted.target_inflation,
                    "target_scale": fitted.target_scale,
                }
            )
            print(
                f"    [{index + 1}/{len(grid)}] phi={penalty} gamma={discount} leaf={leaf}: "
                f"validation PnL {pnl:+.2f}, inflation {fitted.target_inflation:.2f}"
            )
            if best is None or pnl > best[0]:
                best = (pnl, fqi_config, fitted.model)
                best_scale = fitted.target_scale
                best_inflation = fitted.target_inflation

    assert best is not None
    best_pnl, best_config, best_model = best
    print(f"\n  selected: {best_config.label()} with validation PnL {best_pnl:+.2f}")

    with stage("diagnostics"):
        states = np.stack([t.state for t in datasets[best_config.inventory_penalty][:6000]])
        reference = FQIAgent(feature_groups=best_config.feature_groups)
        importances = permutation_importance(
            best_model,
            states,
            reference.feature_names(),
            np.random.default_rng(0),
            n_repeats=4,
        )
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        for name, value in ranked[:8]:
            print(f"    {name:<26} {value:.5f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {"model": best_model, "config": best_config},
            handle,
        )
    print(f"  wrote {MODEL_PATH.relative_to(MODEL_PATH.parents[2])}")

    write_json(
        "training.json",
        {
            "protocol": {
                "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
                "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
                "n_train_episodes": len(TRAIN_SEEDS),
                "behaviour_epsilon": BEHAVIOUR_EPSILON,
                "action_persistence": ACTION_PERSISTENCE,
                "agent_size": AGENT_SIZE,
            },
            "grid_size": len(trials),
            "development_trials": DEVELOPMENT_TRIALS,
            "trials": trials,
            "selected": {
                "label": best_config.label(),
                "validation_pnl": best_pnl,
                "inventory_penalty": best_config.inventory_penalty,
                "discount": best_config.discount,
                "min_samples_leaf": best_config.min_samples_leaf,
                "target_inflation": best_inflation,
                "target_scale": best_scale,
            },
            "model": model_payload(best_model),
            "feature_importance": dict(ranked),
            "reward_summary": reward_summary(datasets[best_config.inventory_penalty]),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _factory(model: QFunction) -> AgentFactory:
    return partial(FQIAgent, model=model, epsilon=0.0, size=AGENT_SIZE)


if __name__ == "__main__":
    main()
