"""Ablations: which modelling choices actually drive the results.

Three, each targeting a different way the headline could be an artefact:

1. **Cancellation queue position.** ``UNIFORM`` versus ``BACK_LOADED``. This is the least
   verifiable assumption in the simulator and it directly controls how fast the queue in front of a
   resting order evaporates.
2. **Feature groups.** Retrain the learned policy without the queue features, and without the flow
   features, and re-measure out of sample. Permutation importance says what the *model* leans on;
   this says what actually matters for PnL.
3. **Double versus single Q estimator.** How much the maximisation-bias correction is worth.

Each ablation retrains where retraining is required, on training seeds only, and evaluates on the
same held-out test seeds as the main result.

Usage: python scripts/run_ablations.py
"""

from __future__ import annotations

import time
from functools import partial

from lobsim.agents.fqi import FQIAgent, FQIConfig, fit_fqi
from lobsim.backtest import BacktestConfig, run_backtest
from lobsim.experiment import RAW_DIR, TEST_SEEDS, TRAIN_SEEDS, read_json, stage, write_json
from lobsim.flow import CancelPolicy, FlowParams
from lobsim.policies import AGENT_SIZE, BASELINES, load_policy
from lobsim.training import collect_dataset

MODEL_PATH = RAW_DIR.parent / "models" / "fqi_policy.pkl"

# Ablations run on the first half of the held-out test seeds rather than all of them. Each
# feature-group variant needs a fresh dataset, a fresh fit and a fresh evaluation, and the full
# grid on 200 seeds does not fit in a reasonable wall clock on an 8 GB laptop. The subset is a
# prefix of the same held-out seeds -- never a re-draw, and never overlapping training or
# validation -- so ablation confidence intervals are simply wider than the headline's, which is
# stated rather than hidden. The headline table itself uses all 200.
ABLATION_SEEDS = TEST_SEEDS[: len(TEST_SEEDS) // 2]

FEATURE_ABLATIONS: dict[str, tuple[str, ...]] = {
    "all_features": ("book", "flow", "queue", "position"),
    "no_queue": ("book", "flow", "position"),
    "no_flow": ("book", "queue", "position"),
    "book_only": ("book", "position"),
}


def selected_config() -> FQIConfig:
    """Rebuild the configuration chosen by train_policy.py, so ablations differ in one thing only."""
    selected = read_json("training.json")["selected"]
    return FQIConfig(
        n_iterations=3,
        discount=float(selected["discount"]),
        max_iter=80,
        max_depth=4,
        min_samples_leaf=int(selected["min_samples_leaf"]),
        inventory_penalty=float(selected["inventory_penalty"]),
        double=True,
    )


def main() -> None:
    started = time.perf_counter()
    base_config = selected_config()
    payload: dict[str, object] = {}

    # ------------------------------------------------------------ cancellation queue position
    with stage("ablation: cancellation queue position"):
        cancel_results: dict[str, dict[str, object]] = {}
        fqi_factory, _ = load_policy(MODEL_PATH)
        policies = {**BASELINES, "fqi": fqi_factory}
        for policy_name in ("at_touch", "inventory_skew", "fqi"):
            cancel_results[policy_name] = {}
            for cancel_policy in (CancelPolicy.UNIFORM, CancelPolicy.BACK_LOADED):
                condition = BacktestConfig(
                    flow=FlowParams(cancel_policy=cancel_policy),
                    label=cancel_policy.value,
                )
                result = run_backtest(
                    policy_name, policies[policy_name], ABLATION_SEEDS, condition, n_jobs=0
                )
                cancel_results[policy_name][cancel_policy.value] = {
                    "pnl_mean": result.mean("pnl"),
                    "pnl": result["pnl"].tolist(),
                    "n_fills": result.mean("n_fills"),
                    "filled_volume": result.mean("filled_volume"),
                    "markout_5s": result.mean("markout_5s"),
                }
                print(
                    f"      {policy_name:<18} {cancel_policy.value:<12} "
                    f"PnL {result.mean('pnl'):+9.2f}  fills {result.mean('n_fills'):7.1f}"
                )
        payload["cancel_policy"] = cancel_results

    # ------------------------------------------------------------ feature groups
    with stage("ablation: feature groups"):
        feature_results: dict[str, dict[str, object]] = {}
        config = BacktestConfig()
        for name, groups in FEATURE_ABLATIONS.items():
            data = collect_dataset(
                TRAIN_SEEDS,
                epsilon=1.0,
                config=config,
                feature_groups=groups,
                inventory_penalty=base_config.inventory_penalty,
                action_persistence=10.0,
            )
            fitted = fit_fqi(data, FQIConfig(**{**vars(base_config), "feature_groups": groups}))
            factory = partial(
                FQIAgent,
                model=fitted.model,
                epsilon=0.0,
                size=AGENT_SIZE,
                feature_groups=groups,
            )
            result = run_backtest(f"fqi_{name}", factory, ABLATION_SEEDS, config, n_jobs=0)
            feature_results[name] = {
                "groups": list(groups),
                "pnl_mean": result.mean("pnl"),
                "pnl": result["pnl"].tolist(),
                "n_fills": result.mean("n_fills"),
                "inventory_rms": result.mean("inventory_rms"),
                "target_inflation": fitted.target_inflation,
            }
            print(
                f"      {name:<16} PnL {result.mean('pnl'):+9.2f}  fills {result.mean('n_fills'):7.1f}"
            )
        payload["feature_groups"] = feature_results

    # ------------------------------------------------------------ double vs single estimator
    with stage("ablation: double vs single Q estimator"):
        estimator_results: dict[str, dict[str, object]] = {}
        config = BacktestConfig()
        data = collect_dataset(
            TRAIN_SEEDS,
            epsilon=1.0,
            config=config,
            inventory_penalty=base_config.inventory_penalty,
            action_persistence=10.0,
        )
        for double in (True, False):
            fitted = fit_fqi(data, FQIConfig(**{**vars(base_config), "double": double}))
            factory = partial(FQIAgent, model=fitted.model, epsilon=0.0, size=AGENT_SIZE)
            result = run_backtest("fqi", factory, ABLATION_SEEDS, config, n_jobs=0)
            key = "double" if double else "single"
            estimator_results[key] = {
                "pnl_mean": result.mean("pnl"),
                "pnl": result["pnl"].tolist(),
                "target_inflation": fitted.target_inflation,
                "target_scale": fitted.target_scale,
            }
            print(
                f"      {key:<8} PnL {result.mean('pnl'):+9.2f}  "
                f"target inflation {fitted.target_inflation:.2f}"
            )
        payload["estimator"] = estimator_results

    payload["n_ablation_episodes"] = len(ABLATION_SEEDS)
    payload["elapsed_seconds"] = time.perf_counter() - started
    write_json("ablations.json", payload)


if __name__ == "__main__":
    main()
