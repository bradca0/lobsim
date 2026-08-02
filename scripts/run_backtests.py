"""Evaluate every policy on the held-out test seeds, under both fill models.

This is where the headline number comes from. Each policy is run twice on the identical seed set:
once with queue-aware fills, once under the optimistic assumption that a trade at your price fills
you. The difference between the two columns is the quantity the repo exists to measure.

Usage: python scripts/run_backtests.py
"""

from __future__ import annotations

import time

from lobsim.backtest import BacktestConfig, run_backtest
from lobsim.experiment import RAW_DIR, TEST_SEEDS, stage, write_json
from lobsim.policies import PRIMARY_BASELINE, all_policies, load_policy
from lobsim.types import FillModel

MODEL_PATH = RAW_DIR.parent / "models" / "fqi_policy.pkl"

CONDITIONS = {
    "queue_aware": BacktestConfig(fill_model=FillModel.QUEUE_AWARE, label="queue_aware"),
    "optimistic": BacktestConfig(fill_model=FillModel.OPTIMISTIC, label="optimistic"),
}


def main() -> None:
    started = time.perf_counter()
    fqi_factory, fqi_info = (None, None)
    if MODEL_PATH.exists():
        fqi_factory, fqi_info = load_policy(MODEL_PATH)
    else:  # pragma: no cover - only when run out of order
        print("  no trained policy found; run scripts/train_policy.py first")

    policies = all_policies(fqi_factory)
    results: dict[str, dict[str, dict[str, object]]] = {}

    for condition_name, condition in CONDITIONS.items():
        with stage(f"backtest [{condition_name}]"):
            results[condition_name] = {}
            for policy_name, factory in policies.items():
                result = run_backtest(
                    policy_name,
                    factory,
                    TEST_SEEDS,
                    condition,
                    n_jobs=0,
                    progress=True,
                )
                results[condition_name][policy_name] = result.to_dict()
                print(
                    f"      {policy_name:<20} PnL {result.mean('pnl'):+9.2f}  "
                    f"fills {result.mean('n_fills'):7.1f}  "
                    f"invRMS {result.mean('inventory_rms'):6.2f}  "
                    f"markout5s {result.mean('markout_5s'):+7.4f}"
                )

    # The control: an inactive policy must report exactly zero PnL in every condition. If it does
    # not, the accounting is broken and no other number in this file means anything.
    for condition_name, block in results.items():
        pnl = block["inactive"]["metrics"]["pnl"]  # type: ignore[index]
        assert all(value == 0.0 for value in pnl), (  # type: ignore[union-attr]
            f"control failed: inactive policy has non-zero PnL under {condition_name}"
        )

    write_json(
        "backtests.json",
        {
            "seeds": {"first": TEST_SEEDS[0], "last": TEST_SEEDS[-1], "n": len(TEST_SEEDS)},
            "primary_baseline": PRIMARY_BASELINE,
            "fqi": fqi_info,
            "conditions": list(CONDITIONS),
            "results": results,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
