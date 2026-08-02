"""Turn raw backtest output into the statistics the README reports.

Everything here is paired: policies are compared seed by seed against a common baseline, so the
shared component of episode difficulty cancels. Confidence intervals are bootstrapped from the
paired differences, p-values are corrected across the family of comparisons with Holm-Bonferroni,
and the learned policy's Sharpe is deflated by the number of configurations actually tried.

Usage: python scripts/analyse_results.py
"""

from __future__ import annotations

import numpy as np

from lobsim.experiment import read_json, stage, write_json
from lobsim.stats import (
    bootstrap_ci,
    deflated_sharpe_ratio,
    holm_bonferroni,
    paired_comparison,
    sharpe_ratio,
)

N_BOOT = 10_000


def _metric(block: dict, policy: str, key: str) -> np.ndarray:
    return np.asarray(block[policy]["metrics"][key], dtype=np.float64)


def main() -> None:
    backtests = read_json("backtests.json")
    training = read_json("training.json")
    ablations = read_json("ablations.json")
    baseline = backtests["primary_baseline"]
    rng = np.random.default_rng(0)

    summary: dict[str, object] = {}

    with stage("per-policy summaries"):
        per_condition: dict[str, dict[str, dict[str, float]]] = {}
        for condition, block in backtests["results"].items():
            per_condition[condition] = {}
            for policy in block:
                pnl = _metric(block, policy, "pnl")
                ci = bootstrap_ci(pnl, n_boot=N_BOOT, rng=rng)
                per_condition[condition][policy] = {
                    "pnl_mean": float(np.mean(pnl)),
                    "pnl_ci_low": ci.low,
                    "pnl_ci_high": ci.high,
                    "pnl_median": float(np.median(pnl)),
                    "sharpe_across_episodes": sharpe_ratio(pnl),
                    "spread_capture": float(np.mean(_metric(block, policy, "spread_capture"))),
                    "inventory_pnl": float(np.mean(_metric(block, policy, "inventory_pnl"))),
                    "n_fills": float(np.mean(_metric(block, policy, "n_fills"))),
                    "filled_volume": float(np.mean(_metric(block, policy, "filled_volume"))),
                    "inventory_rms": float(np.mean(_metric(block, policy, "inventory_rms"))),
                    "max_drawdown": float(np.mean(_metric(block, policy, "max_drawdown"))),
                    "markout_1s": _nanmean(_metric(block, policy, "markout_1s")),
                    "markout_5s": _nanmean(_metric(block, policy, "markout_5s")),
                    "markout_30s": _nanmean(_metric(block, policy, "markout_30s")),
                    "edge_per_lot": _nanmean(_metric(block, policy, "edge_per_lot")),
                    "liquidation_cost": float(np.mean(_metric(block, policy, "liquidation_cost"))),
                    "max_abs_pnl_residual": float(
                        np.max(np.abs(_metric(block, policy, "pnl_residual")))
                    ),
                }
        summary["per_condition"] = per_condition

    with stage("paired comparisons versus the primary baseline"):
        comparisons: dict[str, dict[str, object]] = {}
        p_values: dict[str, float] = {}
        block = backtests["results"]["queue_aware"]
        baseline_pnl = _metric(block, baseline, "pnl")
        for policy in block:
            if policy in (baseline, "inactive"):
                continue
            comparison = paired_comparison(
                _metric(block, policy, "pnl"),
                baseline_pnl,
                name=policy,
                baseline=baseline,
                n_boot=N_BOOT,
                rng=rng,
            )
            comparisons[policy] = comparison.to_dict()
            p_values[policy] = comparison.p_value
            print(
                f"      {policy:<20} diff {comparison.mean_difference:+8.2f}  "
                f"95% CI [{comparison.ci.low:+8.2f}, {comparison.ci.high:+8.2f}]  "
                f"p={comparison.p_value:.4f}"
            )
        adjusted = holm_bonferroni(p_values)
        for policy, record in adjusted.items():
            comparisons[policy]["p_adjusted"] = record["p_adjusted"]  # type: ignore[index]
            comparisons[policy]["significant_holm"] = bool(record["significant"])  # type: ignore[index]
        summary["comparisons_vs_baseline"] = comparisons

    with stage("fill-model effect"):
        fill_effects: dict[str, object] = {}
        queue_block = backtests["results"]["queue_aware"]
        optimistic_block = backtests["results"]["optimistic"]
        for policy in queue_block:
            if policy == "inactive":
                continue
            comparison = paired_comparison(
                _metric(optimistic_block, policy, "pnl"),
                _metric(queue_block, policy, "pnl"),
                name=f"{policy}_optimistic",
                baseline=f"{policy}_queue_aware",
                n_boot=N_BOOT,
                rng=rng,
            )
            optimistic_mean = float(np.mean(_metric(optimistic_block, policy, "pnl")))
            queue_mean = float(np.mean(_metric(queue_block, policy, "pnl")))
            fill_effects[policy] = {
                "optimistic_pnl": optimistic_mean,
                "queue_aware_pnl": queue_mean,
                "difference": comparison.mean_difference,
                "ci_low": comparison.ci.low,
                "ci_high": comparison.ci.high,
                "p_value": comparison.p_value,
                "optimistic_fills": float(np.mean(_metric(optimistic_block, policy, "n_fills"))),
                "queue_aware_fills": float(np.mean(_metric(queue_block, policy, "n_fills"))),
                "fill_inflation": _ratio(
                    float(np.mean(_metric(optimistic_block, policy, "filled_volume"))),
                    float(np.mean(_metric(queue_block, policy, "filled_volume"))),
                ),
            }
            print(
                f"      {policy:<20} optimistic {optimistic_mean:+9.2f} vs "
                f"queue-aware {queue_mean:+9.2f}  (delta {comparison.mean_difference:+8.2f})"
            )
        summary["fill_model_effect"] = fill_effects

    with stage("selection-adjusted Sharpe"):
        n_trials = int(training["development_trials"])
        deflation: dict[str, object] = {}
        for policy in ("fqi", baseline, "fixed_spread_2"):
            if policy not in backtests["results"]["queue_aware"]:
                continue
            pnl = _metric(backtests["results"]["queue_aware"], policy, "pnl")
            # Only the learned policy went through a search; the baselines are single fixed rules,
            # so deflating them by the same trial count would be nonsense.
            trials = n_trials if policy == "fqi" else 1
            deflation[policy] = deflated_sharpe_ratio(pnl, n_trials=trials)
            print(
                f"      {policy:<20} Sharpe {deflation[policy]['sharpe']:+.4f}  "  # type: ignore[index]
                f"benchmark {deflation[policy]['benchmark']:+.4f}  "  # type: ignore[index]
                f"DSR {deflation[policy]['deflated_sharpe']:.4f}  (trials={trials})"  # type: ignore[index]
            )
        summary["deflated_sharpe"] = deflation

    with stage("ablation statistics"):
        ablation_stats: dict[str, object] = {}
        cancel = ablations["cancel_policy"]
        ablation_stats["cancel_policy"] = {
            policy: {
                "uniform": cancel[policy]["uniform"]["pnl_mean"],
                "back_loaded": cancel[policy]["back_loaded"]["pnl_mean"],
                "difference": paired_comparison(
                    np.asarray(cancel[policy]["back_loaded"]["pnl"]),
                    np.asarray(cancel[policy]["uniform"]["pnl"]),
                    name="back_loaded",
                    baseline="uniform",
                    n_boot=N_BOOT,
                    rng=rng,
                ).to_dict(),
            }
            for policy in cancel
        }
        features = ablations["feature_groups"]
        reference = np.asarray(features["all_features"]["pnl"], dtype=np.float64)
        ablation_stats["feature_groups"] = {
            name: {
                "pnl_mean": features[name]["pnl_mean"],
                "vs_all_features": paired_comparison(
                    np.asarray(features[name]["pnl"], dtype=np.float64),
                    reference,
                    name=name,
                    baseline="all_features",
                    n_boot=N_BOOT,
                    rng=rng,
                ).to_dict(),
            }
            for name in features
            if name != "all_features"
        }
        estimator = ablations["estimator"]
        ablation_stats["estimator"] = {
            "double_pnl": estimator["double"]["pnl_mean"],
            "single_pnl": estimator["single"]["pnl_mean"],
            "double_inflation": estimator["double"]["target_inflation"],
            "single_inflation": estimator["single"]["target_inflation"],
            "difference": paired_comparison(
                np.asarray(estimator["double"]["pnl"], dtype=np.float64),
                np.asarray(estimator["single"]["pnl"], dtype=np.float64),
                name="double",
                baseline="single",
                n_boot=N_BOOT,
                rng=rng,
            ).to_dict(),
        }
        summary["ablations"] = ablation_stats

    summary["primary_baseline"] = baseline
    summary["n_test_episodes"] = backtests["seeds"]["n"]
    write_json("analysis.json", summary)


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


if __name__ == "__main__":
    main()
