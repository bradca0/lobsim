"""Inject every number in README.md from results/raw.

The README contains marker pairs like::

    <!-- BEGIN:headline -->
    ... generated ...
    <!-- END:headline -->

and this script rewrites what is between them. Nothing between a marker pair is ever hand-edited,
so a number in the README cannot drift away from the run that produced it. If a marker is missing,
that is an error rather than a silent no-op.

Usage: python scripts/render_readme_tables.py
"""

from __future__ import annotations

import re
from typing import Any

from lobsim.experiment import REPO_ROOT, read_json, stage

README = REPO_ROOT / "README.md"

LABELS = {
    "inactive": "Inactive (control)",
    "at_touch": "Always at touch",
    "fixed_spread_2": "Fixed spread, 2 ticks",
    "inventory_skew": "Inventory skew",
    "avellaneda_stoikov": "Avellaneda–Stoikov",
    "fqi": "**Learned (FQI)**",
}
ORDER = list(LABELS)


def _fmt(value: float, places: int = 2, sign: bool = True) -> str:
    if value != value:  # NaN
        return "—"
    spec = f"{'+' if sign else ''}.{places}f"
    return f"{value:{spec}}"


def claim_block(analysis: dict[str, Any]) -> str:
    """The one-paragraph headline, computed rather than asserted."""
    effects = analysis["fill_model_effect"]
    baseline = analysis["primary_baseline"]
    lines = []
    for policy in ("at_touch", baseline, "fqi"):
        if policy not in effects:
            continue
        record = effects[policy]
        lines.append(
            f"- **{LABELS[policy].strip('*')}**: {_fmt(record['optimistic_pnl'])} ticks under "
            f"optimistic fills, {_fmt(record['queue_aware_pnl'])} under queue-aware fills "
            f"(difference {_fmt(record['difference'])}, 95% CI "
            f"[{_fmt(record['ci_low'])}, {_fmt(record['ci_high'])}]). It executes "
            f"{record['fill_inflation']:.1f}x more volume when the queue is ignored."
        )
    flipped = [
        LABELS[p].strip("*")
        for p, r in effects.items()
        if r["optimistic_pnl"] > 0 >= r["queue_aware_pnl"]
    ]
    if flipped:
        lines.append("")
        lines.append(
            "Policies that look **profitable under optimistic fills and are not** once the queue is "
            "modelled: " + ", ".join(flipped) + "."
        )
    return "\n".join(lines)


def headline_table(analysis: dict[str, Any]) -> str:
    block = analysis["per_condition"]["queue_aware"]
    comparisons = analysis.get("comparisons_vs_baseline", {})
    baseline = analysis["primary_baseline"]
    rows = [
        "| Policy | PnL (ticks) | 95% CI | Fills | Inv. RMS | Edge/lot | 5s markout | vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ORDER:
        if policy not in block:
            continue
        record = block[policy]
        if policy == baseline:
            verdict = "_baseline_"
        elif policy in comparisons:
            comparison = comparisons[policy]
            marker = "✓" if comparison.get("significant_holm") else "n.s."
            verdict = f"{_fmt(comparison['mean_difference'])} ({marker})"
        else:
            verdict = "—"
        rows.append(
            f"| {LABELS[policy]} "
            f"| {_fmt(record['pnl_mean'])} "
            f"| [{_fmt(record['pnl_ci_low'])}, {_fmt(record['pnl_ci_high'])}] "
            f"| {record['n_fills']:.0f} "
            f"| {record['inventory_rms']:.1f} "
            f"| {_fmt(record['edge_per_lot'], 3)} "
            f"| {_fmt(record['markout_5s'], 3)} "
            f"| {verdict} |"
        )
    return "\n".join(rows)


def fill_model_table(analysis: dict[str, Any]) -> str:
    effects = analysis["fill_model_effect"]
    rows = [
        "| Policy | Optimistic fills | Queue-aware fills | Difference | 95% CI | Fill-volume inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ORDER:
        if policy not in effects:
            continue
        record = effects[policy]
        rows.append(
            f"| {LABELS[policy]} "
            f"| {_fmt(record['optimistic_pnl'])} "
            f"| {_fmt(record['queue_aware_pnl'])} "
            f"| {_fmt(record['difference'])} "
            f"| [{_fmt(record['ci_low'])}, {_fmt(record['ci_high'])}] "
            f"| {record['fill_inflation']:.2f}× |"
        )
    return "\n".join(rows)


def validation_table(validation: dict[str, Any]) -> str:
    rows = [
        "| Stylized fact | Measured | Target band | |",
        "|---|---:|:---:|:--:|",
    ]
    for fact in validation["facts"]:
        mark = "pass" if fact["passes"] else "**FAIL**"
        band = f"[{fact['target_low']:g}, {fact['target_high']:g}]"
        rows.append(f"| {fact['name'].replace('_', ' ')} | {fact['value']:.4f} | {band} | {mark} |")
    passing = validation["n_passing"]
    total = validation["n_facts"]
    rows.append("")
    rows.append(f"**{passing} of {total} pass.** Failures are analysed in Limitations.")
    return "\n".join(rows)


def ablation_table(analysis: dict[str, Any]) -> str:
    ablations = analysis["ablations"]
    rows = [
        "| Ablation | Variant | PnL (ticks) | Δ vs reference | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    features = ablations["feature_groups"]
    for name in ("no_queue", "no_flow", "book_only"):
        if name not in features:
            continue
        record = features[name]
        comparison = record["vs_all_features"]
        rows.append(
            f"| Feature groups | {name.replace('_', ' ')} "
            f"| {_fmt(record['pnl_mean'])} "
            f"| {_fmt(comparison['mean_difference'])} "
            f"| [{_fmt(comparison['ci_low'])}, {_fmt(comparison['ci_high'])}] |"
        )
    cancel = ablations["cancel_policy"]
    for policy in ("at_touch", "inventory_skew", "fqi"):
        if policy not in cancel:
            continue
        record = cancel[policy]
        comparison = record["difference"]
        rows.append(
            f"| Cancel position ({policy}) | back-loaded "
            f"| {_fmt(record['back_loaded'])} "
            f"| {_fmt(comparison['mean_difference'])} "
            f"| [{_fmt(comparison['ci_low'])}, {_fmt(comparison['ci_high'])}] |"
        )
    estimator = ablations["estimator"]
    rows.append(
        f"| Q estimator | single (vs double) "
        f"| {_fmt(estimator['single_pnl'])} "
        f"| {_fmt(-estimator['difference']['mean_difference'])} "
        f"| [{_fmt(-estimator['difference']['ci_high'])}, "
        f"{_fmt(-estimator['difference']['ci_low'])}] |"
    )
    rows.append("")
    rows.append(
        f"Q-value target inflation: {estimator['double_inflation']:.2f}× with the double "
        f"estimator versus {estimator['single_inflation']:.2f}× with a single one."
    )
    return "\n".join(rows)


def deflation_table(analysis: dict[str, Any], training: dict[str, Any]) -> str:
    deflation = analysis["deflated_sharpe"]
    rows = [
        "| Policy | Sharpe (per episode) | Selection benchmark | Deflated Sharpe | Trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, record in deflation.items():
        rows.append(
            f"| {LABELS.get(policy, policy)} "
            f"| {_fmt(record['sharpe'], 3)} "
            f"| {_fmt(record['benchmark'], 3)} "
            f"| {record['deflated_sharpe']:.3f} "
            f"| {int(record['n_trials'])} |"
        )
    rows.append("")
    rows.append(
        f"The learned policy is deflated by {training['development_trials']} configurations — "
        f"every FQI variant evaluated on validation seeds across development, not the "
        f"{training['grid_size']} in the final grid."
    )
    return "\n".join(rows)


def provenance_line(backtests: dict[str, Any], validation: dict[str, Any]) -> str:
    revision = backtests["_provenance"]["git_revision"]
    seeds = backtests["seeds"]
    return (
        f"Generated from commit `{revision}` on {backtests['_provenance']['platform']}. "
        f"{seeds['n']} held-out test episodes (seeds {seeds['first']}–{seeds['last']}), "
        f"{validation['n_episodes']} agentless episodes for validation. "
        f"Backtests took {backtests['elapsed_seconds'] / 60:.1f} minutes."
    )


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing the marker pair for '{name}'")
    return pattern.sub(lambda m: f"{m.group(1)}{body}{m.group(2)}", text)


def main() -> None:
    with stage("render README"):
        analysis = read_json("analysis.json")
        validation = read_json("validation.json")
        backtests = read_json("backtests.json")
        training = read_json("training.json")

        text = README.read_text()
        text = replace_block(text, "claim", claim_block(analysis))
        text = replace_block(text, "headline", headline_table(analysis))
        text = replace_block(text, "fillmodel", fill_model_table(analysis))
        text = replace_block(text, "validation", validation_table(validation))
        text = replace_block(text, "ablations", ablation_table(analysis))
        text = replace_block(text, "deflation", deflation_table(analysis, training))
        text = replace_block(text, "provenance", provenance_line(backtests, validation))
        README.write_text(text)
        print(f"  wrote {README.name}")


if __name__ == "__main__":
    main()
