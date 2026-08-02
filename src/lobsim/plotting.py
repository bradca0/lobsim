"""Figures for the README. Excluded from coverage; exercised end-to-end by `make reproduce`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1b1f24"
MUTED = "#8a9199"
ACCENT = "#c8553d"
COOL = "#2e6f95"
GRID = "#e4e7ea"

PALETTE = {
    "inactive": MUTED,
    "at_touch": "#a0522d",
    "fixed_spread_2": "#7a9e7e",
    "inventory_skew": COOL,
    "avellaneda_stoikov": "#9d84b7",
    "fqi": ACCENT,
}

LABELS = {
    "inactive": "Inactive (control)",
    "at_touch": "Always at touch",
    "fixed_spread_2": "Fixed spread (2 ticks)",
    "inventory_skew": "Inventory skew",
    "avellaneda_stoikov": "Avellaneda-Stoikov",
    "fqi": "Learned (FQI)",
}


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def fill_model_figure(analysis: dict[str, Any], path: Path) -> None:
    """The headline: PnL under optimistic versus queue-aware fills, same seeds."""
    effects = analysis["fill_model_effect"]
    policies = [p for p in LABELS if p in effects]
    optimistic = [effects[p]["optimistic_pnl"] for p in policies]
    queue_aware = [effects[p]["queue_aware_pnl"] for p in policies]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    positions = np.arange(len(policies))
    width = 0.38
    ax.bar(positions - width / 2, optimistic, width, label="Optimistic fills", color=MUTED)
    ax.bar(positions + width / 2, queue_aware, width, label="Queue-aware fills", color=ACCENT)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[p] for p in policies], rotation=18, ha="right", fontsize=9)
    _style(ax, "Market-making PnL collapses under queue-aware fills", ylabel="Mean PnL (ticks)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def pnl_decomposition_figure(analysis: dict[str, Any], path: Path) -> None:
    """Spread capture versus inventory PnL: what each policy earns and what it gives back."""
    block = analysis["per_condition"]["queue_aware"]
    policies = [p for p in LABELS if p in block and p != "inactive"]
    spread = [block[p]["spread_capture"] for p in policies]
    inventory = [block[p]["inventory_pnl"] for p in policies]
    total = [block[p]["pnl_mean"] for p in policies]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    positions = np.arange(len(policies))
    width = 0.28
    ax.bar(positions - width, spread, width, label="Spread capture", color=COOL)
    ax.bar(positions, inventory, width, label="Inventory PnL", color=ACCENT)
    ax.bar(positions + width, total, width, label="Total", color=INK)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[p] for p in policies], rotation=18, ha="right", fontsize=9)
    _style(
        ax,
        "Every policy earns the spread and gives it back on inventory",
        ylabel="Mean PnL (ticks)",
    )
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def pnl_distribution_figure(
    backtests: dict[str, Any], analysis: dict[str, Any], path: Path
) -> None:
    """Per-episode PnL distributions with bootstrapped means, queue-aware condition."""
    block = backtests["results"]["queue_aware"]
    stats = analysis["per_condition"]["queue_aware"]
    policies = [p for p in LABELS if p in block and p != "inactive"]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for index, policy in enumerate(policies):
        values = np.asarray(block[policy]["metrics"]["pnl"], dtype=np.float64)
        jitter = np.random.default_rng(index).normal(0, 0.06, values.size)
        ax.scatter(
            np.full(values.size, index) + jitter,
            values,
            s=6,
            alpha=0.25,
            color=PALETTE[policy],
            edgecolors="none",
        )
        record = stats[policy]
        ax.plot([index - 0.28, index + 0.28], [record["pnl_mean"]] * 2, color=INK, linewidth=2)
        ax.plot(
            [index, index],
            [record["pnl_ci_low"], record["pnl_ci_high"]],
            color=INK,
            linewidth=2.4,
            solid_capstyle="round",
        )
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([LABELS[p] for p in policies], rotation=18, ha="right", fontsize=9)
    _style(
        ax,
        "Per-episode PnL with 95% bootstrap CI on the mean",
        ylabel="Episode PnL (ticks)",
    )
    _save(fig, path)


def markout_figure(analysis: dict[str, Any], path: Path) -> None:
    """Adverse selection: how much captured edge survives the next few seconds."""
    block = analysis["per_condition"]["queue_aware"]
    policies = [p for p in LABELS if p in block and p != "inactive"]
    horizons = [0.0, 1.0, 5.0, 30.0]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for policy in policies:
        record = block[policy]
        series = [
            record["edge_per_lot"],
            record["markout_1s"],
            record["markout_5s"],
            record["markout_30s"],
        ]
        ax.plot(
            horizons,
            series,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=PALETTE[policy],
            label=LABELS[policy],
        )
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    _style(
        ax,
        "Markout decay: captured edge versus what survives",
        xlabel="Horizon after fill (seconds)",
        ylabel="PnL per lot (ticks)",
    )
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def validation_figure(validation: dict[str, Any], path: Path) -> None:
    """Simulator fidelity: depth profile and the variance-ratio signature."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    profile = np.asarray(validation["mean_depth_profile"], dtype=np.float64)
    axes[0].bar(np.arange(profile.size), profile, color=COOL, width=0.65)
    _style(
        axes[0],
        "Depth profile (fails: no hump)",
        xlabel="Ticks behind the touch",
        ylabel="Mean resting volume (lots)",
    )

    ratios = validation["variance_ratio_by_horizon"]
    horizons = sorted(int(k) for k in ratios)
    values = [ratios[str(h)] for h in horizons]
    axes[1].axhspan(0.7, 1.3, color=GRID, alpha=0.8)
    axes[1].axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    axes[1].plot(horizons, values, marker="o", color=ACCENT, linewidth=1.8)
    _style(
        axes[1],
        "Variance ratio (passes: mid is near-martingale)",
        xlabel="Horizon (seconds)",
        ylabel="VR(q)",
    )
    axes[1].set_ylim(0.5, 1.5)
    fig.tight_layout()
    _save(fig, path)


def ablation_figure(analysis: dict[str, Any], path: Path) -> None:
    """Feature-group and cancellation-policy ablations side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    features = analysis["ablations"]["feature_groups"]
    names = ["no_queue", "no_flow", "book_only"]
    names = [n for n in names if n in features]
    deltas = [features[n]["vs_all_features"]["mean_difference"] for n in names]
    lows = [features[n]["vs_all_features"]["ci_low"] for n in names]
    highs = [features[n]["vs_all_features"]["ci_high"] for n in names]
    positions = np.arange(len(names))
    axes[0].barh(positions, deltas, color=[ACCENT if d < 0 else COOL for d in deltas], height=0.55)
    for i, (low, high) in enumerate(zip(lows, highs, strict=True)):
        axes[0].plot([low, high], [i, i], color=INK, linewidth=2)
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    _style(axes[0], "Feature ablation vs all features", xlabel="PnL difference (ticks)")

    cancel = analysis["ablations"]["cancel_policy"]
    policies = [p for p in ("at_touch", "inventory_skew", "fqi") if p in cancel]
    uniform = [cancel[p]["uniform"] for p in policies]
    back = [cancel[p]["back_loaded"] for p in policies]
    positions = np.arange(len(policies))
    width = 0.36
    axes[1].bar(positions - width / 2, uniform, width, label="Uniform cancels", color=COOL)
    axes[1].bar(positions + width / 2, back, width, label="Back-loaded cancels", color=ACCENT)
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [LABELS.get(p, p) for p in policies], rotation=12, ha="right", fontsize=8
    )
    _style(axes[1], "Cancellation queue-position assumption", ylabel="Mean PnL (ticks)")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, path)
