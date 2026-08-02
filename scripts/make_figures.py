"""Render every figure the README embeds, from results/raw only.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

from lobsim.experiment import FIGURE_DIR, read_json, stage
from lobsim.plotting import (
    ablation_figure,
    fill_model_figure,
    markout_figure,
    pnl_decomposition_figure,
    pnl_distribution_figure,
    validation_figure,
)


def main() -> None:
    with stage("figures"):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        analysis = read_json("analysis.json")
        backtests = read_json("backtests.json")
        validation = read_json("validation.json")

        fill_model_figure(analysis, FIGURE_DIR / "fill_model.png")
        pnl_decomposition_figure(analysis, FIGURE_DIR / "pnl_decomposition.png")
        pnl_distribution_figure(backtests, analysis, FIGURE_DIR / "pnl_distribution.png")
        markout_figure(analysis, FIGURE_DIR / "markouts.png")
        validation_figure(validation, FIGURE_DIR / "validation.png")
        ablation_figure(analysis, FIGURE_DIR / "ablations.png")


if __name__ == "__main__":
    main()
