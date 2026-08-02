"""Measure the simulator against published stylized facts of real limit order books.

Run on an *agentless* market so the market maker cannot flatter the numbers. Every fact is
reported with the empirical target band it is judged against, and failures are kept in the output
rather than dropped -- the README shows them.

Usage: python scripts/run_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import TRAIN_SEEDS, stage, write_json  # noqa: E402

from lobsim.book import LimitOrderBook  # noqa: E402
from lobsim.engine import SimConfig, run_episode  # noqa: E402
from lobsim.flow import EventKind, FlowParams, OrderFlowGenerator  # noqa: E402
from lobsim.types import Side  # noqa: E402
from lobsim.validation import (  # noqa: E402
    StylizedFact,
    autocorrelation,
    depth_profile_hump,
    excess_kurtosis,
    hill_tail_index,
    log_returns,
    signature_plot,
    variance_ratio,
)

N_EPISODES = 60
CONFIG = SimConfig(horizon_seconds=300.0, burn_in_seconds=30.0)
DEPTH_LEVELS = 6


def collect_market_microstructure(
    n_episodes: int, params: FlowParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run agentless episodes, returning depth profiles, spreads, order sizes and MO signs."""
    profiles: list[np.ndarray] = []
    spreads: list[int] = []
    sizes: list[int] = []
    mo_signs: list[int] = []

    for episode, seed in enumerate(TRAIN_SEEDS[:n_episodes]):
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
        book = LimitOrderBook()
        generator = OrderFlowGenerator(params, rng)
        generator.seed_book(book)
        end = CONFIG.burn_in_seconds + CONFIG.horizon_seconds
        t = 0.0
        next_sample = CONFIG.burn_in_seconds + 1.0
        while t < end:
            t, event = generator.next_event(book, t)
            if t >= end:
                break
            while next_sample <= t:
                spread = book.spread
                if spread is not None:
                    spreads.append(spread)
                    # Indexed by *tick offset* from the touch, not by occupied level: an empty
                    # price inside the book is a zero, not a shift of everything behind it.
                    # Indexing by occupied level silently compresses gaps and can manufacture or
                    # destroy the hump this profile exists to detect.
                    profile = np.zeros(DEPTH_LEVELS)
                    for side in (Side.BUY, Side.SELL):
                        touch = book.best(side)
                        if touch is None:
                            continue
                        for i in range(DEPTH_LEVELS):
                            profile[i] += book.volume_at(side, touch - side.sign * i)
                    profiles.append(profile / 2.0)
                next_sample += 1.0
            if event is None:
                continue
            if event.kind is EventKind.MARKET:
                sizes.append(event.size)
                mo_signs.append(event.side.sign)
                book.add_market(event.ts, event.side, event.size)
            elif event.kind is EventKind.LIMIT:
                sizes.append(event.size)
                book.add_limit(event.ts, event.side, event.price, event.size)
            else:
                book.cancel(event.order_id)
        print(f"    microstructure episode {episode + 1}/{n_episodes}", end="\r", flush=True)
    print()
    return (
        np.asarray(profiles, dtype=np.float64),
        np.asarray(spreads, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64),
        np.asarray(mo_signs, dtype=np.int64),
    )


def main() -> None:
    params = FlowParams()

    with stage("price series"):
        mids = []
        for i, seed in enumerate(TRAIN_SEEDS[:N_EPISODES]):
            mids.append(run_episode(seed, None, CONFIG, params).mid)
            print(f"    price episode {i + 1}/{N_EPISODES}", end="\r", flush=True)
        print()

    with stage("microstructure"):
        profiles, spreads, sizes, mo_signs = collect_market_microstructure(N_EPISODES, params)

    # Statistics that must be pooled across episodes are averaged per episode first, so each
    # episode contributes equally and the standard error is across independent runs.
    returns = [log_returns(m) for m in mids]
    kurtosis = np.array([excess_kurtosis(r) for r in returns])
    acf_abs_1 = np.array([autocorrelation(np.abs(r), 1) for r in returns])
    acf_abs_10 = np.array([autocorrelation(np.abs(r), 10) for r in returns])
    acf_raw_1 = np.array([autocorrelation(r, 1) for r in returns])
    vr_2 = np.array([variance_ratio(m, 2) for m in mids])
    vr_10 = np.array([variance_ratio(m, 10) for m in mids])
    vr_30 = np.array([variance_ratio(m, 30) for m in mids])
    mo_acf = autocorrelation(mo_signs.astype(np.float64), 1)
    mean_profile = profiles.mean(axis=0)

    facts = [
        StylizedFact(
            "return_excess_kurtosis",
            float(np.nanmean(kurtosis)),
            0.5,
            50.0,
            "",
            "Excess kurtosis of 1s mid log-returns; Gaussian is 0, real assets are strongly positive.",
        ),
        StylizedFact(
            "volatility_clustering_lag1",
            float(np.nanmean(acf_abs_1)),
            0.02,
            0.60,
            "",
            "Autocorrelation of |returns| at lag 1: positive and persistent in real data.",
        ),
        StylizedFact(
            "volatility_clustering_lag10",
            float(np.nanmean(acf_abs_10)),
            0.0,
            0.40,
            "",
            "Autocorrelation of |returns| at lag 10: decays slowly, staying above zero.",
        ),
        StylizedFact(
            "raw_return_acf_lag1",
            float(np.nanmean(acf_raw_1)),
            -0.35,
            0.05,
            "",
            "Autocorrelation of raw returns: near zero, slightly negative from bid-ask bounce.",
        ),
        StylizedFact(
            "variance_ratio_10",
            float(np.nanmean(vr_10)),
            0.70,
            1.30,
            "",
            "Lo-MacKinlay variance ratio at 10s. Near 1 means the mid is close to a martingale; "
            "far below 1 would mean a market maker is paid by mean reversion rather than by "
            "providing liquidity.",
        ),
        StylizedFact(
            "variance_ratio_30",
            float(np.nanmean(vr_30)),
            0.70,
            1.30,
            "",
            "Variance ratio at 30s: the mid must stay diffusive at longer horizons too.",
        ),
        StylizedFact(
            "median_spread_ticks",
            float(np.median(spreads)),
            1.0,
            2.0,
            "ticks",
            "Median quoted spread. The target regime is tick-constrained: mostly 1 tick.",
        ),
        StylizedFact(
            "mean_touch_depth_lots",
            float(mean_profile[0]),
            5.0,
            60.0,
            "lots",
            "Mean resting volume at the touch. Must be large enough that queue position matters.",
        ),
        StylizedFact(
            "order_size_tail_index",
            float(hill_tail_index(sizes.astype(np.float64))),
            1.4,
            3.0,
            "alpha",
            "Hill tail index of order sizes; equity and futures data report roughly 1.5-2.5.",
        ),
        StylizedFact(
            "market_order_sign_acf_lag1",
            float(mo_acf),
            0.0,
            0.50,
            "",
            "Autocorrelation of market-order signs: real order flow is strongly persistent.",
        ),
        StylizedFact(
            "depth_profile_hump_index",
            float(depth_profile_hump(mean_profile)),
            1.0,
            4.0,
            "levels",
            "Index of the fullest level away from the touch; real books are hump-shaped.",
        ),
    ]

    signature = signature_plot(np.concatenate(mids), horizons=(1, 2, 5, 10, 30))

    payload = {
        "n_episodes": N_EPISODES,
        "config": {
            "horizon_seconds": CONFIG.horizon_seconds,
            "burn_in_seconds": CONFIG.burn_in_seconds,
            "sample_interval": CONFIG.sample_interval,
        },
        "flow_params": {k: str(v) for k, v in vars(params).items()},
        "branching_ratio": params.branching_ratio,
        "facts": [f.to_dict() for f in facts],
        "n_passing": sum(f.passes for f in facts),
        "n_facts": len(facts),
        "mean_depth_profile": mean_profile.tolist(),
        "signature_plot": {str(k): v for k, v in signature.items()},
        "spread_distribution": {
            str(int(s)): int(c)
            for s, c in zip(*np.unique(spreads, return_counts=True), strict=True)
        },
        "variance_ratio_by_horizon": {
            "2": float(np.nanmean(vr_2)),
            "10": float(np.nanmean(vr_10)),
            "30": float(np.nanmean(vr_30)),
        },
        "diagnostics": {
            "mean_new_orders_per_episode": float(len(sizes) / N_EPISODES),
            "mean_market_orders_per_episode": float(len(mo_signs) / N_EPISODES),
            "kurtosis_sd_across_episodes": float(np.nanstd(kurtosis)),
        },
    }
    write_json("validation.json", payload)

    print()
    print(f"  {'fact':<34} {'value':>10}  {'target':>16}  result")
    for fact in facts:
        band = f"[{fact.target_low:g}, {fact.target_high:g}]"
        mark = "PASS" if fact.passes else "FAIL"
        print(f"  {fact.name:<34} {fact.value:>10.4f}  {band:>16}  {mark}")
    print(f"\n  {payload['n_passing']}/{payload['n_facts']} stylized facts within target bands")


if __name__ == "__main__":
    main()
