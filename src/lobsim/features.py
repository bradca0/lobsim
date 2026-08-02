"""Microstructure feature extraction.

Features are grouped so that whole groups can be ablated, which is how the repo answers "does the
learned policy actually use queue position, or is it just reading order-book imbalance?".

Every feature is computable from information a real market maker has at the decision instant: the
visible book, the public tape, and its own position and resting orders. Nothing here looks at the
latent fundamental, at future prices, or at other participants' identities. That restriction is
the whole reason the numbers downstream mean anything, so it is enforced by the type of the input:
:class:`~lobsim.engine.MarketContext` simply does not carry the forbidden quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from lobsim.types import Side

if TYPE_CHECKING:
    from lobsim.engine import MarketContext

# Feature groups. The keys are also the ablation switches used by scripts/run_ablations.py.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "book": (
        "spread",
        "imbalance_l1",
        "imbalance_l5",
        "microprice_gap",
        "log_touch_depth_bid",
        "log_touch_depth_ask",
    ),
    "flow": (
        "trade_flow_ewma",
        "volume_ewma",
        "momentum_short",
        "momentum_long",
        "volatility",
    ),
    "queue": (
        "queue_ahead_bid",
        "queue_ahead_ask",
        "has_bid",
        "has_ask",
    ),
    "position": (
        "inventory",
        "time_remaining",
        "inventory_time_pressure",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(name for group in FEATURE_GROUPS.values() for name in group)
N_FEATURES = len(FEATURE_NAMES)
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def group_mask(groups: tuple[str, ...]) -> np.ndarray:
    """Boolean mask selecting the features belonging to ``groups``."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature group(s): {sorted(unknown)}")
    mask = np.zeros(N_FEATURES, dtype=bool)
    for group in groups:
        for name in FEATURE_GROUPS[group]:
            mask[_INDEX[name]] = True
    return mask


@dataclass
class FeatureConfig:
    """Smoothing constants, in units of decision steps."""

    flow_halflife: float = 4.0
    volatility_halflife: float = 20.0
    momentum_short: int = 2
    momentum_long: int = 20
    max_inventory: int = 50
    depth_scale: float = 20.0


@dataclass
class FeatureExtractor:
    """Stateful per-episode feature builder.

    Stateful because volatility, momentum and smoothed flow are all path-dependent. ``reset`` must
    be called at the start of every episode or state leaks across independent runs -- which would
    be a subtle form of lookahead, since episode *k*'s features would encode episode *k-1*.
    """

    config: FeatureConfig = field(default_factory=FeatureConfig)
    _mid_history: list[float] = field(default_factory=list)
    _flow_ewma: float = 0.0
    _volume_ewma: float = 0.0
    _variance_ewma: float = 0.0
    _last_mid: float | None = None

    def reset(self) -> None:
        self._mid_history = []
        self._flow_ewma = 0.0
        self._volume_ewma = 0.0
        self._variance_ewma = 0.0
        self._last_mid = None

    @staticmethod
    def _alpha(halflife: float) -> float:
        return float(1.0 - 0.5 ** (1.0 / max(halflife, 1e-9)))

    def extract(self, ctx: MarketContext) -> np.ndarray:
        cfg = self.config
        snap = ctx.snapshot
        mid = snap.mid
        if mid is None:
            # One-sided book: fall back to the last known mid so features stay finite. This is
            # rare but must not produce NaNs, which would silently poison the regression targets.
            mid = self._last_mid if self._last_mid is not None else 0.0

        change = 0.0 if self._last_mid is None else mid - self._last_mid
        self._last_mid = mid
        self._mid_history.append(mid)

        flow_alpha = self._alpha(cfg.flow_halflife)
        self._flow_ewma += flow_alpha * (ctx.trade_flow - self._flow_ewma)
        self._volume_ewma += flow_alpha * (ctx.traded_volume - self._volume_ewma)
        vol_alpha = self._alpha(cfg.volatility_halflife)
        self._variance_ewma += vol_alpha * (change * change - self._variance_ewma)

        bid_depth = snap.bids[0][1] if snap.bids else 0
        ask_depth = snap.asks[0][1] if snap.asks else 0
        microprice = snap.microprice
        spread = snap.spread

        ahead_bid, ahead_ask = snap.agent_volume_ahead
        scale = cfg.depth_scale

        features = np.empty(N_FEATURES, dtype=np.float64)
        features[_INDEX["spread"]] = float(spread) if spread is not None else 0.0
        features[_INDEX["imbalance_l1"]] = snap.imbalance
        features[_INDEX["imbalance_l5"]] = _imbalance(
            snap.depth(Side.BUY, 5), snap.depth(Side.SELL, 5)
        )
        features[_INDEX["microprice_gap"]] = (microprice - mid) if microprice is not None else 0.0
        features[_INDEX["log_touch_depth_bid"]] = np.log1p(bid_depth)
        features[_INDEX["log_touch_depth_ask"]] = np.log1p(ask_depth)

        features[_INDEX["trade_flow_ewma"]] = self._flow_ewma / scale
        features[_INDEX["volume_ewma"]] = self._volume_ewma / scale
        features[_INDEX["momentum_short"]] = self._momentum(cfg.momentum_short)
        features[_INDEX["momentum_long"]] = self._momentum(cfg.momentum_long)
        features[_INDEX["volatility"]] = float(np.sqrt(max(self._variance_ewma, 0.0)))

        # Queue position is reported as the fraction of the level that sits in front of us, so it
        # is comparable across levels of very different absolute depth. A resting order with an
        # empty queue ahead of it is worth far more than one buried behind 40 lots, and the ratio
        # -- not the raw count -- is what carries that information.
        features[_INDEX["queue_ahead_bid"]] = _queue_fraction(ahead_bid, bid_depth)
        features[_INDEX["queue_ahead_ask"]] = _queue_fraction(ahead_ask, ask_depth)
        features[_INDEX["has_bid"]] = 1.0 if ahead_bid is not None else 0.0
        features[_INDEX["has_ask"]] = 1.0 if ahead_ask is not None else 0.0

        inventory = ctx.inventory / max(cfg.max_inventory, 1)
        features[_INDEX["inventory"]] = inventory
        features[_INDEX["time_remaining"]] = ctx.time_remaining
        features[_INDEX["inventory_time_pressure"]] = inventory * (1.0 - ctx.time_remaining)
        return features

    def _momentum(self, window: int) -> float:
        history = self._mid_history
        if len(history) <= window:
            return 0.0
        return history[-1] - history[-1 - window]


def _imbalance(bid_volume: int, ask_volume: int) -> float:
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return (bid_volume - ask_volume) / total


def _queue_fraction(ahead: int | None, depth: int) -> float:
    """Fraction of the level resting in front of our order; 1.0 when we have no order there.

    Encoding "no order" as 1.0 rather than 0.0 or NaN keeps the feature monotone in badness: no
    order and a hopeless queue position are both states in which a fill is not coming soon.
    """
    if ahead is None:
        return 1.0
    if depth <= 0:
        return 0.0
    return min(ahead / depth, 1.0)
