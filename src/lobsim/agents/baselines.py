"""Rule-based market-making baselines.

These exist to be beaten, but they are implemented to be genuinely competitive rather than as
strawmen. In particular ``AlwaysAtTouch`` is a strong baseline in a tick-constrained book: with a
1-tick spread there is nowhere better to quote, so any learned policy has to earn its keep through
*when* it quotes and *which side* it skews, not through cleverer prices.

All baselines are pure functions of the current context plus their own parameters. None of them
looks at the latent fundamental.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lobsim.engine import MarketContext, Quote
from lobsim.types import Fill


@dataclass
class _BaseAgent:
    """Shared plumbing: naming, RNG handling, and fill bookkeeping."""

    size: int = 1
    name: str = "base"
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    _fills: list[Fill] = field(default_factory=list, repr=False)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._fills = []

    def observe_fill(self, fill: Fill) -> None:
        self._fills.append(fill)

    def act(self, ctx: MarketContext) -> Quote:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Inactive(_BaseAgent):
    """Never quotes. The control: its PnL must be exactly zero.

    Worth running on every experiment. If this policy ever reports non-zero PnL, the accounting is
    broken and every other number in the study is suspect.
    """

    name: str = "inactive"

    def act(self, ctx: MarketContext) -> Quote:
        return Quote.flat()


@dataclass
class AlwaysAtTouch(_BaseAgent):
    """Joins both sides of the touch unconditionally.

    The reference policy for a tick-constrained book: maximum fill rate, maximum queue exposure,
    no inventory management whatsoever beyond the engine's hard cap.
    """

    name: str = "at_touch"

    def act(self, ctx: MarketContext) -> Quote:
        snap = ctx.snapshot
        return Quote(bid_price=snap.best_bid, ask_price=snap.best_ask, size=self.size)


@dataclass
class FixedSpread(_BaseAgent):
    """Quotes a fixed half-spread around the mid, rounded outward so it never crosses.

    Rounding *outward* (floor on the bid, ceil on the ask) matters: rounding to nearest would make
    a half-tick quote sometimes marketable, which the post-only engine would reject, silently
    turning a "wide" policy into a "no quote" policy on half its decisions.
    """

    half_spread: int = 1
    name: str = "fixed_spread"

    def __post_init__(self) -> None:
        if self.half_spread < 1:
            raise ValueError("half_spread must be at least 1 tick")
        self.name = f"fixed_spread_{self.half_spread}"

    def act(self, ctx: MarketContext) -> Quote:
        mid = ctx.snapshot.mid
        if mid is None:
            return Quote.flat()
        return Quote(
            bid_price=math.floor(mid - self.half_spread),
            ask_price=math.ceil(mid + self.half_spread),
            size=self.size,
        )


@dataclass
class InventorySkew(_BaseAgent):
    """Quotes at the touch but pulls the side that would worsen an existing position.

    The simplest form of inventory control that is not a strawman: when long past the threshold,
    stop bidding and keep offering. This is what a desk does by hand before anyone writes a model.
    """

    threshold: int = 5
    name: str = "inventory_skew"

    def act(self, ctx: MarketContext) -> Quote:
        snap = ctx.snapshot
        bid: int | None = snap.best_bid
        ask: int | None = snap.best_ask
        if ctx.inventory >= self.threshold:
            bid = None
        elif ctx.inventory <= -self.threshold:
            ask = None
        return Quote(bid_price=bid, ask_price=ask, size=self.size)


@dataclass
class AvellanedaStoikov(_BaseAgent):
    """The Avellaneda-Stoikov (2008) inventory-aware quoting rule, discretised to the tick grid.

    Reservation price ``r = s - q * gamma * sigma^2 * (T - t)`` shifts quotes away from the side
    that would add to inventory, and the optimal total spread is
    ``gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)``.

    Two honest caveats, both of which the results section depends on. The model is derived for a
    *continuous* price with fills arriving at an exponentially decaying rate in quote distance; a
    tick-constrained book with a discrete queue violates both assumptions. And ``kappa`` -- the
    fill-intensity decay -- is not identified from the simulator's parameters, so it is calibrated
    on training seeds only, exactly like the learned policy's hyperparameters.
    """

    gamma: float = 0.05
    kappa: float = 1.5
    volatility_halflife: float = 20.0
    name: str = "avellaneda_stoikov"

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        self._variance = 0.0
        self._last_mid: float | None = None

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self._variance = 0.0
        self._last_mid = None

    def _update_volatility(self, mid: float) -> float:
        alpha = 1.0 - 0.5 ** (1.0 / max(self.volatility_halflife, 1e-9))
        if self._last_mid is not None:
            change = mid - self._last_mid
            self._variance += alpha * (change * change - self._variance)
        self._last_mid = mid
        return self._variance

    def act(self, ctx: MarketContext) -> Quote:
        mid = ctx.snapshot.mid
        if mid is None:
            return Quote.flat()
        variance = self._update_volatility(mid)
        horizon = ctx.time_remaining

        reservation = mid - ctx.inventory * self.gamma * variance * horizon
        spread = self.gamma * variance * horizon + (2.0 / self.gamma) * math.log1p(
            self.gamma / self.kappa
        )
        half = spread / 2.0
        return Quote(
            bid_price=math.floor(reservation - half),
            ask_price=math.ceil(reservation + half),
            size=self.size,
        )
