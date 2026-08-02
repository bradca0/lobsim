"""Synthetic order-flow generator.

The generator is a *zero-intelligence* background market (Cont, Stoikov & Talreja 2010) with one
deliberate departure: market-order arrivals follow a bivariate self- and cross-exciting Hawkes
process rather than a Poisson process. Zero-intelligence flow alone produces a book with realistic
shape but Gaussian, uncorrelated returns; the Hawkes layer is what generates the volatility
clustering and fat tails that real price series exhibit. ``lobsim.validation`` measures whether it
actually does, and the result is reported rather than assumed.

Three event streams compete:

* **Market orders** -- Hawkes intensity per side, tilted toward a latent fundamental value, with
  size from a truncated discrete Pareto. The tilt is what makes aggressive flow *informed*: when
  the fundamental sits above the mid, buy market orders arrive faster, so a market maker resting
  on the offer is systematically run over just before the price moves against it. Without this
  layer, market making in the simulator would be riskless spread capture and the entire exercise
  would be vacuous.
* **Limit orders** -- Poisson, placed ``k`` ticks behind the opposite touch where ``k`` follows a
  discrete power law. Placement relative to the *opposite* best makes crossing structurally
  impossible, so aggression is expressed only through market orders.
* **Cancellations** -- rate proportional to resting non-agent volume, which is what makes the
  steady-state book depth self-correcting rather than a free parameter.

Sampling uses Ogata's thinning: between events the limit and cancel intensities are constant and
the Hawkes intensity only decays, so the total intensity immediately after the previous event is a
valid upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from lobsim.types import Order, Side

if TYPE_CHECKING:
    from lobsim.book import LimitOrderBook


class CancelPolicy(Enum):
    """Where in the queue cancellations land.

    This is the single most consequential modelling choice for a market maker, because it decides
    whether the queue in front of you evaporates or persists. ``UNIFORM`` cancels a lot chosen
    uniformly at random, so on average half of all cancelled volume is ahead of you. ``BACK_LOADED``
    weights orders toward the back of the queue, which is closer to what is observed empirically
    (early-queue orders are placed by participants who wanted the priority) and is markedly less
    generous to a late-arriving market maker. Both are reported; see the cancel-policy ablation.
    """

    UNIFORM = "uniform"
    BACK_LOADED = "back_loaded"


class EventKind(Enum):
    MARKET = "market"
    LIMIT = "limit"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class FlowEvent:
    """One background-market action, timestamped in integer nanoseconds."""

    ts: int
    kind: EventKind
    side: Side
    size: int = 0
    price: int = 0
    order_id: int = 0


@dataclass(frozen=True)
class FlowParams:
    """Order-flow parameters. Rates are per second; prices are in ticks.

    Defaults were tuned so the simulated book reproduces the stylized facts checked in
    ``lobsim.validation`` -- see docs/DECISIONS.md D4 for how they were chosen and what the
    tuning procedure could and could not identify.
    """

    # Hawkes market-order arrivals (per side).
    mo_baseline: float = 0.80
    mo_self_excite: float = 0.22
    mo_cross_excite: float = 0.10
    mo_decay: float = 0.9

    # Limit-order arrivals (total across both sides) and depth placement.
    lo_rate: float = 18.0
    lo_depth_exponent: float = 1.3
    lo_max_depth: int = 10

    # Cancellations: per-lot hazard rate applied to resting non-agent volume.
    cancel_rate_per_lot: float = 0.09
    cancel_policy: CancelPolicy = CancelPolicy.UNIFORM

    # Latent fundamental value: a random walk in tick units that aggressive flow chases.
    fundamental_vol: float = 0.45
    informed_kappa: float = 0.60
    informed_clip: float = 2.5

    # Order sizes: discrete Pareto, truncated.
    size_alpha: float = 1.8
    size_max: int = 60

    # Initial book construction.
    initial_price: int = 10_000
    initial_depth_levels: int = 6
    initial_level_size: int = 16

    def __post_init__(self) -> None:
        branching = (self.mo_self_excite + self.mo_cross_excite) / self.mo_decay
        if branching >= 1.0:
            raise ValueError(
                f"Hawkes branching ratio {branching:.3f} >= 1: the process explodes. "
                "Require (mo_self_excite + mo_cross_excite) < mo_decay."
            )
        if self.lo_max_depth < 1:
            raise ValueError("lo_max_depth must be >= 1")
        if self.size_max < 1:
            raise ValueError("size_max must be >= 1")
        if self.fundamental_vol < 0.0:
            raise ValueError("fundamental_vol must be non-negative")
        if self.informed_clip <= 0.0:
            raise ValueError("informed_clip must be positive")

    @property
    def branching_ratio(self) -> float:
        """Expected offspring per market order. Must be < 1 for stationarity."""
        return (self.mo_self_excite + self.mo_cross_excite) / self.mo_decay


@dataclass
class _Hawkes:
    """Bivariate exponential-kernel Hawkes intensity, tracked by decaying accumulators."""

    params: FlowParams
    excitation: dict[Side, float] = field(default_factory=lambda: {Side.BUY: 0.0, Side.SELL: 0.0})
    last_t: float = 0.0

    def decay_to(self, t: float) -> None:
        dt = t - self.last_t
        if dt <= 0.0:
            return
        factor = math.exp(-self.params.mo_decay * dt)
        for side in self.excitation:
            self.excitation[side] *= factor
        self.last_t = t

    def excitation_at(self, t: float, side: Side) -> float:
        """Decayed excitation for ``side`` at a future time ``t``, without mutating state.

        The baseline is *not* included: it is side-dependent once informed flow tilts it, so the
        caller owns it.
        """
        factor = math.exp(-self.params.mo_decay * (t - self.last_t))
        return self.excitation[side] * factor

    def intensity_at(self, t: float, side: Side, baseline: float) -> float:
        return baseline + self.excitation_at(t, side)

    def excite(self, t: float, side: Side) -> None:
        self.decay_to(t)
        self.excitation[side] += self.params.mo_self_excite
        self.excitation[side.opposite] += self.params.mo_cross_excite


class OrderFlowGenerator:
    """Draws the next background-market event given the current book state."""

    def __init__(self, params: FlowParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self._hawkes = _Hawkes(params)
        self._depth_cdf = self._build_depth_probs(params)
        self.fundamental = float(params.initial_price)

    @staticmethod
    def _build_depth_probs(params: FlowParams) -> np.ndarray:
        ks = np.arange(params.lo_max_depth, dtype=np.float64)
        weights = 1.0 / np.power(ks + 1.0, params.lo_depth_exponent)
        # Stored as a CDF so placement costs one uniform draw and a searchsorted, rather than a
        # full categorical sample per limit order. This is the hottest path in the simulator.
        return np.cumsum(weights / weights.sum())

    # ------------------------------------------------------------------ primitives

    def draw_size(self) -> int:
        """Truncated discrete Pareto: heavy right tail, mass concentrated on small orders.

        ``floor`` -- not ``ceil`` -- is what makes this the discrete Pareto with survival function
        ``P(S >= k) = k**-alpha``. With ``ceil`` the smallest attainable size is 2, because the
        continuous variate is almost surely strictly greater than 1, which silently removes the
        single most common order size in any real book.
        """
        u = 1.0 - self.rng.random()  # (0, 1]; rng.random() can return exactly 0
        raw = math.floor(math.pow(u, -1.0 / self.params.size_alpha))
        return min(raw, self.params.size_max)

    def seed_book(self, book: LimitOrderBook, ts: int = 0) -> None:
        """Populate an empty book with a symmetric ladder around ``initial_price``."""
        p = self.params
        for k in range(1, p.initial_depth_levels + 1):
            book.add_limit(ts, Side.BUY, p.initial_price - k, p.initial_level_size)
            book.add_limit(ts, Side.SELL, p.initial_price + k, p.initial_level_size)

    def _reference_price(self, book: LimitOrderBook, side: Side) -> int:
        """The most aggressive price a ``side`` limit order may take without crossing."""
        opposite_best = book.best(side.opposite)
        if opposite_best is not None:
            return opposite_best - side.sign
        own_best = book.best(side)
        if own_best is not None:
            return own_best
        anchor = book.last_trade_price
        return anchor if anchor is not None else self.params.initial_price

    # ------------------------------------------------------------------ informed flow

    def _informed_baselines(self, book: LimitOrderBook) -> tuple[float, float]:
        """Split the market-order baseline between sides according to the fundamental gap.

        When the fundamental sits ``g`` ticks above the mid, buy market orders arrive at
        ``exp(kappa * g)`` times the baseline and sells at ``exp(-kappa * g)``. The tilt is
        symmetric in log space, so the *total* aggressive rate is only mildly affected while the
        *direction* becomes strongly predictive of where the price is about to go. That predictive
        content is precisely the adverse selection a market maker must survive.
        """
        p = self.params
        mid = book.mid
        if mid is None or p.informed_kappa == 0.0:
            return p.mo_baseline, p.mo_baseline
        gap = self.fundamental - mid
        tilt = max(-p.informed_clip, min(p.informed_clip, p.informed_kappa * gap))
        multiplier = math.exp(tilt)
        return p.mo_baseline * multiplier, p.mo_baseline / multiplier

    def _advance_fundamental(self, now: float, t: float) -> None:
        """Diffuse the latent value over the elapsed interval.

        The fundamental is held piecewise constant *within* an inter-event interval and updated at
        its end. That is what keeps the thinning bound valid: the intensity used to sample the
        interval must not depend on randomness drawn inside it.
        """
        sigma = self.params.fundamental_vol
        if sigma <= 0.0:
            return
        dt = t - now
        if dt > 0.0:
            self.fundamental += float(self.rng.normal(0.0, sigma * math.sqrt(dt)))

    # ------------------------------------------------------------------ sampling

    def next_event(self, book: LimitOrderBook, now: float) -> tuple[float, FlowEvent | None]:
        """Sample the next background event at or after ``now`` (seconds).

        Returns the new simulation time and the event, or ``(t, None)`` when the thinning step
        rejects -- the caller simply advances its clock and asks again, which keeps the rejection
        loop out of this function and bounded by the caller's horizon.
        """
        p = self.params
        self._hawkes.decay_to(now)

        cancellable = book.total_volume(Side.BUY, include_agent=False) + book.total_volume(
            Side.SELL, include_agent=False
        )
        rate_cancel = p.cancel_rate_per_lot * cancellable
        rate_limit = p.lo_rate
        base_buy, base_sell = self._informed_baselines(book)
        # The fundamental and the book are both held constant across the interval, so only the
        # Hawkes term varies -- and it only decays. The intensity at ``now`` therefore bounds the
        # intensity everywhere in (now, t], which is what Ogata thinning requires.
        upper = (
            self._hawkes.intensity_at(now, Side.BUY, base_buy)
            + self._hawkes.intensity_at(now, Side.SELL, base_sell)
            + rate_limit
            + rate_cancel
        )
        t = now + self.rng.exponential(1.0 / upper)

        rate_buy = self._hawkes.intensity_at(t, Side.BUY, base_buy)
        rate_sell = self._hawkes.intensity_at(t, Side.SELL, base_sell)
        rates = (rate_buy, rate_sell, rate_limit, rate_cancel)
        total = sum(rates)
        self._advance_fundamental(now, t)
        if self.rng.random() * upper > total:
            return t, None  # thinning rejection

        u = self.rng.random() * total
        if u < rate_buy:
            return t, self._market_event(t, Side.BUY)
        u -= rate_buy
        if u < rate_sell:
            return t, self._market_event(t, Side.SELL)
        u -= rate_sell
        if u < rate_limit:
            return t, self._limit_event(t, book)
        return t, self._cancel_event(t, book, cancellable)

    def _market_event(self, t: float, side: Side) -> FlowEvent:
        self._hawkes.excite(t, side)
        return FlowEvent(ts=_to_ns(t), kind=EventKind.MARKET, side=side, size=self.draw_size())

    def _limit_event(self, t: float, book: LimitOrderBook) -> FlowEvent:
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        depth = int(np.searchsorted(self._depth_cdf, self.rng.random()))
        price = self._reference_price(book, side) - side.sign * depth
        return FlowEvent(
            ts=_to_ns(t),
            kind=EventKind.LIMIT,
            side=side,
            size=self.draw_size(),
            price=price,
        )

    def _cancel_event(self, t: float, book: LimitOrderBook, cancellable: int) -> FlowEvent | None:
        order = self._choose_cancel_target(book, cancellable)
        if order is None:
            return None
        return FlowEvent(
            ts=_to_ns(t),
            kind=EventKind.CANCEL,
            side=order.side,
            size=order.size,
            price=order.price,
            order_id=order.order_id,
        )

    def _choose_cancel_target(self, book: LimitOrderBook, cancellable: int) -> Order | None:
        """Pick a resting non-agent order to cancel, weighted by the active cancel policy.

        Selection walks price levels outward from the touch rather than materialising every
        resting order, so the cost is O(levels) per cancellation rather than O(orders).
        """
        if cancellable <= 0:
            return None
        buy_volume = book.total_volume(Side.BUY, include_agent=False)
        side = Side.BUY if self.rng.random() * cancellable < buy_volume else Side.SELL
        side_volume = book.total_volume(side, include_agent=False)
        if side_volume <= 0:
            side = side.opposite
            side_volume = book.total_volume(side, include_agent=False)
            if side_volume <= 0:
                return None

        target = self.rng.random() * side_volume
        for price in book.prices(side):
            level_volume = book.other_volume_at(side, price)
            if target < level_volume:
                return self._choose_within_level(book, side, price, target, level_volume)
            target -= level_volume
        return None  # pragma: no cover -- volumes are maintained exactly, so this is unreachable

    def _choose_within_level(
        self,
        book: LimitOrderBook,
        side: Side,
        price: int,
        target: float,
        level_volume: int,
    ) -> Order | None:
        orders = [o for o in book.orders_at(side, price) if not o.is_agent]
        if not orders:
            return None
        if self.params.cancel_policy is CancelPolicy.UNIFORM:
            cumulative = 0.0
            for order in orders:
                cumulative += order.size
                if target < cumulative:
                    return order
            return orders[-1]
        # BACK_LOADED: weight order i (front = 0) by size * (i + 1), so late arrivals are the
        # likeliest to be pulled and the front of the queue is comparatively sticky.
        weights = [order.size * (i + 1) for i, order in enumerate(orders)]
        pick = target / max(level_volume, 1) * sum(weights)
        cumulative = 0.0
        for order, weight in zip(orders, weights, strict=True):
            cumulative += weight
            if pick < cumulative:
                return order
        return orders[-1]


def _to_ns(t: float) -> int:
    """Seconds (float, generator-side) to integer nanoseconds (engine-side)."""
    return int(t * 1e9)
