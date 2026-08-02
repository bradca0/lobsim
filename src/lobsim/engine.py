"""Discrete-event simulation loop tying the order book, the background flow, and an agent together.

The loop advances by sampled inter-event times. Agent decisions are interleaved on a fixed wall
clock so that a policy cannot react faster than its stated latency budget, and mark-to-market is
sampled on a separate, coarser clock to produce the return series used for Sharpe and validation.

Two engine policies deserve attention because they are where most backtests quietly cheat:

* **Re-quoting costs queue priority.** If the agent asks for a price it is already resting at, the
  existing order is *left alone* and keeps its place in line. Any change of price cancels and
  replaces, which sends it to the back of the new queue. A market maker that churns its quotes
  therefore never accumulates queue position -- as it should be.
* **Post-only.** A quote that would cross the opposite touch is rejected rather than silently
  turned into an aggressive order. Rejections are counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lobsim.book import LimitOrderBook
from lobsim.flow import EventKind, FlowEvent, FlowParams, OrderFlowGenerator
from lobsim.types import AgentState, BookSnapshot, Fill, FillModel, Side, Trade


@dataclass(frozen=True, slots=True)
class Quote:
    """A desired two-sided quote in absolute tick prices. ``None`` means "no order on that side"."""

    bid_price: int | None
    ask_price: int | None
    size: int = 1

    @staticmethod
    def flat() -> Quote:
        return Quote(bid_price=None, ask_price=None, size=0)


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Everything an agent may look at when deciding. Deliberately narrow: no lookahead."""

    ts: int
    snapshot: BookSnapshot
    inventory: int
    step: int
    steps_total: int
    # Public tape since the previous decision. A real market maker sees prints, so withholding
    # them would understate what any deployed policy could condition on.
    trade_flow: int = 0
    traded_volume: int = 0

    @property
    def time_remaining(self) -> float:
        """Fraction of the episode left, in [0, 1]. Drives inventory urgency near the close."""
        return 1.0 - self.step / max(self.steps_total, 1)


class Agent(Protocol):
    """The policy interface. Implementations live in ``lobsim.agents``."""

    name: str

    def reset(self, rng: np.random.Generator) -> None:
        """Called once at the start of each episode."""

    def act(self, ctx: MarketContext) -> Quote:
        """Return the desired quote for this decision epoch."""

    def observe_fill(self, fill: Fill) -> None:
        """Called for every execution of the agent's own orders."""


@dataclass(frozen=True)
class SimConfig:
    """Simulation timing and risk limits."""

    horizon_seconds: float = 300.0
    burn_in_seconds: float = 30.0
    decision_interval: float = 0.5
    sample_interval: float = 1.0
    max_inventory: int = 50
    snapshot_levels: int = 5
    liquidate_at_close: bool = True


@dataclass
class EpisodeResult:
    """Everything an episode produces. Arrays are aligned on the mark-to-market sample clock."""

    seed: int
    policy: str
    fill_model: str
    pnl: float
    ts: np.ndarray
    mid: np.ndarray
    inventory: np.ndarray
    mark_to_market: np.ndarray
    fills: list[Fill] = field(default_factory=list)
    trade_prices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    trade_ts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    n_events: int = 0
    n_agent_quotes: int = 0
    post_only_rejections: int = 0
    liquidation_cost: float = 0.0
    # Accumulated exactly, event by event, rather than reconstructed from sampled series -- see
    # the note on _mark_inventory below.
    spread_capture: float = 0.0
    inventory_pnl: float = 0.0
    unmarkable_events: int = 0

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def filled_volume(self) -> int:
        return sum(f.size for f in self.fills)


class Simulation:
    """One episode: a book, a flow generator, and at most one agent."""

    def __init__(
        self,
        config: SimConfig,
        flow_params: FlowParams,
        seed: int,
        agent: Agent | None = None,
        fill_model: FillModel = FillModel.QUEUE_AWARE,
    ) -> None:
        self.config = config
        self.flow_params = flow_params
        self.seed = seed
        self.agent = agent
        self.fill_model = fill_model
        # Common random numbers: the flow and agent streams are independent, so a given seed
        # supplies the same underlying randomness to every policy. This is a variance-reduction
        # device, not a guarantee of identical flow -- the agent is a market participant, so its
        # quotes change the touch and absorb aggressive orders, and the realised event sequence
        # therefore diverges between policies. Comparisons are paired in the CRN sense; see
        # docs/DECISIONS.md D6.
        flow_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(flow_seed)
        self.agent_rng = np.random.default_rng(agent_seed)
        self.book = LimitOrderBook(fill_model=fill_model)
        self.flow = OrderFlowGenerator(flow_params, self.rng)
        self.state = AgentState()
        self._agent_order_ids: dict[Side, int | None] = {Side.BUY: None, Side.SELL: None}
        self._post_only_rejections = 0
        self._n_quotes = 0
        self._tape_flow = 0
        self._tape_volume = 0
        self._spread_capture = 0.0
        self._inventory_pnl = 0.0
        self._mark_mid: float | None = None
        self._unmarkable = 0

    # ------------------------------------------------------------------ agent plumbing

    def _agent_order(self, side: Side) -> int | None:
        order_id = self._agent_order_ids[side]
        if order_id is None:
            return None
        if self.book.get_order(order_id) is None:
            self._agent_order_ids[side] = None
            return None
        return order_id

    def _record_trades(self, trades: list[Trade], ts: int) -> None:
        """Convert engine trades into agent fills and update inventory and cash.

        ``mid_at_fill`` is the mid *after* the trade has been applied to the book, which is the
        correct reference for a markout: it is the price a taker could next transact at.
        """
        mid = self.book.mid
        for trade in trades:
            if not (trade.maker_is_agent or trade.taker_is_agent):
                continue
            is_maker = trade.maker_is_agent
            # The maker sits on the opposite side of the aggressor by construction.
            side = trade.aggressor.opposite if is_maker else trade.aggressor
            fill = Fill(
                ts=ts,
                side=side,
                price=trade.price,
                size=trade.size,
                is_maker=is_maker,
                mid_at_fill=mid,
            )
            if mid is not None:
                self._spread_capture += fill.side.sign * fill.size * (mid - fill.price)
            self.state.apply(fill)
            if self.agent is not None:
                self.agent.observe_fill(fill)

    def _mark_inventory(self, inventory_before: float, mid_before: float | None) -> None:
        """Attribute the mid move across one action to the position held going into it.

        Every action is bracketed by this call, which makes the PnL decomposition an *identity*
        rather than an approximation::

            d(cash + q*m) = sign*size*(m_after - price)   [spread capture, per fill]
                          + q_before * (m_after - m_before)   [inventory PnL]

        Reconstructing the second term from a sampled inventory and mid series -- the usual
        approach -- is only approximate, because both change between samples. Accumulating it here,
        at every event, means ``spread_capture + inventory_pnl`` reproduces realised PnL to
        floating-point precision, and tests assert exactly that.
        """
        mid_after = self.book.mid
        if mid_before is None or mid_after is None:
            self._unmarkable += 1
            return
        if mid_after != mid_before:
            self._inventory_pnl += inventory_before * (mid_after - mid_before)

    def _cancel_agent_side(self, side: Side) -> None:
        order_id = self._agent_order(side)
        if order_id is not None:
            self.book.cancel(order_id)
            self._agent_order_ids[side] = None

    def _place(self, ts: int, side: Side, price: int, size: int) -> None:
        opposite_best = self.book.best(side.opposite)
        # ``side.is_better(opposite_best, price)`` is true exactly when the quote rests behind the
        # opposite touch: a bid strictly below the best ask, or an ask strictly above the best bid.
        if opposite_best is not None and not side.is_better(opposite_best, price):
            # Would cross: post-only rejects rather than crossing into an aggressive order.
            self._post_only_rejections += 1
            return
        order_id, trades = self.book.add_limit(ts, side, price, size, is_agent=True)
        self._record_trades(trades, ts)
        if self.book.get_order(order_id) is not None:
            self._agent_order_ids[side] = order_id

    def _apply_quote(self, ts: int, quote: Quote) -> None:
        """Reconcile the resting agent orders with the desired quote, preserving priority."""
        self._n_quotes += 1
        inventory = self.state.inventory
        limit = self.config.max_inventory
        for side, desired in ((Side.BUY, quote.bid_price), (Side.SELL, quote.ask_price)):
            # Hard risk limit: never quote a side that would push inventory past the cap.
            blocked = (side is Side.BUY and inventory >= limit) or (
                side is Side.SELL and inventory <= -limit
            )
            if desired is None or quote.size <= 0 or blocked:
                self._cancel_agent_side(side)
                continue
            order_id = self._agent_order(side)
            if order_id is not None:
                resting = self.book.get_order(order_id)
                if resting is not None and resting.price == desired:
                    continue  # already there -- keep our place in the queue
                self._cancel_agent_side(side)
            self._place(ts, side, desired, quote.size)

    # ------------------------------------------------------------------ main loop

    def run(self) -> EpisodeResult:
        cfg = self.config
        self.flow.seed_book(self.book)
        if self.agent is not None:
            self.agent.reset(self.agent_rng)

        # The seeded ladder is an arbitrary initial condition. Burn-in lets the background flow
        # relax to its own steady-state depth and spread before the agent trades or anything is
        # recorded, so results measure the market rather than the initialisation.
        burn_in = cfg.burn_in_seconds
        end_time = burn_in + cfg.horizon_seconds
        steps_total = int(cfg.horizon_seconds / cfg.decision_interval)
        next_decision = burn_in + cfg.decision_interval
        next_sample = burn_in + cfg.sample_interval
        step = 0

        ts_log: list[int] = []
        mid_log: list[float] = []
        inv_log: list[int] = []
        mtm_log: list[float] = []
        trade_px: list[int] = []
        trade_ts: list[int] = []

        last_mid = float(self.flow_params.initial_price)
        t = 0.0
        n_events = 0

        while t < end_time:
            t, event = self.flow.next_event(self.book, t)
            if t >= end_time:
                break

            # Interleave scheduled agent decisions and mark-to-market samples that fall before
            # this event. Both clocks are independent of the (stochastic) event clock.
            while next_decision <= t:
                if self.agent is not None:
                    ts_ns = int(next_decision * 1e9)
                    ctx = MarketContext(
                        ts=ts_ns,
                        snapshot=self.book.snapshot(ts_ns, cfg.snapshot_levels),
                        inventory=self.state.inventory,
                        step=step,
                        steps_total=steps_total,
                        trade_flow=self._tape_flow,
                        traded_volume=self._tape_volume,
                    )
                    inv_before, mid_before = self.state.inventory, self.book.mid
                    self._apply_quote(ts_ns, self.agent.act(ctx))
                    self._mark_inventory(inv_before, mid_before)
                self._tape_flow = 0
                self._tape_volume = 0
                next_decision += cfg.decision_interval
                step += 1
            while next_sample <= t:
                mid = self.book.mid
                if mid is not None:
                    last_mid = mid
                ts_log.append(int(next_sample * 1e9))
                mid_log.append(last_mid)
                inv_log.append(self.state.inventory)
                mtm_log.append(self.state.mark_to_market(last_mid))
                next_sample += cfg.sample_interval

            if event is not None:
                n_events += 1
                inv_before, mid_before = self.state.inventory, self.book.mid
                trades = self._apply_event(event)
                self._mark_inventory(inv_before, mid_before)
                for trade in trades:
                    trade_px.append(trade.price)
                    trade_ts.append(trade.ts)
                    self._tape_flow += trade.aggressor.sign * trade.size
                    self._tape_volume += trade.size

        inv_before, mid_before = self.state.inventory, self.book.mid
        liquidation_cost = self._close_out(end_time)
        self._mark_inventory(inv_before, mid_before)
        final_mid = self.book.mid if self.book.mid is not None else last_mid

        return EpisodeResult(
            seed=self.seed,
            policy=self.agent.name if self.agent is not None else "none",
            fill_model=self.fill_model.value,
            pnl=self.state.mark_to_market(final_mid),
            ts=np.asarray(ts_log, dtype=np.int64),
            mid=np.asarray(mid_log, dtype=np.float64),
            inventory=np.asarray(inv_log, dtype=np.int64),
            mark_to_market=np.asarray(mtm_log, dtype=np.float64),
            fills=list(self.state.fills),
            trade_prices=np.asarray(trade_px, dtype=np.int64),
            trade_ts=np.asarray(trade_ts, dtype=np.int64),
            n_events=n_events,
            n_agent_quotes=self._n_quotes,
            post_only_rejections=self._post_only_rejections,
            liquidation_cost=liquidation_cost,
            spread_capture=self._spread_capture,
            inventory_pnl=self._inventory_pnl,
            unmarkable_events=self._unmarkable,
        )

    def _apply_event(self, event: FlowEvent) -> list[Trade]:
        """Route a background event into the book and capture any agent executions."""
        if event.kind is EventKind.MARKET:
            _, trades = self.book.add_market(event.ts, event.side, event.size)
            self._record_trades(trades, event.ts)
            return trades
        if event.kind is EventKind.LIMIT:
            _, trades = self.book.add_limit(event.ts, event.side, event.price, event.size)
            self._record_trades(trades, event.ts)
            return trades
        self.book.cancel(event.order_id)
        return []

    def _close_out(self, end_time: float) -> float:
        """Flatten inventory with a market order and charge the resulting slippage.

        Without this, a policy can hide a large adverse position in the terminal mark and book a
        paper profit it could never realise. The cost is reported separately so the size of the
        effect is visible.
        """
        ts = int(end_time * 1e9)
        for side in (Side.BUY, Side.SELL):
            self._cancel_agent_side(side)
        inventory = self.state.inventory
        if not self.config.liquidate_at_close or inventory == 0:
            return 0.0
        mid_before = self.book.mid
        side = Side.SELL if inventory > 0 else Side.BUY
        _, trades = self.book.add_market(ts, side, abs(inventory), is_agent=True)
        self._record_trades(trades, ts)
        if mid_before is None:
            return 0.0
        # Slippage relative to marking the whole position at mid.
        executed = sum(t.size * t.price for t in trades)
        volume = sum(t.size for t in trades)
        if volume == 0:
            return 0.0
        return float(abs(executed - volume * mid_before))


def run_episode(
    seed: int,
    agent: Agent | None,
    config: SimConfig | None = None,
    flow_params: FlowParams | None = None,
    fill_model: FillModel = FillModel.QUEUE_AWARE,
) -> EpisodeResult:
    """Convenience wrapper: one seed in, one :class:`EpisodeResult` out."""
    sim = Simulation(
        config=config or SimConfig(),
        flow_params=flow_params or FlowParams(),
        seed=seed,
        agent=agent,
        fill_model=fill_model,
    )
    return sim.run()
