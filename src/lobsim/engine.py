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

from collections.abc import Callable
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
    # Accumulated exactly, one book mutation at a time, rather than reconstructed from sampled
    # series -- see the block comment above Simulation._mark.
    spread_capture: float = 0.0
    inventory_pnl: float = 0.0
    one_sided_events: int = 0

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
        on_decision: Callable[[float, int], None] | None = None,
        on_episode_end: Callable[[float], None] | None = None,
    ) -> None:
        # Optional observation hooks used only by training data collection. They receive the
        # mark-to-market and inventory at each decision and at the close, which is information the
        # *agent* is deliberately not given at reward time: rewards are computed by the trainer,
        # outside the policy, so a policy cannot condition on its own PnL when deciding.
        self.on_decision = on_decision
        self.on_episode_end = on_episode_end
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
        self._ref_mid = float(flow_params.initial_price)
        self._one_sided = 0

    # ------------------------------------------------------------------ agent plumbing

    def _agent_order(self, side: Side) -> int | None:
        order_id = self._agent_order_ids[side]
        if order_id is None:
            return None
        if self.book.get_order(order_id) is None:
            self._agent_order_ids[side] = None
            return None
        return order_id

    def _reference_price(self) -> float:
        """The price everything is marked at: the current mid, or the last valid one.

        A market order large enough to sweep a side clean leaves the mid *undefined*, and that
        happens often enough in a thin book to matter. Skipping those moments -- the obvious
        approach -- silently breaks the PnL identity, because the position still has to be marked
        somewhere and realised cash still moves. Carrying the last valid mid forward is what a desk
        actually does with a one-sided book, and it keeps a single consistent reference for spread
        capture, inventory marking, and the reported PnL. Occurrences are counted and reported as
        ``one_sided_events``.
        """
        mid = self.book.mid
        if mid is None:
            self._one_sided += 1
            return self._ref_mid
        self._ref_mid = mid
        return mid

    def _record_trades(self, trades: list[Trade], ts: int, mid: float) -> None:
        """Convert engine trades into agent fills and update inventory and cash.

        ``mid`` is the reference price *after* the trade has been applied to the book, which is the
        correct basis for a markout: it is the price a taker could next transact at.
        """
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
            self._spread_capture += fill.side.sign * fill.size * (mid - fill.price)
            self.state.apply(fill)
            if self.agent is not None:
                self.agent.observe_fill(fill)

    # -------------------------------------------------------- book mutation, exactly accounted
    #
    # Every mutation of the book goes through one of the three wrappers below. Each brackets a
    # *single* operation so that the PnL decomposition is an identity rather than an approximation:
    #
    #     d(cash + q*m) = sum_fills sign*size*(m_after - price)   [spread capture]
    #                   + q_before * (m_after - m_before)         [inventory PnL]
    #
    # The bracketing must be per-operation, not per-decision. Grouping several mutations under one
    # bracket -- for instance reconciling both sides of a two-sided quote together -- breaks the
    # identity whenever a fill happens on the first side and the mid then moves again on the
    # second, because the position that carried the second move is no longer the position recorded
    # at the start of the bracket. That bug was caught by the residual property test, not by
    # inspection, which is exactly why the residual is asserted over many seeds and every policy.

    def _mark(self, inventory_before: int, ref_before: float, ref_after: float) -> None:
        if ref_after != ref_before:
            self._inventory_pnl += inventory_before * (ref_after - ref_before)

    def _submit_limit(
        self, ts: int, side: Side, price: int, size: int, *, is_agent: bool
    ) -> tuple[int, list[Trade]]:
        inventory_before, ref_before = self.state.inventory, self._reference_price()
        order_id, trades = self.book.add_limit(ts, side, price, size, is_agent=is_agent)
        ref_after = self._reference_price()
        self._record_trades(trades, ts, ref_after)
        self._mark(inventory_before, ref_before, ref_after)
        return order_id, trades

    def _submit_market(
        self, ts: int, side: Side, size: int, *, is_agent: bool
    ) -> tuple[int, list[Trade]]:
        inventory_before, ref_before = self.state.inventory, self._reference_price()
        order_id, trades = self.book.add_market(ts, side, size, is_agent=is_agent)
        ref_after = self._reference_price()
        self._record_trades(trades, ts, ref_after)
        self._mark(inventory_before, ref_before, ref_after)
        return order_id, trades

    def _submit_cancel(self, order_id: int) -> bool:
        inventory_before, ref_before = self.state.inventory, self._reference_price()
        cancelled = self.book.cancel(order_id)
        self._mark(inventory_before, ref_before, self._reference_price())
        return cancelled

    def _cancel_agent_side(self, side: Side) -> None:
        order_id = self._agent_order(side)
        if order_id is not None:
            self._submit_cancel(order_id)
            self._agent_order_ids[side] = None

    def _place(self, ts: int, side: Side, price: int, size: int) -> None:
        opposite_best = self.book.best(side.opposite)
        # ``side.is_better(opposite_best, price)`` is true exactly when the quote rests behind the
        # opposite touch: a bid strictly below the best ask, or an ask strictly above the best bid.
        if opposite_best is not None and not side.is_better(opposite_best, price):
            # Would cross: post-only rejects rather than crossing into an aggressive order.
            self._post_only_rejections += 1
            return
        order_id, _ = self._submit_limit(ts, side, price, size, is_agent=True)
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
                    mark = self.state.mark_to_market(self._reference_price())
                    quote = self.agent.act(ctx)
                    if self.on_decision is not None:
                        self.on_decision(mark, self.state.inventory)
                    self._apply_quote(ts_ns, quote)
                self._tape_flow = 0
                self._tape_volume = 0
                next_decision += cfg.decision_interval
                step += 1
            while next_sample <= t:
                reference = self._reference_price()
                ts_log.append(int(next_sample * 1e9))
                mid_log.append(reference)
                inv_log.append(self.state.inventory)
                mtm_log.append(self.state.mark_to_market(reference))
                next_sample += cfg.sample_interval

            if event is not None:
                n_events += 1
                trades = self._apply_event(event)
                for trade in trades:
                    trade_px.append(trade.price)
                    trade_ts.append(trade.ts)
                    self._tape_flow += trade.aggressor.sign * trade.size
                    self._tape_volume += trade.size

        liquidation_cost = self._close_out(end_time)
        final_mid = self._reference_price()
        if self.on_episode_end is not None:
            self.on_episode_end(self.state.mark_to_market(final_mid))

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
            one_sided_events=self._one_sided,
        )

    def _apply_event(self, event: FlowEvent) -> list[Trade]:
        """Route a background event into the book and capture any agent executions."""
        if event.kind is EventKind.MARKET:
            _, trades = self._submit_market(event.ts, event.side, event.size, is_agent=False)
            return trades
        if event.kind is EventKind.LIMIT:
            _, trades = self._submit_limit(
                event.ts, event.side, event.price, event.size, is_agent=False
            )
            return trades
        self._submit_cancel(event.order_id)
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
        mid_before = self._reference_price()
        side = Side.SELL if inventory > 0 else Side.BUY
        _, trades = self._submit_market(ts, side, abs(inventory), is_agent=True)
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
