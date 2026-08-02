"""Simulation-loop tests: determinism, queue-priority accounting, risk limits, close-out."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from lobsim.engine import MarketContext, Quote, SimConfig, Simulation, run_episode
from lobsim.flow import FlowParams
from lobsim.types import Fill, FillModel, Side


@dataclass
class ScriptedAgent:
    """An agent that returns a fixed sequence of quotes, cycling when exhausted."""

    quotes: list[Quote]
    name: str = "scripted"
    seen: list[Fill] = field(default_factory=list)
    contexts: list[MarketContext] = field(default_factory=list)

    def reset(self, rng: np.random.Generator) -> None:
        self.seen = []
        self.contexts = []

    def act(self, ctx: MarketContext) -> Quote:
        self.contexts.append(ctx)
        return self.quotes[min(len(self.contexts) - 1, len(self.quotes) - 1)]

    def observe_fill(self, fill: Fill) -> None:
        self.seen.append(fill)


@dataclass
class TouchAgent:
    """Always quotes one tick behind each touch with a fixed size."""

    size: int = 1
    name: str = "touch"
    fills: list[Fill] = field(default_factory=list)

    def reset(self, rng: np.random.Generator) -> None:
        self.fills = []

    def act(self, ctx: MarketContext) -> Quote:
        snap = ctx.snapshot
        return Quote(bid_price=snap.best_bid, ask_price=snap.best_ask, size=self.size)

    def observe_fill(self, fill: Fill) -> None:
        self.fills.append(fill)


SHORT = SimConfig(horizon_seconds=60.0, burn_in_seconds=10.0)


class TestDeterminism:
    def test_same_seed_reproduces_the_episode_exactly(self) -> None:
        a = run_episode(7, TouchAgent(), SHORT)
        b = run_episode(7, TouchAgent(), SHORT)
        assert a.pnl == b.pnl
        assert a.n_fills == b.n_fills
        np.testing.assert_array_equal(a.mid, b.mid)
        np.testing.assert_array_equal(a.inventory, b.inventory)

    def test_different_seeds_produce_different_paths(self) -> None:
        a = run_episode(1, TouchAgent(), SHORT)
        b = run_episode(2, TouchAgent(), SHORT)
        assert not np.array_equal(a.mid, b.mid)

    def test_agent_stream_is_independent_of_the_flow_stream(self) -> None:
        """Two agents that never trade must still see identical background markets."""
        flat = run_episode(3, ScriptedAgent([Quote.flat()]), SHORT)
        none = run_episode(3, None, SHORT)
        np.testing.assert_array_equal(flat.mid, none.mid)


class TestRecording:
    def test_recording_starts_after_burn_in(self) -> None:
        result = run_episode(0, None, SHORT)
        assert result.ts[0] == int((SHORT.burn_in_seconds + SHORT.sample_interval) * 1e9)
        assert len(result.mid) == pytest.approx(SHORT.horizon_seconds, abs=2)

    def test_an_agentless_episode_has_no_pnl_and_no_fills(self) -> None:
        result = run_episode(0, None, SHORT)
        assert result.pnl == 0.0
        assert result.n_fills == 0
        assert result.policy == "none"
        assert result.n_events > 100

    def test_trades_are_recorded_for_validation(self) -> None:
        result = run_episode(0, None, SHORT)
        assert len(result.trade_prices) > 10
        assert len(result.trade_prices) == len(result.trade_ts)


class TestQueuePriorityOnRequote:
    """The engine must not hand a market maker free queue position when it re-quotes."""

    def _sim(self, quotes: list[Quote]) -> Simulation:
        return Simulation(
            config=SimConfig(horizon_seconds=5.0, burn_in_seconds=0.0, decision_interval=1.0),
            flow_params=FlowParams(),
            seed=0,
            agent=ScriptedAgent(quotes),
        )

    def test_requoting_the_same_price_keeps_the_order_and_its_place_in_line(self) -> None:
        sim = self._sim([Quote(bid_price=9_999, ask_price=None, size=1)])
        sim.flow.seed_book(sim.book)
        sim._apply_quote(0, Quote(bid_price=9_999, ask_price=None, size=1))
        first_id = sim._agent_order(Side.BUY)
        assert first_id is not None
        order = sim.book.get_order(first_id)
        assert order is not None
        ahead_before = order.volume_ahead

        # More volume joins behind us, then we ask for the same price again.
        sim.book.add_limit(1, Side.BUY, 9_999, 20)
        sim._apply_quote(2, Quote(bid_price=9_999, ask_price=None, size=1))

        assert sim._agent_order(Side.BUY) == first_id, "the resting order must be reused"
        assert order.volume_ahead == ahead_before

    def test_changing_price_sends_the_order_to_the_back_of_the_new_queue(self) -> None:
        sim = self._sim([Quote.flat()])
        sim.flow.seed_book(sim.book)
        sim._apply_quote(0, Quote(bid_price=9_999, ask_price=None, size=1))
        first_id = sim._agent_order(Side.BUY)
        sim.book.add_limit(1, Side.BUY, 9_998, 30)
        sim._apply_quote(2, Quote(bid_price=9_998, ask_price=None, size=1))

        second_id = sim._agent_order(Side.BUY)
        assert second_id is not None
        assert second_id != first_id
        assert sim.book.get_order(first_id) is None  # the old order was cancelled
        order = sim.book.get_order(second_id)
        assert order is not None
        assert order.volume_ahead >= 30  # joined behind everything already resting

    def test_quoting_none_cancels_the_resting_order(self) -> None:
        sim = self._sim([Quote.flat()])
        sim.flow.seed_book(sim.book)
        sim._apply_quote(0, Quote(bid_price=9_999, ask_price=None, size=1))
        assert sim._agent_order(Side.BUY) is not None
        sim._apply_quote(1, Quote.flat())
        assert sim._agent_order(Side.BUY) is None


class TestPostOnly:
    def test_a_crossing_quote_is_rejected_and_counted(self) -> None:
        sim = Simulation(SimConfig(), FlowParams(), seed=0, agent=ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        best_ask = sim.book.best_ask
        assert best_ask is not None
        sim._apply_quote(0, Quote(bid_price=best_ask, ask_price=None, size=1))
        assert sim._post_only_rejections == 1
        assert sim._agent_order(Side.BUY) is None
        assert sim.state.inventory == 0  # emphatically not a crossed trade

    def test_a_resting_quote_is_accepted(self) -> None:
        sim = Simulation(SimConfig(), FlowParams(), seed=0, agent=ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        best_bid = sim.book.best_bid
        assert best_bid is not None
        sim._apply_quote(0, Quote(bid_price=best_bid, ask_price=None, size=1))
        assert sim._post_only_rejections == 0
        assert sim._agent_order(Side.BUY) is not None


class TestRiskLimits:
    def test_inventory_cap_blocks_further_buying(self) -> None:
        config = SimConfig(max_inventory=3)
        sim = Simulation(config, FlowParams(), seed=0, agent=ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        sim.state.inventory = 3
        best_bid, best_ask = sim.book.best_bid, sim.book.best_ask
        assert best_bid is not None and best_ask is not None
        sim._apply_quote(0, Quote(bid_price=best_bid, ask_price=best_ask, size=1))
        assert sim._agent_order(Side.BUY) is None, "long at the cap: must not bid"
        assert sim._agent_order(Side.SELL) is not None, "long at the cap: must still offer"

    def test_inventory_cap_blocks_further_selling(self) -> None:
        config = SimConfig(max_inventory=3)
        sim = Simulation(config, FlowParams(), seed=0, agent=ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        sim.state.inventory = -3
        best_bid, best_ask = sim.book.best_bid, sim.book.best_ask
        assert best_bid is not None and best_ask is not None
        sim._apply_quote(0, Quote(bid_price=best_bid, ask_price=best_ask, size=1))
        assert sim._agent_order(Side.SELL) is None
        assert sim._agent_order(Side.BUY) is not None

    def test_inventory_stays_inside_the_cap_over_a_full_episode(self) -> None:
        config = SimConfig(horizon_seconds=120.0, burn_in_seconds=10.0, max_inventory=5)
        result = run_episode(11, TouchAgent(size=3), config)
        # A fill can carry inventory past the cap (the resting order was placed before the cap was
        # reached), but never by more than one order's worth.
        assert np.abs(result.inventory).max() <= config.max_inventory + 3


class TestCloseOut:
    def test_inventory_is_flattened_at_the_close(self) -> None:
        config = SimConfig(horizon_seconds=120.0, burn_in_seconds=10.0)
        sim = Simulation(config, FlowParams(), 4, TouchAgent(size=2))
        result = sim.run()
        assert result.n_fills > 0, "the test is vacuous if the agent never trades"
        assert sim.state.inventory == 0
        assert sim.book.agent_orders() == [], "resting quotes must be pulled before liquidating"

    def test_liquidating_a_position_costs_the_spread(self) -> None:
        """Closing out is not free: the flattening trade pays up, and the cost is reported."""
        config = SimConfig(horizon_seconds=60.0, burn_in_seconds=10.0)
        sim = Simulation(config, FlowParams(), 4, ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        sim.state.inventory = 20  # a long that must be sold into the book
        cost = sim._close_out(60.0)
        assert cost > 0.0
        assert sim.state.inventory == 0
        assert sim.state.cash > 0.0  # sold the position, so cash came in

    def test_disabling_close_out_leaves_inventory_on_the_books(self) -> None:
        config = SimConfig(horizon_seconds=120.0, burn_in_seconds=10.0, liquidate_at_close=False)
        sim = Simulation(config, FlowParams(), 4, TouchAgent(size=2))
        result = sim.run()
        assert result.liquidation_cost == 0.0
        # Whatever position remains is marked, not realised.
        assert result.pnl == pytest.approx(sim.state.mark_to_market(sim.book.mid))

    def test_close_out_of_a_flat_book_is_free(self) -> None:
        config = SimConfig(horizon_seconds=60.0, burn_in_seconds=10.0)
        result = run_episode(4, ScriptedAgent([Quote.flat()]), config)
        assert result.liquidation_cost == 0.0


class TestFillAccounting:
    def test_a_buy_fill_increases_inventory_and_reduces_cash(self) -> None:
        sim = Simulation(SimConfig(), FlowParams(), seed=0, agent=ScriptedAgent([Quote.flat()]))
        sim.flow.seed_book(sim.book)
        best_bid = sim.book.best_bid
        assert best_bid is not None
        sim._apply_quote(0, Quote(bid_price=best_bid, ask_price=None, size=5))
        # Sweep the whole bid level so our order is guaranteed to trade.
        volume = sim.book.volume_at(Side.BUY, best_bid)
        sim._submit_market(1, Side.SELL, volume, is_agent=False)
        assert sim.state.inventory == 5
        assert sim.state.cash == -5 * best_bid
        assert all(f.is_maker for f in sim.state.fills)
        assert all(f.side is Side.BUY for f in sim.state.fills)

    def test_the_agent_is_told_about_every_fill(self) -> None:
        agent = TouchAgent(size=2)
        result = run_episode(4, agent, SimConfig(horizon_seconds=120.0, burn_in_seconds=10.0))
        assert len(agent.fills) == result.n_fills > 0

    def test_pnl_equals_cash_once_flat(self) -> None:
        config = SimConfig(horizon_seconds=120.0, burn_in_seconds=10.0)
        sim = Simulation(config, FlowParams(), 4, TouchAgent(size=2))
        result = sim.run()
        assert sim.state.inventory == 0
        assert result.pnl == pytest.approx(sim.state.cash)


class TestFillModelWiring:
    def test_the_fill_model_reaches_the_book_and_is_reported(self) -> None:
        result = run_episode(5, TouchAgent(), SHORT, fill_model=FillModel.OPTIMISTIC)
        assert result.fill_model == "optimistic"

    def test_optimistic_fills_are_at_least_as_numerous(self) -> None:
        config = SimConfig(horizon_seconds=180.0, burn_in_seconds=10.0)
        queue_aware = run_episode(9, TouchAgent(size=2), config, fill_model=FillModel.QUEUE_AWARE)
        optimistic = run_episode(9, TouchAgent(size=2), config, fill_model=FillModel.OPTIMISTIC)
        assert optimistic.filled_volume > queue_aware.filled_volume
