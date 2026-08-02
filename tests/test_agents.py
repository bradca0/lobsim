"""Baseline policy tests.

Baselines exist to be compared against, so their behaviour has to be exactly what the README claims
it is. The control -- an inactive policy earning exactly zero -- is the single most load-bearing
test in the repo: if it ever fails, the accounting is broken and no other number means anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lobsim.agents import (
    AlwaysAtTouch,
    AvellanedaStoikov,
    FixedSpread,
    Inactive,
    InventorySkew,
)
from lobsim.engine import MarketContext, SimConfig, run_episode
from lobsim.metrics import decompose_pnl
from lobsim.types import BookSnapshot

SHORT = SimConfig(horizon_seconds=90.0, burn_in_seconds=10.0)


def context(
    bid: int = 99,
    ask: int = 101,
    inventory: int = 0,
    step: int = 0,
    steps_total: int = 100,
) -> MarketContext:
    return MarketContext(
        ts=0,
        snapshot=BookSnapshot(ts=0, bids=((bid, 10),), asks=((ask, 10),)),
        inventory=inventory,
        step=step,
        steps_total=steps_total,
    )


def empty_context() -> MarketContext:
    return MarketContext(
        ts=0,
        snapshot=BookSnapshot(ts=0, bids=(), asks=()),
        inventory=0,
        step=0,
        steps_total=10,
    )


def fresh(agent):  # type: ignore[no-untyped-def]
    agent.reset(np.random.default_rng(0))
    return agent


class TestInactiveControl:
    def test_it_never_quotes(self) -> None:
        quote = fresh(Inactive()).act(context())
        assert quote.bid_price is None
        assert quote.ask_price is None

    def test_its_pnl_is_exactly_zero(self) -> None:
        """The control. A non-zero value here means the PnL accounting is broken."""
        for seed in range(6):
            result = run_episode(seed, Inactive(), SHORT)
            assert result.pnl == 0.0
            assert result.n_fills == 0
            assert result.filled_volume == 0
            decomposition = decompose_pnl(result)
            assert decomposition.spread_capture == 0.0
            assert decomposition.inventory_pnl == 0.0


class TestAlwaysAtTouch:
    def test_it_joins_both_touches(self) -> None:
        quote = fresh(AlwaysAtTouch(size=3)).act(context(bid=500, ask=503))
        assert (quote.bid_price, quote.ask_price, quote.size) == (500, 503, 3)

    def test_it_quotes_nothing_on_an_empty_book(self) -> None:
        quote = fresh(AlwaysAtTouch()).act(empty_context())
        assert quote.bid_price is None and quote.ask_price is None

    def test_it_actually_trades(self) -> None:
        result = run_episode(3, AlwaysAtTouch(size=2), SHORT)
        assert result.n_fills > 0


class TestFixedSpread:
    def test_it_quotes_symmetrically_about_the_mid(self) -> None:
        quote = fresh(FixedSpread(half_spread=3)).act(context(bid=98, ask=102))  # mid 100
        assert quote.bid_price == 97
        assert quote.ask_price == 103

    def test_it_rounds_outward_on_a_half_tick_mid(self) -> None:
        """Rounding to nearest would make the quote marketable and get it post-only rejected."""
        quote = fresh(FixedSpread(half_spread=1)).act(context(bid=100, ask=101))  # mid 100.5
        assert quote.bid_price == math.floor(100.5 - 1)
        assert quote.ask_price == math.ceil(100.5 + 1)
        assert quote.bid_price < 100.5 < quote.ask_price

    def test_wider_spreads_trade_less(self) -> None:
        tight = run_episode(4, FixedSpread(size=2, half_spread=1), SHORT)
        wide = run_episode(4, FixedSpread(size=2, half_spread=4), SHORT)
        assert tight.filled_volume > wide.filled_volume

    def test_it_names_itself_by_its_width(self) -> None:
        assert FixedSpread(half_spread=3).name == "fixed_spread_3"

    def test_a_degenerate_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1 tick"):
            FixedSpread(half_spread=0)

    def test_it_stands_down_when_there_is_no_mid(self) -> None:
        assert fresh(FixedSpread()).act(empty_context()).bid_price is None


class TestInventorySkew:
    def test_flat_inventory_quotes_both_sides(self) -> None:
        quote = fresh(InventorySkew(threshold=5)).act(context(inventory=0))
        assert quote.bid_price is not None and quote.ask_price is not None

    def test_being_long_pulls_the_bid(self) -> None:
        quote = fresh(InventorySkew(threshold=5)).act(context(inventory=5))
        assert quote.bid_price is None
        assert quote.ask_price is not None

    def test_being_short_pulls_the_offer(self) -> None:
        quote = fresh(InventorySkew(threshold=5)).act(context(inventory=-5))
        assert quote.ask_price is None
        assert quote.bid_price is not None

    def test_it_holds_less_inventory_than_the_unmanaged_baseline(self) -> None:
        """The property that justifies its existence as the primary baseline."""
        managed = run_episode(7, InventorySkew(size=2, threshold=5), SHORT)
        unmanaged = run_episode(7, AlwaysAtTouch(size=2), SHORT)
        assert np.abs(managed.inventory).max() < np.abs(unmanaged.inventory).max()


class TestAvellanedaStoikov:
    def test_a_long_position_shifts_both_quotes_down(self) -> None:
        """The reservation price moves away from the side that would add to inventory."""
        flat = fresh(AvellanedaStoikov(gamma=0.5, kappa=1.5))
        long = fresh(AvellanedaStoikov(gamma=0.5, kappa=1.5))
        # Feed a volatile path so the variance estimate is non-zero and the skew term bites.
        for step, price in enumerate([100, 104, 98, 106, 96, 105] * 4):
            flat_quote = flat.act(context(bid=price - 1, ask=price + 1, inventory=0, step=step))
            long_quote = long.act(context(bid=price - 1, ask=price + 1, inventory=30, step=step))
        assert long_quote.bid_price is not None and flat_quote.bid_price is not None
        assert long_quote.ask_price is not None and flat_quote.ask_price is not None
        assert long_quote.bid_price < flat_quote.bid_price
        assert long_quote.ask_price < flat_quote.ask_price

    def test_its_quotes_straddle_the_reservation_price(self) -> None:
        agent = fresh(AvellanedaStoikov())
        quote = agent.act(context(bid=99, ask=101))
        assert quote.bid_price is not None and quote.ask_price is not None
        assert quote.bid_price < quote.ask_price

    def test_volatility_estimate_resets_between_episodes(self) -> None:
        agent = AvellanedaStoikov()
        agent.reset(np.random.default_rng(0))
        for step, price in enumerate([100, 130, 90, 140] * 5):
            agent.act(context(bid=price - 1, ask=price + 1, step=step))
        assert agent._variance > 0.0
        agent.reset(np.random.default_rng(0))
        assert agent._variance == 0.0
        assert agent._last_mid is None

    def test_degenerate_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="gamma must be positive"):
            AvellanedaStoikov(gamma=0.0)
        with pytest.raises(ValueError, match="kappa must be positive"):
            AvellanedaStoikov(kappa=-1.0)

    def test_it_stands_down_when_there_is_no_mid(self) -> None:
        assert fresh(AvellanedaStoikov()).act(empty_context()).bid_price is None


class TestFillBookkeeping:
    def test_every_baseline_records_the_fills_it_is_told_about(self) -> None:
        agent = AlwaysAtTouch(size=2)
        result = run_episode(5, agent, SHORT)
        assert len(agent._fills) == result.n_fills

    def test_agents_are_reset_at_the_start_of_each_episode(self) -> None:
        agent = AlwaysAtTouch(size=2)
        run_episode(5, agent, SHORT)
        first = len(agent._fills)
        run_episode(5, agent, SHORT)
        assert len(agent._fills) == first, "fills must not accumulate across episodes"
