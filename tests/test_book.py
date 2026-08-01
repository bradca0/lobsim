"""Matching-engine tests: priority, crossing, cancellation, and queue bookkeeping."""

from __future__ import annotations

import pytest

from lobsim.book import LimitOrderBook
from lobsim.types import FillModel, Side


def book_with_spread() -> LimitOrderBook:
    """Bids 98/99, asks 101/102, 10 lots each."""
    lob = LimitOrderBook()
    for px in (98, 99):
        lob.add_limit(0, Side.BUY, px, 10)
    for px in (101, 102):
        lob.add_limit(0, Side.SELL, px, 10)
    return lob


def test_empty_book_has_no_touch() -> None:
    lob = LimitOrderBook()
    assert lob.best_bid is None
    assert lob.best_ask is None
    assert lob.mid is None
    assert lob.spread is None
    assert lob.volume_at(Side.BUY, 100) == 0
    assert lob.orders_at(Side.SELL, 100) == []


def test_resting_orders_set_the_touch() -> None:
    lob = book_with_spread()
    assert lob.best_bid == 99
    assert lob.best_ask == 101
    assert lob.mid == 100.0
    assert lob.spread == 2
    assert lob.prices(Side.BUY) == [99, 98]
    assert lob.prices(Side.SELL) == [101, 102]


def test_non_marketable_limit_rests_and_does_not_trade() -> None:
    lob = book_with_spread()
    _, trades = lob.add_limit(1, Side.BUY, 100, 5)
    assert trades == []
    assert lob.best_bid == 100
    assert lob.volume_at(Side.BUY, 100) == 5


def test_market_order_walks_the_book_in_price_order() -> None:
    lob = book_with_spread()
    _, trades = lob.add_market(1, Side.BUY, 15)
    assert [(t.price, t.size) for t in trades] == [(101, 10), (102, 5)]
    assert lob.best_ask == 102
    assert lob.volume_at(Side.SELL, 102) == 5


def test_market_order_larger_than_book_discards_remainder() -> None:
    lob = book_with_spread()
    _, trades = lob.add_market(1, Side.BUY, 100)
    assert sum(t.size for t in trades) == 20
    assert lob.best_ask is None
    # The unfilled remainder must not rest: a market order never becomes a resting order.
    assert lob.prices(Side.BUY) == [99, 98]


def test_marketable_limit_stops_at_its_limit_price() -> None:
    lob = book_with_spread()
    _, trades = lob.add_limit(1, Side.BUY, 101, 15)
    assert [(t.price, t.size) for t in trades] == [(101, 10)]
    # The 5 unexecuted lots rest at the limit price, which is now the best bid.
    assert lob.best_bid == 101
    assert lob.volume_at(Side.BUY, 101) == 5


def test_limit_at_the_touch_is_marketable() -> None:
    lob = book_with_spread()
    _, trades = lob.add_limit(1, Side.SELL, 99, 4)
    assert [(t.price, t.size) for t in trades] == [(99, 4)]
    assert lob.volume_at(Side.BUY, 99) == 6


def test_time_priority_within_a_level() -> None:
    lob = LimitOrderBook()
    first, _ = lob.add_limit(0, Side.SELL, 100, 5)
    second, _ = lob.add_limit(1, Side.SELL, 100, 5)
    _, trades = lob.add_market(2, Side.BUY, 6)
    assert trades[0].maker_order_id == first
    assert trades[0].size == 5
    assert trades[1].maker_order_id == second
    assert trades[1].size == 1
    assert lob.get_order(first) is None  # fully filled orders are removed
    assert lob.get_order(second) is not None


def test_cancel_removes_volume_and_reports_success() -> None:
    lob = LimitOrderBook()
    oid, _ = lob.add_limit(0, Side.BUY, 100, 7)
    assert lob.cancel(oid) is True
    assert lob.volume_at(Side.BUY, 100) == 0
    assert lob.best_bid is None
    assert lob.cancel(oid) is False
    assert lob.cancel(9999) is False


def test_cancelling_the_touch_exposes_the_next_level() -> None:
    lob = book_with_spread()
    (oid,) = [o.order_id for o in lob.orders_at(Side.BUY, 99)]
    lob.cancel(oid)
    assert lob.best_bid == 98


def test_price_level_can_be_recreated_after_being_emptied() -> None:
    """Regression: the lazily-cleaned best-price heaps must survive level churn."""
    lob = LimitOrderBook()
    for _ in range(5):
        oid, _ = lob.add_limit(0, Side.BUY, 100, 3)
        assert lob.best_bid == 100
        lob.cancel(oid)
        assert lob.best_bid is None
    lob.add_limit(1, Side.BUY, 100, 3)
    assert lob.best_bid == 100
    assert lob.volume_at(Side.BUY, 100) == 3


def test_rejects_non_positive_size() -> None:
    lob = LimitOrderBook()
    with pytest.raises(ValueError, match="size must be positive"):
        lob.add_limit(0, Side.BUY, 100, 0)
    with pytest.raises(ValueError, match="size must be positive"):
        lob.add_market(0, Side.BUY, -1)


def test_trade_records_carry_aggressor_and_agent_flags() -> None:
    lob = LimitOrderBook()
    lob.add_limit(0, Side.SELL, 100, 5, is_agent=True)
    _, trades = lob.add_market(1, Side.BUY, 5)
    (trade,) = trades
    assert trade.aggressor is Side.BUY
    assert trade.maker_is_agent is True
    assert trade.taker_is_agent is False
    assert trade.price == 100
    assert lob.last_trade_price == 100


class TestQueuePosition:
    def test_agent_joins_the_back_of_the_queue(self) -> None:
        lob = LimitOrderBook()
        lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        order = lob.get_order(oid)
        assert order is not None
        assert order.volume_ahead == 12

    def test_trades_ahead_advance_the_agent(self) -> None:
        lob = LimitOrderBook()
        lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        lob.add_market(2, Side.SELL, 4)
        order = lob.get_order(oid)
        assert order is not None
        assert order.volume_ahead == 8
        assert order.size == 5  # still untouched

    def test_cancellations_ahead_advance_the_agent(self) -> None:
        lob = LimitOrderBook()
        ahead, _ = lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        lob.cancel(ahead)
        order = lob.get_order(oid)
        assert order is not None
        assert order.volume_ahead == 0

    def test_cancellations_behind_do_not_advance_the_agent(self) -> None:
        lob = LimitOrderBook()
        lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        behind, _ = lob.add_limit(2, Side.BUY, 100, 9)
        lob.cancel(behind)
        order = lob.get_order(oid)
        assert order is not None
        assert order.volume_ahead == 12

    def test_agent_fills_only_after_the_queue_ahead_is_exhausted(self) -> None:
        lob = LimitOrderBook()
        lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        _, trades = lob.add_market(2, Side.SELL, 12)
        assert all(not t.maker_is_agent for t in trades)
        _, trades = lob.add_market(3, Side.SELL, 3)
        assert all(t.maker_is_agent for t in trades)
        order = lob.get_order(oid)
        assert order is not None
        assert order.size == 2

    def test_snapshot_reports_agent_queue_position(self) -> None:
        lob = LimitOrderBook()
        lob.add_limit(0, Side.BUY, 100, 12)
        lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        snap = lob.snapshot(2)
        assert snap.agent_volume_ahead == (12, None)


class TestOptimisticFillModel:
    def test_agent_is_moved_to_the_front_of_the_queue(self) -> None:
        lob = LimitOrderBook(fill_model=FillModel.OPTIMISTIC)
        lob.add_limit(0, Side.BUY, 100, 12)
        oid, _ = lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        order = lob.get_order(oid)
        assert order is not None
        assert order.volume_ahead == 0

    def test_agent_fills_on_the_first_trade_at_its_price(self) -> None:
        """The naive backtest assumption: any print at your price is your fill."""
        lob = LimitOrderBook(fill_model=FillModel.OPTIMISTIC)
        lob.add_limit(0, Side.BUY, 100, 12)
        lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
        _, trades = lob.add_market(2, Side.SELL, 3)
        assert all(t.maker_is_agent for t in trades)

    def test_queue_aware_and_optimistic_differ_on_the_same_flow(self) -> None:
        def agent_filled(model: FillModel) -> int:
            lob = LimitOrderBook(fill_model=model)
            lob.add_limit(0, Side.BUY, 100, 20)
            lob.add_limit(1, Side.BUY, 100, 5, is_agent=True)
            _, trades = lob.add_market(2, Side.SELL, 6)
            return sum(t.size for t in trades if t.maker_is_agent)

        assert agent_filled(FillModel.OPTIMISTIC) == 5
        assert agent_filled(FillModel.QUEUE_AWARE) == 0
