"""Price-time-priority limit order book with exact queue-position bookkeeping.

Design notes
------------
*Levels* are ``dict[int, Order]`` keyed by order id. Python dicts preserve insertion order, so the
dict *is* the FIFO queue: the front of the queue is ``next(iter(orders.values()))`` in O(1), and a
cancellation anywhere in the queue is a ``del`` in O(1). A ``deque`` would make mid-queue
cancellation O(n), and cancellations are the most common event type in a real book.

*Best prices* come from lazily-cleaned heaps. A price is pushed once when its level is created and
skipped when popped after the level has been emptied, so the heaps never need deletion.

*Queue position* is maintained incrementally on ``Order.volume_ahead`` for agent orders only. Every
removal at a level -- trade or cancel -- decrements ``volume_ahead`` for agent orders that arrived
later. This is exact, not an approximation, and it is what separates this simulator from a
backtester that assumes a trade at your price fills you.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from lobsim.types import BookSnapshot, FillModel, Order, Side, Trade


@dataclass(slots=True)
class _Level:
    """One price level: an insertion-ordered FIFO of orders plus cached aggregates."""

    price: int
    volume: int = 0
    orders: dict[int, Order] = field(default_factory=dict)
    tracked: list[Order] = field(default_factory=list)

    def add(self, order: Order) -> None:
        self.orders[order.order_id] = order
        self.volume += order.size
        if order.is_agent:
            self.tracked.append(order)

    def remove(self, order: Order) -> None:
        del self.orders[order.order_id]
        self.volume -= order.size
        if order.is_agent:
            self.tracked = [o for o in self.tracked if o.order_id != order.order_id]

    @property
    def is_empty(self) -> bool:
        return not self.orders


class LimitOrderBook:
    """An event-driven matching engine.

    The book is passive: it has no clock of its own and mutates only when the caller submits an
    order or a cancellation with an explicit timestamp. That keeps the simulation deterministic
    and makes the engine trivially replayable.
    """

    __slots__ = (
        "_ask_heap",
        "_ask_in_heap",
        "_ask_levels",
        "_bid_heap",
        "_bid_in_heap",
        "_bid_levels",
        "_next_id",
        "_next_seq",
        "_orders",
        "fill_model",
        "last_trade_price",
    )

    def __init__(self, fill_model: FillModel = FillModel.QUEUE_AWARE) -> None:
        self.fill_model = fill_model
        self._bid_levels: dict[int, _Level] = {}
        self._ask_levels: dict[int, _Level] = {}
        self._bid_heap: list[int] = []  # negated prices -> max-heap
        self._ask_heap: list[int] = []
        self._bid_in_heap: set[int] = set()
        self._ask_in_heap: set[int] = set()
        self._orders: dict[int, Order] = {}
        self._next_id = 1
        self._next_seq = 0
        self.last_trade_price: int | None = None

    # ------------------------------------------------------------------ accessors

    def _levels(self, side: Side) -> dict[int, _Level]:
        return self._bid_levels if side is Side.BUY else self._ask_levels

    @property
    def best_bid(self) -> int | None:
        while self._bid_heap:
            price = -self._bid_heap[0]
            if price in self._bid_levels:
                return price
            heapq.heappop(self._bid_heap)
            self._bid_in_heap.discard(price)
        return None

    @property
    def best_ask(self) -> int | None:
        while self._ask_heap:
            price = self._ask_heap[0]
            if price in self._ask_levels:
                return price
            heapq.heappop(self._ask_heap)
            self._ask_in_heap.discard(price)
        return None

    def best(self, side: Side) -> int | None:
        return self.best_bid if side is Side.BUY else self.best_ask

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2.0

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    def volume_at(self, side: Side, price: int) -> int:
        level = self._levels(side).get(price)
        return 0 if level is None else level.volume

    def orders_at(self, side: Side, price: int) -> list[Order]:
        """Resting orders at a price in time-priority order (front of queue first)."""
        level = self._levels(side).get(price)
        return [] if level is None else list(level.orders.values())

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    @property
    def resting_order_ids(self) -> list[int]:
        return list(self._orders)

    def prices(self, side: Side) -> list[int]:
        """Occupied prices on ``side``, ordered outward from the touch."""
        levels = self._levels(side)
        return sorted(levels, reverse=side is Side.BUY)

    def snapshot(self, ts: int, levels: int = 5) -> BookSnapshot:
        """Top-``levels`` view of both sides plus the agent's queue position at each touch."""
        bids = tuple((p, self._bid_levels[p].volume) for p in self.prices(Side.BUY)[:levels])
        asks = tuple((p, self._ask_levels[p].volume) for p in self.prices(Side.SELL)[:levels])
        return BookSnapshot(
            ts=ts,
            bids=bids,
            asks=asks,
            last_trade_price=self.last_trade_price,
            agent_volume_ahead=(
                self._agent_volume_ahead(Side.BUY),
                self._agent_volume_ahead(Side.SELL),
            ),
        )

    def _agent_volume_ahead(self, side: Side) -> int | None:
        """Queue-ahead of the agent's most aggressive resting order on ``side``."""
        levels = self._levels(side)
        for price in self.prices(side):
            tracked = levels[price].tracked
            if tracked:
                return min(o.volume_ahead for o in tracked)
        return None

    def agent_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_agent]

    # ------------------------------------------------------------------ mutation

    def add_limit(
        self, ts: int, side: Side, price: int, size: int, *, is_agent: bool = False
    ) -> tuple[int, list[Trade]]:
        """Submit a limit order. Crosses the spread if marketable; the remainder rests.

        Returns the new order id and the trades it caused. The order id is valid even if the order
        fully executed and never rested.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        order = self._new_order(ts, side, price, size, is_agent)
        trades = self._match(order, limit_price=price)
        if order.size > 0:
            self._rest(order)
        return order.order_id, trades

    def add_market(
        self, ts: int, side: Side, size: int, *, is_agent: bool = False
    ) -> tuple[int, list[Trade]]:
        """Submit a market order. Any unfilled remainder is discarded (never rests)."""
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        order = self._new_order(ts, side, price=0, size=size, is_agent=is_agent)
        trades = self._match(order, limit_price=None)
        return order.order_id, trades

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. Returns False if it is unknown or already gone."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        level = self._levels(order.side)[order.price]
        self._decrement_queue_ahead(level, ahead_of_seq=order.seq, amount=order.size)
        level.remove(order)
        del self._orders[order_id]
        if level.is_empty:
            del self._levels(order.side)[order.price]
        return True

    # ------------------------------------------------------------------ internals

    def _new_order(self, ts: int, side: Side, price: int, size: int, is_agent: bool) -> Order:
        order = Order(
            order_id=self._next_id,
            seq=self._next_seq,
            side=side,
            price=price,
            size=size,
            ts=ts,
            is_agent=is_agent,
        )
        self._next_id += 1
        self._next_seq += 1
        return order

    def _rest(self, order: Order) -> None:
        levels = self._levels(order.side)
        level = levels.get(order.price)
        if level is None:
            level = _Level(price=order.price)
            levels[order.price] = level
            self._push_price(order.side, order.price)
        # Queue position is the volume already resting at this level. Under the optimistic fill
        # model the agent is treated as arriving at the front, which is exactly the assumption
        # this repo exists to measure.
        if order.is_agent:
            order.volume_ahead = 0 if self.fill_model is FillModel.OPTIMISTIC else level.volume
        level.add(order)
        self._orders[order.order_id] = order

    def _push_price(self, side: Side, price: int) -> None:
        if side is Side.BUY:
            if price not in self._bid_in_heap:
                heapq.heappush(self._bid_heap, -price)
                self._bid_in_heap.add(price)
        elif price not in self._ask_in_heap:
            heapq.heappush(self._ask_heap, price)
            self._ask_in_heap.add(price)

    def _next_maker(self, level: _Level) -> Order:
        """The order at the front of ``level``'s queue under the active fill model."""
        if self.fill_model is FillModel.OPTIMISTIC and level.tracked:
            return level.tracked[0]
        return next(iter(level.orders.values()))

    @staticmethod
    def _decrement_queue_ahead(level: _Level, *, ahead_of_seq: int, amount: int) -> None:
        """Volume left the queue at ``ahead_of_seq``; orders behind it move up by ``amount``."""
        for tracked in level.tracked:
            if tracked.seq > ahead_of_seq:
                tracked.volume_ahead = max(0, tracked.volume_ahead - amount)

    def _match(self, taker: Order, limit_price: int | None) -> list[Trade]:
        """Execute ``taker`` against the opposite book, respecting ``limit_price`` if given."""
        trades: list[Trade] = []
        opposite = taker.side.opposite
        levels = self._levels(opposite)
        while taker.size > 0:
            best = self.best(opposite)
            if best is None:
                break
            # Stop once the best resting price is worse than the taker's limit. Equality is
            # marketable: a buy limit at the ask trades.
            if limit_price is not None and opposite.is_better(limit_price, best):
                break
            level = levels[best]
            self._consume_level(taker, level, trades)
            if level.is_empty:
                del levels[best]
        return trades

    def _consume_level(self, taker: Order, level: _Level, trades: list[Trade]) -> None:
        while taker.size > 0 and not level.is_empty:
            maker = self._next_maker(level)
            qty = min(taker.size, maker.size)
            maker.size -= qty
            taker.size -= qty
            level.volume -= qty
            self._decrement_queue_ahead(level, ahead_of_seq=maker.seq, amount=qty)
            self.last_trade_price = level.price
            trades.append(
                Trade(
                    ts=taker.ts,
                    price=level.price,
                    size=qty,
                    aggressor=taker.side,
                    maker_order_id=maker.order_id,
                    taker_order_id=taker.order_id,
                    maker_is_agent=maker.is_agent,
                    taker_is_agent=taker.is_agent,
                )
            )
            if maker.size == 0:
                level.remove(maker)
                del self._orders[maker.order_id]
