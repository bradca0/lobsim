"""Core value objects for the simulator.

All prices are integer *tick indices* and all sizes are integer *lots*; see docs/DECISIONS.md D1
for why the engine never touches a float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

NS_PER_SECOND: Final[int] = 1_000_000_000


class Side(Enum):
    """Order side. ``sign`` maps the side onto the direction of the resulting position."""

    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        return int(self.value)

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    def is_better(self, price: int, than: int) -> bool:
        """True if ``price`` has strictly higher priority than ``than`` for this side."""
        return price > than if self is Side.BUY else price < than


class FillModel(Enum):
    """How the agent's resting orders are prioritised inside a price level.

    ``QUEUE_AWARE`` gives the agent honest FIFO priority: it is filled only after every order
    that arrived at that level before it. ``OPTIMISTIC`` reproduces the standard naive backtest
    assumption -- the agent fills whenever a trade prints at its price -- by moving its orders to
    the front of the level regardless of arrival time. The gap between the two is the headline
    ablation of this repo.
    """

    QUEUE_AWARE = "queue_aware"
    OPTIMISTIC = "optimistic"


@dataclass(slots=True)
class Order:
    """A resting or in-flight order.

    ``seq`` is a globally monotonic arrival counter and *is* the time-priority key: two orders at
    the same price are ranked by ``seq``, never by wall-clock timestamp (which can tie).

    ``volume_ahead`` is maintained only for agent orders (``is_agent``) and is the exact number of
    lots that must trade or cancel at this price before this order is next in line.
    """

    order_id: int
    seq: int
    side: Side
    price: int
    size: int
    ts: int
    is_agent: bool = False
    volume_ahead: int = 0

    @property
    def is_filled(self) -> bool:
        return self.size == 0


@dataclass(frozen=True, slots=True)
class Trade:
    """An execution. ``price`` is always the resting (maker) order's price."""

    ts: int
    price: int
    size: int
    aggressor: Side
    maker_order_id: int
    taker_order_id: int
    maker_is_agent: bool
    taker_is_agent: bool


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """An immutable view of the top of book, cheap enough to record every event.

    ``bids``/``asks`` are ``(price, volume)`` pairs ordered outward from the touch. Either may be
    empty if that side of the book is exhausted.
    """

    ts: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    last_trade_price: int | None = None
    agent_volume_ahead: tuple[int | None, int | None] = (None, None)

    @property
    def best_bid(self) -> int | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> int | None:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> int | None:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None

    @property
    def mid(self) -> float | None:
        """Arithmetic mid in ticks. ``None`` when either side is empty."""
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2.0
        return None

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid: the touch price weighted by the *opposite* side's volume.

        Leans toward the side with less resting size, which is the side likelier to be taken out.
        """
        if not self.bids or not self.asks:
            return None
        bid_px, bid_sz = self.bids[0]
        ask_px, ask_sz = self.asks[0]
        total = bid_sz + ask_sz
        if total == 0:
            return (bid_px + ask_px) / 2.0
        return (bid_px * ask_sz + ask_px * bid_sz) / total

    @property
    def imbalance(self) -> float:
        """Order-book imbalance at the touch in [-1, 1]; +1 means all size is on the bid."""
        bid_sz = self.bids[0][1] if self.bids else 0
        ask_sz = self.asks[0][1] if self.asks else 0
        total = bid_sz + ask_sz
        if total == 0:
            return 0.0
        return (bid_sz - ask_sz) / total

    def depth(self, side: Side, levels: int) -> int:
        """Total resting volume across the top ``levels`` price levels of ``side``."""
        book = self.bids if side is Side.BUY else self.asks
        return sum(v for _, v in book[:levels])


@dataclass(slots=True)
class Fill:
    """An agent execution, from the agent's own point of view."""

    ts: int
    side: Side
    price: int
    size: int
    is_maker: bool
    mid_at_fill: float | None


@dataclass(slots=True)
class AgentState:
    """Running position and cash of the market-making agent, in ticks and lots."""

    inventory: int = 0
    cash: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def apply(self, fill: Fill) -> None:
        self.inventory += fill.side.sign * fill.size
        self.cash -= fill.side.sign * fill.size * fill.price
        self.fills.append(fill)

    def mark_to_market(self, mid: float | None) -> float:
        """Cash plus inventory marked at ``mid``. Inventory is marked at zero if the book is empty."""
        if mid is None:
            return self.cash
        return self.cash + self.inventory * mid
