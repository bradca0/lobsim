"""Order-flow generator tests: parameter validation, sampling laws, and structural guarantees."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lobsim.book import LimitOrderBook
from lobsim.flow import (
    CancelPolicy,
    EventKind,
    FlowParams,
    OrderFlowGenerator,
    _Hawkes,
)
from lobsim.types import Side


def make(
    params: FlowParams | None = None, seed: int = 0
) -> tuple[OrderFlowGenerator, LimitOrderBook]:
    rng = np.random.default_rng(seed)
    generator = OrderFlowGenerator(params or FlowParams(), rng)
    book = LimitOrderBook()
    generator.seed_book(book)
    return generator, book


class TestParams:
    def test_rejects_explosive_hawkes(self) -> None:
        with pytest.raises(ValueError, match="branching ratio"):
            FlowParams(mo_self_excite=0.6, mo_cross_excite=0.5, mo_decay=1.0)

    def test_branching_ratio_is_reported(self) -> None:
        params = FlowParams(mo_self_excite=0.2, mo_cross_excite=0.1, mo_decay=1.0)
        assert params.branching_ratio == pytest.approx(0.3)

    def test_rejects_degenerate_depth_and_size(self) -> None:
        with pytest.raises(ValueError, match="lo_max_depth"):
            FlowParams(lo_max_depth=0)
        with pytest.raises(ValueError, match="size_max"):
            FlowParams(size_max=0)


class TestHawkes:
    def test_excitation_raises_intensity_then_decays(self) -> None:
        params = FlowParams()
        base = params.mo_baseline
        hawkes = _Hawkes(params)
        assert hawkes.intensity_at(0.0, Side.BUY, base) == pytest.approx(base)
        hawkes.excite(0.0, Side.BUY)
        excited = hawkes.intensity_at(0.0, Side.BUY, base)
        assert excited == pytest.approx(base + params.mo_self_excite)
        assert hawkes.intensity_at(10.0, Side.BUY, base) < excited
        assert hawkes.intensity_at(1e6, Side.BUY, base) == pytest.approx(base)

    def test_a_buy_also_excites_the_sell_side(self) -> None:
        params = FlowParams()
        hawkes = _Hawkes(params)
        hawkes.excite(0.0, Side.BUY)
        assert hawkes.intensity_at(0.0, Side.SELL, params.mo_baseline) == pytest.approx(
            params.mo_baseline + params.mo_cross_excite
        )

    def test_decay_is_monotone_and_never_goes_backwards_in_time(self) -> None:
        hawkes = _Hawkes(FlowParams())
        hawkes.excite(1.0, Side.BUY)
        before = hawkes.excitation[Side.BUY]
        hawkes.decay_to(0.5)  # earlier than last_t -- must be a no-op
        assert hawkes.excitation[Side.BUY] == before


class TestSizes:
    def test_sizes_are_within_bounds(self) -> None:
        generator, _ = make(FlowParams(size_max=7))
        sizes = [generator.draw_size() for _ in range(2000)]
        assert min(sizes) >= 1
        assert max(sizes) <= 7

    def test_size_distribution_has_a_heavy_right_tail(self) -> None:
        """A Pareto tail must put far more mass on 1 lot than on 10, but not zero on 10."""
        generator, _ = make(FlowParams(size_alpha=1.8, size_max=60))
        sizes = np.array([generator.draw_size() for _ in range(20000)])
        assert (sizes == 1).mean() > 0.4
        assert (sizes >= 10).mean() > 0.005
        assert sizes.mean() < 5.0


class TestBookSeeding:
    def test_seeded_book_is_symmetric_around_the_initial_price(self) -> None:
        params = FlowParams(initial_price=500, initial_depth_levels=3, initial_level_size=4)
        _, book = make(params)
        assert book.best_bid == 499
        assert book.best_ask == 501
        assert book.prices(Side.BUY) == [499, 498, 497]
        assert book.volume_at(Side.SELL, 501) == 4


class TestEventGeneration:
    def test_events_are_reproducible_from_a_seed(self) -> None:
        def run() -> list[tuple[int, str, int]]:
            generator, book = make(seed=42)
            out = []
            t = 0.0
            for _ in range(300):
                t, event = generator.next_event(book, t)
                if event is not None:
                    out.append((event.ts, event.kind.value, event.size))
            return out

        assert run() == run()

    def test_limit_events_are_never_marketable(self) -> None:
        """Aggression is expressed only through market orders; limit flow must never cross."""
        generator, book = make(seed=7)
        t = 0.0
        checked = 0
        for _ in range(4000):
            t, event = generator.next_event(book, t)
            if event is None:
                continue
            if event.kind is EventKind.LIMIT:
                opposite = book.best(event.side.opposite)
                if opposite is not None:
                    assert event.side.is_better(opposite, event.price), (
                        f"limit {event.side} @ {event.price} crosses touch {opposite}"
                    )
                    checked += 1
                book.add_limit(event.ts, event.side, event.price, event.size)
            elif event.kind is EventKind.MARKET:
                book.add_market(event.ts, event.side, event.size)
            else:
                book.cancel(event.order_id)
        assert checked > 100

    def test_cancellations_never_target_agent_orders(self) -> None:
        """The background market may not pull the agent's quotes out from under it."""
        generator, book = make(seed=3)
        best_bid, best_ask = book.best_bid, book.best_ask
        assert best_bid is not None and best_ask is not None
        agent_ids = {
            book.add_limit(0, Side.BUY, best_bid, 5, is_agent=True)[0],
            book.add_limit(0, Side.SELL, best_ask, 5, is_agent=True)[0],
        }
        t = 0.0
        cancels = 0
        for _ in range(4000):
            t, event = generator.next_event(book, t)
            if event is None:
                continue
            if event.kind is EventKind.CANCEL:
                assert event.order_id not in agent_ids
                cancels += 1
                book.cancel(event.order_id)
            elif event.kind is EventKind.LIMIT:
                book.add_limit(event.ts, event.side, event.price, event.size)
            else:
                book.add_market(event.ts, event.side, event.size)
        assert cancels > 50
        # Both agent orders must still be resting or filled -- never cancelled by the background.
        assert all(book.get_order(oid) is None or book.get_order(oid).is_agent for oid in agent_ids)  # type: ignore[union-attr]

    def test_timestamps_are_non_decreasing(self) -> None:
        generator, book = make(seed=11)
        t = 0.0
        last_ts = -1
        for _ in range(2000):
            t, event = generator.next_event(book, t)
            if event is not None:
                assert event.ts >= last_ts
                last_ts = event.ts

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_empty_book_still_produces_limit_orders(self, seed: int) -> None:
        """With no resting liquidity the generator must still be able to rebuild the book."""
        rng = np.random.default_rng(seed)
        generator = OrderFlowGenerator(FlowParams(), rng)
        book = LimitOrderBook()  # deliberately not seeded
        t = 0.0
        for _ in range(200):
            t, event = generator.next_event(book, t)
            if event is None:
                continue
            if event.kind is EventKind.LIMIT:
                book.add_limit(event.ts, event.side, event.price, event.size)
            elif event.kind is EventKind.MARKET:
                book.add_market(event.ts, event.side, event.size)
            else:
                book.cancel(event.order_id)
        assert book.best_bid is not None or book.best_ask is not None


class TestCancelPolicy:
    def _cancel_positions(self, policy: CancelPolicy, seed: int = 0) -> list[int]:
        """Queue index of each cancelled order at a single fixed price level."""
        rng = np.random.default_rng(seed)
        generator = OrderFlowGenerator(FlowParams(cancel_policy=policy), rng)
        positions = []
        for _ in range(400):
            book = LimitOrderBook()
            for _ in range(6):
                book.add_limit(0, Side.BUY, 100, 3)
            order_ids = [o.order_id for o in book.orders_at(Side.BUY, 100)]
            target = generator._choose_cancel_target(book, book.total_volume(Side.BUY))
            assert target is not None
            positions.append(order_ids.index(target.order_id))
        return positions

    def test_uniform_policy_is_roughly_flat_across_the_queue(self) -> None:
        positions = np.array(self._cancel_positions(CancelPolicy.UNIFORM))
        # Six equal-sized orders: the mean index should sit near the middle, 2.5.
        assert 2.0 < positions.mean() < 3.0

    def test_back_loaded_policy_concentrates_on_late_arrivals(self) -> None:
        uniform = np.mean(self._cancel_positions(CancelPolicy.UNIFORM))
        back = np.mean(self._cancel_positions(CancelPolicy.BACK_LOADED))
        assert back > uniform + 0.4

    def test_no_cancel_target_in_an_empty_book(self) -> None:
        generator = OrderFlowGenerator(FlowParams(), np.random.default_rng(0))
        assert generator._choose_cancel_target(LimitOrderBook(), 0) is None

    def test_no_cancel_target_when_only_agent_orders_rest(self) -> None:
        generator = OrderFlowGenerator(FlowParams(), np.random.default_rng(0))
        book = LimitOrderBook()
        book.add_limit(0, Side.BUY, 100, 5, is_agent=True)
        assert generator._choose_cancel_target(book, 0) is None


class TestInformedFlow:
    """The tilt of aggressive flow toward the latent fundamental is the source of adverse
    selection, so its sign, symmetry, and bounds all matter."""

    def _baselines(self, gap: float, **kwargs: float) -> tuple[float, float]:
        generator, book = make(FlowParams(**kwargs))  # type: ignore[arg-type]
        mid = book.mid
        assert mid is not None
        generator.fundamental = mid + gap
        return generator._informed_baselines(book)

    def test_no_tilt_when_the_fundamental_sits_at_the_mid(self) -> None:
        buy, sell = self._baselines(0.0)
        assert buy == pytest.approx(sell)
        assert buy == pytest.approx(FlowParams().mo_baseline)

    def test_fundamental_above_mid_favours_buy_aggression(self) -> None:
        buy, sell = self._baselines(2.0)
        assert buy > sell
        assert buy / sell == pytest.approx(math.exp(2 * FlowParams().informed_kappa * 2.0))

    def test_tilt_is_symmetric_in_sign(self) -> None:
        up_buy, up_sell = self._baselines(1.5)
        down_buy, down_sell = self._baselines(-1.5)
        assert up_buy == pytest.approx(down_sell)
        assert up_sell == pytest.approx(down_buy)

    def test_tilt_is_clipped_so_intensities_stay_bounded(self) -> None:
        params = FlowParams()
        buy, _ = self._baselines(1e6)
        assert buy == pytest.approx(params.mo_baseline * math.exp(params.informed_clip))

    def test_the_fundamental_diffuses_over_time(self) -> None:
        generator, book = make(seed=5)
        start = generator.fundamental
        t = 0.0
        for _ in range(2000):
            t, _ = generator.next_event(book, t)
        assert generator.fundamental != start
        # A driftless random walk with vol 0.45 ticks/sqrt(s) should stay within a few standard
        # deviations of its start over the elapsed horizon.
        sd = FlowParams().fundamental_vol * math.sqrt(t)
        assert abs(generator.fundamental - start) < 6 * sd

    def test_a_zero_volatility_fundamental_never_moves(self) -> None:
        generator, book = make(FlowParams(fundamental_vol=0.0), seed=5)
        start = generator.fundamental
        t = 0.0
        for _ in range(500):
            t, _ = generator.next_event(book, t)
        assert generator.fundamental == start

    def test_mid_price_tracks_the_fundamental(self) -> None:
        """The point of informed flow: aggressive orders drag the book toward fair value."""
        params = FlowParams(fundamental_vol=0.0)
        generator, book = make(params, seed=1)
        mid_before = book.mid
        assert mid_before is not None
        generator.fundamental = mid_before + 12.0  # a large, persistent mispricing
        t = 0.0
        for _ in range(6000):
            t, event = generator.next_event(book, t)
            if event is None:
                continue
            if event.kind is EventKind.LIMIT:
                book.add_limit(event.ts, event.side, event.price, event.size)
            elif event.kind is EventKind.MARKET:
                book.add_market(event.ts, event.side, event.size)
            else:
                book.cancel(event.order_id)
        mid_after = book.mid
        assert mid_after is not None
        assert mid_after > mid_before + 4.0
