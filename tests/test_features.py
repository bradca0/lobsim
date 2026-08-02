"""Feature-extraction tests, with emphasis on the properties that keep results honest.

The load-bearing one is :class:`TestNoLookahead`: a feature that peeks at the latent fundamental or
at a future price would make every downstream number meaningless, and no amount of statistical
machinery would catch it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from lobsim.engine import MarketContext
from lobsim.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    N_FEATURES,
    FeatureConfig,
    FeatureExtractor,
    group_mask,
)
from lobsim.types import BookSnapshot


def snapshot(
    bids: tuple[tuple[int, int], ...] = ((99, 10), (98, 20), (97, 30)),
    asks: tuple[tuple[int, int], ...] = ((101, 10), (102, 20), (103, 30)),
    ahead: tuple[int | None, int | None] = (None, None),
) -> BookSnapshot:
    return BookSnapshot(ts=0, bids=bids, asks=asks, agent_volume_ahead=ahead)


def context(
    snap: BookSnapshot | None = None,
    inventory: int = 0,
    step: int = 0,
    steps_total: int = 100,
    trade_flow: int = 0,
    traded_volume: int = 0,
) -> MarketContext:
    return MarketContext(
        ts=step * 1_000_000,
        snapshot=snap if snap is not None else snapshot(),
        inventory=inventory,
        step=step,
        steps_total=steps_total,
        trade_flow=trade_flow,
        traded_volume=traded_volume,
    )


class TestGroups:
    def test_every_named_feature_belongs_to_exactly_one_group(self) -> None:
        flat = [name for group in FEATURE_GROUPS.values() for name in group]
        assert len(flat) == len(set(flat)) == N_FEATURES
        assert set(flat) == set(FEATURE_NAMES)

    def test_mask_selects_only_the_requested_groups(self) -> None:
        mask = group_mask(("queue",))
        selected = {name for name, keep in zip(FEATURE_NAMES, mask, strict=True) if keep}
        assert selected == set(FEATURE_GROUPS["queue"])

    def test_masks_compose_additively(self) -> None:
        both = group_mask(("book", "queue"))
        assert both.sum() == len(FEATURE_GROUPS["book"]) + len(FEATURE_GROUPS["queue"])

    def test_all_groups_selects_everything(self) -> None:
        assert group_mask(tuple(FEATURE_GROUPS)).all()

    def test_unknown_group_is_rejected_loudly(self) -> None:
        with pytest.raises(ValueError, match="unknown feature group"):
            group_mask(("book", "not_a_group"))


class TestShapeAndFiniteness:
    def test_vector_has_the_declared_width(self) -> None:
        extractor = FeatureExtractor()
        assert extractor.extract(context()).shape == (N_FEATURES,)

    def test_features_are_always_finite(self) -> None:
        """NaN or inf here would silently poison the regression targets downstream."""
        extractor = FeatureExtractor()
        awkward = [
            context(),
            context(snapshot(bids=(), asks=((101, 5),))),  # one-sided book
            context(snapshot(bids=((99, 0),), asks=((101, 0),))),  # zero depth
            context(snapshot(bids=(), asks=())),  # empty book
            context(inventory=-500, step=99, steps_total=100),
        ]
        for ctx in awkward:
            values = extractor.extract(ctx)
            assert np.all(np.isfinite(values)), f"non-finite feature for {ctx.snapshot}"


class TestSemantics:
    def test_imbalance_is_signed_toward_the_heavier_side(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("imbalance_l1")
        bid_heavy = extractor.extract(context(snapshot(bids=((99, 30),), asks=((101, 10),))))
        extractor.reset()
        ask_heavy = extractor.extract(context(snapshot(bids=((99, 10),), asks=((101, 30),))))
        assert bid_heavy[index] > 0 > ask_heavy[index]

    def test_microprice_gap_leans_away_from_the_heavier_side(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("microprice_gap")
        values = extractor.extract(context(snapshot(bids=((99, 30),), asks=((101, 10),))))
        # Heavy bid, light ask: the ask is likelier to be taken out, so fair value sits above mid.
        assert values[index] > 0

    def test_spread_is_reported_in_ticks(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("spread")
        values = extractor.extract(context(snapshot(bids=((95, 5),), asks=((101, 5),))))
        assert values[index] == 6.0

    def test_queue_fraction_is_zero_at_the_front_and_one_at_the_back(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("queue_ahead_bid")
        front = extractor.extract(context(snapshot(bids=((99, 20),), ahead=(0, None))))
        extractor.reset()
        buried = extractor.extract(context(snapshot(bids=((99, 20),), ahead=(20, None))))
        assert front[index] == 0.0
        assert buried[index] == 1.0

    def test_no_resting_order_reads_as_the_worst_queue_position(self) -> None:
        """Encoding "no order" as 1.0 keeps the feature monotone in badness."""
        extractor = FeatureExtractor()
        values = extractor.extract(context(snapshot(ahead=(None, None))))
        assert values[FEATURE_NAMES.index("queue_ahead_bid")] == 1.0
        assert values[FEATURE_NAMES.index("has_bid")] == 0.0

    def test_inventory_is_normalised_by_the_risk_limit(self) -> None:
        extractor = FeatureExtractor(FeatureConfig(max_inventory=50))
        values = extractor.extract(context(inventory=25))
        assert values[FEATURE_NAMES.index("inventory")] == pytest.approx(0.5)

    def test_time_pressure_grows_as_the_episode_ends(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("inventory_time_pressure")
        early = extractor.extract(context(inventory=25, step=0, steps_total=100))
        extractor.reset()
        late = extractor.extract(context(inventory=25, step=99, steps_total=100))
        assert abs(late[index]) > abs(early[index])

    def test_momentum_tracks_the_mid_over_its_window(self) -> None:
        extractor = FeatureExtractor(FeatureConfig(momentum_short=2))
        index = FEATURE_NAMES.index("momentum_short")
        for price in (100, 101, 102, 103):
            values = extractor.extract(
                context(snapshot(bids=((price - 1, 10),), asks=((price + 1, 10),)))
            )
        assert values[index] == pytest.approx(2.0)

    def test_volatility_rises_with_a_choppier_mid(self) -> None:
        index = FEATURE_NAMES.index("volatility")

        def run(step_size: int) -> float:
            extractor = FeatureExtractor()
            values = np.zeros(N_FEATURES)
            for i in range(60):
                price = 100 + step_size * (i % 2)
                values = extractor.extract(
                    context(snapshot(bids=((price - 1, 10),), asks=((price + 1, 10),)))
                )
            return float(values[index])

        assert run(4) > run(1) > 0.0

    def test_trade_flow_is_smoothed_and_signed(self) -> None:
        extractor = FeatureExtractor()
        index = FEATURE_NAMES.index("trade_flow_ewma")
        for _ in range(30):
            buys = extractor.extract(context(trade_flow=10, traded_volume=10))
        extractor.reset()
        for _ in range(30):
            sells = extractor.extract(context(trade_flow=-10, traded_volume=10))
        assert buys[index] > 0 > sells[index]


class TestStatefulness:
    def test_reset_clears_all_path_dependent_state(self) -> None:
        """Leaking state between episodes would be a subtle form of lookahead."""
        extractor = FeatureExtractor()
        for price in range(100, 140):
            extractor.extract(context(snapshot(bids=((price - 1, 9),), asks=((price + 1, 9),))))
        dirty = extractor.extract(context())
        extractor.reset()
        clean = extractor.extract(context())
        np.testing.assert_array_equal(clean, FeatureExtractor().extract(context()))
        assert not np.array_equal(dirty, clean)

    def test_a_fresh_extractor_matches_a_reset_one(self) -> None:
        used = FeatureExtractor()
        for _ in range(10):
            used.extract(context(trade_flow=5))
        used.reset()
        np.testing.assert_array_equal(
            used.extract(context()), FeatureExtractor().extract(context())
        )


class TestNoLookahead:
    def test_features_are_a_pure_function_of_the_context(self) -> None:
        """Same context and same history must give the same features, always."""
        a, b = FeatureExtractor(), FeatureExtractor()
        rng = np.random.default_rng(0)
        for _ in range(50):
            price = int(rng.integers(90, 110))
            ctx = context(
                snapshot(bids=((price - 1, 12),), asks=((price + 1, 8),)),
                inventory=int(rng.integers(-10, 10)),
                trade_flow=int(rng.integers(-20, 20)),
            )
            np.testing.assert_array_equal(a.extract(ctx), b.extract(ctx))

    def test_the_context_exposes_no_future_or_latent_information(self) -> None:
        """A structural guard: the agent's observation type must not grow a leak.

        If someone later adds the latent fundamental or a future price to ``MarketContext``, this
        test fails and forces the question to be asked out loud.
        """
        allowed = {
            "ts",
            "snapshot",
            "inventory",
            "step",
            "steps_total",
            "trade_flow",
            "traded_volume",
        }
        actual = set(MarketContext.__dataclass_fields__)
        assert actual == allowed, (
            f"MarketContext gained field(s) {sorted(actual - allowed)}; if this is future or "
            "latent information, the agent must not see it"
        )


class TestDocumentationDoesNotDrift:
    """Prose in the README states counts that code owns. Numbers in prose rot silently."""

    def _readme(self) -> str:
        return (Path(__file__).resolve().parents[1] / "README.md").read_text()

    def test_the_readme_states_the_real_feature_count(self) -> None:
        match = re.search(r"features\.py\s+(\d+) microstructure features", self._readme())
        assert match, "README no longer describes features.py; update this test or the README"
        assert int(match.group(1)) == N_FEATURES

    def test_the_readme_states_the_real_group_count(self) -> None:
        match = re.search(r"microstructure features in (\d+) ablatable groups", self._readme())
        assert match
        assert int(match.group(1)) == len(FEATURE_GROUPS)
