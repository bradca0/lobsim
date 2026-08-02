"""Tests for the learned policy: action space, Q model, fitting, and exploration.

Two of these guard failure modes that are silent rather than loud. A Q function that ignores the
action still runs, still produces a policy, and still reports numbers -- it is just arbitrary. And
i.i.d. exploration still collects data, it just collects data about a market maker that never holds
a queue position. Both are asserted rather than assumed.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from lobsim.agents.fqi import (
    ACTIONS,
    N_ACTIONS,
    AveragedQ,
    FQIAgent,
    FQIConfig,
    QModel,
    Transition,
    action_label,
    fit_fqi,
    model_payload,
    permutation_importance,
)
from lobsim.backtest import BacktestConfig
from lobsim.engine import MarketContext, SimConfig
from lobsim.features import N_FEATURES
from lobsim.training import collect_dataset, collect_episode, reward_summary
from lobsim.types import BookSnapshot

SHORT = BacktestConfig(sim=SimConfig(horizon_seconds=60.0, burn_in_seconds=10.0))


def context(bid: int = 99, ask: int = 101, inventory: int = 0) -> MarketContext:
    return MarketContext(
        ts=0,
        snapshot=BookSnapshot(
            ts=0, bids=((bid, 10), (bid - 1, 20)), asks=((ask, 10), (ask + 1, 20))
        ),
        inventory=inventory,
        step=0,
        steps_total=100,
    )


class TestActionSpace:
    def test_the_action_space_is_the_product_of_the_per_side_choices(self) -> None:
        assert N_ACTIONS == 16
        assert len(set(ACTIONS)) == N_ACTIONS

    def test_both_sides_can_be_pulled_and_both_can_join(self) -> None:
        assert (None, None) in ACTIONS
        assert (0, 0) in ACTIONS

    def test_labels_are_unique_and_readable(self) -> None:
        labels = [action_label(i) for i in range(N_ACTIONS)]
        assert len(set(labels)) == N_ACTIONS
        assert "bid:pull/ask:pull" in labels
        assert "bid:+0/ask:+2" in labels

    def test_quotes_are_placed_relative_to_the_touch(self) -> None:
        agent = FQIAgent()
        ctx = context(bid=500, ask=505)
        for index, (bid_offset, ask_offset) in enumerate(ACTIONS):
            quote = agent.quote_for(ctx, index)
            assert quote.bid_price == (None if bid_offset is None else 500 - bid_offset)
            assert quote.ask_price == (None if ask_offset is None else 505 + ask_offset)

    def test_quotes_never_cross_by_construction(self) -> None:
        agent = FQIAgent()
        ctx = context(bid=100, ask=101)  # the tightest possible book
        for index in range(N_ACTIONS):
            quote = agent.quote_for(ctx, index)
            if quote.bid_price is not None and quote.ask_price is not None:
                assert quote.bid_price < quote.ask_price

    def test_a_one_sided_book_yields_a_one_sided_quote(self) -> None:
        agent = FQIAgent()
        ctx = MarketContext(
            ts=0,
            snapshot=BookSnapshot(ts=0, bids=(), asks=((101, 5),)),
            inventory=0,
            step=0,
            steps_total=10,
        )
        assert agent.quote_for(ctx, 0).bid_price is None


def synthetic_transitions(n: int = 900, seed: int = 0) -> list[Transition]:
    """A dataset where the best action is a known function of the state.

    Action 0 pays 1.0 when the first feature is positive; action 5 pays 1.0 when it is negative.
    Everything else pays nothing, so a correctly fitted Q must separate the actions.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        state = rng.normal(size=N_FEATURES)
        action = int(rng.integers(N_ACTIONS))
        if action == 0:
            reward = 1.0 if state[0] > 0 else 0.0
        elif action == 5:
            reward = 1.0 if state[0] <= 0 else 0.0
        else:
            reward = 0.0
        out.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=rng.normal(size=N_FEATURES),
                terminal=bool(i % 50 == 49),
                episode=i // 50,
            )
        )
    return out


class TestQModel:
    def test_q_values_actually_depend_on_the_action(self) -> None:
        """A tree ensemble handed the action as an ordinary column may never split on it.

        If that happens the Q function is action-independent and the greedy policy is arbitrary
        noise, while everything downstream still runs and still reports numbers. The action is
        declared categorical precisely to prevent this, and this test is what verifies it.
        """
        result = fit_fqi(synthetic_transitions(), FQIConfig(n_iterations=1, double=False))
        states = np.stack([t.state for t in synthetic_transitions(200, seed=1)])
        q = result.model.q_values(states)
        assert q.shape == (200, N_ACTIONS)
        assert (q.max(axis=1) - q.min(axis=1)).mean() > 0.1

    def test_the_fitted_policy_recovers_the_known_optimal_action(self) -> None:
        result = fit_fqi(synthetic_transitions(2000), FQIConfig(n_iterations=1, double=False))
        positive = np.zeros((1, N_FEATURES))
        positive[0, 0] = 2.0
        negative = np.zeros((1, N_FEATURES))
        negative[0, 0] = -2.0
        assert int(result.model.q_values(positive).argmax()) == 0
        assert int(result.model.q_values(negative).argmax()) == 5

    def test_an_unfitted_model_is_neutral_rather_than_random(self) -> None:
        model = QModel(FQIConfig(), N_FEATURES)
        assert not model.is_fitted
        q = model.q_values(np.zeros((3, N_FEATURES)))
        assert q.shape == (3, N_ACTIONS)
        assert np.all(q == 0.0)

    def test_batched_q_values_match_per_action_evaluation(self) -> None:
        """The batched path is an optimisation; it must not change the answer."""
        result = fit_fqi(synthetic_transitions(), FQIConfig(n_iterations=1, double=False))
        states = np.stack([t.state for t in synthetic_transitions(50, seed=2)])
        batched = result.model.q_values(states)
        for action in range(N_ACTIONS):
            per_action = result.model.value_of(states, np.full(states.shape[0], action))
            np.testing.assert_allclose(batched[:, action], per_action)


class TestFitting:
    def test_terminal_transitions_get_no_continuation_value(self) -> None:
        """Otherwise the episode-end liquidation is valued as if trading continued."""
        transitions = [
            Transition(np.zeros(N_FEATURES), 0, 1.0, np.zeros(N_FEATURES), terminal=True, episode=0)
            for _ in range(200)
        ]
        result = fit_fqi(transitions, FQIConfig(n_iterations=3, discount=0.99, double=False))
        # With reward 1 everywhere and no bootstrapping, targets must stay at 1, not compound.
        assert result.target_scale[-1] == pytest.approx(1.0, abs=0.05)

    def test_non_terminal_transitions_do_bootstrap(self) -> None:
        transitions = [
            Transition(
                np.zeros(N_FEATURES), 0, 1.0, np.zeros(N_FEATURES), terminal=False, episode=0
            )
            for _ in range(200)
        ]
        result = fit_fqi(transitions, FQIConfig(n_iterations=3, discount=0.9, double=False))
        assert result.target_scale[-1] > 1.5

    def test_the_double_estimator_reduces_target_inflation(self) -> None:
        """Selecting and evaluating with the same noisy Q biases the max upward.

        The dataset here is pure noise, so the true value of every action is 0 and any inflation is
        maximisation bias by construction.
        """
        rng = np.random.default_rng(3)
        noise = [
            Transition(
                state=rng.normal(size=N_FEATURES),
                action=int(rng.integers(N_ACTIONS)),
                reward=float(rng.normal()),
                next_state=rng.normal(size=N_FEATURES),
                terminal=False,
                episode=i // 40,
            )
            for i in range(2400)
        ]
        config = FQIConfig(n_iterations=4, discount=0.95)
        single = fit_fqi(noise, FQIConfig(**{**vars(config), "double": False}))
        double = fit_fqi(noise, FQIConfig(**{**vars(config), "double": True}))
        assert double.target_inflation < single.target_inflation

    def test_an_empty_dataset_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no transitions"):
            fit_fqi([], FQIConfig())

    def test_a_single_episode_falls_back_to_the_single_estimator(self) -> None:
        transitions = [
            Transition(np.zeros(N_FEATURES), 0, 1.0, np.zeros(N_FEATURES), False, episode=7)
            for _ in range(100)
        ]
        assert not fit_fqi(transitions, FQIConfig(n_iterations=1, double=True)).double

    def test_payload_is_json_friendly(self) -> None:
        result = fit_fqi(synthetic_transitions(), FQIConfig(n_iterations=1))
        payload = model_payload(result.model)
        assert payload["n_actions"] == N_ACTIONS
        assert payload["all_estimators_fitted"] is True
        assert isinstance(result.model, AveragedQ)


class TestExploration:
    def test_persistent_exploration_holds_actions_across_steps(self) -> None:
        """Queue priority is earned by not moving; i.i.d. action noise never earns any."""
        agent = FQIAgent(epsilon=1.0, action_persistence=20.0)
        agent.reset(np.random.default_rng(0))
        actions = [agent.select_action(np.zeros(N_FEATURES)) for _ in range(400)]
        switches = sum(a != b for a, b in pairwise(actions))
        assert switches < 60, "actions should persist for ~20 steps on average"

    def test_iid_exploration_switches_almost_every_step(self) -> None:
        agent = FQIAgent(epsilon=1.0, action_persistence=1.0)
        agent.reset(np.random.default_rng(0))
        actions = [agent.select_action(np.zeros(N_FEATURES)) for _ in range(400)]
        switches = sum(a != b for a, b in pairwise(actions))
        assert switches > 300

    def test_exploration_still_covers_the_whole_action_space(self) -> None:
        agent = FQIAgent(epsilon=1.0, action_persistence=10.0)
        agent.reset(np.random.default_rng(1))
        seen = {agent.select_action(np.zeros(N_FEATURES)) for _ in range(4000)}
        assert seen == set(range(N_ACTIONS))

    def test_a_greedy_agent_is_deterministic_given_a_model(self) -> None:
        result = fit_fqi(synthetic_transitions(), FQIConfig(n_iterations=1))
        agent = FQIAgent(model=result.model, epsilon=0.0)
        agent.reset(np.random.default_rng(0))
        state = np.zeros(N_FEATURES)
        assert len({agent.select_action(state) for _ in range(20)}) == 1

    def test_reset_clears_the_held_exploratory_action(self) -> None:
        agent = FQIAgent(epsilon=1.0, action_persistence=1e6)
        agent.reset(np.random.default_rng(0))
        first = agent.select_action(np.zeros(N_FEATURES))
        assert agent.select_action(np.zeros(N_FEATURES)) == first  # held
        agent.reset(np.random.default_rng(1))
        assert agent._held_action is None


class TestCollection:
    def test_an_episode_produces_one_transition_per_decision(self) -> None:
        transitions = collect_episode(0, epsilon=1.0, config=SHORT)
        expected = int(SHORT.sim.horizon_seconds / SHORT.sim.decision_interval)
        assert abs(len(transitions) - expected) <= 2

    def test_exactly_one_transition_is_terminal(self) -> None:
        transitions = collect_episode(1, epsilon=1.0, config=SHORT)
        assert sum(t.terminal for t in transitions) == 1
        assert transitions[-1].terminal

    def test_transitions_chain_state_to_next_state(self) -> None:
        transitions = collect_episode(2, epsilon=1.0, config=SHORT)
        for current, following in pairwise(transitions[:-1]):
            np.testing.assert_array_equal(current.next_state, following.state)

    def test_episodes_are_tagged_for_the_double_split(self) -> None:
        data = collect_dataset([3, 4], epsilon=1.0, config=SHORT, n_jobs=1)
        assert {t.episode for t in data} == {3, 4}

    def test_the_inventory_penalty_only_subtracts(self) -> None:
        """Same seed, same actions: the penalised reward can never exceed the unpenalised one."""
        plain = collect_episode(5, epsilon=1.0, config=SHORT, inventory_penalty=0.0)
        penalised = collect_episode(5, epsilon=1.0, config=SHORT, inventory_penalty=0.01)
        assert len(plain) == len(penalised)
        for a, b in zip(plain, penalised, strict=True):
            assert b.reward <= a.reward + 1e-12
        assert sum(t.reward for t in penalised) < sum(t.reward for t in plain)

    def test_rewards_are_not_degenerate(self) -> None:
        summary = reward_summary(collect_episode(6, epsilon=1.0, config=SHORT))
        assert summary["std"] > 0.0
        assert 0.0 < summary["frac_nonzero"] < 1.0

    def test_reward_summary_of_nothing_is_empty(self) -> None:
        assert reward_summary([]) == {"n": 0.0}


class TestPermutationImportance:
    def test_the_informative_feature_dominates(self) -> None:
        """On data whose reward depends only on feature 0, feature 0 must rank first."""
        result = fit_fqi(
            synthetic_transitions(4000),
            FQIConfig(n_iterations=1, double=False, max_iter=60, max_depth=3),
        )
        states = np.stack([t.state for t in synthetic_transitions(400, seed=9)])
        names = tuple(f"f{i}" for i in range(N_FEATURES))
        importances = permutation_importance(
            result.model, states, names, np.random.default_rng(0), n_repeats=3
        )
        ranked = sorted(importances, key=lambda k: importances[k], reverse=True)
        assert ranked[0] == "f0"
        assert importances["f0"] > 2 * importances[ranked[1]]

    def test_importances_are_non_negative(self) -> None:
        """Mean absolute change cannot be negative, unlike the drop-in-max formulation."""
        result = fit_fqi(synthetic_transitions(1000), FQIConfig(n_iterations=1, double=False))
        states = np.stack([t.state for t in synthetic_transitions(200, seed=4)])
        names = tuple(f"f{i}" for i in range(N_FEATURES))
        importances = permutation_importance(
            result.model, states, names, np.random.default_rng(1), n_repeats=2
        )
        assert all(v >= 0.0 for v in importances.values())
