"""Fitted Q-Iteration market maker with gradient-boosted trees.

Why batch RL rather than online RL. The simulator produces roughly 600 decisions per episode, and
each decision's consequences unfold over seconds of subsequent order flow. Online methods (DQN,
PPO) would need replay buffers, target networks, and a great deal of wall-clock on a CPU-only
laptop to become stable. Fitted Q-Iteration (Ernst, Geurts & Wehenkel 2005) instead collects a
fixed transition dataset once, then solves the Bellman equation by repeated supervised regression::

    Q_{k+1}(s, a)  <-  r + gamma * max_a' Q_k(s', a')

Each iteration is a single fit of a gradient-boosted tree ensemble. This is stable, has no learning
rate to tune, runs in minutes on 8 cores, and -- unlike a neural policy -- leaves a model that can
be interrogated directly with permutation importance, which is what the feature-group ablation
uses.

**The action space is what makes this a market-making policy rather than a generic controller.**
Each side independently chooses to join the touch, rest one or two ticks behind it, or pull. Prices
are always expressed *relative to the touch*, so the policy learns a quoting rule that transfers
across price levels rather than memorising absolute prices.

Two subtleties that the engine, not this file, enforces, and which the policy must therefore learn
to live with: re-quoting to a different price forfeits queue position, and a quote that would cross
is rejected outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from lobsim.engine import MarketContext, Quote
from lobsim.features import FEATURE_NAMES, FeatureConfig, FeatureExtractor, group_mask
from lobsim.types import Fill

if TYPE_CHECKING:
    from collections.abc import Sequence

# Quote offsets in ticks behind the touch; None means "no quote on this side".
SIDE_OFFSETS: tuple[int | None, ...] = (0, 1, 2, None)
ACTIONS: tuple[tuple[int | None, int | None], ...] = tuple(
    (bid, ask) for bid in SIDE_OFFSETS for ask in SIDE_OFFSETS
)
N_ACTIONS = len(ACTIONS)


def _offset_label(offset: int | None) -> str:
    return "pull" if offset is None else f"+{offset}"


def action_label(index: int) -> str:
    bid, ask = ACTIONS[index]
    return f"bid:{_offset_label(bid)}/ask:{_offset_label(ask)}"


@dataclass(frozen=True)
class FQIConfig:
    """Hyperparameters. Every one of these is selected on validation seeds, never on test seeds."""

    n_iterations: int = 6
    discount: float = 0.98
    max_iter: int = 120
    max_depth: int | None = 6
    learning_rate: float = 0.1
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    feature_groups: tuple[str, ...] = ("book", "flow", "queue", "position")
    size: int = 2
    random_state: int = 0
    double: bool = True
    # Risk aversion: charged as phi * inventory^2 per decision step during training only.
    inventory_penalty: float = 0.0

    def label(self) -> str:
        groups = "+".join(self.feature_groups)
        return (
            f"iters{self.n_iterations}_depth{self.max_depth}_lr{self.learning_rate}"
            f"_leaf{self.min_samples_leaf}_phi{self.inventory_penalty}_g[{groups}]"
        )


@dataclass
class Transition:
    """One decision step, stored for batch fitting.

    ``episode`` is carried so the double estimator can split on *episodes* rather than on rows.
    Splitting on rows would put temporally adjacent -- and therefore strongly correlated --
    transitions on both sides of the split, which destroys the independence the double estimator
    relies on to cancel maximisation bias.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool
    episode: int = -1


class QModel:
    """Q(s, a) as a single regressor with the action as a native categorical feature.

    The obvious alternative -- one regressor per action -- was implemented first and rejected on
    measurement. Two problems. Inference cost: scoring 16 actions meant 16 separate ``predict``
    calls per decision, and sklearn's per-call overhead dominates on a single row, costing ~12 ms
    per decision (~7 s per episode) even after pinning OpenMP to one thread. Tiling the state once
    and scoring all actions in a single call is ~16x faster because it pays that overhead once.

    The reason per-action heads were attractive is real, though: a tree ensemble handed the action
    as an ordinary numeric column is free to never split on it, which would yield an
    action-independent Q and hence an arbitrary policy. Declaring the action as a *categorical*
    feature makes it a first-class split candidate, and
    ``tests/test_fqi.py::TestQModel::test_q_values_actually_depend_on_the_action`` asserts that the
    fitted model discriminates between actions rather than trusting that it does.
    """

    def __init__(self, config: FQIConfig, n_features: int) -> None:
        self.config = config
        self.n_features = n_features
        self.model: HistGradientBoostingRegressor | None = None
        self._action_column = n_features

    def _new_regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            early_stopping=False,
            categorical_features=[self._action_column],
            random_state=self.config.random_state,
        )

    def _design(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.hstack([states, actions.reshape(-1, 1).astype(np.float64)])

    def fit(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> None:
        model = self._new_regressor()
        model.fit(self._design(states, actions), targets)
        self.model = model

    def value_of(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Q(s, a) evaluated at one specific action per row."""
        if self.model is None:
            return np.zeros(states.shape[0], dtype=np.float64)
        return np.asarray(self.model.predict(self._design(states, actions)), dtype=np.float64)

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """(n_samples, n_actions) matrix of Q estimates, in a single batched prediction."""
        if self.model is None:
            return np.zeros((states.shape[0], N_ACTIONS), dtype=np.float64)
        n = states.shape[0]
        tiled = np.repeat(states, N_ACTIONS, axis=0)
        actions = np.tile(np.arange(N_ACTIONS), n)
        return self.value_of(tiled, actions).reshape(n, N_ACTIONS)

    @property
    def is_fitted(self) -> bool:
        return self.model is not None


class AveragedQ:
    """Mean of two independently fitted Q functions -- the policy used after double FQI."""

    def __init__(self, first: QModel, second: QModel) -> None:
        self.members = (first, second)
        self.config = first.config
        self.n_features = first.n_features

    def q_values(self, states: np.ndarray) -> np.ndarray:
        return sum(m.q_values(states) for m in self.members) / len(self.members)  # type: ignore[return-value]

    @property
    def is_fitted(self) -> bool:
        return all(m.is_fitted for m in self.members)

    def value_of(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return sum(m.value_of(states, actions) for m in self.members) / len(self.members)  # type: ignore[return-value]


QFunction = QModel | AveragedQ


@dataclass
class FQIResult:
    """A fitted Q function plus the diagnostics needed to tell whether fitting went wrong."""

    model: QFunction
    target_scale: list[float]
    n_transitions: int
    double: bool

    @property
    def target_inflation(self) -> float:
        """Ratio of final to first mean absolute target.

        Fitted Q-Iteration bootstraps through a ``max``, and a ``max`` over noisy estimates is
        biased upward, so errors can compound across iterations. A ratio near 1 means the value
        estimates settled; a large ratio means they are inflating and the greedy policy is chasing
        overestimates. This is reported rather than hidden because it is the single most useful
        diagnostic for whether an FQI run can be trusted.
        """
        if not self.target_scale or self.target_scale[0] <= 0:
            return float("nan")
        return self.target_scale[-1] / self.target_scale[0]


@dataclass
class _Arrays:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminal: np.ndarray

    def __len__(self) -> int:
        return int(self.rewards.size)


def _to_arrays(transitions: Sequence[Transition]) -> _Arrays:
    return _Arrays(
        states=np.stack([t.state for t in transitions]),
        next_states=np.stack([t.next_state for t in transitions]),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        rewards=np.array([t.reward for t in transitions], dtype=np.float64),
        terminal=np.array([t.terminal for t in transitions], dtype=bool),
    )


def _fit_all_actions(model: QModel, data: _Arrays, targets: np.ndarray) -> None:
    model.fit(data.states, data.actions, targets)


def fit_fqi(
    transitions: Sequence[Transition],
    config: FQIConfig,
    verbose: bool = False,
) -> FQIResult:
    """Run Fitted Q-Iteration over a fixed transition dataset.

    With ``config.double`` the estimator is split in the Double-Q sense: two Q functions are fitted
    on disjoint sets of *episodes*, and each one's bootstrap target is evaluated by the other at the
    action the first considers greedy. Selecting and evaluating with the same noisy estimator is
    what produces upward bias; separating them removes it. The single-estimator path is kept so the
    ablation can measure how much that bias actually costs.
    """
    if not transitions:
        raise ValueError("no transitions to fit on")
    if not config.double:
        return _fit_single(transitions, config, verbose)
    return _fit_double(transitions, config, verbose)


def _fit_single(transitions: Sequence[Transition], config: FQIConfig, verbose: bool) -> FQIResult:
    data = _to_arrays(transitions)
    model = QModel(config, data.states.shape[1])
    targets = data.rewards.copy()
    scale = []

    for iteration in range(config.n_iterations):
        _fit_all_actions(model, data, targets)
        # Terminal steps get no continuation value, which stops the episode-end liquidation from
        # being valued as if trading continued.
        continuation = model.q_values(data.next_states).max(axis=1)
        continuation[data.terminal] = 0.0
        targets = data.rewards + config.discount * continuation
        scale.append(float(np.abs(targets).mean()))
        if verbose:
            print(
                f"    FQI iteration {iteration + 1}/{config.n_iterations}: "
                f"mean|target| = {scale[-1]:.4f}"
            )
    return FQIResult(model=model, target_scale=scale, n_transitions=len(data), double=False)


def _fit_double(transitions: Sequence[Transition], config: FQIConfig, verbose: bool) -> FQIResult:
    episodes = sorted({t.episode for t in transitions})
    left = {e for i, e in enumerate(episodes) if i % 2 == 0}
    halves = [
        [t for t in transitions if t.episode in left],
        [t for t in transitions if t.episode not in left],
    ]
    if not halves[0] or not halves[1]:
        # Only one episode available (unit tests, degenerate runs): fall back rather than crash.
        return _fit_single(transitions, config, verbose)

    data = [_to_arrays(h) for h in halves]
    models = [
        QModel(config, data[0].states.shape[1]),
        QModel(config, data[1].states.shape[1]),
    ]
    targets = [d.rewards.copy() for d in data]
    scale = []

    for iteration in range(config.n_iterations):
        for index in (0, 1):
            _fit_all_actions(models[index], data[index], targets[index])
        for index in (0, 1):
            other = 1 - index
            # Select the greedy action with this half's model, evaluate it with the other's.
            greedy = models[index].q_values(data[index].next_states).argmax(axis=1)
            continuation = models[other].value_of(data[index].next_states, greedy)
            continuation[data[index].terminal] = 0.0
            targets[index] = data[index].rewards + config.discount * continuation
        combined = float(np.abs(np.concatenate(targets)).mean())
        scale.append(combined)
        if verbose:
            print(
                f"    FQI iteration {iteration + 1}/{config.n_iterations}: "
                f"mean|target| = {combined:.4f}"
            )
    return FQIResult(
        model=AveragedQ(models[0], models[1]),
        target_scale=scale,
        n_transitions=len(transitions),
        double=True,
    )


@dataclass
class FQIAgent:
    """Greedy (or epsilon-greedy) policy over a fitted Q function.

    Also serves as the *behaviour* policy during data collection: with ``model=None`` and
    ``epsilon=1.0`` it is a uniform random quoter, which is what generates the initial dataset.
    """

    model: QFunction | None = None
    epsilon: float = 0.0
    # Mean number of decision steps an exploratory action is held before resampling.
    action_persistence: float = 1.0
    size: int = 2
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    feature_groups: tuple[str, ...] = ("book", "flow", "queue", "position")
    name: str = "fqi"
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def __post_init__(self) -> None:
        self._extractor = FeatureExtractor(self.feature_config)
        self._mask = group_mask(self.feature_groups)
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self._held_action: int | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._extractor.reset()
        self.last_state = None
        self.last_action = None
        self._held_action = None

    def observe_fill(self, fill: Fill) -> None:
        """Fills enter the policy through the state, not through a separate channel."""

    def state_of(self, ctx: MarketContext) -> np.ndarray:
        return np.asarray(self._extractor.extract(ctx)[self._mask], dtype=np.float64)

    def select_action(self, state: np.ndarray) -> int:
        if self.model is None or not self.model.is_fitted or self.rng.random() < self.epsilon:
            return self._explore()
        values = self.model.q_values(state.reshape(1, -1))[0]
        # Break ties uniformly rather than by index: np.argmax would systematically favour the
        # lowest-numbered action, which here is "join on both sides".
        best = np.flatnonzero(values == values.max())
        return int(best[0] if best.size == 1 else self.rng.choice(best))

    def _explore(self) -> int:
        """Sample an exploratory action, holding it for a geometrically distributed run.

        Temporally correlated exploration is not a refinement here, it is a requirement. Queue
        priority is *earned by not moving your quote*: an order that stays put advances as the
        volume ahead of it trades and cancels, and re-pricing sends it to the back of the new queue.
        Resampling an independent action every step therefore produces trajectories in which the
        agent perpetually churns its quotes and is almost never near the front of a queue -- so the
        training data contains essentially no examples of the mechanism the policy most needs to
        learn, and the state distribution looks nothing like the one a greedy policy would visit.

        Measured: with i.i.d. exploration the learned policy collapsed onto quoting both touches
        unconditionally and lost money; holding actions for ~10 steps is what makes the dataset
        informative. See docs/DECISIONS.md D9.
        """
        if self._held_action is not None and self.rng.random() > 1.0 / max(
            self.action_persistence, 1.0
        ):
            return self._held_action
        self._held_action = int(self.rng.integers(N_ACTIONS))
        return self._held_action

    def act(self, ctx: MarketContext) -> Quote:
        state = self.state_of(ctx)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        return self.quote_for(ctx, action)

    def quote_for(self, ctx: MarketContext, action: int) -> Quote:
        """Translate an action index into absolute tick prices relative to the touch."""
        bid_offset, ask_offset = ACTIONS[action]
        snap = ctx.snapshot
        best_bid, best_ask = snap.best_bid, snap.best_ask
        bid = None if bid_offset is None or best_bid is None else best_bid - bid_offset
        ask = None if ask_offset is None or best_ask is None else best_ask + ask_offset
        return Quote(bid_price=bid, ask_price=ask, size=self.size)

    def feature_names(self) -> tuple[str, ...]:
        return tuple(n for n, keep in zip(FEATURE_NAMES, self._mask, strict=True) if keep)


def permutation_importance(
    model: QFunction,
    states: np.ndarray,
    feature_names: Sequence[str],
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> dict[str, float]:
    """How much the fitted Q function depends on each feature.

    Measured as the mean absolute change in the whole ``Q(s, a)`` matrix when one feature column is
    shuffled.

    The more familiar formulation -- the *drop* in ``max_a Q`` -- is wrong here, and measurably so.
    ``max`` is convex, so injecting noise into an input tends to *raise* it; a feature that the
    model genuinely uses can therefore score zero or negative, and a pure-noise feature can outrank
    the one that carries all the signal. That is exactly what happened when this was first written:
    on synthetic data whose reward depended only on feature 0, the drop-in-max version ranked an
    irrelevant feature top. Mean absolute change is monotone in dependence and has no such failure.

    This is a statement about the fitted model, not about the market: a feature can matter to the
    model and still be useless in reality. The feature-group ablation -- retrain without a group,
    re-measure PnL out of sample -- is the stronger test; this is the cheap diagnostic that
    motivates it.
    """
    baseline = model.q_values(states)
    importances: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            shuffled = states.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            deltas.append(float(np.abs(model.q_values(shuffled) - baseline).mean()))
        importances[name] = float(np.mean(deltas))
    return importances


def model_payload(model: QFunction) -> dict[str, Any]:
    """Small JSON-friendly description of a fitted model, for the results record."""
    heads = [model] if isinstance(model, QModel) else list(model.members)
    return {
        "n_actions": N_ACTIONS,
        "actions": [action_label(i) for i in range(N_ACTIONS)],
        "n_estimators": len(heads),
        "all_estimators_fitted": all(h.is_fitted for h in heads),
        "n_features": model.n_features,
        "config": {
            "n_iterations": model.config.n_iterations,
            "discount": model.config.discount,
            "max_depth": model.config.max_depth,
            "learning_rate": model.config.learning_rate,
            "min_samples_leaf": model.config.min_samples_leaf,
            "feature_groups": list(model.config.feature_groups),
        },
    }
