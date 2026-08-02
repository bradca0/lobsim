"""The policy registry shared by every experiment script.

Defined once, in the package, so the backtest, the ablations and the figures cannot silently
disagree about what "the baselines" means. Factories are module-level partials so they survive
pickling into worker processes.
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from typing import Any

from lobsim.agents import (
    AlwaysAtTouch,
    AvellanedaStoikov,
    FixedSpread,
    Inactive,
    InventorySkew,
)
from lobsim.agents.fqi import FQIAgent
from lobsim.backtest import AgentFactory

# Every policy quotes the same size, so differences in PnL are differences in *when and where* a
# policy quotes rather than in how much risk it is permitted to take.
AGENT_SIZE = 2

BASELINES: dict[str, AgentFactory] = {
    "inactive": partial(Inactive, size=AGENT_SIZE),
    "at_touch": partial(AlwaysAtTouch, size=AGENT_SIZE),
    "fixed_spread_2": partial(FixedSpread, size=AGENT_SIZE, half_spread=2),
    "inventory_skew": partial(InventorySkew, size=AGENT_SIZE, threshold=5),
    "avellaneda_stoikov": partial(AvellanedaStoikov, size=AGENT_SIZE, gamma=0.05, kappa=1.5),
}

# The reference the learned policy is judged against in the headline table. Chosen as the strongest
# baseline on validation seeds, not the weakest -- comparing against `at_touch` alone would be a
# strawman, since it has no inventory control at all.
PRIMARY_BASELINE = "inventory_skew"


def load_policy(path: Path) -> tuple[AgentFactory, dict[str, Any]]:
    """Load the trained FQI policy and return a factory plus a description of it."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model = payload["model"]
    config = payload["config"]
    factory = partial(
        FQIAgent,
        model=model,
        epsilon=0.0,
        size=AGENT_SIZE,
        feature_groups=config.feature_groups,
    )
    return factory, {"label": config.label(), "feature_groups": list(config.feature_groups)}


def all_policies(fqi_factory: AgentFactory | None) -> dict[str, AgentFactory]:
    policies = dict(BASELINES)
    if fqi_factory is not None:
        policies["fqi"] = fqi_factory
    return policies
