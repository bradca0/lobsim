# Decisions

Non-obvious choices, the alternatives considered, and why. Append-only; newest at the bottom.

## D1 — Integer ticks and integer nanoseconds, never floats, in the engine

**Decision.** Prices are `int` tick indices, sizes are `int` lots, timestamps are `int` nanoseconds.
Floats appear only at the analysis boundary.

**Alternatives.** Float prices (what most toy simulators do); `Decimal`.

**Why.** Price-time priority is an equality-driven algorithm: level lookup, "is this order at the
touch", and queue arithmetic all compare keys for exact equality. Float prices make
`0.1 + 0.2 != 0.3` a matching-engine bug rather than a rounding nuisance, and they silently break
dict-keyed price levels. `Decimal` is exact but ~50x slower than `int` in the hot loop, and we run
hundreds of millions of events. Real exchanges disseminate integer prices for the same reason.

## D2 — Coverage omits `plotting.py`

**Decision.** `src/lobsim/plotting.py` is excluded from the coverage denominator.

**Alternatives.** Image-comparison tests; counting it and writing trivial smoke tests to pad.

**Why.** The brief demands 85% coverage on *core logic, not padded with trivial tests*. Asserting on
matplotlib output is either brittle (pixel comparison) or vacuous (`assert fig is not None`). The
figure scripts are exercised end-to-end by `make reproduce`, which is the honest test for them.

## D3 — Zero-intelligence background flow *plus* a latent fundamental, not one or the other

**Decision.** Background flow is zero-intelligence (Poisson limit orders placed by a power-law
depth profile, volume-proportional cancellations) with two additions: market-order arrivals follow
a bivariate Hawkes process, and their *direction* is tilted toward a latent fundamental price that
diffuses as a random walk.

**Alternatives.** (a) Pure zero-intelligence, as in Smith et al. and Cont-Stoikov-Talreja.
(b) A fully agent-based market with strategic participants. (c) Replaying real historical LOB data.

**Why.** Pure ZI was implemented first and measured — it fails in a way that would have silently
invalidated the entire study. With a realistically deep queue (~20 lots at the touch) the mid moved
a total of **1-4 ticks over a 300-second episode**: price can only move when a level is fully
cleared, and limit-order refill at the touch is faster than market orders can deplete it. Two
consequences, both fatal: there is no inventory risk, and there is no adverse selection, so market
making degenerates into riskless spread capture and any policy comparison measures nothing. The
latent-fundamental tilt fixes both — aggressive flow becomes predictive of the next price move, so
a resting quote is systematically run over just before the market moves against it, which is the
actual risk a market maker is paid to bear. (b) Was rejected as unfalsifiable: strategic agents
introduce many more free parameters without any more ground truth to fit them to. (c) Was rejected
because published, redistributable message-level LOB data does not exist for free, and the
project's claim is about *fill mechanics*, which a synthetic market can express exactly.

## D4 — Flow parameters were tuned to book-shape targets, and only those are identified

**Decision.** `FlowParams` defaults were chosen by sweeping arrival, cancel and volatility rates
until the simulated book matched a target regime: median spread of 1 tick, ~15-25 lots resting at
the touch, and a mid price that moves ~10-15 ticks over a 5-minute episode.

**Alternatives.** Fitting parameters by maximum likelihood to real LOB data; leaving textbook
defaults unchanged.

**Why.** The regime matters more than the exact numbers: a 1-tick spread with a deep queue is the
tick-constrained regime in which queue position, rather than quoted price, dominates a market
maker's fill probability — which is exactly the mechanism this repo exists to measure. This is a
calibration to *stylized targets*, not an estimation: the parameters are not identified in any
statistical sense, and a different parameter vector reproducing the same book shape would be
equally admissible. The honest reading is that results are conditional on this regime, which is
why the cancel-policy and fill-model ablations exist and why sensitivity is reported rather than
assumed away.

## D5 — Cancellation queue position is a first-class, switchable assumption

**Decision.** `CancelPolicy` selects between `UNIFORM` (a cancelled lot is drawn uniformly from
the queue) and `BACK_LOADED` (orders nearer the back are likelier to be pulled). `UNIFORM` is the
default and both are reported.

**Alternatives.** Hard-coding one; cancelling only from the back of the queue.

**Why.** This single assumption sets how fast the queue ahead of a resting order evaporates, and
therefore how often a market maker at the touch gets filled at all. Hard-coding it would bury the
most consequential unverifiable choice in the model. `UNIFORM` is the more *generous* of the two to
the agent, so using it as the default means the headline result is not manufactured by a pessimistic
assumption.

## D6 — Common random numbers, not identical order flow

**Decision.** Every policy is evaluated on the same seed set, with independent RNG streams for the
background flow and the agent.

**Alternatives.** Claiming exactly identical flow across policies; using independent seeds per
policy.

**Why.** A shared seed supplies the same underlying randomness to every policy, which is a standard
variance-reduction device (common random numbers) and lets comparisons be paired — a large win in
statistical power. It is *not* true that the realised order flow is identical across policies, and
the code says so: the agent is a market participant, so its quotes change the touch and absorb
aggressive orders that would otherwise have hit someone else. Independent seeds would have thrown
away the pairing for no benefit.
