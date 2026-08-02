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

## D7 — Informed-flow strength calibrated to the realized/effective spread ratio

**Decision.** `informed_kappa` was set to 0.60 by matching the *realized-to-effective spread ratio*
of a fixed reference policy (`AlwaysAtTouch`) to roughly 0.25, inside the 0.2-0.5 band reported in
the empirical microstructure literature (Huang & Stoll 1996 and successors).

**Alternatives.** Picking the tilt by eye; picking it so that some policy looks profitable; not
tilting aggressive flow at all.

**Why.** The strength of informed flow directly sets how much adverse selection a market maker
faces, so it is the parameter most capable of manufacturing whatever conclusion the author wants.
Tying it to a published, policy-independent statistic removes that freedom. The criterion was fixed
*before* any learned policy existed, it is measured with a reference baseline rather than with the
policy under test, and it is now reported as a stylized fact in its own right. The measured ratio is
in `results/raw/validation.json`.

Two things worth noting about what this calibration bought. It also improved the variance ratio
substantially -- from 0.73 at `kappa = 1.0` to 0.98 at 10 seconds -- so the mid became a far better
martingale as a side effect. And it is a *single-statistic* calibration: it identifies the strength
of informed flow given everything else, not the joint parameter vector.

## D8 — A two-timescale Hawkes kernel was implemented, measured, and rejected

**Decision.** Market-order arrivals use a *single* exponential Hawkes kernel. A two-timescale
kernel (fast bursts plus slow memory) was implemented in full, measured against a pre-registered
acceptance criterion, failed it, and was reverted.

**Alternatives.** Keeping the two-timescale kernel; pushing the branching ratio up until volatility
clustering appears.

**Why.** The simulator does not reproduce volatility clustering beyond about a second: the
autocorrelation of |returns| is ~0.24 at lag 1 but ~0 at lag 10 and lag 30, where real markets show
slowly decaying positive values. The obvious fix is a second, slower excitation component, which is
the standard construction for long memory.

Before implementing it, the acceptance test was fixed in advance: *keep the change only if it
improves clustering at both lag 10 and lag 30 while the variance ratio at 10s and 30s stays inside
[0.7, 1.3]*. Measured, across a sweep of slow-component amplitudes:

| slow self / cross | branching | acf&#124;r&#124; lag 1 | lag 10 | lag 30 | VR(10) | VR(30) |
|---|---|---|---|---|---|---|
| 0 (control)       | 0.29 | 0.186 | 0.008 | -0.011 | 1.087 | 1.234 |
| 0.0035 / 0.0015   | 0.39 | 0.204 | 0.012 | -0.016 | 0.998 | 1.037 |
| 0.0070 / 0.0030   | 0.49 | 0.236 | 0.002 |  0.011 | 0.774 | 0.812 |
| 0.0120 / 0.0050   | 0.63 | 0.320 | 0.040 |  0.010 | 0.489 | 0.368 |
| 0.0180 / 0.0080   | 0.81 | -0.042 | 0.041 | 0.036 | 0.937 | 1.195 |

No setting passes. Amplitudes small enough to preserve the martingale leave lag-30 clustering
unchanged or *worse*; amplitudes large enough to create clustering collapse the variance ratio to
0.49 or below, and a mean-reverting mid would pay a market maker for bearing no risk, invalidating
every PnL number in the repo. At the largest amplitude the process degenerates entirely (negative
lag-1 autocorrelation).

The same sweep was repeated over the *single* kernel's decay rate with the slow component disabled,
and the result is identical in character: across every configuration with an acceptable variance
ratio, clustering at lag 10 is ~0.01 and at lag 30 is ~0.

The mechanism is specific to this model. Volatility clustering in the mid requires *bursts that move
the price*, but a persistent, low-amplitude increase in the market-order rate is absorbed by a deep
queue without moving the touch at all. Extra excitation therefore buys order-flow persistence --
which shows up as mean reversion in the mid via imbalance -- rather than volatility persistence.
Reproducing long-memory volatility would need a mechanism this simulator does not have, most
plausibly stochastic volatility in the latent fundamental itself or a liquidity process that
withdraws depth during bursts.

The kernel was reverted to exactly its pre-investigation parameters, so this excursion changed no
number in the repo. The failure is reported as a failing stylized fact rather than hidden, and the
limitation is stated in the README.

## D9 — Exploration must be temporally correlated, because queue priority is earned by waiting

**Decision.** The behaviour policy that generates training data holds each exploratory action for a
geometrically distributed run of ~10 decision steps rather than resampling every step.

**Alternatives.** Independent uniform action selection each step, the textbook default.

**Why.** Queue priority in this simulator -- as in a real book -- is earned by *not moving a quote*.
An order that stays put advances as the volume ahead of it trades and cancels; re-pricing cancels
it and sends it to the back of the new queue. An i.i.d. behaviour policy therefore churns its
quotes on every decision, spends the entire episode near the back of whatever queue it just joined,
and produces a dataset containing essentially no examples of the single mechanism the policy most
needs to learn. It also makes the behaviour state distribution wildly unlike the one a greedy
policy would visit, which is exactly the distribution shift batch RL is most fragile to.

Measured: with i.i.d. exploration the learned policy collapsed onto quoting both touches
unconditionally and returned -76.8 ticks on validation seeds; with persistent exploration and the
same everything else it improved to -33.8. Both numbers are still negative -- see Limitations --
but the gap is the cost of the wrong exploration scheme.

## D10 — A running inventory penalty is required to make the control problem well-posed

**Decision.** Training rewards are ``delta(mark-to-market) - phi * inventory^2``. Evaluation and
every reported number use unpenalised PnL.

**Alternatives.** Pure PnL reward; terminal-only inventory penalty; constraining inventory only
through the engine's hard cap.

**Why.** The simulator's mid is a near-martingale by design and by measurement (variance ratio 0.98
at 10s). Under a martingale, carrying inventory has *zero expected PnL* -- it is pure variance. A
risk-neutral value function therefore sees no reason to control position at all, and the greedy
policy correctly concludes that quoting both sides unconditionally maximises expected reward. That
is exactly what was observed: with ``phi = 0``, the learned policy's most-chosen action was
"join both touches", its inventory RMS was 27 lots, and it reproduced the always-at-touch baseline's
losses.

The penalty is the discrete-time analogue of the running inventory term in Cartea-Jaimungal, and of
the exponential utility that yields the Avellaneda-Stoikov reservation price. It is a device for
expressing risk aversion, not a thumb on the scale: model selection and every reported result use
unpenalised PnL, so a policy cannot win by being scored on its own training objective.

## D11 — The deflated Sharpe ratio is corrected for *development* trials, not just the final grid

**Decision.** The number of trials fed to the deflated Sharpe ratio is 24 -- every FQI
configuration evaluated on validation seeds across the whole of development -- while
`make reproduce` re-runs a grid of only 4.

**Alternatives.** Using the grid size (4); not deflating at all.

**Why.** The multiple-testing burden comes from every configuration whose validation result
influenced the final choice, not from the subset that survived into the reproduction script.
Reporting 4 would understate the search and inflate the deflated Sharpe, which is precisely the
failure mode the statistic exists to catch. The count is maintained by hand in
`scripts/train_policy.py::DEVELOPMENT_TRIALS`, which is an honesty mechanism rather than an
automated one, and is stated as such. It excludes market-calibration sweeps, which did not select
on policy performance -- see D7.

## D12 — Worker count is capped by memory, and ablations run on half the test seeds

**Decision.** `MAX_WORKERS = 3` regardless of core count, and the ablation suite evaluates on the
first 100 of the 200 held-out test seeds. The headline table uses all 200.

**Alternatives.** Scaling workers with `cpu_count()`; running every ablation on the full test set.

**Why.** Both are concessions to an 8 GB laptop, and both are stated rather than hidden. Worker
processes are memory-bound, not CPU-bound: each spawned worker re-imports numpy, scipy and
scikit-learn before running a single episode. Six workers drove free memory to ~65 MB, and the run
stalled with every process pegged at 100% CPU and none making progress -- a memory stall that
presents exactly like a deadlock. A single FQI episode costs 1.3 seconds of real work, so the extra
parallelism was being spent on page faults; three workers is barely slower.

The ablation subset is a *prefix* of the same held-out seeds, never a re-draw and never overlapping
training or validation, so the only cost is statistical power. That cost is visible in the results:
the feature-group ablations show a monotone ordering but only the book-only variant reaches
significance. The README says so rather than presenting the ordering as established.
