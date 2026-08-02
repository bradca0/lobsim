# The fifteen hardest questions about this repo

Written adversarially: these are the questions a skeptical reviewer should ask, including the ones
whose honest answer is "that's a real weakness". Every answer points at code or at a file in
`results/raw/` rather than asking to be believed.

---

### 1. Your simulator is synthetic. Why should anyone believe a single number that comes out of it?

They shouldn't believe it as a forecast of a real venue's PnL, and the README says so. What the
simulator is used for is a *comparative* claim — the same policy, on the same order flow, under two
fill assumptions — and that comparison is far more robust to model error than any level.

Fidelity is measured rather than asserted. `src/lobsim/validation.py` estimates eleven stylized
facts from the empirical microstructure literature, `scripts/run_validation.py` runs them on an
*agentless* market so the market maker cannot flatter them, and the results table in the README
includes the failures. The one that matters most is the variance ratio: if the mid mean-reverted,
market making would be paid for bearing no risk and every PnL number would be an artefact. It comes
out at 0.98 at 10 seconds and 1.11 at 30 seconds, both inside [0.7, 1.3].

The honest limit: this is calibration to *stylized targets*, not estimation. The parameters are not
statistically identified, and a different parameter vector reproducing the same book shape would be
equally admissible. That is why the two most consequential assumptions — fill model and
cancellation queue position — are both shipped as ablations rather than as settings.

### 2. Queue-position tracking is easy to get subtly wrong. How do you know yours is right?

`Order.volume_ahead` is maintained by O(1) increments on every trade and cancellation
(`src/lobsim/book.py`, `_decrement_queue_ahead`). Incremental bookkeeping is exactly the kind of
thing that drifts, so it is checked against a full recomputation:
`tests/test_book_properties.py::test_queue_ahead_equals_ground_truth` replays arbitrary
Hypothesis-generated sequences of submissions, market orders and cancellations, then recomputes
each agent order's queue position from scratch as the total size of every order at the same price
with an earlier arrival sequence, and demands exact equality.

Six further property tests cover the invariants that make the answer meaningful: the book never
crosses, level aggregates never drift from the orders they summarise, time priority is never
violated, trades respect price priority, and cancelling everything empties the book.

### 3. Isn't "optimistic fills" a strawman you built to knock down?

It is the standard assumption in the overwhelming majority of retail and academic backtests: if a
trade prints at your price, you were filled. It is what you get for free from OHLCV or trade-tape
data, because those data contain no queue at all. `FillModel.OPTIMISTIC` implements exactly that
and nothing worse — the agent's order is moved to the front of its price level, so it fills on the
first print at its price.

`tests/test_book_properties.py::test_optimistic_fills_dominate_queue_aware_fills` establishes the
direction of the bias: on identical flow, optimistic filling weakly dominates queue-aware filling
in executed agent volume. The size of that bias is the headline table.

### 4. Your learned policy loses money and loses to a five-line heuristic. Why is that in the README?

Because it is what the experiment produced, and because hiding it would make everything else in the
repo untrustworthy. The learned policy substantially beats the naive always-at-touch baseline and
still fails to beat `InventorySkew`, which is about five lines of code.

The analysis is in the README's Limitations section, and the short version is a
signal-to-noise argument. The per-step reward is the change in mark-to-market, whose standard
deviation is roughly 7 ticks, while the difference between a good and a bad action is worth
perhaps a hundredth of that. Fitted Q-Iteration has to resolve that gap from ~120k transitions
spread across 16 actions and a continuous state. Persistent exploration does improve the result
substantially over i.i.d. exploration (-76.8 to -33.8 on validation seeds), so the machinery is not
inert — but working machinery on an unfavourable SNR still loses.

The honest read is that this is a data and reward-design problem, not a "needs a bigger model"
problem, and the README says which specific changes would be tried next.

### 5. How do I know the agent isn't peeking at the future or at the latent fundamental?

Structurally. The agent's entire observation is `MarketContext` (`src/lobsim/engine.py`), which
carries the timestamp, a top-of-book snapshot, its own inventory, the step index, and the public
tape since its last decision. The latent fundamental lives in `OrderFlowGenerator` and is never
placed on that object.

`tests/test_features.py::TestNoLookahead::test_the_context_exposes_no_future_or_latent_information`
pins the field set: if anyone later adds a field, the test fails and forces the question to be
asked out loud. Rewards are computed by the trainer from engine hooks (`src/lobsim/training.py`),
outside the policy, so a policy cannot condition on its own PnL when deciding.

### 6. The agent trades in the market it is being measured in. Doesn't it move the market?

Yes, and that is deliberate — a market maker that had no impact would be the unrealistic case. Its
quotes change the touch, and its resting orders absorb aggressive flow that would otherwise have
hit someone else.

The consequence is stated in `docs/DECISIONS.md` D6: seeds give every policy the same underlying
randomness (common random numbers, a variance-reduction device that makes comparisons paired), but
the *realised* order flow does diverge between policies. The code comment in `Simulation.__init__`
says this explicitly rather than claiming identical flow. One thing the agent cannot do is have its
own orders cancelled by the background market — cancellation targets exclude agent orders
(`tests/test_flow.py::test_cancellations_never_target_agent_orders`).

### 7. What stops the PnL accounting from being quietly wrong?

Two mechanisms, one structural and one a control.

The structural one: PnL is decomposed into spread capture and inventory PnL, accumulated by the
engine one book mutation at a time, and the two must sum to realised PnL exactly.
`tests/test_metrics.py::TestDecompositionIdentity` asserts a residual of zero as a Hypothesis
property over random seeds and all five policies, plus both fill models.

This caught a real bug. Marking was originally bracketed per *decision* rather than per book
mutation, and a two-sided quote mutates the book twice; separately, a market order that sweeps a
side clean leaves the mid undefined, and spread capture and marking both silently skipped those
moments while realised PnL still marked at a fallback. The residual test failed at up to 77 ticks
per episode. The fix — a single always-defined reference price used everywhere — is in
`Simulation._reference_price`, and `one_sided_events` is reported per episode.

The control: the `Inactive` policy must report exactly 0.0 PnL. `scripts/run_backtests.py` asserts
this for every condition and aborts the run if it fails.

### 8. Your baselines look weak. Did you tune them at all?

`AlwaysAtTouch` is deliberately untuned because it is the reference, not a competitor: in a
tick-constrained book with a one-tick spread there is nowhere better to quote, so it is a strong
fill-rate baseline and a weak risk baseline. The primary baseline the learned policy is compared
against is `InventorySkew`, chosen because it was the *strongest* baseline on validation seeds —
comparing only against `at_touch` would be the strawman.

`AvellanedaStoikov` has two free parameters that the model does not identify from the simulator's
parameters, and they were calibrated on training seeds only, exactly like the learned policy's
hyperparameters. Its docstring is explicit that the model's assumptions — continuous prices,
exponentially decaying fill intensity in quote distance — are violated by a discrete book with a
queue, so it is being run outside its derivation.

### 9. You ran a hyperparameter search. Why isn't the reported result just the luckiest configuration?

Three separate guards. Model selection uses validation seeds (500–559) that are disjoint from both
training seeds (0–199) and test seeds (1000–1199), and `scripts/train_policy.py` never loads a test
seed. Comparisons against the baseline are corrected across the family with Holm–Bonferroni. And
the Sharpe ratio is deflated (Bailey & López de Prado) by the number of configurations tried.

The number fed to that deflation is 24 — every FQI configuration evaluated on validation seeds
across all of development — not the 4 that survive into `make reproduce`. Reporting the grid size
would understate the search and flatter the statistic, which is the exact failure the deflated
Sharpe exists to catch. See `docs/DECISIONS.md` D11; this is a hand-maintained honesty mechanism
and is labelled as one.

### 10. Fitted Q-Iteration bootstraps through a `max`. Aren't your Q-values inflated? And did your fix work?

The first half: `mean|target|` grew 1.08 → 3.71 across four iterations on an early run, which looks
exactly like maximisation bias compounding through the Bellman backup.

The fix implemented is a double estimator (`_fit_double` in `src/lobsim/agents/fqi.py`): two Q
functions fitted on disjoint sets of *episodes* — not rows, because adjacent rows are correlated and
would destroy the independence the correction relies on — with each one's bootstrap target evaluated
by the other at the action the first considers greedy. On a synthetic pure-noise dataset, where
every action's true value is zero by construction and therefore *all* inflation is bias, it works:
`tests/test_fqi.py::test_the_double_estimator_reduces_target_inflation` passes.

The second half is where the honest answer is "no". On the real data it made no measurable
difference: inflation 1.8519 with the double estimator versus 1.8516 with a single one, and a paired
PnL difference of -2.99 ticks with a 95% CI of [-20.41, +13.80] (p = 0.73). The ablation is in the
README reported as the null result it is.

The diagnosis is that `target_inflation` is not a clean bias measurement on real data, and I would
not use it that way again. With a discount of 0.97 and three iterations, targets are *supposed* to
grow as the value function integrates over a longer effective horizon; the metric conflates that
legitimate accumulation with maximisation bias and cannot separate them. It is a useful alarm for
"something is diverging" and a poor instrument for "how much bias is there". The synthetic test,
where the true value is known to be zero, is the only place in this repo where the metric means
what its name suggests.

### 11. A gradient-boosted tree can just ignore the action column. How do you know your Q function isn't action-independent?

This is the silent failure that motivated the design. An action-independent Q still runs, still
produces a policy, and still reports numbers — it is simply arbitrary.

The action is declared to sklearn as a *categorical* feature (`QModel._new_regressor`), which makes
it a first-class split candidate rather than one numeric column among eighteen. And
`tests/test_fqi.py::TestQModel::test_q_values_actually_depend_on_the_action` asserts that the fitted
Q spreads meaningfully across actions, while
`test_the_fitted_policy_recovers_the_known_optimal_action` checks on synthetic data — where the
optimal action is a known function of one feature — that the greedy policy recovers it.

An earlier design used one regressor per action, which makes action-dependence structural. It was
rejected on measurement: 16 separate single-row `predict` calls cost 1229 ms per decision, and
batching all actions into one call plus pinning OpenMP to one thread brought that to 1.4 ms, an
890× speedup. The trade was made explicitly and the test replaces the guarantee the old design gave
for free.

### 12. Why Fitted Q-Iteration rather than PPO or DQN?

Compute honesty. The target hardware is an 8 GB M1 laptop with no CUDA. FQI collects one fixed
transition dataset and solves the Bellman equation by repeated supervised regression, which has no
learning rate to tune, no replay buffer, no target network, and no failure mode where a run
diverges after an hour. Each iteration is a single gradient-boosted fit.

It also leaves an inspectable model. `permutation_importance` interrogates the fitted Q directly,
which is what motivates the feature-group ablation — and the ablation, which retrains from scratch
without a group and re-measures out-of-sample PnL, is the stronger evidence of the two.

### 13. Your permutation importance measures a fitted model, not the market. Isn't that circular?

Yes, and the docstring says so: a feature can matter to the model and be useless in reality. That
is exactly why the feature-group ablation exists — it retrains the policy from scratch without a
group and measures out-of-sample PnL, which is a claim about the environment rather than about the
model's internals.

The importance implementation is also not the textbook one, for a reason worth stating. Measuring
the *drop* in `max_a Q` is wrong here: `max` is convex, so injecting noise into an input tends to
raise it, and a feature the model genuinely uses can score zero or negative. On synthetic data whose
reward depended only on feature 0, that formulation ranked an irrelevant feature top. The
implementation uses mean absolute change in the whole Q matrix, which is monotone in dependence.

### 14. The simulator fails two of its own stylized facts. Doesn't that invalidate the results?

It qualifies them, and the failures are reported in the README rather than dropped.

The depth profile is monotone decreasing from the touch instead of hump-shaped. This was
re-measured by tick offset rather than by occupied level to rule out a measurement artefact; the
conclusion held.

Volatility clustering dies within about a second — the autocorrelation of |returns| is ~0.24 at lag
1 but ~0 at lags 10 and 30, where real markets show slow decay. A two-timescale Hawkes kernel was
implemented in full to fix this, tested against a *pre-registered* acceptance criterion, failed it,
and was reverted. `docs/DECISIONS.md` D8 carries the sweep table: every setting that produced
clustering collapsed the variance ratio to 0.49 or below, and a mean-reverting mid would have been
far more damaging to the results than absent long memory. The kernel was restored to its exact
pre-investigation parameters, so the excursion changed no number in the repo.

The bearing on the conclusions: neither failure is on the causal path from queue mechanics to fill
probability, which is what the headline measures. Both would matter more for a claim about
tail risk or about volatility-timing strategies, which this repo does not make.

### 15. What would change your mind, and what would you do with another month?

What would falsify the headline: real message-level LOB data. The claim is that queue-aware filling
removes most of the apparent edge, and the decisive test is to replay a real book, place synthetic
orders in the real queue, and measure the same gap. If the gap were small on real data, the result
would be an artefact of the flow model's cancellation dynamics — which is precisely why the
cancellation-position ablation is in the README.

What would most likely fix the learned policy, in order of expected value:

1. **Reward variance reduction.** The mark-to-market term is dominated by price noise that no
   action controls. A per-fill advantage formulation, or a baseline subtracted from the reward,
   attacks the actual bottleneck.
2. **More data.** 120k transitions across 16 actions is thin. This is the cheapest lever and the
   least interesting one.
3. **A smaller, better-shaped action space.** Sixteen actions over two independent sides wastes
   capacity on combinations no sensible market maker would use.
4. **Distributional or risk-sensitive RL.** The objective is genuinely risk-sensitive, and the
   inventory penalty is a crude proxy for that.

What I would not do is add capacity to the function approximator; nothing in the diagnostics
suggests the model class is the binding constraint. I would also drop the double estimator unless a
cleaner bias diagnostic showed it earning its cost — see Q10.
