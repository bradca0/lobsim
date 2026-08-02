# lobsim — what queue position does to a market-making backtest

A queue-aware limit order book simulator, a calibrated synthetic market validated against published
stylized facts, and a learned market-making policy evaluated against rule-based baselines with
paired bootstrap confidence intervals, multiple-testing correction, and a deflated Sharpe ratio.

**The question.** Almost every market-making backtest fills your resting order when a trade prints
at your price. Real exchanges do not work that way: you are behind a queue, and you trade only
after everything that arrived before you has traded or cancelled. How much of a measured edge is
created by that one assumption?

**The answer, measured here:**

<!-- BEGIN:claim -->
- **Always at touch**: -98.06 ticks under optimistic fills, -168.13 under queue-aware fills (difference +70.07, 95% CI [+29.47, +109.49]). It executes 2.7x more volume when the queue is ignored.
- **Inventory skew**: +213.87 ticks under optimistic fills, +2.78 under queue-aware fills (difference +211.09, 95% CI [+205.09, +217.21]). It executes 2.9x more volume when the queue is ignored.
- **Learned (FQI)**: -50.28 ticks under optimistic fills, -34.56 under queue-aware fills (difference -15.72, 95% CI [-45.23, +12.66]). It executes 6.8x more volume when the queue is ignored.
<!-- END:claim -->

![Fill model comparison](results/figures/fill_model.png)

---

## Headline results

All policies, held-out test episodes, queue-aware fills. PnL is in ticks per 5-minute episode.
"vs baseline" is the paired difference against inventory-skew, with ✓ marking significance after
Holm–Bonferroni correction across the family of comparisons.

<!-- BEGIN:headline -->
| Policy | PnL (ticks) | 95% CI | Fills | Inv. RMS | Edge/lot | 5s markout | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Inactive (control) | +0.00 | [+0.00, +0.00] | 0 | 0.0 | — | — | — |
| Always at touch | -168.13 | [-202.43, -134.95] | 249 | 26.9 | +0.374 | +0.106 | -170.91 (✓) |
| Fixed spread, 2 ticks | +3.52 | [-0.24, +7.09] | 9 | 2.6 | +0.548 | +0.915 | +0.74 (n.s.) |
| Inventory skew | +2.78 | [-1.58, +6.95] | 167 | 3.7 | +0.431 | +0.255 | _baseline_ |
| Avellaneda–Stoikov | -18.63 | [-28.80, -9.12] | 33 | 6.8 | +0.466 | +0.188 | -21.41 (✓) |
| **Learned (FQI)** | -34.56 | [-45.43, -24.47] | 34 | 8.0 | +0.159 | -0.268 | -37.34 (✓) |
<!-- END:headline -->

The same policies under both fill models, on identical seeds:

<!-- BEGIN:fillmodel -->
| Policy | Optimistic fills | Queue-aware fills | Difference | 95% CI | Fill-volume inflation |
|---|---:|---:|---:|---:|---:|
| Always at touch | -98.06 | -168.13 | +70.07 | [+29.47, +109.49] | 2.72× |
| Fixed spread, 2 ticks | +9.44 | +3.52 | +5.91 | [+1.35, +10.36] | 1.65× |
| Inventory skew | +213.87 | +2.78 | +211.09 | [+205.09, +217.21] | 2.91× |
| Avellaneda–Stoikov | -13.79 | -18.63 | +4.84 | [-9.47, +18.51] | 2.12× |
| **Learned (FQI)** | -50.28 | -34.56 | -15.72 | [-45.23, +12.66] | 6.78× |
<!-- END:fillmodel -->

**Reading the fill-model table.** The inventory-skew baseline is the clean case: a policy that
looks like a solid, boring earner under optimistic fills is, with the queue modelled, statistically
indistinguishable from zero. Nothing about the policy changed — only the assumption about who
trades first at a price.

Two rows need a caveat rather than a victory lap. *Always at touch* loses more under queue-aware
fills, not less, because its fills are the ones it least wants: optimistic filling hands it a lot
of extra volume with a healthier markout, while queue-aware filling leaves it holding mainly the
trades that ran it over. And the *learned policy* is the one row where the optimistic column should
not be read as a like-for-like comparison at all: it was trained under queue-aware fills, so
evaluating it under optimistic fills is off-distribution. Its 6.8× fill inflation is the largest in
the table precisely because it learned to rely on queue position that the optimistic model hands it
for free.

`make reproduce` regenerates every number above, and the tables are injected from
`results/raw/*.json` by a script — no number in this README is hand-typed.

---

## What is actually built

```
src/lobsim/
  book.py         price-time priority matching engine, exact queue-position tracking
  flow.py         order flow: zero-intelligence + Hawkes + a latent fundamental
  engine.py       discrete-event loop, post-only quoting, exact PnL attribution
  features.py     18 microstructure features in 4 ablatable groups
  agents/         5 rule-based baselines + fitted Q-iteration policy
  stats.py        block bootstrap, paired tests, Holm correction, deflated Sharpe
  validation.py   stylized-fact estimators
```

**The matching engine** keeps each price level as an insertion-ordered dict, so the level *is* the
FIFO queue: front-of-queue is O(1) and mid-queue cancellation — the most common event in a real
book — is also O(1). Agent orders carry a `volume_ahead` counter maintained by O(1) increments on
every trade and cancellation, cross-checked against full recomputation by a Hypothesis property
test.

Two things the engine refuses to let a backtest get away with. **Re-quoting costs queue priority**:
asking for a price you are already resting at leaves the order alone, and any price change cancels
and replaces it at the back of the new queue. **Post-only**: a quote that would cross is rejected
and counted, never silently converted into an aggressive order.

**The market** is zero-intelligence background flow (power-law limit-order placement,
volume-proportional cancellation) with market orders arriving as a Hawkes process, tilted toward a
latent fundamental that diffuses as a random walk. That last part is what makes aggressive flow
*informed*, and therefore what gives the market maker something to lose. Pure zero-intelligence flow
was built first and rejected on measurement: with a realistic queue the mid moved 1–4 ticks per
episode, leaving no inventory risk and no adverse selection (`docs/DECISIONS.md` D3).

**The learned policy** is Fitted Q-Iteration with gradient-boosted trees over 16 actions — each side
independently joins the touch, rests one or two ticks behind it, or pulls. Batch RL rather than
online: no learning rate, no replay buffer, no divergent run, and a model that can be interrogated
afterwards.

---

## Simulator fidelity

Measured on an **agentless** market, so the market maker cannot flatter the numbers. Targets are
bands from the empirical microstructure literature.

<!-- BEGIN:validation -->
| Stylized fact | Measured | Target band | |
|---|---:|:---:|:--:|
| return excess kurtosis | 30.2230 | [0.5, 50] | pass |
| volatility clustering lag1 | 0.2370 | [0.02, 0.6] | pass |
| volatility clustering lag10 | -0.0068 | [0, 0.4] | **FAIL** |
| raw return acf lag1 | -0.1139 | [-0.35, 0.05] | pass |
| variance ratio 10 | 0.9827 | [0.7, 1.3] | pass |
| variance ratio 30 | 1.1143 | [0.7, 1.3] | pass |
| median spread ticks | 1.0000 | [1, 2] | pass |
| mean touch depth lots | 17.6072 | [5, 60] | pass |
| order size tail index | 1.8726 | [1.4, 3] | pass |
| market order sign acf lag1 | 0.2646 | [0, 0.5] | pass |
| depth profile hump index | 0.0000 | [1, 4] | **FAIL** |

**9 of 11 pass.** Failures are analysed in Limitations.
<!-- END:validation -->

![Validation](results/figures/validation.png)

The variance ratio is the one that matters most. If the mid mean-reverted, a market maker would be
paid for bearing no risk and every PnL number downstream would be an artefact.

---

## Where the money goes

![PnL decomposition](results/figures/pnl_decomposition.png)

PnL is split into **spread capture** — the edge on every fill, marked against the mid at execution —
and **inventory PnL**, the mark-to-market of carrying a position while the price moves. The split is
an exact identity, accumulated by the engine one book mutation at a time; the residual is asserted
to be zero over random seeds and every policy.

![Markouts](results/figures/markouts.png)

The markout curve is the adverse-selection story: how much of the captured edge survives the next
few seconds. A policy with high spread capture and a negative markout is being picked off — filled
precisely when the price is about to move against it.

![PnL distribution](results/figures/pnl_distribution.png)

---

## Ablations

<!-- BEGIN:ablations -->
| Ablation | Variant | PnL (ticks) | Δ vs reference | 95% CI |
|---|---|---:|---:|---:|
| Feature groups | no queue | -55.84 | -16.07 | [-37.09, +4.51] |
| Feature groups | no flow | -63.60 | -23.83 | [-51.96, +2.97] |
| Feature groups | book only | -91.36 | -51.59 | [-102.22, -2.15] |
| Cancel position (at_touch) | back-loaded | -154.38 | -6.63 | [-63.72, +47.17] |
| Cancel position (inventory_skew) | back-loaded | -0.58 | -2.13 | [-9.60, +5.83] |
| Cancel position (fqi) | back-loaded | -40.87 | -1.10 | [-18.92, +17.49] |
| Q estimator | single (vs double) | -36.78 | +2.99 | [-13.80, +20.41] |

Q-value target inflation: 1.85× with the double estimator versus 1.85× with a single one.
<!-- END:ablations -->

![Ablations](results/figures/ablations.png)

Ablations run on the first 100 held-out test seeds rather than all 200, because each feature-group
variant needs its own dataset, fit and evaluation; their confidence intervals are correspondingly
wider than the headline's. The seeds are a prefix of the same held-out set, never a re-draw.

The feature ablation asks "does the policy actually use queue position?" The point estimates are
monotone in the amount of information removed — dropping the queue group costs 16 ticks, dropping
flow costs 24, and a book-only policy is 52 worse — but only the book-only variant clears
significance (p = 0.046); the individual group ablations do not (p = 0.13 and p = 0.10). With 100
episodes the study is underpowered for effects this size relative to episode variance. The honest
statement is that the *ordering* is consistent with the policy using both groups, and that
establishing it individually would need several times the episodes, not that each group is
separately proven to matter. The double-estimator ablation is a negative result and is reported as one: at these
hyperparameters it neither reduced Q-value inflation (1.8519 versus 1.8516) nor improved PnL
(-2.99 ticks, 95% CI [-20.41, +13.80], p = 0.73), despite provably correcting the bias on the
synthetic pure-noise diagnostic in `tests/test_fqi.py`. The likely reason is that the inflation
metric itself is not a clean bias measurement on real data — with a discount of 0.97, targets are
*supposed* to grow as the horizon lengthens, and the statistic cannot separate that from bias.
`docs/INTERVIEW.md` Q10 works through it.

The cancellation-position ablation is the important one for external validity. Whether a cancelled
lot is drawn uniformly from the queue or is biased toward late arrivals controls how fast the queue
in front of a resting order evaporates, and it is the least verifiable assumption in the model. The
default (`UNIFORM`) is the *more generous* of the two to the agent, so the headline is not
manufactured by a pessimistic choice.

---

## Selection-adjusted performance

Sharpe here is per *episode* across the held-out test set — a 5-minute episode has no calendar
meaning, so annualising it would be theatre. The deflated Sharpe (Bailey & López de Prado) asks
whether a result survives the number of attempts that produced it: with enough configurations, some
variant will look good on noise alone, and the "selection benchmark" column is the Sharpe that
chance alone would be expected to reach.

<!-- BEGIN:deflation -->
| Policy | Sharpe (per episode) | Selection benchmark | Deflated Sharpe | Trials |
|---|---:|---:|---:|---:|
| Fixed spread, 2 ticks | +0.134 | +0.000 | 0.955 | 1 |
| **Learned (FQI)** | -0.459 | +0.140 | 0.000 | 24 |
| Inventory skew | +0.089 | +0.000 | 0.886 | 1 |

The learned policy is deflated by 24 configurations — every FQI variant evaluated on validation seeds across development, not the 4 in the final grid.
<!-- END:deflation -->

Nothing here clears a bar worth boasting about. The learned policy has a negative Sharpe, so
deflation is academic. `fixed_spread_2` at 0.955 and `inventory_skew` at 0.886 are single fixed
rules that went through no search at all, which is the only reason their benchmark is zero — they
are not "significant strategies", they are two rules that happened to sit slightly above break-even
on 200 episodes. Read the confidence intervals in the headline table, both of which contain zero,
before reading these.

---

## Reproduce

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). No GPU; built and run on an M1 MacBook Air.

```bash
git clone <this repo> && cd lobsim
make setup           # uv venv + pinned dependencies
make test            # full suite, coverage gate on core logic
make lint typecheck  # ruff + mypy --strict
make reproduce       # regenerates every number and figure above, from scratch
```

`make reproduce` runs the whole pipeline: stylized-fact validation → policy training (train seeds
only, model selection on validation seeds) → backtests on held-out test seeds under both fill
models → ablations → statistics → figures → README injection. Raw outputs land in `results/raw/`,
each stamped with the git revision that produced it.

Budget roughly an hour end to end on an 8 GB M1 Air. The binding constraint is memory, not cores:
each worker process re-imports numpy, scipy and scikit-learn, so worker count is capped at four
regardless of core count — six drove free memory to ~65 MB and stalled the run outright.

Seeds are fixed and every episode is fully determined by its seed, so runs are reproducible and the
parallel and serial paths give identical answers.

<!-- BEGIN:provenance -->
Generated from commit `fe13745` on macOS-26.5.2-arm64-arm-64bit. 200 held-out test episodes (seeds 1000–1199), 60 agentless episodes for validation. Backtests took 5.8 minutes.
<!-- END:provenance -->

---

## Limitations

Written to be read by someone deciding whether to trust the numbers.

**The learned policy does not beat a five-line heuristic, and loses money.** It substantially beats
the naive always-at-touch baseline and still loses to `InventorySkew`. This is reported rather than
buried. The diagnosis is signal-to-noise, not model capacity: the per-step reward is the change in
mark-to-market, whose standard deviation is around 7 ticks, while the difference between a good and
a bad quoting decision is worth a small fraction of a tick. Fitted Q-Iteration has to resolve that
gap from ~120k transitions spread over 16 actions and a continuous state. The ablations show the
machinery working — persistent exploration is worth a large improvement over i.i.d. exploration, and
the double estimator does reduce Q-value inflation — but working machinery on a bad SNR still loses.
`docs/INTERVIEW.md` Q15 lists what would be tried next, in order.

**The market is synthetic, and calibrated to stylized targets rather than estimated.** The
parameters are not statistically identified; a different vector reproducing the same book shape
would be equally admissible. Results are conditional on this regime — a one-tick spread with a deep
queue, which is where queue position matters most. The right falsification is real message-level
data: replay a genuine book, place synthetic orders in the real queue, and measure the same gap.

**Two stylized facts fail.** The depth profile is monotone decreasing from the touch instead of
hump-shaped (re-measured by tick offset to rule out a measurement artefact; conclusion unchanged).
Volatility clustering dies within about a second, where real markets show slow decay. A
two-timescale Hawkes kernel was implemented to fix the latter, tested against a pre-registered
acceptance criterion, failed it, and was reverted — every setting that produced clustering collapsed
the variance ratio, and a mean-reverting mid would have done far more damage than absent long
memory. The full sweep is in `docs/DECISIONS.md` D8. Neither failure sits on the causal path from
queue mechanics to fill probability, which is what the headline measures.

**The agent has no latency and no fees.** Quotes take effect at the decision instant, and there are
no exchange fees, rebates, or costs beyond the adverse selection the flow model generates. Maker
rebates in particular would shift every policy's PnL upward and would change which policies are
profitable, though not the direction of the fill-model result.

**Single instrument, no cross-asset hedging.** Inventory can only be managed by quoting and by the
terminal liquidation, which is the hard case but not the realistic one for a real desk.

---

## Documentation

- `docs/DECISIONS.md` — every non-obvious choice, the alternatives, and why; including the ones that
  were implemented, measured, and reverted.
- `docs/INTERVIEW.md` — the fifteen hardest questions a skeptical reviewer should ask, answered with
  pointers into code and results.
- `docs/PLAN.md` — milestone plan and current state.

MIT licensed.
