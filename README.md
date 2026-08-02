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
<!-- END:claim -->

![Fill model comparison](results/figures/fill_model.png)

---

## Headline results

All policies, held-out test episodes, queue-aware fills. PnL is in ticks per 5-minute episode.
"vs baseline" is the paired difference against inventory-skew, with ✓ marking significance after
Holm–Bonferroni correction across the family of comparisons.

<!-- BEGIN:headline -->
<!-- END:headline -->

The same policies under both fill models, on identical seeds:

<!-- BEGIN:fillmodel -->
<!-- END:fillmodel -->

`make reproduce` regenerates every number above, and the tables are injected from
`results/raw/*.json` by a script — no number in this README is hand-typed.

---

## What is actually built

```
src/lobsim/
  book.py         price-time priority matching engine, exact queue-position tracking
  flow.py         order flow: zero-intelligence + Hawkes + a latent fundamental
  engine.py       discrete-event loop, post-only quoting, exact PnL attribution
  features.py     22 microstructure features in 4 ablatable groups
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
<!-- END:ablations -->

![Ablations](results/figures/ablations.png)

The cancellation-position ablation is the important one for external validity. Whether a cancelled
lot is drawn uniformly from the queue or is biased toward late arrivals controls how fast the queue
in front of a resting order evaporates, and it is the least verifiable assumption in the model. The
default (`UNIFORM`) is the *more generous* of the two to the agent, so the headline is not
manufactured by a pessimistic choice.

---

## Selection-adjusted performance

<!-- BEGIN:deflation -->
<!-- END:deflation -->

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

Seeds are fixed and every episode is fully determined by its seed, so runs are reproducible and the
parallel and serial paths give identical answers.

<!-- BEGIN:provenance -->
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
