# PLAN

Living document. Updated as state changes so any future session can resume from files alone.

## Goal

A queue-aware event-driven limit order book (LOB) simulator, a calibrated synthetic order-flow
generator validated against published stylized facts, and a learned market-making policy evaluated
against rule-based baselines under a statistically rigorous backtest.

The central claim the repo must support or refute with its own numbers:

> Queue position is not a second-order detail. A market-making edge measured under the standard
> optimistic fill assumption largely does not survive queue-aware fills.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Repo scaffolding: uv, pinned deps, src layout, ruff/mypy/pytest, CI, Makefile | done |
| M1 | Core matching engine: price-time priority, order lifecycle | done |
| M2 | Queue-position tracking and the two fill models (optimistic / queue-aware) | done |
| M3 | Synthetic order-flow generator (Hawkes market orders, ZI limit/cancel flow) | todo |
| M4 | Stylized-facts validation suite — simulator fidelity as a *measured* result | todo |
| M5 | Market-making agent API, rule-based baselines, backtest harness, metrics | todo |
| M6 | Learned policy: Fitted Q-Iteration with gradient-boosted trees | todo |
| M7 | Statistical evaluation: paired block bootstrap, deflated Sharpe, Holm correction | todo |
| M8 | Ablations (fill realism, feature groups), figures, `make reproduce` | todo |
| M9 | README, DECISIONS, INTERVIEW, hostile-reviewer pass | todo |

## Module map

```
src/lobsim/
  types.py        value objects: Side, OrderType, Order, Trade, BookSnapshot
  book.py         LimitOrderBook — price-time priority matching, O(1) amortised level ops
  queue_model.py  cancellation-position model; queue-ahead bookkeeping for tracked orders
  engine.py       discrete-event simulation loop, agent callback protocol
  flow.py         synthetic order-flow generator (Hawkes + zero-intelligence)
  features.py     microstructure feature extraction from BookSnapshot history
  agents/
    base.py       Agent protocol + Quote action type
    baselines.py  FixedSpread, AvellanedaStoikov, AlwaysAtTouch, Inactive
    fqi.py        Fitted Q-Iteration policy with HistGradientBoostingRegressor
  backtest.py     episode runner, seed management, per-episode records
  metrics.py      PnL decomposition, Sharpe, drawdown, markout adverse selection
  stats.py        stationary block bootstrap, paired tests, deflated Sharpe, Holm
  validation.py   stylized-fact estimators (tail index, ACF, Hurst-ish diffusivity)
  plotting.py     figures (excluded from coverage; asserted only for non-crash)
```

## Experiment protocol (frozen before results were generated)

* Seeds 0–199 are **training** episodes, seeds 1000–1199 are **test** episodes. Disjoint, and the
  learned policy never sees a test seed during fitting or model selection.
* All policies are evaluated on the *same* test seeds so every comparison is **paired**.
* One episode = 30 simulated minutes of order flow at ~exchange event rates.
* Model selection (FQI hyperparameters) uses a validation split carved out of training seeds only.
* The number of configurations tried is recorded and fed to the deflated Sharpe ratio.

## Resume notes

* `make reproduce` regenerates every number and figure in the README from scratch.
* Raw experiment outputs land in `results/raw/*.json`; figures in `results/figures/*.png`.
* Anything in the README that looks like a number is read from `results/raw/` by
  `scripts/render_readme_tables.py` — no hand-typed results.
