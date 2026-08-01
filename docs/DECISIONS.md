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
