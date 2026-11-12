# babelfix: relaxed clocks with Babel's SDollo likelihood

## The problem (Babel 0.4.2, BEAST.base 2.7.8)

`babel.evolution.likelihood.ALSTreeLikelihood` delegates part of the SDollo
likelihood to an observation process (`AnyTipObservationProcess`). That object:

1. **Has its own `branchRateModel` input and silently defaults to a strict clock
   (rate 1) when it isn't set.** `ALSTreeLikelihood` doesn't pass its own branch
   rate model on, and Babel's `examples/testSDollo.xml` doesn't set it either. With a
   relaxed clock, the pattern partials use the relaxed rates, while the birth-placement
   weights and the expected number of observable traits use a strict clock. The
   model is internally inconsistent.
2. **Caches the log tree weight and invalidates it only when μ or the tree
   changes.** If (1) is fixed by passing the clock explicitly, any proposal that
   changes branch rates alone leaves a stale tree weight.

### Evidence

From milestone 1 pilot replicate `rep_000` (10 taxa, true σ = 0.24, μ = 2.0e-4/yr,
tree length 21,782 yr):

| Configuration | Result |
|---|---|
| Relaxed clock wired as in Babel's example | Posterior σ ∈ [3.5, 4.0], μ ∈ [7.9e-4, 1.6e-3], tree length ∈ [7.9k, 11.6k]; posterior ESS 7. Truth outside every HPD. |
| Relaxed clock, `branchRateModel` passed to the observation process, run with `-Dbeast.debug=true` | `Likelihood incorrectly calculated: -6583.31 != -6602.76 (19.45) Operator: UniformOperator` (the rate-quantile operator) |
| Strict clock (stock Babel) | Truth inside all 95% HPDs, ESS > 1,600; see the milestone 1 coverage study |

## The fix

`babelfix.AnyTipObservationProcessRC` subclasses `AnyTipObservationProcess`. It
requires `branchRateModel` to be set, and it clears the cached tree weight
whenever the branch rate model is dirty.

```bash
# needs a JDK 17 (e.g. `brew install openjdk@17`, then JAVA_HOME=$(brew --prefix openjdk@17))
infer/beast/babelfix/build.sh
```

Status: **compiled (JDK 17.0.21) and passes the `-Dbeast.debug=true` check** — 20,000
states on relaxed-clock replicate `rep_000` with full likelihood recomputation every
step and no discrepancies (`make beast-debug-relaxed`).

This should also be reported upstream to Babel (github.com/rbouckaert/Babel).
