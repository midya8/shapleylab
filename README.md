# ShapleyLab

Exact small-feature Shapley attribution from scratch, with every coalition
available for audit. Dependency-free Python 3.10+, scalar callback API, strict
JSON CLI and 43 tests. No model training or paid APIs. Exact mode enumerates all coalitions; an
optional, separately labeled permutation mode handles larger individual games.

## Exact arithmetic is not a causal explanation

```sh
python -m shapleylab demo
bash examples/demo.sh
```

| Constructed fixture | Baseline | Prediction | Attributions | Score calls |
|---|---:|---:|---|---:|
| Product, zero background | 0 | 6 | [3, 3, 0] | 15 |
| Same product/point, background shifted to ones | 1 | 6 | [2, 3, 0] | 15 |
| Product, jointly correlated binary background | 0.5 | 1 | [0.25, 0.25] | 7 |
| Squared difference, on-diagonal background/point | 0 | 0 | [0, 0] | 7 |
| Sigmoid polynomial, three background rows | 0.4036271 | 0.8354835 | [0.4454752, -0.0136188, 0] | 22 |

All rows are hand-constructed development fixtures, not trained-model accuracy,
explanation usefulness or causal benchmarks. The same product prediction changes
attribution when its reference background changes. The squared-difference game
has coalition values `[0, 0.5, 0.5, 0]`: singleton hybrid rows leave the observed
`a=b` manifold even though the final attributions and efficiency residual are
zero. Additivity is an arithmetic check, not validation of those hybrids.

## The game is explicit

For a target point `x`, background rows `b` and included-feature set `S`:

`v(S) = mean_b score(x_S, b_notS)`

Missing features come from the **same background row**, preserving their joint
dependence with each other. Their dependence on the fixed features is broken.
This is empirical-background replacement, often called marginal/interventional
masking. It does not estimate the conditional distribution given `x_S`, infer a
causal graph, or justify real-world interventions.

`phi[j] = sum_{S not containing j} (v(S union {j}) - v(S)) / (d * choose(d-1, |S|))`

Every coalition is enumerated. Exact means exact for this chosen finite game,
up to floating-point arithmetic, not exact for an unknown population or a
conditional/causal explanation. Background selection is part of the question.
Raw-score versus probability-space attribution are different games: explain the
actual chosen score, do not sigmoid individual raw-score attributions.

## Explain your own model

```python
from shapleylab import explain

result = explain(
    score=lambda row: row[0] * row[1],
    point=[2, 3, 99],
    background=[[0, 0, 0], [0, 0, 9]],
    feature_names=["a", "b", "unused"],
)
assert result["attributions"] == [3.0, 3.0, 0.0]
```

Pass a deterministic, side-effect-free scalar score callback, including a wrapper
around a fitted model. It receives immutable tuples, not your mutable input rows.
Callback determinism and model provenance are caller obligations, not verified.
No external Python code is loaded by the CLI.

```sh
python -m shapleylab explain --input examples/product.json
python -m shapleylab explain --input examples/product.json --max-evaluations 14
# Second command rejects: 15 calls are required, before any model call/output.
```

The JSON CLI requires `feature_names`, `point`, `background`, `model`,
plus optionally `groups` and no other fields.
Its built-in model is deliberately small: `bias`, aligned `linear` coefficients,
unique pair terms `{i, j, weight}` with `0 <= i < j < d`, and `output` equal to
`raw` or `sigmoid`. These are supplied coefficients, not a fitted or validated
model. The callback API is not limited to polynomials. `--input -` reads stdin.
Invalid input exits 2, writes stderr and produces no partial JSON report.

## Atomic feature groups

Optional `groups` are an explicit, disjoint, exhaustive partition of raw feature
indices. Every feature in a group switches together between target and the same
background row. `explain_groups` applies the ordinary Shapley formula to these
groups as atomic players; it does not compute individual values then add them.

```python
from shapleylab import explain_groups

result = explain_groups(
    lambda row: row[0] * row[1] * row[2],
    point=[1, 1, 1], background=[[0, 0, 0]],
    feature_names=["a", "b", "c"],
    groups=[{"name": "ab", "features": [0, 1]},
            {"name": "c", "features": [2]}],
)
assert result["group_names"] == ["ab", "c"]
assert result["attributions"] == [0.5, 0.5]
```

The ungrouped three-way product gives `[1/3, 1/3, 1/3]`. Summing the first two
individual values gives 2/3, **not** the grouped value 1/2. Grouping changes the
coalition game and the question; no arbitrary credit split back to members is
reported. This mode is not an Owen-value allocation within groups.

The same optional `groups` key is accepted by the JSON CLI:

```sh
python -m shapleylab explain --input examples/grouped.json
```

In group mode, `attributions` align with `group_names`, not `feature_names`.
`feature_names` retains the raw coordinate order; `group_feature_indices` records
the partition; `attribution_unit` explicitly identifies atomic groups. Coalition
mask bit j includes all members of group j. Names must be distinct, groups
nonempty, indices valid integers, with every feature covered exactly once.

Up to 64 raw features and 12 groups, still at most 1000 background rows and the
same call budget. Replace d by group count G in `(2**G - 1)*B + 1`. The three-way
fixture uses 4 calls instead of 8 because its game has 2 players instead of 3;
this is not an equal-question speed benchmark. A tested 64-feature/eight-group
sum uses 256 calls with one background row, returning eight values of 8.

Grouping a known correlated block prevents hybrids that split that block. It
does not preserve dependencies across different groups, infer groups for you,
make reference rows representative, or establish conditional/causal validity.
Singleton groups recover the original individual-feature game; one group
receives prediction minus baseline. Group ordering changes output order only.

Source for the grouping concept and distinction from post-grouped sums:
[Jullum, Redelmeier and Aas, groupShapley (2021)](https://ar5iv.labs.arxiv.org/html/2106.12228).
ShapleyLab retains its declared empirical replacement contribution function;
it does not implement that paper's conditional-distribution estimation or claim
new grouping theory.

## Optional permutation approximation

Exact mode remains the default and is preferable for small games. For up to
64 individual features, `explain_sampled` instead samples independent uniform
feature orderings with an isolated seeded RNG. For each ordering, it adds every
feature in sequence and records that feature's marginal score increment. The
mean increment is its estimated attribution. Every visited coalition is still
averaged over **all** background rows; backgrounds are not subsampled.

```python
from shapleylab import explain_sampled

result = explain_sampled(
    lambda row: row[0] * row[1] * row[2],
    point=[1, 1, 1], background=[[0, 0, 0]],
    feature_names=["a", "b", "c"], permutations=2, seed=0,
)
assert result["attributions"] == [0.5, 0.5, 0.0]
assert result["monte_carlo_standard_errors"][2] == 0.0
# Exact third attribution is 1/3, despite zero observed Monte Carlo error.
```

```sh
python -m shapleylab explain --input examples/product.json --permutations 64 --seed 0
```

Sampled output has `method=independent_permutation_monte_carlo`, P/seed, per-feature
MC standard errors, cached coalition count, actual score calls and a preflight
upper call bound. No complete coalition-value array is reported in sampled mode.
Sampled grouped mode is not implemented and is rejected, not silently ignored.
P must be 2-10000 (default API 256), seed an integer in [0, 2**32). Exact CLI mode
rejects a nondefault seed without `--permutations`.

The diagnostic is sample SD of per-permutation increments divided by sqrt(P).
It describes permutation randomness for this **fixed** model/point/background,
not model-training uncertainty, background representativeness or confidence in
feature importance. No confidence intervals, adaptive stopping, antithetic
pairing or convergence certificate. Zero observed MC error at small P can miss
a nonzero exact attribution, as the executable example shows. Each ordering
telescopes to prediction minus baseline, so efficiency can pass while individual
allocations remain wrong.

On the three-way product, P=2/seed 0 gives `[0.5, 0.5, 0]`, standard errors
`[0.5, 0.5, 0]`, six calls. P=4096/seed 0 gives
`[0.3220215, 0.3415527, 0.3364258]`, standard errors about
`[0.0073017, 0.0074108, 0.0073835]`, eight calls with caching. Exact is
`[1/3, 1/3, 1/3]`. These are fixed development examples, not a guarantee that
more samples always improve a realized estimate. Even when all eight coalition
values enter the cache, sampled output still averages the drawn increments;
it does not silently switch to the exact coalition-weight formula. Prefer exact
mode when that game fits its budget.

For d features and B rows, at most `min(2**d, 2 + P*(d-1))` unique masks are visited.
The preflight upper call bound is `(max_masks - 1)*B + 1`, accounting for the
single full-mask call. A bound exceeding the budget rejects before any callback;
this can be conservative because orderings can share masks. A tested 64-feature
sum with P=32/seed 4 uses 2002 calls (upper bound 2018), producing 64 unit
attributions and zero MC errors since every linear marginal is constant. This
is a call count for a constructed score, not a walltime or quality benchmark.

Standard permutation-sampling idea:
[Captum Shapley Value Sampling documentation](https://captum.ai/api/shapley_value_sampling.html).
Independent educational implementation, no Captum dependency. This mode is not
SHAP's forward/reverse antithetic PermutationExplainer or a new estimator.

## Audit and limits

- Output contains baseline, prediction, each attribution, their sum, the signed
  efficiency residual and the complete coalition-value array. Integer mask bit
  `j` means feature `feature_names[j]` is included. Nothing is silently adjusted
  to force an efficiency residual of zero.
- For `d` features and `B` background rows: `2**d` coalitions and exactly
  `(2**d - 1) * B + 1` model calls. The full coalition needs one call because it
  is the same point for every background row. This counts calls, not latency.
- Hard input limits: 1-12 individual features (or 1-64 raw features partitioned
  into 1-12 atomic groups), 1-1000 background rows. Sampled individual mode permits up to 64 features. Default budget 200000
  calls, configurable up to 1000000. A rejected budget makes no model calls.
  Exponential enumeration is for small explanations, not a TreeSHAP replacement.
- Finite aligned numeric inputs only, no automatic normalization, optional explicit grouping but no conditional imputation. Nonfinite outputs/arithmetic fail rather
  than emit NaN/Infinity JSON. Constant background outputs retain their exact
  float value; other means use scaled `fsum` to avoid raw-sum overflow.
- Hybrid inputs can be unrealistic, especially with correlated features.
  Attribution signs/magnitudes depend on the reference and score scale. No
  uncertainty interval, explanation faithfulness guarantee, causal effect,
  fairness verdict, model-quality certificate or deployment recommendation.

## Verification

```sh
python -m unittest discover -s tests -v
```

43 tests cover product symmetry/dummy/efficiency, linear closed form, linearity,
feature permutation equivariance, independent factorial permutation references
for dimensions 1-6, correlated joint background, off-manifold hybrids, background
sensitivity, full coalition audit, exact call counts, early budget rejection,
finite extremes, immutable inputs, model schema and CLI failures. Group tests cover singleton equivalence,
whole-block masking, strict partition validation, group ordering, independent
group-permutation references, no aliases and 64-feature grouped work counts.
Sampling tests check seeded ordering means/standard errors, caching/work bounds,
RNG isolation, 64-feature linear results and the zero-MC-error counterexample. CI tests Python
3.10-3.13 and runs the offline demo with an intentionally rejected budget gate.

## Sources and credit

Standard Shapley formula and empirical-background masking, not new theory:

- [Lundberg and Lee, A Unified Approach to Interpreting Model Predictions (2017)](https://arxiv.org/abs/1705.07874)
- [SHAP LinearExplainer documentation: background masking and feature dependence](https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html)
- [PiML documentation: exact coalition formula and masking choices](https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/explain/shap.html)

This is an independent educational implementation, not affiliated with SHAP or
PiML and not their optimized algorithms. MIT license.
