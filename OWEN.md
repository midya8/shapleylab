# Together changes the share

ShapleyLab 0.3 two-level Owen contract. Original exact, atomic-group, sampled and ledger APIs retain their meanings. This is established coalitional-value theory, not a new allocation algorithm.

## Same game, different allowed orders

A deterministic scalar score, target point and empirical background define the original replacement game. For coalition S, included coordinates take target values. Every missing coordinate takes its value from one complete background row; the score is then averaged over all rows. Dependence among missing coordinates is preserved, but included/missing dependence is broken. This is not conditional imputation or a causal intervention.

Ordinary individual Shapley averages marginal increments over all feature orderings. A supplied, disjoint, exhaustive partition instead allows only orderings in which every group's members stay contiguous. Uniformly order the groups and independently uniformly order members within each group. The **two-level Owen value** is each member's average marginal over those admissible complete orders. This is a two-level partition, not arbitrary nested hierarchical Partition SHAP, overlapping coalitions or an automatically inferred grouping.

For feature i in size-s group k among G groups, let R range over subsets of other whole groups, T over subsets of own-group members except i, and Q be the union of groups in R. Retain the term

```text
[v(Q union T union {i}) - v(Q union T)] /
[G * choose(G-1, |R|) * s * choose(s-1, |T|)]
```

Every before/after mask and value, outer and inner denominator, marginal and weighted marginal is retained. Each feature's coefficients sum to 1. The separate group quotient game evaluates unions of entire groups and applies ordinary group Shapley. Summed Owen members equal that group's quotient-game share, up to floating arithmetic. This is not a rule for dividing a fixed group value equally, or rescaling ordinary individual credits to force group agreement.

For a three-way product with all target coordinates1 and zero background, ordinary individual credit is [1/3,1/3,1/3]. Partition{a,b}/{c} gives atomic group shares [1/2,1/2] and Owen members [1/4,1/4,1/2]. The linear score 2a+b+3c instead gives member credits [2,1,3], not equal splitting [1.5,1.5,3]. Singleton partitions and one grand group both recover ordinary individual Shapley. A dummy member receives zero even inside a nontrivial group.

## Native APIs and declared-score reports

```python
from shapleylab.owen import allocate_owen, explain_owen
from shapleylab.owen_verify import verify_allocation
from shapleylab.owen_report import audit_owen
from shapleylab.owen_report_verify import verify_audit
```

`allocate_owen(values, feature_names, groups)` accepts a complete finite coalition-value table in integer-mask order, 1..8 features. It performs no score calls and returns the complete allocation trace. `verify_allocation` checks allocation of this **supplied table**. It cannot authenticate where those values came from.

`explain_owen(score, point, background, feature_names, groups, max_evaluations=200000)` is a callback convenience wrapper. It validates the partition before calling the unchanged original individual engine once, reuses that engine's full coalition table and retains its original individual result for a same-game comparison. Allocation adds zero score calls. Original callback assumptions, finite-value checks and budget apply. Model calls are `(2**d-1)*B+1`, not the smaller atomic-group call count. Groups do not reduce the full-game evaluations in this mode. The wrapper object has extra fields and is not the standalone allocation checker's schema. Callback authenticity is not claimed.

`audit_owen(request)` is the end-to-end declared-score route. Its exact keys are feature_names, point, background, groups and model. Report schema is `shapleylab-owen-audit-v1`. Bounds: 1..6 features, 1..32 background rows, 0..32 unique multiaffine monomials, coordinates/weights/bias in [-100,100]. Each monomial support is a nonempty sorted list of distinct valid feature indices; no repeated support. The model has exactly bias/terms/output, each term features/weight, and output raw or stable sigmoid. Names are distinct nonempty Unicode strings of at most 40 characters without unpaired surrogates. Group member order is preserved. Index tokens must be JSON integers in the native CLI and browser parser.

The report logs every actual native callback evaluation by the original engine, including raw-feature mask, background row, hybrid coordinate values, score and call index. Full target is evaluated once, not once per background row. It retains ordinary individual coefficient terms, Owen coefficient terms, quotient group terms, group sums, every complete admissible ordering, each labeled marginal step and all efficiency residuals. Every allocation uses the exact same full replacement table. Original controls are hand-written synthetic models, not learned predictors, measured agent data or empirical explanation-quality results.

## Independent replay and numerical limits

The generator computes Owen coefficients with reciprocal binomial factors. The standalone checker imports no generator, enumerates every admissible group/member ordering and reconstructs each marginal's occurrence count. The full declared-score checker independently evaluates the supplied monomial/link score at each replacement row, reconstructs background means, then uses independent ordinary permutation counts and the standalone checker for all allocation terms. It reconstructs every full admissible path and verifies the entire schema, fields, list order, metadata, counts, values and wording. Request validation and public wording constants are shared assumptions; score and allocation arithmetic are not shared generator calls.

Computed **float leaves** compare with absolute/relative tolerance 1e-12, admitting JSON integer forms of an expected float and small cross-runtime summation differences. Integer metadata require exact integer types, not bools or floats. This tolerance is disclosed and tested; it is not exact binary64 identity, an interval bound, cryptographic authentication or formal proof. A source value is not authenticated merely because its arithmetic replays. Whole-game constants shift the baseline, not member effects. No residual renormalization or arbitrary residual allocation is used.

The JavaScript summation helper accumulates supplied binary64 values as signed BigInt units of 2^-1074, then rounds once to nearest, ties to even. This original finite helper avoids double rounding when collapsing a floating expansion. It does not turn multiplication, sigmoid, score evaluation or the overall method into exact-real arithmetic.

The independent JavaScript engine computes the declared score and derives coefficient weights from complete admissible-order occurrence counts, not the native binomial formula. Actual JSON-serialized browser reports are independently accepted by the native checker. The browser is not running a Python callback. It shares only request/control descriptions and public wording, not native numerical reports.

## Checked evidence

Finite implementation checks, not a theorem or scaling benchmark:

- 72 Python tests: 49 prior + 23 new.
- 556 signed game/partition cases through 6 features, compared to independent Fraction arithmetic across 50,362 block-contiguous orders. Every partition at each dimension is included, with two signed random tables per partition.
- 14,668 nonempty-unanimity-support/partition cases through 6 features, checked against a separate closed formula. Each active member receives 1/(number of touched groups × number of active own-group members), with constant offset irrelevant.
- Independent standalone checker replay on every 556 case and 8-feature singleton/grand/mixed boundaries; generator-disabled, weight sums, group efficiency, partition/feature reordering, strict schema, preflight zero calls, finite arithmetic and isolation checks.
- 160 non-input primitive leaves individually changed and rejected in one standalone allocation; 427 non-request primitive leaves individually changed and rejected in one full report. Missing/extra fields, wrong list types, metadata bool/int/float substitutions and list order changes are checked. This does not claim every possible mutation is rejected; tiny computed-float changes within the disclosed tolerance can pass.
- 120 seeded raw/sigmoid full declared-score reports through 6 features, including background-row counts and complete end-to-end independent replay. Six-feature/32-background extremes retain 2017 calls and 720 complete paths.
- 12 original supplied controls retain 245 declared evaluations, 194 Owen terms, 114 admissible paths and 516 steps.
- 132 complete native/independent-JavaScript reports, 5045 evaluations, 3326 Owen terms, 2774 paths, 14,948 steps. All actual JSON.stringify outputs are native-verified, not only numerically compared in Node.
- 206 additional broad-magnitude real-coordinate and half-even/subnormal/cancellation requests produce native-verifiable JavaScript reports. A broader pre-release test initially found 48/200 failures from expansion-collapse double rounding, which small-integer controls did not expose. The exact binary64 summation helper fixes that issue without relaxing checker tolerance. All 206 cases now replay in both generators and the native checker. This is finite adversarial coverage, not a proof for every permitted request.
- Actual Chrome 1280/390/320 checks 36 control reports: 735 replacement rows, 582 Owen terms, 1128 ordinary terms, 174 group terms, 342 paths and 1548 steps. Every selected row is compared to native. 12 trusted completed downloads are read from disk, compared and verified by the native CLI. Additional actual Chrome custom raw/sigmoid six-feature/32-background requests retain 4034 evaluations and 1440 complete paths in total. Trusted keyboard edits clear old output; noninteger lexical index tokens and unpaired-surrogate names are rejected. Both signed-bar rules use one symmetric scale. Invalid input clears stale results and disables old downloads.
- Fresh isolated 0.3 wheel outside checkout checks 19 installed module origins, 12 new controls through both new entrypoints and 24 forged CLI rejections. All 95 old edit archives, original demo, exact/grouped/sampled/audit/ledger modes and original commands reproduce. Hosted matrix and deployment verification remain separate release gates, not inferred from local success.

## CLI and reproduction

```sh
python -m unittest discover -s tests -v
node --test tests/*.cjs
PYTHONPATH=. python examples/make_owen_controls.py
bash examples/check_owen_wheel.sh
python -m shapleylab.owen_cli request.json --output report.json
python -m shapleylab.owen_cli report.json --verify
# After installation:
shapleylab-owen request.json --output report.json
shapleylab-owen report.json --verify
python -m http.server 8000 --directory docs
# Open owen.html; original index.html remains available.
```

Standard-library Python 3.10+ runtime, Node 22+ and actual Chrome for optional cross-language/browser tests. CLI `-` means stdin or stdout. Generation requests are at most 65,536 bytes; verification input and serialized output at most 16 MiB. Duplicate JSON keys, nonfinite constants, invalid UTF8 and malformed schemas fail. Output is fully computed and size-checked before stdout or atomic replacement. Errors exit 2 with prefixed stderr, no partial stdout or traceback. These lexical limits are not a runtime/memory guarantee. Size limits can reject a report even when its request is in bounds.

The browser's explicit JSON parser rejects duplicate keys, nonfinite values, oversized UTF8 input and noninteger index tokens rather than silently applying ordinary JSON.parse behavior. Edits clear all old results and exports before inspecting again. A completed download has time to read its Blob URL before revocation. Core summaries and comparisons fit narrow widths; exact trace tables intentionally scroll horizontally with a cue. Values displayed in the UI are rounded, full values remain in JSON.

## Source ledger

Sources inspected for mathematics and interpretation, not copied implementations:

- Gimenez and Puente, *A New Procedure to Calculate the Owen Value*, 2017, Section 2.2.1 standard definition and quotient property: https://www.scitepress.org/PublishedPapers/2017/61137/61137.pdf . This implementation uses the standard definition, not that paper's new multilinear-extension procedure.
- Rozemberczki etal., *The Shapley Value in Machine Learning*, 2022 survey, Section 5.2.1 contiguous-group permutations and explicit-game cautions: https://arxiv.org/pdf/2202.05594 . Direct IJCAI PDF/DOI fetches returned no content; the arxiv paper was read.
- Official SHAP PartitionExplainer documentation, hierarchical Owen distinction and group-credit semantics: https://shap.readthedocs.io/en/stable/generated/shap.PartitionExplainer.html . No SHAP code, arbitrary hierarchy, quadratic-runtime or dependence-recovery claim adopted.
- University-hosted *Owen-type value for games with two-level communication structures* slides, quotient/within-group construction and group efficiency: https://people.unipmn.it/fragnelli/dispense/si110131.pdf . Communication-restricted extensions are not implemented.
- VanKerm, *Shapley value calculations: Implementation and illustrations*, 2025 slides, recursive group-additivity and algorithm context: https://www.stata.com/meeting/switzerland25/slides/Switzerland25_Van_Kerm.pdf . No Stata code or sampling implementation adopted.
- Au etal., *Grouped feature importance and combined features effect plot*, 2022, distinction from post-summed feature values and higher-order interactions: https://link.springer.com/article/10.1007/s10618-022-00840-5 . Its performance game, data and empirical findings are not reproduced.

A third-party variance-game documentation page was also inspected: https://frbennett.github.io/shapleyx/explanation/owen-effects/ . Its displayed inner formula has(K-1)! where the standard definition's outer factor requires K!, giving total coefficient weight K rather than 1. That expression was excluded; no claim about its software implementation was tested. Source discrepancies were not silently merged.

## Boundaries

A supplied partition changes the allocation convention, not the score or the validity of its reference game. Owen member decomposition can still evaluate off-reference within-block hybrids. A zero result in the diagonal binary-disagreement control hides nonzero off-diagonal hybrids. Efficiency, group agreement and code replay cannot validate a background, discover meaningful groups, establish conditional/causal effects, guarantee manifold-valid samples, prove model quality or certify explanation usefulness. No paid API, telemetry, model download, hosted inference or private data. MIT educational implementation of established methods.
