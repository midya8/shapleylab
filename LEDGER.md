# The replacement game before the credit

ShapleyLab v0.2 adds a bounded transparent ledger and independent browser inspector. All original exact, grouped and sampled demo reports, coefficients, points, backgrounds and counterexamples remain unchanged. Original core/grouped/sampled/model/demo implementations are not edited. No trained predictor, accuracy improvement, explanation-quality or causal claim is introduced.

## The chosen question

For atomic player set S, each included raw feature comes from target x. Every missing feature comes from the same background row b. The contribution function is the arithmetic mean of score(x_S, b_notS) over the complete finite reference set. Missing features retain their joint relation with each other, not with fixed features. This is neither a conditional expectation given x_S nor evidence that such hybrids are realistic or causally meaningful.

The original product at x=(2,3,99) predicts 6 under both reference sets. Zero reference gives credits[3,3,0], while reference ones give[2,3,0] with baseline 1. The background changes the allocation question without changing the prediction. The diagonal squared-difference example has coalition values[0,0.5,0.5,0] and zero final credits. Its singleton hybrid(1,0) violates the supplied fixture relation a=b. Zero efficiency residual does not validate the hybrid. The UI checks only that declared relation for that named example; it is not a general manifold detector.

For grouped games, every member of a group switches together. The three-way product gives individual values[1/3,1/3,1/3] but atomic groups ab/c get[1/2,1/2]. Summing a and b is not the group value. No within-group credit split, Owen allocation, group discovery or preservation of dependencies across different groups is implemented.

## Where each weight comes from

A predecessor coalition S of size k appears before player j in exactly k! (d-k-1)! of d! complete orderings. Its marginal difference gets weight 1/[d * choose(d-1,k)]. The ledger stores that denominator and ordering multiplicity with both coalition masks, the difference and weighted credit. Each player's weights sum to 1. Weighted terms sum to the original exact result; nothing adjusts the residual to force zero.

The path view enumerates all small orderings, reusing cached coalition values. Each path's increment sum telescopes from the same baseline to the target score. Averaging all d! paths recovers the coalition-weight formula up to floating rounding. A single path is not the average. The original P=2/seed0 sampling example remains static: credits[0.5,0.5,0] with estimated error 0 for c, despite exact c=1/3. Every sampled path can satisfy efficiency while its allocation is wrong. This inspector does not rerun sampling, promise monotonic Monte Carlo improvement or add a convergence certificate.

## Actual calls versus a browser plan

Native `ledger.audit` wraps the existing exact callback engine and records each actual immutable tuple argument and returned score. It then associates those calls with coalition masks and complete background rows. The full coalition is evaluated once because every reference gives the same target. All later marginal and path computations use cached values: zero additional score calls. Native call count remains (2^d-1)*B+1, not d! times a scorer cost. It is not a walltime claim.

Ledger limits are1-6 atomic players and1-32 background rows, below the original engine's separate bounds, to bound retained rows and factorial paths. Atomic groups may still span up to64 raw coordinates under original validation. Invalid groups, inputs and budgets reject before a score callback. Callback side effects, determinism and provenance remain the caller's responsibility; this is not a callback monitor or sandbox. Nonfinite callback/arithmetic values reject, with no partial JSON result.

The independent browser implements only seven fixed public synthetic score definitions: the original five exact fixtures and individual/grouped three-way products. It evaluates the declared plan itself, not Python callbacks or your model. Native actual call indexes therefore correspond to browser plan indexes, not proof of a browser-to-native invocation. Live point editing is restricted to finite [-100,100] coordinates; backgrounds, coefficients, grouping and score units stay fixed. Invalid values clear all derived rows and stale exports. No upload or dynamic code evaluation.

## Numerical and verification boundary

Native results retain original `math.fsum`, scaled means and model validation. Browser means and weighted/path sums use a compensated floating sum, with a separate stable sigmoid implementation. This is not exact-real arithmetic or a proof for arbitrary extreme callback outputs. A JSON witness records a calculation and public synthetic inputs, not source authenticity.

- Seven native baselines plus95 coordinate edits, including no-op/original values, retain all hybrid rows, scores, coalitions, marginal terms and paths. Original demo includes both sample controls unchanged.
- Independent Fraction arithmetic checks128 generated small callback games with1-5players. It averages rational representations of each binary64 score output over complete background rows and all player orderings. This checks allocation arithmetic conditional on those represented outputs, not exact-real callback evaluation.
- A separate BigInt rational browser-test oracle checks64 integer product games with a dummy player and two complete reference rows. It does not claim general rational sigmoid arithmetic.
- Independent JavaScript matches every shared required numerical/schema field of the native baselines and95edit ledgers at2e-10*(1+abs(reference)); only descriptive scope strings differ. All keys/shapes and nonnumeric values are checked, not merely fields that happen to exist.
- Actual Chrome controls replay95 edits,44original coalition masks,18players and30 complete ordering selections. It checks visible path increments, hybrid rows, rounded target scores, both adverse examples, physical numeric-input keyboard edits, invalidation, clipboard denial/success, JSON Blob downloads, touch sizes and expanded320/390/1280widths.
- Archive reproduction requires exact keys, shapes, types/nonfloat values; floats allow1e-12absolute/relative rounding across platforms. Tests reject missing fields, shape changes,1e-8 perturbations and nonfinite values. No original tolerance or test is weakened.
- Installed reproduction imports all11modules, executes all4commands, checks original exact/grouped/sampled CLI modes and rebuilds every archive outside checkout. Importing `__main__` no longer executes a CLI exit. Duplicate JSON keys reject rather than silently replacing values.

## Reproduce

```sh
python -m shapleylab.ledger_export
python -m shapleylab ledger-demo
python -m shapleylab audit --input examples/product.json
python -m unittest discover -s tests -v
node --test tests/*.cjs
bash examples/demo.sh
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/shapleylab-0.2.0-py3-none-any.whl
bash examples/verify_wheel.sh /tmp/installed
```

## Sources and differences

- [Sundararajan and Najmi, The Many Shapley Values for Model Explanation,2019/2020](https://arxiv.org/pdf/1908.08474v2.pdf), sections2-3: set-function choice and equivalent uniform-ordering description. We declare one empirical replacement game, not a preferred universal explanation semantics.
- [Jullum, Redelmeier and Aas, groupShapley,2021](https://ceur-ws.org/Vol-3014/paper3.pdf), sections2-3: groups as atomic players and distinction from post-summed individual credits/Owen allocations. We do not implement conditional-density estimation or the paper's group selection procedures.
- [David Rosenberg, Shapley Values lecture,2021](https://davidrosenberg.github.io/ttml2021fall/interpretable-ml/3.shapley-values.pdf), permutation formula and conditional/marginal distinction. No causal interpretation is inferred from notation.
- [SHAP LinearExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html), masking/background averaging and dependence choices. We do not call SHAP, implement its correlation transform or attribute its optimized algorithms to this project.
- [Captum Shapley Value Sampling documentation](https://captum.ai/api/shapley_value_sampling.html), permutation increments and exact versus sampled modes. This ledger caches2^dcoalitions rather than evaluating each ordering again. No speed benchmark or Captum dependency. Captum's within-mask presentation convention is not an Owen within-group allocation here.
- [Christoph Molnar, Interpretable Machine Learning, Shapley Values chapter](https://christophm.github.io/interpretable-ml-book/shapley.html), weighted coalition differences, background replacement and efficiency. Arithmetic axioms are not proof of realistic hybrids, fairness or model quality.

Primary papers, lecture notes, software documentation and textbook were read. Established theory, independent educational code. MIT; attribution is not endorsement.
