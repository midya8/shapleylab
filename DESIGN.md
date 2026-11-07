# ShapleyLab design

## Boundary

A local explanation is a choice of score function, point and reference game.
ShapleyLab enumerates the empirical-background replacement game. It does not
learn a model or infer feature causality. The CLI evaluates only an explicitly
serialized polynomial scorer; the Python API can wrap any deterministic finite
scalar predictor. No network requests, dynamic imports or model downloads.

## Joint replacement semantics

For each mask, replace excluded coordinates using one complete background row,
then average scores over rows. This preserves dependence among missing features
but breaks fixed/missing dependence. It is not conditional imputation and does
not factor the empirical background into independent coordinate marginals.
For product score with background (0,0)/(1,1), v(empty)=0.5, not 0.25.

## Exact enumeration

For d features, store 2**d coalition values by integer bit mask. Each feature's
Shapley value averages its marginal increment using weight
`1 / (d * choose(d-1, |S|))`. An independent factorial ordering implementation in
tests checks dimensions 1-6. No Monte Carlo or regression approximation occurs.
The full mask invokes the score once instead of B times. Every other mask uses
all B rows, so calls equal `(2**d - 1)*B + 1`. The callback must be deterministic;
we do not spend extra calls trying to prove determinism. Counts are not walltime.

## Numerical behavior

Check finite coordinates and scalar outputs. Average by fsum of contributions
divided by B, avoiding an overflowing raw sum. When all background outputs are
identical, return that float unchanged, preserving equality to the single full
mask score. Attribution increments/means and the efficiency residual must remain
finite or fail. Do not renormalize attributions or allocate numerical residuals
to arbitrary features. Exact refers to coalition enumeration, not symbolic
arithmetic or unknown population attribution.

## Work limits

Reject dimensions outside 1-12, background sizes outside 1-1000, and budgets
outside 1-1000000. Default call budget 200000. Validate budget before invoking the
callback. Lists are copied where output could otherwise alias input; callbacks
receive immutable tuples. A malicious or stateful callback is outside the model
contract and cannot be made safe by an attribution algorithm.

## Counterexamples are part of the demo

Product at x=(2,3,99) yields [3,3,0] relative to zero background but [2,3,0]
relative to a background with the first two coordinates one. Prediction remains
6; baseline changes from 0 to 1. A different background asks a different question.
For squared difference with diagonal binary background and diagonal target,
coalitions [0,0.5,0.5,0] yield zero attribution and zero residual while singleton
hybrids are off the observed manifold. Passing efficiency is not manifold proof.
The sigmoid polynomial is hand-specified, not trained or accuracy-benchmarked.

## Source credit

The standard formula and masking distinction are documented at:

- https://arxiv.org/abs/1705.07874
- https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html
- https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/explain/shap.html

Independent educational implementation. No new explanation theorem, conditional
SHAP, TreeSHAP, causal identification or efficiency advantage is claimed.
