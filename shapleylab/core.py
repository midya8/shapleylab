"""Exact Shapley values for an explicit empirical-background replacement game."""
import math


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def _sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('arithmetic exceeds finite float range') from exc
    if not math.isfinite(value):
        raise ValueError('arithmetic exceeds finite float range')
    return value


def explain(score, point, background, feature_names, max_evaluations=200000):
    """Enumerate every feature coalition, with no sampling approximation.

    v(S) averages score(point_S, background_row_notS) across complete background
    rows. Dependencies among missing features are preserved within those rows;
    dependence between fixed and missing features is broken. This is not a
    conditional expectation given point_S or a causal-effect estimator.
    score must be a deterministic, side-effect-free scalar function.
    """
    if not callable(score):
        raise ValueError('score must be callable')
    if not isinstance(feature_names, list) or not 1 <= len(feature_names) <= 12:
        raise ValueError('feature_names must be a list with 1 through 12 names')
    if any(not isinstance(v, str) or not v.strip() for v in feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('feature names must be distinct nonempty strings')
    d = len(feature_names)
    def vector(row):
        return isinstance(row, (list, tuple)) and len(row) == d and all(_finite(v) for v in row)
    if not vector(point):
        raise ValueError('point must be a finite feature vector')
    if not isinstance(background, list) or not 1 <= len(background) <= 1000 or any(not vector(r) for r in background):
        raise ValueError('background requires 1 through 1000 finite aligned rows')
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 1000000:
        raise ValueError('max_evaluations must be an integer in [1, 1000000]')
    n = len(background); full = (1 << d) - 1
    planned = full * n + 1
    if planned > max_evaluations:
        raise ValueError('exact enumeration exceeds the model-evaluation budget')
    calls = 0
    def invoke(row):
        nonlocal calls
        value = score(tuple(row))
        calls += 1
        if not _finite(value):
            raise ValueError('score returned a nonfinite or nonscalar value')
        return value
    values = []
    for mask in range(full):
        outputs = [invoke([point[j] if mask & (1 << j) else row[j]
                           for j in range(d)]) for row in background]
        # Preserve a constant output exactly, including v(full) equivalence.
        values.append(outputs[0] if all(v == outputs[0] for v in outputs)
                      else _sum(v / n for v in outputs))
    values.append(invoke(point))  # Every background row gives this same hybrid.
    phis = []
    for j in range(d):
        terms = []
        for mask in range(full + 1):
            if not mask & (1 << j):
                k = mask.bit_count()
                delta = values[mask | (1 << j)] - values[mask]
                if not math.isfinite(delta):
                    raise ValueError('coalition difference exceeds finite float range')
                terms.append(delta / (d * math.comb(d - 1, k)))
        phis.append(_sum(terms))
    total = _sum(phis)
    residual = _sum([values[0], total, -values[-1]])
    return {'feature_names': list(feature_names), 'attributions': phis,
            'baseline': values[0], 'prediction': values[-1],
            'attribution_sum': total, 'efficiency_residual': residual,
            'coalition_values': values,
            'coalition_index': 'integer mask, bit j includes feature_names[j]',
            'background_rows': n, 'coalitions': full + 1,
            'model_evaluations': calls,
            'game': 'empirical_background_replacement',
            'scope': 'Exact enumeration of this finite-background game, up to floating-point arithmetic. '
                     'Not conditional SHAP, a confidence interval, causal attribution, data-manifold validity '
                     'or proof of model quality. Hybrid rows can be out of distribution. Background choice '
                     'and score units/link change the explanation. Callback determinism is required but not verified.'}
