"""Naive independent permutation Monte Carlo for the fixed replacement game."""
import math
import random
from .core import _finite, _sum


def explain_sampled(score, point, background, feature_names, permutations=256,
                    seed=0, max_evaluations=200000):
    """Sample feature orderings; integrate each visited coalition over all rows.

    MC standard errors describe permutation randomness for the fixed callback,
    point and background only. They are not confidence or accuracy guarantees.
    No early stopping or antithetic pairing is used.
    """
    if not callable(score):
        raise ValueError('score must be callable')
    if not isinstance(feature_names, list) or not 1 <= len(feature_names) <= 64:
        raise ValueError('sampled mode requires 1 through 64 feature names')
    if any(not isinstance(v, str) or not v.strip() for v in feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('feature names must be distinct nonempty strings')
    d = len(feature_names)
    def vector(row):
        return isinstance(row, (list, tuple)) and len(row) == d and all(_finite(v) for v in row)
    if not vector(point):
        raise ValueError('point must be a finite feature vector')
    if not isinstance(background, list) or not 1 <= len(background) <= 1000 or any(not vector(r) for r in background):
        raise ValueError('background requires 1 through 1000 finite aligned rows')
    if type(permutations) is not int or not 2 <= permutations <= 10000:
        raise ValueError('permutations must be an integer in [2, 10000]')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 1000000:
        raise ValueError('max_evaluations must be an integer in [1, 1000000]')
    n = len(background); full = (1 << d) - 1
    max_coalitions = min(1 << d, 2 + permutations * (d - 1))
    upper_calls = (max_coalitions - 1) * n + 1
    if upper_calls > max_evaluations:
        raise ValueError('permutation-mode upper work bound exceeds model-evaluation budget')
    x = tuple(point); rows = tuple(tuple(r) for r in background)
    cache = {}; calls = 0
    def invoke(row):
        nonlocal calls
        value = score(tuple(row)); calls += 1
        if not _finite(value):
            raise ValueError('score returned a nonfinite or nonscalar value')
        return value
    def value(mask):
        if mask not in cache:
            if mask == full:
                cache[mask] = invoke(x)
            else:
                outputs = [invoke([x[j] if mask & (1 << j) else row[j]
                                   for j in range(d)]) for row in rows]
                cache[mask] = (outputs[0] if all(v == outputs[0] for v in outputs)
                               else _sum(v / n for v in outputs))
        return cache[mask]
    baseline = value(0); prediction = value(full)
    columns = [[] for _ in range(d)]; rng = random.Random(seed)
    for _ in range(permutations):
        order = list(range(d)); rng.shuffle(order)
        mask = 0; previous = baseline
        for j in order:
            mask |= 1 << j; current = value(mask)
            delta = current - previous
            if not math.isfinite(delta):
                raise ValueError('coalition difference exceeds finite float range')
            columns[j].append(delta); previous = current
    phis = [_sum(v / permutations for v in column) for column in columns]
    ses = []
    for mean, column in zip(phis, columns):
        scale = max(abs(v) for v in column)
        variance = (_sum((v / scale - mean / scale)**2 for v in column) /
                    (permutations - 1)) if scale else 0.
        se = scale * math.sqrt(variance / permutations)
        if not math.isfinite(se):
            raise ValueError('Monte Carlo standard error exceeds finite float range')
        ses.append(se)
    total = _sum(phis)
    return {'method': 'independent_permutation_monte_carlo',
            'feature_names': list(feature_names), 'attributions': phis,
            'monte_carlo_standard_errors': ses,
            'baseline': baseline, 'prediction': prediction, 'attribution_sum': total,
            'efficiency_residual': _sum([baseline, total, -prediction]),
            'permutations': permutations, 'seed': seed,
            'background_rows': n, 'cached_coalitions': len(cache),
            'model_evaluations': calls, 'preflight_upper_model_evaluations': upper_calls,
            'game': 'empirical_background_replacement',
            'scope': 'Permutation approximation for the fixed finite-background replacement game. '
                     'Each visited coalition is integrated over all supplied background rows. '
                     'Standard errors are sample SD of per-ordering increments divided by sqrt(P), '
                     'for permutation randomness only. Not confidence bounds, background/model uncertainty, '
                     'conditional or causal validity. Requires a deterministic side-effect-free score callback. '
                     'Zero observed MC error can miss a nonzero exact value '
                     'at small P. Efficiency can hold despite inaccurate allocations. No early stopping, '
                     'antithetic pairing, feature groups or convergence certificate.'}
