"""Bounded transparent audit of the existing finite replacement game."""
import itertools
import math
from .core import explain, _sum
from .grouped import explain_groups


def audit(score, point, background, feature_names, groups=None, max_evaluations=200000):
    """Log the existing exact engine's actual calls, then inspect cached values.

    At most six atomic players and32background rows. Path arithmetic performs
    no additional score calls. Callback determinism remains the caller's duty.
    """
    if not callable(score):
        raise ValueError('score must be callable')
    players = groups if groups is not None else feature_names
    if not isinstance(players, list) or not 1 <= len(players) <= 6:
        raise ValueError('ledger requires1-6atomic players')
    if not isinstance(background, list) or not 1 <= len(background) <= 32:
        raise ValueError('ledger requires1-32background rows')
    calls = []
    def logged(row):
        value = score(row)
        calls.append({'row': list(row), 'score': value})
        return value
    if groups is None:
        result = explain(logged, point, background, feature_names, max_evaluations)
        names = list(feature_names)
        members = [[j] for j in range(len(feature_names))]
    else:
        result = explain_groups(logged, point, background, feature_names, groups, max_evaluations)
        names = result['group_names'][:]
        members = [g[:] for g in result['group_feature_indices']]
    d = len(names); n = len(background); full = (1 << d) - 1
    assert len(calls) == result['model_evaluations'] == full*n+1
    values = result['coalition_values']
    coalitions = []
    cursor = 0
    for mask in range(full+1):
        included = sorted(j for g, indices in enumerate(members) if mask & (1 << g) for j in indices)
        rows = []
        # The full coalition is evaluated once, not once per background row.
        count = 1 if mask == full else n
        for k in range(count):
            call = calls[cursor]; cursor += 1
            expected = [point[j] if j in included else background[k][j] for j in range(len(feature_names))]
            assert call['row'] == expected
            rows.append({'background_index': None if mask == full else k,
                         'call_index': cursor-1, **call})
        coalitions.append({'mask': mask, 'players': [names[j] for j in range(d) if mask & (1 << j)],
                           'included_features': included, 'value': values[mask], 'evaluated_rows': rows})
    marginals = []
    for j in range(d):
        terms = []
        for mask in range(full+1):
            if mask & (1 << j):
                continue
            k = mask.bit_count(); denominator = d*math.comb(d-1, k)
            delta = values[mask | (1 << j)] - values[mask]
            terms.append({'from_mask': mask, 'to_mask': mask | (1 << j), 'predecessor_count': k,
                          'weight_numerator': 1, 'weight_denominator': denominator,
                          'permutation_multiplicity': math.factorial(k)*math.factorial(d-k-1),
                          'delta': delta, 'weighted_delta': delta/denominator})
        marginals.append({'player': names[j], 'player_index': j, 'terms': terms,
                          'sum': _sum(t['weighted_delta'] for t in terms)})
    paths = []
    for order in itertools.permutations(range(d)):
        mask = 0; increments = [0.]*d; steps = []
        for j in order:
            nxt = mask | (1 << j); delta = values[nxt]-values[mask]
            increments[j] = delta
            steps.append({'player_index': j, 'from_mask': mask, 'to_mask': nxt, 'delta': delta})
            mask = nxt
        total = _sum(increments)
        paths.append({'order': list(order), 'steps': steps, 'increments': increments,
                      'sum': total, 'efficiency_residual': _sum([values[0],total,-values[-1]])})
    path_means = [_sum(p['increments'][j]/len(paths) for p in paths) for j in range(d)]
    return {'version':1, 'feature_names':list(feature_names), 'point':list(point),
            'background':[list(row) for row in background], 'player_names':names,
            'player_feature_indices':members, 'original_exact_result':result,
            'coalitions':coalitions, 'marginals':marginals, 'permutation_paths':paths,
            'permutation_mean_attributions':path_means,
            'permutation_mean_discrepancy':max(abs(a-b) for a,b in zip(path_means,result['attributions'])),
            'additional_score_calls':0,
            'scope':'Actual calls to the existing exact engine; cached values supply marginal/path audits. '
                    'Complete small permutation enumeration, not sampled permutations. Binary64 arithmetic, '
                    'not exact-real guarantees. Fixed empirical-background replacement game, not conditional '
                    'imputation, causal effects, representative backgrounds or explanation quality. '
                    'Groups are atomic players, not sums or an Owen within-group allocation. '
                    'No data-manifold test; caller owns callback determinism and model provenance.'}
