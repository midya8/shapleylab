"""Disjoint atomic feature groups as players in the same replacement game."""
from .core import explain, _finite


def explain_groups(score, point, background, feature_names, groups,
                   max_evaluations=200000):
    """Return group-level Shapley values, not sums/splits of feature values.

    groups is an exhaustive disjoint list of {name, features:[indices]}.
    Up to 64 raw features and 12 groups; every member of a group moves together.
    """
    if not callable(score):
        raise ValueError('score must be callable')
    if not isinstance(feature_names, list) or not 1 <= len(feature_names) <= 64:
        raise ValueError('grouped explanation requires 1 through 64 feature names')
    if any(not isinstance(v, str) or not v.strip() for v in feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('feature names must be distinct nonempty strings')
    d = len(feature_names)
    def vector(row):
        return isinstance(row, (list, tuple)) and len(row) == d and all(_finite(v) for v in row)
    if not vector(point):
        raise ValueError('point must be a finite feature vector')
    if not isinstance(background, list) or not 1 <= len(background) <= 1000 or any(not vector(r) for r in background):
        raise ValueError('background requires 1 through 1000 finite aligned rows')
    if not isinstance(groups, list) or not 1 <= len(groups) <= 12:
        raise ValueError('groups must be a list with 1 through 12 groups')
    group_names = []; indices = []; seen = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {'name', 'features'}:
            raise ValueError('each group requires name and features only')
        name = group['name']; features = group['features']
        if not isinstance(name, str) or not name.strip() or name in group_names:
            raise ValueError('group names must be distinct nonempty strings')
        if not isinstance(features, list) or not features:
            raise ValueError('each group requires a nonempty feature-index list')
        for j in features:
            if type(j) is not int or not 0 <= j < d or j in seen:
                raise ValueError('groups must be disjoint valid feature indices with no repeats')
            seen.add(j)
        group_names.append(name); indices.append(tuple(features))
    if seen != set(range(d)):
        raise ValueError('groups must cover every feature exactly once')
    # Reuse the validated exact player-game engine without enumerating raw-feature
    # coalitions. Coordinate 1 denotes target; negative integers identify the
    # shared background row. These internal selector coordinates never reach score.
    x = tuple(point); rows = tuple(tuple(r) for r in background)
    def grouped_score(selectors):
        hybrid = [None] * d
        for selector, members in zip(selectors, indices):
            source = x if selector == 1 else rows[-int(selector) - 1]
            for j in members:
                hybrid[j] = source[j]
        return score(tuple(hybrid))
    g = len(groups)
    result = explain(grouped_score, [1] * g,
                     [[-i-1] * g for i in range(len(rows))], group_names,
                     max_evaluations)
    result['group_names'] = result.pop('feature_names')
    result['feature_names'] = list(feature_names)
    result['group_feature_indices'] = [list(v) for v in indices]
    result['attribution_unit'] = 'atomic_group_in_group_names_order'
    result['coalition_index'] = 'integer mask, bit j includes all features in group_names[j]'
    result['game'] = 'empirical_background_atomic_group_replacement'
    result['scope'] += (' Groups are atomic players and change the coalition game. Group attribution '
                        'generally differs from summed individual-feature attribution. No within-group '
                        'credit allocation, automatic group discovery, conditional masking or causal guarantee. '
                        'Grouping prevents only within-group hybrid splits, not dependence breaks across groups.')
    return result
