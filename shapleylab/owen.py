"""Exact two-level Owen allocation, distinct from atomic group Shapley."""
import math
from .core import _finite, _sum, explain
from .owen_contract import SCOPE


def _partition(feature_names, groups):
    if not isinstance(feature_names, list) or not 1 <= len(feature_names) <= 8:
        raise ValueError('Owen requires1-8feature names')
    if any(not isinstance(x,str) or not x.strip() for x in feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('feature names must be distinct nonempty strings')
    if not isinstance(groups,list) or not 1 <= len(groups) <= len(feature_names):
        raise ValueError('groups require a nonempty partition')
    names=[];members=[];seen=set()
    for g in groups:
        if not isinstance(g,dict) or set(g) != {'name','features'}:
            raise ValueError('group needs name and features only')
        name=g['name'];indices=g['features']
        if not isinstance(name,str) or not name.strip() or name in names:
            raise ValueError('distinct nonempty group names required')
        if not isinstance(indices,list) or not indices:
            raise ValueError('nonempty feature list required')
        for j in indices:
            if type(j)is not int or not 0<=j<len(feature_names) or j in seen:
                raise ValueError('disjoint integer feature indices required')
            seen.add(j)
        names.append(name);members.append(list(indices))
    if seen != set(range(len(feature_names))):
        raise ValueError('groups must cover every feature')
    return names,members


def allocate_owen(values, feature_names, groups):
    """Allocate one supplied full game; no model calls or sampling.

    Uniformly order groups, then uniformly order members within each group.
    A group stays contiguous. Every feature marginal is weighted by the product
    of outer and inner predecessor-subset probabilities, not a naive equal
    split or ordinary feature-Shapley sum. Binary64, not exact-real arithmetic.
    """
    names,members=_partition(feature_names,groups);d=len(feature_names);g=len(groups)
    if not isinstance(values,list) or len(values)!=1<<d or any(not _finite(v)for v in values):
        raise ValueError('finite complete coalition table required')
    masks=[sum(1<<j for j in row)for row in members]
    unions=[0]*(1<<g)
    for mask in range(1<<g):
        for k,whole in enumerate(masks):
            if mask>>k&1:unions[mask]|=whole
    terms=[[]for _ in range(d)]
    for k,row in enumerate(members):
        size=len(row)
        for j in row:
            others=[i for i in row if i!=j]
            for outer in range(1<<g):
                if outer>>k&1:continue
                outer_den=g*math.comb(g-1,outer.bit_count())
                for within in range(1<<len(others)):
                    t=sum(1<<i for q,i in enumerate(others)if within>>q&1)
                    before=unions[outer]|t;after=before|1<<j
                    inner_den=size*math.comb(size-1,within.bit_count())
                    delta=values[after]-values[before]
                    if not _finite(delta):raise ValueError('nonfinite marginal')
                    terms[j].append({'outer_group_mask':outer,'within_feature_mask':t,
                        'from_mask':before,'to_mask':after,
                        'weight_numerator':1,'weight_denominator':outer_den*inner_den,
                        'outer_weight_denominator':outer_den,'inner_weight_denominator':inner_den,
                        'before':values[before],'after':values[after],
                        'delta':delta,'weighted_delta':delta/(outer_den*inner_den)})
    phis=[_sum(t['weighted_delta']for t in row)for row in terms]
    quotient=[values[u]for u in unions];group_terms=[];group_values=[]
    for k in range(g):
        row=[]
        for mask in range(1<<g):
            if mask>>k&1:continue
            delta=quotient[mask|1<<k]-quotient[mask]
            if not _finite(delta):raise ValueError('nonfinite group marginal')
            den=g*math.comb(g-1,mask.bit_count())
            row.append({'from_group_mask':mask,'to_group_mask':mask|1<<k,
                'weight_denominator':den,'delta':delta,'weighted_delta':delta/den})
        group_terms.append(row);group_values.append(_sum(t['weighted_delta']for t in row))
    sums=[_sum(phis[j]for j in row)for row in members];total=_sum(phis)
    return {'method':'exact_two_level_owen','feature_names':feature_names[:],
        'group_names':names,'group_feature_indices':members,'coalition_values':values[:],
        'attributions':phis,'feature_terms':terms,'group_coalition_values':quotient,
        'group_shapley_attributions':group_values,'group_terms':group_terms,
        'within_group_sums':sums,'group_efficiency_residuals':[_sum([a,-b])for a,b in zip(sums,group_values)],
        'baseline':values[0],'prediction':values[-1],'attribution_sum':total,
        'efficiency_residual':_sum([values[0],total,-values[-1]]),
        'admissible_orderings':math.factorial(g)*math.prod(math.factorial(len(row))for row in members),
        'scope':SCOPE}


def explain_owen(score, point, background, feature_names, groups, max_evaluations=200000):
    _partition(feature_names,groups)  # Reject partitions before a callback runs.
    original=explain(score,point,background,feature_names,max_evaluations)
    result=allocate_owen(original['coalition_values'],feature_names,groups)
    result['original_individual_result']=original
    result['model_evaluations']=original['model_evaluations']
    result['additional_allocation_model_calls']=0
    result['game']='empirical_background_replacement_with_two_level_owen_allocation'
    return result
