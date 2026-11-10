"""Independent bounded Owen replay via every block-contiguous ordering."""
import collections
import itertools
import math
from .owen_contract import SCOPE


def _equal(a,b):
    # JSON has one number type. Integer metadata remain exact; computed float
    # leaves allow a disclosed cross-runtime comparison tolerance.
    if type(b)is float:
        return type(a)in(int,float)and math.isfinite(a)and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if type(a)is not type(b):return False
    if isinstance(b,dict):return set(a)==set(b)and all(_equal(a[k],b[k])for k in b)
    if isinstance(b,list):return len(a)==len(b)and all(_equal(x,y)for x,y in zip(a,b))
    return a==b


def verify_allocation(report):
    """Check the complete standalone finite-game allocation, not its score origin.

    Accepting a coalition table says nothing about model provenance or masking.
    No call to the coefficient generator or callback is made. Strict recursive
    metadata types and keys apply; computed floats compare at1e-12absolute/relative. Bound8 makes complete ordering replay feasible.
    """
    try:
        if type(report)is not dict:raise ValueError('object required')
        names=report['feature_names'];gn=report['group_names'];p=report['group_feature_indices'];v=report['coalition_values']
        if type(names)is not list or not 1<=len(names)<=8 or any(type(n)is not str or not n.strip()for n in names)or len(set(names))!=len(names):raise ValueError('bad features')
        n=len(names)
        if type(gn)is not list or not 1<=len(gn)<=n or any(type(z)is not str or not z.strip()for z in gn)or len(set(gn))!=len(gn):raise ValueError('bad group names')
        if type(p)is not list or len(p)!=len(gn)or any(type(row)is not list or not row for row in p):raise ValueError('bad partition')
        flat=[i for row in p for i in row]
        if any(type(i)is not int or not 0<=i<n for i in flat)or sorted(flat)!=list(range(n)):raise ValueError('bad membership')
        if type(v)is not list or len(v)!=1<<n or any(type(x)not in(int,float)or not math.isfinite(x)for x in v):raise ValueError('bad game')
        g=len(p);owner={j:k for k,row in enumerate(p)for j in row};counts=[collections.Counter()for _ in range(n)];total=0
        outer_counts=[collections.Counter()for _ in p];outer_total=0
        for outer in itertools.permutations(range(g)):
            om=0;outer_total+=1
            for k in outer:outer_counts[k][om]+=1;om|=1<<k
            for inners in itertools.product(*(itertools.permutations(p[k])for k in outer)):
                mask=0;total+=1
                for block in inners:
                    for j in block:counts[j][mask]+=1;mask|=1<<j
        inner_counts=[collections.Counter()for _ in names]
        for row in p:
            for perm in itertools.permutations(row):
                pred=0
                for j in perm:inner_counts[j][pred]+=1;pred|=1<<j
        unions=[]
        for om in range(1<<g):unions.append(sum(1<<j for k,row in enumerate(p)if om>>k&1 for j in row))
        terms=[[]for _ in names];phis=[]
        for j in range(n):
            k=owner[j];others=[i for i in p[k]if i!=j];ownmask=sum(1<<i for i in p[k])
            def sortkey(mask):
                outer=sum(1<<q for q,row in enumerate(p)if q!=k and any(mask>>i&1 for i in row))
                within=sum(1<<q for q,i in enumerate(others)if mask>>i&1)
                return outer,within
            for before in sorted(counts[j],key=sortkey):
                after=before|1<<j;outer,_=sortkey(before);within=before&ownmask
                inner_total=math.factorial(len(p[k]));inner_count=inner_counts[j][within]
                outer_den=outer_total//outer_counts[k][outer];inner_den=inner_total//inner_count;den=total//counts[j][before]
                if den!=outer_den*inner_den:raise ValueError('inconsistent probability')
                delta=v[after]-v[before]
                if not math.isfinite(delta):raise ValueError('nonfinite difference')
                terms[j].append({'outer_group_mask':outer,'within_feature_mask':within,'from_mask':before,'to_mask':after,
                    'weight_numerator':1,'weight_denominator':den,'outer_weight_denominator':outer_den,'inner_weight_denominator':inner_den,
                    'before':v[before],'after':v[after],'delta':delta,'weighted_delta':delta/den})
            phis.append(math.fsum(t['weighted_delta']for t in terms[j]))
        quotient=[v[u]for u in unions];gt=[];gp=[]
        for k in range(g):
            row=[]
            for before in sorted(outer_counts[k]):
                after=before|1<<k;den=outer_total//outer_counts[k][before];delta=quotient[after]-quotient[before]
                if not math.isfinite(delta):raise ValueError('nonfinite group difference')
                row.append({'from_group_mask':before,'to_group_mask':after,'weight_denominator':den,'delta':delta,'weighted_delta':delta/den})
            gt.append(row);gp.append(math.fsum(t['weighted_delta']for t in row))
        sums=[math.fsum(phis[j]for j in row)for row in p];allocation_total=math.fsum(phis)
        expected={'method':'exact_two_level_owen','feature_names':names,'group_names':gn,'group_feature_indices':p,'coalition_values':v,
            'attributions':phis,'feature_terms':terms,'group_coalition_values':quotient,'group_shapley_attributions':gp,'group_terms':gt,
            'within_group_sums':sums,'group_efficiency_residuals':[math.fsum([a,-b])for a,b in zip(sums,gp)],
            'baseline':v[0],'prediction':v[-1],'attribution_sum':allocation_total,'efficiency_residual':math.fsum([v[0],allocation_total,-v[-1]]),
            'admissible_orderings':total,'scope':SCOPE}
        if not _equal(report,expected):raise ValueError('allocation does not match independent replay')
        return True
    except (KeyError,TypeError,OverflowError,ZeroDivisionError)as exc:
        raise ValueError('invalid Owen allocation report')from exc
