"""Replay a declared-score report without importing either report generator."""
import collections
import itertools
import math
from .owen_input import validate
from .owen_verify import verify_allocation,_equal
from .owen_contract import REPORT_SCOPE


def verify_audit(report):
    try:
        if type(report)is not dict:raise ValueError('report must be an object')
        req=validate(report['request']);d=len(req['feature_names']);full=(1<<d)-1;groups=req['groups'];model=req['model']
        evaluations=[];values=[]
        for mask in range(full+1):
            outputs=[]
            source_rows=[req['point']]if mask==full else req['background']
            for b,source in enumerate(source_rows):
                row=[req['point'][j]if mask>>j&1 else source[j]for j in range(d)]
                components=[model['bias']]
                for term in model['terms']:
                    value=term['weight']
                    for coordinate in term['features']:value=value*row[coordinate]
                    components.append(value)
                result=math.fsum(components)
                if model['output']=='sigmoid':
                    if result>=0:result=1/(1+math.exp(-result))
                    else:
                        e=math.exp(result);result=e/(1+e)
                outputs.append(result)
                evaluations.append({'call_index':len(evaluations),'mask':mask,'background_index':None if mask==full else b,'row':row,'score':result})
            values.append(outputs[0]if all(x==outputs[0]for x in outputs)else math.fsum(x/len(outputs)for x in outputs))
        allocation=report['allocation']
        verify_allocation(allocation)
        if not _equal(allocation['coalition_values'],values)or not _equal(allocation['feature_names'],req['feature_names'])or not _equal(allocation['group_names'],[g['name']for g in groups])or not _equal(allocation['group_feature_indices'],[g['features']for g in groups]):
            raise ValueError('allocation game or partition differs from declared score')
        counts=[collections.Counter()for _ in range(d)];order_count=0
        for order in itertools.permutations(range(d)):
            mask=0;order_count+=1
            for j in order:counts[j][mask]+=1;mask|=1<<j
        terms=[];phis=[]
        for j in range(d):
            row=[]
            for before in sorted(counts[j]):
                after=before|1<<j;den=order_count//counts[j][before];delta=values[after]-values[before]
                row.append({'from_mask':before,'to_mask':after,'weight_denominator':den,'delta':delta,'weighted_delta':delta/den})
            terms.append(row);phis.append(math.fsum(t['weighted_delta']for t in row))
        total=math.fsum(phis)
        ordinary={'attributions':phis,'feature_terms':terms,'group_sums':[math.fsum(phis[j]for j in g['features'])for g in groups],
            'attribution_sum':total,'efficiency_residual':math.fsum([values[0],total,-values[-1]])}
        paths=[]
        for group_order in itertools.permutations(range(len(groups))):
            for blocks in itertools.product(*(itertools.permutations(groups[k]['features'])for k in group_order)):
                order=[j for block in blocks for j in block];steps=[];increments=[0.]*d;mask=0
                for j in order:
                    k=next(k for k,g in enumerate(groups)if j in g['features']);nxt=mask|1<<j;delta=values[nxt]-values[mask]
                    steps.append({'group_index':k,'feature_index':j,'from_mask':mask,'to_mask':nxt,'before':values[mask],'after':values[nxt],'delta':delta});increments[j]=delta;mask=nxt
                path_total=math.fsum(increments)
                paths.append({'group_order':list(group_order),'feature_order':order,'steps':steps,'increments':increments,'sum':path_total,
                    'efficiency_residual':math.fsum([values[0],path_total,-values[-1]])})
        expected={'schema':'shapleylab-owen-audit-v1','request':req,'allocation':allocation,'ordinary_individual':ordinary,
            'replacement_evaluations':evaluations,'model_evaluations':len(evaluations),'additional_allocation_model_calls':0,
            'admissible_paths':paths,'scope':REPORT_SCOPE}
        if not _equal(report,expected):raise ValueError('report differs from independent declared-score replay')
        return True
    except (KeyError,TypeError,OverflowError,ZeroDivisionError)as exc:
        raise ValueError('invalid Owen audit report')from exc
