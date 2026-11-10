"""Full declared-score Owen audit with every actual replacement evaluation."""
import itertools
import math
from .core import explain
from .owen_contract import REPORT_SCOPE
from .owen import allocate_owen
from .owen_input import validate,score_function


def audit_owen(request):
    req=validate(request);names=req['feature_names'];d=len(names);groups=req['groups']
    score=score_function(req['model']);calls=[]
    def logged(row):
        value=score(row);calls.append({'row':list(row),'score':value});return value
    original=explain(logged,req['point'],req['background'],names)
    values=original['coalition_values'];full=(1<<d)-1;hybrids=[];cursor=0
    for mask in range(full+1):
        for k in range(1 if mask==full else len(req['background'])):
            call=calls[cursor]
            hybrids.append({'call_index':cursor,'mask':mask,'background_index':None if mask==full else k,**call});cursor+=1
    if cursor!=len(calls):raise ValueError('evaluation provenance mismatch')
    terms=[]
    for j in range(d):
        row=[]
        for mask in range(full+1):
            if mask>>j&1:continue
            den=d*math.comb(d-1,mask.bit_count());delta=values[mask|1<<j]-values[mask]
            row.append({'from_mask':mask,'to_mask':mask|1<<j,'weight_denominator':den,'delta':delta,'weighted_delta':delta/den})
        terms.append(row)
    ordinary={'attributions':original['attributions'],'feature_terms':terms,
        'group_sums':[math.fsum(original['attributions'][j]for j in group['features'])for group in groups],
        'attribution_sum':original['attribution_sum'],'efficiency_residual':original['efficiency_residual']}
    paths=[]
    for outer in itertools.permutations(range(len(groups))):
        for inners in itertools.product(*(itertools.permutations(groups[k]['features'])for k in outer)):
            mask=0;increments=[0.]*d;steps=[];order=[]
            for k,block in zip(outer,inners):
                for j in block:
                    nxt=mask|1<<j;delta=values[nxt]-values[mask];increments[j]=delta;order.append(j)
                    steps.append({'group_index':k,'feature_index':j,'from_mask':mask,'to_mask':nxt,'before':values[mask],'after':values[nxt],'delta':delta});mask=nxt
            total=math.fsum(increments)
            paths.append({'group_order':list(outer),'feature_order':order,'steps':steps,'increments':increments,
                'sum':total,'efficiency_residual':math.fsum([values[0],total,-values[-1]])})
    allocation=allocate_owen(values,names,groups)
    return {'schema':'shapleylab-owen-audit-v1','request':req,'allocation':allocation,'ordinary_individual':ordinary,
        'replacement_evaluations':hybrids,'model_evaluations':len(calls),'additional_allocation_model_calls':0,
        'admissible_paths':paths,'scope':REPORT_SCOPE}
