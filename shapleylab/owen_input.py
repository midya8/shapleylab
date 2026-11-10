"""Bounded declared monomial model and request schema for Owen audits."""
import math


def number(value):
    return type(value)in(int,float)and -100<=value<=100 and math.isfinite(value)


def valid_name(x):
    return type(x)is str and bool(x.strip())and len(x)<=40 and not any(0xD800<=ord(c)<=0xDFFF for c in x)


def validate(request):
    if type(request)is not dict or set(request)!={'feature_names','point','background','groups','model'}:
        raise ValueError('request requires feature_names/point/background/groups/model only')
    names=request['feature_names']
    if type(names)is not list or not 1<=len(names)<=6 or any(not valid_name(x)for x in names)or len(set(names))!=len(names):
        raise ValueError('one to six distinct nonempty feature names of at most40characters required')
    d=len(names)
    def vector(x):return type(x)is list and len(x)==d and all(number(v)for v in x)
    if not vector(request['point']):raise ValueError('point must be aligned numbers in[-100,100]')
    rows=request['background']
    if type(rows)is not list or not 1<=len(rows)<=32 or not all(vector(row)for row in rows):
        raise ValueError('one to32aligned background rows in[-100,100] required')
    groups=request['groups'];seen=set();gn=[]
    if type(groups)is not list or not 1<=len(groups)<=d:raise ValueError('nonempty partition required')
    for group in groups:
        if type(group)is not dict or set(group)!={'name','features'}:raise ValueError('group needs name/features only')
        name=group['name'];indices=group['features']
        if not valid_name(name)or name in gn:raise ValueError('distinct nonempty group names of at most40characters required')
        if type(indices)is not list or not indices:raise ValueError('nonempty group indices required')
        for i in indices:
            if type(i)is not int or not 0<=i<d or i in seen:raise ValueError('disjoint valid integer feature indices required')
            seen.add(i)
        gn.append(name)
    if seen!=set(range(d)):raise ValueError('groups must cover every feature exactly once')
    model=request['model']
    if type(model)is not dict or set(model)!={'bias','terms','output'}or not number(model['bias'])or type(model['output'])is not str or model['output']not in('raw','sigmoid'):
        raise ValueError('model requires finite bounded bias, terms and raw/sigmoid output only')
    terms=model['terms'];seen_terms=set()
    if type(terms)is not list or len(terms)>32:raise ValueError('model allows0..32multiaffine monomials')
    for term in terms:
        if type(term)is not dict or set(term)!={'features','weight'}or not number(term['weight']):raise ValueError('term requires features and weight in[-100,100]')
        support=term['features']
        if type(support)is not list or not support or any(type(i)is not int or not 0<=i<d for i in support)or support!=sorted(set(support))or tuple(support)in seen_terms:
            raise ValueError('monomial supports must be nonempty sorted distinct valid indices and unique')
        seen_terms.add(tuple(support))
    return {'feature_names':names[:],'point':request['point'][:],'background':[row[:]for row in rows],
        'groups':[{'name':g['name'],'features':g['features'][:]}for g in groups],
        'model':{'bias':model['bias'],'terms':[{'features':t['features'][:],'weight':t['weight']}for t in terms],'output':model['output']}}


def score_function(model):
    bias=model['bias'];terms=[(tuple(t['features']),t['weight'])for t in model['terms']];output=model['output']
    def score(row):
        terms_values=[bias]
        for indices,weight in terms:
            value=weight
            for i in indices:value*=row[i]
            terms_values.append(value)
        value=math.fsum(terms_values)
        if output=='raw':return value
        if value>=0:return 1/(1+math.exp(-value))
        exp=math.exp(value);return exp/(1+exp)
    return score
