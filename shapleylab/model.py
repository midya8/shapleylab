"""Small serialized polynomial scorer for CLI examples, not a trained model."""
import math
from .core import _finite, _sum


def polynomial(spec, width):
    if not isinstance(spec, dict) or set(spec) != {'bias', 'linear', 'pairs', 'output'}:
        raise ValueError('model requires bias/linear/pairs/output exactly')
    bias=spec['bias'];linear=spec['linear'];pairs=spec['pairs'];output=spec['output']
    if not _finite(bias) or not isinstance(linear,list) or len(linear)!=width or any(not _finite(v) for v in linear):
        raise ValueError('bias and aligned linear coefficients must be finite')
    if output not in ('raw','sigmoid') or not isinstance(pairs,list) or len(pairs)>width*(width-1)//2:
        raise ValueError('output must be raw/sigmoid and pairs must be unique feature pairs')
    seen=set();validated=[]
    for p in pairs:
        if not isinstance(p,dict) or set(p)!={'i','j','weight'}:
            raise ValueError('each pair requires i/j/weight exactly')
        i,j,w=p['i'],p['j'],p['weight']
        if type(i) is not int or type(j) is not int or not 0<=i<j<width or not _finite(w) or (i,j) in seen:
            raise ValueError('pairs require unique 0 <= i < j < feature count and finite weight')
        seen.add((i,j));validated.append((i,j,w))
    # Copy coefficients: later changes to spec cannot change this score function.
    linear=tuple(linear)
    def score(x):
        value=_sum([bias]+[w*v for w,v in zip(linear,x)]+[w*x[i]*x[j] for i,j,w in validated])
        if output=='raw':return value
        if value>=0:return 1/(1+math.exp(-value))
        exp=math.exp(value);return exp/(1+exp)
    return score
