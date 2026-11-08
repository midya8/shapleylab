"""Rebuild retained native audit witnesses without network or dependencies."""
import json
import math
from pathlib import Path
from .ledger_fixtures import build
ROOT=Path(__file__).with_name('artifacts')


def assert_reproduces(actual,expected,path='root'):
    if isinstance(expected,float):
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):raise AssertionError(path)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise AssertionError(path)
        for k in expected:assert_reproduces(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise AssertionError(path)
        for i,(a,e) in enumerate(zip(actual,expected)):assert_reproduces(a,e,path+'['+str(i)+']')
    elif type(actual)!=type(expected) or actual!=expected:raise AssertionError(path)


def saved():
    r=json.loads((ROOT/'ledger-meta.json').read_text());r['arms']={}
    for name in r.pop('arm_names'):
        a=json.loads((ROOT/('ledger-'+name+'.json')).read_text())
        a['edits']=sum((json.loads(p.read_text()) for p in sorted(ROOT.glob('ledger-'+name+'-edits-*.json'))),[])
        r['arms'][name]=a
    return r


def write():
    r=build();ROOT.mkdir(exist_ok=True)
    def dump(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    dump(ROOT/'ledger-meta.json',{**{k:v for k,v in r.items() if k!='arms'},'arm_names':list(r['arms'])})
    for name,a in r['arms'].items():
        dump(ROOT/('ledger-'+name+'.json'),{k:v for k,v in a.items() if k!='edits'})
        for i,start in enumerate(range(0,len(a['edits']),5)):dump(ROOT/('ledger-'+name+'-edits-'+str(i)+'.json'),a['edits'][start:start+5])
    docs=ROOT.parent.parent/'docs';docs.mkdir(exist_ok=True)
    (docs/'data.js').write_text('globalThis.SHAPLEY_DATA='+json.dumps(dict(arms={k:v['definition'] for k,v in r['arms'].items()},original_demo=r['original_demo']),separators=(',',':'),allow_nan=False)+';\n')
    assert_reproduces(saved(),r)
    return r


if __name__=='__main__':write()
