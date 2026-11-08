#!/usr/bin/env bash
set -euo pipefail
installed=${1}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,shapleylab,json
from shapleylab.ledger_export import saved,assert_reproduces
from shapleylab.ledger_fixtures import build
mods=list(pkgutil.walk_packages(shapleylab.__path__,shapleylab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==10
r=saved();assert_reproduces(build(),r)
q=dict(feature_names=['a','b','unused'],point=[2,3,99],background=[[0,0,0],[0,0,9]],model=dict(bias=0,linear=[0,0,0],pairs=[dict(i=0,j=1,weight=1)],output='raw'))
json.dump(q,open('request.json','w'));json.dump(dict(q,groups=[dict(name='ab',features=[0,1]),dict(name='unused',features=[2])]),open('grouped.json','w'))
print('all11modules import;95edit archives and original exact/grouped/sample reports reproduce')
PY
python -m shapleylab demo > demo.json
python -m shapleylab explain --input request.json > exact.json
python -m shapleylab explain --input grouped.json > grouped.json.out
python -m shapleylab explain --input request.json --permutations 64 --seed 0 > sampled.json
python -m shapleylab audit --input request.json > audit.json
python -m shapleylab audit --input grouped.json > group-audit.json
python -m shapleylab ledger-demo > ledger-demo.json
python - <<'PY'
import json
from shapleylab.ledger_export import saved,assert_reproduces
r=saved();load=lambda p:json.load(open(p))
assert_reproduces(load('demo.json'),r['original_demo']);assert_reproduces(load('ledger-demo.json'),r)
assert load('audit.json')['original_exact_result']==load('exact.json')
assert load('group-audit.json')['original_exact_result']==load('grouped.json.out')
assert load('exact.json')['attributions']==[3,3,0]
assert load('grouped.json.out')['attributions']==[6,0]
assert load('group-audit.json')['original_exact_result']['attributions']==[6,0]
assert load('sampled.json')['method']=='independent_permutation_monte_carlo'
print('all4commands plus exact/grouped/sampled modes run outside checkout')
PY
