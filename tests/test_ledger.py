import copy
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from fractions import Fraction as F
from shapleylab.ledger import audit
from shapleylab.ledger_fixtures import build,inspect,definitions
from shapleylab.ledger_export import saved,assert_reproduces


class LedgerTests(unittest.TestCase):
    def test_original_reports_and95retained_edits(self):
        r=build();assert_reproduces(r,saved())
        self.assertEqual(sum(len(a['edits']) for a in r['arms'].values()),95)
        for name,a in r['arms'].items():self.assertEqual(a['baseline']['original_exact_result'],r['original_demo'][name])
        for mutate in [lambda d:d['arms'].pop('three_way_grouped'),lambda d:d['arms']['three_way_grouped']['baseline']['coalitions'].pop(),lambda d:d['arms']['three_way_grouped']['baseline'].__setitem__('permutation_mean_discrepancy',1e-8),lambda d:d['arms']['three_way_grouped']['baseline'].__setitem__('permutation_mean_discrepancy',float('nan'))]:
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(AssertionError):assert_reproduces(bad,r)

    def test128independent_fraction_permutation_games(self):
        rng=random.Random(823);cases=0
        for _ in range(128):
            d=rng.randrange(1,6);x=[rng.randrange(-8,9)/4 for _ in range(d)];bg=[[rng.randrange(-8,9)/4 for _ in range(d)] for _ in range(rng.randrange(1,5))];weights=[rng.randrange(-8,9)/4 for _ in range(d)]
            def exact(row):
                q=list(map(F,row));return sum((F(w)*v for w,v in zip(weights,q)),F(0))+math.prod(q)
            r=audit(lambda row:float(exact(row)),x,bg,[str(j) for j in range(d)])
            values=[]
            for mask in range(1<<d):values.append(sum((F(float(exact([x[j] if mask&(1<<j) else b[j] for j in range(d)]))) for b in bg),F(0))/len(bg))
            phis=[F(0)]*d
            for order in itertools.permutations(range(d)):
                included=set()
                for j in order:
                    before=sum(1<<i for i in included);included.add(j);after=sum(1<<i for i in included)
                    phis[j]+=(values[after]-values[before])/math.factorial(d)
            for a,e in zip(r['original_exact_result']['attributions'],phis):self.assertAlmostEqual(a,float(e),places=11)
            for c,v in zip(r['coalitions'],values):self.assertAlmostEqual(c['value'],float(v),places=11)
            for j,m in enumerate(r['marginals']):
                self.assertEqual(sum((F(t['weight_numerator'],t['weight_denominator']) for t in m['terms']),F(0)),1)
                self.assertEqual(sum(t['permutation_multiplicity'] for t in m['terms']),math.factorial(d))
                self.assertAlmostEqual(m['sum'],float(phis[j]),places=11)
            for p in r['permutation_paths']:self.assertAlmostEqual(p['efficiency_residual'],0,places=11)
            cases+=1
        self.assertEqual(cases,128)

    def test_actual_calls_once_and_atomic_group_rows(self):
        seen=[]
        def f(row):self.assertIsInstance(row,tuple);seen.append(row);return math.prod(row)
        x=[1,1,1];bg=[[0,0,0],[2,3,4]];groups=[dict(name='ab',features=[0,1]),dict(name='c',features=[2])];before=copy.deepcopy([x,bg,groups])
        r=audit(f,x,bg,['a','b','c'],groups)
        self.assertEqual(len(seen),7);self.assertEqual(r['additional_score_calls'],0);self.assertEqual(before,[x,bg,groups]);self.assertEqual(sum(len(c['evaluated_rows']) for c in r['coalitions']),7)
        self.assertEqual(r['coalitions'][-1]['evaluated_rows'],[dict(background_index=None,call_index=6,row=x,score=1)])
        for c in r['coalitions']:
            for row in c['evaluated_rows']:
                if c['mask']&1:self.assertEqual(row['row'][:2],[1,1])
                else:self.assertEqual(row['row'][:2],bg[row['background_index']][:2])
        for budget in [1,6]:
            count=len(seen)
            with self.assertRaises(ValueError):audit(f,x,bg,['a','b','c'],groups,max_evaluations=budget)
            self.assertEqual(len(seen),count)

    def test_exact_adverse_examples_preserved(self):
        r=build();off=r['arms']['off_manifold_hybrids']['baseline'];self.assertEqual(off['original_exact_result']['attributions'],[0,0]);self.assertEqual(off['coalitions'][1]['evaluated_rows'][0]['row'],[1,0]);self.assertEqual(off['coalitions'][1]['value'],.5)
        individual=r['arms']['three_way_individual']['baseline'];grouped=r['arms']['three_way_grouped']['baseline'];self.assertEqual(grouped['original_exact_result']['attributions'],[.5,.5]);self.assertNotEqual(sum(individual['original_exact_result']['attributions'][:2]),.5)
        tiny=r['original_demo']['permutation_small_sample'];self.assertEqual(tiny['attributions'][2],0);self.assertEqual(tiny['monte_carlo_standard_errors'][2],0);self.assertEqual(individual['original_exact_result']['attributions'][2],1/3)

    def test_bounds_validate_before_callback_and_no_alias(self):
        def fail(_):raise AssertionError('should not invoke')
        for names,bg in [([],[[0]]),(['a']*7,[[0]*7]),(['a'],[]),(['a'],[[0]]*33)]:
            with self.assertRaises(ValueError):audit(fail,[0]*len(names),bg,names)
        with self.assertRaises(ValueError):audit(fail,[1,2],[[0,0]],['a','b'],[dict(name='both',features=[0])])
        with self.assertRaises(ValueError):audit(lambda _:float('inf'),[0],[[0]],['a'])
        a=definitions()['product_zero_background'];r=inspect('product_zero_background');r['background'][0][0]=99;r['player_feature_indices'][0].append(2);self.assertEqual(a,definitions()['product_zero_background'])
        with self.assertRaises(ValueError):inspect('unknown')

    def test_cli_new_and_original_commands(self):
        p=subprocess.run([sys.executable,'-m','shapleylab','audit','--input','examples/grouped.json'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);r=json.loads(p.stdout);self.assertEqual(r['original_exact_result']['attributions'],[6.,0.])
        p=subprocess.run([sys.executable,'-m','shapleylab','ledger-demo'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);assert_reproduces(json.loads(p.stdout),saved())
        body=open('examples/product.json').read().replace('"point":','"point":[0,0,0],"point":',1)
        p=subprocess.run([sys.executable,'-m','shapleylab','explain','--input','-'],input=body,capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')
