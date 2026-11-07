import copy
import io
import json
import math
import random
import statistics
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from shapleylab import explain,explain_sampled
from shapleylab.cli import main


class TestSampled(unittest.TestCase):
    def test_manual_ordering_mean_and_sample_standard_errors(self):
        x=[2,3,4];b=[[0,0,0],[1,1,1]];f=lambda z:math.prod(z)+z[0]**2
        result=explain_sampled(f,x,b,list('abc'),permutations=17,seed=31)
        rng=random.Random(31);contributions=[[],[],[]]
        def v(s):return sum(f([x[j] if j in s else r[j] for j in range(3)]) for r in b)/2
        for _ in range(17):
            order=list(range(3));rng.shuffle(order);s=set();previous=v(s)
            for j in order:
                s.add(j);nxt=v(s);contributions[j].append(nxt-previous);previous=nxt
        for j in range(3):
            self.assertAlmostEqual(result['attributions'][j],statistics.mean(contributions[j]))
            self.assertAlmostEqual(result['monte_carlo_standard_errors'][j],statistics.stdev(contributions[j])/math.sqrt(17))

    def test_small_sample_zero_se_is_not_exact(self):
        r=explain_sampled(math.prod,[1,1,1],[[0,0,0]],list('abc'),permutations=2,seed=0)
        exact=explain(math.prod,[1,1,1],[[0,0,0]],list('abc'))
        self.assertEqual(r['attributions'],[.5,.5,0]);self.assertEqual(r['monte_carlo_standard_errors'],[.5,.5,0])
        self.assertAlmostEqual(exact['attributions'][2],1/3)
        self.assertEqual(r['efficiency_residual'],0);self.assertEqual(r['model_evaluations'],6)

    def test_seeded_longer_run_is_fixture_not_general_guarantee(self):
        r=explain_sampled(math.prod,[1,1,1],[[0,0,0]],list('abc'),permutations=4096,seed=0)
        self.assertEqual(r['attributions'],[.322021484375,.341552734375,.33642578125])
        self.assertLess(max(abs(v-1/3) for v in r['attributions']),.012)
        self.assertEqual(r['cached_coalitions'],8)
        self.assertNotEqual(r['attributions'],[1/3]*3)  # Full cache does not reweight sampled increments.

    def test_linear_exact_with_zero_mc_error_for_64_features(self):
        r=explain_sampled(sum,[1]*64,[[0]*64],[str(i) for i in range(64)],permutations=32,seed=4)
        self.assertEqual(r['attributions'],[1]*64);self.assertEqual(r['monte_carlo_standard_errors'],[0]*64)
        self.assertLessEqual(r['model_evaluations'],r['preflight_upper_model_evaluations'])
        self.assertLess(r['model_evaluations'],2100)

    def test_cache_call_accounting_and_budget_preflight(self):
        calls=[];f=lambda x:calls.append(x) or math.prod(x)
        r=explain_sampled(f,[1,1,1],[[0,0,0],[.5,.5,.5]],list('abc'),permutations=12,seed=9)
        self.assertEqual(len(calls),r['model_evaluations'])
        self.assertEqual(len(calls),(r['cached_coalitions']-1)*2+1)
        self.assertLessEqual(len(calls),r['preflight_upper_model_evaluations'])
        calls.clear()
        with self.assertRaises(ValueError):explain_sampled(f,[1]*64,[[0]*64],[str(i) for i in range(64)],permutations=100,seed=0,max_evaluations=100)
        self.assertEqual(calls,[])

    def test_deterministic_independent_rng_no_mutation(self):
        x=[2,3];b=[[0,0],[1,1]];names=['a','b'];before=copy.deepcopy([x,b,names]);state=random.getstate()
        r=explain_sampled(math.prod,x,b,names,permutations=12,seed=10)
        self.assertEqual(r,explain_sampled(math.prod,x,b,names,permutations=12,seed=10))
        self.assertEqual(random.getstate(),state);self.assertEqual([x,b,names],before)
        r['feature_names'][0]='changed';self.assertEqual([x,b,names],before)

    def test_one_feature_and_joint_background(self):
        r=explain_sampled(lambda x:x[0]**2,[3],[[1],[2]],['a'],permutations=3)
        self.assertEqual(r['attributions'],[6.5]);self.assertEqual(r['monte_carlo_standard_errors'],[0]);self.assertEqual(r['model_evaluations'],3)
        r=explain_sampled(math.prod,[1,1],[[0,0],[1,1]],list('ab'),permutations=8)
        self.assertEqual(r['baseline'],.5)

    def test_invalid_parameters_rows_and_callback(self):
        for kw in [{'permutations':True},{'permutations':1},{'permutations':10001},{'seed':True},{'seed':-1},{'seed':2**32},{'max_evaluations':True}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):explain_sampled(sum,[1],[[0]],['a'],**kw)
        for x,b,names in [([True],[[0]],['a']),([1],[],['a']),([1],[[math.nan]],['a']),([1,1],[[0,0]],['a','a']),([1]*65,[[0]*65],[str(i) for i in range(65)])]:
            with self.assertRaises(ValueError):explain_sampled(sum,x,b,names)
        with self.assertRaises(ValueError):explain_sampled(lambda x:math.inf,[1],[[0]],['a'])

    def test_cli_sampled_and_grouped_conflict_no_partial_report(self):
        p={'feature_names':['a','b'],'point':[2,3],'background':[[0,0]],
           'model':{'bias':0,'linear':[1,2],'pairs':[],'output':'raw'}}
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out):
            self.assertEqual(main(['explain','--input','-','--permutations','8','--seed','12']),0)
        r=json.loads(out.getvalue());self.assertEqual(r['attributions'],[2,6]);self.assertEqual(r['seed'],12)
        p['groups']=[{'name':'all','features':[0,1]}];out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(main(['explain','--input','-','--permutations','8']),2)
        self.assertEqual(out.getvalue(),'');self.assertIn('sampled groups',err.getvalue())

        p.pop('groups');out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(main(['explain','--input','-','--seed','12']),2)
        self.assertEqual(out.getvalue(),'');self.assertIn('--seed applies only',err.getvalue())
