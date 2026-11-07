import copy
import io
import itertools
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from shapleylab import explain,explain_groups
from shapleylab.cli import main


def groups():return [{'name':'ab','features':[0,1]},{'name':'c','features':[2]}]

class TestGrouped(unittest.TestCase):
    def test_grouping_is_not_posthoc_sum(self):
        f=lambda x:math.prod(x)
        individual=explain(f,[1,1,1],[[0,0,0]],['a','b','c'])
        grouped=explain_groups(f,[1,1,1],[[0,0,0]],['a','b','c'],groups())
        self.assertEqual(grouped['attributions'],[.5,.5])
        self.assertAlmostEqual(sum(individual['attributions'][:2]),2/3)
        self.assertNotEqual(sum(individual['attributions'][:2]),grouped['attributions'][0])
        self.assertEqual(grouped['coalition_values'],[0,0,0,1])
        self.assertEqual(grouped['model_evaluations'],4);self.assertEqual(individual['model_evaluations'],8)

    def test_singleton_groups_equal_individual_game(self):
        f=lambda z:2*z[0]+z[1]*z[2];x=[2,3,4];b=[[0,0,0],[1,2,3]];names=['a','b','c']
        a=explain(f,x,b,names);g=[{'name':n,'features':[i]} for i,n in enumerate(names)]
        c=explain_groups(f,x,b,names,g)
        for key in ['attributions','baseline','prediction','coalition_values','model_evaluations','efficiency_residual']:
            self.assertEqual(a[key],c[key])

    def test_single_group_whole_prediction_difference(self):
        r=explain_groups(sum,[2,3,4],[[0,0,0],[1,1,1]],['a','b','c'],[{'name':'all','features':[2,0,1]}])
        self.assertEqual(r['attributions'],[7.5]);self.assertEqual(r['model_evaluations'],3)
        self.assertEqual(r['group_feature_indices'],[[2,0,1]])

    def test_no_within_group_hybrid_split(self):
        seen=[]
        def f(x):seen.append(x);return (x[0]-x[1])**2+x[2]
        r=explain_groups(f,[1,1,2],[[0,0,0],[1,1,1]],['a','b','c'],groups())
        self.assertTrue(all(x[0]==x[1] for x in seen));self.assertEqual(r['attributions'][0],0)
        self.assertEqual(r['model_evaluations'],7)

    def test_group_order_equivariance(self):
        f=lambda x:x[0]**2+x[0]*x[2];args=(f,[2,3,4],[[0,0,0],[1,1,1]],['a','b','c'])
        a=explain_groups(*args,groups());b=explain_groups(*args,list(reversed(groups())))
        self.assertEqual(a['attributions'],list(reversed(b['attributions'])))
        self.assertEqual(b['group_names'],['c','ab'])

    def test_group_permutation_reference(self):
        x=[2,3,4,5];bg=[[0,1,2,3],[1,2,3,4]];g=[{'name':'ab','features':[0,1]},
                                              {'name':'c','features':[2]},{'name':'d','features':[3]}]
        f=lambda z:math.prod(z)+z[0]**2
        result=explain_groups(f,x,bg,list('abcd'),g);totals=[0.]*3
        def v(players):
            indices={j for player in players for j in g[player]['features']}
            return sum(f([x[j] if j in indices else r[j] for j in range(4)]) for r in bg)/2
        for order in itertools.permutations(range(3)):
            s=set();prev=v(s)
            for player in order:
                s.add(player);nxt=v(s);totals[player]+=nxt-prev;prev=nxt
        for actual,total in zip(result['attributions'],totals):self.assertAlmostEqual(actual,total/6)

    def test_up_to_64_raw_features_without_raw_coalitions(self):
        names=[str(i) for i in range(64)];g=[{'name':str(k),'features':list(range(k*8,(k+1)*8))} for k in range(8)]
        r=explain_groups(sum,[1]*64,[[0]*64],names,g)
        self.assertEqual(r['attributions'],[8]*8);self.assertEqual(r['model_evaluations'],256)
        self.assertEqual(r['coalitions'],256);self.assertEqual(r['feature_names'],names)

    def test_strict_partition_validation_no_callback(self):
        invalid=[[],[{'name':'x','features':[]}],[{'name':'x','features':[0,1]}],
                 [{'name':'x','features':[0,1]},{'name':'x','features':[2]}],
                 [{'name':'x','features':[0,1]},{'name':'y','features':[1,2]}],
                 [{'name':'x','features':[0,1,1,2]}],[{'name':'x','features':[False,1,2]}],
                 [{'name':'x','features':[0,1,3]}],[{'name':'x','features':[0,1,2],'extra':1}],
                 [{'name':' ','features':[0,1,2]}],None]
        calls=[]
        for g in invalid:
            with self.subTest(g=g),self.assertRaises(ValueError):
                explain_groups(lambda x:calls.append(x) or 0,[1,1,1],[[0,0,0]],['a','b','c'],g)
        self.assertEqual(calls,[])

    def test_group_budget_precedes_callback(self):
        calls=[]
        with self.assertRaises(ValueError):
            explain_groups(lambda x:calls.append(x) or 0,[1,1,1],[[0,0,0]],['a','b','c'],groups(),3)
        self.assertEqual(calls,[])

    def test_no_input_or_result_aliases(self):
        x=[2,3,4];bg=[[0,0,0]];names=['a','b','c'];g=groups();before=copy.deepcopy([x,bg,names,g])
        r=explain_groups(sum,x,bg,names,g);self.assertEqual([x,bg,names,g],before)
        r['group_feature_indices'][0][0]=2;r['group_names'][0]='new';r['feature_names'][0]='new'
        self.assertEqual([x,bg,names,g],before)

    def test_raw_data_validation_and_dimensions(self):
        for x,bg,names in [([True,1,1],[[0,0,0]],['a','b','c']),([1,1,1],[[math.nan,0,0]],['a','b','c']),
                           ([1,1,1],[],['a','b','c']),([1]*65,[[0]*65],[str(i) for i in range(65)])]:
            with self.assertRaises(ValueError):explain_groups(sum,x,bg,names,groups())

    def test_cli_grouped_and_invalid_partition(self):
        p={'feature_names':['a','b','c'],'point':[1,1,1],'background':[[0,0,0]],'groups':groups(),
           'model':{'bias':0,'linear':[1,2,3],'pairs':[],'output':'raw'}}
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out):self.assertEqual(main(['explain','--input','-']),0)
        r=json.loads(out.getvalue());self.assertEqual(r['group_names'],['ab','c'])
        self.assertEqual(r['attributions'],[3,3]);self.assertEqual(r['feature_names'],['a','b','c'])
        p['groups'][0]['features']=[0]
        out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(main(['explain','--input','-']),2)
        self.assertEqual(out.getvalue(),'');self.assertIn('input error',err.getvalue())
