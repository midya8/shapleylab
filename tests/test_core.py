import copy
import itertools
import math
import unittest
from shapleylab import explain


class TestExact(unittest.TestCase):
    def test_product_symmetry_dummy_efficiency_cost(self):
        r=explain(lambda x:x[0]*x[1],[2,3,99],[[0,0,0],[0,0,9]],['a','b','unused'])
        self.assertEqual(r['attributions'],[3,3,0]);self.assertEqual(r['baseline'],0)
        self.assertEqual(r['prediction'],6);self.assertEqual(r['efficiency_residual'],0)
        self.assertEqual(r['coalitions'],8);self.assertEqual(r['model_evaluations'],15)

    def test_linear_closed_form_correlated_background(self):
        b=[[0,0,1],[1,1,3],[2,2,5]];x=[3,4,7];coef=[2,-3,.5]
        r=explain(lambda z:4+sum(a*v for a,v in zip(coef,z)),x,b,['a','b','c'])
        for j in range(3):self.assertAlmostEqual(r['attributions'][j],coef[j]*(x[j]-sum(v[j] for v in b)/3))

    def test_joint_missing_features_not_independent_marginals(self):
        r=explain(lambda x:x[0]*x[1],[1,1],[[0,0],[1,1]],['a','b'])
        self.assertEqual(r['baseline'],.5);self.assertEqual(r['attributions'],[.25,.25])

    def test_against_independent_permutation_definition(self):
        # Alternate reference accumulates each ordering's marginal increments.
        for d in range(1,7):
            b=[[0]*d,[1]*d];x=list(range(1,d+1))
            f=lambda z:sum(z)+math.prod(z)
            result=explain(f,x,b,[str(j) for j in range(d)])
            def value(subset):
                return sum(f([x[j] if j in subset else row[j] for j in range(d)]) for row in b)/len(b)
            totals=[0.]*d
            for order in itertools.permutations(range(d)):
                subset=set();prev=value(subset)
                for j in order:
                    subset.add(j);nxt=value(subset);totals[j]+=nxt-prev;prev=nxt
            expected=[v/math.factorial(d) for v in totals]
            for actual,reference in zip(result['attributions'],expected):self.assertAlmostEqual(actual,reference)

    def test_linearity(self):
        x=[2,3];b=[[0,0],[1,2]];names=['a','b'];f=lambda z:z[0]*z[1];g=lambda z:2*z[0]-z[1]
        a=explain(f,x,b,names);c=explain(g,x,b,names);r=explain(lambda z:3*f(z)-2*g(z),x,b,names)
        for i in range(2):self.assertAlmostEqual(r['attributions'][i],3*a['attributions'][i]-2*c['attributions'][i])

    def test_one_feature_and_duplicate_background(self):
        r=explain(lambda z:z[0]**2,[3],[[1],[2]],['a'])
        self.assertEqual(r['attributions'],[6.5]);self.assertEqual(r['model_evaluations'],3)
        self.assertEqual(r['attributions'],explain(lambda z:z[0]**2,[3],[[1],[2],[1],[2]],['a'])['attributions'])

    def test_background_sensitivity_same_prediction(self):
        f=lambda z:z[0]*z[1]
        a=explain(f,[2,3],[[0,0]],['a','b']);b=explain(f,[2,3],[[1,1]],['a','b'])
        self.assertEqual(a['prediction'],b['prediction']);self.assertNotEqual(a['attributions'],b['attributions'])
        self.assertEqual(b['attributions'],[2,3]);self.assertEqual(b['baseline'],1)

    def test_feature_permutation_equivariance(self):
        a=explain(lambda z:z[0]**2+2*z[0]*z[1],[2,3],[[0,1],[1,2]],['a','b'])
        b=explain(lambda z:z[1]**2+2*z[1]*z[0],[3,2],[[1,0],[2,1]],['b','a'])
        self.assertEqual(a['attributions'],list(reversed(b['attributions'])))

    def test_budget_checked_before_any_callback(self):
        calls=[]
        with self.assertRaises(ValueError):explain(lambda x:calls.append(x) or 0,[0]*12,[[0]*12]*1000,[str(i) for i in range(12)])
        self.assertEqual(calls,[])
        with self.assertRaises(ValueError):explain(lambda x:calls.append(x) or 0,[1,2],[[0,0]],['a','b'],3)
        self.assertEqual(calls,[])

    def test_no_mutation_and_callback_gets_immutable_copy(self):
        point=[2,3];bg=[[0,1],[1,2]];names=['a','b'];before=copy.deepcopy([point,bg,names]);seen=[]
        r=explain(lambda x:seen.append(type(x)) or sum(x),point,bg,names)
        self.assertTrue(all(t is tuple for t in seen));self.assertEqual([point,bg,names],before)
        r['feature_names'][0]='changed';self.assertEqual(names,['a','b'])

    def test_invalid_inputs(self):
        for point,bg,names in [([True],[[0]],['a']),([math.nan],[[0]],['a']),([1],[],['a']),
                               ([1],[[math.inf]],['a']),([1,2],[[0,0]],['a','a']),
                               ([1],[[0]],[' ']),([],[],[]),([0]*13,[[0]*13],[str(i) for i in range(13)])]:
            with self.subTest(point=point),self.assertRaises(ValueError):explain(sum,point,bg,names)
        for budget in [True,0,1000001,1.5]:
            with self.assertRaises(ValueError):explain(sum,[1],[[0]],['a'],budget)

    def test_callback_nonfinite_nonscalar_and_overflow_fail(self):
        for f in [lambda x:math.inf,lambda x:math.nan,lambda x:True,lambda x:[1],
                  lambda x:1e308 if x[0] else -1e308]:
            with self.assertRaises(ValueError):explain(f,[1],[[0]],['a'])

    def test_coalition_values_are_auditable(self):
        r=explain(lambda z:2*z[0]+3*z[1],[1,2],[[0,0]],['a','b'])
        self.assertEqual(r['coalition_values'],[0,2,6,8])

    def test_constant_callback_and_finite_extreme_mean(self):
        r=explain(lambda z:1e308,[1],[[0],[1]],['a'])
        self.assertEqual(r['baseline'],1e308);self.assertEqual(r['attributions'],[0])

    def test_efficiency_does_not_validate_hybrid_manifold(self):
        r=explain(lambda x:(x[0]-x[1])**2,[1,1],[[0,0],[1,1]],['a','b'])
        self.assertEqual(r['coalition_values'],[0,.5,.5,0])
        self.assertEqual(r['attributions'],[0,0]);self.assertEqual(r['efficiency_residual'],0)
