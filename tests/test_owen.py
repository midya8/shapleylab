import itertools
import math
import random
import unittest
from fractions import Fraction
from shapleylab.owen import allocate_owen,explain_owen
from shapleylab.grouped import explain_groups
from shapleylab.owen_verify import verify_allocation


def partitions(items):
    if not items:
        yield []
        return
    a,*rest=items
    for p in partitions(rest):
        yield [[a]]+[row[:]for row in p]
        for k in range(len(p)):
            yield [([a]+row if k==q else row[:])for q,row in enumerate(p)]


def groups(p):return [{'name':'g'+str(k),'features':row[:]}for k,row in enumerate(p)]


def ordering_oracle(values,p):
    sums=[Fraction()for _ in range(len(values).bit_length()-1)];count=0
    for outer in itertools.permutations(range(len(p))):
        for inners in itertools.product(*(itertools.permutations(p[k])for k in outer)):
            mask=0;count+=1
            for block in inners:
                for j in block:
                    nxt=mask|1<<j;sums[j]+=Fraction(values[nxt])-Fraction(values[mask]);mask=nxt
    return [x/count for x in sums],count


class OwenPrototypeTests(unittest.TestCase):
    def assert_same(self,result,expected):
        self.assertEqual(len(result),len(expected))
        for a,b in zip(result,expected):self.assertAlmostEqual(a,float(b),places=11)

    def test_all_partitions_signed_tables_against_fraction_block_permutations(self):
        rng=random.Random(803577);cases=0;paths=0
        for n in range(1,7):
            for p in partitions(list(range(n))):
                for _ in range(2):
                    values=[rng.randrange(-100,101)for _ in range(1<<n)]
                    r=allocate_owen(values,[str(j)for j in range(n)],groups(p));expected,count=ordering_oracle(values,p)
                    self.assert_same(r['attributions'],expected);self.assertEqual(r['admissible_orderings'],count);self.assertTrue(verify_allocation(r))
                    self.assertAlmostEqual(r['efficiency_residual'],0,places=11)
                    for row in r['feature_terms']:
                        self.assertEqual(sum((Fraction(t['weight_numerator'],t['weight_denominator'])for t in row),Fraction()),1)
                    self.assert_same(r['group_efficiency_residuals'],[0]*len(p));cases+=1;paths+=count
        self.assertEqual(cases,556);print('Owen signed-game partition cases',cases,'independent block orderings',paths)

    def test_unanimity_basis_every_partition_and_nonempty_support(self):
        count=0
        for n in range(1,7):
            for p in partitions(list(range(n))):
                for support in range(1,1<<n):
                    values=[7+int(mask&support==support)for mask in range(1<<n)]
                    r=allocate_owen(values,[str(j)for j in range(n)],groups(p))
                    touched=[row for row in p if any(support>>j&1 for j in row)]
                    expected=[Fraction()for _ in range(n)]
                    for row in touched:
                        active=[j for j in row if support>>j&1]
                        for j in active:expected[j]=Fraction(1,len(touched)*len(active))
                    self.assert_same(r['attributions'],expected);count+=1
        print('Owen unanimity supports/partitions',count)

    def test_three_way_interaction_not_equal_feature_shapley(self):
        r=explain_owen(lambda x:math.prod(x),[1,1,1],[[0,0,0]],['a','b','c'],groups([[0,1],[2]]))
        self.assert_same(r['attributions'],[.25,.25,.5]);self.assert_same(r['group_shapley_attributions'],[.5,.5]);self.assert_same(r['original_individual_result']['attributions'],[1/3]*3)
        self.assertEqual(r['model_evaluations'],8);self.assertEqual(r['additional_allocation_model_calls'],0)

    def test_nonuniform_inner_credit_not_naive_equal_split(self):
        r=explain_owen(lambda x:2*x[0]+x[1]+3*x[2],[1]*3,[[0]*3],['a','b','c'],groups([[0,1],[2]]))
        self.assert_same(r['attributions'],[2,1,3]);self.assert_same(r['group_shapley_attributions'],[3,3])

    def test_same_replacement_quotient_not_independent_coordinate_imputation(self):
        fn=lambda x:x[0]*x[1]+x[2]*x[0];x=[2,3,1];b=[[0,0,0],[1,1,1]];p=groups([[1,0],[2]])
        r=explain_owen(fn,x,b,['a','b','c'],p);old=explain_groups(fn,x,b,['a','b','c'],p)
        self.assertEqual(r['baseline'],1);self.assert_same(r['group_shapley_attributions'],old['attributions'])
        self.assertEqual(r['group_coalition_values'],old['coalition_values'])

    def test_partition_and_feature_reordering(self):
        values=[0,1,2,5,4,9,-2,11];p=[[0,1],[2]]
        a=allocate_owen(values,['a','b','c'],groups(p));b=allocate_owen(values,['a','b','c'],groups([[2],[1,0]]))
        self.assert_same(a['attributions'],b['attributions']);self.assert_same(a['group_shapley_attributions'],b['group_shapley_attributions'][::-1])
        order=[2,0,1];new_values=[values[sum(1<<order[j]for j in range(3)if m>>j&1)]for m in range(8)]
        c=allocate_owen(new_values,['c','a','b'],groups([[1,2],[0]]));self.assert_same(c['attributions'],[a['attributions'][i]for i in order])

    def test_strict_preflight_zero_calls_and_budget(self):
        calls=[]
        def fn(row):calls.append(row);return sum(row)
        bad=[[],[{'name':'a','features':[0]}],[{'name':'a','features':[0,True]}],[{'name':'a','features':[0,0]}],[{'name':'a','features':[-1,1]}],[{'name':'a','features':[0]},{'name':'a','features':[1]}],[{'name':'a','features':[0,1],'extra':1}]]
        for p in bad:
            with self.assertRaises(ValueError):explain_owen(fn,[1,1],[[0,0]],['a','b'],p)
        with self.assertRaises(ValueError):explain_owen(fn,[1,1],[[0,0]],['a','b'],groups([[0],[1]]),3)
        self.assertEqual(calls,[])
        r=explain_owen(fn,[1,1],[[0,0]],['a','b'],groups([[0],[1]]),4);self.assertEqual(len(calls),4)
        self.assertTrue(all(isinstance(row,tuple)for row in calls));self.assert_same(r['attributions'],[1,1])

    def test_invalid_values_overflow_and_isolation(self):
        for v in [[0,1,2], [0,True,2,3],[0,float('inf'),1,2],[0,1,float('nan'),2],(-1,0,0,1),[-1e308,1e308,0,0]]:
            with self.assertRaises(ValueError):allocate_owen(v,['a','b'],groups([[0,1]]))
        v=[0,1,2,3];names=['a','b'];p=groups([[0],[1]]);r=allocate_owen(v,names,p)
        r['coalition_values'][0]=999;r['feature_names'][0]='changed';r['group_feature_indices'][0].append(42)
        self.assertEqual(v,[0,1,2,3]);self.assertEqual(names,['a','b']);self.assertEqual(p,groups([[0],[1]]))

    def test_linear_game_largest_bound_and_constant_shift(self):
        n=8;v=[sum(j+1 for j in range(n)if m>>j&1)for m in range(1<<n)];p=groups([[0,2,4,6],[1,3,5,7]])
        a=allocate_owen(v,[str(i)for i in range(n)],p);b=allocate_owen([x+8192for x in v],[str(i)for i in range(n)],p)
        self.assert_same(a['attributions'],list(range(1,n+1)));self.assertEqual(a['attributions'],b['attributions']);self.assertEqual(a['admissible_orderings'],1152)

    def test_checker_generator_disabled_strict_schema_and_each_primitive(self):
        import copy
        from unittest.mock import patch
        r=allocate_owen([0,1,2,3,4,7,-2,9],['a','b','c'],groups([[1,0],[2]]))
        with patch('shapleylab.owen.allocate_owen',side_effect=AssertionError('generator called')):
            self.assertTrue(verify_allocation(r))
        paths=[]
        def walk(x,path):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k not in ('feature_names','group_names','group_feature_indices','coalition_values'):walk(v,path+[k])
            elif isinstance(x,list):
                for k,v in enumerate(x):walk(v,path+[k])
            else:paths.append(path)
        walk(r,[])
        for path in paths:
            bad=copy.deepcopy(r);target=bad
            for key in path[:-1]:target=target[key]
            old=target[path[-1]];target[path[-1]]=old+'x'if isinstance(old,str)else old+1
            with self.assertRaises(ValueError):verify_allocation(bad)
        for bad in [{**r,'extra':0},{k:v for k,v in r.items()if k!='attributions'},dict(r,attributions=tuple(r['attributions'])),dict(r,admissible_orderings=float(r['admissible_orderings']))]:
            with self.assertRaises(ValueError):verify_allocation(bad)
        print('Owen individual non-input primitive mutations rejected',len(paths))

    def test_independent_checker_largest_singleton_grand_and_mixed_partitions(self):
        n=8;v=[sum(j+1 for j in range(n)if m>>j&1)+3*int(m&7==7)for m in range(1<<n)]
        for p in [[[j]for j in range(n)],[list(range(n))],[[0,2,4,6],[1,3,5,7]]]:
            self.assertTrue(verify_allocation(allocate_owen(v,[str(i)for i in range(n)],groups(p))))
