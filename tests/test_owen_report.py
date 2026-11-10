import copy
import math
import random
import unittest
from unittest.mock import patch
from shapleylab.owen_input import validate
from shapleylab.owen_report import audit_owen
from shapleylab.owen_report_verify import verify_audit


def control():
    return {'feature_names':['a','b','c'],'point':[1,1,1],'background':[[0,0,0]],
        'groups':[{'name':'ab','features':[0,1]},{'name':'c','features':[2]}],
        'model':{'bias':0,'terms':[{'features':[0,1,2],'weight':1}],'output':'raw'}}


class OwenReportTests(unittest.TestCase):
    def test_original_interaction_paths_provenance_and_generator_disabled_replay(self):
        r=audit_owen(control());self.assertEqual(r['allocation']['attributions'],[.25,.25,.5])
        self.assertEqual(r['model_evaluations'],8);self.assertEqual(len(r['replacement_evaluations']),8);self.assertEqual(len(r['admissible_paths']),4)
        self.assertEqual(r['additional_allocation_model_calls'],0)
        for j in range(3):self.assertEqual(math.fsum(p['increments'][j]for p in r['admissible_paths'])/4,r['allocation']['attributions'][j])
        with patch('shapleylab.owen_report.audit_owen',side_effect=AssertionError('generator called')),patch('shapleylab.owen.allocate_owen',side_effect=AssertionError('solver called')):
            self.assertTrue(verify_audit(r))

    def test_all_non_request_primitive_leaves_mutated_and_strict_structure(self):
        r=audit_owen(control());paths=[]
        def walk(x,path):
            if type(x)is dict:
                for k,v in x.items():
                    if k!='request':walk(v,path+[k])
            elif type(x)is list:
                for k,v in enumerate(x):walk(v,path+[k])
            else:paths.append(path)
        walk(r,[])
        for path in paths:
            bad=copy.deepcopy(r);target=bad
            for k in path[:-1]:target=target[k]
            old=target[path[-1]];target[path[-1]]=old+'x'if type(old)is str else 1 if old is None else old+1
            with self.assertRaises(ValueError):verify_audit(bad)
        for bad in [{**r,'extra':1},{k:v for k,v in r.items()if k!='ordinary_individual'},dict(r,admissible_paths=tuple(r['admissible_paths'])),dict(r,model_evaluations=float(r['model_evaluations']))]:
            with self.assertRaises(ValueError):verify_audit(bad)
        print('Owen fullreport non-request primitive mutations rejected',len(paths))

    def test_seeded_raw_sigmoid_full_native_replay_and_group_background_cases(self):
        rng=random.Random(803578);count=0
        for n in range(1,7):
            for seed in range(20):
                names=[str(j)for j in range(n)];part=[[],[]]
                for j in range(n):part[rng.randrange(2)].append(j)
                part=[x for x in part if x]
                supports=[]
                for mask in range(1,1<<n):
                    if rng.randrange(3)==0:supports.append([j for j in range(n)if mask>>j&1])
                req={'feature_names':names,'point':[rng.randrange(-2,3)for _ in names],
                    'background':[[rng.randrange(-2,3)for _ in names]for _ in range(1+seed%3)],
                    'groups':[{'name':'g'+str(k),'features':x}for k,x in enumerate(part)],
                    'model':{'bias':rng.randrange(-3,4),'terms':[{'features':x,'weight':rng.randrange(-3,4)}for x in supports[:32]],'output':'sigmoid'if seed%2 else 'raw'}}
                r=audit_owen(req);self.assertTrue(verify_audit(r));self.assertEqual(r['model_evaluations'],((1<<n)-1)*len(req['background'])+1);count+=1
        self.assertEqual(count,120);print('Owen full declared-score reports independently replayed',count)

    def test_declared_schema_strict_bounds_no_original_engine_calls_and_no_alias(self):
        r=control();bad=[]
        for key,value in [('point',[float('nan'),1,1]),('point',[10**10000,1,1]),('point',[True,1,1]),('background',[]),('feature_names',['a']*3),('groups',[{'name':'g','features':[0,1]}])]:
            q=copy.deepcopy(r);q[key]=value;bad.append(q)
        for model in [{'bias':0,'terms':[{'features':[1,0],'weight':1}],'output':'raw'}, {'bias':0,'terms':[{'features':[True],'weight':1}],'output':'raw'}, {'bias':101,'terms':[],'output':'raw'},{'bias':0,'terms':[],'output':'other'}]:
            q=copy.deepcopy(r);q['model']=model;bad.append(q)
        bad.append({**r,'extra':0})
        with patch('shapleylab.owen_report.explain',side_effect=AssertionError('callback engine invoked')):
            for q in bad:
                with self.assertRaises(ValueError):audit_owen(q)
        result=audit_owen(r);result['request']['point'][0]=100;self.assertEqual(r,control())
        with self.assertRaises(ValueError):verify_audit(result)

    def test_full_largest_constant_shift_and_extreme_bounded_finite_input(self):
        r=control();r['feature_names']=list('abcdef');r['point']=[100]*6;r['background']=[[-100]*6]*32;r['groups']=[{'name':'all','features':list(range(6))}];r['model']={'bias':100,'terms':[{'features':list(range(6)),'weight':100}],'output':'raw'}
        a=audit_owen(r);self.assertTrue(verify_audit(a));self.assertEqual(a['model_evaluations'],2017);self.assertEqual(len(a['admissible_paths']),720)
        r['model']['output']='sigmoid';self.assertTrue(verify_audit(audit_owen(r)))

    def test_disclosed_float_tolerance_and_exact_integer_metadata(self):
        r=audit_owen(control());r['allocation']['attributions'][0]+=1e-14
        self.assertTrue(verify_audit(r))
        r['allocation']['attributions'][0]+=.000001
        with self.assertRaises(ValueError):verify_audit(r)
        r=audit_owen(control());r['model_evaluations']=float(r['model_evaluations'])
        with self.assertRaises(ValueError):verify_audit(r)
        r=audit_owen(control());r['replacement_evaluations'][0]['score']=0
        self.assertTrue(verify_audit(r))
