import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from shapleylab.model import polynomial
from shapleylab.cli import main
from shapleylab.demo import run_demo


def spec():return {'bias':1,'linear':[2,-1],'pairs':[{'i':0,'j':1,'weight':3}],'output':'raw'}

class TestModelCLI(unittest.TestCase):
    def test_polynomial_hand_calculation(self):
        self.assertEqual(polynomial(spec(),2)([2,3]),20)

    def test_model_snapshot_not_mutable_alias(self):
        s=spec();f=polynomial(s,2);s['linear'][0]=99;s['pairs'][0]['weight']=0
        self.assertEqual(f([2,3]),20)

    def test_sigmoid_extremes_and_link_changes_game(self):
        s=spec();s['output']='sigmoid';f=polynomial(s,2)
        self.assertEqual(f([1000,1000]),1);self.assertEqual(f([-1000,1000]),0)
        self.assertAlmostEqual(f([0,0]),1/(1+math.exp(-1)))

    def test_model_validation(self):
        for field,value in [('bias',True),('linear',[1]),('output','softmax'),('pairs',[{'i':1,'j':0,'weight':1}]),
                            ('pairs',[{'i':0,'j':1,'weight':1}]*2),('pairs',[{'i':0,'j':1,'weight':math.inf}])]:
            s=spec();s[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):polynomial(s,2)

    def test_cli_roundtrip(self):
        p={'feature_names':['a','b'],'point':[2,3],'background':[[0,0]],'model':spec()};out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(p))),redirect_stdout(out):self.assertEqual(main(['explain','--input','-']),0)
        r=json.loads(out.getvalue());self.assertEqual(r['prediction'],20);self.assertEqual(r['baseline'],1)
        self.assertAlmostEqual(sum(r['attributions']),19)

    def test_cli_errors_no_partial_json(self):
        for text in ['{','{}','{"feature_names":1,"point":[],"background":[],"model":{}}']:
            out,err=io.StringIO(),io.StringIO()
            with patch('sys.stdin',io.StringIO(text)),redirect_stdout(out),redirect_stderr(err):self.assertEqual(main(['explain','--input','-']),2)
            self.assertEqual(out.getvalue(),'');self.assertIn('input error',err.getvalue())

    def test_demo_invariants(self):
        r=run_demo();self.assertEqual(r['product_zero_background']['attributions'],[3,3,0])
        self.assertEqual(r['product_shifted_background']['attributions'],[2,3,0])
        self.assertAlmostEqual(r['sigmoid_polynomial']['efficiency_residual'],0)
