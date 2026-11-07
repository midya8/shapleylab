import argparse
import json
import sys
from .core import explain
from .model import polynomial
from .demo import run_demo


def main(argv=None):
    parser=argparse.ArgumentParser(prog='shapleylab')
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    p=sub.add_parser('explain');p.add_argument('--input',required=True)
    p.add_argument('--max-evaluations',type=int,default=200000)
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            if not isinstance(payload,dict) or set(payload)!={'feature_names','point','background','model'}:
                raise ValueError('input requires feature_names/point/background/model exactly')
            names=payload['feature_names']
            if not isinstance(names,list):raise ValueError('feature_names must be a list')
            result=explain(polynomial(payload['model'],len(names)),payload['point'],
                           payload['background'],names,args.max_evaluations)
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('shapleylab input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
