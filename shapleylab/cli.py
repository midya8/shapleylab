import argparse
import json
import sys
from .core import explain
from .grouped import explain_groups
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
            if not isinstance(payload,dict) or set(payload) not in ({'feature_names','point','background','model'},
                                                                     {'feature_names','point','background','model','groups'}):
                raise ValueError('input requires feature_names/point/background/model and optionally groups only')
            names=payload['feature_names']
            if not isinstance(names,list):raise ValueError('feature_names must be a list')
            score=polynomial(payload['model'],len(names))
            if 'groups' in payload:
                result=explain_groups(score,payload['point'],payload['background'],names,
                                      payload['groups'],args.max_evaluations)
            else:
                result=explain(score,payload['point'],payload['background'],names,args.max_evaluations)
        body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as exc:
        print('shapleylab input error: '+str(exc),file=sys.stderr);return 2
    print(body);return 0
