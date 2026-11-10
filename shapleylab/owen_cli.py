"""Bounded strict atomic CLI for declared-score Owen reports."""
import argparse
import json
import os
import sys
import tempfile
from .owen_report import audit_owen
from .owen_report_verify import verify_audit

REQUEST_LIMIT=65536
REPORT_LIMIT=16*1024*1024


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def _constant(value):raise ValueError('nonfinite JSON constant: '+value)


def read_json(path,limit):
    if path=='-':raw=sys.stdin.buffer.read(limit+1)
    else:
        with open(path,'rb')as source:raw=source.read(limit+1)
    if len(raw)>limit:raise ValueError('input exceeds'+str(limit)+'bytes')
    return json.loads(raw.decode('utf-8'),object_pairs_hook=_object,parse_constant=_constant)


def emit(result,path):
    raw=(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    if len(raw)>REPORT_LIMIT:raise ValueError('output exceeds16MiB')
    if path is None or path=='-':
        sys.stdout.buffer.write(raw);return
    target=os.path.abspath(path);directory=os.path.dirname(target);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix='.shapleylab-owen-',delete=False)as handle:
            temp=handle.name;handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,target);temp=None
    finally:
        if temp is not None:
            try:os.unlink(temp)
            except OSError:pass


def main(argv=None):
    parser=argparse.ArgumentParser(prog='shapleylab-owen')
    parser.add_argument('input',help='declared-score request, or full report with --verify; - for stdin')
    parser.add_argument('--verify',action='store_true');parser.add_argument('--output',help='atomic destination; default stdout')
    args=parser.parse_args(argv)
    try:
        payload=read_json(args.input,REPORT_LIMIT if args.verify else REQUEST_LIMIT)
        if args.verify:
            verify_audit(payload)
            result={'schema':'shapleylab-owen-verification-v1','verified':True,'model_evaluations':payload['model_evaluations']}
        else:result=audit_owen(payload)
        emit(result,args.output)
    except (ValueError,TypeError,OSError,OverflowError,RecursionError,UnicodeError)as exc:
        print('shapleylab-owen input error: '+str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
