import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from shapleylab.owen_controls import definitions
from shapleylab.owen_report import audit_owen
from shapleylab.owen_report_verify import verify_audit
from shapleylab.owen_cli import emit


def cli(path,*args,input=None):
    return subprocess.run([sys.executable,'-m','shapleylab.owen_cli',str(path),*map(str,args)],input=input,capture_output=True)


class OwenCLITests(unittest.TestCase):
    def test_twelve_original_controls_all_reports_and_native_cli_replay(self):
        counts={'controls':0,'evaluations':0,'terms':0,'paths':0,'steps':0}
        with tempfile.TemporaryDirectory()as tmp:
            tmp=Path(tmp)
            for c in definitions():
                report=audit_owen(c['request']);self.assertTrue(verify_audit(report));counts['controls']+=1
                counts['evaluations']+=report['model_evaluations'];counts['terms']+=sum(map(len,report['allocation']['feature_terms']));counts['paths']+=len(report['admissible_paths']);counts['steps']+=sum(len(p['steps'])for p in report['admissible_paths'])
                req=tmp/'request.json';out=tmp/'report.json';req.write_text(json.dumps(c['request']))
                p=cli(req,'--output',out);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,b'');self.assertEqual(json.loads(out.read_text()),report)
                p=cli(out,'--verify');self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(json.loads(p.stdout)['verified'])
        self.assertEqual(counts['controls'],12);print('Owen original controls',json.dumps(counts,sort_keys=True))

    def test_stdin_stdout_unicode_and_same_input_output(self):
        r=definitions()[0]['request'];r['feature_names'][0]='member α'
        p=cli('-',input=json.dumps(r,ensure_ascii=False).encode());self.assertEqual(p.returncode,0,p.stderr);self.assertIn('member α',p.stdout.decode());self.assertTrue(verify_audit(json.loads(p.stdout)))
        q=cli('-','--verify',input=p.stdout);self.assertEqual(q.returncode,0,q.stderr);self.assertTrue(json.loads(q.stdout)['verified'])
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'same.json';path.write_text(json.dumps(r));p=cli(path,'--output',path);self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(verify_audit(json.loads(path.read_text())))

    def test_duplicate_nonfinite_invalid_utf8_oversize_and_deep_json_atomic_failure(self):
        req=definitions()[0]['request'];body=json.dumps(req)
        invalid=[b'',b'null',b'[]',b'\xff',body.replace('"point":','"point":[1,1,1],"point":',1).encode(),body.replace('"bias": 0','"bias": NaN').encode(),b' '*65537,(b'['*1500+b'0'+b']'*1500)]
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'out.json';out.write_bytes(b'KEEP')
            for raw in invalid:
                p=cli('-','--output',out,input=raw);self.assertEqual(p.returncode,2,p.stderr);self.assertEqual(p.stdout,b'');self.assertIn(b'shapleylab-owen input error:',p.stderr);self.assertNotIn(b'Traceback',p.stderr);self.assertEqual(out.read_bytes(),b'KEEP')
            self.assertEqual(sorted(x.name for x in Path(tmp).iterdir()),['out.json'])

    def test_tampered_report_rejections_and_invalid_destination(self):
        r=audit_owen(definitions()[0]['request']);r['allocation']['attributions'][0]+=1
        p=cli('-','--verify',input=json.dumps(r).encode());self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,b'');self.assertNotIn(b'Traceback',p.stderr)
        p=cli('-','--output','/directory-that-does-not-exist/out.json',input=json.dumps(definitions()[0]['request']).encode());self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,b'')

    def test_atomic_replace_failure_cleanup_and_output_size_preflight(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'out.json';path.write_bytes(b'KEEP')
            with patch('shapleylab.owen_cli.os.replace',side_effect=OSError('replace failed')):
                with self.assertRaises(OSError):emit({'a':1},str(path))
            self.assertEqual(path.read_bytes(),b'KEEP');self.assertEqual(len(list(Path(tmp).iterdir())),1)
            with patch('shapleylab.owen_cli.REPORT_LIMIT',16):
                with self.assertRaises(ValueError):emit({'a':'x'*17},str(path))
            self.assertEqual(path.read_bytes(),b'KEEP');self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_generation_and_verification_limits_differ(self):
        raw=json.dumps(audit_owen(definitions()[-1]['request'])).encode();self.assertGreater(len(raw),65536)
        q=cli('-','--verify',input=raw);self.assertEqual(q.returncode,0,q.stderr)
        q=cli('-',input=raw);self.assertEqual(q.returncode,2);self.assertIn(b'65536',q.stderr)
        q=cli('-','--verify',input=b' '*(16*1024*1024+1));self.assertEqual(q.returncode,2);self.assertEqual(q.stdout,b'');self.assertIn(b'16777216',q.stderr)
