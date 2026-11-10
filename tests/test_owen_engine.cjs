'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),cp=require('node:child_process');
require('../docs/owen-data.js');const I=require('../docs/owen-engine.js');
const script=`import json,random
from shapleylab.owen_controls import definitions
from shapleylab.owen_report import audit_owen
r=random.Random(803579)
requests=[c['request'] for c in definitions()]
for n in range(1,7):
 for seed in range(20):
  p=[[],[],[]]
  for j in range(n):p[r.randrange(3)].append(j)
  p=[x for x in p if x]
  supports=[[j for j in range(n)if mask>>j&1]for mask in range(1,1<<n)if r.randrange(3)==0]
  requests.append(dict(feature_names=[str(j)for j in range(n)],point=[r.randrange(-2,3)for _ in range(n)],background=[[r.randrange(-2,3)for _ in range(n)]for _ in range(1+seed%3)],groups=[dict(name='g'+str(k),features=x)for k,x in enumerate(p)],model=dict(bias=r.randrange(-3,4),terms=[dict(features=s,weight=r.randrange(-3,4))for s in supports[:32]],output='sigmoid'if seed%2 else 'raw')))
print(json.dumps([audit_owen(q)for q in requests]))`;
function close(a,b,path='root'){
 if(typeof b==='number')assert.ok(typeof a==='number'&&Number.isFinite(a)&&Math.abs(a-b)<=1e-12*(1+Math.abs(b)),path+': '+a+' vs '+b);
 else if(Array.isArray(b)){assert.ok(Array.isArray(a),path);assert.equal(a.length,b.length,path);b.forEach((v,j)=>close(a[j],v,path+'['+j+']'));}
 else if(b&&typeof b==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path+' keys');for(const k of Object.keys(b))close(a[k],b[k],path+'.'+k);}
 else assert.equal(a,b,path);
}
test('complete independent permutation-count JavaScript agrees with native coefficient reports and is native-verifiable after actual JSON serialization',()=>{
 const refs=JSON.parse(cp.execFileSync(process.env.PYTHON||'python',['-c',script],{cwd:require('node:path').join(__dirname,'..'),maxBuffer:256*1024*1024})),reports=[],counts={reports:0,evaluations:0,terms:0,paths:0,steps:0};
 for(const r of refs){const a=I.inspect(r.request);close(a,r);reports.push(a);counts.reports++;counts.evaluations+=a.model_evaluations;counts.terms+=a.allocation.feature_terms.reduce((v,row)=>v+row.length,0);counts.paths+=a.admissible_paths.length;counts.steps+=a.admissible_paths.reduce((v,p)=>v+p.steps.length,0);}
 assert.equal(refs.length,132);
 const out=cp.execFileSync(process.env.PYTHON||'python',['-c',`import json,sys
from shapleylab.owen_report_verify import verify_audit
reports=json.load(sys.stdin)
for report in reports:assert verify_audit(report)
print(len(reports))`],{cwd:require('node:path').join(__dirname,'..'),input:JSON.stringify(reports),maxBuffer:256*1024*1024});assert.equal(Number(out.toString()),132);console.log(JSON.stringify(counts));
});
test('strict duplicate/nonfinite/oversize/index-token parsing, schema, input isolation and adverse allocation controls',()=>{
 const r=structuredClone(OWEN_CONTROLS[0].request),before=structuredClone(r);const a=I.inspect(r);assert.deepEqual(r,before);assert.deepEqual(a.allocation.attributions,[.25,.25,.5]);assert.deepEqual(a.ordinary_individual.attributions,[1/3,1/3,1/3]);
 const text=JSON.stringify(r);assert.deepEqual(I.parse(text),r);
 for(const s of ['', 'null x','{"a":1,"a":2}','[NaN]','[1e309]',' '.repeat(65537),text.replace('"features":[0,1]','"features":[0.0,1]'),text.replace('"features":[0,1]','"features":[0e0,1]')])assert.throws(()=>I.validate(I.parse(s)));
 for(const changes of [{point:[true,1,1]},{point:[Infinity,1,1]},{point:[101,1,1]},{groups:[{name:'a',features:[0]}]},{extra:0},{feature_names:['\ud800','b','c']}])assert.throws(()=>I.inspect({...r,...changes}));
 const danger=I.parse('{"__proto__":{"polluted":1}}');assert.equal(Object.getPrototypeOf(danger),Object.prototype);assert.equal({}.polluted,undefined);assert.equal(Object.hasOwn(danger,'__proto__'),true);assert.throws(()=>I.validate(danger));
 for(const c of OWEN_CONTROLS){const result=I.inspect(c.request);assert.equal(result.allocation.feature_terms.length,c.request.feature_names.length);}
 const diagonal=I.inspect(OWEN_CONTROLS.find(c=>c.id==='binary-hybrid-cancellation').request);assert.deepEqual(diagonal.allocation.attributions,[0,0]);assert.ok(diagonal.replacement_evaluations.some(x=>x.score===1));
});
module.exports={close};
