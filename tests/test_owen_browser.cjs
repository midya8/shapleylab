'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn,execFileSync}=require('node:child_process');
require('../docs/owen-data.js');const {waitForDebugger}=require('./chrome_support.js');const pause=ms=>new Promise(r=>setTimeout(r,ms));
const fmt=x=>Number.isInteger(x)?String(x):Number(x.toPrecision(7)).toString();
function near(a,b){assert.ok(Math.abs(a-b)<=1e-12*(1+Math.abs(b)),a+' vs '+b);}
function close(a,b){if(typeof b==='number')near(a,b);else if(Array.isArray(b)){assert.ok(Array.isArray(a));assert.equal(a.length,b.length);b.forEach((v,j)=>close(a[j],v));}else if(b&&typeof b==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(b))close(a[k],b[k]);}else assert.equal(a,b);}
test('actual Chrome every original control at three widths, replacement/coefficient/group/path rows and trusted downloaded native verification',{timeout:120000},async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'owen-chrome-')),downloads=fs.mkdtempSync(path.join(os.tmpdir(),'owen-download-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,error,logs='';browser.on('error',x=>error=x);browser.stderr.on('data',x=>logs=(logs+x).slice(-4000));
 try{
  for(let i=0;i<1200&&!error&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await pause(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),String(error||logs));const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0],tab=await waitForDebugger('http://127.0.0.1:'+port+'/json/list',{alive:()=>browser.exitCode===null&&!error});
  ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.rej(Error(JSON.stringify(m.error))):p.res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));}),js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const change=async(id,value)=>js(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event('change'))`),rows=id=>js(`Array.from(document.querySelectorAll('#${id} tbody tr')).map(r=>Array.from(r.cells).map(x=>x.textContent))`);
  await call('Page.enable');await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads,eventsEnabled:true});await call('Page.navigate',{url:process.env.SHAPLEY_OWEN_URL||'file://'+path.resolve(__dirname,'../docs/owen.html')});
  for(let i=0;i<250;i++){if(await js('Boolean(globalThis.OWEN_REPORT)'))break;await pause(20);}assert.equal(await js(`document.getElementById('error').textContent`),'');
  const refs=JSON.parse(execFileSync(process.env.PYTHON||'python',['-c',`import json
from shapleylab.owen_controls import definitions
from shapleylab.owen_report import audit_owen
print(json.dumps([audit_owen(c['request'])for c in definitions()]))`],{cwd:path.join(__dirname,'..'),maxBuffer:64*1024*1024}));
  const counts={reports:0,evaluations:0,terms:0,ordinaryTerms:0,groupTerms:0,paths:0,steps:0,downloads:0};
  const screenDir=process.env.OWEN_SCREEN_DIR;if(screenDir)fs.mkdirSync(screenDir,{recursive:true});
  async function shot(name,width){if(!screenDir)return;await js('document.activeElement.blur();window.scrollTo(0,0);document.fonts.ready.then(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))');const metrics=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(path.join(screenDir,name+'.png'),Buffer.from(image.data,'base64'));}
  for(const width of [1280,390,320]){
   await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});
   for(let k=0;k<refs.length;k++){
    await change('control',k);const r=refs[k];close(await js('OWEN_REPORT'),r);counts.reports++;
    assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'),'page overflow '+k+'/'+width);
    assert.ok(await js(`Array.from(document.querySelectorAll('button,select,textarea,summary')).filter(e=>e.getClientRects().length).every(e=>e.getBoundingClientRect().height>=44)`),'touch');
    for(let m=0;m<r.allocation.coalition_values.length;m++){
     await change('mask',m);const expected=r.replacement_evaluations.filter(x=>x.mask===m).map(x=>[String(x.call_index),x.background_index===null?'full target':String(x.background_index),'['+x.row.map(fmt).join(', ')+']',fmt(x.score)]);assert.deepEqual(await rows('hybrids'),expected);counts.evaluations+=expected.length;
    }
    for(let j=0;j<r.request.feature_names.length;j++){
     await change('member',j);const a=r.allocation,expected=a.feature_terms[j].map(t=>[t.outer_group_mask,t.within_feature_mask,t.from_mask,t.to_mask,fmt(t.before),fmt(t.after),'1/'+t.outer_weight_denominator,'1/'+t.inner_weight_denominator,fmt(t.delta),fmt(t.weighted_delta)].map(String));assert.deepEqual(await rows('terms'),expected);counts.terms+=expected.length;
     const ordinary=r.ordinary_individual.feature_terms[j].map(t=>[t.from_mask,t.to_mask,'1/'+t.weight_denominator,fmt(t.delta),fmt(t.weighted_delta)].map(String));assert.deepEqual(await rows('ordinary-terms'),ordinary);counts.ordinaryTerms+=ordinary.length;
     assert.ok((await js(`document.getElementById('member-summary').textContent`)).includes(a.feature_names[j]));
    }
    for(let q=0;q<r.request.groups.length;q++){
     await change('group',q);const expected=r.allocation.group_terms[q].map(t=>[t.from_group_mask,t.to_group_mask,'1/'+t.weight_denominator,fmt(t.delta),fmt(t.weighted_delta)].map(String));assert.deepEqual(await rows('group-terms'),expected);counts.groupTerms+=expected.length;
    }
    for(let p=0;p<r.admissible_paths.length;p++){
     await change('path',p);const expected=r.admissible_paths[p].steps.map(t=>[r.allocation.group_names[t.group_index],r.allocation.feature_names[t.feature_index],t.from_mask,t.to_mask,fmt(t.before),fmt(t.after),fmt(t.delta)].map(String));assert.deepEqual(await rows('path-steps'),expected);counts.paths++;counts.steps+=expected.length;
    }
    const max=Math.max(...r.allocation.attributions.map(Math.abs),...r.ordinary_individual.attributions.map(Math.abs)),bars=await js(`Array.from(document.querySelectorAll('#members .fill')).map(e=>({left:parseFloat(e.style.left),width:parseFloat(e.style.width)}))`),values=r.allocation.attributions.flatMap((v,j)=>[v,r.ordinary_individual.attributions[j]]);values.forEach((v,j)=>{assert.ok(Math.abs(bars[j].width-(max?50*Math.abs(v)/max:0))<.0001);assert.ok(Math.abs(bars[j].left-(v<0?50-(max?50*Math.abs(v)/max:0):50))<.0001);});
    await change('mask',0);await change('member',0);await change('group',0);await change('path',0);await shot('control-'+k+'-'+width,width);
    if(width===1280){await js(`document.getElementById('download').scrollIntoView({block:'center'});new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);const box=await js(`JSON.stringify(document.getElementById('download').getBoundingClientRect().toJSON())`),b=JSON.parse(box);assert.equal(await js(`document.elementFromPoint(${b.x+b.width/2},${b.y+b.height/2}).id`),'download');await call('Input.dispatchMouseEvent',{type:'mousePressed',x:b.x+b.width/2,y:b.y+b.height/2,button:'left',clickCount:1});await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:b.x+b.width/2,y:b.y+b.height/2,button:'left',clickCount:1});
     let filename;for(let t=0;t<300;t++){filename=fs.readdirSync(downloads).find(x=>x.endsWith('.json'));if(filename)break;await pause(20);}assert.ok(filename,'completed trusted download');const file=path.join(downloads,filename),raw=fs.readFileSync(file);close(JSON.parse(raw),r);const verified=JSON.parse(execFileSync(process.env.PYTHON||'python',['-m','shapleylab.owen_cli',file,'--verify'],{cwd:path.join(__dirname,'..')}));assert.equal(verified.verified,true);fs.unlinkSync(file);counts.downloads++;
    }
   }
  }
  for(const width of [1280,390,320]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await change('control',0);await js(`for(const e of document.querySelectorAll('details'))e.open=true`);await shot('expanded-'+width,width);assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'));
   await change('control',6);await change('member',2);await change('mask',3);await change('group',1);await change('path',3);await js(`for(const e of document.querySelectorAll('.table-wrap'))e.scrollLeft=e.scrollWidth`);await shot('negative-rightmost-'+width,width);
  }
  await call('Emulation.setDeviceMetricsOverride',{width:320,height:1000,deviceScaleFactor:1,mobile:true});await change('control',0);
  for(const invalid of ['{"x":1,"x":2}','{"point":[NaN]}',' '.repeat(65537),JSON.stringify({...OWEN_CONTROLS[0].request,point:[101,1,1]})]){
   await js(`document.getElementById('request').value=${JSON.stringify(invalid)};document.getElementById('request').dispatchEvent(new Event('input'))`);assert.equal(await js('OWEN_REPORT'),null);assert.equal(await js(`document.getElementById('results').hidden`),true);assert.equal(await js(`document.getElementById('download').disabled`),true);await js(`document.getElementById('inspect').click()`);assert.equal(await js('OWEN_REPORT'),null);assert.match(await js(`document.getElementById('error').textContent`),/^Invalid input:/);
  }
  await shot('invalid-320',320);
  // Trusted keyboard edit must invalidate a real report before a new run.
  await change('control',0);await js(`document.getElementById('request').focus()`);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',modifiers:2,windowsVirtualKeyCode:65});
  await call('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',modifiers:2,windowsVirtualKeyCode:65});
  await call('Input.insertText',{text:'{'});assert.equal(await js('OWEN_REPORT'),null);assert.equal(await js(`document.getElementById('download').disabled`),true);
  const boundary={feature_names:['α','b','c','d','e','雪😀'],point:[1,2,3,4,5,6],background:Array.from({length:32},(_,j)=>Array.from({length:6},(_,i)=>(j+i)%5-2)),groups:[{name:'all',features:[0,1,2,3,4,5]}],model:{bias:1,terms:[{features:[0,1,2],weight:2},{features:[2,5],weight:-3}],output:'raw'}};
  for(const output of ['raw','sigmoid']){
   boundary.model.output=output;const raw=JSON.stringify(boundary);
   await js(`document.getElementById('request').value=${JSON.stringify(raw)};document.getElementById('request').dispatchEvent(new Event('input'));document.getElementById('inspect').click()`);
   const actual=await js('OWEN_REPORT');assert.ok(actual);assert.equal(actual.replacement_evaluations.length,2017);assert.equal(actual.admissible_paths.length,720);
   const native=JSON.parse(execFileSync(process.env.PYTHON||'python',['-c','import json,sys;from shapleylab.owen_report import audit_owen;print(json.dumps(audit_owen(json.load(sys.stdin))))'],{input:raw,cwd:path.join(__dirname,'..'),maxBuffer:32*1024*1024}));close(actual,native);
   assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'));await shot('boundary-'+output+'-320',320);
  }
  for(const invalid of [JSON.stringify(boundary).replace('"features":[0,1,2,3,4,5]','"features":[0.0,1,2,3,4,5]'),JSON.stringify(boundary).replace('"features":[0,1,2,3,4,5]','"features":[0e0,1,2,3,4,5]'),JSON.stringify({...boundary,feature_names:['\ud800','b','c','d','e','f']})]){
   await js(`document.getElementById('request').value=${JSON.stringify(invalid)};document.getElementById('request').dispatchEvent(new Event('input'));document.getElementById('inspect').click()`);assert.equal(await js('OWEN_REPORT'),null);assert.match(await js(`document.getElementById('error').textContent`),/^Invalid input:/);
  }
  assert.equal(counts.downloads,12);console.log(JSON.stringify({...counts,boundaryReports:2,boundaryEvaluations:4034,boundaryPaths:1440,trustedKeyboardEdit:true}));
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(r=>browser.once('exit',r));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});await fs.promises.rm(downloads,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
