'use strict';
(function(root){
const clone=x=>JSON.parse(JSON.stringify(x));
const exactKeys=(x,keys)=>x!==null&&typeof x==='object'&&!Array.isArray(x)&&Object.keys(x).sort().join('|')===keys.slice().sort().join('|');
const sum=values=>{const partials=[];for(let x of values){if(!Number.isFinite(x))throw Error('Nonfinite arithmetic');let i=0;for(let y of partials){if(Math.abs(x)<Math.abs(y))[x,y]=[y,x];const hi=x+y,lo=y-(hi-x);if(lo)partials[i++]=lo;x=hi;}partials.length=i;partials.push(x);}let result=0;for(let k=partials.length-1;k>=0;k--)result+=partials[k];if(!Number.isFinite(result))throw Error('Nonfinite sum');return result;};
const perms=a=>a.length?a.flatMap((x,i)=>perms(a.filter((_,j)=>j!==i)).map(rest=>[x,...rest])):[[]];
const factorial=n=>{let value=1;for(let k=2;k<=n;k++)value*=k;return value;};
function validate(r){
 if(!exactKeys(r,['feature_names','point','background','groups','model']))throw Error('Request keys must be feature_names/point/background/groups/model only');
 const name=x=>typeof x==='string'&&x.trim()&&[...x].length<=40&&![...x].some(c=>c.codePointAt(0)>=0xD800&&c.codePointAt(0)<=0xDFFF),number=x=>typeof x==='number'&&Number.isFinite(x)&&x>=-100&&x<=100;
 const names=r.feature_names;if(!Array.isArray(names)||names.length<1||names.length>6||!names.every(name)||new Set(names).size!==names.length)throw Error('One to six distinct feature names required');
 const d=names.length,vector=x=>Array.isArray(x)&&x.length===d&&x.every(number);
 if(!vector(r.point)||!Array.isArray(r.background)||r.background.length<1||r.background.length>32||!r.background.every(vector))throw Error('Aligned bounded point and1..32background rows required');
 if(!Array.isArray(r.groups)||r.groups.length<1||r.groups.length>d)throw Error('Nonempty partition required');
 const seen=new Set(),gn=new Set();
 for(const g of r.groups){if(!exactKeys(g,['name','features'])||!name(g.name)||gn.has(g.name)||!Array.isArray(g.features)||!g.features.length)throw Error('Invalid group');gn.add(g.name);for(const i of g.features){if(!Number.isInteger(i)||i<0||i>=d||seen.has(i))throw Error('Disjoint integer member indices required');seen.add(i);}}
 if(seen.size!==d)throw Error('Partition must cover all features');
 const m=r.model;if(!exactKeys(m,['bias','terms','output'])||!number(m.bias)||!['raw','sigmoid'].includes(m.output)||!Array.isArray(m.terms)||m.terms.length>32)throw Error('Invalid bounded monomial model');
 const supports=new Set();for(const t of m.terms){if(!exactKeys(t,['features','weight'])||!number(t.weight)||!Array.isArray(t.features)||!t.features.length||t.features.some((x,i)=>!Number.isInteger(x)||x<0||x>=d||(i&&x<=t.features[i-1])))throw Error('Invalid monomial support');const key=t.features.join(',');if(supports.has(key))throw Error('Repeated monomial');supports.add(key);}
 return clone(r);
}
function parse(text){
 if(typeof text!=='string'||new TextEncoder().encode(text).length>65536)throw Error('Input exceeds65536bytes');
 let i=0;const ws=()=>{while(i<text.length&&/[\x20\t\r\n]/.test(text[i]))i++;};
 function value(path){ws();const c=text[i];if(c==='"'){const start=i++;let escaped=false;for(;i<text.length;i++){const ch=text[i];if(!escaped&&ch==='"'){i++;return JSON.parse(text.slice(start,i));}if(!escaped&&ch==='\\')escaped=true;else escaped=false;}throw Error('Unterminated JSON string');}
 if(c==='['){i++;ws();const a=[];if(text[i]===']'){i++;return a;}for(;;){a.push(value([...path,a.length]));ws();if(text[i]===']'){i++;return a;}if(text[i++]!==',')throw Error('Expected array comma');}}
 if(c==='{'){i++;ws();const o={},keys=new Set();if(text[i]==='}'){i++;return o;}for(;;){ws();if(text[i]!=='"')throw Error('Object key must be a string');const k=value(path);if(keys.has(k))throw Error('Duplicate JSON key: '+k);keys.add(k);ws();if(text[i++]!==':')throw Error('Expected colon');Object.defineProperty(o,k,{value:value([...path,k]),writable:true,enumerable:true,configurable:true});ws();if(text[i]==='}'){i++;return o;}if(text[i++]!==',')throw Error('Expected object comma');}}
 for(const [token,result]of [['true',true],['false',false],['null',null]])if(text.startsWith(token,i)){i+=token.length;return result;}
 const token=text.slice(i).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);if(!token)throw Error('Invalid JSON value');i+=token[0].length;const result=Number(token[0]);if(!Number.isFinite(result))throw Error('Nonfinite JSON number');if(path.at(-2)==='features'&&!/^-?(?:0|[1-9]\d*)$/.test(token[0]))throw Error('Feature indices require integer JSON tokens');return result;
 }
 const r=value([]);ws();if(i!==text.length)throw Error('Trailing JSON content');return r;
}
function evaluate(model,row){const parts=[model.bias];for(const term of model.terms){let v=term.weight;for(const i of term.features)v*=row[i];parts.push(v);}const z=sum(parts);if(model.output==='raw')return z;return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}
function inspect(request){
 const r=validate(request),d=r.feature_names.length,g=r.groups.length,full=(1<<d)-1,values=[],evaluations=[];
 for(let mask=0;mask<=full;mask++){const outputs=[],sources=mask===full?[r.point]:r.background;for(let b=0;b<sources.length;b++){const row=r.point.map((x,j)=>mask&(1<<j)?x:sources[b][j]),score=evaluate(r.model,row);outputs.push(score);evaluations.push({call_index:evaluations.length,mask,background_index:mask===full?null:b,row,score});}values.push(outputs.every(x=>x===outputs[0])?outputs[0]:sum(outputs.map(x=>x/outputs.length)));}
 const members=r.groups.map(x=>x.features),groupOrders=perms(members.map((_,i)=>i)),outerCounts=members.map(()=>new Map()),innerCounts=Array.from({length:d},()=>new Map()),counts=Array.from({length:d},()=>new Map()),paths=[];
 for(const row of members)for(const order of perms(row)){let mask=0;for(const j of order){innerCounts[j].set(mask,(innerCounts[j].get(mask)||0)+1);mask|=1<<j;}}
 const product=arrays=>arrays.reduce((out,a)=>out.flatMap(p=>a.map(v=>[...p,v])),[[]]);
 for(const outer of groupOrders){let om=0;for(const k of outer){outerCounts[k].set(om,(outerCounts[k].get(om)||0)+1);om|=1<<k;}
 for(const blocks of product(outer.map(k=>perms(members[k])))){let mask=0;const increments=Array(d).fill(0),steps=[],order=[];for(let b=0;b<blocks.length;b++)for(const j of blocks[b]){counts[j].set(mask,(counts[j].get(mask)||0)+1);const next=mask|1<<j,delta=values[next]-values[mask];increments[j]=delta;order.push(j);steps.push({group_index:outer[b],feature_index:j,from_mask:mask,to_mask:next,before:values[mask],after:values[next],delta});mask=next;}const total=sum(increments);paths.push({group_order:outer.slice(),feature_order:order,steps,increments,sum:total,efficiency_residual:sum([values[0],total,-values[full]])});}}
 const featureTerms=[],phis=[];
 for(let j=0;j<d;j++){const k=members.findIndex(row=>row.includes(j)),others=members[k].filter(i=>i!==j),key=mask=>{let om=0,im=0;members.forEach((row,q)=>{if(q!==k&&row.some(i=>mask&(1<<i)))om|=1<<q;});others.forEach((i,q)=>{if(mask&(1<<i))im|=1<<q;});return [om,im];},beforeMasks=[...counts[j].keys()].sort((a,b)=>key(a)[0]-key(b)[0]||key(a)[1]-key(b)[1]),row=[];
 for(const before of beforeMasks){const after=before|1<<j,outer=key(before)[0],within=members[k].reduce((v,i)=>v|(before&(1<<i)),0),outerDen=groupOrders.length/outerCounts[k].get(outer),innerDen=factorial(members[k].length)/innerCounts[j].get(within),den=paths.length/counts[j].get(before),delta=values[after]-values[before];row.push({outer_group_mask:outer,within_feature_mask:within,from_mask:before,to_mask:after,weight_numerator:1,weight_denominator:den,outer_weight_denominator:outerDen,inner_weight_denominator:innerDen,before:values[before],after:values[after],delta,weighted_delta:delta/den});}
 featureTerms.push(row);phis.push(sum(row.map(t=>t.weighted_delta)));}
 const unions=Array.from({length:1<<g},(_,mask)=>members.reduce((v,row,k)=>mask&(1<<k)?row.reduce((v,j)=>v|(1<<j),v):v,0)),quotient=unions.map(m=>values[m]),groupTerms=[],groupValues=[];
 for(let k=0;k<g;k++){const row=[...outerCounts[k].keys()].sort((a,b)=>a-b).map(before=>{const after=before|1<<k,den=groupOrders.length/outerCounts[k].get(before),delta=quotient[after]-quotient[before];return {from_group_mask:before,to_group_mask:after,weight_denominator:den,delta,weighted_delta:delta/den};});groupTerms.push(row);groupValues.push(sum(row.map(t=>t.weighted_delta)));}
 const groupSums=members.map(row=>sum(row.map(j=>phis[j]))),total=sum(phis),allocation={method:'exact_two_level_owen',feature_names:r.feature_names.slice(),group_names:r.groups.map(x=>x.name),group_feature_indices:members.map(x=>x.slice()),coalition_values:values.slice(),attributions:phis,feature_terms:featureTerms,group_coalition_values:quotient,group_shapley_attributions:groupValues,group_terms:groupTerms,within_group_sums:groupSums,group_efficiency_residuals:groupSums.map((x,k)=>sum([x,-groupValues[k]])),baseline:values[0],prediction:values[full],attribution_sum:total,efficiency_residual:sum([values[0],total,-values[full]]),admissible_orderings:paths.length,scope:root.OWEN_WORDING.SCOPE};
 const ordinaryCounts=Array.from({length:d},()=>new Map()),ordinaryOrders=perms(Array.from({length:d},(_,j)=>j));for(const order of ordinaryOrders){let mask=0;for(const j of order){ordinaryCounts[j].set(mask,(ordinaryCounts[j].get(mask)||0)+1);mask|=1<<j;}}
 const ordinaryTerms=ordinaryCounts.map((counts,j)=>[...counts.keys()].sort((a,b)=>a-b).map(before=>{const after=before|1<<j,den=ordinaryOrders.length/counts.get(before),delta=values[after]-values[before];return {from_mask:before,to_mask:after,weight_denominator:den,delta,weighted_delta:delta/den};})),ordinaryPhis=ordinaryTerms.map(row=>sum(row.map(t=>t.weighted_delta))),ordinaryTotal=sum(ordinaryPhis),ordinary={attributions:ordinaryPhis,feature_terms:ordinaryTerms,group_sums:members.map(row=>sum(row.map(j=>ordinaryPhis[j]))),attribution_sum:ordinaryTotal,efficiency_residual:sum([values[0],ordinaryTotal,-values[full]])};
 return {schema:'shapleylab-owen-audit-v1',request:r,allocation,ordinary_individual:ordinary,replacement_evaluations:evaluations,model_evaluations:evaluations.length,additional_allocation_model_calls:0,admissible_paths:paths,scope:root.OWEN_WORDING.REPORT_SCOPE};
}
const api={inspect,parse,validate,sum,evaluate};root.OwenEngine=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
