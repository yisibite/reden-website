'use strict';
// Generate an explicit browser test page. Tests run only after its button is clicked.
const fs=require('node:fs'),path=require('node:path');
const source=['core.js','browser_compute.js'].map(name=>fs.readFileSync(path.join(__dirname,name),'utf8')).join('\n');
const worker=source+`\nself.onmessage=async()=>{try{
 const clean=r=>{const {elapsedMs,execution,...rest}=r;return Object.fromEntries(Object.entries(rest).sort(([a],[b])=>a.localeCompare(b)));};let passed=0;
 const results=[];
 for(const range of [4000,108000,216000]){
  const p={seed:'0',range,centerX:0,centerZ:0,compute:'gpu',spawnY:1};
  let begun=performance.now();const b=SlimeFarm.searchBounds(range,0,0);for(const k of ['cmin','zmin'])b[k]-=10;for(const k of ['cmax','zmax'])b[k]+=10;
  const prepared=await SlimeLocalCompute.grid('0',b);
  const baseline=SlimeFarm.withGrid(prepared,()=>SlimeFarm.search('0',range));const oldMs=performance.now()-begun;
  const actual=await SlimeLocalCompute.run('farm',p);const fields=r=>({score:r.score,x:r.x,z:r.z,ties:r.ties});
  if(JSON.stringify(fields(actual))!==JSON.stringify(fields(baseline)))throw new Error('Benchmark result mismatch');
  results.push({range,oldMs,newMs:actual.elapsedMs, ...fields(actual),readbackBytes:actual.execution.candidateReadbackBytes});self.postMessage({results});
 }
 self.postMessage({done:true,results});

}catch(e){self.postMessage({error:e.message});}};`;
const json=JSON.stringify(worker).replace(/</g,String.fromCharCode(92)+'u003c');
const html=`<!doctype html><meta charset="utf-8"><title>GPU farm oracle check</title><h1>GPU farm oracle check</h1><button id="start">运行 GPU 对照</button><pre id="result">尚未运行</pre><script>
document.querySelector('#start').onclick=()=>{const out=document.querySelector('#result');out.textContent='RUNNING';const url=URL.createObjectURL(new Blob([${json}],{type:'text/javascript'}));const w=new Worker(url);URL.revokeObjectURL(url);w.onmessage=e=>{out.textContent=JSON.stringify(e.data,null,2);if(e.data.done||e.data.error)w.terminate();};w.onerror=e=>{out.textContent='ERROR '+e.message;w.terminate();};w.postMessage({});};</script>`;
const output=process.argv[2];if(!output)throw new Error('Pass a temporary HTML output path.');
fs.writeFileSync(output,html);console.log('Created GPU oracle browser page: '+output);
