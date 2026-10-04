'use strict';
// Generate an explicit browser test page. Tests run only after its button is clicked.
const fs=require('node:fs'),path=require('node:path');
const source=['core.js','browser_compute.js'].map(name=>fs.readFileSync(path.join(__dirname,name),'utf8')).join('\n');
const worker=source+`\nself.onmessage=async()=>{try{
 const clean=r=>{const {elapsedMs,execution,...rest}=r;return Object.fromEntries(Object.entries(rest).sort(([a],[b])=>a.localeCompare(b)));};let passed=0;
 for(const [centerX,centerZ] of [[17,-19],[29996000,-29996000]])for(const seed of ['0','-1','9007199254740993'])for(const shape of ['square','rectangle'])for(const range of [1,257,2100]){
  const p={seed,range,shape,target:'slime',centerX,centerZ,compute:'gpu'};
  const actual=await SlimeLocalCompute.run('cluster',p);
  const expected=SlimeFarm.searchChunkCluster(seed,range,'slime',shape,()=>{},null,centerX,centerZ);
  if(actual.execution.strategy!=='gpu-shape'||JSON.stringify(clean(actual))!==JSON.stringify(clean(expected)))throw new Error(JSON.stringify({p,actual,expected}));
  self.postMessage({passed:++passed,p});
 }
 self.postMessage({done:true,passed});
}catch(e){self.postMessage({error:e.message});}};`;
const json=JSON.stringify(worker).replace(/</g,'\u003c');
const html=`<!doctype html><meta charset="utf-8"><title>GPU shape oracle check</title><h1>GPU shape oracle check</h1><button id="start">运行 GPU 对照</button><pre id="result">尚未运行</pre><script>
document.querySelector('#start').onclick=()=>{const out=document.querySelector('#result');out.textContent='RUNNING';const url=URL.createObjectURL(new Blob([${json}],{type:'text/javascript'}));const w=new Worker(url);URL.revokeObjectURL(url);w.onmessage=e=>{out.textContent=JSON.stringify(e.data,null,2);if(e.data.done||e.data.error)w.terminate();};w.onerror=e=>{out.textContent='ERROR '+e.message;w.terminate();};w.postMessage({});};</script>`;
const output=process.argv[2];if(!output)throw new Error('Pass a temporary HTML output path.');
fs.writeFileSync(output,html);console.log('Created GPU oracle browser page: '+output);
