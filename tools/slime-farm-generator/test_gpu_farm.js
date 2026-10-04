'use strict';
// Generate an explicit browser test page. Tests run only after its button is clicked.
const fs=require('node:fs'),path=require('node:path');
const source=['core.js','browser_compute.js'].map(name=>fs.readFileSync(path.join(__dirname,name),'utf8')).join('\n');
const worker=source+`\nself.onmessage=async()=>{try{
 const clean=r=>{const {elapsedMs,execution,...rest}=r;return Object.fromEntries(Object.entries(rest).sort(([a],[b])=>a.localeCompare(b)));};let passed=0;
 for(const seed of ['0','-1','9007199254740993'])for(const range of [1,257,4000])for(const imported of [false,true]){
  const centerX=17,centerZ=-19;
  const biomes=imported?{y:1,read:1,chunks:new Map([['1,-2',new Uint8Array(16).fill(1)]]),boxes:[[10,0,-30,30,2,-10]]}:null;
  const p={seed,range,centerX,centerZ,compute:'gpu',spawnY:1,biomes};
  const actual=await SlimeLocalCompute.run('farm',p);
  const expected=SlimeFarm.search(seed,range,()=>{},biomes,1,centerX,centerZ);
  if(actual.execution.strategy!=='gpu-farm-screen'||JSON.stringify(clean(actual))!==JSON.stringify(clean(expected)))throw new Error(JSON.stringify({p,actual:clean(actual),expected:clean(expected)}));
  self.postMessage({passed:++passed,seed,range,imported});
 }
 self.postMessage({done:true,passed});
}catch(e){self.postMessage({error:e.message});}};`;
const json=JSON.stringify(worker).replace(/</g,String.fromCharCode(92)+'u003c');
const html=`<!doctype html><meta charset="utf-8"><title>GPU farm oracle check</title><h1>GPU farm oracle check</h1><button id="start">运行 GPU 对照</button><pre id="result">尚未运行</pre><script>
document.querySelector('#start').onclick=()=>{const out=document.querySelector('#result');out.textContent='RUNNING';const url=URL.createObjectURL(new Blob([${json}],{type:'text/javascript'}));const w=new Worker(url);URL.revokeObjectURL(url);w.onmessage=e=>{out.textContent=JSON.stringify(e.data,null,2);if(e.data.done||e.data.error)w.terminate();};w.onerror=e=>{out.textContent='ERROR '+e.message;w.terminate();};w.postMessage({});};</script>`;
const output=process.argv[2];if(!output)throw new Error('Pass a temporary HTML output path.');
fs.writeFileSync(output,html);console.log('Created GPU oracle browser page: '+output);
