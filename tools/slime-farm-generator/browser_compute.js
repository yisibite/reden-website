/* Browser-only acceleration. No fetch, uploads, backend or CUDA dependency. */
'use strict';
const SlimeLocalCompute = (() => {
  const shader = `
struct Params { width:u32, height:u32, unused0:u32, unused1:u32 }
@group(0) @binding(0) var<storage,read> xs:array<vec2<u32>>;
@group(0) @binding(1) var<storage,read> zs:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> flags:array<u32,256>;
fn highProduct(a:u32,b:u32)->u32 {
 let a0=a&65535u; let a1=a>>16u; let b0=b&65535u; let b1=b>>16u;
 let w0=a0*b0; let t=a1*b0+(w0>>16u);
 let w1=(t&65535u)+a0*b1;
 return a1*b1+(t>>16u)+(w1>>16u);
}
fn next(s:vec2<u32>)->vec2<u32> {
 let product=s.x*0xdeece66du; let low=product+11u;
 let carry=select(0u,1u,low<product);
 let hi=(s.y*0xdeece66du+s.x*5u+highProduct(s.x,0xdeece66du)+carry)&65535u;
 return vec2<u32>(low,hi);
}
@compute @workgroup_size(256)
fn grid(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
 let i=id.x+id.y*p.unused0; flags[lane]=0u;
 if(i<p.width*p.height){
 let a=xs[i%p.width]; let b=zs[i/p.width]; let lo=a.x+b.x;
 var s=vec2<u32>(lo,(a.y+b.y+select(0u,1u,lo<a.x))&65535u);
 s=s^vec2<u32>(987234911u^0xdeece66du,5u);
 loop {
  s=next(s); let bits=(s.x>>17u)|(s.y<<15u); let v=bits%10u;
  if(bits<2147483640u){flags[lane]=select(0u,1u<<(lane&31u),v==0u);break;}
 }
 }
 workgroupBarrier();
 if((lane&31u)==0u && i<p.width*p.height){
  var packed=0u;
  for(var j=0u;j<32u;j++){packed|=flags[lane+j];}
  output[i/32u]=packed;
 }
}`;
  // Exact maximum shapes, not a fixed-size/threshold substitute. Histogram state
  // survives batches; every positive column is considered on the device.
  const shapeShader=`
struct ShapeParams { width:u32, row:u32, base:u32, square:u32 }
struct Candidate { low:u32, high:u32, x:u32, z:u32, width:u32, height:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> bits:array<u32>;
@group(0) @binding(1) var<storage,read_write> heights:array<u32>;
@group(0) @binding(2) var<storage,read_write> best:array<Candidate>;
@group(0) @binding(3) var<uniform> p:ShapeParams;
var<workgroup> choices:array<Candidate,256>;
${shader.slice(shader.indexOf('fn highProduct'),shader.indexOf('fn next'))}
fn better(a:Candidate,b:Candidate)->bool {
 if(a.high!=b.high){return a.high>b.high;}
 if(a.low!=b.low){return a.low>b.low;}
 if(a.z!=b.z){return a.z<b.z;}
 if(a.x!=b.x){return a.x<b.x;}
 if(a.z+a.height!=b.z+b.height){return a.z+a.height<b.z+b.height;}
 return a.x+a.width<b.x+b.width;
}
@compute @workgroup_size(256)
fn advance(@builtin(global_invocation_id) id:vec3<u32>){
 let x=id.x;if(x>=p.width){return;}
 let i=p.row*p.width+x;
 heights[x]=select(0u,heights[x]+1u,((bits[i>>5u]>>(i&31u))&1u)!=0u);
}
@compute @workgroup_size(256)
fn score(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 let x=id.x;var c=Candidate(0u,0u,0u,0u,0u,0u,0u,0u);
 if(x<p.width){
  let h=heights[x];let z=p.base+p.row;
  if(h>0u){
   if(p.square!=0u){
    var side=0u;var minimum=h;
    loop {
     if(side>=h || side>x){break;}
     minimum=min(minimum,heights[x-side]);
     if(minimum<side+1u){break;}
     side+=1u;
    }
    c=Candidate(side*side,highProduct(side,side),x-side+1u,z-side+1u,side,side,0u,0u);
   }else{
    var left=x;var right=x+1u;
    loop {if(left==0u){break;}if(heights[left-1u]<h){break;}left-=1u;}
    loop {if(right>=p.width){break;}if(heights[right]<h){break;}right+=1u;}
    let w=right-left;c=Candidate(w*h,highProduct(w,h),left,z-h+1u,w,h,0u,0u);
   }
  }
 }
 choices[lane]=c;workgroupBarrier();
 var stride=128u;
 loop {
  if(lane<stride){let other=choices[lane+stride];if(better(other,choices[lane])){choices[lane]=other;}}
  workgroupBarrier();if(stride==1u){break;}stride>>=1u;
 }
 if(lane==0u && better(choices[0],best[group.x])){best[group.x]=choices[0];}
}`;
  const mask=(1n<<48n)-1n;
  const pair=(array,i,value)=>{value&=mask;array[i*2]=Number(value&0xffffffffn);array[i*2+1]=Number(value>>32n);};
  function reduceCandidates(words,best={count:0,minX:0,minZ:0,width:0,height:0}){
    for(let i=0;i<words.length;i+=8){
      const count=words[i]+words[i+1]*4294967296,minX=words[i+2],minZ=words[i+3],width=words[i+4],height=words[i+5];
      if(count>best.count||count===best.count&&count>0&&(minZ<best.minZ||minZ===best.minZ&&(minX<best.minX||minX===best.minX&&(minZ+height<best.minZ+best.height||minZ+height===best.minZ+best.height&&minX+width<best.minX+best.width))))best={count,minX,minZ,width,height};
    }
    return best;
  }
  async function gpuShapes(data,onProgress=()=>{},gpu=globalThis.navigator?.gpu){
    if(!gpu)throw new Error('浏览器未提供 WebGPU；请使用本机 CPU 或支持 WebGPU 的安全页面。');
    const started=performance.now(),{seed,range,shape,centerX=0,centerZ=0}=data;
    const bounds=SlimeFarm.searchBounds(range,centerX,centerZ),n=bounds.cmax-bounds.cmin+1,h=bounds.zmax-bounds.zmin+1;
    const adapter=await gpu.requestAdapter();if(!adapter)throw new Error('没有可用的本机 WebGPU 设备。');
    const device=await adapter.requestDevice(),buffers=[];
    let lost=null;device.lost.then(info=>{lost=info.message||'GPU 设备已断开';});
    try{
      if(n*8>device.limits.maxStorageBufferBindingSize)throw new Error('GPU 不支持该搜索行宽。');
      const groups=Math.ceil(n/256),batchRows=Math.max(1,Math.min(128,Math.floor(device.limits.maxStorageBufferBindingSize*8/n),Math.floor(0xffffffff/n)));
      const flagBytes=Math.ceil(n*batchRows/32)*4,candidateBytes=groups*32;
      const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
      const xbuf=make(n*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),zbuf=make(batchRows*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const flags=make(flagBytes,GPUBufferUsage.STORAGE),heights=make(n*4,GPUBufferUsage.STORAGE);
      const bestBuffer=make(candidateBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(candidateBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const gridParams=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),shapeParams=make(batchRows*256,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      device.pushErrorScope('validation');
      const gridModule=device.createShaderModule({code:shader}),shapeModule=device.createShaderModule({code:shapeShader});
      for(const module of [gridModule,shapeModule]){const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>m.message).join('; '));}
      const pipelines=[];for(const [module,entryPoint] of [[gridModule,'grid'],[shapeModule,'advance'],[shapeModule,'score']])pipelines.push(await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}}));
      const error=await device.popErrorScope();if(error)throw new Error(error.message);
      const gridBind=device.createBindGroup({layout:pipelines[0].getBindGroupLayout(0),entries:[xbuf,zbuf,flags,gridParams].map((buffer,binding)=>({binding,resource:{buffer}}))});
      // Auto-layout omits bindings unused by an entry point.
      const rowBinds=Array.from({length:batchRows},(_,row)=>pipelines.slice(1).map((pipeline,index)=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:(index===0?[0,1,3]:[1,2,3]).map(binding=>({binding,resource:binding===3?{buffer:shapeParams,offset:row*256,size:16}:{buffer:[flags,heights,bestBuffer][binding]}}))})));
      const xs=new Uint32Array(n*2),zs=new Uint32Array(batchRows*2),params=new Uint32Array(batchRows*64),b=BigInt(seed);
      for(let x=0;x<n;x++){const c=bounds.cmin+x;pair(xs,x,b+BigInt(Math.imul(Math.imul(c,c),4987142))+BigInt(Math.imul(c,5947611)));}
      device.queue.writeBuffer(xbuf,0,xs);
      let best={count:0,minX:0,minZ:0,width:0,height:0},readbackBytes=0;
      for(let row=0;row<h;row+=batchRows){
        if(lost)throw new Error(lost);
        const rows=Math.min(batchRows,h-row),workgroups=Math.ceil(n*rows/256),gx=Math.min(workgroups,device.limits.maxComputeWorkgroupsPerDimension);
        for(let z=0;z<rows;z++){const c=bounds.zmin+row+z;pair(zs,z,BigInt(Math.imul(c,c))*4392871n+BigInt(Math.imul(c,389711)));params.set([n,z,row,shape==='square'?1:0],z*64);}
        device.queue.writeBuffer(zbuf,0,zs);device.queue.writeBuffer(gridParams,0,new Uint32Array([n,rows,gx*256,0]));device.queue.writeBuffer(shapeParams,0,params);
        device.pushErrorScope('validation');
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(pipelines[0]);pass.setBindGroup(0,gridBind);pass.dispatchWorkgroups(gx,Math.ceil(workgroups/gx));
        for(let z=0;z<rows;z++)for(let index=0;index<2;index++){pass.setPipeline(pipelines[index+1]);pass.setBindGroup(0,rowBinds[z][index]);pass.dispatchWorkgroups(groups);}
        pass.end();encoder.copyBufferToBuffer(bestBuffer,0,read,0,candidateBytes);device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);try{best=reduceCandidates(new Uint32Array(read.getMappedRange()),best);}finally{read.unmap();}
        const validation=await device.popErrorScope();if(validation)throw new Error(validation.message);
        readbackBytes+=candidateBytes;
        onProgress({phase:'shape',progress:(row+rows)/h,message:`本机 GPU 求最大${shape==='square'?'正方形':'矩形'}：${row+rows} / ${h} 行；当前最多 ${best.count} 个区块`});
      }
      if(lost)throw new Error(lost);
      const result={seed:String(seed),range,centerX,centerZ,target:'slime',shape,gridChunks:n*h,found:best.count>0,biome:null,elapsedMs:performance.now()-started,execution:{backend:'gpu',fallbackReason:'',strategy:'gpu-shape',peakGridCells:0,stateBytes:n*4,readbackBytes}};
      if(!best.count)return result;
      const chunks={minX:bounds.cmin+best.minX,maxX:bounds.cmin+best.minX+best.width-1,minZ:bounds.zmin+best.minZ,maxZ:bounds.zmin+best.minZ+best.height-1};
      return {...result,count:best.count,width:best.width,height:best.height,chunks,blocks:{minX:chunks.minX*16,maxX:chunks.maxX*16+15,minZ:chunks.minZ*16,maxZ:chunks.maxZ*16+15},coordinates:null,rectangles:null};
    }finally{for(const b of buffers)b.destroy();device.destroy();}
  }
  async function grid(seed,bounds,onProgress=()=>{},gpu=globalThis.navigator?.gpu,consume=null) {
    if(!gpu)throw new Error('浏览器未提供 WebGPU；请使用本机 CPU 或支持 WebGPU 的安全页面。');
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error('没有可用的本机 WebGPU 设备。');
    const device=await adapter.requestDevice(),buffers=[];
    let lost=null;device.lost.then(info=>{lost=info.message||'GPU 设备已断开';});
    try {
      const {cmin,cmax,zmin,zmax}=bounds,n=cmax-cmin+1,h=zmax-zmin+1;
      if(![n,h].every(v=>Number.isInteger(v)&&v>0)||!consume&&n*h>800000000)throw new Error('区块网格尺寸超出支持范围。');
      if(n*8>device.limits.maxStorageBufferBindingSize)throw new Error('GPU 不支持该搜索行宽。');
      const batchRows=Math.max(1,Math.min(128,Math.floor(1048576/n),Math.floor(device.limits.maxComputeWorkgroupsPerDimension*256/n)));
      if(n*batchRows>device.limits.maxComputeWorkgroupsPerDimension*256)throw new Error('GPU 单行计算超出设备限制。');
      const batchCells=n*batchRows,maxBytes=Math.ceil(batchCells/32)*4;
      const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
      const xbuf=make(n*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const zbuf=make(batchRows*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const out=make(maxBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      const read=make(maxBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const params=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      device.pushErrorScope('validation');
      const module=device.createShaderModule({code:shader});
      const info=await module.getCompilationInfo();
      const errors=info.messages.filter(m=>m.type==='error');
      if(errors.length)throw new Error(errors.map(m=>m.message).join('; '));
      const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'grid'}});
      const validation=await device.popErrorScope();if(validation)throw new Error(validation.message);
      const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[xbuf,zbuf,out,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const xs=new Uint32Array(n*2),zs=new Uint32Array(batchRows*2),result=new Uint8Array(consume?batchCells:n*h),b=BigInt(seed);
      for(let x=0;x<n;x++){const c=cmin+x;pair(xs,x,b+BigInt(Math.imul(Math.imul(c,c),4987142))+BigInt(Math.imul(c,5947611)));}
      device.queue.writeBuffer(xbuf,0,xs);
      for(let row=0;row<h;row+=batchRows){
        if(lost)throw new Error(lost);
        const rows=Math.min(batchRows,h-row),cells=n*rows,bytes=Math.ceil(cells/32)*4;
        for(let z=0;z<rows;z++){const c=zmin+row+z;pair(zs,z,BigInt(Math.imul(c,c))*4392871n+BigInt(Math.imul(c,389711)));}
        device.queue.writeBuffer(zbuf,0,zs);device.queue.writeBuffer(params,0,new Uint32Array([n,rows,0,0]));
        device.pushErrorScope('validation');
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(cells/256));pass.end();
        encoder.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ,0,bytes);
        try{const words=new Uint32Array(read.getMappedRange(0,bytes));
          for(let i=0;i<cells;i++)result[(consume?0:row*n)+i]=(words[i>>>5]>>>(i&31))&1;
        }finally{read.unmap();}
        const error=await device.popErrorScope();if(error)throw new Error(error.message);
        if(consume)consume({grid:result.subarray(0,cells),minChunk:cmin,maxChunk:cmax,minChunkZ:zmin+row,maxChunkZ:zmin+row+rows-1,n,h:rows});
        onProgress({phase:'grid',progress:consume?(row+rows)/h:0.05+0.25*(row+rows)/h,message:`本机 GPU ${consume?'逐行搜索':'计算史莱姆区块'}：${row+rows} / ${h} 行`});
      }
      if(lost)throw new Error(lost);
      return consume?null:{seed:String(seed),grid:result,minChunk:cmin,maxChunk:cmax,minChunkZ:zmin,maxChunkZ:zmax,n,h};
    } finally {for(const b of buffers)b.destroy();device.destroy();}
  }
  async function stream(data,onProgress){
    const {seed,range,centerX=0,centerZ=0,target,shape,biomes,compute='auto'}=data;
    let scan=SlimeFarm.chunkRowScan(seed,range,target,shape,biomes,centerX,centerZ),backend='cpu',reason='',peakGridCells=0;
    const accept=g=>{peakGridCells=Math.max(peakGridCells,g.grid.length);scan.consume(g);};
    if(compute==='gpu'||compute==='auto'&&scan.n*scan.h>=4194304){
      try{if(target==='slime')return await gpuShapes(data,onProgress);await grid(seed,scan.bounds,onProgress,globalThis.navigator?.gpu,accept);backend='gpu';}
      catch(error){
        if(compute==='gpu')throw error;
        reason=error.message;scan=SlimeFarm.chunkRowScan(seed,range,target,shape,biomes,centerX,centerZ);
        onProgress({phase:'grid',progress:0,message:'本机 GPU 不可用，从首行重新以本机 CPU 搜索…'});
      }
    }
    if(backend==='cpu'){
      const rows=Math.max(1,Math.min(128,Math.floor(1048576/scan.n))),b=scan.bounds;
      for(let row=0;row<scan.h;row+=rows){
        const count=Math.min(rows,scan.h-row);
        accept(SlimeFarm.makeGrid(seed,b.cmin,b.cmax,()=>{},b.zmin+row,b.zmin+row+count-1));
        onProgress({phase:'shape',progress:(row+count)/scan.h,message:`本机 CPU 逐行搜索：${row+count} / ${scan.h} 行`});
        await new Promise(resolve=>setTimeout(resolve,0));
      }
    }
    const result=scan.finish();result.execution={backend,fallbackReason:reason,strategy:'row-stream',peakGridCells,stateBytes:scan.stateBytes};return result;
  }
  const farmShader=`
struct P {n:u32,cols:u32,rows:u32,threshold:u32}
@group(0) @binding(0) var<storage,read> flags:array<u32>;
@group(0) @binding(1) var<storage,read_write> vertical:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<u32>;
@group(0) @binding(3) var<storage,read_write> hits:array<vec3<u32>>;
@group(0) @binding(4) var<storage,read_write> count:atomic<u32>;
@group(0) @binding(5) var<uniform> p:P;
fn flag(i:u32)->u32{return (flags[i>>5u]>>(i&31u))&1u;}
@compute @workgroup_size(256)
fn columns(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p.n*p.rows){return;}var sum=0u;
 for(var z=0u;z<17u;z++){sum+=flag(i+z*p.n);}vertical[i]=sum;
}
@compute @workgroup_size(256)
fn screen(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p.cols*p.rows){return;}let x=i%p.cols+2u;let row=i/p.cols;
 var sum=0u;for(var dx=0u;dx<17u;dx++){sum+=vertical[row*p.n+x+dx];}
 if(sum*256u<p.threshold){return;}
 var upper=0u;for(var dz=0u;dz<17u;dz++){for(var dx=0u;dx<17u;dx++){upper+=flag((row+dz)*p.n+x+dx)*weights[dz*17u+dx];}}
 if(upper>=p.threshold){let slot=atomicAdd(&count,1u);hits[slot]=vec3<u32>(x-2u,row,upper);}
}`;
  async function farm(data,prepared,onProgress){
    const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('没有可用 GPU。');
    const device=await adapter.requestDevice(),buffers=[];let lost=null;
    device.lost.then(info=>{lost=info.message||'GPU 设备已断开';});
    const steps=SlimeFarm.searchSteps(data.seed,data.range,onProgress,data.biomes,data.spawnY,data.centerX||0,data.centerZ||0);
    // Iterator execution must remain inside withGrid while it obtains its initial grid.
    let next;const started=performance.now();
    try{
      next=SlimeFarm.withGrid(prepared,()=>steps.next());
      const {g,k,cn}=next.value,n=g.n,batchRows=128,cells=cn*batchRows;
      const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
      const input=make(Math.ceil(n*(batchRows+16)/32)*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),vertical=make(n*batchRows*4,GPUBufferUsage.STORAGE);
      const weights=make(289*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),hits=make(cells*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      const count=make(4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),params=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      const read=make(cells*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const xs=new Uint32Array(n*2),zs=new Uint32Array((batchRows+16)*2),seed=BigInt(data.seed);
      for(let x=0;x<n;x++){const c=g.minChunk+x;pair(xs,x,seed+BigInt(Math.imul(Math.imul(c,c),4987142))+BigInt(Math.imul(c,5947611)));}
      const xbuf=make(xs.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),zbuf=make(zs.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),gridParams=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(xbuf,0,xs);
      const gridModule=device.createShaderModule({code:shader});const gridPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:gridModule,entryPoint:'grid'}});
      const gridBind=device.createBindGroup({layout:gridPipeline.getBindGroupLayout(0),entries:[xbuf,zbuf,input,gridParams].map((buffer,binding)=>({binding,resource:{buffer}}))});
      device.pushErrorScope('validation');const module=device.createShaderModule({code:farmShader});
      const info=await module.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw new Error(info.messages.map(m=>m.message).join('; '));
      const pipelines=[];for(const entryPoint of ['columns','screen'])pipelines.push(await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}}));
      const bind=pipelines.map((pipeline,i)=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:(i===0?[0,1,5]:[0,1,2,3,4,5]).map(binding=>({binding,resource:{buffer:[input,vertical,weights,hits,count,params][binding]}}))}));
      let error=await device.popErrorScope();if(error)throw new Error(error.message);
      device.queue.writeBuffer(weights,0,new Uint32Array(k.upper));let transferred=0,cachedUntil=0,cache=[];
      while(!next.done){
        if(lost)throw new Error(lost);const {band,zn,best}=next.value;
        if(band<cachedUntil){next=steps.next(cache.filter(c=>c.tz>=band&&c.tz<band+16&&c.bound>=best));continue;}
        const rows=Math.min(batchRows,zn-band);
        for(let z=0;z<rows+16;z++){const c=g.minChunkZ+band+2+z;pair(zs,z,BigInt(Math.imul(c,c))*4392871n+BigInt(Math.imul(c,389711)));}
        device.queue.writeBuffer(zbuf,0,zs);device.queue.writeBuffer(gridParams,0,new Uint32Array([n,rows+16,0,0]));
        device.queue.writeBuffer(count,0,new Uint32Array([0]));device.queue.writeBuffer(params,0,new Uint32Array([n,cn,rows,Math.max(0,best)]));
        device.pushErrorScope('validation');let encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(gridPipeline);pass.setBindGroup(0,gridBind);pass.dispatchWorkgroups(Math.ceil(n*(rows+16)/256));
        for(let i=0;i<2;i++){pass.setPipeline(pipelines[i]);pass.setBindGroup(0,bind[i]);pass.dispatchWorkgroups(Math.ceil((i===0?n:cn)*rows/256));}pass.end();
        encoder.copyBufferToBuffer(count,0,read,0,4);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ,0,4);
        let length;try{length=new Uint32Array(read.getMappedRange(0,4))[0];}finally{read.unmap();}
        if(length>cn*rows)throw new Error('GPU 候选数量异常。');const candidates=[];
        if(length){encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(hits,0,read,0,length*16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ,0,length*16);
          try{const a=new Uint32Array(read.getMappedRange(0,length*16));for(let i=0;i<length;i++)candidates.push({tx:a[i*4],tz:band+a[i*4+1],bound:a[i*4+2]});}finally{read.unmap();}}
        error=await device.popErrorScope();if(error)throw new Error(error.message);transferred+=4+length*16;
        cache=candidates;cachedUntil=band+rows;next=steps.next(cache.filter(c=>c.tz<band+16&&c.bound>=best));
      }
      const result=next.value;result.elapsedMs=performance.now()-started;result.execution={backend:'gpu',fallbackReason:'',strategy:'gpu-farm-screen',candidateReadbackBytes:transferred};return result;
    }finally{for(const b of buffers)b.destroy();device.destroy();}
  }
  async function run(kind,data,onProgress=()=>{}) {
    const started=performance.now(),{seed,range,centerX=0,centerZ=0,compute='auto'}=data;
    if(!['auto','cpu','gpu'].includes(compute))throw new Error('未知本机计算方式。');
    if(kind==='cluster'&&['square','rectangle'].includes(data.shape))return stream(data,onProgress);
    if(!Number.isInteger(range)||range<1||range>SlimeFarm.MAX_SEARCH_RANGE)throw new Error('搜索范围必须是 1～216000 的整数。');
    const bounds=SlimeFarm.searchBounds(range,centerX,centerZ),halo=kind==='farm'||data.shape==='spawnRange'?10:0;
    for(const k of ['cmin','zmin'])bounds[k]-=halo;
    for(const k of ['cmax','zmax'])bounds[k]+=halo;
    const cells=(bounds.cmax-bounds.cmin+1)*(bounds.zmax-bounds.zmin+1);
    let prepared=null,reason='',backend='cpu';
    if(compute==='gpu'||compute==='auto'&&cells>=4194304){
      try{prepared=await grid(seed,bounds,onProgress);backend='gpu';if(kind==='farm'){const result=await farm(data,prepared,onProgress);result.elapsedMs=performance.now()-started;return result;}}
      catch(error){if(compute==='gpu')throw error;reason=error.message;onProgress({phase:'grid',progress:0,message:'本机 GPU 不可用，改用本机 CPU 计算…'});}
    }
    const result=SlimeFarm.withGrid(prepared,()=>kind==='farm'
      ?SlimeFarm.search(seed,range,onProgress,data.biomes,data.spawnY,centerX,centerZ)
      :data.shape==='spawnRange'?SlimeFarm.searchLeastSlime(seed,range,onProgress,data.biomes,centerX,centerZ)
      :SlimeFarm.searchChunkCluster(seed,range,data.target,data.shape,onProgress,data.biomes,centerX,centerZ));
    result.execution={backend,fallbackReason:reason};return result;
  }
  return {grid,run,reduceCandidates};
})();
if(typeof module!=='undefined')module.exports=SlimeLocalCompute;
