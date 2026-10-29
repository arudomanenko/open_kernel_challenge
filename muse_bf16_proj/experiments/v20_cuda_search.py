# okc: problem=muse_bf16_proj
"""H200 dense projections: measured GEMV, split-K router, and library GEMM paths.

BF16 weights retain their original values. Only the router uses the explicitly
permitted BF16 hi/lo representation, with separate FP32 accumulators.
"""
from typing import TypedDict
from threading import RLock

import torch


# Serialize configuration and launch among callers of this module. External
# DeepGEMM users must coordinate separately if they change runtime settings.
_DEEP_LOCK = RLock()


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nvoid gemv(uint64_t,uint64_t,uint64_t,int,int,int,int,bool,uint64_t);\nvoid gemm(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,bool,uint64_t);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("gemv",&gemv); m.def("gemm",&gemm); }\n'
_CUDA = '#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <mma.h>\n#include <cooperative_groups.h>\n#include <stdint.h>\n#include <stdexcept>\nnamespace wm = nvcuda::wmma;\nusing bf = __nv_bfloat16;\n\n__device__ __forceinline__ float warp_sum(float v) {\n  for(int s=16;s;s/=2) v += __shfl_down_sync(0xffffffff,v,s);\n  return v;\n}\ntemplate<bool FLOAT_W,int ROWS,int THREADS>\n__global__ void vec_gemv(const bf* __restrict__ x,const void* wp,void* yp,int N,int K) {\n  int tid=threadIdx.x, lane=tid%32, warp=tid/32;\n  constexpr int GROUP = THREADS/ROWS;\n  int row=blockIdx.x*ROWS+tid/GROUP, t=tid%GROUP;\n  float v=0;\n  if(row<N) {\n    for(int k=t*8;k<K;k+=GROUP*8) {\n      uint4 a=*reinterpret_cast<const uint4*>(x+k);\n      bf* av=reinterpret_cast<bf*>(&a);\n      if constexpr(FLOAT_W) {\n        float4 b0=*reinterpret_cast<const float4*>((const float*)wp+row*K+k);\n        float4 b1=*reinterpret_cast<const float4*>((const float*)wp+row*K+k+4);\n        float w[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};\n        #pragma unroll\n        for(int i=0;i<8;i++) v=fmaf(__bfloat162float(av[i]),w[i],v);\n      } else {\n        uint4 b=*reinterpret_cast<const uint4*>((const bf*)wp+row*K+k);\n        bf* bv=reinterpret_cast<bf*>(&b);\n        #pragma unroll\n        for(int i=0;i<8;i++) v=fmaf(__bfloat162float(av[i]),__bfloat162float(bv[i]),v);\n      }\n    }\n  }\n  v=warp_sum(v);\n  if constexpr(GROUP==32) {\n    if(lane==0 && row<N) {\n      if constexpr(FLOAT_W) ((float*)yp)[row]=v;\n      else ((bf*)yp)[row]=__float2bfloat16_rn(v);\n    }\n  } else {\n    __shared__ float sums[THREADS/32];\n    if(lane==0) sums[warp]=v;\n    __syncthreads();\n    if(t==0 && row<N) {\n      float z=0;\n      #pragma unroll\n      for(int j=0;j<GROUP/32;j++) z+=sums[(tid/GROUP)*(GROUP/32)+j];\n      if constexpr(FLOAT_W) ((float*)yp)[row]=z;\n      else ((bf*)yp)[row]=__float2bfloat16_rn(z);\n    }\n  }\n}\n\n__device__ __forceinline__ void cp16(void* dst,const void* src,bool valid) {\n#if __CUDA_ARCH__ >= 800\n  unsigned addr=__cvta_generic_to_shared(dst);\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(addr),"l"(src),"r"(valid?16:0));\n#endif\n}\n__device__ __forceinline__ void cpwait() {\n  asm volatile("cp.async.commit_group;\\n cp.async.wait_group 0;":::"memory");\n  __syncthreads();\n}\n\ntemplate<int BN,int BK,int THREADS,bool ROUTER,bool CLUSTER=false>\n__global__ void small_gemm(const bf* __restrict__ x,const bf* __restrict__ w,\n const bf* __restrict__ lo,float* __restrict__ p,void* __restrict__ y,int M,int N,int K,int SX,int split) {\n  __shared__ __align__(32) bf a[16][BK+8];\n  __shared__ __align__(32) bf b[BN][BK+8];\n  __shared__ __align__(32) bf bl[ROUTER?BN:1][BK+8];\n  __shared__ __align__(32) float out[16][BN];\n  int tid=threadIdx.x, warp=tid/32;\n  int mo=blockIdx.x*16,no=blockIdx.y*BN,si=blockIdx.z;\n  wm::fragment<wm::accumulator,16,16,16,float> acc, low;\n  wm::fill_fragment(acc,0.f); wm::fill_fragment(low,0.f);\n  for(int ko=si*BK;ko<K;ko+=split*BK) {\n    for(int i=tid;i<16*(BK/8);i+=THREADS) {\n      int r=i/(BK/8),c=(i%(BK/8))*8;\n      cp16(&a[r][c], x+(mo+r)*SX+ko+c, mo+r<M && ko+c<K);\n    }\n    for(int i=tid;i<BN*(BK/8);i+=THREADS) {\n      int r=i/(BK/8),c=(i%(BK/8))*8;\n      cp16(&b[r][c], w+(no+r)*K+ko+c, no+r<N && ko+c<K);\n      if constexpr(ROUTER) cp16(&bl[r][c],lo+(no+r)*K+ko+c,no+r<N && ko+c<K);\n    }\n    cpwait();\n    if(warp<BN/16) {\n      #pragma unroll\n      for(int kk=0;kk<BK;kk+=16) {\n        wm::fragment<wm::matrix_a,16,16,16,bf,wm::row_major> af;\n        wm::fragment<wm::matrix_b,16,16,16,bf,wm::col_major> bf1;\n        wm::load_matrix_sync(af,&a[0][kk],BK+8);\n        wm::load_matrix_sync(bf1,&b[warp*16][kk],BK+8);\n        wm::mma_sync(acc,af,bf1,acc);\n        if constexpr(ROUTER) {\n          wm::load_matrix_sync(bf1,&bl[warp*16][kk],BK+8);\n          wm::mma_sync(low,af,bf1,low);\n        }\n      }\n    }\n    __syncthreads();\n  }\n  if(warp<BN/16) {\n    if constexpr(ROUTER) {\n      #pragma unroll\n      for(int i=0;i<acc.num_elements;i++) acc.x[i]+=low.x[i];\n    }\n    wm::store_matrix_sync(&out[0][warp*16],acc,BN,wm::mem_row_major);\n  }\n  __syncthreads();\n  if constexpr(CLUSTER) {\n#if __CUDA_ARCH__ >= 900\n    auto cluster=cooperative_groups::this_cluster();\n    cluster.sync();\n    if(cluster.block_rank()==0) {\n      for(int i=tid;i<16*BN;i+=THREADS) {\n        int r=i/BN,c=i%BN;\n        float v=0;\n        for(int j=0;j<split;j++) {\n          float* remote=cluster.map_shared_rank(&out[0][0],j);\n          v+=remote[i];\n        }\n        if(mo+r<M && no+c<N) {\n          if constexpr(ROUTER) ((float*)y)[(mo+r)*N+no+c]=v;\n          else ((bf*)y)[(mo+r)*N+no+c]=__float2bfloat16_rn(v);\n        }\n      }\n    }\n    cluster.sync();\n#endif\n  } else {\n    for(int i=tid;i<16*BN;i+=THREADS) {\n      int r=i/BN,c=i%BN;\n      if(mo+r<M && no+c<N) p[si*M*N+(mo+r)*N+no+c]=out[r][c];\n    }\n  }\n}\ntemplate<bool FP32>\n__global__ void finish(const float* p,void* y,int size,int split) {\n  int i=blockIdx.x*256+threadIdx.x;\n  if(i<size) {\n    float v=0;\n    for(int s=0;s<split;s++) v+=p[s*size+i];\n    if constexpr(FP32) ((float*)y)[i]=v;\n    else ((bf*)y)[i]=__float2bfloat16_rn(v);\n  }\n}\nvoid gemv(uint64_t x,uint64_t w,uint64_t y,int n,int k,int rows,int threads,bool fp,uint64_t stream) {\n  cudaStream_t s=(cudaStream_t)stream;\n#define GV(R,T) if(rows==R && threads==T) {if(fp) vec_gemv<true,R,T><<<(n+R-1)/R,T,0,s>>>((bf*)x,(void*)w,(void*)y,n,k);else vec_gemv<false,R,T><<<(n+R-1)/R,T,0,s>>>((bf*)x,(void*)w,(void*)y,n,k);return;}\n  GV(1,128) GV(1,256) GV(4,128) GV(8,256) GV(2,128)\n  throw std::runtime_error("unsupported gemv configuration");\n}\nvoid gemm(uint64_t x,uint64_t w,uint64_t lo,uint64_t p,uint64_t y,int m,int n,int k,int sx,int bn,int bk,int split,bool router,uint64_t stream) {\n  cudaStream_t s=(cudaStream_t)stream;\n  bool cluster=split<0;\n  if(cluster) split=-split;\n  dim3 grid((m+15)/16,(n+bn-1)/bn,split);\n  if(cluster) {\n    cudaLaunchConfig_t cfg={};\n    cfg.gridDim=grid;cfg.blockDim=dim3(128);cfg.stream=s;\n    cudaLaunchAttribute attr={};\n    attr.id=cudaLaunchAttributeClusterDimension;\n    attr.val.clusterDim.x=1;attr.val.clusterDim.y=1;attr.val.clusterDim.z=split;\n    cfg.attrs=&attr;cfg.numAttrs=1;\n#define GC(N,K) if(bn==N && bk==K) {cudaError_t e; if(router) e=cudaLaunchKernelEx(&cfg,small_gemm<N,K,128,true,true>,(const bf*)x,(const bf*)w,(const bf*)lo,(float*)p,(void*)y,m,n,k,sx,split);else e=cudaLaunchKernelEx(&cfg,small_gemm<N,K,128,false,true>,(const bf*)x,(const bf*)w,(const bf*)lo,(float*)p,(void*)y,m,n,k,sx,split); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));return;}\n    GC(16,64) GC(32,64) GC(64,64) GC(16,128) GC(32,128) GC(64,128)\n    throw std::runtime_error("unsupported cluster configuration");\n  }\n#define GM(N,K) if(bn==N && bk==K) {if(router) small_gemm<N,K,128,true><<<grid,128,0,s>>>((bf*)x,(bf*)w,(bf*)lo,(float*)p,(void*)y,m,n,k,sx,split);else small_gemm<N,K,128,false><<<grid,128,0,s>>>((bf*)x,(bf*)w,(bf*)lo,(float*)p,(void*)y,m,n,k,sx,split);}\n  GM(16,64) else GM(32,64) else GM(64,64) else GM(16,128) else GM(32,128) else GM(64,128) else throw std::runtime_error("unsupported gemm configuration");\n  if(router) finish<true><<<(m*n+255)/256,256,0,s>>>((float*)p,(void*)y,m*n,split);\n  else finish<false><<<(m*n+255)/256,256,0,s>>>((float*)p,(void*)y,m*n,split);\n}\n'


def _run(x,layer,n,state,tactic):
    m,k=x.shape
    router=layer=="router"
    if tactic[0]=="torch":
        w=state["transposed"][layer][:,:n] if tactic[1] else state[layer][:n].T
        return torch.mm(x.float() if router else x,w)
    out=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if tactic[0]=="deep":
        dg=state["dg"]
        old_sms,old_det=dg.get_num_sms(),dg.get_deterministic_algorithms()
        try:
            dg.use_deterministic_algorithms(True)
            dg.set_num_sms(tactic[1])
            dg.bf16_gemm_nt(x,state[layer][:n],out)
        finally:
            dg.set_num_sms(old_sms);dg.use_deterministic_algorithms(old_det)
        return out
    ext=state["ext"]
    stream=torch.cuda.current_stream(x.device).cuda_stream
    if tactic[0]=="v":
        ext.gemv(x.data_ptr(),state[layer].data_ptr(),out.data_ptr(),n,k,tactic[1],tactic[2],router,stream)
    else:
        bn,bk,split=tactic[1:]
        p=torch.empty((max(1,split),m,n),device=x.device,dtype=torch.float32) if split>0 else out
        w=state["hi"] if router else state[layer]
        lo=state["lo"] if router else w
        ext.gemm(x.data_ptr(),w.data_ptr(),lo.data_ptr(),p.data_ptr(),out.data_ptr(),m,n,k,x.stride(0),bn,bk,split,router,stream)
    return out


def prepare(weights: Weights):
    import statistics,json
    from torch.utils.cpp_extension import load_inline
    state={name:w.contiguous() for name,w in weights.items()}
    state["ext"]=load_inline(name="muse_cuda_v20",cpp_sources=_CPP,cuda_sources=_CUDA,
        functions=None,with_cuda=True,no_implicit_headers=True,extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3","--expt-relaxed-constexpr"],verbose=False)
    state["transposed"]={name:w.T.contiguous() for name,w in weights.items()}
    state["hi"]=state["router"].bfloat16()
    state["lo"]=(state["router"]-state["hi"].float()).bfloat16()
    state["dg"]=None
    if torch.cuda.get_device_capability()[0]==9:
        try:
            import deep_gemm
            state["dg"]=deep_gemm
            from pathlib import Path
            print("CUTLASS_HEADERS",str(Path(deep_gemm.__file__).parent/'include'),(Path(deep_gemm.__file__).parent/'include/cutlass/cutlass.h').exists(),flush=True)
        except ImportError:
            pass
    flush=torch.empty(64*1024*1024,device=state["router"].device,dtype=torch.uint8)
    state["choices"]={}
    for layer,w in weights.items():
        for m in (1,8,16,32,128,256):
            n,k=w.shape
            if layer=='unpooler' and m<=8:n=1024
            router=layer=='router'
            x=torch.randn((m,k),device=w.device,dtype=torch.bfloat16)
            ref=(x.double() @ w[:n].double().T).to(torch.float32 if router else torch.bfloat16)
            choices=[('torch',False),('torch',True)]
            if m==1:
                choices += [('v',r,t) for r,t in [(1,128),(1,256),(2,128),(4,128),(8,256)]]
            if router or m in (8,16,32):
                choices += [('g',bn,bk,sp) for bn,bk in [(16,128),(32,128),(64,128),(32,64)] for sp in (4,8,16,32)]
                if torch.cuda.get_device_capability()[0]==9:
                    choices += [('g',bn,128,-sp) for bn in (16,32,64) for sp in (2,4,8)]
            if not router and m>=16 and state['dg'] is not None:
                choices += [('deep',sms) for sms in (128,132)]
            results=[]
            for tactic in choices:
                out=_run(x,layer,n,state,tactic)
                err=((out.float()-ref.float()).norm()/ref.float().norm()).item()
                if err>(1e-4 if router else .01):
                    print('BAD',layer,m,tactic,err,flush=True)
                    continue
                graph=torch.cuda.CUDAGraph()
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                with torch.cuda.graph(graph):
                    flush.zero_(); start.record()
                    out=_run(x,layer,n,state,tactic)
                    end.record()
                times=[]
                for _ in range(7):
                    graph.replay();end.synchronize();times.append(start.elapsed_time(end)*1000)
                results.append((statistics.median(times),tactic))
            results.sort()
            state['choices'][(layer,m,n,k)]=results[0][1]
            print('CUDA_TUNE',json.dumps({'layer':layer,'m':m,'best':results[:6],'torch':[r for r in results if r[1][0]=='torch']}),flush=True)
    return state


def kernel_fn(data: Inputs,state):
    x=data['x'];m,k=x.shape
    layer,n=data['config']['layer'],data['config']['n']
    router=layer=='router'
    if m==0 or n==0:return torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if x.stride(1)!=1 or x.data_ptr()%16 or x.stride(0)*2%16 or k%8:
        return torch.mm(x.float() if router else x,state[layer][:n].T)
    choice=state['choices'].get((layer,m,n,k))
    if choice is None:
        choice=('v',1,128) if m==1 else ('g',32,128,16) if router else ('torch',False)
    return _run(x,layer,n,state,choice)
