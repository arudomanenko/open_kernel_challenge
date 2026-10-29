# okc: problem=muse_bf16_proj
"""CUDA H200 projections: GEMV, pipelined WGMMA, and CUDA library GEMM.

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



_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
void gemv(uint64_t,uint64_t,uint64_t,int,int,int,int,bool,uint64_t);
void simt(uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t);
void combine(uint64_t,uint64_t,int,int,bool,uint64_t);
void reverse_pipe(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,bool,uint64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("gemv",&gemv);m.def("simt",&simt);m.def("combine",&combine);m.def("reverse_pipe",&reverse_pipe);
}
"""

_CUDA = r"""
// CUDA implementation of BF16 projections and the FP32 router.
// WGMMA layout and register mappings follow NVIDIA's public PTX ISA and
// CUTLASS SM90 descriptor/accumulator layouts. Streaming GEMV is adapted
// from SGLang a55ab7d5 (Apache-2.0); see reports/research.md for sources.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdexcept>
using bf=__nv_bfloat16;
__device__ __forceinline__ float warp_sum(float v) {
  for(int s=16;s;s/=2) v += __shfl_down_sync(0xffffffff,v,s);
  return v;
}
__device__ __forceinline__ void cp16(void* dst,const void* src,bool valid) {
#if __CUDA_ARCH__ >= 800
  unsigned addr=__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(addr),"l"(src),"r"(valid?16:0));
#endif
}
__device__ __forceinline__ void cpwait() {
  asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;":::"memory");
  __syncthreads();
}

template<bool FP32>
__global__ void finish(const float* p,void* y,int size,int split) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<size) {
    float v=0;
    for(int s=0;s<split;s++) v+=p[s*size+i];
    if constexpr(FP32) ((float*)y)[i]=v;
    else ((bf*)y)[i]=__float2bfloat16_rn(v);
  }
}
// Adapted from SGLang a55ab7d5 Hopper GEMV: shared activations and
// streaming weight loads. The original project uses the Apache-2.0 license.
template<int K,int ROWS,int UNROLL,int NW>
__global__ void stream_gemv(const bf* __restrict__ x,const bf* __restrict__ w,bf* __restrict__ y,int N) {
  __shared__ __align__(32) bf sx[K];
  int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  for(int i=tid*8;i<K;i+=NW*32*8) *(uint4*)(sx+i)=*(const uint4*)(x+i);
  __syncthreads();
  int row=(blockIdx.x*NW+warp)*ROWS;
  float acc[ROWS]={};
  for(int kk=lane*8*UNROLL;kk<K;kk+=32*8*UNROLL) {
    uint4 xv[UNROLL];
    #pragma unroll
    for(int u=0;u<UNROLL;u++) xv[u]=*(const uint4*)(sx+kk+u*8);
    #pragma unroll
    for(int r=0;r<ROWS;r++) {
      if(row+r<N) {
        uint4 wv[UNROLL];
        #pragma unroll
        for(int u=0;u<UNROLL;u++) wv[u]=__ldcs((const uint4*)(w+(row+r)*K+kk+u*8));
        #pragma unroll
        for(int u=0;u<UNROLL;u++) {
          const __nv_bfloat162* a=(const __nv_bfloat162*)&xv[u];
          const __nv_bfloat162* b=(const __nv_bfloat162*)&wv[u];
          float dot=0;
          #pragma unroll
          for(int j=0;j<4;j++) {
            float2 av=__bfloat1622float2(a[j]),bv=__bfloat1622float2(b[j]);
            dot=fmaf(av.x,bv.x,dot);dot=fmaf(av.y,bv.y,dot);
          }
          acc[r]+=dot;
        }
      }
    }
  }
  #pragma unroll
  for(int r=0;r<ROWS;r++) {
    float v=warp_sum(acc[r]);
    if(lane==0 && row+r<N) y[row+r]=__float2bfloat16_rn(v);
  }
}
template<int K,int T,bool FP32>
__global__ void prefetch_gemv(const bf* __restrict__ x,const void* __restrict__ w,void* __restrict__ y,int N) {
 int row=blockIdx.x,tid=threadIdx.x;
 constexpr int V=(K+T*8-1)/(T*8);
 uint4 xs[V],ws[V];float4 wf[2*V];
 #pragma unroll
 for(int v=0;v<V;v++) {
   int k=tid*8+v*T*8;
   xs[v]=k<K?__ldg((const uint4*)(x+k)):uint4{};
   if constexpr(FP32) {
     wf[2*v]=k<K?__ldg((const float4*)((const float*)w+row*K+k)):float4{};
     wf[2*v+1]=k<K?__ldg((const float4*)((const float*)w+row*K+k+4)):float4{};
   } else ws[v]=k<K?__ldg((const uint4*)((const bf*)w+row*K+k)):uint4{};
 }
 float acc[8]={};
 #pragma unroll
 for(int v=0;v<V;v++) {
   bf* a=(bf*)&xs[v];
   if constexpr(FP32) {
     float* b=(float*)&wf[2*v];
     #pragma unroll
     for(int j=0;j<8;j++) acc[j]=fmaf(__bfloat162float(a[j]),b[j],acc[j]);
   } else {
     bf* b=(bf*)&ws[v];
     #pragma unroll
     for(int j=0;j<8;j++) acc[j]=fmaf(__bfloat162float(a[j]),__bfloat162float(b[j]),acc[j]);
   }
 }
 float val=(acc[0]+acc[1])+(acc[2]+acc[3])+(acc[4]+acc[5])+(acc[6]+acc[7]);
 val=warp_sum(val);
 __shared__ float sm[T/32];
 if(tid%32==0)sm[tid/32]=val;
 __syncthreads();
 if(tid==0) {
   float v=0;
   #pragma unroll
   for(int j=0;j<T/32;j++)v+=sm[j];
   if constexpr(FP32)((float*)y)[row]=v;
   else ((bf*)y)[row]=__float2bfloat16_rn(v);
 }
}

void gemv(uint64_t x,uint64_t w,uint64_t y,int n,int k,int kind,int threads,bool fp,uint64_t stream) {
 cudaStream_t s=(cudaStream_t)stream;
 if(kind==-1) {
   if(k==8192 && threads==512) {
     if(fp) prefetch_gemv<8192,512,true><<<n,512,0,s>>>((bf*)x,(void*)w,(void*)y,n);
     else prefetch_gemv<8192,512,false><<<n,512,0,s>>>((bf*)x,(void*)w,(void*)y,n);
     return;
   }
   if(k==4096 && threads==256 && !fp) {
     prefetch_gemv<4096,256,false><<<n,256,0,s>>>((bf*)x,(void*)w,(void*)y,n);return;
   }
 }
 if(kind==112 && k==1024 && threads==128 && !fp) {
   stream_gemv<1024,1,2,4><<<(n+3)/4,128,0,s>>>((bf*)x,(bf*)w,(bf*)y,n);return;
 }
 if(kind==114 && k==8192 && threads==256 && !fp) {
   stream_gemv<8192,1,4,8><<<(n+7)/8,256,0,s>>>((bf*)x,(bf*)w,(bf*)y,n);return;
 }
 throw std::runtime_error("unsupported GEMV configuration");
}
template<int MT,int T>
__global__ void router_simt(const bf* __restrict__ x,const float* __restrict__ w,float* __restrict__ y,int M,int N,int K,int SX) {
  int n=blockIdx.x,mo=blockIdx.y*MT,tid=threadIdx.x;
  float acc[MT]={};
  for(int ko=tid*8;ko<K;ko+=T*8) {
    float4 w0=__ldg((const float4*)(w+n*K+ko));
    float4 w1=__ldg((const float4*)(w+n*K+ko+4));
    float wv[8]={w0.x,w0.y,w0.z,w0.w,w1.x,w1.y,w1.z,w1.w};
    #pragma unroll
    for(int m=0;m<MT;m++) {
      if(mo+m<M) {
        uint4 xv=*(const uint4*)(x+(mo+m)*SX+ko);
        bf* xb=(bf*)&xv;
        float dot=0;
        #pragma unroll
        for(int j=0;j<8;j++) dot=fmaf(wv[j],__bfloat162float(xb[j]),dot);
        acc[m]+=dot;
      }
    }
  }
  __shared__ float sm[MT][T/32];
  #pragma unroll
  for(int m=0;m<MT;m++) {
    float v=warp_sum(acc[m]);
    if(tid%32==0) sm[m][tid/32]=v;
  }
  __syncthreads();
  if(tid<MT && mo+tid<M) {
    float v=0;
    #pragma unroll
    for(int j=0;j<T/32;j++) v+=sm[tid][j];
    y[(mo+tid)*N+n]=v;
  }
}

void simt(uint64_t x,uint64_t w,uint64_t y,int m,int n,int k,int sx,uint64_t stream) {
 router_simt<4,256><<<dim3(n,(m+3)/4),256,0,(cudaStream_t)stream>>>((bf*)x,(float*)w,(float*)y,m,n,k,sx);
}
__global__ void combine_kernel(const float* __restrict__ p,float* __restrict__ y,int M,int N,bool interleave) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<M*N) {
    int m=i/N,n=i%N;
    if(interleave) y[i]=p[m*(2*N)+n*2]+p[m*(2*N)+n*2+1];
    else y[i]=p[m*(2*N)+n]+p[m*(2*N)+N+n];
  }
}
void combine(uint64_t p,uint64_t y,int m,int n,bool interleave,uint64_t stream) {
 combine_kernel<<<(m*n+255)/256,256,0,(cudaStream_t)stream>>>((float*)p,(float*)y,m,n,interleave);
}
template<int BN> __device__ __forceinline__ void wg(float*,uint64_t,uint64_t);

template<> __device__ __forceinline__ void wg<32>(float* c,uint64_t a,uint64_t b) {
#if __CUDA_ARCH__ >= 900
 asm volatile("{.reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]):"l"(a),"l"(b));
#endif
}
template<> __device__ __forceinline__ void wg<64>(float* c,uint64_t a,uint64_t b) {
#if __CUDA_ARCH__ >= 900
 asm volatile("{.reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]):"l"(a),"l"(b));
#endif
}

template<> __device__ __forceinline__ void wg<8>(float* c,uint64_t a,uint64_t b) {
#if __CUDA_ARCH__ >= 900
 asm volatile("{.reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"l"(a),"l"(b));
#endif
}
template<> __device__ __forceinline__ void wg<16>(float* c,uint64_t a,uint64_t b) {
#if __CUDA_ARCH__ >= 900
 asm volatile("{.reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;}"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]):"l"(a),"l"(b));
#endif
}

__device__ __forceinline__ uint64_t gmma_desc_sw(const bf* p,int BK) {
 uint64_t addr=__cvta_generic_to_shared(p);
 return (addr>>4)|(uint64_t(1)<<16)|((uint64_t(8*BK*2)>>4)<<32)|(uint64_t(1)<<62);
}
template<int TN,int BK,bool ROUTER>
__global__ void reverse_pipe_gemm(const bf* __restrict__ x,const bf* __restrict__ w,const bf* __restrict__ lo,
 float* __restrict__ p,void* __restrict__ y,int M,int N,int K,int SX,int split) {
#if __CUDA_ARCH__ >= 900
 extern __shared__ __align__(1024) unsigned char storage[];
 bf* a=(bf*)storage;bf* b=a+2*64*BK;bf* al=b+2*TN*BK;
 int tid=threadIdx.x,mo=blockIdx.x*TN,no=blockIdx.y*64,si=blockIdx.z;
 float acc[TN/2]={},low[TN/2]={};
 auto load_stage=[&](int ko,int stage) {
   for(int i=tid;i<64*BK/8;i+=128) {
     int r=i/(BK/8),c=(i%(BK/8))*8;
     int off=(r/8)*(8*BK)+(c/64)*512+(r%8)*64+(c%64);
     off ^= (off>>3)&56;
     cp16(a+stage*64*BK+off,w+(no+r)*K+ko+c,no+r<N && ko+c<K);
     if constexpr(ROUTER)cp16(al+stage*64*BK+off,lo+(no+r)*K+ko+c,no+r<N && ko+c<K);
   }
   for(int i=tid;i<TN*BK/8;i+=128) {
     int r=i/(BK/8),c=(i%(BK/8))*8;
     int off=(r/8)*(8*BK)+(c/64)*512+(r%8)*64+(c%64);
     off ^= (off>>3)&56;
     cp16(b+stage*TN*BK+off,x+(mo+r)*SX+ko+c,mo+r<M && ko+c<K);
   }
 };
 load_stage(si*BK,0);cpwait();
 int stage=0;
 for(int ko=si*BK;ko<K;ko+=split*BK) {
   int next=ko+split*BK;
   if(next<K) {load_stage(next,stage^1);asm volatile("cp.async.commit_group;":::"memory");}
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   asm volatile("wgmma.fence.sync.aligned;":::"memory");
   #pragma unroll
   for(int kk=0;kk<BK;kk+=16) {
     uint64_t ad=gmma_desc_sw(a+stage*64*BK+(kk/64)*512+kk%64,BK),bd=gmma_desc_sw(b+stage*TN*BK+(kk/64)*512+kk%64,BK);
     wg<TN>(acc,ad,bd);
     if constexpr(ROUTER)wg<TN>(low,gmma_desc_sw(al+stage*64*BK+(kk/64)*512+kk%64,BK),bd);
   }
   asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;":::"memory");
   __syncthreads();
   if(next<K) {asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();}
   stage^=1;
 }
 int lane=tid%32,warp=tid/32;
 #pragma unroll
 for(int i=0;i<TN/2;i++) {
   int r=warp*16+lane/4+(i%4>=2?8:0),c=(lane%4)*2+i%2+(i/4)*8;
   if(no+r<N && mo+c<M) {
     float v=acc[i]+(ROUTER?low[i]:0.f);
     int idx=(mo+c)*N+no+r;
     if(split>1) p[si*M*N+idx]=v;
     else {
       if constexpr(ROUTER)((float*)y)[idx]=v;
       else ((bf*)y)[idx]=__float2bfloat16_rn(v);
     }
   }
 }
#endif
}
void reverse_pipe(uint64_t x,uint64_t w,uint64_t lo,uint64_t p,uint64_t y,int m,int n,int k,int sx,int tn,int bk,int split,bool router,uint64_t stream) {
 cudaStream_t s=(cudaStream_t)stream;
 dim3 grid((m+tn-1)/tn,(n+63)/64,split);
#define PBF(N,K) if(!router && tn==N && bk==K) {constexpr int bytes=2*(64+N)*K*2;cudaFuncSetAttribute(reverse_pipe_gemm<N,K,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);reverse_pipe_gemm<N,K,false><<<grid,128,bytes,s>>>((bf*)x,(bf*)w,(bf*)lo,(float*)p,(void*)y,m,n,k,sx,split);}
#define PRT(N,K) if(router && tn==N && bk==K) {constexpr int bytes=2*(128+N)*K*2;cudaFuncSetAttribute(reverse_pipe_gemm<N,K,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);reverse_pipe_gemm<N,K,true><<<grid,128,bytes,s>>>((bf*)x,(bf*)w,(bf*)lo,(float*)p,(void*)y,m,n,k,sx,split);}
 PBF(8,128) else PBF(8,256) else PBF(16,128) else PRT(16,256) else PRT(32,256) else PRT(64,128) else throw std::runtime_error("unsupported pipelined GEMM configuration");
#undef PBF
#undef PRT
 if(split>1) {
   if(router)finish<true><<<(m*n+255)/256,256,0,s>>>((float*)p,(void*)y,m*n,split);
   else finish<false><<<(m*n+255)/256,256,0,s>>>((float*)p,(void*)y,m*n,split);
 }
}
"""

# Frozen from paired cold-cache H200 CUDA-graph experiments.
_TACTICS = {('qkv_gate', 1, 2304, 8192): ('v', -1, 512), ('qkv_gate', 8, 2304, 8192): ('rev', 16, 128, 16), ('qkv_gate', 16, 2304, 8192): ('rev', 16, 128, 16), ('qkv_gate', 32, 2304, 8192): ('torch', True), ('qkv_gate', 128, 2304, 8192): ('deep', 128), ('qkv_gate', 256, 2304, 8192): ('deep', 128), ('o', 1, 8192, 1024): ('v', 112, 128), ('o', 8, 8192, 1024): ('torch', True), ('o', 16, 8192, 1024): ('torch', True), ('o', 32, 8192, 1024): ('torch', True), ('o', 128, 8192, 1024): ('torch', True), ('o', 256, 8192, 1024): ('torch', True), ('pooler', 1, 4096, 8192): ('v', 114, 256), ('pooler', 8, 4096, 8192): ('rev', 8, 128, 8), ('pooler', 16, 4096, 8192): ('deep', 132), ('pooler', 32, 4096, 8192): ('deep', 128), ('pooler', 128, 4096, 8192): ('deep', 128), ('pooler', 256, 4096, 8192): ('torch', True), ('unpooler', 1, 1024, 4096): ('v', -1, 256), ('unpooler', 8, 1024, 4096): ('rev', 8, 256, 16), ('unpooler', 16, 8192, 4096): ('torch', True), ('unpooler', 32, 8192, 4096): ('torch', True), ('unpooler', 128, 8192, 4096): ('torch', True), ('unpooler', 256, 8192, 4096): ('torch', False), ('router', 1, 256, 8192): ('v', -1, 512), ('router', 8, 256, 8192): ('simt', 4, 256), ('router', 16, 256, 8192): ('rev', 16, 256, 32), ('router', 32, 256, 8192): ('rev', 32, 256, 32), ('router', 128, 256, 8192): ('rev', 64, 128, 32), ('router', 256, 256, 8192): ('cat', 0, True)}


def _run(x, layer, n, state, tactic):
    m, k = x.shape
    router = layer == "router"
    kind = tactic[0]
    if kind == "torch":
        w = state["transposed"][layer][:, :n] if tactic[1] else state[layer][:n].T
        return torch.mm(x, w)
    if kind == "deep" and state["deep"] is None:
        return torch.mm(x, state[layer][:n].T)
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    if kind == "deep":
        dg = state["deep"]
        with _DEEP_LOCK:
            old_sms, old_det = dg.get_num_sms(), dg.get_deterministic_algorithms()
            try:
                dg.set_num_sms(min(tactic[1], state["sms"]))
                dg.use_deterministic_algorithms(True)
                dg.bf16_gemm_nt(x, state[layer][:n], out)
            finally:
                dg.use_deterministic_algorithms(old_det)
                dg.set_num_sms(old_sms)
        return out
    stream = torch.cuda.current_stream(x.device).cuda_stream
    ext = state["ext"]
    if kind == "v":
        ext.gemv(x.data_ptr(), state[layer].data_ptr(), out.data_ptr(), n, k,
                 tactic[1], tactic[2], router, stream)
    elif kind == "simt":
        ext.simt(x.data_ptr(), state[layer].data_ptr(), out.data_ptr(),
                 m, n, k, x.stride(0), stream)
    elif kind == "cat":
        p = torch.mm(x, state["router_cat"], out_dtype=torch.float32)
        ext.combine(p.data_ptr(), out.data_ptr(), m, n, False, stream)
    elif kind == "rev":
        if not state["hopper"]:
            return (torch.mm(x, state[layer][:n].T) if not router else
                    _run(x, layer, n, state, ("cat", 0, True)))
        tn, bk, split = tactic[1:]
        p = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        w = state["hi"] if router else state[layer]
        lo = state["lo"] if router else w
        ext.reverse_pipe(x.data_ptr(), w.data_ptr(), lo.data_ptr(), p.data_ptr(),
                         out.data_ptr(), m, n, k, x.stride(0), tn, bk, split,
                         router, stream)
    else:
        raise ValueError("Unknown tactic")
    return out


def prepare(weights: Weights):
    from torch.utils.cpp_extension import load_inline
    state = {name: w.contiguous() for name, w in weights.items()}
    device = state["router"].device
    state["hopper"] = torch.cuda.get_device_capability(device)[0] == 9
    state["sms"] = torch.cuda.get_device_properties(device).multi_processor_count
    flags = ["-O3", "--expt-relaxed-constexpr"]
    if state["hopper"]:
        flags += ["-gencode=arch=compute_90a,code=sm_90a"]
    state["ext"] = load_inline(name="muse_cuda_final_v32", cpp_sources=_CPP,
        cuda_sources=_CUDA, functions=None, with_cuda=True, no_implicit_headers=True,
        extra_cflags=["-O3"], extra_cuda_cflags=flags, verbose=False)
    packed = {key[0] for key, tactic in _TACTICS.items()
              if tactic[0] == "torch" and tactic[1]}
    state["transposed"] = {name: state[name].T.contiguous() for name in packed}
    state["hi"] = state["router"].bfloat16()
    state["lo"] = (state["router"] - state["hi"].float()).bfloat16()
    state["router_cat"] = torch.cat((state["hi"], state["lo"]), dim=0).T.contiguous()
    state["deep"] = None
    if state["hopper"]:
        try:
            import deep_gemm
            for name in ("get_num_sms", "set_num_sms", "get_deterministic_algorithms",
                         "use_deterministic_algorithms", "bf16_gemm_nt"):
                getattr(deep_gemm, name)
            state["deep"] = deep_gemm
        except (ImportError, AttributeError):
            pass
    # Warm synthetic specializations. Retain no inputs, outputs or captured graphs.
    for (layer, m, n, k), tactic in _TACTICS.items():
        if tactic[0] != "torch":
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            _run(x, layer, n, state, tactic)
    return state


def kernel_fn(data: Inputs, state):
    x = data["x"]
    m, k = x.shape
    layer, n = data["config"]["layer"], data["config"]["n"]
    router = layer == "router"
    if m == 0 or n == 0:
        return torch.empty((m, n), device=x.device,
                           dtype=torch.float32 if router else torch.bfloat16)
    # Vectorized CUDA loads require aligned dense columns. DeepGEMM additionally
    # requires a dense activation matrix. The library handles unusual views.
    aligned = (x.stride(1) == 1 and x.data_ptr() % 16 == 0
               and x.stride(0) * x.element_size() % 16 == 0
               and state[layer].data_ptr() % 16 == 0)
    tactic = _TACTICS.get((layer, m, n, k))
    if not aligned or (tactic is not None and tactic[0] == "deep" and not x.is_contiguous()):
        return torch.mm(x.float() if router else x, state[layer][:n].T)
    if tactic is None:
        if router and n == 256 and k == 8192:
            tactic = ("cat", 0, True)
        else:
            return torch.mm(x, state[layer][:n].T)
    return _run(x, layer, n, state, tactic)
