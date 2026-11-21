// Proof-of-concept: training loop C++/CUDA murni utk SATU TRANSFORMER BLOCK (blok VIR).
// Pre-norm: RMSNorm -> causal self-attention (QKV, NoPE) -> RMSNorm -> SwiGLU.
// Forward + backward manual, SGD. Bandingkan vs PyTorch transformer block.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} } while(0)

// ---- row-major gemm helpers (verified) ----
static inline void gemmNN(cublasHandle_t h, const float* A, const float* B, float* C, int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&b,C,N); }        // C=A@B
static inline void gemmTN(cublasHandle_t h, const float* A, const float* B, float* C, int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,A,K,&b,C,N); }        // C=A@B^T (B [N,K])
static inline void gemmNT(cublasHandle_t h, const float* A, const float* B, float* C, int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,A,M,&b,C,N); }        // C=A^T@B (A [K,M])
static inline void bgemmNN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,sB,A,K,sA,&b,C,N,sC,bt); }
static inline void bgemmTN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,sB,A,K,sA,&b,C,N,sC,bt); }
static inline void bgemmNT(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,sB,A,M,sA,&b,C,N,sC,bt); }

// ---- RMSNorm (one block per row, D threads) ----
__global__ void rmsnorm_fwd(const float* x,const float* g,float* y,float* rinv,int N,int D){
    int row=blockIdx.x; if(row>=N)return; int t=threadIdx.x;
    const float* xr=x+row*D; float* yr=y+row*D; __shared__ float sd[64];
    float xi=xr[t]; sd[t]=xi*xi; __syncthreads();
    for(int s=32;s>0;s>>=1){ if(t<s) sd[t]+=sd[t+s]; __syncthreads(); }
    float r=rsqrtf(sd[0]/D+1e-6f); rinv[row]=r; yr[t]=xi*r*g[t];
}
__global__ void rmsnorm_bwd(const float* x,const float* g,const float* dy,const float* rinv,float* dx,float* dg,int N,int D){
    int row=blockIdx.x; if(row>=N)return; int t=threadIdx.x;
    const float* xr=x+row*D; const float* dyr=dy+row*D; float* dxr=dx+row*D;
    float xi=xr[t],gi=g[t],dyi=dyr[t],r=rinv[row]; __shared__ float sd[64];
    sd[t]=gi*xi*dyi; __syncthreads();
    for(int s=32;s>0;s>>=1){ if(t<s) sd[t]+=sd[t+s]; __syncthreads(); }
    float dot=sd[0];
    dxr[t]=gi*r*dyi - xi*r*r*r*dot/D;
    atomicAdd(&dg[t], dyi*xi*r);
}
// ---- causal softmax (one block per (bh,i), T threads) ----
__global__ void softmax_causal(const float* s,float* p,int rows,int T){
    int row=blockIdx.x; if(row>=rows)return; int t=threadIdx.x; int i=row%T,bh=row/T;
    const float* sr=s+bh*T*T+i*T; float* pr=p+bh*T*T+i*T; __shared__ float sd[32];
    float m=-INFINITY; for(int j=t;j<=i;j+=T) m=fmaxf(m,sr[j]);
    sd[t]=m; __syncthreads();
    for(int st=16;st>0;st>>=1){ if(t<st) sd[t]=fmaxf(sd[t],sd[t+st]); __syncthreads(); } m=sd[0];
    float e=(t<=i)?__expf(sr[t]-m):0.0f; sd[t]=e; __syncthreads();
    for(int st=16;st>0;st>>=1){ if(t<st) sd[t]+=sd[t+st]; __syncthreads(); } float sum=sd[0];
    if(t<=i) pr[t]=e/sum; else pr[t]=0.0f;
}
__global__ void softmax_bwd(const float* p,const float* dp,float* ds,int rows,int T){
    int row=blockIdx.x; if(row>=rows)return; int t=threadIdx.x; int i=row%T,bh=row/T;
    const float* pr=p+bh*T*T+i*T; const float* dpr=dp+bh*T*T+i*T; float* dsr=ds+bh*T*T+i*T; __shared__ float sd[32];
    float v=(t<=i)?pr[t]*dpr[t]:0.0f; sd[t]=v; __syncthreads();
    for(int st=16;st>0;st>>=1){ if(t<st) sd[t]+=sd[t+st]; __syncthreads(); } float dot=sd[0];
    if(t<=i) dsr[t]=pr[t]*(dpr[t]-dot); else dsr[t]=0.0f;
}
// ---- permutes ----
__global__ void qkv_split(const float* qkv,float* Q,float* K,float* V,int N,int T,int HH,int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int D=HH*hd; if(i>=N*D)return;
    int row=i/D,d=i%D; int b=row/T,t=row%T; int h=d/hd,e=d%hd; int dst=((b*HH+h)*T+t)*hd+e;
    Q[dst]=qkv[i]; K[dst]=qkv[(long)N*D+i]; V[dst]=qkv[(long)2*N*D+i];
}
__global__ void attn_merge(const float* attn,float* out,int N,int T,int HH,int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int D=HH*hd; if(i>=N*D)return;
    int row=i/D,d=i%D; int b=row/T,t=row%T; int h=d/hd,e=d%hd;
    out[i]=attn[((b*HH+h)*T+t)*hd+e];
}
__global__ void attn_scatter(const float* in,float* out,int N,int T,int HH,int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int D=HH*hd; if(i>=N*D)return;
    int row=i/D,d=i%D; int b=row/T,t=row%T; int h=d/hd,e=d%hd;
    out[((b*HH+h)*T+t)*hd+e]=in[i];
}
__global__ void dqkv_merge(const float* dQ,const float* dK,const float* dV,float* dQKV,int N,int T,int HH,int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int D=HH*hd; if(i>=N*D)return;
    int row=i/D,d=i%D; int b=row/T,t=row%T; int h=d/hd,e=d%hd; int src=((b*HH+h)*T+t)*hd+e;
    dQKV[i]=dQ[src]; dQKV[(long)N*D+i]=dK[src]; dQKV[(long)2*N*D+i]=dV[src];
}
// ---- elementwise ----
__global__ void add_(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void silu_mul(const float* g,const float* u,float* h,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float x=g[i]; h[i]=(x/(1.0f+__expf(-x)))*u[i]; }
__global__ void swiglu_bwd(const float* g,const float* u,const float* dh,float* dg,float* du,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float x=g[i],uu=u[i],d=dh[i];
    float sig=1.0f/(1.0f+__expf(-x)); float silu=x*sig; float ds=sig+x*sig*(1.0f-sig);
    dg[i]=d*uu*ds; du[i]=d*silu; }
__global__ void zero_(float* a,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=0.0f; }
__global__ void sgd(float* w,const float* gw,float lr,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) w[i]-=lr*gw[i]; }
__global__ void mse_grad(const float* y,const float* t,float* dy,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dy[i]=(y[i]-t[i])/(float)n; }

int main(){
    const int B=64,T=32,D=64,HH=4,hd=D/HH,HI=256,N=B*T,BH=B*HH;
    const int STEPS=500,WARMUP=20;
    const long sQ=(long)T*hd, sS=(long)T*T;
    cublasHandle_t h; CK(cublasCreate(&h)); cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST);
    const float alpha=1.0f,beta=0.0f,lr=1e-3f,scale=1.0f/sqrtf((float)hd);
    srand(0);
    auto rnd=[&](int n,float s){ std::vector<float> v(n); for(auto&x:v)x=((rand()%1000)/1000.0f-0.5f)*s; return v; };
    std::vector<float> hWqkv(D*3*D),hWo(D*D),hg1(D,1.0f),hWg(D*HI),hWu(D*HI),hWd(HI*D),hg2(D,1.0f);
    { auto v=rnd(D*3*D,0.1f); std::copy(v.begin(),v.end(),hWqkv.begin()); }
    { auto v=rnd(D*D,0.1f); std::copy(v.begin(),v.end(),hWo.begin()); }
    { auto v=rnd(D*HI,0.1f); std::copy(v.begin(),v.end(),hWg.begin()); }
    { auto v=rnd(D*HI,0.1f); std::copy(v.begin(),v.end(),hWu.begin()); }
    { auto v=rnd(HI*D,0.1f); std::copy(v.begin(),v.end(),hWd.begin()); }
    float *Wqkv,*dWqkv,*Wo,*dWo,*g1,*dg1,*Wg,*dWg,*Wu,*dWu,*Wd,*dWd,*g2,*dg2;
    #define ALLOC1(name,sz) CHECK(cudaMalloc(&name,sizeof(float)*(sz)))
    #define ALLOC2(a,b,sz) CHECK(cudaMalloc(&a,sizeof(float)*(sz))); CHECK(cudaMalloc(&b,sizeof(float)*(sz)))
    ALLOC2(Wqkv,dWqkv,D*3*D); ALLOC2(Wo,dWo,D*D); ALLOC2(g1,dg1,D);
    ALLOC2(Wg,dWg,D*HI); ALLOC2(Wu,dWu,D*HI); ALLOC2(Wd,dWd,HI*D); ALLOC2(g2,dg2,D);
    CHECK(cudaMemcpy(Wqkv,hWqkv.data(),sizeof(float)*D*3*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Wo,hWo.data(),sizeof(float)*D*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(g1,hg1.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Wg,hWg.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Wu,hWu.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Wd,hWd.data(),sizeof(float)*HI*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(g2,hg2.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
    float *Xin,*Xa,*Xo,*Xn1,*Ri1,*Xn2,*Ri2,*QKV,*Q,*K,*V,*SC,*PR,*AT,*ATN,*ATO,*GT,*UP,*HA,*MO;
    float *dXo,*dXa,*dXn1,*dXn2,*dTmp,*dQKV,*dQ,*dK,*dV,*dSC,*dPR,*dAT,*dATN,*dGT,*dUP,*dHH,*Tg;
    ALLOC2(Xin,Xa,N*D); ALLOC1(Xo,N*D); ALLOC2(Xn1,Ri1,N*D); ALLOC2(Xn2,Ri2,N*D);
    ALLOC1(QKV,N*3*D); ALLOC1(Q,BH*T*hd); ALLOC1(K,BH*T*hd); ALLOC1(V,BH*T*hd);
    ALLOC1(SC,BH*T*T); ALLOC1(PR,BH*T*T); ALLOC1(AT,BH*T*hd); ALLOC1(ATN,N*D); ALLOC1(ATO,N*D);
    ALLOC1(GT,N*HI); ALLOC1(UP,N*HI); ALLOC1(HA,N*HI); ALLOC1(MO,N*D);
    ALLOC1(dXo,N*D); ALLOC1(dXa,N*D); ALLOC1(dXn1,N*D); ALLOC1(dXn2,N*D); ALLOC1(dTmp,N*D);
    ALLOC1(dQKV,N*3*D); ALLOC1(dQ,BH*T*hd); ALLOC1(dK,BH*T*hd); ALLOC1(dV,BH*T*hd);
    ALLOC1(dSC,BH*T*T); ALLOC1(dPR,BH*T*T); ALLOC1(dAT,BH*T*hd); ALLOC1(dATN,N*D);
    ALLOC1(dGT,N*HI); ALLOC1(dUP,N*HI); ALLOC1(dHH,N*HI); ALLOC1(Tg,N*D);
    #undef ALLOC1
    #undef ALLOC2
    { auto v=rnd(N*D,1.0f); CHECK(cudaMemcpy(Xin,v.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice)); }
    { auto v=rnd(N*D,1.0f); CHECK(cudaMemcpy(Tg,v.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice)); }
    cudaStream_t s; CHECK(cudaStreamCreate(&s)); cublasSetStream(h,s);
    int th=256; int bHD=(N*D+th-1)/th, bHI=(N*HI+th-1)/th, bHN=(N+th-1)/th;
    auto step = [&](){
        zero_<<<64,64,0,s>>>(dg1,D); zero_<<<64,64,0,s>>>(dg2,D);
        // ===== FORWARD =====
        rmsnorm_fwd<<<N,64,0,s>>>(Xin,g1,Xn1,Ri1,N,D);
        gemmNN(h,Xn1,Wqkv,QKV,N,3*D,D,alpha,beta);
        qkv_split<<<bHD,th,0,s>>>(QKV,Q,K,V,N,T,HH,hd);
        bgemmTN(h,Q,K,SC,T,T,hd,sQ,sQ,sS,BH,scale,beta);
        softmax_causal<<<BH*T,32,0,s>>>(SC,PR,BH,T);
        bgemmNN(h,PR,V,AT,T,hd,T,sS,sQ,sQ,BH,alpha,beta);
        attn_merge<<<bHD,th,0,s>>>(AT,ATN,N,T,HH,hd);
        gemmNN(h,ATN,Wo,ATO,N,D,D,alpha,beta);
        add_<<<bHD,th,0,s>>>(Xin,ATO,N*D);                 // Xa = Xin + attnout
        rmsnorm_fwd<<<N,64,0,s>>>(Xa,g2,Xn2,Ri2,N,D);
        gemmNN(h,Xn2,Wg,GT,N,HI,D,alpha,beta);
        gemmNN(h,Xn2,Wu,UP,N,HI,D,alpha,beta);
        silu_mul<<<bHI,th,0,s>>>(GT,UP,HA,N*HI);
        gemmNN(h,HA,Wd,MO,N,D,HI,alpha,beta);
        add_<<<bHD,th,0,s>>>(Xa,MO,N*D);                   // Xo = Xa + mlpout
        // ===== LOSS GRAD =====
        mse_grad<<<bHD,th,0,s>>>(Xo,Tg,dXo,N*D);
        // ===== BACKWARD MLP =====
        gemmNT(h,HA,dXo,dWd,HI,D,N,alpha,beta);             // dWd = hh^T @ dXo
        gemmTN(h,dXo,Wd,dHH,N,HI,HI,alpha,beta);            // dhh = dXo @ Wd^T
        swiglu_bwd<<<bHI,th,0,s>>>(GT,UP,dHH,dGT,dUP,N*HI);
        gemmNT(h,Xn2,dGT,dWg,D,HI,N,alpha,beta);            // dWg = Xn2^T @ dGT
        gemmNT(h,Xn2,dUP,dWu,D,HI,N,alpha,beta);            // dWu = Xn2^T @ dUP
        gemmTN(h,dGT,Wg,dXn2,N,D,HI,alpha,beta);            // dXn2 = dGT @ Wg^T
        gemmTN(h,dUP,Wu,dXn2,N,D,HI,alpha,1.0f);            //      += dUP @ Wu^T
        rmsnorm_bwd<<<N,64,0,s>>>(Xa,g2,dXn2,Ri2,dTmp,dg2,N,D); // dXa_mlp = rmsnorm_bwd(Xa,...)
        // ===== dXa = dXo + dXa_mlp =====
        CHECK(cudaMemcpyAsync(dXa,dXo,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,s));
        add_<<<bHD,th,0,s>>>(dXa,dTmp,N*D);
        // ===== BACKWARD ATTENTION (dATO = dXa) =====
        gemmNT(h,ATN,dXa,dWo,D,D,N,alpha,beta);             // dWo = ATN^T @ dXa
        gemmTN(h,dXa,Wo,dATN,N,D,D,alpha,beta);             // dATN = dXa @ Wo^T
        attn_scatter<<<bHD,th,0,s>>>(dATN,dAT,N,T,HH,hd);   // dAT head-major
        bgemmNT(h,PR,dAT,dV,T,hd,T,sS,sQ,sQ,BH,alpha,beta); // dV = PR^T @ dAT
        bgemmTN(h,dAT,V,dPR,T,T,hd,sQ,sQ,sS,BH,alpha,beta); // dPR = dAT @ V^T
        softmax_bwd<<<BH*T,32,0,s>>>(PR,dPR,dSC,BH,T);
        bgemmNN(h,dSC,K,dQ,T,hd,T,sS,sQ,sQ,BH,scale,beta);  // dQ = dSC @ K * scale
        bgemmNT(h,dSC,Q,dK,T,hd,T,sS,sQ,sQ,BH,scale,beta);  // dK = dSC^T @ Q * scale
        dqkv_merge<<<bHD,th,0,s>>>(dQ,dK,dV,dQKV,N,T,HH,hd);
        gemmNT(h,Xn1,dQKV,dWqkv,D,3*D,N,alpha,beta);        // dWqkv = Xn1^T @ dQKV
        gemmTN(h,dQKV,Wqkv,dXn1,N,D,3*D,alpha,beta);        // dXn1 = dQKV @ Wqkv^T
        rmsnorm_bwd<<<N,64,0,s>>>(Xin,g1,dXn1,Ri1,dTmp,dg1,N,D); // dg1 (dXin tak dipakai)
        // ===== SGD =====
        sgd<<<bHD,th,0,s>>>(Wqkv,dWqkv,lr,D*3*D);
        sgd<<<bHD,th,0,s>>>(Wo,dWo,lr,D*D);
        sgd<<<bHI,th,0,s>>>(Wg,dWg,lr,D*HI);
        sgd<<<bHI,th,0,s>>>(Wu,dWu,lr,D*HI);
        sgd<<<bHI,th,0,s>>>(Wd,dWd,lr,HI*D);
        sgd<<<64,64,0,s>>>(g1,dg1,lr,D);
        sgd<<<64,64,0,s>>>(g2,dg2,lr,D);
    };
    for(int i=0;i<WARMUP;i++){ step(); } CHECK(cudaStreamSynchronize(s));
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<STEPS;i++){ step(); }
    CHECK(cudaStreamSynchronize(s));
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double tok=(double)STEPS*N;
    // sanity: finite check
    float sX[4],sW[4],sG[4];
    CHECK(cudaMemcpy(sX,Xa,sizeof(float)*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sW,Wqkv,sizeof(float)*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sG,GT,sizeof(float)*4,cudaMemcpyDeviceToHost));
    int finite=1; for(int i=0;i<4;i++){ if(!std::isfinite(sX[i])||!std::isfinite(sW[i])||!std::isfinite(sG[i])) finite=0; }
    // MSE akhir (verifikasi backward: loss harus finite & kecil)
    std::vector<float> hXo(N*D), hTg(N*D);
    CHECK(cudaMemcpy(hXo.data(), Xo, sizeof(float)*N*D, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hTg.data(), Tg, sizeof(float)*N*D, cudaMemcpyDeviceToHost));
    double mse=0.0; for(int i=0;i<N*D;i++){ double d=hXo[i]-hTg[i]; mse+=d*d; } mse/=(double)(N*D);
    printf("TB: %d steps, %.2f ms, %.1f us/step, %.1f tok/s | finite=%d mse=%.4f Xa0=%.4f Wqkv0=%.4f\n",
           STEPS,ms,ms*1000.0/STEPS,tok/(ms/1000.0),finite,mse,sX[0],sW[0]);
    return 0;
}