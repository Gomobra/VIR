// Proof-of-concept: training loop C++/CUDA murni utk 6 TRANSFORMER BLOCKS bertumpuk.
// Struktur VIR: prelude(1) + core×loops + coda(1) = 6 blok (recurrent depth).
// Tiap blok: RMSNorm -> causal attention (QKV, NoPE) -> RMSNorm -> SwiGLU (residual).
// Forward + backward manual, SGD. Bandingkan vs PyTorch 6-block.
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
#define NBLK 6

static inline void gemmNN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&b,C,N); }
static inline void gemmTN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,A,K,&b,C,N); }
static inline void gemmNT(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,float a,float b){
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,A,M,&b,C,N); }
static inline void bgemmNN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,sB,A,K,sA,&b,C,N,sC,bt); }
static inline void bgemmTN(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,sB,A,K,sA,&b,C,N,sC,bt); }
static inline void bgemmNT(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K,long sA,long sB,long sC,int bt,float a,float b){
    cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,sB,A,M,sA,&b,C,N,sC,bt); }

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

// ---- Muon (Newton-Schulz) elementwise kernels ----
__global__ void mom_update(float* buf,const float* g,float mom,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) buf[i]=mom*buf[i]+g[i]; }
__global__ void sum_sq(const float* x,float* out,int n){
    __shared__ float sd[256]; int t=threadIdx.x; float s=0;
    for(int i=t;i<n;i+=blockDim.x) s+=x[i]*x[i];
    sd[t]=s; __syncthreads();
    for(int st=128;st>0;st>>=1){ if(t<st) sd[t]+=sd[t+st]; __syncthreads(); }
    if(t==0) atomicAdd(out,sd[0]); }
__global__ void scale_norm(const float* x,float* y,const float* norm,float eps,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float rinv=rsqrtf(*norm+eps); y[i]=x[i]*rinv; }
__global__ void ns_combine(float* X,const float* AX,const float* A2X,float c0,float c1,float c2,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    X[i]=c0*X[i]+c1*AX[i]+c2*A2X[i]; }
__global__ void w_update(float* W,const float* X,float lr,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) W[i]-=lr*X[i]; }

int main(){
    const int B=64,T=32,D=64,HH=4,hd=D/HH,HI=256,N=B*T,BH=B*HH;
    const int STEPS=500,WARMUP=20;
    const long sQ=(long)T*hd, sS=(long)T*T;
    cublasHandle_t h; CK(cublasCreate(&h)); cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST);
    const float alpha=1.0f,beta=0.0f,lr=1e-4f,scale=1.0f/sqrtf((float)hd);
    srand(0);
    auto rnd=[&](int n,float s){ std::vector<float> v(n); for(auto&x:v)x=((rand()%1000)/1000.0f-0.5f)*s; return v; };
    // weights per block
    float *Wqkv[NBLK],*dWqkv[NBLK],*Wo[NBLK],*dWo[NBLK],*g1[NBLK],*dg1[NBLK];
    float *Wg[NBLK],*dWg[NBLK],*Wu[NBLK],*dWu[NBLK],*Wd[NBLK],*dWd[NBLK],*g2[NBLK],*dg2[NBLK];
    #define ALLOC1(name,sz) CHECK(cudaMalloc(&name,sizeof(float)*(sz)))
    for(int b=0;b<NBLK;b++){
        ALLOC1(Wqkv[b],D*3*D); ALLOC1(dWqkv[b],D*3*D);
        ALLOC1(Wo[b],D*D); ALLOC1(dWo[b],D*D);
        ALLOC1(g1[b],D); ALLOC1(dg1[b],D);
        ALLOC1(Wg[b],D*HI); ALLOC1(dWg[b],D*HI);
        ALLOC1(Wu[b],D*HI); ALLOC1(dWu[b],D*HI);
        ALLOC1(Wd[b],HI*D); ALLOC1(dWd[b],HI*D);
        ALLOC1(g2[b],D); ALLOC1(dg2[b],D);
        std::vector<float> tmp;
        tmp=rnd(D*3*D,0.1f); CHECK(cudaMemcpy(Wqkv[b],tmp.data(),sizeof(float)*D*3*D,cudaMemcpyHostToDevice));
        tmp=rnd(D*D,0.1f); CHECK(cudaMemcpy(Wo[b],tmp.data(),sizeof(float)*D*D,cudaMemcpyHostToDevice));
        tmp.assign(D,1.0f); CHECK(cudaMemcpy(g1[b],tmp.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
        tmp=rnd(D*HI,0.1f); CHECK(cudaMemcpy(Wg[b],tmp.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice));
        tmp=rnd(D*HI,0.1f); CHECK(cudaMemcpy(Wu[b],tmp.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice));
        tmp=rnd(HI*D,0.1f); CHECK(cudaMemcpy(Wd[b],tmp.data(),sizeof(float)*HI*D,cudaMemcpyHostToDevice));
        tmp.assign(D,1.0f); CHECK(cudaMemcpy(g2[b],tmp.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
    }
    // per-block stored activations
    float *XB[NBLK],*XA[NBLK],*Xn1[NBLK],*Ri1[NBLK],*Xn2[NBLK],*Ri2[NBLK];
    float *Q[NBLK],*K[NBLK],*V[NBLK],*PR[NBLK],*ATN[NBLK],*GT[NBLK],*UP[NBLK];
    for(int b=0;b<NBLK;b++){
        ALLOC1(XB[b],N*D); ALLOC1(XA[b],N*D);
        ALLOC1(Xn1[b],N*D); ALLOC1(Ri1[b],N);
        ALLOC1(Xn2[b],N*D); ALLOC1(Ri2[b],N);
        ALLOC1(Q[b],BH*T*hd); ALLOC1(K[b],BH*T*hd); ALLOC1(V[b],BH*T*hd);
        ALLOC1(PR[b],BH*T*T); ALLOC1(ATN[b],N*D);
        ALLOC1(GT[b],N*HI); ALLOC1(UP[b],N*HI);
    }
    // temps (forward)
    float *Xin,*X,*QKV,*SC,*AT,*ATO,*HA,*MO,*Tg;
    ALLOC1(Xin,N*D); ALLOC1(X,N*D); ALLOC1(QKV,N*3*D); ALLOC1(SC,BH*T*T); ALLOC1(AT,BH*T*hd);
    ALLOC1(ATO,N*D); ALLOC1(HA,N*HI); ALLOC1(MO,N*D); ALLOC1(Tg,N*D);
    // temps (backward, reused across blocks)
    float *dX,*dXA,*dTmp,*dXn1,*dXn2,*dQKV,*dQ,*dK,*dV,*dSC,*dPR,*dAT,*dATN,*dGT,*dUP,*dHH;
    ALLOC1(dX,N*D); ALLOC1(dXA,N*D); ALLOC1(dTmp,N*D); ALLOC1(dXn1,N*D); ALLOC1(dXn2,N*D);
    ALLOC1(dQKV,N*3*D); ALLOC1(dQ,BH*T*hd); ALLOC1(dK,BH*T*hd); ALLOC1(dV,BH*T*hd);
    ALLOC1(dSC,BH*T*T); ALLOC1(dPR,BH*T*T); ALLOC1(dAT,BH*T*hd); ALLOC1(dATN,N*D);
    ALLOC1(dGT,N*HI); ALLOC1(dUP,N*HI); ALLOC1(dHH,N*HI);
    // Muon: momentum buffers per 2D weight + shared NS workspace + norm scalar
    float *bufWqkv[NBLK],*bufWo[NBLK],*bufWg[NBLK],*bufWu[NBLK],*bufWd[NBLK];
    for(int b=0;b<NBLK;b++){
        ALLOC1(bufWqkv[b],D*3*D); ALLOC1(bufWo[b],D*D);
        ALLOC1(bufWg[b],D*HI); ALLOC1(bufWu[b],D*HI); ALLOC1(bufWd[b],HI*D);
        CHECK(cudaMemset(bufWqkv[b],0,sizeof(float)*D*3*D)); CHECK(cudaMemset(bufWo[b],0,sizeof(float)*D*D));
        CHECK(cudaMemset(bufWg[b],0,sizeof(float)*D*HI)); CHECK(cudaMemset(bufWu[b],0,sizeof(float)*D*HI));
        CHECK(cudaMemset(bufWd[b],0,sizeof(float)*HI*D));
    }
    float *Xw,*Aw,*AXw,*A2w,*A2Xw,*norm_dev;
    int maxMN = D*HI; if(D*3*D>maxMN) maxMN=D*3*D; if(HI*D>maxMN) maxMN=HI*D;
    int maxMM = HI*HI; if(D*D>maxMM) maxMM=D*D; if(3*D*3*D>maxMM) maxMM=3*D*3*D;
    ALLOC1(Xw,maxMN); ALLOC1(AXw,maxMN); ALLOC1(A2Xw,maxMN);
    ALLOC1(Aw,maxMM); ALLOC1(A2w,maxMM); ALLOC1(norm_dev,1);
    #undef ALLOC1
    { auto v=rnd(N*D,1.0f); CHECK(cudaMemcpy(Xin,v.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice)); }
    { auto v=rnd(N*D,1.0f); CHECK(cudaMemcpy(Tg,v.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice)); }
    cudaStream_t st; CHECK(cudaStreamCreate(&st)); cublasSetStream(h,st);
    int th=256; int bHD=(N*D+th-1)/th, bHI=(N*HI+th-1)/th;
    const float mom=0.95f, nseps=1e-7f; const int NS_STEPS=5;
    // Muon update: W -= lr * NewtonSchulz(momentum_buf + grad)
    auto muon = [&](float* W,const float* G,float* buf,int M,int N){
        int sz=M*N;
        mom_update<<<bHD,th,0,st>>>(buf,G,mom,sz);
        CHECK(cudaMemsetAsync(norm_dev,0,sizeof(float),st));
        sum_sq<<<1,256,0,st>>>(buf,norm_dev,sz);
        scale_norm<<<bHD,th,0,st>>>(buf,Xw,norm_dev,nseps,sz);
        for(int i=0;i<NS_STEPS;i++){
            gemmTN(h,Xw,Xw,Aw,M,M,N,alpha,beta);        // A = X @ X^T
            gemmNN(h,Aw,Xw,AXw,M,N,M,alpha,beta);       // AX = A @ X
            gemmNN(h,Aw,Aw,A2w,M,M,M,alpha,beta);       // A2 = A @ A
            gemmNN(h,A2w,Xw,A2Xw,M,N,M,alpha,beta);     // A2X = A2 @ X
            ns_combine<<<bHD,th,0,st>>>(Xw,AXw,A2Xw,3.4445f,-4.7750f,2.0315f,sz);
        }
        w_update<<<bHD,th,0,st>>>(W,Xw,lr,sz);
    };

    auto step = [&](){
        CHECK(cudaMemcpyAsync(X,Xin,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));  // reset input (fixed)
        for(int b=0;b<NBLK;b++){ zero_<<<64,64,0,st>>>(dg1[b],D); zero_<<<64,64,0,st>>>(dg2[b],D); }
        // ===== FORWARD =====
        for(int b=0;b<NBLK;b++){
            CHECK(cudaMemcpyAsync(XB[b],X,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
            rmsnorm_fwd<<<N,64,0,st>>>(X,g1[b],Xn1[b],Ri1[b],N,D);
            gemmNN(h,Xn1[b],Wqkv[b],QKV,N,3*D,D,alpha,beta);
            qkv_split<<<bHD,th,0,st>>>(QKV,Q[b],K[b],V[b],N,T,HH,hd);
            bgemmTN(h,Q[b],K[b],SC,T,T,hd,sQ,sQ,sS,BH,scale,beta);
            softmax_causal<<<BH*T,32,0,st>>>(SC,PR[b],BH,T);
            bgemmNN(h,PR[b],V[b],AT,T,hd,T,sS,sQ,sQ,BH,alpha,beta);
            attn_merge<<<bHD,th,0,st>>>(AT,ATN[b],N,T,HH,hd);
            gemmNN(h,ATN[b],Wo[b],ATO,N,D,D,alpha,beta);
            add_<<<bHD,th,0,st>>>(X,ATO,N*D);
            CHECK(cudaMemcpyAsync(XA[b],X,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
            rmsnorm_fwd<<<N,64,0,st>>>(X,g2[b],Xn2[b],Ri2[b],N,D);
            gemmNN(h,Xn2[b],Wg[b],GT[b],N,HI,D,alpha,beta);
            gemmNN(h,Xn2[b],Wu[b],UP[b],N,HI,D,alpha,beta);
            silu_mul<<<bHI,th,0,st>>>(GT[b],UP[b],HA,N*HI);
            gemmNN(h,HA,Wd[b],MO,N,D,HI,alpha,beta);
            add_<<<bHD,th,0,st>>>(X,MO,N*D);
        }
        // ===== LOSS GRAD =====
        mse_grad<<<bHD,th,0,st>>>(X,Tg,dX,N*D);
        // ===== BACKWARD =====
        for(int b=NBLK-1;b>=0;b--){
            // MLP backward
            silu_mul<<<bHI,th,0,st>>>(GT[b],UP[b],HA,N*HI);          // recompute hh
            gemmNT(h,HA,dX,dWd[b],HI,D,N,alpha,beta);                 // dWd = HA^T @ dX
            gemmTN(h,dX,Wd[b],dHH,N,HI,D,alpha,beta);               // dhh = dX @ Wd^T
            swiglu_bwd<<<bHI,th,0,st>>>(GT[b],UP[b],dHH,dGT,dUP,N*HI);
            gemmNT(h,Xn2[b],dGT,dWg[b],D,HI,N,alpha,beta);           // dWg
            gemmNT(h,Xn2[b],dUP,dWu[b],D,HI,N,alpha,beta);           // dWu
            gemmTN(h,dGT,Wg[b],dXn2,N,D,HI,alpha,beta);              // dXn2 = dGT@Wg^T
            gemmTN(h,dUP,Wu[b],dXn2,N,D,HI,alpha,1.0f);              //     += dUP@Wu^T
            rmsnorm_bwd<<<N,64,0,st>>>(XA[b],g2[b],dXn2,Ri2[b],dTmp,dg2[b],N,D);
            // dXA = dX + dTmp
            CHECK(cudaMemcpyAsync(dXA,dX,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
            add_<<<bHD,th,0,st>>>(dXA,dTmp,N*D);
            // attention backward (dATO = dXA)
            gemmNT(h,ATN[b],dXA,dWo[b],D,D,N,alpha,beta);            // dWo = ATN^T @ dXA
            gemmTN(h,dXA,Wo[b],dATN,N,D,D,alpha,beta);               // dATN = dXA @ Wo^T
            attn_scatter<<<bHD,th,0,st>>>(dATN,dAT,N,T,HH,hd);
            bgemmNT(h,PR[b],dAT,dV,T,hd,T,sS,sQ,sQ,BH,alpha,beta);  // dV = PR^T @ dAT
            bgemmTN(h,dAT,V[b],dPR,T,T,hd,sQ,sQ,sS,BH,alpha,beta);  // dPR = dAT @ V^T
            softmax_bwd<<<BH*T,32,0,st>>>(PR[b],dPR,dSC,BH,T);
            bgemmNN(h,dSC,K[b],dQ,T,hd,T,sS,sQ,sQ,BH,scale,beta);   // dQ = dSC@K*scale
            bgemmNT(h,dSC,Q[b],dK,T,hd,T,sS,sQ,sQ,BH,scale,beta);   // dK = dSC^T@Q*scale
            dqkv_merge<<<bHD,th,0,st>>>(dQ,dK,dV,dQKV,N,T,HH,hd);
            gemmNT(h,Xn1[b],dQKV,dWqkv[b],D,3*D,N,alpha,beta);       // dWqkv = Xn1^T @ dQKV
            gemmTN(h,dQKV,Wqkv[b],dXn1,N,D,3*D,alpha,beta);          // dXn1 = dQKV @ Wqkv^T
            rmsnorm_bwd<<<N,64,0,st>>>(XB[b],g1[b],dXn1,Ri1[b],dTmp,dg1[b],N,D);
            add_<<<bHD,th,0,st>>>(dXA,dTmp,N*D);                    // dX(prev) = dXA + dTmp
            CHECK(cudaMemcpyAsync(dX,dXA,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
        }
        // ===== Muon (2D) + SGD (1D) =====
        for(int b=0;b<NBLK;b++){
            muon(Wqkv[b],dWqkv[b],bufWqkv[b],D,3*D);
            muon(Wo[b],dWo[b],bufWo[b],D,D);
            muon(Wg[b],dWg[b],bufWg[b],D,HI);
            muon(Wu[b],dWu[b],bufWu[b],D,HI);
            muon(Wd[b],dWd[b],bufWd[b],HI,D);
            sgd<<<64,64,0,st>>>(g1[b],dg1[b],lr,D);
            sgd<<<64,64,0,st>>>(g2[b],dg2[b],lr,D);
        }
    };
    // diagnostic: 1 step, cek besar gradien (untuk cari bug divergensi)
    step(); CHECK(cudaStreamSynchronize(st));
    auto maxabs=[&](const float* p,int n){ std::vector<float> v(n); CHECK(cudaMemcpy(v.data(),p,sizeof(float)*n,cudaMemcpyDeviceToHost)); float m=0; for(float x:v) m=fmaxf(m,fabsf(x)); return m; };
    printf("DIAG 1-step: maxX=%.4f maxdWqkv5=%.4f maxdX=%.4f maxdWo5=%.4f maxdWd5=%.4f\n",
           maxabs(X,N*D), maxabs(dWqkv[NBLK-1],D*3*D), maxabs(dX,N*D), maxabs(dWo[NBLK-1],D*D), maxabs(dWd[NBLK-1],HI*D));
    for(int i=1;i<WARMUP;i++){ step(); } CHECK(cudaStreamSynchronize(st));
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<STEPS;i++){ step(); }
    CHECK(cudaStreamSynchronize(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double tok=(double)STEPS*N;
    float sX[4],sW[4]; int finite=1;
    CHECK(cudaMemcpy(sX,X,sizeof(float)*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sW,Wqkv[NBLK-1],sizeof(float)*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<4;i++){ if(!std::isfinite(sX[i])||!std::isfinite(sW[i])) finite=0; }
    std::vector<float> hX(N*D),hTg(N*D);
    CHECK(cudaMemcpy(hX.data(),X,sizeof(float)*N*D,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hTg.data(),Tg,sizeof(float)*N*D,cudaMemcpyDeviceToHost));
    double mse=0.0; for(int i=0;i<N*D;i++){ double d=hX[i]-hTg[i]; mse+=d*d; } mse/=(double)(N*D);
    printf("MUON6 (6-block + Muon): %d steps, %.2f ms, %.1f us/step, %.1f tok/s | finite=%d mse=%.4f X0=%.4f\n",
           STEPS,ms,ms*1000.0/STEPS,tok/(ms/1000.0),finite,mse,sX[0]);
    return 0;
}