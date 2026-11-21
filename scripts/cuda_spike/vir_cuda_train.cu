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
#include <string>

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

// ---- VIR: VQ encode (one block per position, CB threads) ----
__global__ void vq_encode(const float* z, const float* cb, float* zres, int N, int D, int CB){
    int i=blockIdx.x; if(i>=N) return; int t=threadIdx.x;
    float d=0; for(int j=0;j<D;j++){ float df=z[i*D+j]-cb[t*D+j]; d+=df*df; }
    __shared__ float sd[256]; __shared__ int si[256];
    sd[t]=d; si[t]=t; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(t<s){ if(sd[t+s]<sd[t]){ sd[t]=sd[t+s]; si[t]=si[t+s]; } } __syncthreads(); }
    int best=si[0];
    for(int j=t;j<D;j+=CB) zres[i*D+j]=z[i*D+j]-cb[best*D+j];
}
// ---- VIR: gather first l rows of each batch [B,L,D]->[B*l,D] ----
__global__ void gather_rows(const float* X, float* out, int B, int L, int l, int D){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=B*l*D; if(idx>=total)return;
    int d=idx%D; int i=(idx/D)%l; int b=idx/(D*l);
    out[idx]=X[(b*L+i)*D+d];
}
__global__ void scatter_rows(float* X, const float* dhk, int B, int L, int l, int D){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=B*l*D; if(idx>=total)return;
    int d=idx%D; int i=(idx/D)%l; int b=idx/(D*l);
    atomicAdd(&X[(b*L+i)*D+d], dhk[idx]);
}
// ---- VIR: CE forward+backward (one block per row, V threads) ----
__global__ void ce_fwd_bwd(const float* logits, const int* tgt, float* dlog, float* loss, int M, int V){
    int row=blockIdx.x; if(row>=M)return; int t=threadIdx.x;
    __shared__ float sd[512];
    float lv=logits[row*V+t]; sd[t]=lv; __syncthreads();
    for(int s=256;s>0;s>>=1){ if(t<s) sd[t]=fmaxf(sd[t],sd[t+s]); __syncthreads(); }
    float mx=sd[0]; __syncthreads();
    float e=__expf(lv-mx); sd[t]=e; __syncthreads();
    for(int s=256;s>0;s>>=1){ if(t<s) sd[t]+=sd[t+s]; __syncthreads(); }
    float sum=sd[0]; float sm=e/sum; int tg=tgt[row];
    dlog[row*V+t]=(sm-(t==tg?1.0f:0.0f))/(float)M;
    if(t==tg) atomicAdd(loss,-logf(sm+1e-9f));
}

// ---- embedding + target gather ----
__global__ void embed_gather(const float* emb, const int* tok, float* z, int N, int D){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N*D)return;
    int row=i/D, d=i%D; z[i]=emb[tok[row]*D+d];
}
__global__ void embed_scatter(const float* dz, const int* tok, float* demb, int N, int D){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N*D)return;
    int row=i/D, d=i%D; atomicAdd(&demb[tok[row]*D+d], dz[i]);
}
__global__ void gather_tgt(const int* tok, int* out, int B, int T, int l, int offset){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=B*l; if(idx>=total)return;
    int i=idx%l; int b=idx/l; out[idx]=tok[offset+b*T+i];
}

static std::string g_out=".";
static void write_checkpoint(const char* out, int VOCAB,int D,int CB,int NBLK,int K,
    float* emb, float** cb, float** Wqkv, float** Wo, float** Wg, float** Wu, float** Wd, float** g1, float** g2, float** head){
    std::string dir=out;
    std::string manifest=dir+"/manifest.json";
    std::string binp=dir+"/model.bin";
    FILE* mf=fopen(manifest.c_str(),"w");
    fprintf(mf,"{\"format\":\"vir-cuda-1\",\"config\":{\"vocab\":%d,\"d_model\":%d,\"codebook\":%d,\"n_blocks\":%d,\"levels\":5},\"tensors\":[",VOCAB,D,CB,NBLK);
    auto dims=[&](const char* name,int rows,int cols,bool last){
        fprintf(mf,"%s{\"name\":\"%s\",\"shape\":[%d,%d]}",last?"":",",name,rows,cols); };
    dims("embed",VOCAB,D,false);
    for(int k=0;k<K;k++){ char nm[32]; snprintf(nm,32,"codebooks.%d",k); dims(nm,CB,D,false); }
    for(int b=0;b<NBLK;b++){ char nm[64];
        snprintf(nm,64,"blocks.%d.ln1",b); dims(nm,1,D,false);
        snprintf(nm,64,"blocks.%d.wqkv",b); dims(nm,D,3*D,false);
        snprintf(nm,64,"blocks.%d.wo",b); dims(nm,D,D,false);
        snprintf(nm,64,"blocks.%d.ln2",b); dims(nm,1,D,false);
        snprintf(nm,64,"blocks.%d.wg",b); dims(nm,D,4*D,false);
        snprintf(nm,64,"blocks.%d.wu",b); dims(nm,D,4*D,false);
        snprintf(nm,64,"blocks.%d.wd",b); dims(nm,4*D,D,false);
    }
    for(int k=0;k<K;k++){ char nm[32]; snprintf(nm,32,"heads.%d",k); dims(nm,VOCAB,D,k==K-1); }
    fprintf(mf,"]}\n"); fclose(mf);
    FILE* bf=fopen(binp.c_str(),"wb");
    auto wr=[&](const float* p,int n){ fwrite(p,sizeof(float),n,bf); };
    wr(emb,VOCAB*D);
    for(int k=0;k<K;k++) wr(cb[k],CB*D);
    for(int b=0;b<NBLK;b++){ wr(g1[b],D); wr(Wqkv[b],D*3*D); wr(Wo[b],D*D); wr(g2[b],D); wr(Wg[b],D*4*D); wr(Wu[b],D*4*D); wr(Wd[b],4*D*D); }
    for(int k=0;k<K;k++) wr(head[k],VOCAB*D);
    fclose(bf);
    printf("checkpoint: %s + %s\n",manifest.c_str(),binp.c_str());
}

int main(int argc,char** argv){
    int D=64,HH=4,T=32,B=64,NBLK=6,VOCAB=512,CB=256,STEPS=500; float lr=1e-4f;
    const char* tokens_path=nullptr;
    for(int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--tokens") tokens_path=argv[++i];
        else if(a=="--out") g_out=argv[++i];
        else if(a=="--vocab") VOCAB=atoi(argv[++i]);
        else if(a=="--d-model") D=atoi(argv[++i]);
        else if(a=="--n-heads") HH=atoi(argv[++i]);
        else if(a=="--n-blocks") NBLK=atoi(argv[++i]);
        else if(a=="--seq-len") T=atoi(argv[++i]);
        else if(a=="--batch") B=atoi(argv[++i]);
        else if(a=="--steps") STEPS=atoi(argv[++i]);
        else if(a=="--lr") lr=atof(argv[++i]);
    }
    if(!tokens_path){ printf("usage: --tokens <file> [--out dir] [--vocab V] [--d-model D] ...\n"); return 1; }
    int hd=D/HH, HI=4*D, N=B*T, BH=B*HH, NK=5, LEV[5]={2,4,8,16,32};
    const long sQ=(long)T*hd, sS=(long)T*T;
    cublasHandle_t h; CK(cublasCreate(&h)); cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST);
    const float alpha=1.0f,beta=0.0f,scale=1.0f/sqrtf((float)hd);
    srand(0);
    auto rnd=[&](int n,float s){ std::vector<float> v(n); for(auto&x:v)x=((rand()%1000)/1000.0f-0.5f)*s; return v; };
    #define ALLOC1(name,sz) CHECK(cudaMalloc(&name,sizeof(float)*(sz)))
    // load tokens
    std::vector<int> htok; { FILE* f=fopen(tokens_path,"rb"); if(!f){printf("no tokens file\n");return 1;}
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); htok.resize(sz/4);
        if(fread(htok.data(),4,htok.size(),f)!=htok.size()){ printf("read tokens fail\n"); return 1; } fclose(f); }
    int n_tok=htok.size(); if(n_tok < N){ printf("tokens %d < batch %d\n",n_tok,N); return 1; }
    int *dtok; CHECK(cudaMalloc(&dtok,sizeof(int)*n_tok)); CHECK(cudaMemcpy(dtok,htok.data(),sizeof(int)*n_tok,cudaMemcpyHostToDevice));
    // weights
    float *emb,*demb; ALLOC1(emb,VOCAB*D); ALLOC1(demb,VOCAB*D);
    { auto v=rnd(VOCAB*D,0.1f); CHECK(cudaMemcpy(emb,v.data(),sizeof(float)*VOCAB*D,cudaMemcpyHostToDevice)); }
    float *cb[5]; for(int k=0;k<NK;k++){ ALLOC1(cb[k],CB*D); auto v=rnd(CB*D,0.05f); CHECK(cudaMemcpy(cb[k],v.data(),sizeof(float)*CB*D,cudaMemcpyHostToDevice)); }
    float *Wqkv[NBLK],*dWqkv[NBLK],*Wo[NBLK],*dWo[NBLK],*g1[NBLK],*dg1[NBLK],*Wg[NBLK],*dWg[NBLK],*Wu[NBLK],*dWu[NBLK],*Wd[NBLK],*dWd[NBLK],*g2[NBLK],*dg2[NBLK];
    float *bufWqkv[NBLK],*bufWo[NBLK],*bufWg[NBLK],*bufWu[NBLK],*bufWd[NBLK];
    for(int b=0;b<NBLK;b++){
        ALLOC1(Wqkv[b],D*3*D); ALLOC1(dWqkv[b],D*3*D); ALLOC1(Wo[b],D*D); ALLOC1(dWo[b],D*D);
        ALLOC1(g1[b],D); ALLOC1(dg1[b],D); ALLOC1(Wg[b],D*HI); ALLOC1(dWg[b],D*HI);
        ALLOC1(Wu[b],D*HI); ALLOC1(dWu[b],D*HI); ALLOC1(Wd[b],HI*D); ALLOC1(dWd[b],HI*D); ALLOC1(g2[b],D); ALLOC1(dg2[b],D);
        { auto v=rnd(D*3*D,0.1f); CHECK(cudaMemcpy(Wqkv[b],v.data(),sizeof(float)*D*3*D,cudaMemcpyHostToDevice)); }
        { auto v=rnd(D*D,0.1f); CHECK(cudaMemcpy(Wo[b],v.data(),sizeof(float)*D*D,cudaMemcpyHostToDevice)); }
        { auto v=rnd(D*HI,0.1f); CHECK(cudaMemcpy(Wg[b],v.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice)); }
        { auto v=rnd(D*HI,0.1f); CHECK(cudaMemcpy(Wu[b],v.data(),sizeof(float)*D*HI,cudaMemcpyHostToDevice)); }
        { auto v=rnd(HI*D,0.1f); CHECK(cudaMemcpy(Wd[b],v.data(),sizeof(float)*HI*D,cudaMemcpyHostToDevice)); }
        std::vector<float> one(D,1.0f); CHECK(cudaMemcpy(g1[b],one.data(),sizeof(float)*D,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(g2[b],one.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
        ALLOC1(bufWqkv[b],D*3*D); ALLOC1(bufWo[b],D*D); ALLOC1(bufWg[b],D*HI); ALLOC1(bufWu[b],D*HI); ALLOC1(bufWd[b],HI*D);
        CHECK(cudaMemset(bufWqkv[b],0,sizeof(float)*D*3*D)); CHECK(cudaMemset(bufWo[b],0,sizeof(float)*D*D));
        CHECK(cudaMemset(bufWg[b],0,sizeof(float)*D*HI)); CHECK(cudaMemset(bufWu[b],0,sizeof(float)*D*HI)); CHECK(cudaMemset(bufWd[b],0,sizeof(float)*HI*D));
    }
    float *head[5],*dhead[5]; for(int k=0;k<NK;k++){ ALLOC1(head[k],VOCAB*D); ALLOC1(dhead[k],VOCAB*D); auto v=rnd(VOCAB*D,0.05f); CHECK(cudaMemcpy(head[k],v.data(),sizeof(float)*VOCAB*D,cudaMemcpyHostToDevice)); }
    // activations
    float *X,*QKV,*SC,*AT,*ATO,*HA,*MO,*ZX;
    ALLOC1(X,N*D); ALLOC1(QKV,N*3*D); ALLOC1(SC,BH*T*T); ALLOC1(AT,BH*T*hd); ALLOC1(ATO,N*D); ALLOC1(HA,N*HI); ALLOC1(MO,N*D); ALLOC1(ZX,N*D);
    float *XB[NBLK],*XA[NBLK],*Xn1[NBLK],*Ri1[NBLK],*Xn2[NBLK],*Ri2[NBLK],*Q[NBLK],*K[NBLK],*V[NBLK],*PR[NBLK],*ATN[NBLK],*GT[NBLK],*UP[NBLK];
    for(int b=0;b<NBLK;b++){ ALLOC1(XB[b],N*D);ALLOC1(XA[b],N*D);ALLOC1(Xn1[b],N*D);ALLOC1(Ri1[b],N);ALLOC1(Xn2[b],N*D);ALLOC1(Ri2[b],N);
        ALLOC1(Q[b],BH*T*hd);ALLOC1(K[b],BH*T*hd);ALLOC1(V[b],BH*T*hd);ALLOC1(PR[b],BH*T*T);ALLOC1(ATN[b],N*D);ALLOC1(GT[b],N*HI);ALLOC1(UP[b],N*HI); }
    float *dX,*dXA,*dTmp,*dXn1,*dXn2,*dQKV,*dQ,*dK,*dV,*dSC,*dPR,*dAT,*dATN,*dGT,*dUP,*dHH;
    ALLOC1(dX,N*D);ALLOC1(dXA,N*D);ALLOC1(dTmp,N*D);ALLOC1(dXn1,N*D);ALLOC1(dXn2,N*D);ALLOC1(dQKV,N*3*D);
    ALLOC1(dQ,BH*T*hd);ALLOC1(dK,BH*T*hd);ALLOC1(dV,BH*T*hd);ALLOC1(dSC,BH*T*T);ALLOC1(dPR,BH*T*T);ALLOC1(dAT,BH*T*hd);ALLOC1(dATN,N*D);
    ALLOC1(dGT,N*HI);ALLOC1(dUP,N*HI);ALLOC1(dHH,N*HI);
    float *hk[5],*logits[5],*dlog[5]; int *tgt[5]; float *dhk,*loss_dev;
    for(int k=0;k<NK;k++){ int lk=LEV[k]<T?LEV[k]:T; int rows=B*lk; ALLOC1(hk[k],rows*D);ALLOC1(logits[k],rows*VOCAB);ALLOC1(dlog[k],rows*VOCAB);
        CHECK(cudaMalloc(&tgt[k],sizeof(int)*rows)); }
    ALLOC1(dhk,N*D); ALLOC1(loss_dev,1);
    float *Xw,*Aw,*AXw,*A2w,*A2Xw,*norm_dev;
    int maxMN=D*HI; if(D*3*D>maxMN)maxMN=D*3*D; if(HI*D>maxMN)maxMN=HI*D;
    int maxMM=HI*HI; if(3*D*3*D>maxMM)maxMM=3*D*3*D;
    ALLOC1(Xw,maxMN);ALLOC1(AXw,maxMN);ALLOC1(A2Xw,maxMN);ALLOC1(Aw,maxMM);ALLOC1(A2w,maxMM);ALLOC1(norm_dev,1);
    #undef ALLOC1
    cudaStream_t st; CHECK(cudaStreamCreate(&st)); cublasSetStream(h,st);
    int th=256; int bHD=(N*D+th-1)/th, bHI=(N*HI+th-1)/th;
    const float mom=0.95f,nseps=1e-7f; const int NS_STEPS=5;
    auto muon=[&](float* W,const float* G,float* buf,int M,int N){
        int sz=M*N; mom_update<<<bHD,th,0,st>>>(buf,G,mom,sz);
        CHECK(cudaMemsetAsync(norm_dev,0,sizeof(float),st)); sum_sq<<<1,256,0,st>>>(buf,norm_dev,sz);
        scale_norm<<<bHD,th,0,st>>>(buf,Xw,norm_dev,nseps,sz);
        for(int i=0;i<NS_STEPS;i++){ gemmTN(h,Xw,Xw,Aw,M,M,N,alpha,beta); gemmNN(h,Aw,Xw,AXw,M,N,M,alpha,beta);
            gemmNN(h,Aw,Aw,A2w,M,M,M,alpha,beta); gemmNN(h,A2w,Xw,A2Xw,M,N,M,alpha,beta);
            ns_combine<<<bHD,th,0,st>>>(Xw,AXw,A2Xw,3.4445f,-4.7750f,2.0315f,sz); }
        w_update<<<bHD,th,0,st>>>(W,Xw,lr,sz);
    };
    auto step = [&](){
        int offset = (int)((long)0); // sequential from start
        CHECK(cudaMemsetAsync(demb,0,sizeof(float)*VOCAB*D,st));
        for(int b=0;b<NBLK;b++){ zero_<<<64,64,0,st>>>(dg1[b],D); zero_<<<64,64,0,st>>>(dg2[b],D); }
        embed_gather<<<bHD,th,0,st>>>(emb,dtok,X,N,D);
        for(int k=0;k<NK;k++) vq_encode<<<N,CB,0,st>>>(X,cb[k],X,N,D,CB);
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
        CHECK(cudaMemsetAsync(loss_dev,0,sizeof(float),st));
        zero_<<<bHD,th,0,st>>>(dX,N*D);
        for(int k=0;k<NK;k++){ int lk=LEV[k]<T?LEV[k]:T; int rows=B*lk;
            gather_rows<<<bHD,th,0,st>>>(X,hk[k],B,T,lk,D);
            gemmTN(h,hk[k],head[k],logits[k],rows,VOCAB,D,alpha,beta);
            gather_tgt<<<bHD,th,0,st>>>(dtok,tgt[k],B,T,lk,offset);
            ce_fwd_bwd<<<rows,VOCAB,0,st>>>(logits[k],tgt[k],dlog[k],loss_dev,rows,VOCAB);
        }
        for(int k=0;k<NK;k++){ int lk=LEV[k]<T?LEV[k]:T; int rows=B*lk;
            gemmTN(h,dlog[k],hk[k],dhead[k],VOCAB,D,rows,alpha,beta);
            gemmNN(h,dlog[k],head[k],dhk,rows,D,VOCAB,alpha,beta);
            scatter_rows<<<bHD,th,0,st>>>(dX,dhk,B,T,lk,D);
        }
        for(int b=NBLK-1;b>=0;b--){
            silu_mul<<<bHI,th,0,st>>>(GT[b],UP[b],HA,N*HI);
            gemmNT(h,HA,dX,dWd[b],HI,D,N,alpha,beta);
            gemmTN(h,dX,Wd[b],dHH,N,HI,D,alpha,beta);
            swiglu_bwd<<<bHI,th,0,st>>>(GT[b],UP[b],dHH,dGT,dUP,N*HI);
            gemmNT(h,Xn2[b],dGT,dWg[b],D,HI,N,alpha,beta);
            gemmNT(h,Xn2[b],dUP,dWu[b],D,HI,N,alpha,beta);
            gemmTN(h,dGT,Wg[b],dXn2,N,D,HI,alpha,beta);
            gemmTN(h,dUP,Wu[b],dXn2,N,D,HI,alpha,1.0f);
            rmsnorm_bwd<<<N,64,0,st>>>(XA[b],g2[b],dXn2,Ri2[b],dTmp,dg2[b],N,D);
            CHECK(cudaMemcpyAsync(dXA,dX,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
            add_<<<bHD,th,0,st>>>(dXA,dTmp,N*D);
            gemmNT(h,ATN[b],dXA,dWo[b],D,D,N,alpha,beta);
            gemmTN(h,dXA,Wo[b],dATN,N,D,D,alpha,beta);
            attn_scatter<<<bHD,th,0,st>>>(dATN,dAT,N,T,HH,hd);
            bgemmNT(h,PR[b],dAT,dV,T,hd,T,sS,sQ,sQ,BH,alpha,beta);
            bgemmTN(h,dAT,V[b],dPR,T,T,hd,sQ,sQ,sS,BH,alpha,beta);
            softmax_bwd<<<BH*T,32,0,st>>>(PR[b],dPR,dSC,BH,T);
            bgemmNN(h,dSC,K[b],dQ,T,hd,T,sS,sQ,sQ,BH,scale,beta);
            bgemmNT(h,dSC,Q[b],dK,T,hd,T,sS,sQ,sQ,BH,scale,beta);
            dqkv_merge<<<bHD,th,0,st>>>(dQ,dK,dV,dQKV,N,T,HH,hd);
            gemmNT(h,Xn1[b],dQKV,dWqkv[b],D,3*D,N,alpha,beta);
            gemmTN(h,dQKV,Wqkv[b],dXn1,N,D,3*D,alpha,beta);
            rmsnorm_bwd<<<N,64,0,st>>>(XB[b],g1[b],dXn1,Ri1[b],dTmp,dg1[b],N,D);
            add_<<<bHD,th,0,st>>>(dXA,dTmp,N*D);
            CHECK(cudaMemcpyAsync(dX,dXA,sizeof(float)*N*D,cudaMemcpyDeviceToDevice,st));
        }
        embed_scatter<<<bHD,th,0,st>>>(dX,dtok,demb,N,D);
        for(int b=0;b<NBLK;b++){ muon(Wqkv[b],dWqkv[b],bufWqkv[b],D,3*D); muon(Wo[b],dWo[b],bufWo[b],D,D);
            muon(Wg[b],dWg[b],bufWg[b],D,HI); muon(Wu[b],dWu[b],bufWu[b],D,HI); muon(Wd[b],dWd[b],bufWd[b],HI,D);
            sgd<<<64,64,0,st>>>(g1[b],dg1[b],lr,D); sgd<<<64,64,0,st>>>(g2[b],dg2[b],lr,D); }
        sgd<<<bHD,th,0,st>>>(emb,demb,lr,VOCAB*D);
        for(int k=0;k<NK;k++) sgd<<<bHD,th,0,st>>>(head[k],dhead[k],lr,VOCAB*D);
    };
    for(int i=0;i<20;i++) step(); CHECK(cudaStreamSynchronize(st));
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<STEPS;i++) step();
    CHECK(cudaStreamSynchronize(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    float hl=0; CHECK(cudaMemcpy(&hl,loss_dev,sizeof(float),cudaMemcpyDeviceToHost));
    printf("vir-cuda-train: %d steps, %.1f us/step, %.0f tok/s | loss=%.3f vocab=%d D=%d blocks=%d\n",
        STEPS, ms*1000.0/STEPS, (double)STEPS*N/(ms/1000.0), hl, VOCAB, D, NBLK);
    write_checkpoint(g_out.c_str(),VOCAB,D,CB,NBLK,NK,emb,cb,Wqkv,Wo,Wg,Wu,Wd,g1,g2,head);
    return 0;
}
