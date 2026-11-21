// Proof-of-concept: training loop C++/CUDA murni (cuBLAS) utk MLP 2-layer.
// Bobot disimpan TRANSPOSED [in,out] -> matmul standar (leading-dim valid).
//   W1[D,H], b1[H], W2[H,D], b2[D]
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} } while(0)

__global__ void gelu_fwd(const float* z, float* a, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float x=z[i]; a[i]=0.5f*x*(1.0f+erff(x*0.7071067811865476f));
}
__global__ void gelu_bwd(const float* z, const float* da, float* dz, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float x=z[i];
    float g=0.5f*(1.0f+erff(x*0.7071067811865476f)) + 0.5f*x*expf(-x*x)*0.7978845608f;
    dz[i]=da[i]*g;
}
__global__ void mse_grad(const float* y, const float* t, float* dy, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    dy[i]=(y[i]-t[i])/(float)n;
}
__global__ void sgd(float* w, const float* gw, float lr, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    w[i]-=lr*gw[i];
}

int main(){
    const int B=64, L=32, D=64, H=256, N=B*L;
    const int STEPS=500, WARMUP=20;
    cublasHandle_t h; CK(cublasCreate(&h));
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    const float alpha=1.0f, beta=0.0f, lr=1e-3f;
    // row-major, transposed: W1[D,H], b1[H], W2[H,D], b2[D]
    std::vector<float> hW1(D*H), hb1(H), hW2(H*D), hb2(D);
    srand(0);
    for(auto&v:hW1) v=((rand()%1000)/1000.0f-0.5f)*0.1f;
    for(auto&v:hb1) v=0.0f;
    for(auto&v:hW2) v=((rand()%1000)/1000.0f-0.5f)*0.1f;
    for(auto&v:hb2) v=0.0f;
    float *dW1,*db1,*dW2,*db2,*gW1,*gb1,*gW2,*gb2;
    CHECK(cudaMalloc(&dW1,sizeof(float)*D*H)); CHECK(cudaMalloc(&db1,sizeof(float)*H));
    CHECK(cudaMalloc(&dW2,sizeof(float)*H*D)); CHECK(cudaMalloc(&db2,sizeof(float)*D));
    CHECK(cudaMalloc(&gW1,sizeof(float)*D*H)); CHECK(cudaMalloc(&gb1,sizeof(float)*H));
    CHECK(cudaMalloc(&gW2,sizeof(float)*H*D)); CHECK(cudaMalloc(&gb2,sizeof(float)*D));
    CHECK(cudaMemcpy(dW1,hW1.data(),sizeof(float)*D*H,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db1,hb1.data(),sizeof(float)*H,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dW2,hW2.data(),sizeof(float)*H*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db2,hb2.data(),sizeof(float)*D,cudaMemcpyHostToDevice));
    float *X,*Z1,*A1,*dZ1,*dA1,*Y,*dY,*T;
    CHECK(cudaMalloc(&X,sizeof(float)*N*D)); CHECK(cudaMalloc(&Z1,sizeof(float)*N*H));
    CHECK(cudaMalloc(&A1,sizeof(float)*N*H)); CHECK(cudaMalloc(&dZ1,sizeof(float)*N*H));
    CHECK(cudaMalloc(&dA1,sizeof(float)*N*H)); CHECK(cudaMalloc(&Y,sizeof(float)*N*D));
    CHECK(cudaMalloc(&dY,sizeof(float)*N*D)); CHECK(cudaMalloc(&T,sizeof(float)*N*D));
    std::vector<float> hX(N*D), hT(N*D);
    for(auto&v:hX) v=((rand()%1000)/1000.0f-0.5f);
    for(auto&v:hT) v=((rand()%1000)/1000.0f-0.5f);
    CHECK(cudaMemcpy(X,hX.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(T,hT.data(),sizeof(float)*N*D,cudaMemcpyHostToDevice));
    cudaStream_t s; CHECK(cudaStreamCreate(&s)); cublasSetStream(h,s);
    int th=256; int bH=(N*H+th-1)/th; int bD=(N*D+th-1)/th;

    // Row-major C=A@B: cublasSgemm(OP_N,OP_N, n,m,k, alpha, B,n, A,k, beta, C,n)
    auto step = [&](){
        // Row-major C=A@B^T: cublasSgemm(OP_T,OP_T, n,m,k, A,k, B,k, C,m?) ; ldc=row stride
        // Z1[N,H] = X[N,D] @ W1^T[H,D]
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, H, N, D, &alpha, dW1, H, X, D, &beta, Z1, H));
        gelu_fwd<<<bH,th,0,s>>>(Z1, A1, N*H);
        // Y[N,D] = A1[N,H] @ W2^T[D,H]
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, D, N, H, &alpha, dW2, D, A1, H, &beta, Y, D));
        mse_grad<<<bD,th,0,s>>>(Y, T, dY, N*D);
        // dA1[N,H] = dY[N,D] @ W2^T[D,H]  (C=A@B^T, B=W2[H,D])
        CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, H, N, D, &alpha, dW2, D, dY, D, &beta, dA1, H));
        gelu_bwd<<<bH,th,0,s>>>(Z1, dA1, dZ1, N*H);
        // dW2[H,D] = A1^T[N,H] @ dY[N,D]  (C=A^T@B, A=A1)
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, D, H, N, &alpha, dY, D, A1, H, &beta, gW2, D));
        // dW1[D,H] = X^T[N,D] @ dZ1[N,H]  (C=A^T@B, A=X)
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, H, D, N, &alpha, dZ1, H, X, D, &beta, gW1, H));
        sgd<<<bH,th,0,s>>>(dW1, gW1, lr, D*H);
        sgd<<<bH,th,0,s>>>(dW2, gW2, lr, H*D);
    };
    for(int i=0;i<WARMUP;i++){ step(); } CHECK(cudaStreamSynchronize(s));
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<STEPS;i++){ step(); }
    CHECK(cudaStreamSynchronize(s));
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double tok=(double)STEPS*N;
    // sanity: cek bobot & aktivasi finite (bukan NaN/Inf -> throughput palsu)
    float hW1s[4], hA1s[4], hW2s[4];
    CHECK(cudaMemcpy(hW1s, dW1, sizeof(float)*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hA1s, A1, sizeof(float)*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hW2s, dW2, sizeof(float)*4, cudaMemcpyDeviceToHost));
    int finite = 1;
    for(int i=0;i<4;i++){ if(!std::isfinite(hW1s[i])||!std::isfinite(hA1s[i])||!std::isfinite(hW2s[i])) finite=0; }
    printf("C++/CUDA: %d steps, %.2f ms, %.1f us/step, %.1f tok/s | finite=%d W1[0]=%.4f A1[0]=%.4f W2[0]=%.4f\n",
           STEPS, ms, ms*1000.0/STEPS, tok/(ms/1000.0), finite, hW1s[0], hA1s[0], hW2s[0]);
    return 0;
}