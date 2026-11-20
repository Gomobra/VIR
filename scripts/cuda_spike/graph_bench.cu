// Bukti konsep: CUDA graph capture vs eager utk matmul (cuBLAS).
// Mengukur overhead luncuran kernel — yang jadi dinding Burn (~0.7x PyTorch).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} } while(0)

int main(){
    const int M=64, N=64, K=64;   // matmul [M,K]@[K,N]
    const int ITERS=500;
    cublasHandle_t h; CK(cublasCreate(&h));
    const float alpha=1.0f, beta=0.0f;
    float *A,*B,*C;
    CHECK(cudaMalloc(&A,sizeof(float)*M*K));
    CHECK(cudaMalloc(&B,sizeof(float)*K*N));
    CHECK(cudaMalloc(&C,sizeof(float)*M*N));
    std::vector<float> hA(M*K,1.0f), hB(K*N,1.0f);
    CHECK(cudaMemcpy(A,hA.data(),sizeof(float)*M*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B,hB.data(),sizeof(float)*K*N,cudaMemcpyHostToDevice));
    cudaStream_t s; CHECK(cudaStreamCreate(&s)); cublasSetStream(h,s);

    // Eager: launch matmul ITERS times.
    auto matmul = [&](){ CK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,K,A,K,&beta,C,N)); };
    matmul(); CHECK(cudaStreamSynchronize(s)); // warmup
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<ITERS;i++) matmul();
    CHECK(cudaStreamSynchronize(s));
    auto t1=std::chrono::high_resolution_clock::now();
    double eager_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    // CUDA graph: capture ITERS matmuls into one graph, replay.
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for(int i=0;i<ITERS;i++) matmul();
    cudaStreamEndCapture(s, &graph);
    CHECK(cudaGraphInstantiate(&gexec, graph, 0));
    cudaGraphLaunch(gexec, s); CHECK(cudaStreamSynchronize(s)); // warmup
    auto t2=std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(gexec, s);
    CHECK(cudaStreamSynchronize(s));
    auto t3=std::chrono::high_resolution_clock::now();
    double graph_ms=std::chrono::duration<double,std::milli>(t3-t2).count();

    printf("eager:  %d matmul, %.3f ms (%.1f us/call)\n", ITERS, eager_ms, eager_ms*1000.0/ITERS);
    printf("graph:  %d matmul, %.3f ms (%.1f us/call)\n", ITERS, graph_ms, graph_ms*1000.0/ITERS);
    printf("speedup: %.2fx\n", eager_ms/graph_ms);
    return 0;
}