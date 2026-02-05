#include <cuda_runtime.h>

// Day 4: AXPBY - y = alpha * x + beta * y

__global__ void axpby_kernel(const float* a, const float* b, float* out, 
                             float alpha, float beta, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        out[tid] = a[tid] * alpha + b[tid] * beta;
    }
}

extern "C" void solve(const float* a, const float* b, float* out, 
                      float alpha, float beta, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    axpby_kernel<<<blocks, threads>>>(a, b, out, alpha, beta, N);
    cudaDeviceSynchronize();
}
