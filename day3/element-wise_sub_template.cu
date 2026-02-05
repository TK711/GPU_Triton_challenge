#include <cuda_runtime.h>

// Day 3: Element-wise Subtraction - y = a - b

__global__ void vector_sub_kernel(const float* a, const float* b, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        out[tid] = a[tid] - b[tid];
    }
}

extern "C" void solve(const float* a, const float* b, float* out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    vector_sub_kernel<<<blocks, threads>>>(a, b, out, N);
    cudaDeviceSynchronize();
}
