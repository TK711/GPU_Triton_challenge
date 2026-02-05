#include <cuda_runtime.h>

// Day 5: ReLU - max(0, x)

__global__ void relu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        output[tid] = (x > 0.0f) ? x : 0.0f;
        // 또는: output[tid] = fmaxf(x, 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
