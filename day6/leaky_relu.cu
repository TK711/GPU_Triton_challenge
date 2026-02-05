#include <cuda_runtime.h>

// Day 6: Leaky ReLU - x > 0 ? x : alpha * x

__global__ void leaky_relu_kernel(const float* input, float* output, 
                                  float alpha, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        output[tid] = (x > 0.0f) ? x : (alpha * x);
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}
