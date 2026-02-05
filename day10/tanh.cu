#include <cuda_runtime.h>
#include <math.h>

// Day 10: Tanh - (exp(2x) - 1) / (exp(2x) + 1)

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        
        // Method 1: tanh(x) = 2 * sigmoid(2x) - 1
        float sigmoid_2x = 1.0f / (1.0f + expf(-2.0f * x));
        output[tid] = 2.0f * sigmoid_2x - 1.0f;
        
        // Method 2: 직접 계산
        // float exp_2x = expf(2.0f * x);
        // output[tid] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
