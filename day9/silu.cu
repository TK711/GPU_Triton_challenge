#include <cuda_runtime.h>
#include <math.h>

// Day 9: SiLU (Swish) - x * sigmoid(x)

__global__ void silu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        
        // sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        
        // SiLU = x * sigmoid(x)
        output[tid] = x * sigmoid_x;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    silu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
