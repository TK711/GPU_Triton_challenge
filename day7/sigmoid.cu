#include <cuda_runtime.h>
#include <math.h>

// Day 7: Sigmoid - 1 / (1 + exp(-x))

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        output[tid] = 1.0f / (1.0f + expf(-x));
    }
}

// Numerical stable version
__global__ void sigmoid_stable_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        float result;
        
        // 수치 안정성: x의 부호에 따라 다르게 계산
        if (x >= 0.0f) {
            result = 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            result = exp_x / (1.0f + exp_x);
        }
        
        output[tid] = result;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    sigmoid_stable_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
