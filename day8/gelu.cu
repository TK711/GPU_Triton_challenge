#include <cuda_runtime.h>
#include <math.h>

// Day 8: GeLU - 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

__device__ float tanh_approx(float x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    float sigmoid_2x = 1.0f / (1.0f + expf(-2.0f * x));
    return 2.0f * sigmoid_2x - 1.0f;
}

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float x = input[tid];
        
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        
        // GeLU 근사
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_inner = tanh_approx(inner);
        
        output[tid] = 0.5f * x * (1.0f + tanh_inner);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
