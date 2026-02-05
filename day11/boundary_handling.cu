#include <cuda_runtime.h>

// Day 11: Boundary Handling - Safe memory access

__global__ void safe_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 경계 체크 (이게 핵심!)
    if (tid < N) {
        output[tid] = input[tid] * 2.0f;
    }
    // tid >= N이면 아무것도 안 함 (안전)
}

// 위험한 버전 (비교용 - 실제로 사용하지 말 것!)
__global__ void unsafe_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ❌ 경계 체크 없음 - 위험!
    output[tid] = input[tid] * 2.0f;
    // tid >= N이면 범위 초과 접근!
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;  // 올림 나눗셈 중요!
    
    safe_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
