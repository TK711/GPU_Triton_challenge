#include <cuda_runtime.h>
#include <stdio.h>

// Day 2: Vector Scale - y = alpha * x

__global__ void vector_scale_kernel(const float* input, float* output, float alpha, int N) {
    // 스레드 ID 계산
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 경계 체크
    if (tid < N) {
        output[tid] = input[tid] * alpha;
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    // 블록당 스레드 개수
    int threads = 256;
    // 블록 개수 (올림 나눗셈)
    int blocks = (N + threads - 1) / threads;
    
    // 커널 실행
    vector_scale_kernel<<<blocks, threads>>>(input, output, alpha, N);
    
    // GPU 완료 대기
    cudaDeviceSynchronize();
}
