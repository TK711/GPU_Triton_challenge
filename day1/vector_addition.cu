#include <cuda_runtime.h>
#include <stdio.h>

// A 배열, B 배열, C 배열, 전체 일렬로 변경시 데이터 크기 N, 
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // printf("hi! %p",A);
    // tID : 데이터 위치
    // 블록 개수 * 블록 크기 + 블록 내 위치 - 1차원
    int tID = (blockIdx.x * blockDim.x) + threadIdx.x;
    // 범위 N 안인가
    if (tID < N) {
        C[tID] = A[tID] + B[tID];
        // printf("hi! %p", A);
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // 블록내 스레레드 개수
    int threadsPerBlock = 256;
    // 그리드내 블록 수
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 벡터 합 함수 실행
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    // 종료료시까지 대기
    cudaDeviceSynchronize();
}
