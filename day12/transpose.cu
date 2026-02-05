#include <cuda_runtime.h>

// Day 12: 2D Strided Load - Transpose

// ============================================================================
// 1D Grid Transpose
// ============================================================================
__global__ void transpose_kernel(const float* input, float* output, int M, int N) {
    // 각 스레드가 하나의 요소를 처리
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < M * N) {
        // 1D index를 2D로 변환
        int i = tid / N;  // row
        int j = tid % N;  // col
        
        // input[i, j] → output[j, i]
        int input_idx = i * N + j;
        int output_idx = j * M + i;
        
        output[output_idx] = input[input_idx];
    }
}


// ============================================================================
// 2D Grid Transpose (더 효율적)
// ============================================================================
__global__ void transpose_2d_kernel(const float* input, float* output, int M, int N) {
    // 2D 스레드 인덱스
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col
    
    if (i < M && j < N) {
        // input[i, j] → output[j, i]
        int input_idx = i * N + j;
        int output_idx = j * M + i;
        
        output[output_idx] = input[input_idx];
    }
}


// ============================================================================
// Shared Memory를 사용한 최적화 버전
// ============================================================================
#define TILE_SIZE 32

__global__ void transpose_tiled_kernel(const float* input, float* output, int M, int N) {
    // Shared memory 타일
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    
    // 글로벌 메모리 인덱스
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Input에서 타일로 로드 (coalesced)
    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    
    __syncthreads();
    
    // Transpose된 위치 계산
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // 타일에서 output으로 쓰기 (coalesced)
    if (y < N && x < M) {
        output[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}


// ============================================================================
// Host Functions
// ============================================================================

extern "C" void transpose_1d(const float* input, float* output, int M, int N) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    transpose_kernel<<<blocks, threads>>>(input, output, M, N);
    cudaDeviceSynchronize();
}

extern "C" void transpose_2d(const float* input, float* output, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    
    transpose_2d_kernel<<<blocks, threads>>>(input, output, M, N);
    cudaDeviceSynchronize();
}

extern "C" void transpose_tiled(const float* input, float* output, int M, int N) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    
    transpose_tiled_kernel<<<blocks, threads>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
