import torch
import triton
import triton.language as tl

# ============================================================================
# Transpose: Row-major (M, N) → (N, M)
# ============================================================================
@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Transpose: (M, N) → (N, M)
    
    각 프로그램이 input의 한 행을 읽어서
    output의 한 열에 쓴다
    """
    row_id = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Input에서 row_id 행 읽기
    # input[row_id, :] = input_ptr + row_id * N + col_offsets
    input_offset = row_id * N + col_offsets
    data = tl.load(input_ptr + input_offset, mask=mask)
    
    # Output에 열로 쓰기
    # output[col_offsets, row_id] = output_ptr + col_offsets * M + row_id
    output_offset = col_offsets * M + row_id
    tl.store(output_ptr + output_offset, data, mask=mask)


# ============================================================================
# 2D Grid Transpose (더 효율적)
# ============================================================================
@triton.jit
def transpose_2d_kernel(input_ptr, output_ptr, M, N, 
                        BM: tl.constexpr, BN: tl.constexpr):
    """
    2D Grid를 사용한 Transpose
    각 블록이 (BM x BN) 타일을 처리
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 2D 인덱스
    i = pid_m * BM + tl.arange(0, BM)[:, None]
    j = pid_n * BN + tl.arange(0, BN)[None, :]
    
    mask = (i < M) & (j < N)
    
    # Input 읽기: input[i, j]
    input_offset = i * N + j
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Output 쓰기: output[j, i]
    output_offset = j * M + i
    tl.store(output_ptr + output_offset, data, mask=mask)


def transpose_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton transpose wrapper"""
    assert x.is_cuda and x.ndim == 2
    
    M, N = x.shape
    out = torch.empty(N, M, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    grid = (M,)
    transpose_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def transpose_triton_2d(x: torch.Tensor, BM: int = 32, BN: int = 32) -> torch.Tensor:
    """Triton 2D transpose wrapper"""
    assert x.is_cuda and x.ndim == 2
    
    M, N = x.shape
    out = torch.empty(N, M, device=x.device, dtype=x.dtype)
    
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    transpose_2d_kernel[grid](x, out, M, N, BM=BM, BN=BN, num_warps=4)
    
    return out


if __name__ == "__main__":
    print("=" * 80)
    print("Day 12: 2D Strided Load - Transpose")
    print("=" * 80)
    
    # ========================================================================
    # 테스트 1: 작은 행렬 (3x4)
    # ========================================================================
    print("\n[테스트 1] 작은 행렬 Transpose (3x4)")
    print("-" * 80)
    
    x = torch.arange(12, device='cuda', dtype=torch.float32).reshape(3, 4)
    print(f"Input (3x4):\n{x}")
    print(f"Input 메모리: {x.flatten()}")
    
    out = transpose_triton(x)
    print(f"\nOutput (4x3):\n{out}")
    print(f"Output 메모리: {out.flatten()}")
    
    expected = x.t().contiguous()
    is_correct = torch.allclose(out, expected)
    print(f"\n검증: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # ========================================================================
    # 테스트 2: 큰 행렬 (1D Grid)
    # ========================================================================
    print("\n[테스트 2] 큰 행렬 Transpose - 1D Grid (512x1024)")
    print("-" * 80)
    
    M, N = 512, 1024
    x = torch.randn(M, N, device='cuda')
    
    out = transpose_triton(x)
    expected = x.t().contiguous()
    
    is_correct = torch.allclose(out, expected, atol=1e-5)
    max_diff = (out - expected).abs().max().item()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Max diff: {max_diff:.2e}")
    print(f"결과: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # ========================================================================
    # 테스트 3: 큰 행렬 (2D Grid)
    # ========================================================================
    print("\n[테스트 3] 큰 행렬 Transpose - 2D Grid (512x1024)")
    print("-" * 80)
    
    out_2d = transpose_triton_2d(x, BM=32, BN=32)
    
    is_correct = torch.allclose(out_2d, expected, atol=1e-5)
    max_diff = (out_2d - expected).abs().max().item()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_2d.shape}")
    print(f"Max diff: {max_diff:.2e}")
    print(f"결과: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # ========================================================================
    # 핵심 요약
    # ========================================================================
    print("\n" + "=" * 80)
    print("핵심 요약")
    print("=" * 80)
    print("1. Transpose: input[i,j] → output[j,i]")
    print("2. Input offset:  row_id * N + col_offsets (행으로 읽기)")
    print("3. Output offset: col_offsets * M + row_id (열로 쓰기)")
    print("4. 2D Grid: 더 효율적인 메모리 접근 패턴")
    print("=" * 80)
