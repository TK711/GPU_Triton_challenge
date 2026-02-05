"""
Day 3: Matrix Subtraction 학습 템플릿
Day 2의 벡터 연산을 2D 행렬로 확장

학습 목표:
1. 2D 그리드 설정 이해
2. 2D 인덱싱 패턴
3. 2D mask 처리
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Day 2 복습: 1D 벡터 처리
# ============================================================================

@triton.jit
def vector_sub_day2(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Day 2 스타일: 1D 벡터"""

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask = mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    out = a - b
    tl.store(out_ptr + offsets, out, mask=mask)    


# ============================================================================
# Day 3: 2D 행렬 처리
# ============================================================================

@triton.jit
def matrix_sub_kernel(
    a_ptr, b_ptr, out_ptr,
    M: int,  # 행 개수
    N: int,  # 열 개수
    BLOCK_SIZE: tl.constexpr
):
    """
    Step 1: 2D 그리드 설정
    - pid_x: 행 방향 프로그램 ID
    - pid_y: 열 방향 프로그램 ID
    """
    # TODO: 작성해보세요
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pass


@triton.jit
def matrix_sub_kernel_step2(
    a_ptr, b_ptr, out_ptr,
    M: int, N: int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Step 2: 2D 인덱스 계산
    - 각 프로그램이 BLOCK_SIZE × BLOCK_SIZE 블록 처리
    """
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # TODO: 행/열 인덱스 계산
    row_indices = pid_x * BLOCK_SIZE # + tl.arange(0,BLOCKSIZE)
    col_indices = row_indices * BLOCK_SIZE # + tl.arange(0,BLOCKSIZE)
    pass


@triton.jit
def matrix_sub_kernel_step3(
    a_ptr, b_ptr, out_ptr,
    M: int, N: int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Step 3: 2D Mask 처리
    - row_mask: 행 범위 체크
    - col_mask: 열 범위 체크
    - valid_mask: 둘 다 유효한 영역
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # offsets, 데이터 위치
    row_indices = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_indices = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # TODO: 2D mask 생성
    row_mask = row_indices < N
    col_mask = col_indices < N
    valid_mask = row_mask[:None] & col_mask[None:]  # 브로드캐스팅 사용: row_mask[:, None] & col_mask[None, :]
    pass


@triton.jit
def matrix_sub_kernel_step4(
    a_ptr, b_ptr, out_ptr,
    M: int, N: int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Step 4: Flat Index 계산 및 데이터 로드/저장
    - 2D 인덱스를 1D flat index로 변환
    - 행렬은 row-major로 저장됨
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    row_indices = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_indices = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    row_mask = row_indices < M
    col_mask = col_indices < N
    valid_mask = row_mask[:, None] & col_mask[None, :]

    flat_indices = row_indices[:None] * N + col_indices[None:]
    
    a = tl.load(a_ptr + flat_indices, mask=valid_mask)
    b = tl.load(b_ptr + flat_indices, mask = valid_mask)
    out = a-b
    tl.store(out_ptr + valid_mask,out,mask=valid_mask)

    # TODO: Flat index 계산
    # flat_indices = row_indices[:, None] * N + col_indices[None, :]
    
    # TODO: 데이터 로드
    # a = tl.load(a_ptr + flat_indices, mask=valid_mask)
    # b = tl.load(b_ptr + flat_indices, mask=valid_mask)
    
    # TODO: 연산 및 저장
    # out = ?
    # tl.store(?, ?, mask=valid_mask)
    pass


@triton.jit
def matrix_sub_kernel(
    a_ptr, b_ptr, out_ptr,
    M: int, N: int,
    BLOCK_SIZE: tl.constexpr
):
    """
    완성된 Matrix Subtraction 커널
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # 행/열 인덱스
    row_indices = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_indices = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2D mask
    row_mask = row_indices < M
    col_mask = col_indices < N
    valid_mask = row_mask[:, None] & col_mask[None, :]
    
    # Flat index (row-major)
    flat_indices = row_indices[:, None] * N + col_indices[None, :]
    
    # 데이터 로드
    a = tl.load(a_ptr + flat_indices, mask=valid_mask, other=0.0)
    b = tl.load(b_ptr + flat_indices, mask=valid_mask, other=0.0)
    
    # 연산
    out = a - b
    
    # 저장
    tl.store(out_ptr + flat_indices, out, mask=valid_mask)


def matrix_sub_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix Subtraction 래퍼 함수
    
    Args:
        a: Input tensor (M, N) on CUDA
        b: Input tensor (M, N) on CUDA
    
    Returns:
        a - b
    """
    M, N = a.shape
    out = torch.empty_like(a)
    
    # TODO: 2D 그리드 설정
    # grid = lambda meta: (?, ?)
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    
    matrix_sub_kernel[grid](a, b, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


# ============================================================================
# 테스트 함수
# ============================================================================

def test_matrix_sub():
    """단계별 테스트"""
    
    # 작은 크기로 시작
    M, N = 8, 8
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # PyTorch 정답
    expected = a - b
    
    # Triton 결과
    result = matrix_sub_triton(a, b)
    
    # 비교
    print(f"Input shape: {a.shape}")
    print(f"Max difference: {(result - expected).abs().max().item():.6f}")
    print(f"All close: {torch.allclose(result, expected, rtol=1e-5)}")
    
    return torch.allclose(result, expected, rtol=1e-5)


def test_various_sizes():
    """다양한 크기 테스트"""
    test_cases = [
        (1, 1),
        (8, 8),
        (16, 16),
        (32, 32),
        (100, 50),
        (50, 100),
    ]
    
    for M, N in test_cases:
        a = torch.randn(M, N, device='cuda', dtype=torch.float32)
        b = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        expected = a - b
        result = matrix_sub_triton(a, b)
        
        is_close = torch.allclose(result, expected, rtol=1e-5)
        print(f"Shape ({M}, {N}): {'✅' if is_close else '❌'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 3: Matrix Subtraction 학습")
    print("=" * 60)
    print("\n각 Step을 순서대로 완성해보세요!")
    print("Step 1 → Step 2 → Step 3 → Step 4 순서로 진행하세요.\n")
    
    # 테스트 실행
    test_matrix_sub()
    test_various_sizes()
