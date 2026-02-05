import torch
import triton
import triton.language as tl


@triton.jit
# 더할 텐서 a, b 정답 기록할 텐서 c, 전체 텐서 데이터 개수 n, 블록내 스레드개수 BLOCK_SIZE
def vector_scale_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    
    # 프로그램 인스턴스 ID로 각 프로그램은 독립 실행
    pid = tl.program_id(axis=0)
    # 블럭 시작 위치
    block_start=pid * BLOCK_SIZE
    # 이 프로그램이 처리할 모든 인덱스 배열, 벡터 연산용용
    # tl.arange가 뭐지
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 전체 데이터 개수 안넘는지 확인
    # 범위 초과 시 로드 건너뜀, mask 없으면 undefined behavior
    mask = offsets < n_elements
    # 텐서 a에서 범위안 데이터 가져오기, 범위 넘어가면 out-of-boundary 에러
    x = tl.load(a + offsets, mask=mask)
    # 더한 결과 생성 (실제 연산)
    output=x*c
    # 텐서 c에 출력값 저장
    # c에 오프셋은 왜 더하지 -> 포인터 + 오프셋 방식 (c배열 첫 위치 + 오프셋으로 위치)
    # triton은 벡터화된 포인터 연산 사용
    #     # mask는 뭐지지
    
    if pid==0:
        tl.static_print("output")
        tl.device_print("x values",x, output)

    tl.store(b + offsets, output, mask=mask)
    # print("testing")


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: int, N: int):
    # 블럭내 스레드 개수
    BLOCK_SIZE = 128 # 1024
    # 그리드 - 프로그램 인스턴스 개수,각 프로그램은 BLOCK_SIZE만큼 처리
    # 전체 데이터 개수 / 블럭 크기 -> 블럭 개수?
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    # 실제 함수 실행 - CUDA의 <<<>>> 역할
    vector_scale_kernel[grid](a, b, c, N, BLOCK_SIZE)

def main():
    # 설정
    c = 5
    N = 100
    
    # 1. 실제 데이터 생성 (torch.ones, torch.randn 등 사용)
    # 반드시 장치를 'cuda'로 지정해야 Triton이 작동합니다.
    a = torch.arange(0,N, device='cuda', dtype=torch.float32)*0.1 + 1.0
    b = torch.empty(N, device='cuda', dtype=torch.float32) # 결과 저장용 빈 텐서
    
    print(f"입력값 a (앞부분): {a[:10]}")
    print(f"곱할 값 c: {c}")

    # 2. 실행
    solve(a, b, c, N)

    # 3. 결과 확인
    print(f"결과값 b (앞부분): {b[:10]}")

if __name__ == "__main__":
    main()