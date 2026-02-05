import torch
import triton
import triton.language as tl

@triton.jit
def axpby(a_ptr, b_ptr, out_ptr, alpha, beta, N, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offsets= pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask = mask)
    b = tl.load(b_ptr + offsets, mask = mask)

    # axpby
    out = a * alpha + b * beta
    tl.store(out_ptr + offsets,out,mask = mask)



def runner(a: torch.Tensor, b: torch.Tensor, alpha=float, beta=float) -> torch.Tensor:
    
    # 이건 1차원에서만 동작!
    # N = a.shape[0]
    N = a.numel()
    BLOCK_SIZE=128
    # alp = 10
    # beta = 20
    # a = torch.arange(0, N, device='cuda', dtype=torch.float32)
    # b = torch.arange(0, N, device='cuda', dtype=torch.float32)
    out=torch.empty_like(a)
    
    # 없으면 에러남, grid는 몇개의 프로세스를 실행할지 GPU에게 전달해준다
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    axpby[grid](a,b,out,alpha, beta,N,BLOCK_SIZE)

    return out

if __name__ == "__main__":
    print("=" * 60)
    print("Day 4: axpby 검증")
    print("=" * 60)
    
    # 테스트 케이스
    alpha, beta = 10.0, 20.0
    
    test_cases = [
        (10,),      # 1D
        (5, 5),     # 2D
        (3, 4, 2),  # 3D
    ]
    
    for shape in test_cases:
        a = torch.randn(shape, device='cuda', dtype=torch.float32)
        b = torch.randn(shape, device='cuda', dtype=torch.float32)
        
        # Triton 결과
        triton_out = runner(a, b, alpha, beta)
        
        # PyTorch 참조 결과 (정답)
        torch_out = a * alpha + b * beta
        
        # 검증
        is_correct = torch.allclose(triton_out, torch_out, atol=1e-5)
        max_diff = (triton_out - torch_out).abs().max().item()
        
        print(f"Shape {shape}: {'✅ PASS' if is_correct else '❌ FAIL'}, "
              f"max diff = {max_diff:.2e}, dtype = {triton_out.dtype}")