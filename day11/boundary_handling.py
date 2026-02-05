import torch
import triton
import triton.language as tl

@triton.jit
def boundary(a_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """안전한 경계 처리 예시"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    a = tl.load(a_ptr+offsets,mask=mask)

    out = a * 2.0
    
    tl.store(out_ptr+offsets,out,mask=mask)

if __name__ == "__main__" :
    print("=" * 60)
    print("Day 11: boundary handling 검증")
    print("=" * 60)
    alpha=5.00
    N = 500
    BLOCK_SIZE = 128
    
    x = torch.linspace(-1,1,N, device='cuda')
    out = torch.empty_like(x)
    
    print(f"before {x}")
    print(f"")
    # grid = ((num_elements+block_size -1)//block_size,)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    boundary[grid](x,out,N,BLOCK_SIZE = BLOCK_SIZE)

    torch_out = x * 2
    is_correct = torch.allclose(out, torch_out, atol=1e-5)
    max_diff = (out - torch_out).abs().max().item()

    
    print(f"after {out}")
    print(f"✅ PASS" if is_correct else f"❌ FAIL")
    print(f"max diff: {max_diff:.2e}")
    print(f"Triton 결과 (일부): {out[:5]}")
    print(f"PyTorch 결과 (일부): {torch_out[:5]}")
