import torch
import triton
import triton.language as tl

@triton.jit
def tanh(a_ptr,out_ptr,N,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets,mask=mask)
    sigmoid = 1 / (1 + tl.exp(-a))

    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    # tanh = 2*sigmoid(2*x)-1로 대체
    # sigmoid x = 1 / (1 + tl.exp(-a))
    # out = 0.5 * a *(1 + 2 * tl.sigmoid(2 * sqrt_2_over_pi*(a + coeff*a*a*a))-1)
    out = 2 * (1 / (1 + tl.exp(-a*2))) -1
    tl.store(out_ptr+offsets,out,mask=mask)

if __name__ == "__main__" :
    print("=" * 60)
    print("Day 10: tanh 검증")
    print("=" * 60)
    N = 500
    BLOCK_SIZE = 128
    
    x = torch.linspace(-1,1,N, device='cuda')
    out = torch.empty_like(x)
    
    print(f"before {x}")
    print(f"")
    # grid = ((num_elements+block_size -1)//block_size,)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    tanh[grid](x,out,N,BLOCK_SIZE = BLOCK_SIZE)

    torch_out = torch.nn.functional.tanh(x)
    is_correct = torch.allclose(out, torch_out, atol=1e-5)
    max_diff = (out - torch_out).abs().max().item()

    
    print(f"after {out}")

    print(f"✅ PASS" if is_correct else f"❌ FAIL")
    print(f"max diff: {max_diff:.2e}")
    print(f"Triton 결과 (일부): {out[:5]}")
    print(f"PyTorch 결과 (일부): {torch_out[:5]}")