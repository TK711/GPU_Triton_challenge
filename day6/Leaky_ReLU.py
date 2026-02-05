import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu(a_ptr,out_ptr,alpha,N,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets= pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask=offsets<N

    a = tl.load(a_ptr+offsets,mask=mask)
    out=tl.where(a>0,a,a * alpha)
    tl.store(out_ptr+offsets,out,mask=mask)

if __name__ == "__main__" :
    print("=" * 60)
    print("Day 6: leaky relu 검증")
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
    leaky_relu[grid](x,out,alpha,N,BLOCK_SIZE = BLOCK_SIZE)
    print(f"after {out}")