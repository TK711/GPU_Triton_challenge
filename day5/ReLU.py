import torch
import triton
import triton.language as tl

@triton.jit
def relu(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask=offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 1 maximun
    # tl.maximum(x,0)
    # 2 where
    out=tl.where(x>0,x,0)
    tl.store(out_ptr + offsets, out, mask=mask)

if __name__ == "__main__" :
    print("=" * 60)
    print("Day 5: relu 검증")
    print("=" * 60)
    num_elements = 1000
    block_size = 256
    
    x = torch.linspace(-1,1,num_elements, device='cuda')
    out = torch.empty_like(x)
    
    print(f"before {x}")
    print(f"")
    grid = ((num_elements+block_size -1)//block_size,)
    relu[grid](x,out,num_elements,BLOCK_SIZE = block_size)
    print(f"after {out}")
    # print(out)
