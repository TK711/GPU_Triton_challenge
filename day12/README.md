# Day 12: 2D Strided Load

## 📋 실습 목표
Row-major/Column-major 레이아웃 변환 - Stride를 활용한 메모리 접근

---

## 📚 실습 전 복습 사항

### 1. 메모리 레이아웃 이해
```python
# Row-major (C-style, PyTorch 기본)
# shape (3, 4):
# [0  1  2  3]
# [4  5  6  7]
# [8  9 10 11]
# 메모리: [0,1,2,3,4,5,6,7,8,9,10,11]

# Column-major (Fortran-style)
# [0  3  6  9]
# [1  4  7 10]
# [2  5  8 11]
# 메모리: [0,1,2,3,4,5,6,7,8,9,10,11] (같지만 해석이 다름)
```

### 2. Stride 개념
```python
# shape (M, N) row-major
# element[i, j] = base + i * N + j
#                       ↑ stride

# stride = 한 행을 건너뛰기 위한 오프셋
```

---

## 🎯 실습으로 배울 사항

### 1. 2D 인덱싱

```python
@triton.jit
def load_2d_kernel(input_ptr, output_ptr, M, N, 
                   input_stride, output_stride,
                   BLOCK_SIZE: tl.constexpr):
    # 각 프로그램이 한 행 처리
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Row-major 접근
    input_offset = row_id * input_stride + col_offsets
    data = tl.load(input_ptr + input_offset, mask=mask)
    
    # 다른 stride로 저장 (레이아웃 변환)
    output_offset = row_id * output_stride + col_offsets
    tl.store(output_ptr + output_offset, data, mask=mask)
```

### 2. Transpose 구현
```python
# input: (M, N)
# output: (N, M)
# 
# input[i, j]  → output[j, i]
# input offset: i * N + j
# output offset: j * M + i
```

---

## ⚠️ 주의 사항

### 1. Stride 계산
```python
# PyTorch tensor
a = torch.randn(M, N)
stride_row = a.stride(0)  # N (다음 행으로 가는 거리)
stride_col = a.stride(1)  # 1 (다음 열로 가는 거리)
```

### 2. Contiguous vs Non-contiguous
```python
a = torch.randn(3, 4)       # contiguous
b = a.t()                   # non-contiguous!
b = b.contiguous()          # contiguous로 변환
```

---

## 📝 실습 과제

1. Row-major → Column-major 변환
2. Transpose 구현
3. Strided 메모리 접근 패턴 이해

---

## 💡 핵심

**2D 접근 = base + row * stride + col**
