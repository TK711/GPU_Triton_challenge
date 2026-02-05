# Day 4: Fused AXPBY

## 📋 실습 목표
`y = alpha * x + beta * y` - Fused 연산 및 Load/Store 최적화

---

## 📚 실습 전 복습 사항

### 1. Day 1-3 복습
- [ ] 기본 커널 구조 (program_id, offsets, mask)
- [ ] `tl.load`, `tl.store` 사용법
- [ ] Element-wise 연산 패턴

### 2. Scalar Broadcasting (Day 2)
- [ ] Scalar 값과 텐서의 곱셈
- [ ] `alpha * x` 패턴

### 3. 핵심 개념
```python
# 기본 구조
@triton.jit
def kernel(x_ptr, y_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Fused 연산: alpha * x + beta * y
    result = alpha * x + beta * y
    tl.store(y_ptr + offsets, result, mask=mask)  # in-place
```

---

## 🎯 실습으로 배울 사항

### 1. Fused 연산 개념 ⭐ 핵심!
- [ ] 여러 연산을 하나의 커널로 통합
- [ ] 메모리 접근 최소화
- [ ] 중간 결과를 SRAM에 저장

### 2. Load/Store 최적화
- [ ] **Naive 방식**: `y = alpha * x` → `y = y + beta * y` (2번의 메모리 접근)
- [ ] **Fused 방식**: `y = alpha * x + beta * y` (1번의 메모리 접근)
- [ ] 메모리 대역폭 절약

### 3. In-place 연산
- [ ] 출력을 입력 텐서에 직접 저장
- [ ] 추가 메모리 할당 불필요

### 4. Scalar 파라미터 전달
```python
# Python 래퍼에서
def axpby_triton(x, y, alpha, beta):
    # alpha, beta는 scalar 값
    kernel[grid](x, y, alpha, beta, N, BLOCK_SIZE=...)
```

---

## ⚠️ 주의 사항

### 1. 메모리 접근 순서
- **Load 최적화**: `x`와 `y`를 동시에 로드
- **Store 최적화**: 결과를 `y`에 직접 저장 (in-place)

### 2. In-place 연산 주의
- 입력 `y`가 수정됨!
- 원본 보존이 필요하면 복사본 사용
```python
# 원본 보존
y_copy = y.clone()
result = axpby_triton(x, y_copy, alpha, beta)
```

### 3. Scalar 타입
- `alpha`, `beta`는 Python float 또는 torch scalar
- Triton에서 자동으로 처리됨

### 4. 수치 안정성
- 큰 `alpha`, `beta` 값 주의
- 오버플로우 가능성 체크

### 5. 디버깅 팁
```python
# 작은 크기로 테스트
N = 8
x = torch.ones(N, device='cuda') * 2.0
y = torch.ones(N, device='cuda') * 3.0
alpha, beta = 0.5, 0.7

# PyTorch 결과
expected = alpha * x + beta * y
result = axpby_triton(x, y, alpha, beta)

# y가 in-place로 수정되었는지 확인
assert torch.allclose(result, expected)
assert torch.allclose(y, expected)  # y도 수정됨
```

---

## 📝 체크리스트

- [ ] Day 1-3 복습 완료
- [ ] Fused 연산 개념 이해
- [ ] `y = alpha * x + beta * y` 구현
- [ ] In-place 연산 이해
- [ ] Load/Store 최적화 이해
- [ ] 다양한 alpha, beta 값 테스트
- [ ] PyTorch 결과와 비교

---

## 🔗 참고 자료

- Day 1-3 코드
- Fused 연산은 Fused Softmax, Layer Norm, Fused Attention의 기초

---

## 💡 다음 단계

이 Day 4에서 배운 **Fused 연산** 개념은:
- Fused Softmax에서 활용
- Layer Normalization에서 활용
- Fused Attention에서 핵심!