# Day 9: SiLU (Swish)

## 📋 실습 목표
`x * sigmoid(x)` - 두 함수의 곱셈 (Fused 연산 패턴)

---

## 📚 실습 전 복습 사항

### 1. Day 7 (Sigmoid) 복습 ⭐ 필수!
- [ ] Sigmoid 구현
- [ ] Numerical stability 기법
- [ ] `tl.exp()` 사용법

### 2. Day 4 (Fused AXPBY) 복습
- [ ] Fused 연산 개념
- [ ] 여러 연산을 하나로 통합

### 3. Element-wise 곱셈
- [ ] `x * y` 패턴

### 4. 핵심 개념
```python
# SiLU = x * sigmoid(x)
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Sigmoid 계산 (Day 7 코드 재활용)
    sigmoid_x = sigmoid_stable(x)  # Day 7의 구현
    
    # SiLU: x * sigmoid(x)
    silu = x * sigmoid_x
    tl.store(out_ptr + offsets, silu, mask=mask)
```

---

## 🎯 실습으로 배울 사항

### 1. 함수 조합 (Fused 패턴)
- [ ] 두 함수의 곱셈
- [ ] Day 4의 Fused 연산 개념 재활용
- [ ] 중간 결과 재사용

### 2. SiLU 수학적 이해
- [ ] Swish 활성화 함수
- [ ] `x * sigmoid(x)` 형태
- [ ] Smooth하고 non-monotonic

### 3. 구현 방법
```python
# 방법 1: Sigmoid 함수 재사용 (권장)
def sigmoid_stable(x):
    # Day 7의 구현
    max_x = tl.maximum(x, 0.0)
    exp_neg = tl.exp(-(x - max_x))
    return 1.0 / (1.0 + exp_neg)

sigmoid_x = sigmoid_stable(x)
silu = x * sigmoid_x

# 방법 2: 인라인 구현 (비효율적)
# sigmoid를 매번 계산하면 비효율적
```

### 4. 코드 재사용
- [ ] Day 7의 Sigmoid 함수 재활용
- [ ] 모듈화의 중요성

---

## ⚠️ 주의 사항

### 1. Numerical Stability
- [ ] Day 7의 안정적인 Sigmoid 구현 사용
- [ ] `x * sigmoid(x)`에서도 오버플로우 가능성 있음
- [ ] 큰 x 값 처리 주의

### 2. 성능
- [ ] Sigmoid 계산이 비용이 큼
- [ ] 한 번만 계산하고 재사용
- [ ] 불필요한 중간 계산 방지

### 3. 수치 정확도
- [ ] `x = 0`일 때: `silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0`
- [ ] 매우 큰 x: `silu(x) ≈ x` (sigmoid(x) ≈ 1)
- [ ] 매우 작은 x: `silu(x) ≈ 0` (sigmoid(x) ≈ 0)

### 4. 디버깅 팁
```python
# 다양한 입력 테스트
x = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0], device='cuda')

# PyTorch 결과
expected = x * torch.sigmoid(x)
# 또는
expected = torch.nn.functional.silu(x)

result = silu_triton(x)
assert torch.allclose(result, expected, rtol=1e-4)
```

---

## 📝 체크리스트

- [ ] Day 7 (Sigmoid) 복습 완료 ⭐
- [ ] Day 4 (Fused AXPBY) 복습 완료
- [ ] SiLU 수학적 이해
- [ ] Sigmoid 함수 재사용
- [ ] `x * sigmoid(x)` 구현
- [ ] 다양한 입력 값 테스트
- [ ] PyTorch `torch.nn.SiLU()`와 비교

---

## 🔗 참고 자료

- Day 7 코드: `../day7/` (Sigmoid 구현)
- Day 4 코드: `../day4/` (Fused 연산 개념)
- PyTorch: `torch.nn.SiLU()` 또는 `x * torch.sigmoid(x)`

---

## 💡 다음 단계

Day 3-9 완료! 이제 고급 튜토리얼로 넘어갈 준비가 되었습니다:
- **Fused Softmax**: Day 7의 numerical stability + Reduction 연산
- **Layer Normalization**: Day 4의 Fused 연산 + Reduction
- **Fused Attention**: 모든 개념의 종합

화이팅! 🎉
