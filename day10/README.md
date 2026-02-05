# Day 10: Tanh

## 📋 실습 목표
Hyperbolic Tangent 함수 구현 - Sigmoid를 활용한 수치 안정적 구현

---

## 📚 실습 전 복습 사항

### 1. Day 7 (Sigmoid) 복습 ⭐ 필수!
- [ ] Sigmoid 구현
- [ ] `tl.sigmoid()` 사용법
- [ ] Numerical stability

### 2. 수학적 관계
```python
# Tanh와 Sigmoid의 관계
# tanh(x) = 2 * sigmoid(2x) - 1
# 
# 또는 직접 계산:
# tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
#         = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

---

## 🎯 실습으로 배울 사항

### 1. Tanh 구현 방법

#### 방법 1: Sigmoid 활용 (권장)
```python
@triton.jit
def tanh_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # tanh(x) = 2 * sigmoid(2x) - 1
    out = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

#### 방법 2: Exp 직접 계산
```python
# tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
exp_2x = tl.exp(2.0 * x)
out = (exp_2x - 1.0) / (exp_2x + 1.0)
```

### 2. Tanh 특성
- 출력 범위: (-1, 1)
- `tanh(0) = 0` (원점 대칭)
- Sigmoid보다 기울기가 큼
- `tanh(x) = 2 * sigmoid(2x) - 1`

---

## ⚠️ 주의 사항

### 1. 수치 안정성
- 큰 x 값: `exp(2x)` 오버플로우 가능
- `sigmoid` 활용하면 이미 안정적

### 2. Sigmoid 재사용
- Day 7 코드 재활용
- `tl.sigmoid()` 사용 가능

### 3. 검증
```python
# PyTorch 비교
torch_out = torch.tanh(x)
is_correct = torch.allclose(triton_out, torch_out, atol=1e-5)
```

---

## 📝 체크리스트

- [ ] Day 7 (Sigmoid) 복습 완료
- [ ] Tanh 수학적 이해
- [ ] `tanh(x) = 2*sigmoid(2x) - 1` 공식 이해
- [ ] Tanh 구현 완료
- [ ] PyTorch와 비교 검증
- [ ] 다양한 입력 테스트

---

## 💡 참고

Tanh는 RNN, LSTM에서 자주 사용되는 활성화 함수입니다.
Sigmoid와의 관계를 이해하면 쉽게 구현할 수 있습니다.
