# Day 7: Sigmoid

## 📋 실습 목표
`1 / (1 + exp(-x))` - Numerical stability를 고려한 Sigmoid 구현

---

## 📚 실습 전 복습 사항

### 1. Day 1-6 복습
- [ ] 기본 커널 구조
- [ ] Element-wise 연산
- [ ] 조건부 연산

### 2. 지수 함수 이해
- [ ] `exp(x)` 함수의 특성
- [ ] 큰 x 값에서 `exp(x)`는 매우 큼 (오버플로우)
- [ ] 작은 x 값에서 `exp(-x)`는 매우 큼

### 3. 핵심 개념
```python
# Naive 구현 (문제 있음)
sigmoid_naive = 1.0 / (1.0 + tl.exp(-x))  # x가 크면 exp(-x) → 0, 괜찮음
# 하지만 x가 매우 작으면 exp(-x) → inf, 오버플로우!

# Numerical stable 구현
max_x = tl.maximum(x, 0.0)  # x가 양수면 그대로, 음수면 0
exp_neg = tl.exp(-(x - max_x))  # x - max_x는 항상 <= 0
sigmoid = 1.0 / (1.0 + exp_neg)
```

---

## 🎯 실습으로 배울 사항

### 1. Numerical Stability ⭐ 핵심!
- [ ] 오버플로우 방지 기법
- [ ] `exp(-x)` 대신 `exp(-(x - max))` 사용
- [ ] 수학적으로 동일하지만 수치적으로 안정적

### 2. Sigmoid 수학적 이해
- [ ] 출력 범위: (0, 1)
- [ ] S자 형태의 곡선
- [ ] `x = 0`일 때 `sigmoid(0) = 0.5`

### 3. 구현 방법
```python
# 방법 1: Numerical stable (권장)
max_x = tl.maximum(x, 0.0)
exp_neg = tl.exp(-(x - max_x))
sigmoid = 1.0 / (1.0 + exp_neg)

# 방법 2: 더 안정적 (추천)
# x > 0: 1 / (1 + exp(-x))
# x <= 0: exp(x) / (1 + exp(x))
sigmoid = tl.where(
    x > 0,
    1.0 / (1.0 + tl.exp(-x)),
    tl.exp(x) / (1.0 + tl.exp(x))
)
```

---

## ⚠️ 주의 사항

### 1. 오버플로우 방지 ⚠️ 매우 중요!
- **Naive 구현의 문제**:
  ```python
  # x가 매우 작으면 (예: -100)
  exp_neg = exp(-(-100)) = exp(100) → inf (오버플로우!)
  ```
- **안정적 구현**:
  ```python
  # x가 작아도 안전
  max_x = max(x, 0)  # x가 음수면 0
  exp_neg = exp(-(x - max_x))  # 항상 <= 1
  ```

### 2. 수치 정확도
- 매우 큰 x: `sigmoid(x) ≈ 1`
- 매우 작은 x: `sigmoid(x) ≈ 0`
- 경계값 처리 중요

### 3. `tl.exp()` 사용법
- `tl.exp(x)`는 x가 클수록 매우 큰 값
- 항상 numerical stability 고려

### 4. 디버깅 팁
```python
# 다양한 입력 테스트
test_cases = [
    torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], device='cuda'),
    torch.tensor([-100.0, 100.0], device='cuda'),  # 극단값
]

for x in test_cases:
    expected = torch.sigmoid(x)
    result = sigmoid_triton(x)
    assert torch.allclose(result, expected, rtol=1e-4)
    print(f"Input: {x.cpu()}, Output: {result.cpu()}")
```

---

## 📝 체크리스트

- [ ] Day 1-6 복습 완료
- [ ] Numerical stability 개념 이해 ⭐
- [ ] `tl.exp()` 사용법
- [ ] Sigmoid 구현 완료
- [ ] 다양한 입력 값 테스트 (음수, 0, 양수, 극단값)
- [ ] PyTorch `torch.sigmoid`와 비교
- [ ] 오버플로우 방지 확인

---

## 🔗 참고 자료

- Day 1-6 코드
- PyTorch: `torch.sigmoid()`
- 다음 Day: GeLU (복잡한 수학 함수)
- **Fused Softmax에서도 같은 numerical stability 기법 사용!** ⭐

---

## 💡 다음 단계

이 Day 7에서 배운 **Numerical Stability**는:
- **Fused Softmax에서 필수!** (exp(x - max) 패턴)
- Layer Normalization에서도 중요
- 모든 지수 함수 사용 시 고려 필요

반드시 이해하고 넘어가세요! ⭐
