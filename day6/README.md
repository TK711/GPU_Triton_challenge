# Day 6: Leaky ReLU

## 📋 실습 목표
`x < 0 ? alpha * x : x` - Leaky ReLU 구현

---

## 📚 실습 전 복습 사항

### 1. Day 5 (ReLU) 복습
- [ ] `tl.maximum(x, 0.0)` 사용법
- [ ] `tl.where(condition, x, y)` 사용법
- [ ] 조건부 연산 패턴

### 2. Scalar Broadcasting (Day 2, 4)
- [ ] Scalar 값과 텐서의 곱셈
- [ ] `alpha * x` 패턴

### 3. 핵심 개념
```python
# 기본 구조
@triton.jit
def kernel(x_ptr, out_ptr, alpha, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # Leaky ReLU: x < 0 ? alpha * x : x
    out = tl.where(x < 0, alpha * x, x)
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 🎯 실습으로 배울 사항

### 1. 조건부 연산 확장
- [ ] `tl.where`를 활용한 복잡한 조건
- [ ] Day 5의 ReLU 확장
- [ ] 음수 영역에서 기울기 유지

### 2. Leaky ReLU 수학적 이해
- [ ] ReLU: `max(0, x)` → 음수 영역에서 기울기 0
- [ ] Leaky ReLU: `x < 0 ? alpha * x : x` → 음수 영역에서 작은 기울기 유지
- [ ] `alpha`는 보통 0.01 ~ 0.2

### 3. 구현 방법 비교
```python
# 방법 1: tl.where 사용 (권장)
out = tl.where(x < 0, alpha * x, x)

# 방법 2: 조건 분리 (비효율적)
negative = x < 0
out = negative * (alpha * x) + (~negative) * x
```

---

## ⚠️ 주의 사항

### 1. Alpha 값
- 일반적으로 `0.01` ~ `0.2` 범위
- `alpha = 0`이면 ReLU와 동일
- `alpha = 1`이면 항등 함수 (의미 없음)

### 2. 조건 체크
- `x < 0` 조건이 정확히 작동하는지 확인
- `x = 0`일 때는 `x` 선택 (양수로 취급)

### 3. 수치 안정성
- 매우 작은 `alpha` 값 주의
- 오버플로우/언더플로우 가능성

### 4. 디버깅 팁
```python
# 다양한 입력 테스트
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
alpha = 0.01

# PyTorch 결과
expected = torch.where(x < 0, alpha * x, x)
# [-0.02, -0.01, 0.0, 1.0, 2.0]

result = leaky_relu_triton(x, alpha)
assert torch.allclose(result, expected)
```

---

## 📝 체크리스트

- [ ] Day 5 (ReLU) 복습 완료
- [ ] `tl.where` 조건부 연산 이해
- [ ] Leaky ReLU 구현 완료
- [ ] 다양한 alpha 값 테스트 (0.01, 0.1, 0.2)
- [ ] 양수, 음수, 0 값 테스트
- [ ] PyTorch와 비교 검증

---

## 🔗 참고 자료

- Day 5 코드: `../day5/`
- PyTorch: `torch.nn.LeakyReLU(alpha)`
- 다음 Day: Sigmoid (Numerical stability)
