# Day 5: ReLU

## 📋 실습 목표
`max(0, x)` - Rectified Linear Unit 구현

---

## 📚 실습 전 복습 사항

### 1. Day 1-4 복습
- [ ] 기본 커널 구조
- [ ] `tl.load`, `tl.store` 사용법
- [ ] Element-wise 연산 패턴

### 2. 조건부 연산 개념
- [ ] Python의 `if` 문과는 다름
- [ ] Triton은 벡터화된 조건부 연산 사용

### 3. 핵심 개념
```python
# 기본 구조
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # ReLU: max(0, x)
    out = tl.maximum(x, 0.0)  # 또는 tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 🎯 실습으로 배울 사항

### 1. 조건부 연산 방법
- [ ] `tl.maximum(a, b)`: 두 값 중 큰 값 선택
- [ ] `tl.where(condition, x, y)`: 조건에 따라 선택
- [ ] 벡터화된 조건부 연산

### 2. ReLU 구현 방법
```python
# 방법 1: tl.maximum 사용
out = tl.maximum(x, 0.0)

# 방법 2: tl.where 사용
out = tl.where(x > 0, x, 0.0)

# 방법 3: 수동 구현 (비효율적)
out = x * (x > 0)  # x > 0이면 x, 아니면 0
```

### 3. 성능 비교
- `tl.maximum`이 일반적으로 더 빠름
- `tl.where`는 더 유연하지만 약간 느릴 수 있음

---

## ⚠️ 주의 사항

### 1. 데이터 타입
- 입력이 음수일 수 있음
- 출력은 항상 >= 0
- dtype은 그대로 유지 (float32 → float32)

### 2. 0의 처리
- `x = 0`일 때 결과는 `0`
- `max(0, 0) = 0` 정확히 처리

### 3. 벡터화
- `tl.maximum`은 벡터 전체에 대해 동시에 연산
- 각 요소가 독립적으로 처리됨

### 4. 디버깅 팁
```python
# 다양한 입력 테스트
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
expected = torch.relu(x)  # [0, 0, 0, 1, 2]
result = relu_triton(x)
assert torch.allclose(result, expected)
```

---

## 📝 체크리스트

- [ ] Day 1-4 복습 완료
- [ ] `tl.maximum` 사용법 이해
- [ ] `tl.where` 사용법 이해
- [ ] ReLU 구현 완료
- [ ] 양수, 음수, 0 값 테스트
- [ ] PyTorch `torch.relu`와 비교

---

## 🔗 참고 자료

- Day 1-4 코드
- PyTorch: `torch.relu()`
- 다음 Day: Leaky ReLU (조건부 연산 확장)
