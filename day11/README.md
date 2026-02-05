# Day 11: Boundary Handling

## 📋 실습 목표
비정렬 크기 처리 - `N != block_size * K`인 경우 안전한 메모리 접근

---

## 📚 실습 전 복습 사항

### 1. Day 1-10 복습
- [ ] `mask` 사용법
- [ ] `offsets < N` 패턴
- [ ] 경계 체크의 중요성

### 2. 문제 상황
```python
N = 1000
BLOCK_SIZE = 128

# 1000 / 128 = 7.8125
# → 7개 블록: 896개 원소 처리
# → 104개 원소 남음!
```

---

## 🎯 실습으로 배울 사항

### 1. Mask의 중요성

```python
# ❌ Mask 없이 (위험!)
a = tl.load(a_ptr + offsets)  # 범위 초과 시 undefined behavior

# ✅ Mask 사용 (안전)
mask = offsets < N
a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
```

### 2. Grid 계산

```python
# 올림 나눗셈 필수!
grid = (triton.cdiv(N, BLOCK_SIZE),)  # (N + BLOCK_SIZE - 1) // BLOCK_SIZE
```

### 3. 경계 처리 패턴

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 핵심: 경계 체크
    mask = offsets < N
    
    # mask로 안전하게 로드
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 연산
    out = x * 2.0
    
    # mask로 안전하게 저장
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## ⚠️ 주의 사항

### 1. Mask 미사용 시 문제
- Out-of-bounds 메모리 접근
- Segmentation fault 가능
- 예측 불가능한 결과

### 2. `other` 파라미터
```python
# mask=False인 곳에 대체값 지정
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 3. 테스트 케이스
```python
# 다양한 N 테스트
test_sizes = [
    127,   # BLOCK_SIZE보다 작음
    128,   # 정확히 맞음
    129,   # 1개 초과
    1000,  # 여러 블록
    1023,  # 거의 8블록
]
```

---

## 📝 실습 과제

다양한 크기로 테스트:
1. N < BLOCK_SIZE
2. N = BLOCK_SIZE
3. N = BLOCK_SIZE * k
4. N = BLOCK_SIZE * k + 1
5. 큰 N (예: 100000)

---

## 💡 핵심 요약

**항상 mask 사용!**
```python
mask = offsets < N
tl.load(..., mask=mask)
tl.store(..., mask=mask)
```
