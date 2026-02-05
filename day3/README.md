# Day 3: Element-wise Sub

## 📋 실습 목표
`y = a - b` - 두 벡터/행렬의 element-wise 뺄셈 구현

---

## 📚 실습 전 복습 사항

### 1. Day 1 (Vector Add) 복습
- [ ] `tl.program_id(axis=0)` 사용법
- [ ] `tl.arange(0, BLOCK_SIZE)` 벡터화된 인덱스 생성
- [ ] `mask = offsets < N` 범위 체크
- [ ] `tl.load(ptr + offsets, mask=mask)` 안전한 메모리 로드
- [ ] `tl.store(ptr + offsets, data, mask=mask)` 안전한 메모리 저장

### 2. Day 2 (Vector Scale) 복습
- [ ] Scalar broadcasting 개념
- [ ] `y = alpha * x` 패턴

### 3. 핵심 개념
```python
# 기본 구조
@triton.jit
def kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a - b  # 뺄셈만 다름!
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 🎯 실습으로 배울 사항

### 1. Element-wise 연산 패턴
- [ ] 두 텐서의 같은 위치 요소끼리 연산
- [ ] Day 1의 덧셈과 거의 동일한 구조
- [ ] 연산자만 `+` → `-`로 변경

### 2. 다양한 크기 처리
- [ ] 작은 크기 (N < BLOCK_SIZE) 처리
- [ ] 큰 크기 (N >> BLOCK_SIZE) 처리
- [ ] Mask를 통한 경계 처리

### 3. 그리드 설정
```python
grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
```

---

## ⚠️ 주의 사항

### 1. 메모리 접근
- **반드시 mask 사용**: 범위 밖 접근 방지
- **other 파라미터**: `tl.load(ptr + offsets, mask=mask, other=0.0)`

### 2. 데이터 타입
- 입력 텐서 `a`, `b`의 dtype이 같아야 함
- 출력 텐서도 동일한 dtype 사용

### 3. 텐서 크기
- `a`와 `b`의 크기가 같아야 함
- Shape 검증 필요: `assert a.shape == b.shape`

### 4. 디버깅 팁
```python
# 작은 크기로 시작
N = 8
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')

# PyTorch와 비교
expected = a - b
result = elementwise_sub_triton(a, b)
assert torch.allclose(result, expected)
```

---

## 📝 체크리스트

- [ ] Day 1-2 코드 복습 완료
- [ ] Element-wise 뺄셈 구현
- [ ] 다양한 크기 테스트 (8, 64, 1024, 10000)
- [ ] PyTorch 결과와 비교 검증
- [ ] 에러 처리 (shape 불일치 등)

---

## 🔗 참고 자료

- Day 1 코드: `../day1/vector_addition.py`
- Day 2 코드: `../day2/vector_scale.py`
- Benchmark 참고: `/home/members/donghyun/workspace/benchmark/gpu-100days/`
