# 바렌 플래토 직접 확정(A) + near-identity 초기화 완화 시험(B)

- 작성: 2026-06-30 06:08 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). 기존 코드 무수정(import/래핑). 단일 프로세스. 단정 금지(이 ansatz·옵티마이저·초기화 한정). 판정은 웹 AI.

## 1. 작업 목적
직전 n↑ 스케일링서 QCBM-고전 격차가 n↑일수록 확대 → "BP류 trainability 벽" 정황만 있고 gradient 미측정.
(A) n별 gradient 분산 직접 측정 → BP(지수감소) 확정. (B) near-identity(소각) 초기화로 큰 n 학습 개선 여부. 우위와 별개.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (초기화/gradient 접근)
- `qcbm_kl_variant.train_kl_one`(@37): init `w=rng.uniform(0,2π,shp)`(full-range, BP 취약), gradient `qml.grad(cost)`, cost=−Σq·log(p+ε),
  ansatz `E.make_circuit`=StronglyEntanglingLayers, params=L·n·3, Adam. **초기 파라미터 인자 없음** → 완화(B)용 init 변형을 새 파일에 작성.
- 측정: 랜덤 init R개에서 ∂C/∂θ₀(=qml.grad(cost)[0,0,0]) + canonical ∂⟨Z₀⟩/∂θ₀(별도 qnode). near-identity=N(0,σ=0.1).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: init/grad 접근, n-generic 재사용. 직전 n10 L6≈0.42(baseline 재현 대상).
- 추정: BP면 Var(grad) n↑서 지수감소; near-identity면 초기 gradient 보존→개선 가능(단 초기 p≈δ₀).
- 확인 불가: 정확 Grant identity-block(여기선 소각 near-identity 프록시), 다른 ansatz/옵티마이저.

## 5. (A) gradient 진단 — 랜덤 init R=50, probe=중간층 qubit0 RY(인덱스 (L//2,0,1)), L=6
| n | P | **Var(∂⟨Z₀⟩/∂θ) (canonical)** | mean\|∂Z₀\| | Var(∂C_KL/∂θ) | mean\|∂C_KL\| |
|---|---|---|---|---|---|
| 6 | 108 | **6.24e-3** | 0.0645 | 0.738 | 0.618 |
| 8 | 144 | **1.28e-3** | 0.0256 | 1.473 | 0.755 |
| 10 | 180 | **6.36e-4** | 0.0194 | 0.606 | 0.556 |
| 12 | 216 | 2.97e-34 (이상값) | 1.4e-17 | 3.772 | 1.110 |
- **canonical ⟨Z₀⟩ 분산이 n=6→8→10에서 단조 지수 감소**(6.2e-3 → 1.3e-3 → 6.4e-4, +2큐빗마다 ~2~5배↓) → **바렌 플래토 확정** ✅(이 ansatz·L=6 범위).
- **n=12 ⟨Z₀⟩는 2.97e-34=기계 0의 이상값**(수치 언더플로 또는 그 probe의 퇴화로 의심) → BP 곡선에서 제외, 사실로 표기. n=6~10만 신뢰.
- KL-cost gradient는 비단조(0.74→1.47→0.61→3.77) — 1/pₛ 스케일로 희소상태 비율에 휘둘려 BP probe 부적합. **⟨Z₀⟩가 올바른 척도**.
- (probe 주의) (0,0,0)=|0⟩에 처음 걸리는 RZ(위상)각은 확률 gradient 항등 0 → 부적합이라 중간층 RY 사용(초기 버그 수정).
- 깊이: n=8 Var(Z₀) L=3 1.54e-3 vs L=6 1.28e-3 (큰 차이 아님; R=50 노이즈 범위).

## 6. (B) near-identity 완화 — uniform vs N(0,σ=0.1), L=6, epochs=500, seeds=3 best
| n | uniform best (spread) | near-identity best (spread) | 개선? |
|---|---|---|---|
| 8 | **0.0051** (0.0015) | 0.0090 (0.0062) | 아니오(둘 다 바닥, 충분학습시 uniform도 OK) |
| 10 | **0.6431** (0.8177) | **0.3911** (0.3658) | **부분 개선**(0.64→0.39, 분산도 0.82→0.37↓) |
- **n=8**: 충분 학습(500ep)이면 uniform도 바닥(0.005) 도달 → smoke(120ep)의 "uniform 0.38 vs near-id 0.02, 17배"는 **과소학습 artifact**였음(정직히 정정).
- **n=10**: near-identity가 uniform보다 **개선(0.64→0.39)·seed 분산 감소(0.82→0.37)** → 초기화/BP가 한 원인임을 지지. **단 바닥(MLP~0.001)엔 여전히 미도달** = 부분 완화(해결 아님).

## 7. self-check
- 랜덤 init서 mean(grad)≈0(대칭) — randMean작음 True. Var(Z₀) n간 비증가 True.
- baseline 재현: uniform n10 L6 best=**0.643**(직전 스케일링 0.42와 같은 차수=바닥서 한참 멈; 정확값은 seed 분산 큼(0.64~1.46)으로 차이, 정성 일치).

## 8. 사실 관찰 (판정 아님, 단정 금지)
- **바렌 플래토 확정(A)**: gradient 분산이 n=6→10서 지수 감소 — 큰 n QCBM 학습 실패가 표현력이 아니라 **trainability(BP)** 임을 직접 측정으로 뒷받침.
- **완화는 부분적(B)**: near-identity 초기화가 n=10서 KL·분산을 줄였으나(0.64→0.39) **고전(MLP~1e-3) 수준엔 미도달** → BP가 한 원인이지만 **소각 init 단독으론 큰 n 경쟁력 회복 못 함**. n=8은 충분학습이면 init 무관(BP 영향 미미).
- ★ 모두 **이 ansatz(StronglyEntangling)·Adam·이 초기화 한정**. "양자 일반 BP" 또는 "해결/우위"로 단정하지 않음. BP 완화가 됐어도 **우위와는 별개**(여전히 고전 우세).
- 한계: n=10 seed 분산 큼(3 seed 소수), n=12 ⟨Z₀⟩ 이상값, near-identity는 정확 Grant identity-block 아닌 소각 프록시.

## 9. 생성 파일 / 실행 / tmux
신규: `tools/exp_barren_plateau.py`, `results2/barren_plateau/{barren_plateau.json,barren_plateau.png,run.log,smoke.log,atest.log}`, 본 로그.
```
python -m py_compile tools/exp_barren_plateau.py            # OK
python tools/exp_barren_plateau.py --smoke                 # (probe 버그 발견→중간층 RY로 수정)
tmux new -s hoc_bp -d 'python3 tools/exp_barren_plateau.py --ns-A 6,8,10,12 --ns-B 8,10 --R 50 --L 6 --epochs 500 --seeds 3 ...'
```
- **tmux 세션 hoc_bp**, 로그 `results2/barren_plateau/run.log`. (A) ~7분(n별 47~167s), (B) n8 2582s·n10 4109s. 총 ~2.5h. 자동 종료.
- 기존 코드 무수정(E.make_circuit/wshape, SY, OD import만; init 변형은 새 파일 train_kl_init). 합성만. dependency 무설치. 실패 없음(probe 버그는 새 파일 내 수정).

## 10. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 11. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] init(uniform[0,2π])/grad(qml.grad) 접근·wshape 확인·기록.
- [x] (A) n=6,8,10,12 Var(grad) 측정, ⟨Z₀⟩ 지수감소(n6~10) 확정·json/png. n12 이상값 표기.
- [x] (B) n=10,8 uniform vs near-identity KL·분산 비교(n10 부분개선, n8 동급).
- [x] self-check(랜덤 mean≈0, baseline n10 같은 차수 재현).
- [x] 해석 "이 ansatz·옵티마이저·초기화 한정·우위와 별개". 기존 무변경·합성만·무설치.

**남은 불확실성**
- n=10 seed 분산 큼(3 seed) → 더 많은 seed면 값 안정. n=12 ⟨Z₀⟩ 이상값(수치) 원인 미규명.
- near-identity는 소각 프록시 → 정확 identity-block/layerwise면 완화 더 클 수 있음(미검).
- BP가 "한 원인"이나, 큰 n 미도달이 BP 외 요인(국소최적, 표현력 경계)도 섞였을 가능성(완전분리 안 됨).

**다음 추천 단계**
1. **더 강한 완화 시험**: 정확 identity-block(Grant)·layerwise 학습으로 n=10이 바닥 도달하는지 — "init 단독 부분완화" 넘어서는지.
2. n=10 seed↑(예: 10)로 분산 줄여 near-identity 개선폭 안정 측정. n=12 ⟨Z₀⟩ 이상값 원인(probe/precision) 점검.
3. (해석 입력) 웹 AI에 "BP 직접 확정(지수감소), near-identity 부분완화하나 바닥 미도달 → 큰 n 한계는 trainability(BP)가 주원인이나 init 단독으론 미해결, 우위와 별개" 전달.
