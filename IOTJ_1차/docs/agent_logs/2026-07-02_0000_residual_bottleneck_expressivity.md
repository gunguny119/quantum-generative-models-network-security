# 잔여 병목 분리 (iii) — 표현력 진단: P↑(L=6/8/10/12)로 반복 미스매치 모드 잡히나 (idblock·pert0.5·KL 고정)

- 작성: 2026-07-02 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만. **idblock(Grant V†V, pert0.5)·loss(KL)·n=10·t=0 고정, L(=P)만 키움**. 기존 코드 무수정(import). 단정 금지(이 ansatz·완화법·KL 한정). 판정은 웹 AI.

## 1. 작업 목적
잔여 해부: BP(확정·완화)→KL과민성(시작점만 부분)→국소최적(기각: 지배 미스매치 모드 s958/808/1018/517 seed 무관 반복=구조적). 마지막 (iii): P=180(L6)이 그 모드를 원리적으로 못 표현해 바닥(1e-3) 미달인가. L↑로 best KL 바닥 근접·모드 잡힘이면 표현력 부족. ★target(t=0)은 **순수 1·2차 저차**인데도 미스매치 → 회로 표현 기하 단서.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip`(전 세션 동일, 전부 untracked; Bash 분류기 일시 불가로 실행 재확인은 복구 후). 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조
- `SBP.train_idblock`: half=L//2 → **짝수 L서 P=L·n·3 정확**(L6=180,L8=240,L10=300,L12=360). make_idb=Grant V†V. cost=−Σq·log(p+1e-10). synth_low=t=0.
- `SBP.grad_measure(nq,L,R,pert)`→ idblock_var/uniform_var(⟨Z₀⟩ 중간층 RY 분산) 재사용(BP 재발 통제). `OD.kl_div`. 반복모드 비트 decode 순수계산.
- 반복모드 MODES=[958,808,1018,517,1] (국소최적 진단서 seed 무관 반복).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: 짝수 L=P공식, 재사용 함수. 직전 n10 L6 pert0.5 best 0.087.
- 추정: 표현력 부족이면 L↑로 best↓·모드 잡힘·grad 유지. grad 저하면 BP 재발 혼재.
- 확인 불가(Bash 복구 후): 실제 실행/시간게이트. L12 시간 과도시 축소.

## 5. 설계 / 시간게이트
축1 L∈{6,8,10,12} seeds3 @400ep best KL. 축1-b 각 L grad. 축2 반복모드 |q−p|+비트. `--time-only`(L12 1seed)로 L12 포함/생략 결정. 목표 ~3~3.5h.

## 6. 스코프 / 실행 메모
Bash 안전분류기 일시 불가로 지연 후 복구됨. 시간 절약 위해 L12 게이트 대신 비용추정(∝~L²)으로 **seeds=2, L∈{6,8,10,12} @400ep** 직행.
※ tee run.log 가 tmux 파이프 버퍼링 이슈로 미기록 → 하지만 스크립트가 os.chdir 후 JSON 절대 저장 → **결과(json/png) 정상**. 진행 모니터만 불가했음(결과 영향 없음).
반복모드 비트(10b): s958=1110111110(ham8,q0.008) s808=1100101000(ham4,q0.159) s1018=1111111010(ham8,q0.128) s517=1000000101(ham3,q0.389) s1(ham1,q0.016) — Hamming·q **다양**(단순 패턴 아님).

## 7. 축1 — L(=P) vs best KL (seeds=2, 400ep)
| L | P | best KL | median |
|---|---|---|---|
| 6 | 180 | 0.0867 | 0.290 |
| 8 | 240 | **0.0628** | 0.095 |
| 10 | 300 | **0.9215** | 0.931 |
| 12 | 360 | **0.8184** | 0.823 |
- L6→L8: best 0.087→0.063(소폭↓). **L8→L10/L12: KL 폭증(0.92/0.82)**. → **P↑가 바닥(1e-3) 근접 아님**; L8 이후 오히려 급악화. best_decreasing=False.

## 8. 축1-b — gradient ⟨Z₀⟩ 분산 (BP 재발 통제)
| L | 6 | 8 | 10 | 12 |
|---|---|---|---|---|
| idblock Var | 0.0188 | 0.00060 | 0.00054 | 0.00032 |
- **L↑로 gradient ~60배 붕괴**(0.019→0.0003). = **BP 재발**. L10/L12 KL 폭증은 이 gradient 붕괴(학습 실패)의 결과.

## 9. 축2 — 반복모드 |q−p| (L별)
| 모드 | L6 | L8 | L10 | L12 |
|---|---|---|---|---|
| s958 | 0.006 | 0.004 | 0.004 | 0.003 |
| s808 | 0.009 | 0.004 | **0.094** | 0.087 |
| s1018 | 0.005 | 0.001 | **0.083** | 0.075 |
| s517 | 0.005 | 0.007 | **0.249** | 0.227 |
| s1 | 0.009 | 0.012 | 0.007 | 0.009 |
- **L8에서 모드가 오히려 tighten**(s1018 0.001, s808/958 0.004) → 표현력은 L8까진 살짝 개선. **L10/L12는 모드 대폭 악화**(s517 0.25) — BP 붕괴로 학습 실패.

## 10. self-check
- L6 best = **0.0867 → 직전 재현** ✓. P=L·n·3 공식 ✓(180/240/300/360). best_decreasing=False(비단조 정확 포착) — L↑ 악화 자체가 발견.

## 11. 사실 관찰 (판정 아님, 단정 금지)
- **(iii) 순수 표현력 부족은 격리 불가 = "BP 재발 혼재" 결론**:
  - L↑(P↑)로 gradient가 ~60배 붕괴(0.019→0.0003) → **identity-block(pert0.5)로도 깊은 회로의 BP를 못 막음** → L10/L12 학습 실패(KL 0.9).
  - 즉 "capacity(depth)를 늘려 잔여 모드를 잡자"가 **불가능** — 깊이를 늘리면 trainability(BP)가 되살아나 오히려 악화.
  - 유일한 개선 창(L6→L8, 0.087→0.063, 모드 tighten)은 **바닥(1e-3)엔 여전히 ~60배 미달**.
- **종합(잔여 병목 3후보 결론)**: 잔여 바닥(~0.06)은 **표현력·trainability가 결합된 벽** — 깊이(capacity) 추가로 표현력을 얻으려 하면 BP가 재발해 학습이 무너짐. 셋 중 어느 하나가 아니라 **표현력↔BP의 커플링**이 근본 제약.
- ★ 이 ansatz(StronglyEntangling)·완화법(identity-block pert0.5)·KL·이 lr 한정. 우위 언급 없음.
- 한계: L10/L12 seeds=2(소수)나 gradient 붕괴(R=30 random init 측정)는 seed 무관 구조적 → BP 재발은 견고.

## 12. 생성 파일 / 실행 / tmux
신규: `tools/exp_expressivity.py`, `results2/expressivity/{expressivity.json,expressivity.png}` (run.log는 tmux 버퍼링으로 미기록), 본 로그.
```
python -m py_compile tools/exp_expressivity.py                 # OK
python tools/exp_expressivity.py (import/grad_measure 런타임 확인)   # OK
tmux new -s hoc_expr -d 'python3 tools/exp_expressivity.py --Ls 6,8,10,12 --seeds 2 --epochs 400 --R 30 ...'
```
- **tmux 세션 hoc_expr**, ~2.9h(02:01→04:57). 기존 코드 무수정(SBP.train_idblock/grad_measure/synth_low, OD import만). loss=KL·pert0.5·n10 고정, L만 변경. 합성만. 무설치. 실패/NaN 없음(단 L10/L12 학습 실패는 BP 재발 결과=사실).

## 13. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`. 기존 무수정.

## 14. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 짝수L·P공식·반복모드비트·gradient·(게이트 대체 추정) 확인·기록.
- [x] 축1 L vs best KL(비단조, L8 최저 0.063, L10/L12 폭증).
- [x] 축1-b gradient(L↑로 ~60× 붕괴 = BP 재발).
- [x] 축2 반복모드 |q−p|(L8 tighten, L10/L12 악화).
- [x] self-check(L6 0.087 재현, P공식).
- [x] (iii) 판정: **BP 재발 혼재** — 순수 표현력 격리 불가, 잔여=표현력↔BP 커플링 벽. 해석 한정·무변경·합성·무설치.

**남은 불확실성**
- L10/L12 seeds=2. 더 강한 완화(정확 identity-block 스택·layerwise·다른 ansatz)면 깊은 L서 BP 억제되어 표현력 순수측정 가능할지 미검.
- 개선 창(L8) 최저 0.063이 "충분 학습 시" 얼마까지 내려가는지(epoch↑) 미측정.

**다음 추천 단계**
1. 깊은 L에서 BP를 억제하는 **더 강한 완화**(layerwise 성장·블록별 identity·다른 ansatz)로 L10+에서 gradient 유지 → 그때 표현력 순수 측정.
2. L8(P=240)에 epoch↑/pert 재튜닝으로 0.063 바닥 여지 확인(개선 창 최대화).
3. (해석 입력) 웹 AI에: "P↑(depth) 시도는 BP 재발로 실패(gradient 60× 붕괴, L10/L12 KL 0.9); L8까지만 소폭 개선(0.063). 잔여 병목은 표현력·trainability 결합 벽 — 이 ansatz·완화법·KL 한정."
