# 깊이-표현력 곡선 (n=6, L 스윕) — QCBM이 동일 P로 고전 동급 되는 교차점 정량화 (n↑ 기준선)

- 작성: 2026-06-29 10:42 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). n=6 고정·한 점 집중. 기존 코드 무수정(import/재사용). 단일 프로세스. 판정은 웹 AI.

## 1. 작업 목적
직전 확정: QCBM-KL 저차 0.197은 L=3/P=54 표현력 포화(유한학습 아님), L=6/P=108은 0.0002(바닥). 이 "파라미터당 표현
효율" 차를 깊이 L 스윕(2~8)으로 곡선화 → P별 QCBM 포화 KL vs 동일 P 고전(MLP/NMF). 교차점(QCBM이 같은/더 적은 P로
고전 동급/우위) 존재 여부 정량화. **n=6 기준선**(이후 n=8,10 비교 토대). 양자 우위 단정 금지.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (train_kl_one · 고전 P맞춤 · L별 P)
- `qcbm_kl_variant.train_kl_one(q,nq,layers,epochs,lr,seed)` → (p_final,CE,loss_hist). params=L·nq·3.
- 고전 P맞춤(직전 EXP): NMF rank=round(P/(2·side)), MLP hidden=round(P/(n+2)); side=8.
- L별 P (n=6, wshape 검증):
  | L | P | NMF rank(params) | MLP hidden(params) |
  |---|---|---|---|
  | 2 | 36 | 2 (32) | 4 (33) |
  | 3 | 54 | 3 (48) | 7 (57) |
  | 4 | 72 | 4 (64) | 9 (73) |
  | 5 | 90 | 6 (96) | 11 (89) |
  | 6 | 108 | 7 (112) | 14 (113) |
  | 8 | 144 | 9 (144) | 18 (145) |
- 재현 대상: L=3 QCBM 0.195, L=6 0.0002, 샘플바닥 0.0009. 평가 KL(p_true‖model).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: train_kl_one 인자화·곡선, 고전 P맞춤 공식, L별 P. 학습량 고정(800ep/5restart)로 포화 보장.
- 추정: L↑→P↑→QCBM KL 단조감소(L3→6 확인). 고전도 P↑로 감소. 교차/평행/열위가 결론.
- 확인 불가: n=8,10(다음 작업), 다른 ansatz.

## 5. 설계
n=6, t=0 고정(seed0 disjoint, N=80000, 샘플 seed12345). L∈[2,3,4,5,6,8]: QCBM-KL(800ep,5restart,best) + 동일 P NMF/MLP/marginal.
P vs KL 곡선. 교차점=QCBM_KL≤MLP_KL 최소 P. 고랭크 random t=0.6 핵심 L([3,6,8]) 곡선. self-check: L↑→KL 비증가, L3/L6 재현.

## 6. 깊이 곡선 (n=6, t=0 저차) — KL(p_true‖model), epochs=700 restarts=4
| L | P | QCBM-KL | MLP(같은 P) | NMF(같은 P) | marginal |
|---|---|---|---|---|---|
| 2 | 36 | 0.0534 | 0.0035(p33) | 0.0052(r2) | 2.415 |
| 3 | 54 | **0.1966** | 0.0009(p57) | 0.0051(r3) | 2.415 |
| 4 | 72 | **0.0004** | 0.0004(p73) | 0.0051(r4) | 2.415 |
| 5 | 90 | 0.0004 | 0.0003(p89) | 0.0051(r6) | 2.415 |
| 6 | 108 | 0.0002 | 0.0002(p113) | 0.0051(r7) | 2.415 |
| 8 | 144 | 0.0003 | 0.0005(p145) | 0.0050(r9) | 2.415 |
- 샘플바닥 0.0009. **QCBM 곡선 비단조**: L=3(0.197)이 L=2(0.053)·L=4(0.0004)보다 나쁨 — 파라미터 더 많은데 적합 더 나쁨 →
  표현력 아닌 **L=3 최적화 landscape의 나쁜 attractor**(4 restart로도 못 벗어남). L≥4부터 QCBM이 바닥(~3e-4) 도달.

## 7. P vs KL 교차점 분석 (저차)
- **L=2,3에선 QCBM > MLP**(MLP가 동일 P로 훨씬 낮음). 그러나 **L≥4(P≥72)부터 QCBM ≈ MLP**(둘 다 샘플바닥 ~3e-4).
- L=6: QCBM 0.0002 = MLP 0.0002(동률). **L=8: QCBM 0.0003 < MLP 0.0005** → 스크립트 교차점 **P=144**(QCBM≤MLP 최초).
- 즉 동일 P에서 QCBM이 고전 MLP **동급에 도달**(L≥4), 깊은 끝(L=8)에선 미세 우위. NMF는 전 구간 0.005에서 평탄(저랭크 한계).

## 8. 고랭크 random t=0.6 (share 0.178, rank99=4)
| L | P | QCBM-KL | MLP | NMF |
|---|---|---|---|---|
| 3 | 54 | 0.2773 | 0.0029 | **3.0530** |
| 6 | 108 | 0.0078 | 0.0005 | **0.4848** |
| 8 | 144 | **0.0003** | 0.0050 | **0.2563** |
- NMF 전 구간 붕괴(rank99=4 > NMF rank, 깊어져도 0.26에서 못 회복). QCBM은 깊이↑로 0.277→0.0003 급감.
- **L=8: QCBM 0.0003 < MLP 0.0050 < NMF 0.256** → QCBM이 둘 다 넘음(교차점 P=144). (MLP L=8 hidden18이 다소 미수렴/불운일 수 있음 — 주의.)

## 9. self-check
- L↑→QCBM KL 단조 = **False**(L=3 anomaly로). 학습 자체는 정상(L=2/4 양호, 발산 없음).
- 직전 재현: L=3 QCBM = **0.1966**(직전 포화 0.195 일치 ✓), L=6 = **0.0002**(직전 0.0002 일치 ✓). 실험 일관성 확인.

## 10. 사실 관찰 (판정 아님, n=6 한정, 단정 금지)
- **QCBM은 깊이를 키우면 동일 P 고전(MLP)과 동급에 도달**(L≥4 저차, 둘 다 샘플바닥). 깊은 끝(L=8)에선 QCBM이 MLP를 미세히
  넘고, **고랭크에선 L=8 QCBM이 MLP·NMF 둘 다 넘음**(교차점 P=144). 즉 "QCBM이 항상 더 많은 P 필요"는 아님.
- 단 **QCBM 곡선은 비단조**(L=3 최적화 landscape 나쁨) — 깊이당 효율이 매끄럽지 않고 최적화 운에 민감.
- NMF(저랭크)는 고랭크에서 깊어져도 회복 불가(구조적 한계), 반면 QCBM·MLP는 회복.
- ★ n=6 단일 분포·단일 seed 기준선. 교차점이 양자 "우위"인지는 n↑(8,10)에서 교차점이 더 낮은 P로/더 뚜렷이 이동하는지
  봐야 판정 가능(이 작업 범위 밖). 단정 금지. MLP L=8 미세 결과(0.005)는 재확인 필요.

## 11. 생성 파일 / 실행 / tmux
신규: `tools/exp_depth_expressivity.py`, `results2/depth_expressivity/{depth_n6.json,depth_n6.png,run.log,smoke.log}`, 본 로그.
```
python -m py_compile tools/exp_depth_expressivity.py     # OK
python tools/exp_depth_expressivity.py --smoke           # 비단조·교차점 확인(L=2,3,6)
tmux new -s hoc_depth -d 'python3 tools/exp_depth_expressivity.py --epochs 700 --restarts 4 2>&1 | tee results2/depth_expressivity/run.log'
```
- **tmux 세션 hoc_depth**, 로그 `results2/depth_expressivity/run.log`. 저차 L=2~8 + 고랭크 L=3,6,8. 단일 프로세스. 완료 후 자동 종료(약 3시간, L↑일수록 느림).
- 기존 코드 무수정(qcbm_kl_variant/exp_synth_*/experiment2 import만). 합성만(실데이터 미사용). dependency 무설치. 실패 없음.
- (참고) 중간에 nohup 백그라운드 시도는 sandbox에서 미유지 → tmux로 수행(검증된 방식). 감시 워처 1회 종료됐으나 실험은 tmux서 완주.

## 12. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 13. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] train_kl_one·고전 P맞춤·L별 P 확인·기록(§3).
- [x] L=2~8 QCBM 포화 KL + 동일 P MLP/NMF 표(§6,§8), json/png.
- [x] 교차점 분석: L≥4 동급, L=8 미세우위, 교차점 P=144(§7).
- [x] 고랭크 곡선(§8). self-check(L3 0.197/L6 0.0002 재현, 비단조 사실 기록)(§9).
- [x] n=6 한정·n↑ 기준선 명시·양자우위 단정 없음. 기존 무변경·합성만·무설치.

**남은 불확실성**
- QCBM 비단조(L=3 anomaly): 최적화 landscape 문제 — 다른 init/restart수/옵티마이저면 완화될지 미검.
- MLP L=8(0.0050) 다소 높음(불운/미수렴 가능) → 고랭크 L=8 "QCBM>MLP" 결론은 재확인 필요.
- 단일 분포·seed. n↑ 미수행(교차점 이동 방향이 핵심인데 이건 다음 작업).

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "n=6에서 QCBM은 깊이↑로 동일 P MLP 동급 도달(L≥4), L=8 고랭크선 MLP·NMF 둘 다 넘음; 단 비단조(L=3 landscape).
   교차점 양자우위 여부는 n↑ 필요." 정리.
2. **n=8,10에서 동일 깊이 곡선** → 교차점이 더 낮은 P로/더 뚜렷이 이동하는지(양자 우위 점근성)가 핵심 다음 실험.
3. L=3 anomaly 원인(다른 seed·restart수·init 분산)·MLP L=8 재확인으로 곡선 노이즈 제거.
