# 합성 고차-상관 분포에서 양자(QCBM) vs 고전(NMF/MLP/marginal) 통제 비교 — "교차점" 탐색

- 작성: 2026-06-29 06:04 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성 데이터만(실데이터/체크포인트 미사용). 긴 학습 없음. 기존 코드 import/재사용. 단일 프로세스. 판정은 웹 AI.

## 1. 작업 목적
고차 비중을 0→100% 조절하는 합성 분포에서 동일 파라미터 예산의 QCBM vs 고전(NMF/MLP/marginal) 적합도를 비교해,
"고전이 무너지고 양자가 버티는 교차점"의 존재를 통제 실험으로 검증(A: 양자유리 분포 실재 vs B: 회로/학습 한계 구별).

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. tmux=/usr/bin/tmux. 보존: 모든 기존 파일(읽기/import).

## 3. 확인한 QCBM 구조 / maxent / baseline / 패키지
- QCBM(experiment2.py): ansatz `StronglyEntanglingLayers`(@145), **loss=MMD²**(Hamming-RBF 멀티시그마 K):
  `cost=p·Kp−2p·Kq+q·Kq`(@188), p=qml.probs(exact), 수동 Adam(@197), **params=L·nq·3**. **타깃 분포 벡터 q 직접 입력**(@176).
  재사용: `E._init(q,K,nq,L,epochs,lr,log_every,tag)`+`E.train_one(seed)`, `E.build_kernel`.
- maxent: `diagnose_order_decomposition.maxent_upto/kl_div/bit_matrix`(≥3차 share 사후측정).
- 패키지: pennylane/scipy/numpy/matplotlib O, **torch·sklearn X** → NMF/MLP numpy 자체구현, 무설치.

## 4. 설계 요약
- 합성: log p(s)=α·(1·2차 랜덤항)+β·(k≥3 parity항), softmax. β 스윕 → ≥3차 share(=KL(p‖p₂)/KL(p‖p₁)) 사후라벨.
- 모델 예산 P=L·nq·3 맞춤: NMF rank r≈P/(2·side), MLP hidden h≈P/(n+2), marginal=n(하한).
- 각 β: p_true→N샘플→q_emp→4모델 적합→KL(p_true‖model)/TV. 교차점=QCBM_KL<NMF_KL 지점.

## 5. 확실한 사실 / 추정 / 확인 불가
- 확실(코드/측정): QCBM loss=MMD²·ansatz=StronglyEntangling·params=L·nq·3, 타깃 분포 직접 입력. 합성 self-check. 4모델 KL.
- torch·sklearn 부재 확인 → NMF/MLP numpy 자체구현(무설치).
- 확인 불가 없음. 단 n=6 소규모·단일 seed·QCBM은 MMD²로 학습/KL로 평가(목적 불일치)라는 해석상 한계(아래 11/14).

## 6. 합성 생성기 + self-check (PASS)
log p(s)=(1−t)·(랜덤 1·2차)+t·HIGH_SCALE·(k≥3 parity), softmax. ≥3차 share=KL(p‖p₂)/KL(p‖p₁) 사후측정.
- self-check(disjoint, n=6): t=0→share 0.000, 0.25→0.45, 0.5→0.73, 0.75→0.90, **1.0→1.000** (단조). PASS.
- 고차 모드 2종: **disjoint**(겹치지 않는 3비트 parity → 고-Pauli차이나 reshape에 대해 **저랭크/분리가능**),
  **random**(겹치는 3·4비트 parity 다수 → 고-Pauli차 **+ 고랭크/비분리**). `eff_rank99`(reshape SVD 99%)로 진단.

## 7. 스윕 설계
n=6(64상태, side=8), t=0,0.2,0.4,0.6,0.8,1.0. N=80,000 샘플→q_emp. 예산 P=QCBM(L=3)=**54**.
모델 params: QCBM 54 / NMF rank3=48 / MLP hidden7=57 / marginal 6. QCBM **250 epoch, 3 restart(best)** (공정 학습).

## 8. 모델별 적합도 (KL(p_true‖model), nats)
**disjoint (고-Pauli차, 저랭크):**
| t | ≥3차 share | rank99 | QCBM | NMF | MLP | marginal |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 2 | 0.521 | 0.005 | 0.001 | 2.415 |
| 0.4 | 0.034 | 2 | 0.278 | 0.009 | 0.005 | 2.210 |
| 0.6 | 0.341 | 2 | 0.329 | 0.010 | 0.007 | 1.594 |
| 0.8 | 0.790 | 3 | 0.150 | 0.003 | 0.002 | 1.465 |
| 1.0 | 1.000 | 1 | 0.004 | 0.0002 | 0.001 | 1.352 |

**random (고-Pauli차, 고랭크):**
| t | ≥3차 share | rank99 | QCBM | NMF | MLP | marginal |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 2 | 0.521 | 0.005 | 0.001 | 2.415 |
| 0.2 | 0.017 | 2 | 1.309 | 0.576 | 0.002 | 0.910 |
| 0.4 | 0.240 | 3 | 0.831 | **1.241** | 0.002 | 1.250 |
| 0.6 | 0.178 | **4** | 0.745 | **2.857** | 0.003 | 1.990 |
| 0.8 | 0.025 | 1 | 0.328 | 0.939 | 0.001 | 1.013 |
| 1.0 | 0.003 | 1 | 0.069 | 0.203 | 0.001 | 0.154 |
(random은 t↔share/rank가 비단조: 항선택이 seed 의존. 실제 driver는 **eff_rank vs NMF rank**.)

## 9. 교차점 분석
- **"고전(NMF)이 무너지고 QCBM이 버티는 교차점" = 없음.** disjoint에선 NMF가 전 구간 우세(저랭크라 안 무너짐).
  random에선 NMF가 무너지나(rank99=4>NMF rank3 → KL 2.86), **그 지점에서 버티는 건 QCBM이 아니라 MLP**(KL 0.003).
- QCBM은 모든 구간·모든 모드에서 NMF 또는 MLP보다 나쁨(예외: disjoint t=1의 자명한 분리분포에서만 NMF와 동급).
- 즉 **양자 우위 교차점 부재**; 고전 저랭크가 깨지는 고랭크 영역도 동일예산 고전 신경망(MLP)이 완전 커버.

## 10. self-check (관찰)
- 고차≈0(저차 지배): NMF/MLP가 QCBM 압도(0.005/0.001 vs 0.52) — 실데이터 "고전 우위" 정성 재현. ✓
- 고랭크 영역(random t=0.4~0.6): NMF **붕괴**(KL 1.2~2.9). ✓ 단 MLP는 안 붕괴.
- 순수 parity(disjoint t=1): 저랭크라 NMF/MLP/QCBM 모두 양호(붕괴 아님) — "Pauli고차 자체는 어렵지 않음"의 직접 증거.

## 11. 사실 관찰 (판정 아님)
- **양자 우위 교차점은 이 통제 실험(n=6, 동일예산)에서 나타나지 않음.** 고전이 깨지는 유일 지점(고랭크)도 MLP가 버팀.
- "고전이 깨진다"는 **저랭크 NMF 한정**이고 matrix-rank가 driver(Pauli-order가 아님). disjoint 고차는 저랭크라 안 깨짐.
- **QCBM이 저차(share0)에서도 KL 0.52로 나쁜 것**은 핵심 단서: 학습(250ep/3restart) 강화 후에도 그러함 →
  (B 쪽: 회로/학습 한계) 또는 **MMD² 목적 vs KL 평가 불일치**(이 프로젝트의 적합-탐지 해리와 같은 결). QCBM MMD²는 낮으나 KL 큼.
- "양자가 이긴다/진다"를 이 소규모로 단정하지 않음 — 큰 n·구조적 hardness·KL목적 QCBM은 별도(아래 14).

## 12. 생성 파일 / 실행 / tmux
신규: `tools/synth_highorder_distribution.py`, `tools/exp_synth_quantum_vs_classical.py`,
`results2/synth_highorder/{synth_main_disjoint,synth_main_random,synth_smoke,synth_smoke_random}.{json,png}`, `run.log`, 본 로그.
```
python -m py_compile tools/synth_highorder_distribution.py tools/exp_synth_quantum_vs_classical.py   # OK
python tools/synth_highorder_distribution.py --selftest          # PASS(share 0→1 단조)
python tools/exp_synth_quantum_vs_classical.py --smoke [--high-mode random]   # smoke 확인
tmux new -s hoc_synth -d '... --high-mode disjoint ...; ... --high-mode random ...'  # 본 실행
```
- **tmux 세션 hoc_synth**, 로그 `results2/synth_highorder/run.log`. 두 모드 순차(단일 프로세스). 각 점 ~110s, 총 ~22분. 자동 종료.
- 합성만(실데이터/체크포인트 미사용). dependency 무설치. 긴 학습 없음(QCBM 점당 ≤2분). 실패/크래시 없음.

## 13. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 14. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] QCBM loss=MMD²/ansatz/params(L·nq·3) 확인·기록.
- [x] 합성 생성기 + self-check(β↔≥3차 share 단조) 통과.
- [x] NMF/marginal/QCBM **+MLP(numpy)** 비교곡선 json/png(2 모드).
- [x] self-check: 고차0 고전 우세 재현 / 고랭크서 NMF 붕괴.
- [x] **교차점 부재** 명확 기록(고전붕괴=NMF한정, MLP가 커버; 양자 우위 없음).
- [x] 기존 무변경·합성만·긴학습 없음·무설치.

**남은 불확실성**
- n=6 소규모(64상태) — 고전이 자명히 강함. 양자가 유리할 수 있는 큰 n·구조적 hardness 미탐.
- QCBM은 MMD²로 학습/KL로 평가(목적 불일치). KL목적 QCBM(예: NLL/KL loss)이면 결과 다를 수 있음(미검).
- random 모드 t↔share/rank 비단조(seed 의존). 단일 seed. eff_rank가 실제 driver.

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "통제 합성서 양자 교차점 부재; 고전 NMF가 깨지는 고랭크 영역도 동일예산 MLP가 완전 커버;
   QCBM은 저차에서도 KL 나쁨(MMD≠KL/회로한계). 즉 B(회로·목적) 쪽 신호." 양자 정당화 최종 정리.
2. QCBM을 **KL/NLL 목적**(또는 더 깊은 ansatz·더 큰 n=8~10)으로 재학습해 MMD-목적 confound 제거 후 재비교.
3. 고랭크+고차+**비-MLP-친화** 분포(예: 학습 어려운 구조) 설계로 "어떤 고전도 깨지는" 영역이 존재하는지 탐색.
