# QCBM loss MMD²→KL(forward) 교체 후 동일 합성 실험 재비교 — "MMD 목적 confound" 제거

- 작성: 2026-06-29 06:55 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). loss 외 변경 없음. 기존 코드 무수정(import/래핑). 단일 프로세스. 판정은 웹 AI.

## 1. 작업 목적
직전 합성 실험서 QCBM(MMD² loss)이 저차에서도 KL 0.52로 나빴음(NMF 0.005). confound: QCBM은 MMD²로 학습/KL로 평가.
QCBM cost를 **forward KL(cross-entropy: −Σ q·log p)**로 교체하고 직전과 동일 설정에서 재비교 → "MMD 목적 탓인가(개선됨) vs 회로 한계인가(여전히 나쁨)" 판별.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용만).

## 3. 확인한 QCBM cost/학습 구조 + 교체 지점
- `experiment2.train_one`(@175): cost(MMD²)가 **함수 내부 클로저**(@188-190) `p·Kp−2p·Kq+q·Kq`. → "cost만 교체" 불가 →
  변형 파일에서 **Adam 루프 재구현**(ansatz `E.make_circuit`@140 / `E.wshape`@150, 파라미터수 L·nq·3, pnp Adam @197 동일), cost만 KL.
- pnp: `from pennylane import numpy as pnp`(@43), `import pennylane as qml`(@42). 타깃 분포 q는 학습 입력(@176).
- 직전 실험 재사용: `tools/exp_synth_quantum_vs_classical.py`의 `qcbm_fit`(MMD)/`nmf_reconstruct`/`mlp_energy_fit`/`fit_marginal`/`eval_model`/`eff_rank`,
  `tools/synth_highorder_distribution.py`의 `synth_dist`/`make_terms`/`ge3_share`. → 동일 분포/예산/baseline/평가 보장.

## 4. KL 변형 정의 / self-test
- cost(w) = −Σ_s q_target[s]·log(circuit(w)[s]+ε) = forward KL(q_target‖p) − H(q_target) (상수차). ε=1e-10. qml.grad 미분.
- 학습 입력 q_target = **q_emp**(직전 MMD-QCBM과 동일 입력; 동일 설정 보장). 평가 = KL(p_true‖model)(직전과 동일).
- self-test: 임의 타깃서 KL이 epoch마다 감소 확인.

## 5. 확실한 사실 / 추정 / 확인 불가
- 확실: cost는 train_one 내부 클로저(분리 불가)→루프 재구현(ansatz/Adam/params 동일, cost만 KL). KL self-test 감소. 4모델+2 QCBM KL.
- 확인 불가 없음. 한계(해석): n=6 소규모, QCBM 250ep/3restart(여전히 유한 학습), 단일 seed.

## 6. self-test 결과 (KL 변형 정상)
임의 n=4 타깃: cross-entropy 3.12→2.46, KL(q‖p) 0.66→**0.0001**(거의 완벽 적합). KL 목적 QCBM은 분포를 정확히 학습함 → MMD-QCBM(KL~0.5)과 대조.

## 7. 재비교 설정 (직전과 동일)
n=6(64상태), t=0,0.2,0.4,0.6,0.8,1.0, N=80,000, 예산 P=L·nq·3=**54**(QCBM-MMD/KL 동일), NMF rank3·MLP hidden7·marginal,
평가 KL(p_true‖model)/TV, 250 epoch·3 restart(best), 샘플 seed 12345. 고차 모드 disjoint·random 둘 다. → **QCBM loss만 차이**.

## 8. 비교표 (KL(p_true‖model), nats)
**disjoint (고-Pauli차·저랭크):**
| t | share | rank99 | QCBM-MMD | **QCBM-KL** | NMF | MLP | marginal |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 2 | 0.521 | **0.197** | 0.005 | 0.001 | 2.415 |
| 0.2 | 0.001 | 2 | 1.050 | **0.156** | 0.023 | 0.002 | 2.600 |
| 0.4 | 0.034 | 2 | 0.278 | **0.258** | 0.009 | 0.005 | 2.210 |
| 0.6 | 0.341 | 2 | 0.329 | **0.188** | 0.010 | 0.007 | 1.594 |
| 0.8 | 0.790 | 3 | 0.150 | **0.055** | 0.003 | 0.002 | 1.465 |
| 1.0 | 1.000 | 1 | 0.004 | **0.0003** | 0.0002 | 0.001 | 1.352 |

**random (고-Pauli차·고랭크):**
| t | share | rank99 | QCBM-MMD | **QCBM-KL** | NMF | MLP | marginal |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 2 | 0.521 | **0.197** | 0.005 | 0.001 | 2.415 |
| 0.2 | 0.017 | 2 | 1.309 | **0.252** | 0.576 | 0.002 | 0.910 |
| 0.4 | 0.240 | 3 | 0.831 | **0.215** | 1.241 | 0.002 | 1.250 |
| 0.6 | 0.178 | **4** | 0.745 | **0.364** | 2.857 | 0.003 | 1.990 |
| 0.8 | 0.025 | 1 | 0.328 | **0.206** | 0.939 | 0.001 | 1.013 |
| 1.0 | 0.003 | 1 | 0.069 | **0.043** | 0.203 | 0.001 | 0.154 |

## 9. self-check
- **KL-QCBM ≤ MMD-QCBM 전 구간 = True**(두 모드 모두). 같은 목적 학습/평가라 정상, 그리고 KL이 항상 개선.
- 직전 MMD 재현: QCBM-MMD 저차 KL=0.521(직전 0.521과 동일) — 실험 일관성 ✓.
- KL이 NMF 넘는 구간: disjoint=False(NMF가 저랭크라 0.005로 매우 강함), **random=True**(NMF 붕괴 영역서 QCBM-KL이 NMF 넘음).
- KL이 MLP 넘는 구간: disjoint=**True**(t=1.0서 KL 0.0003 < MLP 0.001), random=False. (MLP는 거의 모든 곳 최강.)

## 10. 사실 관찰 (판정 아님)
- **MMD 목적 confound는 실재**: QCBM loss를 KL로 바꾸자 **전 구간 개선**(저차 0.521→0.197, random 고랭크 0.831→0.215, 0.745→0.364).
  즉 직전 "QCBM이 저차에서도 나쁨"의 상당부는 **MMD≠KL 목적 불일치 탓**(회로 한계만이 아님).
- **그러나 QCBM-KL도 동일예산 MLP를 일반적으로 못 넘음**(MLP 0.001~0.007로 거의 전 구간 최강). NMF가 무너지는 고랭크에선 QCBM-KL이 NMF는 넘지만 MLP는 못 넘음.
- 즉 KL 목적으로 confound를 제거해도 **이 통제 소규모에서 양자 우위(어떤 고전도 못 하는 영역)는 나타나지 않음** — 동일예산 고전 MLP가 여전히 우세.
- (추정) QCBM-KL 저차 KL 0.197은 여전히 MLP보다 크나, self-test(n=4→0.0001)로 보아 더 많은 epoch/restart·깊은 회로면 더 내려갈 여지(미검). "회로 한계"인지 "유한 학습"인지는 추가 학습으로 가려야 함.

## 11. 생성 파일 / 실행 / tmux
신규: `tools/qcbm_kl_variant.py`, `tools/exp_synth_kl_vs_mmd.py`,
`results2/synth_highorder_kl/{klmmd_main_disjoint,klmmd_main_random,klmmd_smoke_random}.{json,png}`, `run.log`, 본 로그.
```
python -m py_compile tools/qcbm_kl_variant.py tools/exp_synth_kl_vs_mmd.py     # OK
python tools/qcbm_kl_variant.py --selftest          # PASS(KL 0.66→0.0001)
python tools/exp_synth_kl_vs_mmd.py --smoke --high-mode random   # smoke
tmux new -s hoc_klmmd -d '... --high-mode disjoint ...; ... --high-mode random ...'  # 본 실행
```
- **tmux 세션 hoc_klmmd**, 로그 `results2/synth_highorder_kl/run.log`. 두 모드 순차, 점당 ~230s, 총 ~46분. 자동 종료.
- experiment2 무수정(make_circuit/wshape import만, cost는 변형 파일서 재구현). 직전 합성 스크립트 import 재사용. 합성만, 무설치, 실패 없음.

## 12. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 13. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] QCBM cost/학습 구조 확인, loss 교체 지점(클로저→루프 재구현) 기록.
- [x] KL 변형 기존 무수정 구현(import/재사용) + self-test(KL 감소) 통과.
- [x] 동일 설정 QCBM-KL vs QCBM-MMD vs NMF/MLP/marginal json/png(disjoint+random).
- [x] self-check: KL≤MMD 전구간, 직전 MMD 재현(0.521).
- [x] 핵심 질문 사실 기록: 저차서 KL-QCBM 개선(0.52→0.20), NMF는 고랭크서 넘으나 MLP는 못 넘음.
- [x] 기존 무변경·합성만·긴학습 없음·무설치.

**남은 불확실성**
- QCBM-KL 저차 0.197 > MLP 0.001: 유한 학습(250ep/3restart) 탓인지 회로 표현력 탓인지 미분리.
- n=6 소규모·단일 seed. 큰 n·깊은 ansatz 미검.

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "MMD confound 실재(KL로 전구간 개선); 그래도 동일예산 MLP가 우세, 양자 우위 영역 부재(소규모)." 정리.
2. QCBM-KL을 epoch↑/restart↑/L↑로 더 학습해 저차 0.197이 MLP 수준(0.001)까지 내려가는지 → "회로 한계 vs 유한 학습" 판별.
3. n=8~10 확장 + KL목적으로 재비교(고전이 어려워지는 규모에서 교차점 재탐색).
