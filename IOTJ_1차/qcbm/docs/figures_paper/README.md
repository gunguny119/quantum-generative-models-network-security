# Figures (논문/판정용 정리)

QCBM 양자 우위 검증 핵심 그림 모음. 원본은 `results2/`에 그대로 보존되어 있고, 여기에는 **복사본**만 들어 있다(원본 무수정).

| Figure | 무엇을 보여줌 | 뒷받침하는 주장 | Exp | 우선순위 | 파일 | 원본 |
|---|---|---|---|---|---|---|
| **F1** | joint 이득 비교 (HOIC/Bot/UNSW) | joint > marginal 이 데이터를 가로질러 성립 | Exp 1, 6 (핵심) | 높음 | `F1_joint_gain_cross_dataset.png` | `results2/cross_dataset_diagnosis.png` |
| **F2** | 복잡도 vs 양자우위 경계 | rank↑일수록 양자/고전 경계 규명 | Exp 2, 4 | 높음 | `F2_complexity_vs_quantum_boundary.png` | `results2/attack_complexity.png` |
| **F3a** | 파라미터 효율 곡선 (Bot) | params vs AUC, 고전 vs QCBM (효율 부재) | Exp 4 | 높음 | `F3a_param_efficiency_bot.png` | `results2/param_efficiency_bot.png` |
| **F3b** | 파라미터 효율 곡선 (UNSW) | params vs AUC, 고전 vs QCBM (효율 부재) | Exp 4 | 높음 | `F3b_param_efficiency_unsw.png` | `results2/param_eff_unsw.png` |
| **F4** | TV–AUC / NLL–AUC 산점도 | 적합도–탐지 해리/역상관 | Exp 5 (핵심 발견) | 높음 | `F4_tv_auc_dissociation.png` | `results2/tv_auc_dissociation.png` |
| **F4b** | 비지도 선택 분석 (보조) | min-MMD²/TV 선택 가능성 | Exp 5 | 높음 | `F4b_unsup_selection.png` | `results2/unsup_selection.png` |
| **F5a** | QCBM restart 분포 (Bot, 16개) | best vs median, 견고성 (best 운) | Exp 4 | 중간 | `F5a_restart_dist_bot.png` | `results2/bot_efficiency_robustness.png` |
| **F5b** | QCBM restart 분포 (UNSW, 16개) | best vs median, 견고성 (우측 패널) | Exp 4 | 중간 | `F5b_restart_dist_unsw.png` | `results2/param_eff_unsw.png` |
| **F6a** | 학습 곡선 MMD/gradient (Bot) | 학습 정상, barren plateau 없음 | Exp 3 (보조) | 낮음 | `F6a_train_curve_bot.png` | `results2/train_qcbm_bot.png` |
| **F6b** | 학습 곡선 MMD/gradient (UNSW) | 학습 정상, barren plateau 없음 | Exp 3 (보조) | 낮음 | `F6b_train_curve_unsw.png` | `results2/train_qcbm_unsw.png` |

## 비고
- **F3·F5 (UNSW)**: 원본 `param_eff_unsw.png`는 좌(효율 곡선)+우(16-restart AUC 분포) 2패널이라 F3b·F5b가 동일 원본을 공유한다 — 효율은 좌측, restart 분포는 우측 패널을 본다.
- **F4b**(`unsup_selection.png`)는 F4의 보조 그림(실전 선택 가능성). 단일 그림만 필요하면 F4를 쓴다.
- 데이터셋별 그림이 둘인 경우(F3/F5/F6) `a`=Bot, `b`=UNSW. F1은 HOIC/Bot/UNSW를 한 그림에 포함.

## 핵심 수치 (참고)
- joint vs marginal (UNSW/ALL, M2): emp_joint AUC ≈ 0.871 > marginal ≈ 0.703.
- 효율: UNSW 동일 180p 예산에서 고전(NMF r=2, 128p) 0.835 > QCBM best 0.793 (고전 약 1/3 params로 도달).
- 견고성: 고전최고 초과 QCBM restart — Bot 5/16, UNSW 0/16.
- 선택: corr(MMD², AUC) — Bot 0.077(무상관), UNSW −0.375.
