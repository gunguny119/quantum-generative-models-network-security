# psh 제거 11비트 QCBM(MMD) 본 학습 — 강신호 없는 순수 상호작용을 QCBM이 학습하는가

> **전제: 기존 소스 무수정(import + 모듈속성 대입). 학습은 tmux 보호 실행.**

- **작성일시**: 2026-06-26 11:10 (로컬), smoke 11:12, 본 학습 11:13~ (tmux)
- **작성 주체**: 로컬 Claude Code agent (default mode)

---

## 1. 작업 목적

psh 통제 분석에서 psh 제거 시 marginal 0.5136(붕괴)·emp_joint 0.7773(유지), joint이득 +0.2637. 이 0.7773은 emp_joint(상한).
**QCBM(MMD, 표현력 제한)이 psh 없는 순수 상호작용을 학습으로 실현하는지** 검증.
질문: QCBM AUC > marginal(0.5136)? emp_joint 상한(0.7773) 근접도?(B1 99.5%, B2 99.2% 실현)
- 인코딩: proto2+syn1+size4+FwdIAT2+FlowIAT2 = **11bit, 2048상태**(psh 제거). 대상 HOIC.

---

## 2. 시작 git status / tmux

```
?? docs/ ?? qcbm/ ?? qcbm_share_20260626.zip (전부 미추적; 추적 변경 0)
which tmux: /usr/bin/tmux (3.2a), 잔여 세션 없음
```
- 추가 파일: `tools/train_qcbm_iat_no_psh.py`, `results2/qcbm_iat_no_psh.json`, `..._train.log`, `..._smoke.json`, `results2/{train,dist,anomaly}_qcbm_iat_no_psh.png`, 본 `.md`.
- experiment2.py 등 **무수정**.

---

## 3. 재사용 경로 / 11비트 인코딩 / 정합 검증

- 학습: `E.run_config(SPEC(nq=11), q_train, size_edges)` 직접(B1·B2 검증된 경로, 내부 discretize 재호출 없음). E.LAYERS/EPOCHS/RESTARTS/LR/OUT 모듈속성 대입.
- q_train: `INFO.load_split`(seed=7,0.7) + `INFO.feature_levels`로 feature별 레벨 → **psh 제외** [proto2,syn1,size4,fwd2,flow2] MSB→LSB 조립 → 2048상태. IAT edges는 B2와 동일(Fwd [5.739,5.907,7.858], Flow [5.335,5.503,7.455]).
- 평가: `s=-log p+1e-12`, attack=positive, `auc_score`/`anomaly_eval`(raw MWU). marginal=`marginal_dist`(비트독립 곱).
- **정합 검증(학습 전)**: emp_joint **0.7773**, marginal **0.5136**, 점유 **90** → 기대(0.7773/0.5136/90) **완전 일치** ✅. 11비트 인코딩 재현 정확.

---

## 4. smoke 시간 + 본 학습 설정

- smoke(R=1,EP=50,L=6): wall **76.2s → 1.52s/ep** (B1 1.47, B2 2.26 사이, 예상대로). best MMD² 4.86e-3, TV 0.441(50ep/1restart라 미수렴).
- 추정: **est_300 ≈ 457s (~7.6분)** ≤3h → **확정 R=16, EP=300, L=6, LR=0.1**.
- 점유 90/2048(4.4%) — B2(2.3%)보다 덜 희소(psh 제거로 상태공간 절반).

---

## 5. 본 학습 실행 방식 (tmux 보호)

```bash
# Claude가 detached로 기동 (cwd=qcbm)
tmux new-session -d -s qcbm_nopsh "bash -lc 'export OMP_NUM_THREADS=1 ...; \
  python3 -u tools/train_qcbm_iat_no_psh.py 2>&1 | tee results2/qcbm_iat_no_psh_train.log; echo EXIT_CODE=\$?'"
```
사용자 표준 절차:
```bash
cd /home/elicer/IOTJ/qcbm
tmux new -s qcbm_nopsh
python3 -u tools/train_qcbm_iat_no_psh.py 2>&1 | tee results2/qcbm_iat_no_psh_train.log
# Ctrl+b 후 d (detach) / tmux attach -t qcbm_nopsh / tmux kill-session -t qcbm_nopsh
```
- detached라 연결 끊겨도 지속. 로그 `results2/qcbm_iat_no_psh_train.log`.

---

## 6. 학습 결과 (완료)

- **best MMD² = 6.66e-5**, **best TV = 0.062**(분포 학습 우수), KL 0.051.
- 중앙값 MMD² 3.12e-4, **수렴 ~220 epoch**.
- **gradient norm: 초기 1.07e-2 → 최종 1.00e-5** (정상 감소, 11큐빗 barren plateau 심각 조짐 없음).
- nparams 198 (L=6, nq=11). full wall **893.7s (≈14.9분)**. 점유 90/2048.

---

## 7. 핵심 비교표 (완료)

| 모델 | AUC | Youden J | logp격차 | 비고 |
|------|-----|----------|----------|------|
| marginal | 0.5136 | — | — | psh 제거 시 붕괴(거의 무작위) |
| emp_joint | 0.7773 | — | — | train 결합분포 암기 = 표현 상한 |
| **QCBM (11q, MMD, L=6)** | **0.8023** | 0.6719 | 1.022 | 우리 모델 |

- **QCBM > marginal(0.5136)? → 예** (+0.2887, 약 +0.29).
- **QCBM(0.8023) > emp_joint(0.7773)**, 상한근접도 = **103.2%** — QCBM이 train-암기 상한을 **초과**.
- 해석(사실): emp_joint는 benign **train** 결합분포를 그대로 외운 분포 → train 적합엔 상한이지만, **test/공격에 대한 탐지에선 과적합**. QCBM의 MMD 평활화가 **더 잘 일반화**해 held-out 평가에서 emp_joint를 넘음. (B1/B2는 상한의 99%대였으나, psh 없는 이 희소 상호작용 영역에선 평활화의 일반화 이득이 상한을 초과.)
- B1·B2 맥락: B1 QCBM 0.8811(상한 99.5%), B2 0.9185(99.2%), **no-psh 0.8023(103.2%)**.

---

## 8. 실행 명령과 결과

```bash
git status; which tmux                                  # ?? ... ; tmux 3.2a
python3 -m py_compile tools/train_qcbm_iat_no_psh.py    # PY_COMPILE_OK
python3 -u tools/train_qcbm_iat_no_psh.py --smoke-only  # 정합 OK(0.7773/0.5136/90), smoke 76.2s, 확정 16/300
# 본 학습: tmux detached (results2/qcbm_iat_no_psh_train.log)
```

---

## 9. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → 비어 있음. experiment2.py / train_qcbm_iat_b2.py / diagnose_*.py **무수정**(import만).

---

## 10. 사실 관찰 (판정 아님 — 웹 AI)

1. **QCBM이 marginal(0.5136)을 +0.29 압도** → psh(쉬운 강신호) 없이도 QCBM이 **순수 joint 상호작용만으로 탐지**. joint>marginal이 양자모델 학습으로도 실현됨(빈도 상한뿐 아니라).
2. **QCBM(0.8023)이 emp_joint 상한(0.7773)까지 초과**(103.2%) — train-암기보다 QCBM 평활화가 **더 잘 일반화**(held-out 공격 탐지). 분포탐지에서 양자모델의 일반화 이점 정황.
3. TV 0.062로 분포 학습 우수, barren plateau 심각 조짐 없음(11큐빗).
4. 종합: psh 통제 후에도 (a) joint>marginal이 빈도 상한에서 +0.26, (b) QCBM이 그 상한을 학습·초과(0.80) → **"joint 상호작용 가치 + QCBM의 학습·일반화"** 라인이 강신호 artifact 없이 성립.

---

## 11. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → 비어 있음. experiment2.py / train_qcbm_iat_b2.py / diagnose_*.py **무수정**(import만).
- 신규: `tools/train_qcbm_iat_no_psh.py`, `results2/qcbm_iat_no_psh.{json,png 3종,_train.log,_smoke.json}`, 본 `.md`.

---

## 12. 다음 추천

1. (서사 완성) 핵심 결과 3종 묶기: ① psh 통제 시 joint 이득 +0.26(빈도) ② QCBM이 순수 상호작용 학습(marginal 0.51 → QCBM 0.80) ③ QCBM이 train-암기 상한(0.78)을 일반화로 초과(103%). → "쉬운 강신호에 가려진 joint 가치를 QCBM이 학습·일반화로 드러낸다"가 논문 핵심.
2. (견고화) QCBM 0.80 > emp_joint 0.78의 "일반화 초과"를 다중 seed/분할로 재확인(분산 점검).
3. (양자 정당성) 파라미터 효율은 닫혔으므로(B1·B2), 정당성을 **정확도+일반화**로 정렬.

---

## 최종 보고

```markdown
## 작업 요약
- psh 제거 11비트(2048상태)로 QCBM(MMD,L=6) tmux 학습. 정합검증(emp 0.7773/marg 0.5136/occ90) 통과.
  QCBM AUC=0.8023 > marginal 0.5136(+0.29), 그리고 emp_joint 상한 0.7773까지 초과(103.2%, 일반화).
  TV 0.062, 수렴~220ep, barren plateau 없음. 기존 소스 무수정, 5h 예산 내(14.9분).

## 본 학습 실행 방법 (tmux)
cd /home/elicer/IOTJ/qcbm
tmux new -s qcbm_nopsh
python3 -u tools/train_qcbm_iat_no_psh.py 2>&1 | tee results2/qcbm_iat_no_psh_train.log

## 11비트 인코딩 / 정합 검증
- proto2+syn1+size4+FwdIAT2+FlowIAT2 (psh 제거), 점유 90/2048. emp 0.7773/marg 0.5136 일치(인코딩 재현 OK).

## 학습 경로 및 설정
- E.run_config 직접(11bit SPEC + 2048 q_train). smoke 1.52s/ep -> est_300 457s -> 확정 R=16 EP=300 L=6.

## 학습 결과
- best MMD² 6.66e-5, TV 0.062, KL 0.051, 수렴~220ep, gradient 1.07e-2->1.00e-5, wall 893.7s.

## 핵심 비교표
| 모델 | AUC | Youden J | logp격차 |
| marginal | 0.5136 | — | — |
| emp_joint | 0.7773 | — | — |
| QCBM 11q L=6 | 0.8023 | 0.6719 | 1.022 |
- 상한근접도(QCBM/emp_joint) = 103.2% (상한 초과).

## 사실 관찰 (판정 아님)
- QCBM이 marginal(0.5136)을 +0.29 압도 -> 순수 상호작용을 QCBM이 학습.
- QCBM(0.8023)이 emp_joint 상한(0.7773) 초과 -> train-암기보다 QCBM 평활화가 더 잘 일반화.
- B1 99.5%/B2 99.2% 대비 no-psh는 103.2%(일반화 이득이 상한 초과).

## git diff --stat
- 비어 있음(미추적). 기존 소스 무수정(import만).

## 다음 추천 단계
1. 핵심 결과 3종(psh통제 +0.26 / QCBM 순수상호작용 학습 / 상한 초과 일반화)을 논문 라인으로.
2. QCBM>emp_joint 일반화 초과를 다중 seed로 재확인(분산).
```
