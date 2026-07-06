# B2(12큐빗) QCBM 결과 박제 — 재학습 없이 로그에서 JSON·그림 복원 + 231줄 버그 수정

- **작성일시**: 2026-06-26 10:01 (로컬)
- **작성 주체**: 로컬 Claude Code agent (조사→수정)
- **작업 성격**: **재학습 없음.** 로그 기반 박제(경로 A) + train_qcbm_iat_b2.py 231줄 한 줄 수정.

---

## 1. 작업 목적

12큐빗 QCBM(B2) 학습은 성공적으로 끝났으나(full wall 1319.8s≈22분), 학습 직후 `[B1 대비]` 로그 한 줄의
`%` 이스케이프 버그(`99.5%` → `% |` 오해석)로 **JSON 저장·그림 생성 직전에 죽음**. 학습/평가는 모두 끝난
뒤라 **과학적 결과는 무손실**. → (1) 결과를 JSON·그림으로 박제, (2) 231줄 버그 수정. 재학습 금지.

---

## 2. 시작 시점 git status

```
?? docs/   ?? qcbm/   (미추적, git diff --stat 비어 있음)
```
- 보존해야 할 무관한 변경 없음. 이번 작업 외 변경 없음.

---

## 3. train_qcbm_iat_b2.py 231줄 이후 구조

`[B1 대비]` 로그 라인([train_qcbm_iat_b2.py:231](../../qcbm/tools/train_qcbm_iat_b2.py#L231)) **다음**에:
- `E.plot_training(cfg, TAG)` — 16재시작 loss/gnorm 히스토리(메모리) 필요 → `train_qcbm_iat_b2.png`
- `E.plot_dist(d["q_train"], best, TAG)` — q_train + best p_final(메모리) 필요 → `dist_qcbm_iat_b2.png`
- `E.plot_anomaly(an_qcbm, TAG)` — ROC fpr/tpr 배열(메모리) 필요 → `anomaly_qcbm_iat_b2.png`
- `json.dump(payload, ...)` → `results2/qcbm_iat_b2.json`

→ 이 모든 코드가 231줄 ValueError로 **미실행**. 4개 산출물(그림 3 + JSON 1) 모두 미생성.

---

## 4. results2/ 조사 결과

| 항목 | 존재? |
|------|-------|
| tee 로그 `qcbm_iat_b2_train.log` | ✅ 25,292 bytes, 모든 스칼라값 포함 |
| smoke `qcbm_iat_b2_smoke.json` | ✅ (decision 값) |
| `qcbm_iat_b2.json` | ❌ 미생성(크래시) |
| best 모델/분포 배열(.npy/.npz/checkpoint) | ❌ **없음** — 스크립트가 p_final을 디스크에 저장하지 않음 |

---

## 5. 선택한 복원 경로: **경로 A** (로그 기반 스칼라 박제)

근거: best 모델/분포 배열이 디스크에 **전혀 저장되지 않음**(스크립트에 .npy 저장 코드 없음). 따라서
원본 배열이 필요한 그림(분포 real vs QCBM, ROC 곡선, 학습곡선)은 **재학습 없이는 복원 불가**.
경로 B(저장 모델 로드)는 적용 불가. → 로그의 확정 스칼라값으로 JSON 박제 + 실스칼라 비교 그림만 생성.
(재학습은 금지 사항이자, seed 난수로 미세 변동 위험이 있어 "그" 결과를 보존하는 것이 옳음.)

---

## 6. 확실한 사실 / 추정 / 확인 불가

**확실(로그에서 직접):**
- HOIC AUC: marginal **0.8741**(J 0.7558, logp격차 3.874), emp_joint **0.9258**(J 0.8448, 8.054), **QCBM 12q L=6 0.9185**(J 0.8374, 7.475).
- QCBM−marginal **+0.0445**, emp_joint 상한근접 **99.2%**. QCBM p: ben 1.528e-2 vs atk 2.384e-3(6.41x); logp ben −4.260 vs atk −11.735.
- 학습: best MMD² **1.8865e-4**(seed 14), 중앙값 3.9601e-4, **TV 0.0930**, KL 0.1141, **수렴~284ep**, full wall 1319.8s.
- gradient 중앙값 초기 8.110e-3 → 최종 1.252e-5. 16재시작 중 seed 0/1/12만 국소최적(TV≈0.50/0.50/0.75), 나머지 13개 TV 0.08~0.12.
- decision(smoke.json): smoke 112.9s, 2.259s/ep, est_300 677.6s, 확정 16/300.
- IAT 분위수 경계(iat_joint.json, 동일 benign train): fwd `[5.7390,5.9067,7.8581]`, flow `[5.3351,5.5026,7.4548]`.

**확인 불가(원본 배열 없음):** best p_final 분포 배열, 16재시작 loss/gnorm 히스토리, ROC fpr/tpr 배열.

---

## 7. 생성한 파일

- **`results2/qcbm_iat_b2.json`** (2,118 B): train_qcbm_iat_b1.json 스키마를 따라 위 모든 스칼라값 박제
  (encoding/nq/nstates/occupied/split/IAT edges/layers/lr/decision/full_wall/train{…}/auc{…}/compare_b1/`_rescue_note`).
- **`results2/compare_b1_b2_qcbm.png`**: B1 vs B2 × {marginal, emp_joint, QCBM} 실제 AUC 막대그림(로그 실스칼라).
  - 주의: 그림 제목의 한글 일부가 폰트 미탑재로 □로 표시될 수 있음(막대/숫자/범례는 정상). cosmetic.

---

## 8. 복원하지 못한 항목과 사유

| 미복원 그림 | 사유 |
|-------------|------|
| `train_qcbm_iat_b2.png` (학습곡선+gradient) | 16재시작 loss/gnorm 히스토리 배열 미저장 |
| `dist_qcbm_iat_b2.png` (분포 real vs QCBM) | best p_final(4096차원) 배열 미저장 |
| `anomaly_qcbm_iat_b2.png` (점수 히스토그램+ROC) | ROC fpr/tpr·점수 배열 미저장 |

→ 모두 **재학습해야만** 복원 가능(금지). 빈/가짜 그림은 만들지 않음.

---

## 9. 231줄 버그 수정 전/후

- **수정 전**: `"  [B1 대비] B1: QCBM-marg=+0.0085, 상한근접 99.5% | B2: QCBM-marg=%+.4f, 상한근접 %.1f%%"`
  → 리터럴 `%`(`99.5%`) 다음의 ` |`를 포맷 지정자로 오해석 → `ValueError: unsupported format character '|'`.
- **수정 후**: `... 상한근접 99.5%% | ...` (리터럴 `%`를 `%%`로 이스케이프). **출력 숫자/내용 동일**, 로직 불변.
- 검증: `python3 -c "...그 포맷 문자열만 출력..."` → `... 상한근접 99.5% | B2: QCBM-marg=+0.0445, 상한근접 99.2%` 정상 출력, ValueError 없음.

---

## 10. 실행한 명령과 결과

```bash
git status                                                     # ?? docs ?? qcbm
python3 -m py_compile tools/train_qcbm_iat_b2.py tools/rescue_b2_from_log.py   # PY_COMPILE_OK
python3 -u tools/rescue_b2_from_log.py                         # JSON + compare png 생성, DONE
python3 -c "<231줄 포맷만 출력>"                                # 정상(ValueError 없음)
python3 -c "json.load(qcbm_iat_b2.json) 키/값 확인"            # QCBM 0.9185, -marg +0.0445, 근접 99.2%
```
- **재학습 실행 안 함.** rescue 스크립트는 로그값 기록 + 막대그림만(학습 코드 없음).

---

## 11. git diff --stat / 변경 한정 확인

- `qcbm/` 미추적이라 `git diff --stat`은 비어 있음(추적 변경 0).
- **train_qcbm_iat_b2.py 변경 = 231줄 단 한 줄**(`99.5%`→`99.5%%`). 학습/평가/AUC/인코딩 로직 무변경.
- 신규 파일: `tools/rescue_b2_from_log.py`, `results2/qcbm_iat_b2.json`, `results2/compare_b1_b2_qcbm.png`, 본 `.md`.

---

## 12. 성공 기준 충족 여부

- [x] qcbm_iat_b2.json 생성, 핵심 수치(QCBM 0.9185 / marginal 0.8741 / emp_joint 0.9258 / −marg +0.0445 / 근접 99.2% / MMD²·TV·gradient·수렴) 포함.
- [x] 가능한 그림(실스칼라 비교) 생성, 복원 불가 3그림은 사유 명시.
- [x] 231줄 버그 수정 + py_compile 통과 + 포맷 출력 검증.
- [x] 재학습 없음(로그 mtime 09:14 유지, rescue는 학습 코드 없음).
- [x] train_qcbm_iat_b2.py 변경이 231줄 한 줄로 한정.
- [x] .md에 경로 A 결정·근거·실행·미복원 항목 기록.

---

## 13. 남은 불확실성

1. 로그 스칼라는 **출력 정밀도(주로 소수 4자리)** 한정. b1 JSON 같은 풀정밀도는 재학습해야만 확보(불필요).
2. 분포/ROC 시각 자료는 논문 figure에 필요할 수 있음 → 필요 시 재학습(결정론적, seed 고정이라 동일 수치 재현)으로만 생성 가능.

---

## 14. 다음 추천 작업

1. (논문 figure 필요 시) 231줄 고쳐진 `train_qcbm_iat_b2.py`를 tmux로 **1회 재실행** → 동일 수치 + 그림 3종 정식 생성. (향후 재발 방지로, 학습 스크립트가 best p_final을 `.npy`로 저장하도록 개선 권장 — 단 이는 별도 작업.)
2. B1·B2 결과를 웹 AI에 전달: **"joint 이득↑(+0.013→+0.052)에 따라 QCBM 우위↑(+0.0085→+0.0445), 상한 99%대 실현"** — 단조 관계로 핵심 주장 강화.

---

## 최종 보고

```markdown
## 작업 요약
- B2 학습 결과를 재학습 없이 로그에서 박제. qcbm_iat_b2.json + 실스칼라 비교그림 생성. 231줄 % 버그 수정.
  QCBM 12q AUC 0.9185 > marginal 0.8741 (+0.0445), emp_joint 0.9258 상한 99.2% 실현. 분포/ROC 그림은 모델 미저장으로 복원 불가.

## 선택한 복원 경로 (A/B)와 근거
- 경로 A. best 모델/분포 배열이 디스크에 없음(스크립트가 p_final 미저장) -> 배열 필요 그림 복원 불가, 로그 스칼라만 박제.

## 생성한 파일
- results2/qcbm_iat_b2.json: 인코딩/분할/IAT경계/decision/train{best_mmd2,tv,kl,conv,gnorm,nparams,best_seed,국소최적seed}/auc{전체}/compare_b1.
- results2/compare_b1_b2_qcbm.png: B1 vs B2 × {marginal,emp_joint,QCBM} 실제 AUC 막대(실스칼라).
- (복원 불가) train_/dist_/anomaly_qcbm_iat_b2.png: 히스토리/p_final/ROC 배열 미저장 -> 재학습 필요.

## 231줄 버그 수정
- 수정 전: "... 상한근접 99.5% | B2: ..."  (리터럴 % 미이스케이프 -> ValueError)
- 수정 후: "... 상한근접 99.5%% | B2: ..." (출력 내용 동일)

## 실행한 검증
- py_compile OK; rescue 실행 DONE; 231줄 포맷 단독 출력 정상(ValueError 없음); JSON 키/값 확인.

## 재학습 없음 확인
- train.log mtime 09:14 유지, rescue 스크립트에 학습 코드 없음.

## git diff --stat
- qcbm 미추적이라 비어 있음. train_qcbm_iat_b2.py 변경은 231줄 한 줄로 한정.

## 확인된 사실
- QCBM 0.9185 / marginal 0.8741 / emp_joint 0.9258 / QCBM-marg +0.0445 / 상한근접 99.2% / TV 0.093 / 수렴 284ep.

## 복원 불가 항목과 사유
- 분포·ROC·학습곡선 그림: best p_final/히스토리/ROC 배열 미저장.

## 다음 추천 단계
1. 논문 figure 필요 시 고쳐진 스크립트를 tmux로 1회 재실행(결정론적 동일수치 + 그림). 차후 p_final .npy 저장 추가 권장.
2. B1·B2 단조관계 결과를 웹 AI에 전달해 핵심 주장 강화.
```
