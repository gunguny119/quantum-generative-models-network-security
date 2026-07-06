# psh 제거 11비트 학습의 16 restart 분산 정찰 — best가 운인가 견고한가

> **전제: 기존 소스 무수정. QCBM 재학습 없음. 저장 결과/로그 분석만.**

- **작성일시**: 2026-06-26 11:39 (로컬), 실행 11:41
- **작성 주체**: 로컬 Claude Code agent (조사→분석)
- **결론 선요약**: ★ **학습 품질은 견고**: 16 restart 중 **15개가 잘 학습(TV≤0.15)**, 명백한 실패(TV>0.4) **0개**. best(seed15, TV 0.062)는 분포의 좋은 끝이지만 **다수가 비슷한 품질**(median TV 0.109). 단 **AUC-vs-emp_joint(0.7773) 초과가 몇 restart인지는 p_final 미저장으로 측정 불가 → 1회 재실행 필요**(다음 단계).

---

## 1. 작업 목적

QCBM AUC 0.8023 > emp_joint 0.7773(103.2%)이 "견고"인가 "best 선택의 운"인가? 재학습 없이 저장 결과/로그로 16 restart 분포를 정찰. 다중 seed 본격 재확인(비용 큰 다음 실험) 투자 가치 판단. 판정은 웹 AI.

---

## 2. 시작 git status / 보존 대상

```
?? docs/ ?? qcbm/ ?? qcbm_share_20260626.zip (전부 미추적; 추적 변경 0)
```
- 보존 대상 무관 변경 없음. 작업 후에도 기존 파일 무변경(§9).

---

## 3. 저장물 확인 → 경우 B 판정

| 항목 | 결과 |
|------|------|
| `qcbm_iat_no_psh.json` 키 | encoding/nq/.../train/auc/... — **`train`은 best 요약만**(best_mmd2, best_tv...), `results` 배열·`p_final` **없음** |
| no_psh restart .npy/.npz | **없음** |
| 학습 로그 완료 줄 | **17개** = full 16 + smoke 1(seed0, 50ep) |

→ **경우 B**: 16개 p_final 디스크 미저장 → 각 restart의 **AUC 재학습 없이 복원 불가**. 로그의 **MMD²/TV 분포만** 분석 가능. (train_qcbm_iat_no_psh.py가 best만 json에 저장 — B2 박제 전례와 동일.)

---

## 4. 파싱 방법 / 검증

- 로그에서 **full 학습 구간**(라인 22 `[full] R=16 EP=300` ~ 251 `[full] wall=`) 안의 "완료" 줄만 파싱 → **smoke(앞쪽) 제외**, full 16개 정확 추출.
- 검증: 파싱 best = seed15, MMD² 6.66e-5, TV 0.0622 → json `best_tv`(0.0622)와 **정확 일치**. median MMD² 3.12e-4 → json `median_mmd2`(0.000312)와 **일치**. 파싱 정확.

---

## 5. 확실 / 추정 / 확인 불가

- **확실**: 16 restart의 MMD²/TV(로그), 분포 통계, best 위치.
- **추정**: 15/16이 잘 학습(TV≤0.15) → best는 운이 아니라 견고한 학습의 좋은 끝일 가능성 높음. 다만 이는 **분포 품질** 기준.
- **확인 불가**: 각 restart의 이상탐지 AUC(p_final 미저장). "emp_joint 0.7773 초과 restart 수"는 재실행해야 측정.

---

## 6. 결과 — (경우 B) 16 restart 품질 분포

| 지표 | min | median | max | mean | std |
|------|-----|--------|-----|------|-----|
| MMD² | 6.66e-5 | 3.12e-4 | 2.05e-3 | 4.08e-4 | 4.32e-4 |
| TV | 0.0622 | 0.1086 | 0.2671 | 0.1153 | 0.0410 |

- **best restart**: seed 15, MMD² 6.66e-5, TV 0.0622 (MMD²·TV 모두 최소 = 분포의 가장 좋은 끝).
- **잘 학습(TV ≤ 0.15): 15/16** | 명백한 국소최적 실패(TV > 0.4): **0/16**.
- (대조: B2는 16 중 3개가 TV 0.5~0.75 실패였음 → no-psh는 **분포 학습이 더 안정적**, 상태공간 절반·덜 희소 덕으로 추정.)
- best TV(0.062)는 16개 중 최저(상위 ~6%). 그러나 median 0.109, 15개가 ≤0.15에 몰려 **품질이 좁게 군집**(best가 외딴 outlier 아님).

### AUC per restart
- **UNAVAILABLE** — p_final 미저장. "emp_joint(0.7773) 초과 restart 수"는 측정 불가.

---

## 7. 사실 관찰 (판정 아님 — 웹 AI)

1. **분포 학습 품질은 견고**: 15/16이 잘 학습(TV≤0.15), 실패 0개, 품질이 median 0.109 부근에 좁게 군집. best(0.062)는 그 군집의 좋은 끝이지 외딴 운이 아님.
2. **그러나 핵심 주장(QCBM AUC > emp_joint)의 restart 견고성은 직접 미확정**: AUC는 분포 품질과 단조가 아닐 수 있어, "15/16이 잘 학습"이 곧 "15/16이 emp_joint AUC 초과"를 보장하지 않음. p_final 없어 측정 불가.
3. 정황: 학습이 안정적(군집)이므로 best 0.8023이 단일 운일 가능성은 낮아 보이나, **AUC 분포는 재실행으로만 확정** 가능.

---

## 8. 생성 파일 / 실행

- `results2/no_psh_restart_analysis.json`: case B, 16 restart(seed/mmd2/tv/grad), MMD²·TV 통계, best, well-learned 수, AUC 불가 명시.
- `results2/no_psh_restart_analysis.png`: TV 히스토그램(+best/0.15선), log10 MMD² 히스토그램.
- `tools/analyze_no_psh_restarts.py` (신규).
```bash
python3 -m py_compile tools/analyze_no_psh_restarts.py   # PY_COMPILE_OK
python3 -u tools/analyze_no_psh_restarts.py              # full 16 파싱, best seed15 일치, DONE
```
- 재학습 없음(로그 파싱만). dependency 무설치.

---

## 9. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → 비어 있음. experiment2.py / train_qcbm_iat_no_psh.py 등 **무수정**.
- 신규: `tools/analyze_no_psh_restarts.py`, `results2/no_psh_restart_analysis.{json,png}`, 본 `.md`.

---

## 10. 성공 기준 / 남은 불확실성 / 다음 추천

충족: 저장물 확인(경우 B 판정·근거) ✅, MMD²/TV 분포·best 위치 ✅, AUC 분석엔 재실행 필요 명시 ✅, json/png ✅, 무수정·무재학습·무설치 ✅.

불확실: 16 restart의 **AUC 분포**(emp_joint 초과 수) — p_final 미저장으로 미측정.

다음 추천:
1. **(핵심) 16 restart AUC 견고성 확정에는 1회 재실행 필요**: train_qcbm_iat_no_psh.py를 응용해 **각 restart의 p_final로 AUC를 계산·저장**하는 1회 학습(결정론적, seed 고정 → best는 동일 0.8023 재현). 그러면 "emp_joint 0.7773 초과 restart N/16"을 직접 측정. (tmux, ~15분.)
2. 분포 품질 견고성(15/16 well-learned)은 이미 확인 → 1번이 나오면 "QCBM>emp_joint 일반화 우위"가 견고한지 확정.
3. 그 전까지는 best 0.8023을 "단일 best, 단 학습은 안정적(15/16 well-learned)"로 정직 보고.

---

## 최종 보고

```markdown
## 작업 요약
- psh 제거 11비트 16 restart를 재학습 없이 로그 분석. 저장물 확인 결과 경우 B(best p_final만, 16 AUC 복원 불가).
  로그 MMD²/TV 분포: 15/16 well-learned(TV<=0.15), 실패 0, best(seed15 TV0.062)는 군집의 좋은 끝(외딴 운 아님).
  단 AUC-vs-emp_joint 초과 restart 수는 p_final 미저장으로 미측정 -> 1회 재실행 필요. 기존 소스 무수정.

## 저장물 확인 (경우 A/B)
- 16개 p_final 저장 여부: 없음(json은 best 요약만, npy/npz 없음).
- 판정: 경우 B (로그 MMD²/TV만 분석 가능).

## 결과 (경우 B) 16 restart 품질 분포
- MMD² min/median/max/std = 6.66e-5 / 3.12e-4 / 2.05e-3 / 4.32e-4
- TV   min/median/max/std = 0.0622 / 0.1086 / 0.2671 / 0.0410
- best: seed15 MMD²6.66e-5 TV0.0622 (json 일치). well-learned(TV<=0.15) 15/16, 실패(TV>0.4) 0/16.
- AUC per restart: 불가(p_final 미저장).

## 사실 관찰 (판정 아님)
- 분포 학습 견고(15/16 well-learned, 실패 0, 좁은 군집) -> best는 외딴 운 아님.
- 그러나 AUC>emp_joint의 restart 견고성은 직접 미확정(AUC≠분포품질 단조). 재실행 필요.

## 생성 파일 / git diff (무변경)
- tools/analyze_no_psh_restarts.py, results2/no_psh_restart_analysis.{json,png}. 기존 소스 무수정.

## 다음 추천 단계
1. 1회 재실행으로 각 restart p_final->AUC 저장 -> emp_joint(0.7773) 초과 restart N/16 직접 측정(tmux, ~15분).
2. 그 결과로 'QCBM>emp_joint 일반화 우위'의 견고성 확정.
```
