# psh 강신호 통제 + 약신호 joint 가치 검증 — joint>marginal이 진짜인가 psh artifact인가

> **전제: 기존 소스 무수정(import/재사용만). QCBM 학습 없음. numpy/scipy/pandas/matplotlib만.**

- **작성일시**: 2026-06-26 10:56 (로컬), 실행 10:58
- **작성 주체**: 로컬 Claude Code agent (조사→스크립트)
- **결론 선요약**: ★ **joint>marginal은 진짜 joint(상호작용) 가치다. psh artifact 아님 — 오히려 psh가 그 가치를 가렸다.** psh 제거 시 joint 이득이 +0.0518 → **+0.2637** 로 폭증(joint 0.777 vs marginal 0.514). marginal(0.874)은 사실상 psh 단독(0.878)과 동일(차 −0.004). emp_joint는 psh 단독 위에 **+0.048**을 상호작용으로 더함. 상호작용은 psh×fwd_iat(+0.044)·fwd×flow(+0.094) 등에 실재.

---

## 1. 작업 목적

직전 진단에서 psh 단독 AUC 0.878 발견 → "12비트 joint(0.9185)가 psh 하나(0.878)보다 +0.04뿐 → joint 가치가 psh artifact 아닌가?"라는 의심. 논문 핵심 라인("joint>marginal")의 생존을 가르는 검증.
(1) psh 제거 통제, (2) joint 순수 기여 분리, (3) 상호작용 위치(쌍별). 빈도 기반(emp_joint/marginal), QCBM 학습 없음. 판정은 웹 AI.

---

## 2. 시작 git status / 보존 대상

```
?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip   (전부 미추적; 추적 변경 0)
```
- 보존 대상 무관 변경 없음. 작업 후에도 기존 파일 무변경(§10).

---

## 3. 재사용 경로 (실제 확인)

- `import experiment2 as E`: `empirical_dist`:119, `auc_score`:266, `size_edges_from_benign`:90, `discretize`:98.
- `import diagnose_iat_joint as DIAG`: `log1p_quantile_edges`(IAT 인코딩).
- `import diagnose_encoding_info_loss as INFO`: `load_split`(seed=7,0.7), `feature_levels`(feature별 양자화 레벨).
- `from train_qcbm_iat_b2 import marginal_dist`: 비트독립 곱(프로젝트 기준선 정의, 0.8741 재현).
- 비트 패킹(MSB→LSB): proto2 syn1 psh1 size4 fwd_iat2 flow_iat2 (build_b2와 동일). psh 제거는 스크립트에서 feature 조립으로 구성(기존 소스 무수정).
- 평가: `s=-log p_benign(x)+1e-12`, attack=positive, `auc_score`(raw MWU). anomaly_eval과 동일 방식.

**정의 주의**: (1)(2) marginal = **비트독립 곱**(기준선). (3) 쌍 marginal = **두 feature 독립 곱**(feature-level, 두 feature 상호작용 분리). 둘 다 .md/json에 명시.

---

## 4. 정합 검증

- 12비트 재계산: emp_joint **0.9258**, marginal **0.8741**, 점유 **94** → 기대(0.9258/0.8741/94)와 **완전 일치** ✅ (불일치 시 중단; 통과). 평가/인코딩 재현 정확.

---

## 5. (1) psh 제거 통제

| 인코딩 | marginal AUC | emp_joint AUC | **joint 이득** | 점유 |
|--------|--------------|----------------|----------------|------|
| 12비트 (psh 포함) | 0.8741 | 0.9258 | +0.0518 | 94/4096 |
| **11비트 (psh 제거)** | **0.5136** | **0.7773** | **+0.2637** | 90/2048 |

- psh 제거 시 **marginal이 0.514(거의 무작위)로 붕괴** → marginal의 0.874는 거의 전부 psh 덕.
- 그러나 **emp_joint는 0.777 유지**, joint 이득이 **+0.0518 → +0.2637 (5배 폭증)**.
- → **joint 가치는 진짜이고, psh가 그것을 가리고 있었다.** psh(쉬운 신호)가 marginal·joint를 모두 높여 격차를 +0.05로 압축했을 뿐, psh를 빼면 joint(상호작용)가 탐지를 떠받침.

---

## 6. (2) joint 순수 기여 (12비트)

| 모델 | AUC |
|------|-----|
| psh 단독 | 0.8779 |
| marginal (비트독립) | 0.8741 |
| emp_joint | 0.9258 |

- **emp_joint − marginal = +0.0518** (상호작용이 독립가정 위에 더하는 값).
- **emp_joint − psh단독 = +0.0480** (joint가 "psh 하나"보다 더하는 값 — 순수 추가 가치).
- **marginal − psh단독 = −0.0038** → **marginal ≈ psh 단독**. marginal의 검출력은 사실상 psh 하나.
- → "12비트 joint가 psh보다 +0.04뿐"의 그 +0.04는 **psh로는 못 얻는 상호작용 가치**(artifact 아님).

---

## 7. (3) 쌍별 상호작용 (joint vs 두 feature 독립곱)

| 쌍 | joint AUC | marginal AUC | 이득 |
|----|-----------|--------------|------|
| **proto×size** | 0.7522 | 0.6319 | **+0.1204** |
| **fwd_iat×flow_iat** | 0.6501 | 0.5558 | **+0.0944** |
| **psh×fwd_iat** | 0.9366 | 0.8926 | **+0.0440** |
| psh×flow_iat | 0.9116 | 0.8823 | +0.0293 |
| psh×size | 0.8784 | 0.8791 | −0.0007 |
| size×fwd_iat | 0.5422 | 0.6366 | −0.0944 |
| size×flow_iat | 0.4984 | 0.6375 | −0.1391 |

- 최대 이득: **proto×size (+0.120)** — 단, proto는 단독 거의 죽음(AUC 0.497, 거의 상수 TCP)이라 희소 셀(드문 proto값×size) 과적합 정황 가능 → 해석 주의.
- **해석 견고한 상호작용**: **psh×fwd_iat (+0.044)**(시간×플래그), **fwd_iat×flow_iat (+0.094)**(두 시간축 상호작용). 살아있는 feature 간 진짜 상호작용.
- 음의 이득(size×fwd/flow −0.09~−0.14): 그 쌍에선 결합 경험분포가 독립곱보다 못함(희소 셀/반정보) — joint가 항상 이기는 건 아님(정직 보고).

---

## 8. 사실 관찰 (판정 아님 — 웹 AI)

1. **psh 제거 시 joint 이득 유지·폭증**(+0.0518 → +0.2637). joint>marginal은 psh artifact가 **아님**.
2. **marginal ≈ psh 단독**(차 −0.004): marginal 기준선의 힘은 사실상 psh 하나에서 옴.
3. **joint가 psh 단독에 더하는 값 = +0.048**: 쉬운 신호(psh)로는 못 얻는 상호작용 가치를 joint가 포착.
4. **상호작용 위치**: psh×fwd_iat(+0.044), fwd_iat×flow_iat(+0.094)가 해석 견고한 진짜 상호작용. proto×size(+0.120)는 최대지만 proto 사망으로 희소 artifact 가능.
5. 종합 정황(사실): "joint>marginal"의 +0.04~0.05는 **상호작용 실체**. 다만 psh라는 강신호가 절대 AUC를 0.87+로 끌어올려 격차를 작아 보이게 함. **psh를 통제하면 joint의 우위가 +0.26로 드러남** → 논문 핵심 라인 견고.

---

## 9. 확실 / 추정 / 확인 불가

- **확실**: 12비트 정합(0.9258/0.8741/94), 위 모든 AUC·이득(결정론 빈도 계산).
- **추정**: proto×size +0.120은 proto 사망(거의 상수)으로 희소 셀 과적합 가능 — train 빈도라 test 일반화는 별개.
- **확인 불가(범위 밖)**: 이 상호작용을 QCBM이 학습으로 얼마나 재현하는지(이미 B2에서 emp_joint의 99.2% 실현은 측정됨). 11비트(psh 제거)에서 QCBM 재학습 시 joint 0.777 근접 여부(미측정).

---

## 10. 생성 파일 / 실행 / git

- `results2/psh_control_joint.json`: verify_12bit, control_remove_psh(12/11비트), joint_contribution, pairwise_interaction(7쌍), max_interaction_pair.
- `results2/psh_control_joint.png`: (좌)psh有/無 marginal·emp_joint + psh단독선, (우)쌍별 이득 막대.
- `tools/diagnose_psh_control_joint.py` (신규).
```bash
python3 -m py_compile tools/diagnose_psh_control_joint.py   # PY_COMPILE_OK
python3 -u tools/diagnose_psh_control_joint.py              # DONE, 12비트 정합 통과
```
- `git diff --stat` 비어 있음(미추적). experiment2.py/train_qcbm_iat_b2.py/diagnose_*.py **무수정**(import만). QCBM 무학습, 무설치.

---

## 11. 성공 기준 / 남은 불확실성 / 다음 추천

충족: 12비트 정합(0.9258/0.8741) ✅, (1) psh 제거 11비트 결과 ✅, (2) marginal/emp/psh단독 + 두 차이 ✅, (3) 쌍별 이득 + 최대쌍 ✅, "psh 제거 시 joint 이득 유지" 사실 ✅, json/png ✅, 무수정·무학습·무설치 ✅.

불확실: proto×size 희소 artifact 여부(test 일반화 미측정). 11비트에서 QCBM 학습 재현도 미측정.

다음 추천:
1. (핵심 서사) **joint>marginal은 진짜 상호작용 가치이며 psh가 그것을 가린다 — psh 통제 시 +0.26**를 논문 핵심 결과로. "쉬운 강신호가 있을 때 joint의 진가가 가려진다"는 메시지가 강력.
2. (보강) **11비트(psh 제거) 인코딩으로 QCBM 재학습**해 모델이 joint 0.777(상한)을 재현하는지 측정 → "QCBM이 psh 없는 순수 상호작용을 학습한다"가 입증되면 양자 라인까지 직접 보강(tmux 학습).
3. (선택) proto×size를 test 분할로 재확인(희소 artifact 배제), 쌍별 분석을 test 기준으로 재측정.

---

## 최종 보고

```markdown
## 작업 요약
- "joint>marginal이 psh artifact인가" 검증. 결과: 아님 — psh가 오히려 joint 가치를 가림.
  psh 제거 시 joint 이득 +0.0518 -> +0.2637(joint 0.777 vs marginal 0.514). marginal≈psh단독(차 -0.004).
  emp_joint는 psh단독 위에 +0.048을 상호작용으로 더함. 진짜 상호작용: psh×fwd_iat(+0.044), fwd×flow(+0.094).
  기존 소스 무수정, QCBM 무학습.

## 정합 검증 (12비트 = 0.9258/0.8741?)
- emp_joint 0.9258, marginal 0.8741, 점유 94 -> 기대와 완전 일치.

## (1) psh 제거 통제
| 인코딩 | marginal | emp_joint | joint이득 |
| 12비트(psh포함) | 0.8741 | 0.9258 | +0.0518 |
| 11비트(psh제거) | 0.5136 | 0.7773 | +0.2637 |

## (2) joint 순수 기여
- marginal 0.8741 / emp_joint 0.9258 / psh단독 0.8779
- emp-marginal +0.0518 | emp-psh단독 +0.0480 | marginal-psh단독 -0.0038 (marginal≈psh)

## (3) 쌍별 상호작용 이득 (top)
| 쌍 | joint | marginal | 이득 |
| proto×size | 0.7522 | 0.6319 | +0.1204 (proto 사망->희소 주의) |
| fwd_iat×flow_iat | 0.6501 | 0.5558 | +0.0944 |
| psh×fwd_iat | 0.9366 | 0.8926 | +0.0440 |
| size×flow_iat | 0.4984 | 0.6375 | -0.1391 (joint가 짐) |

## 사실 관찰 (판정 아님)
- psh 제거 시 joint 이득 +0.0518->+0.2637 (유지·폭증). joint는 psh artifact 아님, 오히려 psh가 가림.
- marginal의 힘은 사실상 psh 하나(marginal-psh단독 -0.004). joint가 psh 위에 +0.048 추가.
- 진짜 상호작용: psh×시간(fwd_iat), 두 시간축(fwd×flow). 일부 쌍은 joint가 짐(정직).

## 생성 파일 / git diff (무변경)
- tools/diagnose_psh_control_joint.py, results2/psh_control_joint.{json,png}. 기존 소스 무수정.

## 다음 추천 단계
1. 핵심 서사: "joint>marginal은 진짜 상호작용 가치, psh가 가림(통제 시 +0.26)"를 논문 결과로.
2. 11비트(psh 제거) QCBM 재학습으로 모델이 joint 0.777을 재현하는지 측정(양자 라인 보강, tmux).
```
