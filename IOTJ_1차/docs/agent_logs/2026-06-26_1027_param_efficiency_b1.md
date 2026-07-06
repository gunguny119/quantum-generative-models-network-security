# 양자 justification 1단계 — B1(10q): 고전 생성모델 2종 vs QCBM 파라미터 효율

> **전제: 기존 소스 무수정(import/재사용만). QCBM 재학습 없음. numpy/scipy만 사용(새 dependency 없음).**

- **작성일시**: 2026-06-26 10:27 (로컬), 실행 10:30
- **작성 주체**: 로컬 Claude Code agent (조사→스크립트)
- **결론 선요약**: ★ **이 B1 규모(1024상태)에서는 양자 파라미터 효율 우위 없음.** 고전 저랭크(r=2, **128 params**)·비트그룹(7+3, **134 params**) 모두 QCBM의 TV(0.037@180)를 **더 적은 파라미터로** 달성(그리고 더 낮은 TV).

---

## 1. 작업 목적

QCBM은 B1에서 180 파라미터로 TV=0.03694를 냄. AUC만 보면 emp_joint(0.8856) > QCBM(0.8811)이라 "왜 양자"가 정당화 안 됨.
양자 정당성은 정확도가 아니라 **표현 효율**(같은 분포를 더 적은 파라미터로). 질문: **고전이 같은 q_train을 같은 TV로 표현하려면 파라미터가 몇 개 필요한가? 180보다 많은가?**
많으면 양자 효율 우위, 적으면 정직하게 "이 규모 우위 없음". 이번은 B1만.

---

## 2. 시작 git status / 보존 대상

```
?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip   (모두 미추적; 추적 변경 0)
```
- 보존 대상 무관한 변경 없음. 작업 후에도 기존 파일 무변경(§11).

---

## 3. 재사용 경로 (실제 확인)

- **TV**: `from experiment2 import tv_dist` (= `0.5*Σ|p−q|`, [experiment2.py:160](../../qcbm/experiment2.py#L160)). 새 구현 안 함(정의 불일치 방지).
- **q_train**: `import train_qcbm_iat_b1 as TB1; d = TB1.build_b1()` ([train_qcbm_iat_b1.py:98](../../qcbm/tools/train_qcbm_iat_b1.py#L98)). main 가드 있어 import 시 학습 안 돎. 동일 seed=7 분할·동일 Fwd IAT log1p 분위수 경계·동일 비트 패킹으로 q_train(1024) 정확 재현.
- **QCBM 점**: `results2/qcbm_iat_b1.json`에서 (nparams 180, best_tv 0.03694) 읽음. 재학습 없음.

---

## 4. q_train 재현 검증

- 재현된 q_train: 길이 **1024**, **점유 79** → json `occupied_train_states=79`와 **일치**. ✅ (불일치 시 중단하도록 구현; 통과.)

---

## 5. 고전 baseline 정의와 파라미터 카운팅 (보수적 = 고전에 유리하게)

- **(A) 저랭크 행렬 분해**: q_train(1024) → **32×32 reshape** → SVD 랭크 r 근사 → `clip(0,None)` 후 재정규화(QCBM p_final 처리와 동일). **params = 64r** (= 32×r + r×32, 전역 스케일/정규화 자유도 차감 없이 단순·보수적).
- **(B) 비트그룹 분해**: 10비트(bit0..bit9)를 연속 그룹으로 분할, 그룹 내부는 경험적 joint, 그룹 간 독립(곱). **params = Σ_g (2^{b_g} − 1)**.
- **emp_joint**: q_train 자체(TV=0). params는 **저장할 비0 셀 수 = 점유 79**로 보수 카운트(생성식 일반 카운트 2^10−1=1023도 병기).
- marginal = 비트그룹 전부 1비트(params 10).

---

## 6. 결과 표 (파라미터 수 vs TV)

| 모델/구성 | 파라미터 수 | TV(vs q_train) |
|-----------|-------------|----------------|
| **marginal** (1bit×10) | 10 | 0.08530 |
| 비트그룹 2bit×5 | 15 | 0.08424 |
| 비트그룹 3+3+4 | 29 | 0.06707 |
| 비트그룹 5+5 | 62 | 0.07870 |
| 저랭크 r=1 | 64 | 0.07656 |
| **저랭크 r=2** | **128** | **0.00550** |
| 비트그룹 **7+3** | **134** | **0.01116** |
| **QCBM (B1)** | **180** | **0.03694** |
| 저랭크 r=3 | 192 | 0.00098 |
| 비트그룹 8+2 | 258 | 0.01101 |
| 비트그룹 full(1×10) | 1023 | 0.0 |
| **emp_joint** (점유 저장) | **79** | **0.0** |

(저랭크 r=4~32는 TV≈0; 전체 곡선은 json `lowrank.curve` 참조.)

---

## 7. 사실 관찰 (판정 아님 — 웹 AI)

- **QCBM TV(0.03694, 180 params) 도달에 필요한 고전 파라미터**:
  - 저랭크: **128개**(r=2, TV 0.00550) — 180보다 **적고**, TV는 훨씬 낮음.
  - 비트그룹: **134개**(7+3, TV 0.01116) — 180보다 **적고**, TV 더 낮음.
- 즉 **이 B1 규모에서는 고전 두 모델 모두 QCBM보다 적은 파라미터로 같은(또는 더 좋은) TV를 달성**. 양자의 파라미터 효율 우위 **관측 안 됨**.
- 더 극단적으로, **emp_joint는 79개(점유 셀)만으로 TV=0**(정확) — QCBM 180/TV 0.037보다 적고 정확. (단 emp_joint는 일반화/평활화가 없는 암기 분포라는 한계는 별개 논점.)
- 배경: q_train이 1024상태 중 79개만 점유한 **희소·저랭크적 분포**라, 저랭크/그룹 분해가 매우 효율적. 이 구조적 단순성이 고전에 유리하게 작용.
- **함의(사실)**: B1(1024상태) 규모에서는 "표현 효율" 기반 양자 정당화가 성립하지 않음. 양자 우위 후보는 (a) 분포가 고랭크/비분해적이거나 (b) 상태공간이 훨씬 커서 고전 분해 파라미터가 폭증하는 더 큰 규모에서 찾아야 할 가능성.

---

## 8. 확실 / 추정 / 확인 불가

- **확실**: 위 (params, TV) 수치(닫힌형 SVD + 경험적 빈도, 결정론적). q_train 점유 79 일치.
- **추정**: 더 큰 규모(B2 4096, 또는 그 이상)에서는 고전 분해 비용이 더 빨리 커질 수 있음(미검증).
- **확인 불가(이번 범위 밖)**: QCBM이 더 적은 L/qubit로 같은 TV를 낼 수 있는지(아키텍처 최적화), 일반화(test/OOD) 관점 효율.

---

## 9. 생성 파일

- `results2/param_efficiency_b1.json`: qcbm_point, auc_ref, lowrank.curve(r=1..32), bitgroup.curve(7구성), emp_joint, qcbm_tv_reach(저랭크 128/비트그룹 134), tv_metric/param_def 명시.
- `results2/param_efficiency_b1.png`: x=params(log), y=TV(log). 저랭크 곡선 + 비트그룹 곡선 + QCBM★(180) + QCBM TV 수평선 + emp_joint◆(79).
- `tools/diagnose_param_efficiency_b1.py` (신규 스크립트).

---

## 10. 실행 명령과 결과

```bash
git status                                                      # ?? docs ?? qcbm ?? zip
python3 -m py_compile tools/diagnose_param_efficiency_b1.py     # PY_COMPILE_OK
python3 -u tools/diagnose_param_efficiency_b1.py                # DONE; q_train 점유 79 검증 통과
python3 -c "json.load(param_efficiency_b1.json) reach 128/134"  # 확인
```
- 실패 없음. 빠르게 종료(SVD+빈도). **QCBM 재학습 없음**(json 값 사용). 새 dependency 없음(numpy/scipy/matplotlib만).

---

## 11. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → `git diff --stat` 비어 있음. experiment2.py / train_qcbm_iat_b1.py 등 **무수정**(import만).
- 신규: `tools/diagnose_param_efficiency_b1.py`, `results2/param_efficiency_b1.{json,png}`, 본 `.md`.

---

## 12. 성공 기준 충족

- [x] q_train 재현 + 점유 79 검증.
- [x] 고전 2종 (params, TV) 곡선 생성.
- [x] QCBM 점(180, 0.037) + marginal(10) + emp_joint(79) 그래프 포함.
- [x] "고전이 TV 0.037 도달에 필요한 파라미터" 사실 기록(저랭크 128 / 비트그룹 134).
- [x] json/png 저장, 기존 소스 무수정, QCBM 무재학습, dependency 무변경.

---

## 13. 남은 불확실성

1. 결과가 **B1(1024) 한정**. 희소·저랭크 분포라 고전에 유리. 더 큰 상태공간에서 역전 가능성은 미검증.
2. 파라미터 카운팅 관례(저랭크 64r, 비트그룹 2^b−1, emp_joint 점유수)에 따라 절대 비교가 달라질 수 있음 — 정의를 명시했으나 관례 선택은 논쟁 여지.
3. "표현 효율"은 train 분포 적합 기준. **일반화/이상탐지 성능당 효율**은 다른 축(별도 측정 필요).

---

## 14. 다음 추천 단계

1. **B2(4096상태)로 동일 분석 확장**: 64×64 저랭크 + 비트그룹(2^b−1) vs QCBM 216 params(TV 0.093). 상태공간 4배에서 고전 분해 비용이 QCBM 대비 어떻게 변하는지 — **양자 우위는 규모에서 나올 수 있으므로 핵심**.
2. (서사) B1에서 양자 효율 우위가 없음을 정직 보고하고, 연구 초점을 **(a) 더 큰 규모 스케일링** 또는 **(b) 정확도(joint>marginal, 이미 입증)** 로 재정렬 — 웹 AI 판단.
3. (선택) 고랭크/비분해 분포(예: 다른 공격·feature)에서 고전 분해가 무너지는 사례 탐색.

---

## 최종 보고

```markdown
## 작업 요약
- B1(10q,1024상태)에서 고전 생성모델 2종(저랭크 SVD, 비트그룹)의 파라미터 vs TV를 재서 QCBM(180p, TV0.037)과 비교.
  결과: 고전 저랭크 128p / 비트그룹 134p 가 QCBM TV에 더 적은 파라미터로 도달(그리고 더 낮은 TV).
  -> 이 규모에선 양자 파라미터 효율 우위 없음(정직 보고). 기존 소스 무수정, QCBM 무재학습.

## q_train 재현 검증
- 점유 상태수 79 (기대 79 일치). 1024차원.

## 고전 baseline 정의와 파라미터 카운팅
- 저랭크: 32x32 SVD 랭크 r, params=64r (보수적).
- 비트그룹: 연속비트 그룹 내부 joint x 그룹간 독립, params=sum(2^b-1).
- emp_joint: 점유 셀수 79 (TV0); 일반카운트 2^10-1=1023 병기.

## 결과 표 (파라미터 수 vs TV)
| 모델/구성 | 파라미터 | TV |
| marginal | 10 | 0.0853 |
| 저랭크 r=2 | 128 | 0.0055 |
| 비트그룹 7+3 | 134 | 0.0112 |
| QCBM (B1) | 180 | 0.0369 |
| 저랭크 r=3 | 192 | 0.0010 |
| emp_joint | 79(점유) | 0.0 |

## 사실 관찰 (판정 아님)
- QCBM TV(0.037,180p) 도달 고전 파라미터: 저랭크 128, 비트그룹 134 (둘 다 <180, TV도 더 낮음).
- q_train이 1024중 79점유의 희소·저랭크 분포라 고전 분해가 매우 효율적. B1 규모 양자 효율 우위 없음.

## 생성 파일
- results2/param_efficiency_b1.json, results2/param_efficiency_b1.png, tools/diagnose_param_efficiency_b1.py

## git diff --stat (기존 소스 무변경)
- 비어 있음(qcbm 미추적). experiment2.py/train_qcbm_iat_b1.py 무수정(import만).

## 다음 추천 단계
1. B2(4096)로 확장 — 상태공간 4배에서 고전 분해 비용 증가율 vs QCBM 216p 비교(양자 우위는 규모에서?).
2. B1 우위 없음 정직 보고 + 서사를 스케일링 또는 정확도(joint>marginal)로 재정렬.
```
