# 양자 justification 2단계 — B2(12q, 4096상태) 파라미터 효율 + 공정성 검증(NMF·음수량·MMD)

> **전제: 기존 소스 무수정(import/재사용만). QCBM 재학습 없음. numpy/scipy만(새 dependency 설치 없음).**

- **작성일시**: 2026-06-26 10:39 (로컬), 실행 10:40
- **작성 주체**: 로컬 Claude Code agent (조사→스크립트)
- **결론 선요약**: ★ **B2도 양자 파라미터 효율 우위 없음(B1 확증). 공정성 검증 3종 모두 비교가 공정했음을 확인** — NMF(비음수)도 SVD만큼 효율적, SVD 음수질량 무시 가능(<0.07%), MMD 기준도 TV와 동일 결론. 고전이 더 적은 파라미터로 QCBM(216p)의 TV/MMD를 능가.

---

## 1. 작업 목적

1단계(B1)에서 고전(저랭크 128p/비트그룹 134p)이 QCBM(180p)보다 효율적이었다. 이번엔 (1) **B2(4096상태)로 스케일 확장** 시에도 같은가, (2) **1단계 비교가 공정했나**를 3가지로 검증: (a) NMF(비음수 분해)도 SVD만큼 효율적인가, (b) SVD가 clip 전 음수 질량을 얼마나 만들었나, (c) TV뿐 아니라 MMD(QCBM 학습 척도)로도 결론이 같은가. 판정은 웹 AI.

---

## 2. 시작 git status / 보존 대상

```
?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip   (전부 미추적; 추적 변경 0)
```
- 보존 대상 무관한 변경 없음. 작업 후에도 기존 파일 무변경(§13).

---

## 3. 재사용 경로 (실제 확인)

- `from experiment2 import tv_dist(:160), build_kernel(:127), empirical_dist(:119)` — 새 구현 안 함.
- `import train_qcbm_iat_b2 as TB2; d = TB2.build_b2()` ([train_qcbm_iat_b2.py:94](../../qcbm/tools/train_qcbm_iat_b2.py#L94)) — 동일 seed=7/Fwd·Flow IAT log1p 분위수/MSB→LSB 패킹으로 q_train(4096) 재현. main 가드 있어 import 안전.
- 1단계 정의(저랭크 64r/비트그룹 2^b−1, clip+정규화) 동일 체계로 확장(저랭크 128r=64×r+r×64).
- **sklearn 없음** → NMF는 **numpy 곱셈 업데이트**(400 iter × 3 init, Frobenius 최소 선택) 직접 구현.
- QCBM 점: `results2/qcbm_iat_b2.json`에서 (216p, TV 0.093, MMD² 1.887e-4) 읽음. 재학습 없음.

---

## 4. q_train 재현 검증

- 재현 q_train: 길이 **4096**, **점유 94** → json `occupied_train_states=94`와 **일치** ✅ (불일치 시 중단 구현; 통과).

---

## 5. 고전 baseline 정의 + 공정성 점검 정의

- **(A) SVD 저랭크**: q(4096) → **64×64** → 랭크 r 근사 → `clip(0)`+정규화. **params=128r**.
- **(a) NMF 저랭크**: 같은 64×64 → 비음수 분해 W·H(≥0) → 정규화(클립 무효과). **params=128r**. SVD와 같은 r에서 나란히 비교 → "음수 후처리가 고전을 부당하게 도왔나" 점검.
- **(b) SVD 음수량**: 각 r에서 clip 전 근사의 음수 질량(|Σ 음수|)과 음수 셀 수 기록.
- **(B) 비트그룹**: 12비트 연속 그룹, 그룹내 joint×그룹간 독립. **params=Σ(2^{b_g}−1)**.
- **(c) MMD 평가**: 모든 고전 근사에 대해 `build_kernel(12)`로 MMD²=(p−q)K(p−q) 계산. QCBM은 json 저장 MMD²(p_final 미저장이라 재계산 불가; **고전 MMD²는 동일 K로 새 계산**).
- 끝점: marginal(1bit×12, 12p), emp_joint(점유 94, TV0); QCBM(216p).

---

## 6. 결과 표 (B2: 파라미터 vs TV vs MMD²)

| 모델/구성 | 파라미터 | TV | MMD² |
|-----------|----------|-----|------|
| marginal (1bit×12) | 12 | 0.7527 | 1.09e-2 |
| 비트그룹 3bit×4 | 28 | 0.5056 | 6.06e-3 |
| **비트그룹 4bit×3** | **45** | **0.0201** | **1.87e-5** |
| **SVD r=1** | **128** | **0.0794** | **1.80e-4** |
| **NMF r=1** | **128** | **0.0794** | **1.80e-4** |
| 비트그룹 6+6 | 126 | 0.0797 | 1.79e-4 |
| **QCBM (B2)** | **216** | **0.0930** | **1.89e-4**(저장) |
| SVD r=2 | 256 | 0.0496 | 5.66e-5 |
| NMF r=2 | 256 | 0.0498 | 5.66e-5 |
| 비트그룹 8+4 | 270 | 0.0110 | 1.65e-5 |
| SVD r=6 | 768 | 0.00008 | 2.0e-9 |
| emp_joint (점유) | 94 | 0.0 | 0.0 |

(전체 곡선 r=1..64는 json `svd.curve`/`nmf.curve` 참조.)

---

## 7. SVD 음수 질량 / 셀 수 (r별, clip 전)

| r | 음수질량(\|Σ음수\|) | 음수셀 수 |
|---|---------------------|-----------|
| 1 | **0.0000** | **0** |
| 2 | 0.0004 | 85 |
| 3 | 0.0007 | 97 |
| 4 | 0.0004 | 93 |
| 5 | 0.0002 | 79 |
| 6 | 0.0000 | 93 |

→ 음수 질량이 **전체(=1)의 0.07% 이하**로 무시 가능. 특히 **r=1은 음수 0** → clip이 결과에 영향 없음. SVD의 clip 후처리가 고전을 부당하게 돕지 않았음(공정).

---

## 8. 사실 관찰 (판정 아님 — 웹 AI)

- **QCBM TV(0.093, 216p) 도달 고전 파라미터**: SVD **128p**(r=1), NMF **128p**(r=1), **비트그룹 45p**(4bit×3, TV 0.020). 셋 다 216보다 **적고**, 비트그룹은 **약 1/5 파라미터로 4.6배 낮은 TV**.
- **(a) NMF vs SVD**: r=1에서 **완전히 동일**(둘 다 TV 0.0794, 음수 0). r=2도 거의 동일(0.0496 vs 0.0498). 즉 **비음수 제약을 줘도 고전 우위 그대로** → 양자가 "유효분포 제약" 덕에 공정해지는 것 아님.
- **(b) SVD 음수량**: 미미(<0.07%, r=1은 0). clip이 고전을 부풀리지 않음.
- **(c) MMD 기준**: 결론 동일. SVD r=1(128p) MMD² 1.80e-4 ≈ QCBM 1.89e-4(이미 더 적은 파라미터로 동급), 비트그룹 4bit×3(45p) MMD² 1.87e-5(QCBM의 1/10). **TV든 MMD든 고전이 더 효율적.**
- 배경: q_train이 4096상태 중 **94개만 점유(2.3%)** 한 매우 희소·분해가능 분포. 특히 4bit×3 그룹 분해(45p)가 잘 맞음(feature 블록 구조와 부분 정렬 추정). 상태공간을 4배(1024→4096) 키워도 **고전 우위 유지·강화**.

---

## 9. B1 vs B2 비교표

| 항목 | B1 (10q, 1024상태) | B2 (12q, 4096상태) |
|------|---------------------|---------------------|
| 점유 상태 | 79 (7.7%) | 94 (2.3%) |
| QCBM | 180p / TV 0.0369 | 216p / TV 0.0930 |
| 고전 SVD가 QCBM TV 도달 | 128p (r=2) | **128p (r=1)** |
| 고전 비트그룹이 QCBM TV 도달 | 134p (7+3) | **45p (4bit×3)** |
| emp_joint(점유 저장) | 79p / TV0 | 94p / TV0 |
| 양자 효율 우위 | 없음 | 없음(더 뚜렷) |

→ **스케일을 키울수록 양자 우위가 생기기는커녕, 고전이 더 적은 파라미터(B2 비트그룹 45p)로 능가.** 희소·분해가능 분포에선 고전 분해가 본질적으로 효율적.

---

## 10. 확실 / 추정 / 확인 불가

- **확실**: 위 (params, TV, MMD²) 수치(닫힌형 SVD + 결정론 빈도 + NMF best-of-3). 점유 94 일치. SVD 음수량.
- **추정**: 더 큰/고랭크/비분해 분포에서는 다를 수 있음(미검증). NMF는 비볼록이라 전역최적 보장 없음(3 init 중 best 사용; frob_err 기록).
- **확인 불가**: QCBM MMD²는 저장 스칼라(p_final 미저장이라 동일 파이프라인 재계산 불가) — 단, json MMD²는 학습에 쓴 동일 커널 산출이라 고전 MMD²와 비교 가능.

---

## 11. 생성 파일

- `results2/param_efficiency_b2.json`: svd/nmf/bitgroup curve(각 params·tv·mmd2, svd는 neg_mass/neg_cells), qcbm_point, emp_joint, qcbm_tv_reach(svd128/nmf128/bitgroup45), b1_compare, 메타.
- `results2/param_efficiency_b2.png`: TV 패널 + MMD² 패널. SVD/NMF/비트그룹 곡선 + QCBM★(216) + emp_joint◆(94) + QCBM 수평선.
- `tools/diagnose_param_efficiency_b2.py` (신규).

---

## 12. 실행 명령과 결과

```bash
git status                                                      # ?? docs ?? qcbm ?? zip
python3 -c "import sklearn" -> 없음 -> numpy NMF로 대체
python3 -m py_compile tools/diagnose_param_efficiency_b2.py     # PY_COMPILE_OK
python3 -u tools/diagnose_param_efficiency_b2.py                # DONE; 점유 94 검증 통과
```
- 실패 없음. SVD+NMF(작은 64×64)+빈도+4096² 커널 MMD, 분 단위 이내 완료. **QCBM 재학습 없음**, dependency 설치 없음.

---

## 13. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → `git diff --stat` 비어 있음. experiment2.py / train_qcbm_iat_b2.py / diagnose_param_efficiency_b1.py 등 **무수정**(import만).
- 신규: `tools/diagnose_param_efficiency_b2.py`, `results2/param_efficiency_b2.{json,png}`, 본 `.md`.

---

## 14. 성공 기준 충족 / 남은 불확실성 / 다음 추천

충족: q_train 점유 94 검증 ✅, SVD/NMF/비트그룹 (params,TV,MMD²) 곡선 ✅, SVD 음수량 r별 ✅, QCBM·marginal·emp_joint 포함 ✅, QCBM TV 도달 파라미터(SVD128/NMF128/비트45) ✅, NMF·음수·MMD 사실 기록 ✅, B1 vs B2 표 ✅, json/png 저장 ✅, 무수정·무재학습·무설치 ✅.

불확실:
1. 결과는 이 **희소·분해가능 분포** 한정. 고랭크/비분해 분포에서는 역전 가능(미검증).
2. 파라미터 카운팅 관례(저랭크 128r, 비트그룹 2^b−1, emp 점유수) 선택에 따라 절대 비교 달라질 수 있음(정의 명시함).
3. "표현 효율"은 train 적합 기준. 일반화/이상탐지 성능당 효율은 별도 축.

다음 추천:
1. (서사) **B1·B2 모두 양자 파라미터 효율 우위 없음**을 정직 보고. 양자 정당성 라인은 (표현 효율 ✗) → **정확도(joint>marginal, 시간축 상호작용 학습 ✓, 이미 입증)** 로 재정렬 권고 — 웹 AI 판단.
2. (선택) 양자 우위 가능성이 있는 **고랭크/비분해 분포**(다른 공격·feature, 또는 의도적으로 상관 강한 인코딩)에서 고전 분해가 무너지는지 탐색.
3. (선택) 일반화 관점(test/OOD 이상탐지 AUC 당 파라미터) 효율 비교.

---

## 최종 보고

```markdown
## 작업 요약
- B2(12q,4096상태)에서 고전 SVD/NMF/비트그룹의 params vs TV/MMD²를 재서 QCBM(216p,TV0.093,MMD²1.89e-4)과 비교
  + 공정성 3종(NMF/음수량/MMD) 검증. 결과: 고전 SVD/NMF 128p(r=1), 비트그룹 45p가 QCBM TV를 더 적은 파라미터로 능가.
  공정성 검증 통과(NMF=SVD, SVD음수<0.07%, MMD결론동일). B1 확증: 양자 효율 우위 없음(더 뚜렷). 무수정/무재학습/무설치.

## q_train 재현 검증 (기대 점유 94)
- 점유 94, 4096차원 일치 ✅.

## 결과 표 (B2: 파라미터 vs TV vs MMD²)
| 모델/구성 | 파라미터 | TV | MMD² |
| marginal | 12 | 0.7527 | 1.09e-2 |
| 비트그룹 4bit×3 | 45 | 0.0201 | 1.87e-5 |
| SVD r=1 | 128 | 0.0794 | 1.80e-4 |
| NMF r=1 | 128 | 0.0794 | 1.80e-4 |
| QCBM (B2) | 216 | 0.0930 | 1.89e-4 |
| emp_joint | 94(점유) | 0.0 | 0.0 |

## 공정성 점검 결과
- NMF vs SVD: r=1 완전 동일(음수 0). 비음수 제약 줘도 고전 우위 그대로.
- SVD 음수질량: <0.07%(r=1은 0). clip이 고전을 부풀리지 않음.
- MMD 기준: TV와 동일 결론(SVD r=1 128p가 QCBM MMD² 동급, 비트그룹 45p가 1/10).

## B1 vs B2
- B1: SVD 128p / 비트그룹 134p가 QCBM(180p) 도달. B2: SVD 128p / 비트그룹 45p가 QCBM(216p) 도달. 둘 다 우위 없음, B2가 더 뚜렷.

## 생성 파일
- tools/diagnose_param_efficiency_b2.py, results2/param_efficiency_b2.{json,png}

## git diff --stat
- 비어 있음(미추적). 기존 소스 무수정(import만).

## 다음 추천 단계
1. 양자 정당성을 '표현효율'에서 '정확도(joint>marginal, 시간축 상호작용)'로 재정렬 정직 보고(웹 AI 판단).
2. 고랭크/비분해 분포에서 고전 분해가 무너지는지 탐색(양자 우위 후보 영역).
```
