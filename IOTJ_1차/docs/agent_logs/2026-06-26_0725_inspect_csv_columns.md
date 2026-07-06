# CSE-CIC-IDS2018 CSV 컬럼 사전 조사 — feature 확장 후보 발굴 (조사 전용)

- **작성일시**: 2026-06-26 07:25 (로컬), 실행 07:2x
- **작성 주체**: 로컬 Claude Code agent (default mode)
- **작업 성격**: 진단 스크립트 1개 + 리포트(json/csv) 생성. **학습 없음. 기존 소스 무수정.**

---

## 1. 작업 목적

현재 QCBM 8비트 인코딩(proto2+syn1+psh1+size4)의 4 feature로는 Infiltration AUC가 0.43
(분포기반 상한 emp_joint도 0.43)으로, **"이 4 feature에 공격을 가를 정보가 없음"** 이 측정 확정됨
([classical_baseline 로그](2026-06-26_0710_classical_baseline_auc.md) 참조).
feature를 10~12비트로 늘리기 전에, CSV의 **어떤 컬럼이 benign vs infilteration을 잘 가르고**
이산화에 적합한지(고유값 수/분포)를 학습 없이 통계만으로 조사.

분리지표 = `|mean_ben − mean_atk| / (pooled_std + eps)` (표준화 평균차; 후보 발굴용, 절대 진리 아님).

---

## 2. 시작 시점 git status 요약

```
?? docs/    (이전 로그들 + 본 로그)
?? qcbm/    (QCBM 코드/결과 전체가 여전히 미추적)
git diff --stat: 비어 있음 (추적 파일 변경 0)
```
- 본 작업이 추가한 파일: `qcbm/tools/inspect_csv_columns.py`, `qcbm/results2/csv_columns_report.json`, `qcbm/results2/csv_columns_report.csv`, 본 `.md`.
- **experiment2.py 등 기존 소스는 읽기만 함 (무수정).** `load_frames()`/`proto_code()` 라벨 처리 확인: `Label.astype(str).str.strip().str.lower()` → `benign`/`infilteration` ([experiment2.py:83-85](../../qcbm/experiment2.py#L83-L85)).

---

## 3. 전체 컬럼 수와 목록

- **행=331,125, 컬럼=80.**
- **라벨 분포**: `benign=238,037`, `infilteration=93,063`, **`label`=25** ← CSV 중간에 헤더가 다시 끼어든 **오염 행 25개**. 이 때문에 모든 수치 컬럼의 pandas dtype이 `object`로 잡힘(스크립트는 `pd.to_numeric(errors="coerce")`로 정상 처리). benign/attack 카운트는 experiment2와 **정확히 일치**(238037/93063)하므로 오염 25행은 분리지표에 사실상 영향 없음.
- 80개 컬럼 전체 목록은 `results2/csv_columns_report.json`의 `"columns"` 키 참조. (Flow/Fwd/Bwd 패킷·바이트·IAT 통계 + Flag 카운트 + Pkt Len/Size 통계 등 표준 CICFlowMeter feature.)

---

## 4. 분리력 상위 20 컬럼

| # | 컬럼 | dtype | nunique | mean_ben | mean_atk | 분리지표 |
|---|------|-------|---------|----------|----------|----------|
| 1 | Fwd Seg Size Min | object | 11 | 15.71 | 17.95 | **0.3109** |
| 2 | PSH Flag Cnt | object | 3 | 0.3964 | 0.519 | 0.2481 |
| 3 | Fwd Pkt Len Mean | object | 10071 | 50.11 | 36.05 | 0.2388 |
| 4 | Fwd Seg Size Avg | object | 10071 | 50.11 | 36.05 | 0.2388 (=#3 중복) |
| 5 | Fwd IAT Mean | object | 130083 | 3.508e6 | 1.311e6 | 0.2059 |
| 6 | Pkt Size Avg | object | 18850 | 93.01 | 70.81 | 0.2037 |
| 7 | Fwd Pkt Len Max | object | 1341 | 178.3 | 127.9 | 0.1963 |
| 8 | Flow IAT Mean | object | 146409 | 3.054e6 | 1.026e6 | 0.1949 |
| 9 | Fwd IAT Max | object | 95024 | 6.458e6 | 3.536e6 | 0.1907 |
| 10 | Fwd IAT Min | object | 42618 | 2.579e6 | 6.263e5 | 0.1866 |
| 11 | Pkt Len Mean | object | 19125 | 79.35 | 60.15 | 0.1859 |
| 12 | Flow IAT Min | object | 24731 | 2.456e6 | 5.616e5 | 0.1837 |
| 13 | Flow IAT Max | object | 83987 | 6.537e6 | 3.756e6 | 0.1808 |
| 14 | Idle Max | object | 8000 | 6.24e6 | 3.548e6 | 0.1747 |
| 15 | Pkt Len Max | object | 1027 | 363.3 | 274.3 | 0.1746 |
| 16 | Idle Mean | object | 16390 | 5.956e6 | 3.347e6 | 0.1743 |
| 17 | Idle Min | object | 19572 | 5.687e6 | 3.129e6 | 0.1733 |
| 18 | Fwd Pkt Len Std | object | 13858 | 53.84 | 39.35 | 0.1658 |
| 19 | Fwd IAT Tot | object | 96007 | 1.527e7 | 1.005e7 | 0.1567 |
| 20 | Bwd Pkt Len Max | object | 858 | 342.6 | 264.4 | 0.1542 |

> **관찰(사실)**: 최고 분리지표도 **0.31**에 불과(통상 0.8+면 "잘 가름", 0.2~0.5는 약함). 즉 단일 컬럼으로 강하게 가르는 feature는 **없음** → 이전 결론(이 데이터에서 Infiltration은 본질적으로 약분리)과 일관. 다만 4 feature(proto/syn/psh/size) 밖에 **size/IAT/idle 계열**이 비슷하거나 더 큰 분리력을 가짐.

---

## 5. 이산화 부적합 컬럼 (nunique≤2 또는 한 값 99%+) — 11개

```
Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, FIN Flag Cnt, CWE Flag Count,
Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg
```
- 대부분 상수(0)에 가까운 플래그/블록레이트 컬럼 → 비트 인코딩 시 정보량 0, **feature 후보에서 제외**.

---

## 6. 실행한 명령어와 결과

```bash
git status                                                  # ?? docs/  ?? qcbm/
python3 -m py_compile qcbm/tools/inspect_csv_columns.py     # PY_COMPILE_OK
python3 -u qcbm/tools/inspect_csv_columns.py                # DONE (에러 없음)
```
- 환경: `python3` 3.10.12. 원시 CSV(`data/cic_thursday.csv`, 102MB) 존재 → 전체 실행 성공.
- 출력: `results2/csv_columns_report.json`(전체 80컬럼 통계 + top20 + 부적합목록), `results2/csv_columns_report.csv`(분리력 내림차순 전체 컬럼 통계).
- **실행 실패/우회 없음.** 메모리: CSV 1회 로드 후 boolean mask로 그룹 분리(전체 복제 없음).

---

## 7. feature 확장 후보 (사실 기반 의견)

비트 예산(10~12bit)과 분리력·이산화 적합성을 함께 고려한 **후보**(최종 선택은 웹 AI):

1. **Fwd Seg Size Min** (sep 0.31, nunique 11) — 분리력 1위, 고유값 적어 **이산화 매우 용이**(3~4bit). 0순위 후보.
2. **Pkt Size Avg / Fwd Pkt Len Mean** (sep ~0.20~0.24, 연속) — 현재 size feature의 보강/대체. 분위수 4bit 적합. (단 Fwd Pkt Len Mean ≡ Fwd Seg Size Avg 중복이므로 둘 중 하나만.)
3. **IAT 계열** (Fwd IAT Mean/Min, Flow IAT Mean/Min, sep ~0.18~0.21) — proto/size와 **다른 축(시간 간격)** 정보 → feature 간 상관 확보에 유리. 분위수 3~4bit. 단 스케일 큼(1e6) → 로그+분위수 권장.
4. **Idle 계열** (Idle Mean/Max/Min, sep ~0.17) — IAT와 상관 가능성 → 중복 주의. 하나만.
5. 현행 **PSH Flag Cnt**(sep 0.25)는 유지 가치 있음. SYN/proto는 top20 밖이나 구조정보로 유지 가능.

> 핵심: 단일 분리력은 모두 약하므로(≤0.31), feature 확장의 기대효과는 **개별 분리력보다 "서로 다른 축(size·시간·플래그)의 결합 정보"** 에서 나올 가능성. 다만 본 데이터 특성상 큰 AUC 개선은 보장되지 않음(이전 한계 측정과 정합적).

---

## 8. 아직 불확실한 점

1. 분리지표는 **단변량·선형(평균차)** 지표 → 비선형/joint 분리력은 못 봄. sep 낮아도 조합으로 유효할 수 있음(반대도 가능).
2. IAT/Idle의 **큰 스케일·치우친 분포** → 단순 분위수 16버킷이 정보를 뭉갤 수 있음(로그 변환 검토 필요).
3. 오염 행 25개(`label`)는 카운트엔 영향 없으나, 일부 컬럼 통계에 극단값으로 미세 영향 가능(coerce로 대부분 NaN 처리됨).
4. 컬럼 간 **상관/중복**(Fwd Pkt Len Mean ≡ Fwd Seg Size Avg 등)을 아직 체계적으로 보지 않음 → 후보 선정 시 상관행렬 점검 필요.

---

## 9. 다음 추천 작업

1. 후보 컬럼(예: Fwd Seg Size Min + Pkt Size Avg + Fwd IAT Mean + 기존 proto/psh)으로 **10~12bit 인코딩 시안**을 정하고, 학습 전 **emp_joint AUC**를 먼저 측정(이전 baseline 도구 재사용) → 학습 없이 상한부터 확인.
2. 후보 컬럼들의 **상관행렬**을 뽑아 중복(IAT vs Idle, PktLen vs SegSize) 제거.
3. IAT/Idle은 **log1p 후 분위수 이산화** 적합성 비교.

---

## 최종 보고

```markdown
## 작업 요약
- CSV 80컬럼 전수 통계 + benign vs infilteration 표준화 평균차 분리지표 산출. 학습 없음, 기존 소스 무수정.
  최고 분리력도 0.31로 약함 -> 단일 강분리 feature 없음(이전 한계 측정과 일관). size/IAT/idle 계열이 후보.

## 전체 컬럼 수
- 80 (행 331,125; benign 238,037 / infilteration 93,063 / 오염 'label' 25)

## 분리력 상위 컬럼 (top 20)
| 컬럼 | dtype | nunique | mean_ben | mean_atk | 분리지표 |
|------|-------|---------|----------|----------|----------|
| Fwd Seg Size Min | object | 11 | 15.71 | 17.95 | 0.3109 |
| PSH Flag Cnt | object | 3 | 0.396 | 0.519 | 0.2481 |
| Fwd Pkt Len Mean | object | 10071 | 50.11 | 36.05 | 0.2388 |
| Fwd Seg Size Avg | object | 10071 | 50.11 | 36.05 | 0.2388 (중복) |
| Fwd IAT Mean | object | 130083 | 3.51e6 | 1.31e6 | 0.2059 |
| Pkt Size Avg | object | 18850 | 93.01 | 70.81 | 0.2037 |
| Fwd Pkt Len Max | object | 1341 | 178.3 | 127.9 | 0.1963 |
| Flow IAT Mean | object | 146409 | 3.05e6 | 1.03e6 | 0.1949 |
| Fwd IAT Max | object | 95024 | 6.46e6 | 3.54e6 | 0.1907 |
| Fwd IAT Min | object | 42618 | 2.58e6 | 6.26e5 | 0.1866 |
| ... (11~20: Pkt Len Mean, Flow IAT Min/Max, Idle Max/Mean/Min, Pkt Len Max, Fwd Pkt Len Std, Fwd IAT Tot, Bwd Pkt Len Max; sep 0.15~0.19) |

## 이산화 부적합 컬럼
- Bwd PSH Flags, Fwd/Bwd URG Flags, FIN Flag Cnt, CWE Flag Count,
  Fwd/Bwd Byts/b Avg, Fwd/Bwd Pkts/b Avg, Fwd/Bwd Blk Rate Avg (11개, 상수 근처)

## feature 확장 후보 (사실 기반)
- 0순위: Fwd Seg Size Min(분리1위+이산화 용이, 3~4bit)
- size축: Pkt Size Avg 또는 Fwd Pkt Len Mean(중복 주의)
- 시간축: Fwd/Flow IAT Mean·Min (log1p+분위수), proto/size와 다른 축
- 유지: PSH Flag Cnt
- 제외: 부적합 11개

## 다음 추천 단계
1. 후보로 10~12bit 인코딩 시안 -> 학습 전 emp_joint AUC부터 측정(상한 확인)
2. 후보 컬럼 상관행렬로 중복 제거, IAT/Idle log 변환 적합성 비교
```
