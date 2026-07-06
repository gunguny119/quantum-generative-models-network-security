# 현재 CSV의 공격 라벨 종류 확인 — DDoS 등 강분리 공격 존재 여부 (조사 전용)

- **작성일시**: 2026-06-26 07:49 (로컬)
- **작성 주체**: 로컬 Claude Code agent (default mode)
- **작업 성격**: Label 컬럼만 읽어 고유값 확인. **학습 없음. 기존 소스 무수정.**

---

## 1. 작업 목적

연구 방향("공격 유형별 joint vs marginal 분포 모델링 비교")에서, Infiltration은 joint(0.428) < marginal(0.438)로
**약분리·joint 무이득**이 확정됨. 다음 가설: **DDoS처럼 feature 조합에 시그니처가 있는 공격은 joint > marginal일 수 있다.**
이를 측정하려면 DDoS 데이터가 필요. 본 작업은 **현재 파일(`data/cic_thursday.csv`)에 DDoS 등 다른 공격 라벨이
있는지 먼저 확인**한다(있으면 데이터 추가 없이 즉시 측정 가능).

---

## 2. 시작 시점 git status 요약

```
?? docs/    ?? qcbm/      (둘 다 미추적)
git diff --stat: 비어 있음 (추적 파일 변경 0)
```
- 추가 파일: `qcbm/tools/check_attack_labels.py`, 본 `.md`. **기존 소스 무수정**(`load_frames`의 라벨 처리 `str.strip().str.lower()` 규칙만 재사용).

---

## 3. Label 전체 고유값 + 개수

### 원본(RAW)
| label | count |
|-------|-------|
| Benign | 238,037 |
| Infilteration | 93,063 |
| Label | 25 |

### strip + lower 적용 후
| label | count |
|-------|-------|
| benign | 238,037 |
| infilteration | 93,063 |
| label | 25 |

- `label` 25개 = CSV 중간에 헤더가 다시 끼어든 **오염 행**(공격 아님). 이전 컬럼 조사 결과와 동일.
- 실제 클래스는 **benign, infilteration 단 2종**.

---

## 4. DDoS 등 강분리 공격 존재 여부

**없음.** 자동 키워드 스캔(ddos/dos/portscan/bruteforce/bot/web/sql/xss) 결과 → **매칭 0건**.
이 파일에는 benign과 infilteration 외 다른 공격이 전혀 없음. **현재 데이터로는 DDoS joint/marginal 측정 불가.**

---

## 5. DDoS는 CSE-CIC-IDS2018의 어느 파일에 있나 (데이터 확보 가이드)

현재 파일은 **Thursday-01-03-2018** (Infiltration 전용 요일). CSE-CIC-IDS2018은 요일별로 공격이 다름.
DDoS 및 다른 강분리 공격이 있는 날짜 파일(아는 범위, 검증 권장):

| 날짜 파일 | 주요 공격 |
|-----------|-----------|
| **Tuesday-20-02-2018** | **DDoS-LOIC-HTTP, DDoS-LOIC-UDP** ← DDoS |
| **Wednesday-21-02-2018** | **DDoS-LOIC-UDP, DDoS-HOIC** ← DDoS |
| Wednesday-14-02-2018 | FTP-BruteForce, SSH-BruteForce |
| Thursday-15-02-2018 | DoS-GoldenEye, DoS-Slowloris |
| Friday-16-02-2018 | DoS-Hulk, DoS-SlowHTTPTest |
| Thu/Fri-22~23-02-2018 | Web attacks (BruteForce-Web, XSS, SQL Injection) |
| Wed/Thu-28-02~01-03-2018 | **Infiltration** (← 현재 파일) |
| Friday-02-03-2018 | Bot |

- 다운로드 경로 패턴(`experiment.py`의 `DATA_URL` 참고): S3 버킷
  `cse-cic-ids2018.s3.amazonaws.com/Processed Traffic Data for ML Algorithms/<요일>_TrafficForML_CICFlowMeter.csv`.
  → 현재 `Thursday-01-03-2018...`를 **`Tuesday-20-02-2018...`** 또는 **`Wednesday-21-02-2018...`** 로 바꾸면 DDoS 파일.
- 주의: 컬럼 스키마는 요일별로 동일(CICFlowMeter 80컬럼)하므로 **기존 인코딩/도구 재사용 가능**. (단 파일별 헤더 오염 행 가능성은 동일하게 방어 필요.)

---

## 6. 실행한 명령과 결과

```bash
git status                                              # ?? docs/  ?? qcbm/
python3 -m py_compile qcbm/tools/check_attack_labels.py # PY_COMPILE_OK
python3 -u qcbm/tools/check_attack_labels.py            # DONE
# 빠른 확인(동일 결과):
python3 -c "import pandas as pd; s=pd.read_csv('qcbm/data/cic_thursday.csv', usecols=['Label'], low_memory=False)['Label'].astype(str).str.strip().str.lower(); print(s.value_counts())"
```
- 결과: benign 238037 / infilteration 93063 / label 25. 강분리 공격 후보 **0건**.
- 실패/우회 없음. 학습 미실행.

---

## 7. 다음 추천

1. **DDoS로 joint vs marginal 가설 검증하려면** `Tuesday-20-02-2018` 또는 `Wednesday-21-02-2018` 파일을 받아야 함(현재 파일엔 없음).
2. 받은 뒤에는 **새 학습 없이 먼저** 기존 `tools/diagnose_classical_baseline.py` 방식으로 **emp_joint vs marginal AUC**부터 측정 → DDoS에서 joint > marginal 인지 즉시 확인 가능(QCBM 학습은 그 다음).
3. (대안) 데이터 추가 없이 진행하려면, 현재 파일의 Infiltration에 대해 §feature 확장(이전 inspect_csv_columns 후보: Fwd Seg Size Min, IAT 계열)으로 joint 이득이 생기는지 측정.

---

## 최종 보고

```markdown
## 작업 요약
- data/cic_thursday.csv 의 Label 전수 확인. 클래스는 benign(238037)/infilteration(93063)/오염'label'(25) 2종뿐.
  DDoS 등 강분리 공격 없음 -> 현재 파일로는 DDoS joint/marginal 측정 불가. 학습/소스수정 없음.

## Label 전체 (strip/lower 적용)
| label | count |
|-------|-------|
| benign | 238,037 |
| infilteration | 93,063 |
| label | 25 (헤더 오염, 공격 아님) |

## DDoS 등 강분리 공격 존재 여부
- 없음. 키워드 스캔(ddos/dos/portscan/bruteforce/bot/web/sql/xss) 0건.

## 다음 추천 단계
1. (없으므로) DDoS 파일 확보: Tuesday-20-02-2018 / Wednesday-21-02-2018 (S3, 동일 스키마)
2. 확보 후 새 학습 없이 emp_joint vs marginal AUC부터 측정(diagnose_classical_baseline 재사용)
```
