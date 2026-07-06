# CIC 어려운 공격 데이터 확보 — Kaggle 접근 확인 + 대체 경로(S3) 발견 (확보·검증 전용)

- **작성일시**: 2026-06-26 14:32 (로컬)
- **작성 주체**: 로컬 Claude Code agent (default mode)
- **작업 성격**: 데이터 확보 + 컬럼/라벨 검증만. **QCBM 학습·복잡도 진단 없음. 기존 소스/데이터 무수정.**

---

## 1. 작업 목적

연구 방향: "더 어렵고 복잡한 분포를 가진 공격(Infiltration / Bot / Brute Force / Web 등)"에서
양자 우위 가능성 재탐색. 본 작업은 그 **어려운 공격 데이터를 확보하고 기존 파이프라인 컬럼
호환성을 검증**하는 것(학습·진단은 다음 단계).

---

## 2. 시작 시점 git status

```
On branch main
Untracked files:
  docs/          (미추적)
  qcbm/          (미추적, 실제 작업/데이터 디렉토리)
  qcbm_share_20260626.zip
nothing added to commit (추적 파일 변경 0)
```

- 주의: 작업 제목/태스크는 `data/`로 지칭했으나, 실제 데이터·소스는 **`qcbm/` 하위**에 있음.
  - 기존 데이터: `qcbm/data/cic_0221_ddos.csv` (329MB), `qcbm/data/cic_thursday.csv` (108MB)
  - 핵심 소스: `qcbm/experiment2.py`, `qcbm/tools/train_qcbm_iat_b2.py`
  - 로그 위치: 리포 루트 `docs/agent_logs/`

---

## 3. 기존 컬럼/Label 기준 (대조 기준)

### 핵심 7개 컬럼이 실제 코드에서 쓰이는 곳
- `experiment2.py:77` → `cols = ["Protocol","SYN Flag Cnt","PSH Flag Cnt","Pkt Len Mean","Label"]`
- `tools/train_qcbm_iat_b2.py:95-98` → `["Protocol","SYN Flag Cnt","PSH Flag Cnt","Pkt Len Mean","Fwd IAT Mean","Flow IAT Mean","Label"]` 를 `usecols`로 읽음.

### 기존 파일 헤더 (둘 다 동일한 CICFlowMeter 80컬럼)
- `cic_0221_ddos.csv`, `cic_thursday.csv` 모두 동일 80컬럼.
- 7개 핵심 컬럼 위치: Protocol(2), Flow IAT Mean(19), Fwd IAT Mean(24), Pkt Len Mean(43),
  SYN Flag Cnt(47), PSH Flag Cnt(49), Label(80). → **7개 전부 존재.**

### 기존 Label 분포
| 파일 | Label | count |
|------|-------|-------|
| cic_0221_ddos.csv | DDOS attack-HOIC | 686,012 |
| | Benign | 360,833 |
| | DDOS attack-LOIC-UDP | 1,730 |
| cic_thursday.csv | Benign | 238,037 |
| | **Infilteration** | 93,063 |
| | Label (헤더 오염행) | 25 |

→ **중요 발견**: `cic_thursday.csv`는 이미 **어려운 공격 Infiltration(93,063개)** 을
   동일 스키마로 보유 중. (어려운 공격 1종은 이미 로컬에 존재.)

---

## 4. Kaggle 접근 확인 결과 — **블록됨**

| 항목 | 결과 |
|------|------|
| kaggle CLI (`which kaggle`) | **없음** (미설치) |
| python `kaggle` 모듈 | **없음** (`import kaggle` 실패) |
| pip 패키지 | **없음** (`pip list \| grep kaggle` → 0건) |
| API 토큰 `~/.kaggle/kaggle.json` | **없음** |
| 네트워크 (kaggle.com 도달) | 도달 가능 (HTTPS 응답 옴) |

→ **결론: Kaggle에서 다운로드 불가.** CLI 미설치 + 인증 토큰 부재.
   금지사항 7(의존성 임의 설치 금지) 및 실패행동 4(미설치 시 보고 후 멈춤)에 따라
   **kaggle CLI를 임의 설치하지 않음.** 추측 우회/임의 데이터 생성도 하지 않음(금지사항 5).

---

## 5. 대체 경로 발견 — 프로젝트의 기존(검증된) 확보 방법: AWS Open Data S3 (무인증)

기존 agent 로그(`2026-06-26_0749`, `2026-06-26_0753`)와 `experiment.py:51-53` 확인 결과,
**이 프로젝트는 원래 Kaggle이 아니라 CSE-CIC-IDS2018 AWS 공개 S3 버킷**에서 데이터를 받았음:

```
https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/
  <Weekday>-<DD>-<MM>-<YYYY>_TrafficForML_CICFlowMeter.csv
```

- **무인증 공개 버킷** (S3 Open Data). 기존 `cic_0221_ddos.csv`가 바로 이 경로로 받은 파일.
- 요일별 파일은 **전부 동일한 CICFlowMeter 80컬럼 스키마** → 기존 파이프라인/인코딩 그대로 재사용.
- 따라서 이 경로는 "추측 우회"가 아니라 **프로젝트 자신의 입증된 확보 방법**이며,
  목표(진짜 CIC 어려운 공격 + 호환 컬럼 확보)를 그대로 달성함.

### 어려운 공격 ↔ 날짜 파일 매핑 (요일별 공격 상이)
| 날짜 파일 | 주요 공격(어려움) | S3 HEAD 확인 | 크기 |
|-----------|------------------|--------------|------|
| Wednesday-14-02-2018 | **FTP-BruteForce, SSH-BruteForce** | HTTP 200 | 358 MB |
| Friday-02-03-2018 | **Bot (Botnet)** | HTTP 200 | 352 MB |
| Thursday-22-02-2018 | **Web: BruteForce-Web, XSS, SQL Injection** | HTTP 200 | 383 MB |
| Friday-23-02-2018 | **Web: BruteForce-Web, XSS, SQL Injection** | HTTP 200 | 383 MB |
| (보유) Thursday-01-03-2018 | Infiltration | (이미 로컬) | 108 MB |

→ 4개 후보 모두 S3에서 **HTTP 200 도달 확인**(다운로드 가능 상태). 각 ~350–380MB이므로
   용량 절약 위해 **일부만** 받는 것이 적절.

---

## 6. 현재까지 실행한 명령과 결과 (전부 실제 실행)

```bash
git status                              # untracked: docs/, qcbm/, zip
which kaggle                            # (출력 없음 = 미설치)
ls ~/.kaggle/kaggle.json                # NO TOKEN
python3 -c "import kaggle"              # ModuleNotFoundError
pip list | grep -i kaggle              # 0건
head -1 qcbm/data/cic_0221_ddos.csv     # 80컬럼, 핵심 7개 전부 존재
head -1 qcbm/data/cic_thursday.csv      # 동일 80컬럼
python3 -c "...value_counts() on Label" # 위 §3 분포
curl -I <S3 4개 hard-attack 파일>       # 전부 HTTP 200, text/csv
```

- 실패/우회 없음. **Kaggle 다운로드는 인증 부재로 미실행**(가짜 성공 보고 안 함).

---

## 7. 사용자 결정 결과

- **확보 경로**: (A) **S3 무인증 경로로 지금 받기** 선택 (Kaggle 대기 아님).
- **받을 공격**: **Bot/Botnet (Friday-02-03-2018)** 1개 선택 (용량 절약, Infiltration은 이미 로컬 보유).

---

## 8. 다운로드 결과 — **성공**

```bash
URL="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
curl -sS --max-time 580 -o /home/elicer/IOTJ/qcbm/data/cic_bot_0203.csv "$URL"   # curl exit=0, ~25초
```

- **저장 위치**: `qcbm/data/cic_bot_0203.csv`
- **크기**: **352,368,373 bytes** (337MB) — S3 Content-Length와 **정확히 일치**(손상 없음).
- 기존 `cic_0221_ddos.csv` / `cic_thursday.csv` **무수정**(새 파일명으로 저장).

---

## 9. 컬럼 호환성 검증 — **완전 호환**

- 컬럼 수: **80개** (기존과 동일).
- 핵심 7개 컬럼 존재: `{Protocol, SYN Flag Cnt, PSH Flag Cnt, Pkt Len Mean, Fwd IAT Mean, Flow IAT Mean, Label}` → **전부 True**.
- 헤더 비교: `diff`가 처음엔 차이를 보고 → 원인은 **줄바꿈 문자뿐**. 새 파일은 **CRLF(`\r\n`)**,
  기존은 LF(`\n`). **CR 제거 후 헤더 byte 단위 완전 동일.** pandas가 CRLF를 투명 처리
  (라벨이 `Bot`/`Benign`으로 깔끔히 파싱됨, 80컬럼 정상 인식).
- → **불일치 컬럼 없음.** 기존 인코딩/도구(`experiment2.py`, `train_qcbm_iat_b2.py`) 그대로 재사용 가능.
  (단 attack 라벨이 `bot`이므로 다음 단계 학습 시 attack 마스크를 `bot`으로 지정 필요 — 매핑은 다음 작업.)

---

## 10. 라벨(공격)별 샘플 수

| Label | 샘플 수 |
|-------|---------|
| Benign | 762,384 |
| **Bot** (Botnet, 어려운 공격) | 286,191 |
| (헤더 오염행 "Label") | **0건 — 깨끗** |

- 총 1,048,575행. 어려운 공격 **Bot 286,191개** 확보. 표본 수 충분(분포 학습에 유리).
- `cic_thursday.csv`(Infiltration 93,063) 와 합치면 어려운 공격 **2종**(Bot, Infiltration) 로컬 보유.

---

## 11. 최종 git status (새 파일만 추가 확인)

```
On branch main
Untracked files:
  docs/      qcbm/      qcbm_share_20260626.zip
nothing added to commit (추적 파일 변경 0)
```
- 새로 생긴 것: `qcbm/data/cic_bot_0203.csv`(다운로드), 본 `.md`. **기존 추적 파일 변경 0건.**
- 기존 데이터/소스/results2 **무수정 확인.**

---

## 12. 남은 불확실성 / 다음 추천
- 줄바꿈 CRLF 차이는 pandas에 투명하지만, 만약 향후 다른 도구가 raw 파싱한다면 인지 필요(현 파이프라인엔 무영향).
- 다음 단계(별도 작업): **복잡도 진단** — Bot에 대해 `emp_joint vs marginal AUC`(기존 `tools/diagnose_classical_baseline.py` 방식, 학습 없음) 측정 → Bot 분포가 고랭크·조밀한지(joint > marginal 인지) 확인. 그 다음 QCBM 학습.
- (선택) 추가 어려운 공격이 필요하면 동일 S3 경로로 Brute Force(14-02) / Web(22·23-02) 확보 가능(전부 HTTP 200 확인됨, Web은 공격 표본 희소 주의).
