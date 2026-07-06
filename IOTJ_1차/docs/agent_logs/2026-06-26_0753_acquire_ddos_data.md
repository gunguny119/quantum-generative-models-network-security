# DDoS 데이터 파일(02-21-2018) 확보 — DATA_URL 구조 확인 및 다운로드

- **작성일시**: 2026-06-26 07:53 (로컬), 다운로드 07:55
- **작성 주체**: 로컬 Claude Code agent (default mode)
- **작업 성격**: 공개 데이터셋 1개 다운로드 + 검증. **학습 없음. 기존 소스/데이터 무수정.**

---

## 1. 작업 목적

연구 방향("공격 유형별 joint vs marginal 분포 모델링 비교")에서, 현재 파일은 Infiltration만 있어 약분리·joint 무이득.
**DDoS처럼 feature 조합에 시그니처가 있는 공격은 joint > marginal일 수 있다**는 가설 검증을 위해
DDoS가 포함된 CSE-CIC-IDS2018 **02-21-2018** 파일을 확보(공개 데이터, AWS Open Data S3).

---

## 2. 시작 시점 git status 요약

```
?? docs/    ?? qcbm/      (둘 다 미추적)
git diff --stat: 비어 있음
```
- 추가 파일: `qcbm/data/cic_0221_ddos.csv`(신규 다운로드), 본 `.md`. **기존 소스/`cic_thursday.csv` 무수정.**

---

## 3. 기존 DATA_URL 구조 (실제)

[experiment.py:51-53](../../qcbm/experiment.py#L51-L53):
```
https://cse-cic-ids2018.s3.amazonaws.com/
Processed%20Traffic%20Data%20for%20ML%20Algorithms/
Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
```
- S3 무인증 공개 버킷. `urllib.request.urlretrieve(DATA_URL, path)`로 받음 ([experiment.py:73](../../qcbm/experiment.py#L73)).
- 파일명 패턴: `<Weekday>-<DD>-<MM>-<YYYY>_TrafficForML_CICFlowMeter.csv`. 디렉토리의 공백은 `%20`.
- **02-21-2018 = 수요일** → `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv`.

---

## 4. 02-21-2018 다운로드 명령

```bash
URL="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
curl -s --max-time 580 -o qcbm/data/cic_0221_ddos.csv "$URL"
# (대안) wget -O qcbm/data/cic_0221_ddos.csv "$URL"
```
- HEAD 사전 확인: `HTTP/1.1 200 OK`, `Content-Type: text/csv`, `Content-Length: 328893673`(≈329MB), `Accept-Ranges: bytes`.

---

## 5. 현재 파일 컬럼명 리스트 (비교 기준, 80개)

`Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std, Flow Byts/s, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s, Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var, FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt, URG Flag Cnt, CWE Flag Count, ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg, Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts, Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Label`

(80컬럼. 코드가 실제로 읽는 5개: Protocol, SYN Flag Cnt, PSH Flag Cnt, Pkt Len Mean, Label.)

---

## 6. 다운로드 결과

**성공.**
- 저장: `qcbm/data/cic_0221_ddos.csv`
- 크기: **328,893,673 bytes** (Content-Length와 정확히 일치, 손상 없음).
- 소요: 약 1분 9초 (curl exit 0).
- 기존 `cic_thursday.csv`는 건드리지 않음(새 파일명으로 저장).

---

## 7. 받은 파일 검증

### Label 고유값 (strip/lower)
| label | count |
|-------|-------|
| **ddos attack-hoic** | 686,012 |
| benign | 360,833 |
| **ddos attack-loic-udp** | 1,730 |

→ **DDoS 존재 확인** (HOIC 대량 + LOIC-UDP 소량). 웹 검증(02-21 = LOIC-UDP + HOIC)과 일치.
→ 이번 파일은 헤더 오염 행(`label`) **없음** (cic_thursday와 달리 깨끗).

### 컬럼명 일치 여부
- `diff` 결과 **헤더 완전 동일**(80컬럼, cic_thursday.csv와 identical).
- 코드가 읽는 5개 컬럼 존재 검증: `{Protocol: True, SYN Flag Cnt: True, PSH Flag Cnt: True, Pkt Len Mean: True, Label: True}`.
- → **기존 인코딩/도구 그대로 재사용 가능.** (단 라벨 매칭만 `infilteration` → DDoS 라벨로 바꿔야 함; 아래 §주의.)

### 주의 (다음 단계용)
- 공격 라벨이 **2종**(`ddos attack-hoic`, `ddos attack-loic-udp`)이고, 기존 코드 `load_frames()`는 `infilteration` 하나만 attack으로 잡음. DDoS 측정 시 attack 마스크를 `lab.str.startswith("ddos")` 또는 두 라벨 합집합으로 바꿔야 함 (기존 코드 수정 대신 새 진단 스크립트에서 처리 권장).
- HOIC가 686k로 benign(361k)보다 많음 → train/test 분할·평가 시 클래스 비율 유의.

---

## 8. 다음 추천 단계

1. **새 학습 없이 먼저** `tools/diagnose_classical_baseline.py`를 응용해 이 파일에서 **emp_joint vs marginal AUC**를 측정 → DDoS에서 **joint > marginal**인지(가설 핵심) 5분 내 확인.
   - attack = `ddos attack-hoic` ∪ `ddos attack-loic-udp` (또는 HOIC만으로 먼저).
   - benign train/test 분할은 기존과 동일(seed 7, 0.7/0.3).
2. (가설 지지 시) 동일 8bit 인코딩(proto/syn/psh/size)으로 QCBM 학습 → joint 모델의 DDoS AUC 측정.
3. (선택) 02-20-2018(LOIC-HTTP, **84컬럼** — 컬럼 4개 더 많음, 스키마 주의)은 필요 시 별도 확보.

---

## 최종 보고

```markdown
## 작업 요약
- CSE-CIC-IDS2018 02-21-2018(수) 파일을 S3 공개버킷에서 다운로드 성공(329MB). DDoS 라벨 존재 확인,
  헤더 80컬럼이 기존 cic_thursday.csv와 완전 동일 -> 기존 도구 재사용 가능. 학습/소스수정 없음.

## 기존 DATA_URL 구조
- https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/<Weekday>-<DD>-<MM>-<YYYY>_TrafficForML_CICFlowMeter.csv
- experiment.py:51-53, urllib.urlretrieve 로 받음. 02-21=수요일 -> Wednesday-21-02-2018_...

## 02-21-2018 다운로드 명령
```bash
URL="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
curl -s --max-time 580 -o qcbm/data/cic_0221_ddos.csv "$URL"
```

## 다운로드 결과
- 성공: qcbm/data/cic_0221_ddos.csv, 328,893,673 bytes (Content-Length 일치), ~69초, curl exit 0.

## 받은 파일 검증 (성공)
- Label 고유값: ddos attack-hoic(686012), benign(360833), ddos attack-loic-udp(1730). 헤더 오염 없음.
- 컬럼명 일치 여부: 헤더 80컬럼 cic_thursday.csv와 완전 동일(diff 무차이). 필수 5컬럼 모두 존재.
- 주의: 기존 load_frames는 attack=infilteration만 잡음 -> DDoS 측정 시 attack 마스크를 ddos 라벨로 변경 필요(새 스크립트에서).

## 다음 추천 단계
1. 받은 DDoS 데이터로 emp_joint vs marginal AUC 사전 측정 (학습 없음, 5분) — joint>marginal 가설 확인
2. 가설 지지 시 동일 8bit 인코딩으로 QCBM 학습/측정
```
