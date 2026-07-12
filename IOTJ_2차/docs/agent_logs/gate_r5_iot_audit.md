# Gate R5 — IoT 데이터 지형 contextuality 감사 (다중 기기 × 정의 2종 × 부분집합 5종)

- 작성: 2026-07-08 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **측정(검증·정량화)**, 결정적(seed=20260705), 무설치, 병렬(16 proc)+tmux(`hoc_r5`).
- 재현: `python3 tools/gate_r5_iot_audit.py --k 8 --boot 500 --null 500 --rows 50000 --jobs 16`. 재사용(무수정): `gate_r3_iot_cf`(=R3, def-a/b·부분집합5·no_sig), `kink_origin_v1`(cyclic CF-LP).

## ★ 인식론적 역할 (서두 명시)
**Proposition(Fine 정리 따름정리)**: 동시 기록된 고전 로그에서 def-a형(feature-block) 구성의 CF는 **수학적으로 정확히 0**(단일 결합분포 → 전역 절편 존재). 따라서 본 감사는 **'발견'이 아니라**:
(i) **정리·구현의 검증**, (ii) def-b형(시간창)의 **유한표본 요동 규모 측정**, (iii) 트래픽 종류 불변성 확인.
→ **def-a 에서 CF>0 이 나오면 발견이 아니라 구현 버그 신호** → 원인 규명(본 감사서 실제로 초기 버그 1건 발견·수정, §실행 참조).

## self-check (cyclic CF-LP) — 통과
좌절 3-cycle CF=**1.000**(기대1), 정합 CF=**0.000**(기대0). LP는 데이터 무관이라 1회로 전 셀 공통.

## 데이터 접근성 감사표 (웹 확인 — GREEN=직접링크 작동+수치테이블만)
| 데이터셋 | URL | 포맷 | 크기 | 계정 | 판정 | 코멘트 |
|---|---|---|---|---|---|---|
| **N-BaIoT benign** (UCI 442) | archive.ics.uci.edu/static/public/442/…zip | CSV(수치 115) | 1.7GB zip | 불요 | **GREEN** | **9기기 benign 직접 사용(본 감사)** |
| N-BaIoT attacks (mirai/gafgyt) | 동 zip 내 `*.rar` | RAR→CSV | 186MB/기기 | 불요 | **RED** | unrar/rarfile/7z 부재로 추출 불가(레버2 스킵) |
| IoT-23 (Stratosphere) | mcfp…small.tar.gz / HF parquet | conn.log / parquet | 9.4GB / 28KB | 불요 | **YELLOW** | full=9.4GB 과대; HF parquet=**pyarrow 부재로 미판독**(무설치) |
| CIC-IoT2023 (UNB) | unb.ca/cic/datasets/iotdataset-2023.html / Kaggle | CSV | ~13GB | Kaggle/대용량 | **YELLOW** | 직접 대용량 or Kaggle 계정 |
| Edge-IIoTset | IEEE DataPort / Kaggle | CSV | ~12GB | 계정 | **YELLOW** | Kaggle/DataPort 로그인 |
| MQTT-IoT-IDS2020 | IEEE DataPort | CSV/pcap | - | DataPort 계정 | **YELLOW** | 계정(미검증) |
| WUSTL-IIoT-2021 | cse.wustl.edu/~jain/iiot2 / DataPort | CSV | 106MB/390MB | 직접링크 미해결 | **YELLOW** | 직접 파일 URL 미해결 |
| TON_IoT / Bot-IoT | AARNET CloudStor | CSV | - | - | **RED** | CloudStor 폐쇄(기확인) |
- **R5 포함 권고**: 즉시 실행 가능 GREEN = **N-BaIoT benign 9기기**뿐. 나머지는 YELLOW(계정·대용량·리더부재)/RED. → 본 감사는 **N-BaIoT 9기기**로 수행(정직: 외부셋 미확보 사유 상기 표).

## 데이터 (MD5 · 기기 · 행수)
N-BaIoT zip **MD5=56bf8e22debda482596f6d2be1260afd**. **9기기 benign**(행샘플 ≤50k·중앙값 이진화·상수열 제거):
Danmini(49,548)·Ecobee(13,113)·Ennio(39,100)·Philips(50k)·Provision_737E(50k)·Provision_838(50k)·Samsung(50k)·SimpleHome_1002(46,585)·SimpleHome_1003(19,528). feature 111~115.

## 결과 — 마스터 (9기기 × 정의 2 × 부분집합 5; k=8, boot 500, null 500)
### def-a (feature-block, 45셀) — **검증**
- **전 45셀 CF = 0.0 정확** (max|CF| = **4.4e-16** = 기계 epsilon). no-sig 잔차 0. → **Fine 정리·구현 검증 통과**(def-a CF>0 셀 0건).

### def-b (시간창, 9셀) — **유한표본 요동 + null 보정**
| 기기 | CF | null 95%CI 상단 | null 내? |
|---|---|---|---|
| Danmini | 2.0e-05 | 4.0e-05 | ✔ |
| Ecobee | 7.6e-05 | 1.5e-04 | ✔ |
| Ennio | 0 | 5.1e-05 | ✔ |
| Philips | 0 | 4.0e-05 | ✔ |
| Provision_737E | 2.0e-05 | 4.0e-05 | ✔ |
| Provision_838 | 0 | 4.0e-05 | ✔ |
| Samsung | 4.0e-05 | 4.0e-05 | ✔ |
| SimpleHome_1002 | 4.3e-05 | 4.3e-05 | ✔ |
| SimpleHome_1003 | 0 | 1.0e-04 | ✔ |
- **전 9셀 관측 CF 가 셔플 null 분위수 내(9/9)** → **"실측 CF는 표본 요동과 통계적으로 구별 불가"** 확정. def-b CF 최대 7.6e-5.

### 잔차·상한
- **no-signalling 잔차 L1 ≤ 1.0e-4** (전 셀). CF>0(≈0) 셀도 잔차 동급 → 교란 아님.
- **잔여 gap 상한 = c·CF²_upper ≤ 2.9e-9** (c=1/8 경계근사) = 실질 0.

## 판정
- **def-a 검증 통과**: 전 셀 CF=0(4e-16) → Fine 정리·구현 정확성 확인.
- **def-b 유한표본 확정**: 전 셀 null 내(9/9), CF≤7.6e-5 → 시간창 CF는 표본요동일 뿐.
- **트래픽 불변**: benign **9기기 전부 CF≈0**(기기·모델 무관). ★공격 vs 정상 비교는 **레버2(mirai/gafgyt .rar) unrar 부재로 불가** → 정상 트래픽 내 기기-불변만 확인(공격 비교 미수행, 정직).
- **히트맵**(`r5_heatmap.png`, 9기기×6): 전면 ≈0 → 논문 Fig "CF across the IoT data landscape" 후보(축·수치 라벨 명확, max CF 1.5e-4).

## 실행 메모 (버그 1건 규명)
- 초기 전체 실행서 `demonstrate_structure.csv`(zip 최상위 비-기기 파일)가 10번째 "기기"(0행)로 잡혀 빈 행렬→`cyclic_cf` linprog 차원오류. → **기기 필터를 "benign CSV 보유 top-level"로 수정**(gate_r5_iot_audit.py, 기존 R3 무수정). 재실행 정상(54셀, 23s). def-a CF>0 아닌 **입력 데이터 오염**이 원인이었음(버그 신호 규명 원칙대로 처리).

## 한계 (필수)
- 감사 범위 = **N-BaIoT 9기기 benign 한정**(외부셋 미확보: IoT-23 리더부재·CIC/Edge/MQTT 계정/대용량·WUSTL 링크미해결·TON/Bot CloudStor폐쇄·공격 .rar 미추출). 행샘플 ≤50k·중앙값 이진화·부분집합 선택 의존.
- 정적 결합분포는 이론상 CF=0(측정=검증/정량화, 발견 아님). (a)(b)는 조작적 구성이나 **Proposition이 구성 일반을 커버**. c=1/8 경계근사. **관측/검증이지 supremacy 아님**(모델클래스 분리).

## git 상태
- 신규(untracked): `tools/gate_r5_iot_audit.py`, `results2/gate_r5/{r5.json,r5_heatmap.png,run.log}`, `data/nbaiot/proc5/`, 본 로그. 기존(R3/K) 무수정 import. 커밋은 사용자 지시 시에만.

## [해석입력용 요약 — 웹 AI 전달]
1. **def-a 검증 통과**: N-BaIoT 9기기 benign, 45셀 전부 CF=0.0(max 4.4e-16=기계ε) → Fine 정리·구현 정확성 확인(발견 아님). (초기 crash는 zip 비-기기 파일 오염 버그로 규명·수정.)
2. **def-b 유한표본 요동 확정**: 시간창 9셀 CF ≤7.6e-5, **전부 셔플-null 분위수 내(9/9)** → "실측 CF는 표본요동과 구별 불가" 통계로 받침. no-sig ≤1e-4, gap 상한 ≤2.9e-9.
3. **트래픽**: benign 9기기 CF≈0(기기 무관). 공격 비교는 mirai/gafgyt `.rar` unrar 부재로 불가(정직 스킵) — 정상 트래픽 내 불변만 확인.
4. **히트맵**은 논문 Fig "CF across the IoT data landscape"로 쓸 만함(9기기×6, 전면≈0, max 1.5e-4). 접근성표: 즉시 GREEN은 N-BaIoT뿐, 나머지 YELLOW/RED. supremacy 아님.
