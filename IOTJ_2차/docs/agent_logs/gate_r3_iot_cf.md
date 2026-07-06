# Gate R3 — 실제 IoT 데이터셋 CF 실측 (꺼진 앵커의 IoT 정박)

- 작성: 2026-07-05 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **측정(정박·정량화)**, 결정적(seed=20260705), 무설치(numpy·scipy·pandas). 병렬(16 proc) + tmux.
- 재현: `python3 tools/gate_r3_iot_cf.py --devices 3 --rows 500000 --k 8 --boot 500 --jobs 16` (tmux 세션 `hoc_r3`).
- 재사용(무수정): `kink_origin_v1`(=K) cyclic CF-LP·setup. **★목적=정박/정량화**(정적 결합분포는 이론상 CF=0 자명; 발견 아님). supremacy 아님.

## 데이터 출처 / 해시
- **N-BaIoT** (UCI 442): `https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip` (1.7GB, **MD5=56bf8e22debda482596f6d2be1260afd**).
  - 기기별 `benign_traffic.csv`(115 feature 통계량) 직접 사용. 공격은 `*.rar`(unrar 필요) → 미추출. **benign IoT 텔레메트리 = IoTJ '정상 IoT' 꺼진 앵커의 이상적 대상**.
  - 사용 기기 3종: **Danmini_Doorbell**(49,548행), **Ecobee_Thermostat**(13,113), **Ennio_Doorbell**(39,100). (benign 전량 ≤50만 → 행샘플 상한 미도달.)
- **TON_IoT / BoT-IoT**: CloudStor(AARNET) **2024 폐지**로 공식 배포 링크 접근 불가(401/무응답) → 확보분(N-BaIoT 3기기)으로 진행(정직 보고, 폴백 규칙대로).

## self-check (cyclic CF-LP) — 통과
3-cycle 좌절(반상관) CF=**1.000**(기대1), 정합 CF=**0.000**(기대0). → LP 유효 → 실 IoT CF 측정 신뢰.

## 측정 (1차 방법론: 중앙값 이진화 · k=8 · 부분집합 5종 · 정의 2종 · 부트스트랩 500)
정의: **(a)** feature-block k-cycle context, **(b)** 시간창 3-cycle(행순서=시간대용, lag1,1,2). CF-LP + 부트스트랩 95%CI + no-signalling 잔차.
| 기기 | 정의/부분집합 | CF | 95%CI | no-sig 잔차 | gap 상한 (c·CF²_up) |
|---|---|---|---|---|---|
| Danmini/Ecobee/Ennio | **def-a** highvar·targeted·random×3 (15셀) | **0.0000** | [0,0] | 0.0000 | ≤1.5e-31 |
| Danmini | def-b 시간창 | 0.0000 | [0,0] | 0.0000 | 2.0e-10 |
| Ecobee | def-b 시간창 | 0.0001 | [0, 0.0002] | 0.0001 | **2.9e-09** |
| Ennio | def-b 시간창 | 0.0000 | [0,0] | 0.0000 | 8.2e-11 |
- **전 18셀 CF≈0**: def-a **정확히 0.0**(전 기기·전 부분집합), def-b 최대 8e-5. **CF CI 상단 최대 = 0.00015 < 0.05**. no-signalling 잔차 최대 1.0e-4(무시가능).
- **잔여 gap 상한 = c·CF²_upper ≤ 2.9e-09** (c=1/8 gate_e cyclic 경계값, 경계근사 주의).

## 판정
- **예상대로 CF≈0 실측 확정**: 3개 실제 IoT 기기 텔레메트리(초인종·온도계) × 2정의 × 5부분집합 모두 CF≈0(CI 상단 0.00015). → **실 IoT 텔레메트리는 비맥락(non-contextual)**.
- **정량화**: "잔여 CF로도 학습격차 gap ≤ **2.9e-9**"(CF²법칙·c=1/8 경계근사) — 실질 0. → **IoT 꺼진 앵커 정박 완료**(IoTJ 스코프: contextuality 자원 부재 → 양자 생성모델 우위 여지 없음).
- 유의 CF 뜬 셀 없음(def-b 최대 8e-5 = 유한표본 잡음, no-sig 잔차와 동급 → 교란 아닌 0). 성급한 해석 불필요.
- **곡선상 위치**: KCBS(0.263)·Delft(0.211) 켜진 앵커 반대편, CF≈0 꺼진 앵커. gap≈c·CF² 곡선 원점 근방에 실 IoT 정박 → 곡선 양 끝(켜짐/꺼짐) 모두 실데이터 확보.

## 한계 (필수)
- **N-BaIoT 3기기(단일 저장소)**: TON_IoT/BoT-IoT 는 CloudStor 폐지로 미확보(2데이터셋 미만 폴백 규칙 적용, 정직). benign 전량 사용(≤5만행, 행샘플 상한 미도달).
- **정적 결합분포는 이론상 CF=0**(측정=정박/정량화, 발견 아님). (a)(b) 정의는 **조작적 구성**(1차 contextuality_v0/gate1 준용). 중앙값 이진화·부분집합 선택 의존(단 5부분집합·2정의서 일관 0).
- c=1/8 은 **경계근사(CF≤0.1)** — 여기선 CF≈0 이라 gap 상한이 매우 타이트(2차 소멸). 공격(.rar) 미포함(benign만). **관측/앵커이지 quantum supremacy 아님**(모델클래스 분리).

## git 상태
- 신규(untracked): `IOTJ_2차/tools/gate_r3_iot_cf.py`, `IOTJ_2차/results2/gate_r3/*`, `IOTJ_2차/data/nbaiot/*`(다운로드), 본 로그. 기존(K) 무수정 import. 루트 `D` 항목은 1차 폴더이동 잔재(무관). 커밋은 사용자 지시 시에만.

## [해석입력용 요약 — 웹 AI 전달]
1. **실제 IoT 텔레메트리(N-BaIoT UCI, MD5 검증, 3기기: Danmini/Ecobee/Ennio benign) 에서 CF≈0 실측 확정**: def-a(feature-block) 정확히 0.0, def-b(시간창) ≤8e-5, 전 18셀 CF CI 상단 최대 0.00015<0.05, no-sig 잔차 ≤1e-4.
2. **CF²법칙으로 잔여 gap 상한 = c·CF²_upper ≤ 2.9e-9 (c=1/8 경계근사)** = 실질 0 → **IoT 꺼진 앵커 정박**(contextuality 자원 부재 → 양자 생성모델 우위 여지 없음). gap≈c·CF² 곡선의 원점 근방 실데이터.
3. **곡선 양 끝 실데이터 확보**: 켜진 앵커 KCBS(0.263)·Delft(0.211) ↔ 꺼진 앵커 IoT(≈0). IoTJ 스코프 정박 완료.
4. **비단정·한계**: TON/BoT=CloudStor 폐지로 미확보(N-BaIoT 3기기로 폴백)·정적결합=CF0 자명(정박목적)·(a)(b) 조작적·benign만·경계근사. supremacy 아님(모델클래스 분리).
