# 게이트1 — IAT 시퀀스에 진짜 contextuality가 존재하는가 (정적=0 반례 확인)

- 작성: 2026-07-03 (KST). 작업 디렉터리 `/home/elicer/IOTJ/qcbm`. **측정 전용(새 학습 없음)**, 결정적(seed=31337), 무설치(struct·numpy·scipy만). 단정 금지.
- 근거: arXiv:2507.11604(sequential k) + Abramsky-Brandenburger sheaf contextuality(전역절편 LP·contextual fraction). 개념·알고리즘 출처=그들, IAT/트래픽 적용=우리.
- 재현: `python3 tools/gate1_iat_contextuality.py --bins 3,4`.

## 게이트0 (선행) — PASS
- 데이터: 흐름 CSV(CIC/UNSW)는 **집계 IAT 통계(Mean/Std/Max/Min/Tot)만** 있어 패킷 시퀀스 복원 불가. 그러나 `samples/*.pcap`(Tinba=공격, Facetime/iscx=benign)에 **패킷 타임스탬프 존재**.
- `dpkt` 미설치 → **struct 직접 파싱**(pcap record 헤더 ts_sec/ts_usec)으로 패킷 타임스탬프→IAT 시퀀스 복원. 무설치. → **게이트0 PASS**(Tinba n=22000, email n=30409, facebook n=40000 등).
- ※ 이 pcap 들은 USTC-TFC2016/ISCX 샘플 — 본 프로젝트 QCBM이 쓴 CIC/UNSW **흐름 데이터와 다른 소규모 데이터**(개념검증, 일반화 아님).

## 정의 T-b (시간창=context, 삼각형 3-cycle 커버)
- IAT 로그-분위 d빈 이산화. 변수 V={0,1,2}=연속 3위치 IAT 빈. 컨텍스트 C01=(X_t,X_{t+1})[lag1], C12=(X_{t+1},X_{t+2})[lag1], C02=(X_t,X_{t+2})[lag2], 각각 시퀀스 lag 풀서 **독립 추정**(단일 결합 강요 안 함).
- **no-signalling 위반**=공유변수 marginal 컨텍스트 간 L1. **CF**=전역절편 LP 실현불가 정도(1−max 비접촉질량, scipy.linprog).
- 정적 대조=IAT **셔플(순서 파괴=iid)** 후 동일 측정.
- (T-a 폴백은 no-sig 위반 지배 시 사용 — 아래처럼 위반 미미해 **미발동**.)

## CF-LP self-check (도구 유효성) — 통과
| 입력 empirical model | CF | 기대 |
|---|---|---|
| 좌절 삼각형(완전 반상관, maximally contextual) | **1.000** | >0 ✓ |
| 정합 삼각형(완전 상관) | 0.000 | =0 ✓ |
| 부분(0.75 반상관) | 0.250 | 0<CF<max ✓ |
→ **LP는 contextuality가 있으면 CF>0 반환** 확인. 따라서 IAT의 CF≈0은 "LP가 항상 0"이 아닌 **의미 있는 음성**.

## 결과 표 (pcap × d: 시퀀스 CF / 정적 CF / no-sig 위반)
| pcap | d | 시퀀스 CF | 정적 CF | 시퀀스 no-sig | n |
|---|---|---|---|---|---|
| Tinba (attack) | 3 | 0.0000 | 0.0000 | 0.0001 | 21999 |
| Tinba (attack) | 4 | 0.0000 | 0.0000 | 0.0001 | 21999 |
| Facetime (benign) | 3 | 0.0002 | 0.0002 | 0.0002 | 5953 |
| aim_chat (benign) | 3 | 0.0008 | 0.0008 | 0.0013 | 1242 |
| aim_chat (benign) | 4 | 0.0008 | 0.0006 | 0.0013 | 1242 |
| email (benign) | 3/4 | 0.0000 | 0.0000 | 0.0000 | 30409 |
| facebook (benign) | 3/4 | 0.0000 | 0.0000 | 0.0000 | 39999 |
- **모든 pcap·모든 d: 시퀀스 CF ≈ 0**(최대 0.0008=유한표본 잡음). **정적 CF ≈ 시퀀스 CF**(차이 없음). **no-sig 위반도 ≈ 0**.
- 공격(Tinba) vs benign 차이 없음(둘 다 CF≈0). d=3/4 민감도 없음(견고). (Facetime/email d=4 요청 시 0-IAT 다수로 유효빈 d_eff=3 붕괴 — 표기.)

## 판정
- **주장 B 사망**: **시퀀스 CF ≈ 0** (전역절편 존재=실현가능) → **동역학도 non-contextual**. "동역학이 contextuality를 살린다" **미실증**.
- 정적 CF도 ≈ 0 → 가설 "시퀀스 CF>0 & 정적 CF≈0"의 **전자가 성립 안 함**(둘 다 0).
- **no-sig 위반 미미**(≤0.0013) → signalling 지배도 아님 = **깨끗한 CF≈0**(T-a 폴백 불필요).
- **결론**: 게이트1 **불통과** → **A의 척추 강화("동역학으로도 contextuality 불가")**. IAT 시퀀스는 (이 T-b 커버·이 pcap서) 진짜 contextuality 없음.
- ★ 이론적으로도 예상: **(준)정상 확률과정은 전역 법칙(process law) 보유 → 모든 창-주변분포 정합 → CF=0**. 측정이 이를 확증.

## 한계 (필수)
- **empirical-model 변환은 우리 정의(자의성)**: T-b 컨텍스트를 full 시퀀스 translation-invariant 로 추정 → 정상성 하 CF=0이 **구조적으로 유도**됨(정확한 이론적 답이나, 이 구성으론 CF>0 불가). 진짜 CF>0엔 **공동측정 불가한 비-정상 슬라이스**가 필요하나 그러면 **no-sig 위반=signalling**(contextuality 아님). → **고전 트래픽은 no-signalling contextuality 원리적으로 불가**(정적·시퀀스 공통).
- **CF는 LP 상한/근사**(self-check로 유효성만 확인). **소규모 개념검증**(pcap 5개, flow 미분리, USTC/ISCX≠CIC/UNSW) — 일반화 아님. **관측이지 인과 아님. 양자 우위·성능 주장 없음.**
- IAT 이산화(로그-분위 d=3/4) 의존. Facetime/email d=4 붕괴(0-IAT 다수).

## 생성 파일 / 실행 / git
- 신규: `tools/gate1_iat_contextuality.py`, `results2/gate1_ctx/{gate1.json,gate1.png}`, 본 로그.
- 실행: `py_compile` OK; `--smoke` OK; self-check(좌절삼각형 CF=1.0) OK; 본실행 5 pcap×{3,4}, 수 분, 단일 프로세스, 새 학습 없음.
- `git diff --stat`=기존 무변경. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스·데이터·pcap 원본(읽기만) 무수정.

## [해석입력용 요약 — 웹AI 전달]
게이트0 통과(pcap struct 파싱으로 IAT 시퀀스 복원, 무설치). 게이트1: T-b(시간창=context 삼각형) contextual fraction LP로 측정 → **5개 pcap(Tinba 공격 포함) 모두 시퀀스 CF≈0(≤0.0008), 정적(셔플) CF도 ≈0, no-sig 위반 ≈0**. LP는 좌절삼각형서 CF=1.0 반환(유효성 확인)이라 이 CF≈0은 의미 있는 음성. → **동역학도 non-contextual = 주장 B 사망, A 척추 강화**. 이론적으로 (준)정상 확률과정은 전역법칙 보유→CF=0; 진짜 contextuality엔 비-정상 공동측정불가 시나리오 필요하나 그건 signalling. 소규모 개념검증·이 커버·pcap 한정, 우위 주장 없음.
