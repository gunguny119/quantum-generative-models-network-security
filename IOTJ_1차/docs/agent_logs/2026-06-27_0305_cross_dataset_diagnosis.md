# 교차 데이터셋 재현 진단 — joint>marginal·복잡도·적합도탐지 해리가 UNSW에서도 성립하는가

- 작성: 2026-06-27 03:05 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 빈도/SVD 기반(QCBM 학습 없음). raw MWU. ★전수(샘플링 없음). 판정은 웹 AI.

## 1. 작업 목적
모든 결론(joint>marginal, 복잡도-탐지, 적합도-탐지 해리)이 CIC-IDS2018 한 데이터셋에서만 나왔다.
독립 데이터셋 UNSW-NB15에서 **같은 방법론·같은 공통 feature 세트**로 재현되는지 검증한다. UNSW엔
TCP 플래그(SYN/PSH)가 없으므로 양쪽 공통 feature(proto+size+fwd_iat+flow_iat=10bit)만 쓰고,
공정 비교를 위해 CIC도 동일 세트로 재계산(플래그 제거가 교란변수 안 되도록).

## 2. 시작 git status, 보존 대상
`git status --short`: `?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip` (전체 untracked).
`git diff --stat`(tracked)=비어 있음. 보존 대상: 모든 기존 소스/데이터/결과(import/읽기만).
생성: `tools/diagnose_cross_dataset.py`, `results2/cross_dataset_diagnosis.{json,png}`, 본 로그.

## 3. 확정한 공통 feature 세트 (억지 매핑 없음)
| 우리 feature | 비트 | CIC 컬럼 | UNSW 컬럼 | 상태 |
|---|---|---|---|---|
| proto | 2 | Protocol(숫자) | proto(문자열 tcp/udp) → tcp=6,udp=17,기타=−1 후 proto_code | **정확(의미)** |
| size | 4 | Pkt Len Mean | smean (src 평균 패킷크기) | **근사** |
| fwd_iat | 2 | Fwd IAT Mean | sinpkt (Source interpacket arrival, mSec) | **근사~정확** |
| flow_iat | 2 | Flow IAT Mean | dinpkt (Dest interpacket arrival, mSec) | **불확실** — 전체흐름 단일 대응 없음(대안: rate, sinpkt+dinpkt) |
| ~~syn/psh~~ | — | SYN/PSH Flag Cnt | **UNSW 부재** | 양쪽 제외 → CIC도 10bit 재계산 |

인코딩 구조 동일: proto2 + size4(linear 16버킷) + fwd_iat2/flow_iat2(log1p 4버킷) = 10bit(1024상태).
경계는 각 데이터 benign train(seed=7, 0.7)에서 생성(데이터 간 재사용 안 함). 캐노니컬 컬럼명으로
변환해 기존 `discretize`/`size_edges_from_benign`/`iat_to_bits` 그대로 재사용.
**불확실성**: flow_iat=dinpkt는 UNSW dinpkt에 0값 다수(목적지 무응답 흐름)이나, 활용률 1.0으로
이산화 비퇴화(0이 한 버킷, 나머지 분산). 의미상으로는 "역방향 도착간격"이라 CIC의 전체흐름 IAT와
완전 일치하지 않음 — 결과 해석 시 유의(억지 매핑 아님, 가장 합리적 단일 대응으로 진행 + 명시).

## 4. 샘플 수 (★전수)
- **UNSW** (train 175,341 + test 82,332 합본, dropna 0): benign(Normal) **93,000** (train 65,100 / test 27,900).
  공격별: Generic 58,871 · Exploits 44,525 · Fuzzers 24,246 · DoS 16,353 · Reconnaissance 13,987 ·
  Analysis 2,677 · Backdoor 2,329 · Shellcode 1,511 · Worms 174 · (ALL 164,673).
- **CIC_HOIC** (cic_0221_ddos.csv): benign 360,833 (train 252,583) / attack(ddos-hoic) 686,012.
- **CIC_Bot** (cic_bot_0203.csv): benign 762,384 (train 533,668) / attack(bot) 286,191.

## 5. 확실한 사실 / 추정 / 확인 불가
**확실(재계산)**: §6 비교표·§7 해리·§8 분류의 모든 수치. 13초 내 전수 계산, dropna 0.
**추정(확인됨)**: "플래그 제거가 HOIC을 크게 바꾼다"는 예상 → 사실(HOIC emp 0.926→0.777). Bot은 거의
불변(0.875→0.839) — 사전 "Bot 신호는 플래그 무관" 추정과 일치.
**확인 불가/한계**: flow_iat=dinpkt 의미 정합(상기). UNSW 공격별 정의가 CIC와 달라 절대 AUC 직접
비교는 부적절(데이터별 benign 기준이라 상대 패턴 비교가 타당).

## 6. 비교표 (동일 10-feature 세트, 전수)
benign 복잡도(데이터별): HOIC reach(TV≤0.05)=**2**, cumE(r3)=0.999, 점유 8.30% → 가장 단순/저랭크.
Bot reach=**8**, cumE(r3)=0.915, 점유 24.51%. UNSW reach=**12**, cumE(r3)=0.705, 점유 16.89% → 최고랭크/최복잡.

| 데이터/공격 | joint이득 | emp_joint | marginal | 최대단일AUC | sp(TV,AUC) |
|---|---|---|---|---|---|
| CIC_HOIC/hoic | **+0.2637** | 0.7773 | 0.5136 | size 0.635 | +0.29 |
| CIC_Bot/bot | **+0.5531** | 0.8389 | 0.2858 | proto 0.360 | −0.82 |
| UNSW/Analysis | +0.0735 | 0.8478 | 0.7744 | proto 0.835 | +0.17 |
| UNSW/Backdoor | +0.0796 | 0.9673 | 0.8878 | proto 0.890 | −0.99 |
| UNSW/DoS | +0.0826 | 0.9178 | 0.8352 | proto 0.848 | −0.99 |
| UNSW/Exploits | +0.2535 | 0.7913 | 0.5378 | flow_iat 0.652 | −1.00 |
| UNSW/Fuzzers | +0.1215 | 0.6496 | 0.5281 | flow_iat 0.700 | −0.98 |
| UNSW/Generic | +0.1122 | 0.9930 | 0.8808 | flow_iat 0.872 | −0.95 |
| UNSW/Reconnaissance | +0.2637 | 0.8436 | 0.5799 | flow_iat 0.740 | −0.89 |
| UNSW/Shellcode | +0.1405 | 0.7385 | 0.5979 | flow_iat 0.712 | −0.82 |
| UNSW/Worms | +0.3934 | 0.6961 | 0.3028 | size 0.711 | −0.92 |
| UNSW/ALL | +0.1612 | 0.8624 | 0.7012 | flow_iat 0.766 | −1.00 |

**joint>marginal: 12/12 모두 양(+)** — 모든 UNSW 공격에서 joint이 marginal보다 우수.

**플래그 제거 효과(동일 세트 재계산 대비 기존 12bit)**:
- HOIC: 12bit emp 0.9258/marg 0.8741, psh단독 0.878 → **10bit emp 0.7773/marg 0.5136, 최대단일 size 0.635**.
  psh 플래그가 HOIC 탐지를 거의 혼자 만들었음이 확증(제거 시 emp −0.15, marg −0.36, 단일지배 소멸).
- Bot: 12bit emp 0.8749/marg 0.2870 → **10bit emp 0.8389/marg 0.2858** (거의 불변). Bot 신호는 플래그 무관.

## 7. 해리 진단 (저랭크 TV vs AUC)
방법: benign q_train의 32×32 SVD 저랭크 근사 r=1..20, 각 r의 (TV, 공격탐지 AUC). 두 지표로 본다 —
(i) spearman(TV,AUC): +면 적합 나쁠수록 AUC↑(해리), −면 적합 좋을수록 AUC↑(정렬).
(ii) 저랭크 최대 AUC vs 전체 emp AUC 갭: +면 "최적합(전체)이 최적탐지 아님"(해리).

| 데이터/공격 | sp(TV,AUC) | 저랭크maxAUC−emp | 해석 |
|---|---|---|---|
| CIC_HOIC | **+0.29** | **+0.0746** | **해리** — 저랭크(rank~2)가 전체보다 탐지 우수 |
| UNSW/Analysis | **+0.17** | **+0.0396** | **약한 해리** |
| CIC_Bot | −0.82 | +0.0018 | 정렬(적합↑=탐지↑) |
| UNSW(그 외 8공격+ALL) | −0.82~−1.00 | ≤+0.004 | 정렬 |

**즉 "최적합≠최적탐지" 해리는 HOIC에서 뚜렷, UNSW/Analysis에서 약하게 재현되고, Bot과 나머지
UNSW 공격에서는 재현되지 않음(오히려 적합-탐지 정렬).** 단, 이 고전 저랭크-스윕 해리는 QCBM
restart 해리(고정 랭크에서 모델 16개의 적합도가 AUC 무관)와는 **다른 조작화**다 — 직접 동치 아님.
저랭크 스윕은 랭크↑가 전체 emp로 수렴하므로 본질적으로 정렬 경향이며, 그럼에도 HOIC은 과적합형
해리를 보임(전체 joint이 benign에 과적합돼 탐지가 되레 낮음).

## 8. UNSW 공격 분류
- **복잡+joint 의존(joint이득 큼, feature가 포착)**: Exploits(+0.25, emp 0.79), Reconnaissance(+0.26, 0.84),
  Worms(+0.39, 0.70; 단 n=174 소표본), ALL(+0.16, 0.86).
- **단순(이미 쉬움 — 단일/marginal로 충분, joint이득 작음)**: Generic(gain 0.11, emp 0.99, flow 0.87),
  Backdoor(0.08, 0.97, proto 0.89), DoS(0.08, 0.92, proto 0.85), Analysis(0.07, 0.85, proto 0.83).
- **feature 부적합(emp_joint 낮음 — 우리 feature로 잘 안 풀림)**: **Fuzzers(emp 0.65, 최저)**, Shellcode(0.74).
  → "재현 실패"가 아니라 proto/size/IAT 4-feature가 이 공격의 신호를 충분히 담지 못함(다른 컬럼에 신호).

## 9. 사실 관찰 (판정 아님)
- **joint>marginal 재현**: ✅ **강하게 재현** — CIC(HOIC/Bot) 및 UNSW 9개 공격 전부에서 joint이득>0(12/12).
  독립 데이터셋·다른 추출도구에서도 "결합분포가 주변분포보다 낫다"가 성립.
- **복잡도-joint의존 관계 일관성**: △ **부분적**. benign 복잡도(랭크)는 HOIC<Bot<UNSW로 명확. 그러나
  joint이득의 크기는 benign 랭크와 단조 일치하지 않음(Bot이 랭크 더 낮은데 이득 최대 +0.55). joint이득은
  benign 복잡도뿐 아니라 공격-benign 분리구조·marginal 기저값에 함께 의존하기 때문(예: Bot marginal
  0.29로 매우 낮아 여지 큼). "복잡할수록 joint 의존↑"은 CIC 내부에서는 방향성 성립, 교차 데이터 단조
  법칙으로 단정 불가.
- **적합도-탐지 해리 재현**: △ **제한적**. 고전 저랭크 조작화에서는 HOIC(과적합형, 저랭크>전체 +0.075)와
  UNSW/Analysis에서만 뚜렷/약하게 재현, 나머지는 적합-탐지 정렬. QCBM-restart 해리와는 다른 측정이므로
  "해리가 UNSW 전반에서 재현"이라 말할 수 없음(정직히 제한적).
- **부수 확증**: 플래그(강신호) 없는 자연 환경인 UNSW에서 joint이득이 보편적으로 양 — "강신호(psh) 제거 시
  joint 가치 부각" 발견과 정합. CIC HOIC도 플래그 빼면 단일지배 소멸하고 joint이득이 드러남.

## 10. 생성 파일, 실행 명령과 결과
- `qcbm/tools/diagnose_cross_dataset.py` (신규)
- `qcbm/results2/cross_dataset_diagnosis.json` (60KB), `.png` (165KB) (신규)
- `docs/agent_logs/2026-06-27_0305_cross_dataset_diagnosis.md` (본 로그)
```
python3 -m py_compile tools/diagnose_cross_dataset.py    # OK
python3 tools/diagnose_cross_dataset.py                  # real 0m13.1s, DONE
# CIC_HOIC dropna=0 benign=360833 / CIC_Bot benign=762384 / UNSW benign=93000, 공격 9종+ALL
```
실패/에러 없음. 메모리 문제 없음(usecols 로드). 샘플링 없음(전수). scipy.stats.spearmanr(기존 설치) 사용.

## 11. git diff --stat (기존 무변경)
```
$ git diff --stat        # tracked 변경: 없음
$ git status --short
?? docs/   ?? qcbm/   ?? qcbm_share_20260626.zip
```
기존 CIC 데이터(314M/337M/103M)·소스·결과 무변경. QCBM 무학습, dependency 무설치.

## 12. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 공통 feature 세트 확정(억지 매핑 없음, flow_iat 불확실 명시), 양쪽 동일 인코딩 구조 적용.
- [x] CIC(HOIC/Bot) 동일 10-feature 재계산값 보고(플래그 제외 효과 포함).
- [x] UNSW 각 공격+전체에 joint이득/복잡도/단일feature/해리 측정(★전수, 샘플 수 명시).
- [x] 핵심 비교표(joint>marginal 재현, 복잡도-joint 관계, 해리 재현 여부).
- [x] UNSW 공격을 복잡+joint의존/단순/feature부적합으로 분류.
- [x] json/png 저장, 기존 무수정·무학습·무설치.

**남은 불확실성**
- flow_iat=dinpkt 의미 정합(전체흐름 vs 역방향). 대안(rate, sinpkt+dinpkt) 재계산 시 결과 민감도 미검증.
- joint이득과 benign 복잡도의 비단조성 원인(marginal 기저·분리구조 교란) 정량 분리 안 됨.
- Fuzzers/Shellcode "feature 부적합"의 원인 컬럼(어느 UNSW feature에 신호가 있는지) 미조사.
- 고전 저랭크 해리 ≠ QCBM-restart 해리(조작화 차이) — UNSW에서 QCBM-restart형 해리는 미측정(학습 필요).

**다음 추천 단계**
1. flow_iat 대안 매핑(rate 또는 sinpkt+dinpkt 평균)으로 재계산해 결과 민감도 점검(불확실성 해소).
2. Fuzzers/Shellcode에 대해 UNSW 고유 feature(sttl/state/swin 등) 단독 AUC를 조사해 "어느 신호가
   빠졌는지" 규명(feature 세트 확장 후보 도출).
3. (양자 단계) UNSW 복잡 공격(Exploits/Reconnaissance/ALL)에서 QCBM 학습 후 restart 해리·216p 효율
   우위가 CIC Bot처럼 재현되는지 — 단 이는 학습 작업(이번 범위 밖).
