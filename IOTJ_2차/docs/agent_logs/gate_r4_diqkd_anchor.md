# Gate R4 — DI-QKD/보안 프로토콜 원시데이터를 세 번째 켜진 앵커로? (웹조사 → RED, 게이트 스킵)

- 작성: 2026-07-05 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **조사 전용**(웹 검색). 데이터 다운로드·훈련·앵커 추가 **없음**(GREEN 후보 없어 실행 조건 미충족).
- 목표: KCBS/Delft 켜진 앵커를 "실제 양자 **보안 프로토콜**(DI-QKD/DI-randomness) 실행 데이터"로 업그레이드 가능한지 — trial-level 공개 + CHSH + 우리 CF-LP로 재구성 가능한지 확인.
- 원칙: 접근 못 하는 데이터를 "있다"고 단정 금지. **확인된 것만 GREEN.** supremacy 아님.

## 판정 기준
- GREEN: CHSH 기반 + trial-level(setting x,y·outcome a,b) 공개 + 재구성 가능 → 세 번째 앵커 추가.
- YELLOW: 공개됐으나 집계값만/Eberhard형 → 정성 인용만.
- RED: 미공개/접근 불가 → 현 앵커 유지, DI-QKD는 인용 근거로만.

## 후보별 조사 결과 (우선순위대로)
| 후보 | 출처(확인 링크) | Bell 종류 | 데이터 형태 | 판정 | 코멘트 |
|---|---|---|---|---|---|
| **Nadlinger 2022** (Oxford, 갇힌 이온 DI-QKD, Nature 607:682; arXiv:2109.14600) | ORA `uuid:604c53b9`(=**박사학위논문 PDF만**), Nature(페이월) | CHSH | **thesis PDF만**, 원시 trial 데이터 미공개 | **RED** | ORA엔 학위논문 11.5MB뿐, dataset DOI/Zenodo 없음(확인) |
| **van Leent 2022** (Munich, 단일원자 100km DI-QKD, Nature 607:687) | nature(페이월 PDF), 학위논문(edoc LMU) | CHSH | **집계값만 확인**(1.2M heralded pairs, key rate 0.112 bit/event); 공개 trial repo **미확인** | **RED**(공개 trial 미확인; 페이월로 data-availability 확인 불가) | CHSH지만 trial-level 공개 근거 못 찾음 |
| **Liu 2022** (photonic DI-QKD, PRL 129:050502; arXiv:2110.01480) | PRL/arXiv | CHSH(generalized) | proof-of-concept, 공개 trial repo **미확인** | **RED** | 미확인 |
| **Rosenfeld 2017** (Munich, 원자 event-ready **CHSH Bell**, PRL 119:010402; arXiv:1611.04604) — 대안 | PRL/arXiv | CHSH | 원시데이터 공개 **미확인** | **RED** | 대안 CHSH Bell도 공개 근거 없음 |
| (참고) **Delft Hensen 2015** — 이미 사용 | 4TU DOI 10.4121/…6E19E9B2 (공개, MD5검증) | CHSH | **trial-level 공개** | GREEN(이미 앵커) | loophole-free CHSH Bell = DI 보안의 정초 실험 |
| (참고) **NIST 2015** — R1서 확인 | Zenodo=분석 PDF만 | Eberhard/CH | trial 재구성 불가 | YELLOW | Eberhard형 → 우리 CF-LP 비호환 |

## 핵심 관찰
- **DI-QKD 실험(3종: Oxford 이온·Munich 원자·photonic)은 공통적으로 trial-level 원시데이터를 공개 저장소에 올리지 않음** — 학위논문 PDF·집계 통계(CHSH S·QBER·key rate)·코드만. loophole-free **Bell 실험**(Delft=event list, NIST=timetag)과 달리 DI-QKD **런 데이터**는 공개 관행이 아직 아님(확인 범위 내).
- Munich DI-QKD(van Leent)는 Rosenfeld 2017 CHSH Bell과 **동일 원자 플랫폼** — 즉 우리가 이미 쓴 Delft(CHSH loophole-free Bell)가 **DI-QKD 보안이 딛고 선 바로 그 실험 유형**.

## 최종 권고 — **업그레이드 시도 가치 있는 GREEN 후보: 없음 (no)**
- 근거: 3개 DI-QKD + 대안 CHSH Bell(Rosenfeld) 모두 **trial-level 공개 미확인** → GREEN 0건. (접근 못 한 데이터를 "있다"고 하지 않음.)
- 따라서 **Gate R4 실행 스킵**: 데이터 다운로드·CF 측정·훈련·세 번째 앵커 추가 **하지 않음**. 현 앵커 **KCBS(CF0.263)·Delft(CF0.211) 유지**.
- **Q1(보안 연관성) 처리**: DI-QKD는 **Discussion 인용 근거**로 사용 — "우리 켜진 앵커(Delft)는 loophole-free **CHSH Bell test** 로, DI-QKD/DI-randomness 보안이 정초하는 바로 그 실험 유형이다. 따라서 CF²-법칙이 실측된 앵커는 이미 device-independent 보안 프리미티브(CHSH 위반)에 직접 얹혀 있다." → DIQKD-전용 앵커 없이도 Q1 실질 해소.

## 한계
- Nature/PRL 본문 data-availability 문구는 **페이월**로 직접 확인 불가(van Leent) → "공개 미확인"으로 처리(있다/없다 단정 아님). 저자 문의·supplementary 확인 시 뒤집힐 여지는 있으나, **현재 공개 다운로드 링크는 확인 안 됨**.
- 웹 검색 기반(2차 출처 포함). GREEN 판정은 실제 다운로드·MD5·파싱까지 됐을 때만 부여(이번엔 0건).

## 생성/변경
- 신규: 본 로그 1개. **tools/gate_r4·results2/gate_r4 미생성**(실행 조건 미충족). 기존 무변경.

## [해석입력용 요약 — 웹 AI 전달]
DI-QKD 3종(Nadlinger 이온·van Leent 원자·Liu photonic, 모두 CHSH)의 **trial-level 원시데이터는 공개 저장소서 확인 안 됨**(학위논문·집계·코드만). 대안 CHSH Bell(Rosenfeld 2017)도 미공개. → **GREEN 0건, Gate R4 스킵, KCBS/Delft 앵커 유지.** 단 우리 Delft 앵커 자체가 loophole-free **CHSH Bell test** = DI-QKD 보안의 정초 실험 유형이라, DI-QKD를 Discussion 인용으로 두면 Q1(보안 연관성)은 실질 해소. 접근 못 한 데이터를 있다고 단정하지 않음, supremacy 아님.
