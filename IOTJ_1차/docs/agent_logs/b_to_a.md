# B→A 확보 — contextuality 있는 실제 데이터에서 (양자)우위 실증 (문헌 + 실데이터 2층)

- 작성: 2026-07-03 (KST). 작업 디렉터리 `/home/elicer/IOTJ/qcbm`. **측정 전용**, 결정적, 무설치(numpy·scipy), self-check(CF-LP). 단정 금지·이 세팅 한정·우위 주장 없음.
- 논리: 메인결론 "실보안데이터 CF=0 → 양자우위 없음"의 **대우 "CF>0 → 우위"** 를 실세계서 확립 시도. ★인공생성 금지(도구 self-check 제외) — 실제 존재 데이터/결과만 증거.
- 재현: `python3 tools/prove_b_to_a.py`. 웹조사(층1·후보 공개성)는 아래 출처.

## self-check (CF-LP 유효성) — 통과
좌절 삼각형(maximally contextual) CF=**1.000**, 정합 삼각형 CF=**0.000** → LP가 contextuality 감지 확인(valid=True).

## 층1 — 문헌 기반 B→A (실험 없이)
| 논문 | 물리계 | contextuality 측도 | 보안 우위 | 실데이터 | no-signalling |
|---|---|---|---|---|---|
| arXiv:1512.02256 (Troupe-Farinholt 2015) | 단일 큐빗 QKD(이론) | Spekkens 준비/측정 contextuality(weak-value POVM) | **키레이트 ≥ BB84 + detector 공격 면역** | 이론 | prepare&measure |
| Nature Sci.Rep. srep01627 (2013) | 실험 단일계 | KS/KCBS 위반 | **인증된 양자난수(min-entropy)** | 예 | 단일계 contextuality |
| Phys.Rev.Applied 13,034077 (2020) | 실험 단일계 | contextuality 부등식 | **Randomness Expansion** | 예 | contextuality |
| Sci.Adv. abk1660 / PMC8827658 (2022) | 실험: 두 원자이온종 (loophole-free) | KCBS 주기형, **C=2.526>2** | 상태독립 contextuality 실증(난수/보안) | 예 (본 도구 CF>0 재확인) | loophole-free·state-indep |
| arXiv:2601.08392 (on-chip semi-DI QRNG) | 실험: on-chip 광자 | contextuality(semi-DI) | genuine randomness 21.7 bit/s, min-ent 0.077/round | 예 | semi-DI |

**층1 판정**: **실제 양자실험에서 contextuality(KCBS/Bell 위반)→보안우위(인증난수·키레이트≥BB84·detector면역·randomness expansion)가 문헌에 확립**. → B→A의 **원리축 확보(인용 가능)**.
★ 단 그 우위축 = **DI-난수/키레이트(양자 측정·암호)** 이지, 우리 프로젝트의 **생성모델 KL/AUC(고차 분포 학습) 축과 다름**.

## 층2 — 실제 공개데이터로 우리축 재확인 (게이트)
### 실데이터 CF>0 확인 (실측 KCBS 상관, 인공 아님)
PMC8827658/Sci.Adv. abk1660 Table1 **실측 상관**: ⟨Ô₀Ô₁⟩=0.6164, ⟨Ô₁Ô₂⟩=0.625, ⟨Ô₂Ô₃⟩=0.6678, ⟨Ô₃Ô₀⟩=−0.6166 → C=**2.526>2**(비접촉 한계). 균등 marginal 재구성 empirical model에 CF-LP:
- **CF = 0.2629 (>0)**, no-sig = 0.0000 (state-independent) → **실제 contextual 데이터서 CF>0 실증**. gate1의 malware/IAT(CF≈0)와 **동일 도구가 실 양자데이터선 CF=0.26** → 도구가 실세계 contextuality를 변별.

### 후보 게이트표 (공개성 / 도메인 적합)
| 후보 | (a) 공개다운로드 | (b) CF>0 | (c) 양자>고전 KL/AUC | (d) 인과(제거시 소멸) | 판정 |
|---|---|---|---|---|---|
| 1) loophole-free Bell (NIST-2015) | 공개(NIST+AWS, Zenodo 15655461/509) | >0(CHSH) | **부적합(물리 측정)** | 부적합 | 탈락: 도메인 불일치 |
| 2) KCBS 2-종 이온 (ETH TIQI/Sci.Adv.) | 공개(Table1 실측·ETH raw) | **>0 (CF=0.26 확인)** | **부적합** | 부적합 | **b통과, c/d 도메인 불일치** |
| 3) 공개 QRNG 비트스트림 | 공개 | 후처리로 구조 소거=검증불가 | 부적합 | 부적합 | 탈락 |
| 4) 공개 QKD sifted-key/QBER | 일부/요청필요 | CF 직접측정 대상 아님 | 부적합 | 부적합 | 탈락 |
| 5) IBM Q 공개 contextuality | 계정/실행 필요(비-정적) | >0 가능 | 부적합 | 부적합 | 탈락 |

**층2 판정**: 실제 contextual 공개데이터(KCBS/Bell) **존재·다운로드 가능(a통과)·CF>0 실측(b통과)**. 그러나 이들은 **양자물리 측정데이터**라 (c)양자>고전 생성/탐지 성능·(d)contextual 제거 인과 축과 **도메인 불일치** → **우리 축(생성모델 KL/AUC)에서의 실증은 미완**(실 contextual **보안-탐지** 데이터셋 부재).

## 종합 판정 (단정 금지)
- **B→A는 원리로서 확립**(문헌 인용): 실 양자실험서 contextuality→보안우위 실증됨.
- **그러나 classical 보안-데이터 생성모델링 축으로 전이 안 됨**: contextuality는 **양자 측정 시나리오의 성질**이고, classical 보안데이터(malware/traffic)는 **non-contextual**(gate1 CF=0, gate1 로그). 대우 "CF>0→우위"는 **양자물리/암호에서 참**이나, malware/traffic 같은 classical 데이터엔 **적용 대상(CF>0 데이터)이 없음** → QCBM 무우위 결론과 **정합**.
- **핵심 통찰**: 우리 도구(CF-LP)로 실 양자데이터 CF=0.26 vs 우리 classical 보안데이터 CF≈0 이 **명확히 분리** → "양자우위의 자원(contextuality)이 우리 데이터엔 없다"가 실측으로 뒷받침.

## 선행확인 (비슷하다고 접지 않기)
- 1512.02256 등은 **QKD 키레이트/난수 축**이지 **우리 생성모델(고차 분포 KL) 주장을 정확히 replicate 아님**. 우리 축(생성모델 contextual-우위)의 실증은 문헌서 **미발견** → 우리가 채울 자리이나, **실 contextual 보안-탐지 데이터 부재**가 병목(층2 도메인 불일치가 그 증거).
- 즉 "정확히 우리 것"은 없고(차별점 존재), 다만 원리축은 인접 문헌이 확립.

## 한계 (필수)
- 실데이터 CF는 논문 **실측 상관→균등 marginal 재구성** empirical model(상관은 실측, 재구성은 가정). CF는 LP 상한. no-sig≈0은 state-indep 설계 반영.
- 층2 gate (c)/(d) 미충족은 **도메인 불일치**(물리 측정 ≠ 보안 탐지/생성)이지 우위 부재 증명 아님 — "우리 축의 실 데이터가 없다"는 사실 보고.
- 웹조사 기반 문헌표(2차 출처 포함). **관측·인용이지 인과 아님. 양자 우위·성능 주장 없음(이 세팅 한정).**
- 인공데이터는 증거 아님(도구 self-check만 합성).

## 생성 파일 / 실행 / git
- 신규: `tools/prove_b_to_a.py`, `results2/b_to_a/b_to_a.json`, 본 로그.
- 실행: `py_compile` OK; 실행 OK(self-check valid, 실 KCBS CF=0.2629). 무설치, 결정적, 새 학습 없음.
- `git diff --stat`=기존 무변경. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 무수정.

## [해석입력용 요약 — 웹AI 전달]
B→A 2층: **층1** 문헌서 실 양자실험(KCBS/Bell 위반, PMC8827658 C=2.526; QKD 1512.02256 키레이트≥BB84·detector면역; 인증난수 srep01627/PRApplied)이 contextuality→보안우위를 확립 — 단 **DI-난수/키레이트 축**(우리 생성모델 KL/AUC 축과 다름). **층2** 실 contextual 공개데이터(KCBS/Bell) 존재·CF>0 실측(우리 CF-LP로 실 KCBS **CF=0.26**, 동일 도구가 classical 보안데이터선 CF≈0) — 그러나 물리 측정데이터라 (c)양자>고전 생성성능·(d)인과제거 축과 **도메인 불일치**(실 contextual 보안-탐지 데이터셋 부재). **결론: B→A는 원리로 확립되나 classical 보안-생성모델 축으로 전이 안 됨**(자원 contextuality가 classical 데이터엔 없음=gate1 CF0와 정합). 이 세팅 한정·우위 주장 없음.

## 출처
- arXiv:1512.02256 (Contextuality-Based QKD). Nature Sci.Rep. srep01627 (Certified randomness via contextuality). Phys.Rev.Applied 13,034077 (Randomness Expansion via contextuality). Sci.Adv. abk1660 / PMC8827658 (loophole-free KCBS, 2-species ions, C=2.526). arXiv:2601.08392 (on-chip semi-DI QRNG). NIST Bell Test Data (nist.gov/pml/applied-physics-division/bell-test-research-software-and-data; Zenodo 15655461/15655509). ETH TIQI public datasets (tiqi.ethz.ch).
