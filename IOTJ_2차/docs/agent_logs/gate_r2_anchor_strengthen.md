# Gate R2 — 켜진 앵커 강화: (A) KCBS KL_q 실측 (B) Delft Bell 두 번째 실데이터 앵커

- 작성: 2026-07-05 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **측정+훈련**, 결정적(seed=20260705), 무설치(numpy·scipy).
- 재현: `python3 tools/gate_r2_anchor_strengthen.py`. 산출: `tools/gate_r2_anchor_strengthen.py`, `results2/gate_r2/{r2.json, r2.png}`.
- 재사용(무수정): `gap_vs_cf_v0`(=G, CHSH CF-LP·nc_floor·fit_quantum). ★**모델클래스 분리이지 supremacy 아님**.

## 데이터 출처 / 해시
- **KCBS**: PMC8827658 / Sci.Adv. abk1660 Table1 실측 상관 E=[0.6164,0.625,0.6678,−0.6166]±[0.0079,0.0078,0.0074,0.0079].
- **Delft**: Hensen 2015 loophole-free Bell. 4TU DOI `10.4121/uuid:6E19E9B2-4A2D-40B5-8DD3-A660BF3C0A31` → data.zip **MD5=342f29f8288c46575818acd2acebd535 (검증 통과 ✓)**. `bell_open_data.txt`(4746 attempts) → 예제 스크립트 필터 그대로 적용 → 유효 Bell 시행.

## self-check — 통과
CHSH CF-LP: PR-box CF=**1.000**(기대1), uniform CF=**0.000**(기대0). → LP 유효.

## Track A — KCBS KL_q "추론 → 실측"
★ b_to_a KCBS 는 **4-cycle(짝수 사이클 = CHSH 동치)**, 5-cycle 오각별 아님 → 올바른 양자실현 = **2큐빗 CHSH 모델**. 4 상관을 CHSH 컨텍스트로 매핑(S=C 보존).
| 항목 | 값 |
|---|---|
| CF (CHSH LP) | 0.2629 (매핑 self-check: cyclic 0.2629와 일치) |
| NCfloor | 0.0123 |
| **KL_q 실측 (2큐빗)** | **0.00000** (CI [5e-18, 1.3e-11]) |
| **gap = NCfloor − KL_q** | **0.0123** (CI [0.0109, 0.0137]) |
- **판정 A**: KL_q 실측 = 0.00000 ≈ 0 → R1 의 "실측 양자데이터=양자실현 → KL_q≈0" **추론이 측정으로 확정**. gap 실측 0.0123 = **R1 앵커값(0.0123) 유지**(CI 안, 갱신 불필요).

## Track B — Delft Bell 두 번째 앵커
유효 Bell 시행 **n=245** (Hensen 2015 원문 245와 일치), 컨텍스트별 N=[53,79,62,51].
| 항목 | 값 |
|---|---|
| **CHSH S** | **2.422** (CI [1.95, 2.80]) — 원문 2.42와 일치 |
| CF (CHSH LP) | **0.2112** |
| CF from (S−2)/2 (교차확인) | 0.2112 (**일치 ✓**) |
| CF 95%CI (시행 재표집) | [0.115, 0.418] (n=245 → **넓음**) |
| no-signalling 잔차 (L1) | **0.0895** (유의미 — 아래 주의) |
| NCfloor | 0.0129 |
| **KL_q 실측 (2큐빗)** | **0.00446** |
| **gap = NCfloor − KL_q** | **0.0085** (CI [~0, 0.0265]) |

## 곡선 배치 (CHSH 참조 NCfloor(CF), gate_e CHSH c≈1/6)
| 앵커 | CF | gap 실측 | 곡선 예측 | 편차 | on-curve(≤20%) |
|---|---|---|---|---|---|
| **KCBS ion** | 0.2629 | 0.0123 | 0.0128 | **3.7%** | **예** |
| **Delft Bell** | 0.2112 | 0.0085 | 0.0080 | **5.3%** | **예** |
- **두 실측 양자데이터 점이 모두 gap≈c·CF² 곡선 위**(3.7%, 5.3%, 각 부트스트랩 CI 안). → **두 번째 켜진 앵커 확보**.
- c_fit(경계 c·CF²) = N/A(참조 스윕 CF≤0.1 점 부족, 0.05 격자) → gate_e 기존값 CHSH c=1/6=0.167 인용; 실측 CF(0.21~0.26)는 경계 밖이라 완전 곡선(초선형)으로 배치·평가.

## 판정 (P1/P2 강화)
- **A (KCBS 앵커 확정)**: KL_q 실측=0 → gap=0.0123 유지. R1 앵커 값 **측정으로 확정**(추론 아님).
- **B (Delft 두 번째 앵커)**: CF=0.211, gap=0.0085 가 곡선 예측(0.0080)과 **5.3%** → **곡선 위**. CHSH S=2.42·CF=(S−2)/2 교차확인 통과. → **독립적 실물리계(다이아몬드 NV 스핀)에서도 법칙 성립**.
- **두 앵커 종합**: 이온(KCBS, CF0.263) + 다이아몬드(Delft, CF0.211) 두 개의 서로 다른 실험 플랫폼 데이터가 같은 gap≈c·CF² 곡선 위 → 법칙의 실데이터 지지 강화.

## 한계 (필수)
- **Delft no-signalling 잔차 L1=0.0895 유의미**: 유한표본(n=245)·실험 불완전성. no-signalling 위반이 있으면 CF 해석에 주의 필요(2큐빗 모델은 무신호라 KL_q=0.0045 의 일부는 이 잔차 반영). CF·gap 값은 이 잔차하 해석.
- **Delft n=245 소규모** → CF CI 넓음([0.115,0.418]). 점추정은 곡선 위이나 CI 폭 큼.
- KCBS 는 **4-cycle(=CHSH 동치)**로 2큐빗 실현 — 데이터가 4-term 상관이라 5-cycle 오각별 아님(정직). 상관→균등 marginal 재구성.
- NCfloor 수치최소화(다중초기값). c_fit 경계 미산출(gate_e 값 인용). **관측/앵커이지 quantum supremacy 아님**(메모리매칭 모델클래스 분리). Delft 훈련(2큐빗 KL_q)은 시뮬 적합(데이터는 실측).

## git 상태
- 신규(untracked): `IOTJ_2차/tools/gate_r2_anchor_strengthen.py`, `IOTJ_2차/results2/gate_r2/*`, `IOTJ_2차/data/delft/*`(다운로드), 본 로그. 기존(G) 무수정 import. 루트 레포의 `D` 항목은 1차 폴더 이동(ISCX zip)에서 온 것—본 작업과 무관. 커밋은 사용자 지시 시에만.

## [해석입력용 요약 — 웹 AI 전달]
1. **KCBS KL_q 추론→실측 확정**: 4-cycle(=CHSH 동치) 2큐빗 적합 KL_q=**0.00000**(CI [5e-18,1e-11]) → R1 의 "양자실현→KL_q≈0" 확정, gap=0.0123 **유지**(갱신 불필요).
2. **Delft Bell 두 번째 실데이터 앵커 확보**: 4TU 실데이터(MD5 검증, n=245, CHSH S=2.422, CF=(S−2)/2=0.2112 교차확인) → gap=0.0085 가 CHSH 곡선예측(0.0080)과 **5.3%** = 곡선 위. 이온+다이아몬드 두 플랫폼 모두 gap≈c·CF² 위.
3. **주의**: Delft no-signalling 잔차 L1=0.089(유의미, n=245 유한표본)·CF CI 넓음[0.115,0.418]. KL_q=0.0045 의 일부는 이 잔차 반영.
4. **비단정·한계**: KCBS=4-cycle(오각별 아님)·상관재구성·NCfloor 수치최소화·시뮬훈련. supremacy 아님(모델클래스 분리).
