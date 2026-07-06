# contextuality(k) 측정 v0 — malware 고차가 non-contextual인지 (정의 b→a, NMF-rank 프록시)

- 작성: 2026-07-03 (KST). 작업 디렉터리 `/home/elicer/IOTJ/qcbm`. **측정 전용(새 QCBM 학습 없음)**, 결정적(seed 고정), 무설치. 단정 금지·판정은 웹 AI.
- 근거: arXiv:2507.11604 strong k-contextuality(개념·k추정 greedy 출처). **empirical-model 변환·도메인·차수대조=우리 정의(자의성 있음).**
- 재현: `python3 tools/contextuality_v0.py --splits 0,1,2 --k 10 --tol 0.05` (seed=4242).

## k̂ 조작정의 (v0)
context×outcome empirical model 행렬 **M 의 추정 비음수 랭크**(상대 Frobenius 오차 ≤ tol=0.05 되는 최소 NMF 성분수 r) **− 1** = "E 재현에 필요한 공유 은닉상태(잠재 혼합성분) 최소개수 − 1"의 프록시. NMF=`diagnose_param_efficiency_b2.nmf_factorize`(무수정 재사용, greedy 랭크 증가).
- **정의 b(우선, 가족=context, 지도적)**: M[g,o]=P(o|가족 g), outcome=표적 10비트 상태(가족 합집합 support). label 사용=**지도적**.
- **정의 a(폴백, feature블록=context)**: 10비트 A(5)|B(5) 분할, M[a,b]=P(B=b|A=a). 랜덤 5|5 ×5(seed) 평균.
- 대조 부분집합 3종: 표적(고차)/highvar(저차)/랜덤 × split{0,1,2}. QCBM 상대성능=`qcbm_order_comp`의 gap_QCBM_M2(고차↔targeted, 저차↔highvar).

## 결과 표 (subset × split: ≥3차 | def-b k̂ | def-a k̂ | QCBM gapM2)
| split | subset | ≥3차 | def-b k̂ (rank) | def-a k̂ [min-max] | QCBM gapM2 |
|---|---|---|---|---|---|
| 0 | 표적 고차 | 0.32 | 6 (7) | 15.0 [14-17] | +0.391 |
| 0 | highvar 저차 | 0.15 | 6 (7) | 16.2 [13-21] | +0.731 |
| 0 | 랜덤 | 0.01 | 3 (4) | 2.8 [2-4] | N/A |
| 1 | 표적 고차 | **0.58** | 6 (7) | **21.6 [20-23]** | **−0.178** |
| 1 | highvar 저차 | 0.15 | 6 (7) | 16.4 [13-21] | +0.762 |
| 1 | 랜덤 | 0.04 | 1 (2) | 3.0 [2-4] | N/A |
| 2 | 표적 고차 | **0.59** | 6 (7) | **23.8 [20-26]** | **−0.140** |
| 2 | highvar 저차 | 0.15 | 6 (7) | 16.4 [13-21] | +0.826 |
| 2 | 랜덤 | 0.06 | 2 (3) | 1.8 [1-3] | N/A |

## 정렬 (상관)
- **corr(≥3차, def-a k̂) = +0.834** — 고차일수록 k̂↑. (특히 표적 ≥3차 0.58~0.59서 k̂ 22~24, 저차 16, 랜덤 ~2.5.)
- **corr(def-a k̂, QCBM gapM2) = −0.862** — k̂↑일수록 QCBM이 M2에 근접/앞섬(고차 셀서 QCBM 우세와 정렬).
- **corr(≥3차, QCBM gapM2) = −0.997** — ★ 그러나 **≥3차 자체가 QCBM을 k̂보다 더 잘 예측**. 즉 k̂는 order의 잡음 섞인 버전이라 **order 넘어 추가 설명력 없음**.
- def-b k̂: 구조 부분집합 전부 **6 = 가족수(7)−1 천장 포화**(비변별), 랜덤만 1~3. corr(≥3차,def-b k̂)=0.62(천장 탓 약함).

## 판정 (단정 금지, 판정은 웹 AI)
- **주장 A "우리 고차는 non-contextual(=k 낮음)" — 이 프록시로는 미지지**: def-a k̂가 고차서 오히려 **상승**(표적 0.59→k̂ 24)해 "차수 높은데 k 낮음"이 아님. def-b는 천장 포화라 무변별.
- **주장 B "보안에 양자 유리 조건 실재" — 프록시상 시사되나 확정 불가**: def-a k̂가 order·QCBM 상대성능과 정렬(+0.83/−0.86). 그러나:
  (i) **정적 결합분포는 전역 절편(global section) 항상 존재→자명히 non-contextual**이라 k̂는 **진짜 AB-contextuality 아님**(잠재 혼합성분 수=NMF 랭크 프록시).
  (ii) ≥3차가 QCBM을 k̂보다 **더** 잘 예측(−0.997 vs −0.862) → k̂는 order를 넘는 독립 설명력 없음(공변량).
  (iii) def-b(가족 context)는 no-signalling **위반**(가족 간 주변 L1≈4.1~4.6=지도적/signalling) → 표준 contextuality 프레임 부적용.
- **차수 vs k 분리 — 부분적**: k̂와 ≥3차는 공변(0.83)하나 highvar는 저차(0.15)인데 k̂ 16(중간)·def-b 천장 → k̂ ≠ 순수 order. 다만 "실데이터서 order↔k 깔끔 분리"라는 뚜렷한 반례 셀은 없음(대체로 동조).
- **연결(기둥4)**: k̂는 QCBM 상대성능과 정렬(−0.86)하나 order가 더 강한 예측자 → "보안서 k가 QCBM을 예측"은 **order를 통한 간접**일 뿐, k 고유 기여 미확인.

**종합**: 이 v0(NMF-rank 프록시)에서 "malware 고차 부분집합의 k̂"는 order·QCBM경쟁력과 **양의 공변**(주장 B 방향 시사)하나, ①정적 결합=자명 non-contextual이라 **진짜 contextuality 아님**, ②order가 더 강한 예측자라 **k 고유 설명력 없음**, ③def-b 천장 포화·signalling → **주장 B 확정 불가, 주장 A(순수 저-k)도 미지지 = "프록시상 order와 분리 안 되는 잠재복잡도"로 정직 보고**.

## 폴백 발생 / self-check
- 정의 b가 천장 포화(k̂=6=가족수−1, 전 구조 부분집합)로 변별력 없음 → **정의 a(폴백)로 주 판정**(설계대로 b→a). 랜덤 부분집합은 두 정의 모두 낮은 k̂(비구조=낮은 랭크)로 sanity 통과.
- NMF 랭크 단조: rel 오차 r↑에 감소, tol 교차서 정지(결정적 seed). 랜덤<highvar<표적(고차) 순 k̂ = 구조성 반영 정합.

## 한계 (필수)
- **empirical-model 변환은 우리 정의(자의성)** → 정의 b·a 교차로 방어(단 b는 천장 포화). **k̂=greedy NMF 상한 추정**(정확한 비음수랭크 아님) → "추정 k"로만.
- ★ **정적 결합분포=전역 절편 존재→자명 non-contextual**; def-a는 단일 결합의 조건부라 진짜 contextuality 0. def-b는 가족 간 주변 불일치=**signalling**(비-무신호) → AB-contextuality 프레임 부적용. **k̂는 잠재 혼합복잡도(NMF 랭크) 프록시**이지 contextuality가 아님.
- k=10 표적 부분집합·3 split(9점)·평활 c 의존. **관측 정렬이지 인과 아님. 양자 우위 주장 없음.** def-b는 label 사용=지도적(비지도 아님).

## 생성 파일 / 실행 / git
- 신규: `tools/contextuality_v0.py`, `results2/contextuality/{ctx_v0.json,ctx_v0.png}`, 본 로그.
- 실행: `py_compile` OK; `--smoke`(split0) OK; 본실행 3 split×3 subset×(def-b+def-a 5분할) 수 분, 단일 프로세스, 새 학습 없음.
- `git diff --stat`=기존 무변경. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스·results2 기존 json·Drebin 원본 무수정.

## [해석입력용 요약 — 웹AI 전달]
contextuality v0(NMF-rank 프록시): 표적 고차 부분집합의 추정 k̂(def-a)가 ≥3차 비중과 +0.83, QCBM 상대성능과 −0.86 공변 → 표면상 "고차=고-k, 양자유리(주장 B)" 방향. 그러나 ①정적 결합분포는 전역절편 존재로 **자명 non-contextual**이라 k̂는 진짜 contextuality가 아닌 잠재복잡도(NMF 랭크)이고, ②≥3차가 QCBM을 k̂보다 더 강히 예측(−0.997)해 k 고유 설명력 없음, ③가족-context(def-b)는 signalling·천장 포화. 결론: 주장 B 확정 불가·주장 A(저-k) 미지지 — "실데이터서 k̂는 order와 분리 안 됨". 진짜 contextuality 판정엔 비-정적(측정 disturbance) 시나리오 필요. 이 부분집합·프록시·3점 한정, 우위 주장 없음.
