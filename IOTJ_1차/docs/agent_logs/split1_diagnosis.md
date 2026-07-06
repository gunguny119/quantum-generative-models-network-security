# split1 이상 신호 진단 — drift 탐지에서 split1만 고차−pairwise AUC 이득 +0.15인 원인

- 작성: 2026-07-03 (KST). 작업 디렉터리 `/home/elicer/IOTJ/qcbm`. **측정 전용(새 학습 없음).** 기존 도구 무수정(import), 무설치. 단정 금지·혼재면 정직 보고.
- 대상: `results2/drebin_targeted/drebin_targeted.json`의 표적 k=10 부분집합, `results2/drift_auc/drift_auc.json`의 AUC 격차(고차−M2). 재현: `python3 tools/diagnose_split1.py --splits 0,1,2 --k 10` (seed 없음=결정적, 표본수 병기).
- 정의: **unseen**=y_test에만 있는 가족(split0→f0, 1→f1, 2→f2; 양성), **known**=train 가족(음성). 분포는 표적 10비트 부분집합 위 2^10 상태, KL은 Laplace 평활(c=1/2^k). conn3=연결 3점상관.

## 배경(재확인)
AUC 고차−M2: split0 **−0.069** / split1 **+0.152** / split2 **−0.094** (k10). split1만 양. split1의 **M2 절대 AUC도 유독 낮음(0.605 vs 0.889/0.880)**.

## 측정 결과 표 (split별, AUC 이득과 나란히)
| 지표 | split0 | **split1** | split2 | split1이 이상치? (AUC이득과 정렬) |
|---|---|---|---|---|
| **AUC 고차−M2** | −0.069 | **+0.152** | −0.094 | (기준) |
| M2 절대 AUC | 0.889 | **0.605** | 0.880 | ✓ 유독 낮음(저차 판별 약함) |
| **측정1 구조상속** | | | | |
| ladder_rge3 (≥3차 train 상속) | −3.40 | −3.77 | −3.92 | ✗ 단조, split1 중간 (상속으로 설명 안 됨) |
| conn3 jaccard(unseen,known) | 0.284 | **0.000** | 0.062 | ✓ 유독 낮음(3차 top 구조 전혀 안 겹침) |
| conn3 cos(unseen,known) | 0.102 | 0.108 | 0.170 | ✗ split1 중간 |
| **측정2 이탈 차수비중(known base)** | | | | |
| 1차 비중 | 0.29 | **0.45** | 0.37 | ✓ 유독 높음(이탈이 1차에 몰림) |
| 2차 비중 | 0.31 | **0.24** | 0.32 | ✓ 유독 낮음 |
| ≥3차 비중 | 0.40 | **0.31** | 0.31 | ✗ split1 최저 — **H2(고차 이탈 큼) 직접 반증** |
| **측정3 아티팩트** | | | | |
| n_unseen | 925 | 667 | 625 | ✗ split1 중간 |
| known/unseen 비 | 0.52 | 0.79 | 0.86 | ✗ split1 중간 (표본 불균형으로 설명 안 됨) |
| **feat 활성 L1이동(subset,vs train)** | 3.45 | **2.31** | 3.20 | ✓ 유독 낮음(주변분포가 train과 가장 유사) |
| **측정4 선택정합** | | | | |
| subset 활성 unseen | 0.230 | 0.499 | 0.592 | — |
| subset 활성 train | 0.446 | 0.472 | 0.524 | — |
| \|subset 활성 unseen−train\| | 0.216 | **0.027** | 0.068 | ✓ 유독 작음(표적 subset 주변활성이 split1 unseen서 거의 안 변함) |
| subset∩unseen_top10 | 1 | 0 | 1 | (split1=0, 약함) |

## 핵심 관찰 — split1이 이상치인 지표는 모두 "저차 신호 결핍"을 가리킴
split1에서만 방향 정렬(이상치)한 지표: **M2 AUC 최저 · feat 활성 L1이동 최소 · |subset 활성 unseen−train| 최소 · 측정2 1차 이탈비중 최고/2차 최저 · conn3 jaccard 0.** 종합하면:
- **split1의 unseen 가족은 탐지기 feature(표적 subset) 위에서 주변·저차 구조가 train과 가장 유사** (활성 unseen 0.499≈train 0.472, L1이동 2.31 최소) → **M1/M2(저차) 탐지기가 유독 약함**(AUC 0.53/0.61 최저).
- 동시에 **3차 top 상관 구조는 known과 전혀 안 겹침**(jaccard 0.000) → 저차로 못 잡는 판별 잔여 신호가 고차에 존재 → **M4/M_full(0.76)이 M2(0.61)를 앞섬 = 고차 이득 +0.15**.

## 판정 (H1~H4)
- **H3(아티팩트) 기각**: n_unseen·known/unseen 비가 split1을 단독 지목 안 함(split1 중간). 표본수·불균형은 원인 아님.
- **H1(구조 상속) 기각(주효과 아님)**: ladder_rge3 전 split 음수(unseen은 train ≥3차를 상속 안 함)·split1 비특이. 단 conn3 jaccard=0은 "3차에서 **최대 비**상속"으로 부분 관련.
- **H2(고차 이탈 큼) — 직접형은 반증, 상대형은 성립**: 측정2 ≥3차 **이탈 비중은 split1이 최저(0.31)** → "unseen이 고차에서 더 크게 이탈해서"는 **아님**. 대신 **1·2차 이탈이 유독 작아**(주변분포가 train과 유사) 저차 판별이 무너지고, 남은 판별 신호가 상대적으로 고차에 몰림 → **고차 모델이 유일하게 잡을 수 있는 잔여**가 됨(3차 구조 jaccard 0).
- **H4(선택 정합) 부분 채택**: |subset 활성 unseen−train|=0.027(최소), subset∩unseen_top10=0 → train에서 고른 표적 subset의 주변활성이 split1 unseen에선 거의 안 변함 → 저차 신호 결핍에 직접 기여.

**결론(혼재, 지배 스레드 명확)**: split1의 고차 AUC 이득은 **"고차 이탈이 커서"가 아니라 "저차(주변·pairwise) 이탈이 유독 작아서"** 발생 — 표적 subset 위 unseen의 주변분포가 train과 가장 유사(M2 AUC 최저)해 저차 탐지가 무너지고, known과 전혀 안 겹치는 3차 구조(jaccard 0)가 고차 모델이 잡는 잔여 판별 신호가 됨. **H3 기각, H1 기각(부분: 3차 비상속), H2는 상대형(저차 결핍→고차가 유일 신호)만 성립·직접형 반증, H4 부분 채택.** 즉 지배 원인 = **저차 신호 결핍 + 고차 전용 잔여 구조**.

## 부분상관 시도 / 한계
- **표본 3 split**뿐 → 상관계수·부분상관은 통계적으로 무의미(순위·이상치 정렬로만 판단). 지배 스레드는 4개 독립 지표(M2 AUC, L1이동, |Δ활성|, jaccard)가 일관 정렬해 견고하나, 3점 한계 명시.
- 측정2의 차수 분해는 **robust moment(연결 상관 cosine-dissimilarity)** 기반. 초기 maxent-사다리 KL 버전은 **order-4 maxent가 train 과적합→unseen서 KL 폭발(14.6)·비가법(dev2>dev_full)** 이라 폐기(그 자체가 "train ≥3차는 unseen에 일반화 안 됨"의 방증). 그래서 §측정1 ladder는 smooth(M1/M2/full 평활)만 사용, order-4 제외.
- k=10 표적 subset 한정(k=12도 AUC 부호 동일: split1 +0.116). AUC/탐지는 NLL 기반=본질적으로 약함(직전 로그) → **상대 격차·차수 정렬만** 해석, 절대 탐지력 주장 아님. 우위·QCBM 언급 없음.
- 인과 아님(관측 정렬). "저차 결핍"과 "M2 AUC 낮음"은 부분적으로 동어반복 위험 있으나, feat 활성 L1이동·|Δ활성|·jaccard는 AUC와 독립 계산된 구조 지표로 교차 확인됨.

## 생성 파일 / 실행 / git
- 신규: `tools/diagnose_split1.py`, `results2/split1_diag/{split1_diag.json,split1_diag.png}`, 본 로그.
- 실행: `python3 -m py_compile tools/diagnose_split1.py` OK; `python3 tools/diagnose_split1.py --splits 0,1,2 --k 10` (수 초, 단일 프로세스, 새 학습 없음).
- `git diff --stat`=기존 무변경. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스·results2 기존 json·Drebin 원본 무수정.

## 다음 추천
1. "저차 결핍" 가설 강화: 전 split에서 unseen의 저차(활성·pairwise) train-유사도와 M2 AUC의 관계를 더 많은 가족/부분집합으로(3점 넘게) 재측정.
2. split1 unseen(f1)의 jaccard=0 3차 triple이 실제 어떤 permission/api 조합인지 식별 — 고차 전용 판별 구조의 정체.
3. (해석 입력) 웹 AI에: "split1 고차 이득은 고차 이탈이 커서가 아니라(≥3차 이탈비중 최저 0.31) 저차 이탈이 유독 작아(주변 train-유사, M2 AUC 최저 0.61) 저차 탐지가 무너지고, known과 안 겹치는 3차 구조(jaccard 0)가 고차 잔여 신호가 된 것. H3 기각·H1 기각·H2 상대형만·H4 부분. 3점 한계."
