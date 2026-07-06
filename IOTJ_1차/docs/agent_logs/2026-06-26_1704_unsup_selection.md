# 비지도 모델 선택 기준 탐색 — 라벨 없는 기준이 고AUC restart를 고르는가

- 작성: 2026-06-26 17:04 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- QCBM 재학습 없음 (저장된 16 p_final 재사용). raw MWU 유지. 판정은 웹 AI.

## 1. 작업 목적
직전 결과에서 우리가 best를 고르는 기준 min-MMD²가 탐지 AUC와 무관함이 드러났다
(corr(MMD²,AUC)=0.077≈0). 실전(비지도)에는 공격 라벨이 없어 AUC로 못 고르므로, "216p best가
고전을 이겨도 실전에서 그 best를 못 고른다"는 치명적 약점. 본 작업은 **공격/라벨 일절 없이
benign(train/test)+학습분포 p_final만으로** 계산되는 다른 선택 기준들이 MMD²보다 고AUC restart를
더 잘 고르는지, 앙상블이 고전(0.706)을 넘는지를 사실로 탐색한다.

## 2. 시작 git status, 보존 대상
시작 `git status --short`: `?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip` (전체 untracked).
`git diff --stat`(tracked) = 비어 있음. 보존 대상: 모든 기존 소스/데이터/결과(import/읽기만).
생성: `tools/explore_unsup_selection.py`, `results2/unsup_selection.json`,
`results2/unsup_selection.png`, 본 로그.

## 3. 재현/정합: 16 AUC, q_train/states
- npz 키: `seeds, p_finals(16,4096), aucs, mmd2, tv, weights, q_train(4096,)`.
  **benign_test/attack states는 npz에 없어** `train_qcbm_bot_resumable.build_bot()`로 재현
  (q_train, st_tr, st_te, st_atk, occ 반환).
- 점유 occ=**361/4096** → qcbm_bot.json과 일치. q_train 재현 `np.allclose` 일치.
- 16 AUC 재계산 vs npz `aucs`: **max|Δ|=0.00e+00**, median 0.6650 일치 → 정합 OK.
- TV 재계산 vs 저장 `tv`: max|Δ|=0.00e+00 (재사용 일관성).

## 4. 비지도 조건 준수 (공격 미사용) 확인
- 선택 기준 S1~S6, 앙상블 E1~E3은 **q_train(benign train)·st_te(benign test)·p_final만** 사용.
- `st_atk`(공격)는 **AUC 평가에만** 사용 — "비지도 기준 X로 restart 고름 → 그 AUC 사후 확인" 구조.
  기준 산식 어디에도 공격/라벨 미포함. (코드 `auc_of_dist`만 st_atk 사용, 나머지 점수 함수는 미사용.)
- `experiment2`의 auc_score/tv_dist/kl_div/empirical_dist, `build_bot` import 재사용. scipy.stats만 추가.

## 5. 확실한 사실 / 추정 / 확인 불가
**확실(코드·재계산)** — 비교 기준선: 고전 216p-이하 최고 = 0.706, QCBM median=0.665, best(seed7)=0.746.
각 비지도 기준의 Pearson/Spearman(vs AUC)과 argmin/argmax 선택 결과는 §6 표.
**추정(확인됨)**: "분포적합 계열은 AUC와 무상관"이라는 사전 추정은 **부분적으로 틀림** — MMD²/TV는
무상관이지만 KL(q‖p)·held-out NLL·일반화 갭은 |Spearman| 0.6~0.74로 **강하게 상관**. 단 **부호가
"반대"**(적합이 좋을수록 AUC가 낮음)라 적합-최소화 선택엔 무용.
**확인 불가/한계**: n=16 소표본. 엔트로피 기준이 best를 고른 것이 신호인지 운인지 단정 불가.

## 6. 결과
### 6-1. 비지도 선택 기준 vs AUC (공격 미사용)
MMD² 기준선 Pearson = **0.077** (직전 결과와 정합).

| 기준 | Pearson | Spearman | (자연방향)min-pick AUC | >고전 | max-pick AUC |
|---|---|---|---|---|---|
| S1 MMD²(기준선) | +0.077 | +0.021 | seed7 **0.746** | ✅ | seed13 0.721 |
| S2 TV | −0.014 | +0.038 | seed7 **0.746** | ✅ | seed13 0.721 |
| S3a KL(q‖p) | **+0.709** | **+0.715** | seed10 0.656 | ❌ | seed5 0.736 |
| S3b KL(p‖q) | +0.254 | +0.197 | seed4 0.524 | ❌ | seed1 0.635 |
| **S4 held-out NLL(test)** | **+0.711** | **+0.744** | seed10 0.656 | ❌ | seed5 0.736 |
| S5a 엔트로피 H(p) | −0.355 | −0.371 | seed7 **0.746** | ✅ | seed2 0.620 |
| S5b 유효 support exp(H) | −0.368 | −0.371 | seed7 **0.746** | ✅ | seed2 0.620 |
| S6 일반화 갭(NLL_te−tr) | +0.550 | +0.621 | seed2 0.620 | ❌ | seed7 **0.746** |

해석(중요):
- **강한 예측자가 존재한다** (held-out NLL: Spearman +0.744, KL(q‖p) +0.715, gen_gap +0.621).
  그러나 **부호가 양(+)** = benign 적합이 *나쁠수록* AUC가 *높다*. 즉 자연스러운 비지도 규칙
  "적합 최소화(min)"로 고르면 **고전 아래(0.656)** restart가 뽑힌다 (TV-AUC 해리의 일반화 —
  적합도는 무정보가 아니라 **음(−)의 정보**).
- min-MMD²/min-TV가 seed7(0.746)을 고른 것은 Spearman≈0.02~0.04로 **신호 아닌 운**(직전 발견 재확인).
- 유일하게 "자연방향 + >고전"인 구조적 기준은 **min-엔트로피**(가장 집중된 분포): seed7(0.746=best)을
  고름. 다만 |Spearman|=0.37로 중간, 소표본이라 신호/운 단정 불가.
- 강한 상관을 **역이용**(max-NLL = benign을 가장 못 맞춘 모델 선택)하면 seed5(0.736>고전). 라벨은
  안 쓰지만 "benign 적합 최악 모델 선택"이라는 도착(倒錯)적 원리라 신뢰·정당화 어려움(관찰로만 기록).

### 6-2. 앙상블 (공격 미사용 결합 → AUC 평가)
| 방법 | AUC | vs median 0.665 / 고전 0.706 / best 0.746 |
|---|---|---|
| E1 산술평균 | 0.699 | median 위 / 고전 **아래** / best 아래 |
| E2 기하평균 | 0.696 | median 위 / 고전 **아래** / best 아래 |
| E3 top4 by min-MMD² | 0.687 | median 위 / 고전 아래 / best 아래 |
| E3 top4 by min-NLL | 0.628 | median **아래** / 고전 아래 / best 아래 |

- 앙상블은 median(0.665)보다 안정적으로 높지만 **어느 것도 고전(0.706)을 넘지 못함**(최고 0.699).
  min-NLL 상위4 평균이 가장 낮은 것은 "적합 좋은 4개 = AUC 낮은 4개"라는 §6-1 음상관의 귀결.

## 7. 사실 관찰 (판정 아님)
- **AUC를 잘 예측하는 비지도 기준이 있는가**: 있다(NLL/KL/gen_gap, |Spearman| 0.6~0.74). 단 부호가
  적합-최소화와 반대라, 자연스러운 비지도 선택으로는 고AUC를 못 고른다.
- **자연방향으로 고전을 넘기는 기준**: min-MMD²·min-TV는 운으로 seed7(0.746)을 고름(상관≈0);
  min-엔트로피는 약한 신호로 seed7을 고름(>고전). 적합도 계열(KL/NLL/gen_gap)은 모두 고전 아래.
- **앙상블이 고전을 넘는가**: 아니오(최고 0.699 < 0.706).
- **모델선택 문제 해소 여부**: **미해소(부분적 단서만)**. 적합도 기반 비지도 선택은 오히려 역효과
  (적합 좋을수록 AUC 낮음). 가능성 있는 단서는 (i) min-엔트로피(구조적, 약), (ii) 적합 기준의 역이용
  (도착적). 둘 다 n=16 소표본 검증 필요. 따라서 "216p best를 실전에서 못 고른다"는 약점은
  이번 탐색으로 **확실히 해소되지 않음**.

## 8. 생성 파일, 실행 명령과 결과
- `qcbm/tools/explore_unsup_selection.py` (신규)
- `qcbm/results2/unsup_selection.json`, `qcbm/results2/unsup_selection.png` (신규)
- `docs/agent_logs/2026-06-26_1704_unsup_selection.md` (본 로그)

```
python3 -m py_compile tools/explore_unsup_selection.py   # OK
python3 tools/explore_unsup_selection.py                 # real 0m7.2s, 정합 OK, DONE
# [정합] AUC Δ=0, TV Δ=0 ; best_predictor=S4_NLL_test(Spearman +0.744) ; 앙상블 최고 0.699<고전
```
실패/에러 없음. KL/NLL의 0 처리: eps=1e-12 클립(`kl_div` 기본 eps, NLL은 `-log(p+EPS)`).
환경: /usr/bin/python3, OMP_NUM_THREADS=1, scipy.stats.spearmanr 사용(설치된 것 사용, 무설치).

## 9. git diff --stat (기존 무변경)
```
$ git diff --stat     # tracked 변경: 없음
$ git status --short
?? docs/
?? qcbm/
?? qcbm_share_20260626.zip
```
기존 소스/데이터/결과(tracked) 무수정. QCBM 무재학습, dependency 무설치.

## 10. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 16 AUC 재계산 정합(Δ=0), q_train/states 재현(occ 361).
- [x] S1~S6 각 Pearson/Spearman 계산, MMD²(0.077) 대비 더 나은(절대상관) 기준 보고
      (NLL/KL/gen_gap이 |0.6~0.74| — 단 부호 반대).
- [x] 앙상블 E1~E3 AUC vs median/고전/best 비교(최고 0.699<고전).
- [x] 각 기준 선택 시 뽑히는 restart AUC와 고전 초과 여부 보고(양방향).
- [x] 모든 선택 기준 공격 미사용 확인(비지도 조건).
- [x] json/png 저장, 기존 무수정·무재학습·무설치.

**남은 불확실성**
- n=16 소표본: 엔트로피 기준(−0.37)과 "적합 역상관"이 다른 split/데이터에서도 유지되는지 미검증.
- 적합 역이용(max-NLL) 규칙의 정당성·안정성 불명(원리적으로 도착적).

**다음 추천 단계**
1. 일반화 검증: 다른 benign split seed 여러 개로 "적합 좋을수록 AUC 낮음" 음상관과 min-엔트로피
   효과가 재현되는지(부호 안정성).
2. 더 많은 restart(예: 32~64)로 엔트로피/NLL의 AUC 예측력 신뢰구간 추정 — 신호 vs 운 판별.
3. (재학습 관련) L=10 등 더 깊은 회로 재학습보다, 먼저 "적합-탐지 음상관"의 원인(인코딩/공격 분포
   구조) 진단이 우선 — 음상관이 구조적이면 비지도 선택 자체가 원리적으로 어려움을 시사.
