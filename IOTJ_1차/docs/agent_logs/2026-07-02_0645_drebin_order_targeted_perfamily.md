# Drebin 고차성 최강 버전 — 표적 부분집합 탐색(≥3차 최대화, holdout) + per-family 분해 (측정 전용)

- 작성: 2026-07-02 06:45 UTC (KST 15:45)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- **순수 차수 측정.** QCBM 학습·탐지 없음, 기존 코드 무수정(import), 무설치. 단정 금지·과적합 명시(측정값만, 우위·QCBM 언급 없음).

## 1. 작업 목적
직전: Drebin 일반 선정 ≥3차 최대 21.8%(highvar k12) < flow UNSW 27%, 2차 지배. 추론1(malware>flow 고차) 일반 선정선 미지지.
이번 최강 버전 2가지: (b) ≥3차를 *일부러 최대화*하는 부분집합이 있는가(표적 greedy, **holdout 필수**), (a) 가족 pooling이 mixture로 겉보기 고차를 부풀렸는지 per-family 분리. 결과로 malware 축 완전 종료 여부 판단.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip`(전부 untracked). 보존: 기존 소스·results2 기존 json·Drebin 원본(읽기만).

## 3. 확인한 도구 재사용 / per-family 필터 / 가족 표본 / 표적 비용 / holdout 방법
- **도구 재사용(무수정)**: OD(`maxent_upto`,`kl_div`,`bit_matrix`,`decompose`,`self_check`,`entropy`), DR(`emp_dist`,`select_subset`,`load_split`,`top_triples`). ≥3차 share=KL2/KL1(1·2차 maxent만; 임의 feature 인덱스 지원). 12bit 3·4차 세부는 비수렴이라 headline만(직전과 동일 잣대).
- **holdout 주의(중요)**: `drebin_new_0.npz` train 가족 {1..7}, **test 가족 {0,2,3,6,7}**(가족0=925 train에 없음) — CADE concept-drift라 **X_test는 iid holdout 아님**. → 과적합 격리 1차=**train 내부 random split selA/selB**(같은 가족분포), X_test(drift)는 2차 검증(명시).
- **per-family 표본**(train split0): f1=666,f2=312,f4=329,f3=201,f5=152,f6=145,f7=107. k=10/12엔 희소(occ≤n) → n≥150만 측정(FAMILY_MIN=150), 부족 가족 불가 명시.
- **표적 비용**: k12 ≥3차 1회 평가 ~0.33s(k8 0.02s). 후보풀 top-M(=40) greedy k12 ≈ (2..12)×40 evals ≈ 1분/run → splits×ks feasible.
- **greedy 방법**: selA 고분산 top-M 후보풀, 씨앗=connected-3body 최대 triple, ≥3차 share 최대화하며 k까지 추가(국소최적=하한).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: 도구 재사용·train/test 분리 로드·가족 표본수·greedy 비용·holdout 설계.
- 추정: 표적으로 selA ≥3차는 21.8%보다↑ 가능(예비 top-var40 k12=25.7%). selB에서 유지될지(과적합 여부)가 관건.
- 확인 불가(본실행 전): selB holdout 값·과적합 gap·per-family가 pooling보다 낮은지.

## 5. (b) 표적 greedy 탐색 — ≥3차 최대 부분집합 (selA 선정 → selB iid holdout / X_test drift)
방법: selA(train 50%)서 고분산 top-40 후보풀, 씨앗=connected-3body 최대 triple, ≥3차 share 최대화하며 k까지 greedy 추가(국소최적=하한).

| split | k | selA(선정) | **selB(iid holdout)** | gap(A−B) | X_test(drift) | occA/occB |
|---|---|---|---|---|---|---|
| 0 | 10 | 35.2% | **33.9%** | +1.3% | 14.1% | 104/104 |
| 0 | 12 | 41.4% | **40.8%** | +0.6% | 18.1% | 180/172 |
| 1 | 10 | 60.9% | **64.1%** | −3.2% | 32.5% | 133/132 |
| 1 | 12 | 58.4% | **60.0%** | −1.7% | 27.6% | 188/182 |
| 2 | 10 | 58.1% | **61.2%** | −3.1% | 18.7% | 130/135 |
| 2 | 12 | 54.9% | **57.2%** | −2.3% | 20.8% | 190/202 |

- **selB(iid holdout) ≥3차 = 34~64% — flow UNSW(27%)·일반선정(21.8%)를 뚜렷이 초과, gap≈0(±3%)=과적합 아님**(iid). 즉 ≥3차를 표적으로 크게 올린 부분집합이 **실재하고 iid holdout서 견고**.
- **그러나 X_test(drift) = 14~32%로 ~절반 붕괴** — test는 train과 다른 가족(CADE drift)이라, 이 고차성은 **train 가족 mixture에 특이적**(unseen 가족엔 전이 안 됨).
- 표적 부분집합 구성(k12): permission·call·api_call·real_permission 계열 지배(split0 perm4/call2/api2, split1 call4/perm4, split2 call4/perm3/api3) — 악성 행동 조합 방향과 부합.

## 6. (a) per-family 분해 — 고정 highvar 부분집합, 가족별 ≥3차 vs pooling
표본 n≥150 가족만(occ≪2^k=희소, 표본오차 큼 명시). n<150 가족(대부분 f6/f7 등)=측정불가.

| k | pooling(전가족) | per-family 평균 [범위] (n≥150 5가족) |
|---|---|---|
| 10 | 15.2% | **8.4~9.5%** [6~14%] |
| 12 | 21.8% | **10.1~14.1%** [3~25%] |
- **per-family 평균 < pooling** (k10 ~9% vs 15.2%, k12 ~12% vs 21.8%) → **pooling(mixture)이 ≥3차를 부풀림**(가족 간 서로 다른 feature 패턴의 혼합이 고차 상관 유도). 일부 가족(k12 f2=25%, f1=18%)은 pooling 근처이나 평균은 낮음.
- 즉 pooling 21.8%의 상당 부분이 **가족 mixture 기여**, 단일 가족 내부 고차성은 그보다 낮음(occ 희소로 값 신뢰도 제한).

## 7. flow 대비표
| 데이터 | ≥3차 |
|---|---|
| flow UNSW / Bot / HOIC | 27% / 19% / 1% |
| Drebin 일반선정(직전) | ≤21.8% |
| **Drebin 표적 selB(iid holdout)** | **34~64%** (flow 초과) |
| Drebin 표적 X_test(drift) | 14~32% (flow 이하~근처) |
| Drebin per-family(단일가족) | 3~25% (평균 8~14%) |

## 8. self-check (잣대 일관성)
- pooling highvar k12 ≥3차 = **0.218 재현**(기대 0.218) ✓. 합성 parity/ising 단조 all=True ✓. flow UNSW = **0.270 재현**(기대 0.270) ✓.

## 9. 사실 관찰 (판정 아님, 단정 금지)
- **추론1 전제(malware>flow 고차)의 최강 버전 = 조건부 지지 / 애매 (기각 아님, 완전 지지도 아님)**:
  - **(지지 측면)** 표적 탐색으로 ≥3차 34~64% 부분집합이 실재하고 **iid holdout(selB)서 견고**(gap≈0, 과적합 아님) → flow(27%) 뚜렷이 초과. "일반 선정에선 21.8%"라는 직전 결론은 **표적 선정에선 깨짐**.
  - **(제약 측면)** 그 고차성은 **가족 mixture에 기인** — per-family(단일 가족)는 평균 8~14%로 낮고, 표적 부분집합도 **X_test(다른 가족 drift)서 ~절반 붕괴**(14~32%). 즉 "malware 자체의 전이 가능한 조합적 고차"라기보다 **train 가족 구성에 특이적**.
  - 종합: **iid 분포적으로는 flow 초과 고차가 실재(견고), 그러나 그 원천이 가족 mixture라 novel family엔 전이 안 됨** → malware 축은 "완전 종료"도 "명확한 승산"도 아닌 **좁은·조건부 무대**(고정 가족셋·표적 feature 한정).
- **holdout 신뢰성**: selB는 iid(train과 같은 가족분포)라 과적합 격리에 유효 → 34~64%는 **과적합 아님**(신뢰 가능). X_test는 drift라 전이 실패는 과적합이 아니라 분포 이동. per-family는 occ 희소(12~89/4096)로 개별값 신뢰도 낮음(평균 경향만).
- ★ 측정값 기준. 우위·QCBM 성능 언급 없음. greedy=국소최적(하한). 이 부분집합 규모·이 잣대 한정. 단정 금지.

## 10. 생성 파일 / 실행 명령 / 시간
- 신규: `tools/diagnose_drebin_targeted.py`, `results2/drebin_targeted/{drebin_targeted.json,.png,run.log}`, 본 로그.
```
python3 -m py_compile tools/diagnose_drebin_targeted.py                 # OK
python3 tools/diagnose_drebin_targeted.py --smoke                       # OK (selA50%→selB, 1가족)
python3 tools/diagnose_drebin_targeted.py --splits 0,1,2 --ks 10,12     # exit0, ~1분
```
- 시간: 본실행 ~1분(greedy k10 ~2~3s, k12 ~4~13s ×3 split; per-family 즉시; self-check UNSW 재빌드 포함). **단일 프로세스**. png에 한글 글리프 경고(DejaVu 한글 없음)=박스로 렌더, 무해(값 영향 없음).

## 11. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스·results2 기존 json·Drebin 원본 무수정.

## 12. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 도구 재사용·per-family 필터·가족 표본·표적 비용·holdout(selA/selB + drift) 확인·기록.
- [x] (b) 표적 train/holdout/test ≥3차(k=10,12, 3split) — selB 34~64% flow 초과, 과적합 아님, drift 붕괴.
- [x] (a) per-family ≥3차 vs pooling — mixture 부풀림 확인(per-family 평균 < pooling).
- [x] self-check: pooling 21.8% + flow UNSW 27% 재현(둘 다 정확).
- [x] 최강 버전 판정 재료: **조건부 지지/애매**(iid 초과·견고 but mixture·drift 제약). 과적합 명시.
- [x] 학습 없음·기존 무변경·무설치.

**남은 불확실성**
- X_test=drift라 "전이 실패"가 개념 이동 때문 — 같은 가족셋의 *다른 표본*(더 큰 iid test)로 재현 확인은 미수행(train만 3315 규모라 selA/selB로 대체).
- per-family occ 희소(≤89/4096) → 개별 가족값 신뢰도 낮음(평균 경향만 해석).
- greedy=국소최적 → 진짜 최댓값은 더 높을 수 있음(하한). 표적이 "가족 판별 feature"를 골랐을 가능성(고차=가족 분리 신호)은 미분해.

**다음 추천 단계**
1. 표적 부분집합의 고차 상관이 **가족 판별(분류) 신호**인지 확인 — 그 feature들이 가족 예측력이 높은지(측정, 학습 아님). 높으면 "고차=가족 mixture 라벨 구조"로 해석 강화.
2. (malware 축 판단) iid·고정 가족셋 시나리오로 무대를 한정해 QCBM 소규모 시험을 할지 결정 — 단 drift 전이 실패·mixture 기여를 전제로.
3. (해석 입력) 웹 AI에: "표적 선정 시 ≥3차 34~64%(iid holdout 견고, flow 27% 초과)이나 가족 mixture 기인(per-family 8~14%)·drift서 붕괴(14~32%). 최강버전은 조건부 지지=좁은 무대. 이 잣대·부분집합 한정, 우위 언급 없음."
