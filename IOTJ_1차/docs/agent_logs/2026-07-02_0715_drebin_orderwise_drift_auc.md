# 차수별 drift 탐지 AUC 진단 — 고차(≥3차) 모델링이 pairwise보다 unseen-가족 탐지 AUC를 올리는가 (고전 전용)

- 작성: 2026-07-02 07:15 UTC (KST 16:15)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- **고전 전용, QCBM 없음. maxent 적합·MI 계산 외 학습 없음.** 기존 코드 무수정(import), 무설치. 상대 비교 한정(절대 성능·CADE 비교 금지). 단정 금지·판정은 웹 AI.

## 1. 작업 목적
직전: Drebin 표적 부분집합서 ≥3차 34~64%(iid holdout 견고)나 가족 mixture 기인·drift 붕괴. "분포에 고차 있다 ≠ 탐지에 고차 쓸모 있다"(flow 적합-탐지 해리 −0.7). 이번: 같은 표적 부분집합·train/test에서 **차수만 바꾼 밀도모델(1/2/3/4차·평활 경험분포)** 의 unseen-가족 drift 탐지 AUC를 비교 → 고차가 pairwise(2차)보다 AUC를 올리면 form-2 무대, 안 올리면 malware 축 폐쇄 재료.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip`(전부 untracked). 보존: 기존 소스·results2 기존 json·Drebin 원본(읽기만).

## 3. 확인한 표적 부분집합 형태 / unseen 가족 / 수렴 / 평활화 / 상태 매핑
- **부분집합 인덱스 저장됨**(재현 불필요): `results2/drebin_targeted/drebin_targeted.json` → `targeted[k][i]['subset']`(표적), `per_family[k][i]['subset']`(highvar), split=rec['split']. 12/10개 원본 feature 인덱스.
- **unseen 가족**(y_train에 없고 y_test에 있음): split0→f0(unseen 925/known 478), split1→f1(667/529), split2→f2(625/538).
- **maxent 수렴**(OD.maxent_upto): k=10 order1/2 빠름, **order3=0.6s·order4=6.8s 모두 conv=True·full support**(log 안전). k=12는 3·4차 비수렴 → **제외**(M_full 프록시).
- **평활화**(경험분포 M_full만; maxent는 full support라 불요): Laplace 총 pseudo-mass=1, `p_s=(n_s+c)/(N+c·2^k)`, `c=1/2^k` 고정.
- **상태 매핑**: test 표본 k비트 → 정수코드(bit j<<j, DR.emp_dist 동일) → p_model[code]. score=−log p, unseen=양성. AUC=rank(Mann-Whitney, numpy 직접).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: 부분집합 저장·unseen 가족·k10 3/4차 수렴·평활·상태매핑·재사용 함수.
- 추정: unseen 이탈이 저차에 이미 크면 M2≈고차(고차 무가치); 표적 고차가 가족구성 정보면 고차가 unseen 더 낮은 likelihood로 매길 수도(미검).
- 확인 불가(본실행 전): 실제 격차(고차−M2) 부호·크기·split 일관성, MI 대소.

## 5. (축1) 차수별 drift AUC — 표적 부분집합 (unseen 가족=양성, score=−log p[state])
| split | k | M1(1차) | M2(2차) | M3 | M4 | M_full | 고차−M2 | unseen/known |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 0.866 | **0.889** | 0.817 | 0.820 | 0.815 | −0.069 | 925/478 |
| 1 | 10 | 0.528 | 0.605 | **0.765** | 0.757 | 0.759 | +0.152 | 667/529 |
| 2 | 10 | 0.226 | **0.880** | 0.810 | 0.786 | 0.792 | −0.094 | 625/538 |
| 0 | 12 | 0.818 | 0.767 | — | — | 0.770 | +0.002 | 925/478 |
| 1 | 12 | 0.482 | 0.650 | — | — | **0.766** | +0.116 | 667/529 |
| 2 | 12 | 0.213 | 0.768 | — | — | **0.811** | +0.043 | 625/538 |
- **고차−M2 격차: k10 [−0.069,+0.152,−0.094] 평균 −0.004 / k12 [+0.002,+0.116,+0.043] 평균 +0.054.** → **작고 부호 split마다 뒤바뀜**(견고한 개선 아님). 최고 AUC 모델도 split마다 다름(M2 2회, M3 1회, M_full 2회, M1 1회).
- M1(1차)은 극도로 불안정(0.21~0.87, split2/k10=0.226=역전) → 저차 단독은 신뢰 불가. M2(pairwise)는 대체로 강함(0.61~0.89).
- k12 3·4차 maxent 비수렴 → M3/M4 제외(M_full 프록시), 표에 "—".

## 6. (축2) highvar 일반 부분집합 대조
| split | k | M2 | M_full | 고차−M2 |
|---|---|---|---|---|
| 0/1/2 | 10 | 0.696/0.821/0.728 | 0.723/0.832/0.835 | +0.026/+0.011/+0.111 (평균 +0.050) |
| 0/1/2 | 12 | 0.816/0.837/0.825 | 0.784/0.861/0.866 | −0.033/+0.023/+0.041 (평균 +0.011) |
- highvar도 고차−M2 평균 +0.01~+0.05로 **작음**(부호 혼재). 표적/highvar 모두 "고차가 pairwise를 뚜렷·일관 초과"는 아님.

## 7. (축3) 가족 판별력 MI (feature-family, train known 가족; nats)
| split | 표적 평균 | highvar 평균 | 전체(1376) 평균 | 표적/전체 배수 |
|---|---|---|---|---|
| 0 | 0.277 | 0.220 | 0.055 | 5.0× |
| 1 | 0.288 | 0.333 | 0.064 | 4.5× |
| 2 | 0.277 | 0.293 | 0.059 | 4.7× |
- 표적·highvar 선정 feature 모두 **전체 평균의 ~4.5~5배 가족 MI** → 선정된 고분산/표적 feature는 강한 **가족 판별 신호**(직전 "고차=가족 mixture 구조" 해석과 부합). 표적 ≈ highvar (표적이 특별히 더 높지는 않음).

## 8. self-check
- AUC: 무작위=**0.502**(≈0.5), 완전분리=**1.000**, 역분리=**0.000** → AUC 계산 정확 ✓.
- maxent 수렴: k10 order1~4 모두 conv=True(full support); k12 order3/4 비수렴 → 제외(플래그 기록) ✓.
- train ≥3차 share(full-train, KL2/KL1): 표적 0.34~0.62 — selA 기준 직전(34~64%)과 같은 대역(정합 경향; full-train이라 정확 일치는 아님) ✓.

## 9. 사실 관찰 (판정 아님, 단정 금지)
- **고차(≥3차) 모델링이 pairwise(M2)보다 drift 탐지 AUC를 뚜렷·일관되게 올리지 못함**:
  - 고차−M2 격차가 **작고**(대개 |Δ|<0.1) **split마다 부호가 바뀜**(표적 k10 −0.069/+0.152/−0.094). 평균도 −0.004~+0.054로 미미.
  - 최고 AUC는 split별로 M2/M3/M_full/M1 제각각 — "차수↑가 이득"이라는 견고한 패턴 없음. M_full(전차수)도 M2 대비 거의 동급.
  - 즉 **탐지 신호는 대체로 저차(특히 2차)에서 이미 포착**되고, 고차 추가는 이득이 불안정/미미(때로 악화). "분포에 고차 있다 ≠ 탐지에 고차 쓸모 있다"와 부합.
- 부가: 선정 feature는 가족 MI가 전체의 ~5배(표적·highvar 공통) → 고차 구조는 **가족 판별 신호**에 가까움.
- ★ 상대 비교(차수별 격차) 한정. **절대 성능·CADE 비교 아님**(NLL 탐지는 본질적으로 약함). 우위·QCBM 언급 없음. unseen이 test 다수(불균형)라 표본수 병기함. 판정은 웹 AI.

## 10. 생성 파일 / 실행 명령 / 시간
- 신규: `tools/diagnose_drift_auc.py`, `results2/drift_auc/{drift_auc.json,.png,run.log}`, 본 로그.
```
python3 -m py_compile tools/diagnose_drift_auc.py                    # OK
python3 tools/diagnose_drift_auc.py --smoke                          # OK (AUC self-check 통과)
python3 tools/diagnose_drift_auc.py --splits 0,1,2 --ks 10,12        # exit0, ~1분
```
- 시간 ~1분(k10 M4 ~7s×3split×2subset + 나머지 소액; MI 즉시). **단일 프로세스**. png 한글 글리프 경고=박스 렌더(무해).

## 11. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스·results2 기존 json·Drebin 원본 무수정.

## 12. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 표적 부분집합 재사용·unseen 가족·수렴·평활·상태매핑 확인·기록.
- [x] (축1) split×k×{M1,M2,M3,M4,M_full} AUC 표 + 고차−M2 격차 + unseen/known 표본수.
- [x] (축2) 표적 vs highvar 대조.
- [x] (축3) 가족 MI(표적/highvar/전체, ~5×).
- [x] self-check: AUC 0.5/1.0/0.0, 수렴 플래그, share 정합 경향.
- [x] 판정 재료: **고차−pairwise 격차 작고 부호 불일치 = 견고한 탐지 이득 없음**. 단정 없이 값으로.
- [x] maxent/MI 외 학습 없음·기존 무변경·무설치·CADE 비교 없음.

**남은 불확실성**
- unseen 가족이 test 다수(불균형)·NLL 탐지 본질적 약함 → 절대 AUC 해석 주의(상대 격차만 유효).
- k12 고차는 M_full 프록시(3/4차 비수렴). M_full=memorization·평활 c 민감 → maxent 사다리(k10)가 1차 근거.
- 표적 부분집합이 selA에서 선정(가족구성 특이) → 다른 선정/더 큰 k에서 격차 다를 여지(단 방향성은 "미미·불일치"로 일관).

**다음 추천 단계**
1. (malware 축 판단) 고차 모델링의 탐지 이득이 이 잣대에서 견고치 않음 → form-2(QCBM으로 고차를 적은 파라미터로) 정당화 약함. 축 폐쇄/보류 결정 재료로 웹 AI에 제출.
2. (원하면) 탐지 지표를 NLL 대신 다른 방식(2차 vs 고차 잔차 기반)으로 바꿔 재확인 — 단 여전히 상대 비교.
3. (해석 입력) 웹 AI에: "차수별 drift AUC에서 고차−pairwise 격차 −0.09~+0.15(작고 부호 불일치, 평균≈0), 최고모델 split마다 상이, 선정 feature는 가족 MI ~5×. 고차 모델링의 탐지 이득 견고치 않음. 상대 비교·이 잣대 한정, 우위·QCBM·CADE 비교 없음."
