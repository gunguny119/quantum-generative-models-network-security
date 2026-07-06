# 고차 상관 진단 — k-body Pauli-Z correlator 스펙트럼 (HOIC/Bot/UNSW)

- 작성: 2026-06-28 12:03 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 순수 측정(학습 없음). 기존 인코딩/평가 코드 무수정, import 재사용만. 판정은 웹 AI.

## 1. 작업 목적
benign 경험적 분포의 "고차 상관"을 직접 측정한다. QCBM=Fourier(Pauli-Z correlator) 모델이므로,
분포의 frequency support가 저차에 몰리면 고전 저랭크가 싸게 대체 가능(양자 무망), 고차로 퍼지면
고전이 비싸짐(양자 여지). SVD 랭크(2차 상관만)가 분포 복잡도를 과대평가했는지(특히 UNSW: rank 12인데
고전이 1/3 params로 따라잡음)를 k-body correlator 스펙트럼으로 재진단한다.

## 2. 시작 git status, 보존 대상
`git status --short` (qcbm 기준): `?? ../docs/  ?? ./  ?? ../qcbm_share_20260626.zip` — 전부 untracked.
보존: 모든 기존 소스/데이터/결과(import/재사용만). 신규 파일만 생성.

## 3. 확인한 함수/경로 (코드에서 직접 재확인)
- 인코딩→경험적 분포 빌드(모두 benign-train empirical dist `q_train`(길이 2^n) 반환, import 시 부작용 없음 / `__main__` 가드):
  - HOIC: `tools/train_qcbm_iat_b2.py:94` `build_b2()` — NQ=12 (4096), `data/cic_0221_ddos.csv`
  - Bot:  `tools/train_qcbm_bot_resumable.py:86` `build_bot()` — NQ=12 (4096), `data/cic_bot_0203.csv`
  - UNSW: `tools/train_qcbm_unsw_resumable.py:79` `build_unsw()` — NQ=10 (1024), `unsw_nb15_*-set.csv`
- `empirical_dist(states, nstates)` @ `experiment2.py:119` (bincount 정규화), `marginal_dist(states, nq)` @ `train_qcbm_iat_b2.py:78` (비트독립 곱).
- `E.log` @ `experiment2.py:62` = stdout print만(파일 미수정, 안전).
- import 패턴: `diagnose_param_efficiency_unsw.py:27-36` (QCBM_DIR/TOOLS_DIR sys.path + os.chdir) 재사용.
- 라이브러리: numpy/scipy/pandas/matplotlib (sklearn 없음, scipy.linalg.hadamard 미사용) → FWHT 자체구현.

## 4. 각 데이터셋 비트 수 (확인 결과)
| 데이터셋 | bits | states | 인코딩 | benign train split |
|---|---|---|---|---|
| HOIC | 12 | 4096 | proto2+syn1+psh1+size4+fwdIAT2+flowIAT2 | seed=7, 0.7 |
| Bot  | 12 | 4096 | 동일 12bit | seed=7, 0.7 |
| UNSW | 10 | 1024 | proto2+size4+fwdIAT2+flowIAT2 (syn/psh 제외, flow=M2) | seed=7, 0.7 |

## 5. SVD 랭크 출처(데이터셋 native 인코딩과 일치)
- HOIC/Bot: `results2/attack_complexity.json` (native 12bit) — reach(TV≤0.05): HOIC 2, Bot 7.
- UNSW: `results2/cross_dataset_diagnosis.json` (10bit) — reach(TV≤0.05): UNSW 12.
- (각 데이터셋 내에서 SVD rank와 high-order 스펙트럼이 동일 인코딩으로 계산됨.)

## 6. 핵심 수학 / self-check 설계
- spin s_i=1−2·bit_i. raw k-body correlator p_hat(S)=Σ_x p(x)(−1)^{popcount(S&x)}=E[∏_{i∈S}s_i] = 자연순서 FWHT(p)[S].
  W_k = Σ_{popcount(S)=k} p_hat(S)². (FWHT 1회 O(2^n·n).) k=0(상수항, p_hat=1) 비율에서 제외.
- low_order_ratio=(W1+W2)/Σ_{k≥1}W_k, high_order_ratio=(Σ_{k≥3}W_k)/Σ_{k≥1}W_k.
- self-check: (a) n=4에서 FWHT W_k == 직접 곱-평균 W_k (assert). (b) 독립(곱) 분포: connected 2-body≈0 (assert);
  raw 고차는 0이 아니라 기하 감쇠(정직히 출력).
- 해석 주의(.md 기록): raw 스펙트럼은 분포 peakiness/단일비트 bias에 민감(델타함수는 전 차수 평탄).
  → marginal(독립) 분포의 스펙트럼을 baseline으로 병기하고, connected-2와 누적곡선으로 보완. 단정 금지.

## 7. self-check 결과
- **WHT vs 직접 곱-평균**: n=4 임의 분포에서 max|W_k차이| = **2.78e-17** → 일치(PASS).
- **독립(곱) 분포**: connected 2-body = **4.53e-33 ≈ 0**(PASS, assert). raw high_order_ratio = **0.0034**
  (정확히 0 아님 — raw 고차는 ∏E[s_i]로 기하 감쇠하는 raw 지표의 한계를 그대로 보여줌. 정직히 기록).

## 8. 데이터셋별 차수별 가중치 스펙트럼(W_k) ★전수(no sampling)
benign-train empirical dist 사용. 점유: HOIC 94/4096, Bot 361/4096, UNSW 174/1024.

| 데이터셋 | bits | 점유 | raw low(k≤2) | **raw high(k≥3)** | W_k 정점 차수 | 90% 누적 도달 k | marginal high(k≥3) | connected-2body |
|---|---|---|---|---|---|---|---|---|
| HOIC | 12 | 94   | 0.182 | **0.818** | k=4 | k=6 | 0.334 | 2.022 |
| Bot  | 12 | 361  | 0.064 | **0.936** | k=5 | k=8 | 0.173 | 4.340 |
| UNSW | 10 | 174  | 0.140 | **0.860** | k=4 | k=7 | 0.001 | 2.388 |

- raw 스펙트럼은 세 데이터셋 모두 **중간 차수(k=4~5)에 정점**, 고차 비중(k≥3)이 0.82~0.94로 큼.
- marginal(독립) baseline의 high는 HOIC 0.33 / Bot 0.17 / **UNSW 0.001**(UNSW 단일비트 marginal이 거의 균형 → 독립모델은 사실상 순수 저차).

## 9. SVD 랭크 vs high_order_ratio 대조 표 (각 데이터셋 native 인코딩 일치)
| 데이터셋 | SVD reach(TV≤0.05) | raw high_order_ratio | marginal high | (참고) raw high가 rank를 따라가나? |
|---|---|---|---|---|
| HOIC | **2** (최저) | 0.818 | 0.334 | 아니오 — 랭크 최저인데 raw 고차 큼 |
| Bot  | 7 | **0.936** (최고) | 0.173 | 아니오 |
| UNSW | **12** (최고) | 0.860 | **0.001** | 아니오 — 랭크 최고지만 raw 고차는 Bot보다 낮음 |

## 10. 사실 관찰 (판정 아님)
- **가설("UNSW는 SVD rank 높지만 high_order_ratio 낮다")은 raw 지표로는 지지되지 않음**:
  UNSW raw high = **0.860(높음)**, 낮지 않다. 즉 raw Pauli-Z 스펙트럼만으로는 "UNSW가 사실 저차"라고
  말할 수 없다.
- **raw high_order_ratio는 SVD rank를 따라가지 않는다**(서로 다른 복잡도 척도): HOIC는 SVD rank 최저(2)인데
  raw 고차 0.82, Bot이 raw 고차 최고(0.94). 두 척도(2D-reshape 저랭크 vs Pauli 차수)가 측정 대상이 다름.
- **raw 지표의 confound(정직히 기록)**: raw 스펙트럼은 (1) 분포 peakiness/희소성(델타함수는 전 차수 평탄),
  (2) 단일비트 bias(독립이어도 ∏E[s_i]로 고차 비0)에 동시 민감. 따라서 raw high 단독은 "기약 고차 상관"의
  깨끗한 지표가 아니다. self-check가 이 한계를 수치로 보여줌(독립분포 raw high=0.0034≠0).
- **상대적으로 깨끗한 신호(marginal baseline 대비)**: UNSW는 marginal high≈0.001 인데 empirical high=0.86 →
  UNSW의 고차 성분은 거의 전부 "독립 초과(=진짜 상관)"에서 옴. HOIC는 marginal high=0.33으로 겉보기 고차의
  상당 부분이 단일비트 bias. connected-2body는 Bot(4.34) > UNSW(2.39) > HOIC(2.02).
- 고차 비중(raw)이 상대적으로 가장 높은 데이터셋 = **Bot(0.936)**.

## 11. 생성 파일, 실행 명령과 결과
신규 생성: `tools/diagnose_higher_order_correlation.py`, `results2/higher_order_correlation.json`,
`results2/higher_order_correlation.png`, 본 로그.
```
python -m py_compile tools/diagnose_higher_order_correlation.py      # OK
python tools/diagnose_higher_order_correlation.py --help             # OK
python tools/diagnose_higher_order_correlation.py                    # self-check PASS, 3 데이터셋 저장
```
실패/에러 없음. QCBM 무학습. dependency 무설치(numpy/scipy/pandas/matplotlib). FWHT 자체구현(무의존).
build_* 는 CSV 읽기만(쓰기 없음), E.log는 stdout print만.

## 12. git diff --stat (기존 무변경)
`git diff --stat` = 변경 없음(tracked). `git status --short` = `?? docs/  ?? qcbm/  ?? *.zip`(프로젝트 전체 untracked,
신규 파일은 그 안). 기존 소스/데이터/결과 무수정·무삭제·무덮어쓰기.

## 13. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 인코딩/경험적분포/데이터로드 함수·경로 코드에서 확인·기록(§3).
- [x] 스크립트 compile + --help 동작.
- [x] self-check 통과: WHT==직접계산(2.8e-17), 독립→connected-2body≈0.
- [x] 3 데이터셋(HOIC/Bot/UNSW) W_k/low·high ratio 산출, json/png 저장. ★전수, 점유 명시.
- [x] SVD rank vs high_order 대조표 기록(§9), UNSW 가설 일치 여부 사실 기록(§10 — raw로는 불일치).
- [x] 기존 무변경, QCBM 무학습, dependency 무설치.

**남은 불확실성**
- raw 지표가 peakiness+bias에 confound됨 → "기약 고차 상관"의 깨끗한 척도가 아님. 결론은 raw 단독으로 단정 불가.
- connected은 2-body만 계산(고차 cumulant=Ursell 함수 미산출).
- HOIC/Bot은 12bit native, UNSW는 10bit — bit수 차이로 차수 절대비교에 주의(데이터셋 내 비교는 유효).

**다음 추천 단계**
1. **confound-free 척도**: 전 차수 connected correlator(cumulant/Ursell) 스펙트럼, 또는 저차 marginal 제약
   max-entropy 분포 대비 KL(=정확히 "저차로 설명 안 되는 잔여")로 고차 본질을 재측정.
2. peakiness 통제: 동일 점유/엔트로피의 합성 분포 대비 정규화, 또는 occupied-support 한정 분석.
3. (해석 입력) 웹 AI에 "raw Pauli 차수는 SVD rank와 무관하고 UNSW를 저차로 보여주지 않음; 깨끗한 판정엔
   cumulant/KL 척도 필요" 전달.
