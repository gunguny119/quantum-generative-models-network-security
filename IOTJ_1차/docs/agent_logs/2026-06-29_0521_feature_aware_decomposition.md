# feature-인지 분해 — joint gain / 고차 상관에서 "feature 내부 상관" 오염 제거 후 재측정

- 작성: 2026-06-29 05:21 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 순수 측정(학습 없음). 경험적 분포 + 기존 평가/maxent 재사용. import만. 단일 프로세스. 판정은 웹 AI.

## 1. 작업 목적
현 marginal은 **비트 단위 독립**이라 joint gain·고차 상관에 (a)feature 내부 비트상관·(b)feature 간 상관이 섞임.
feature 경계를 인지해 (a)/(b)를 분리하고, joint gain과 고차가 **feature 간(b)만으로도 살아남는지** 재측정.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/  ?? qcbm/  ?? *.zip` — 전부 untracked. tmux=/usr/bin/tmux. 보존: 모든 기존 파일(읽기/ import).

## 3. 확인한 함수/경로
- AUC: `experiment2.auc_score(scores,labels)` @266 (raw MWU), 점수=−log q[state] (`anomaly_eval` @285 / `auc_of` 패턴).
- 분포: `experiment2.empirical_dist`(joint) @119, `train_qcbm_iat_b2.marginal_dist`(bit_marg) @78.
- maxent: `tools/diagnose_order_decomposition.py` `maxent_upto(p,nq,bitmat,kmax,...,Phi=None)` @102 (L-BFGS, spin Φ; Φ 명시 가능), `kl_div`@72, `bit_matrix`@67 재사용.
- build: `build_b2/build_bot/build_unsw` → st_tr/st_te/st_atk, q_train(joint). **QCBM 체크포인트 불필요**(경험적 분포 비교).
- 해리: `tools/analyze_tv_auc_dissociation.py`는 QCBM 16 restart의 (TV/MMD² vs AUC) 상관 → feature_marginal(단일분포)로 재정의 불가 → **goal4 확인불가**(새 정의 금지).
- 비트→feature: HOIC/Bot proto{10,11} syn{9} psh{8} size{4-7} fwd{2,3} flow{0,1}; UNSW proto{8,9} size{4-7} fwd{2,3} flow{0,1}.

## 4. feature_marginal 정의 / self-check
- feat_marg = 각 feature 비트셋의 결합 marginal(bincount) 추정 후 feature끼리 외적(독립 곱). 적합 없이 정확.
- 정보기하: bit_marg ⊂ feat_marg ⊂ joint (nested maxent) → KL(joint‖bit_marg)=KL(joint‖feat_marg)+KL(feat_marg‖bit_marg).
  within(A)=KL(feat_marg‖bit_marg), cross(B)=KL(joint‖feat_marg).
- self-check: (i)1 feature=전체→feat_marg==joint; (ii)각 1비트 feature→feat_marg==bit_marg; (iii)독립 feature→cross≈0; (iv)단조·음수없음.

## 4-1. self-check 결과 (PASS)
(i) feature 1개=전체비트 → KL(joint,feat_marg)=**0.0** ; (ii) feature=각 1비트 → KL(feat_marg,bit_marg)=**5e-17** ;
(iii) 독립 두 feature → cross B=**1.4e-4≈0**(표본오차) ; (iv) nested 제약열 **단조 True**. → 분해 도구 정상.

## 5. 확실한 사실 / 추정 / 확인 불가
- 확실(코드/측정): feat_marg 정의, AUC 분해, 정확 정보량 within/cross(Pythagoras 잔차~0), 차수×within/cross.
- 추정 없음. 확인 불가: 해리 재검증(goal4, §9) — feature_marginal로 재정의 불가(새 정의 금지).

## 6. AUC 분해표 (score=−log q[state], raw MWU)
| 데이터셋 | bit_marg | feature_marg | joint | contamination(feat−bit, feature내부) | **pure_cross(joint−feat, feature간)** |
|---|---|---|---|---|---|
| UNSW | 0.7027 | 0.7460 | 0.8705 | +0.0433 | **+0.1245** |
| Bot  | 0.2870 | 0.2037 | 0.8749 | −0.0833 | **+0.6712** |
| HOIC | 0.8741 | 0.8774 | 0.9258 | +0.0034 | **+0.0484** |
- 탐지 우위는 대부분 **feature 간**(pure_cross). Bot은 contamination 음수(feat_marg가 bit_marg보다 더 나쁨, AUC<0.5 raw) →
  Bot 탐지신호는 사실상 전부 joint cross-feature.

## 7. 정보량 within/cross (정확, nats)
| 데이터셋 | total=KL(joint‖bit) | within A=KL(feat‖bit) | **cross B=KL(joint‖feat)** | **B/total** | Pythagoras 잔차 |
|---|---|---|---|---|---|
| UNSW | 2.2841 | 0.0837 | **2.2005** | **0.963** | 4.4e-16 |
| Bot  | 2.9191 | 0.2895 | **2.6296** | **0.901** | 0.0 |
| HOIC | 1.4608 | 0.0191 | **1.4416** | **0.987** | −9e-11 |
- joint−bitmarg 정보량의 **90~99%가 순수 feature 간**. within-feature 오염은 1~10%로 작음(A+B=total 정확).

## 8. 차수×{within/cross} 분해 (≥3차 중 feature 간 비중)
| 데이터셋 | ≥3차 within | ≥3차 cross | resid(≥5,전부 cross) | ≥3차 총합 | **≥3차 cross 비중** | (직전 KL₂ 일치) |
|---|---|---|---|---|---|---|
| UNSW | 0.0103 | 0.6067 | 0.0541 | 0.6170 | **0.983** | =0.617 ✓ |
| Bot  | 0.0848 | 0.4645 | 0.0005 | 0.5493 | **0.846** | =0.549 ✓ |
| HOIC | 0.0087 | 0.0086 | 0.0000 | 0.0173 | **0.497** | =0.017 ✓ |
- **UNSW ≥3차 중 feature 간(B) = 98.3%** — 직전 "고차 27%"는 within-feature 오염이 아니라 거의 전부 진짜 feature 간 고차.
- Bot ≥3차도 85% cross. HOIC는 ≥3차 절대값이 미미(0.017)하고 within/cross 반반(절대값 무의미).
- ≥3차 총합이 직전 order_decomposition KL₂와 정확 일치(교차검증). 단조성 모두 True, 음수 기여 없음.
- 수렴: 일부 고차 maxent 적합 maxiter 도달(UNSW within4/cross4, CIC cross3/cross4 등) — 위반 ≤3.6e-6, 단조·정합으로 값 신뢰.

## 9. 해리 재검증 (확인 불가)
`tools/analyze_tv_auc_dissociation.py`는 QCBM 16 restart의 (TV/MMD² vs AUC) 상관 측정(qcbm_iat_no_psh_restarts.npz).
feature_marginal은 단일 분포라 restart-fit-vs-AUC 해리에 끼울 수 없음 → 재측정하려면 새 해리 정의 필요(금지).
→ **확인 불가**로 기록(임의 정의 안 함). goal1–3만 수행.

## 10. 사실 관찰 (판정 아님)
- **joint gain·고차 상관은 within-feature 오염이 작고 거의 전부 feature 간**: 정보량 B/total 90~99%, AUC pure_cross가 우위 대부분,
  ≥3차 cross 비중 UNSW 98%/Bot 85%. → 직전 "고차 부풀림" 우려는 (UNSW/Bot에서) 기각, 진짜 feature 간 고차 상관 존재.
- HOIC만 ≥3차 절대량이 미미(0.017) → 애초 고차가 거의 없는 분포(저차로 충분).
- (해석 주의) 이는 "feature 간 고차 상관이 실재"라는 구조적 사실. "양자가 이긴다"는 별개(단정 금지) —
  직전 결과는 그 고차가 있어도 고전 저랭크가 파라미터 효율 우위였음.

## 11. 생성 파일 / 실행 / tmux / 수렴
신규: `tools/diagnose_feature_aware.py`, `results2/feature_aware_decomposition.json`, `.png`, `feature_aware_run.log`, 본 로그.
```
python -m py_compile tools/diagnose_feature_aware.py     # OK
python tools/diagnose_feature_aware.py --help            # OK
python tools/diagnose_feature_aware.py --dataset unsw    # self-check PASS, UNSW(smoke)
tmux new -s hoc_feat -d 'python3 tools/diagnose_feature_aware.py 2>&1 | tee results2/feature_aware_run.log'  # 전체
```
- **tmux 세션명 hoc_feat**, 로그 `results2/feature_aware_run.log`. UNSW→Bot→HOIC 순차(병렬 없음), 전수. 완료 후 자동 종료.
- QCBM 무학습(경험적 분포만, 체크포인트 불필요). dependency 무설치. 실패/크래시 없음.

## 12. git diff --stat (기존 무변경)
`git diff --stat` = 변경 없음(tracked). `git status --short` = `?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안).

## 13. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] AUC/체크포인트(불필요 확인)/test·attack 로드/해리 코드 경로 확인·기록(§3).
- [x] feature_marginal + self-check 4종 통과(경계조건 i/ii, 독립 cross≈0, 단조).
- [x] 3 데이터셋 AUC(bit/feat/joint)+contamination+pure_cross json/png(§6).
- [x] 정확 within/cross 정보량(A+B=total, Pythagoras 잔차~0)(§7).
- [x] 차수×within/cross로 ≥3차 중 feature간 비중 산출(UNSW 98.3%)(§8), 직전 KL₂ 일치.
- [x] 해리=확인불가 명시(§9). 기존 무변경·무학습·무설치.

**남은 불확실성**
- 일부 고차 maxent 적합 gtol 형식 미달(maxiter, 희소+고차). 위반≤3.6e-6·단조·정합으로 값 신뢰하나 형식 수렴 아님.
- AUC 분해는 비가산(차분 해석). HOIC ≥3차 절대량 미미라 within/cross 비율 해석 주의.
- 비트수/인코딩 의존. feature 내부 차수(예: size 4비트 내부 구조)는 별도.

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "joint gain·고차는 within-feature 오염 작고 거의 전부 feature 간(정보량 90~99%, UNSW ≥3차 cross 98%).
   즉 진짜 feature 간 고차 상관 존재 — 단 그게 있어도 직전엔 고전 저랭크가 효율 우위였음." 양자 정당화 재정리.
2. cross-feature 고차가 '어느 feature 쌍/삼중'에 집중되는지(예: size×IAT×proto) feature-쌍별 KL로 국소화.
3. feature_marginal을 QCBM이 실제로 넘는지(restart 분포의 AUC vs feat_marg 0.746/0.204/0.877) 대조 — joint 학습이 cross를 잡았나.
