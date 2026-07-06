# 잔여 병목 분리 (i) — KL 과민성 진단 (n10 0.14 정체가 KL 1/p 험함 때문인가, loss=KL 유지)

- 작성: 2026-07-01 05:50 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). **loss=KL 유지(안 바꿈)**. 기존 코드 무수정(import). 단정 금지(이 ansatz·완화법·KL 한정). 판정은 웹 AI.

## 1. 작업 목적
직전: identity-block(pert0.1) 완화로 n10 KL 0.679→0.139, gradient 18.7배 보존(BP 완화)했으나 바닥(1e-3) 미도달.
gradient 살아있는데 KL 정체 → 잔여 병목. 후보 (i)KL 과민성 (ii)국소최적 (iii)표현력. 이 작업은 **(i)**를 진단.
가설: near-δ₀ 시작 → 대부분 상태 p≈0 → forward KL −Σq·log(p+ε)의 log 폭발(1/p 지형) 험함. 두 축으로 진단.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (cost/ε, pert 초기분포, 상태별 KL기여)
- SBP(exp_strong_bp_mitigation): cost=−Σq·log(p+EPS), **EPS=1e-10**. `train_idblock(q,nq,L,epochs,seed,lr,pert)`→(p_final,ce).
  `make_idb`=Grant V†V. init: rng(2000+seed), wA0=uniform, wB0=wA0+N(0,pert). `synth_low`=t=0 저차. n10 P=180(L6).
- 재사용(무수정): SBP.train_idblock/make_idb/synth_low, OD.kl_div. 초기 p_init 은 동일 init 규약 재현 후 회로 평가.
- 상태별 KL 기여 c_s=q_s·log((q_s+ε)/(p_s+ε)) (q_s>0), ε 민감도(1e-10~1e-4) 순수 계산 가능.

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: cost/ε, identity-block init, 재사용 함수. 직전 n10 idblock 0.139.
- 추정: pert↑ 초기확산 → 1/p 험함 완화 → KL↓ 가능(단 gradient도 커져 혼재). 축2 소수 상태가 KL 지배면 과민.
- 확인 불가: 이번은 KL "진단"까지(해결=loss 교체는 별도).

## 5. 설계
축1: n10 identity-block pert∈{0.1,0.3,0.5,0.8}, 각 초기 max p·엔트로피 + 학습(seeds2, 300ep) best KL. pert↑→KL↓?
축2: best 최종 p 에서 상태별 KL기여 정렬→상위k 지배율, ε민감도, 상위상태(q vs p) 성격. self-check: pert0.1≈0.139 재현, 초기maxp 단조↓.

## 맥락 정정: target 은 희소하지 않음
p_true(t=0 저차 합성)는 **full support 1024/1024**, H=2.21. 즉 "희소 분포" 자체가 아님 — 과민성 가설은 "학습 *시작점* near-δ₀"의 1/p 험함에 관한 것.

## 6. 축1 — pert(초기확산) 스윕, n=10, 400ep, seeds=2
| pert | 초기 max p | 초기 H | best KL | spread |
|---|---|---|---|---|
| 0.1 | 0.854 | 1.155 | **0.1389** | 0.068 |
| 0.3 | 0.232 | 4.975 | 0.1112 | 0.053 |
| **0.5** | 0.0166 | 6.299 | **0.0867** | 0.407 |
| 0.8 | 0.0080 | 6.494 | 0.1012 | 0.169 |
- **pert↑(초기 확산↑) → best KL 하락**(0.139→0.087, ~37%↓, pert0.5 최저; 0.8은 반등). → **near-δ₀ 시작이 장애의 일부**(가설 지지).
  단 바닥(MLP~1e-3)엔 여전히 **~87배 미도달**. ★pert↑는 확산+gradient 동시 변경이라 완전 격리 아님(명시).

## 7. 축2 — 최종 p(pert0.5, KL 0.0867)의 KL 과민성
- 상태별 KL 기여: **상위5 = 64.4%, 상위10 = 84.3%** 집중(소수 모드가 잔여 KL 지배).
- **ε 민감도: 0.0867(1e-10) → 0.0861(1e-6) → 0.0713(1e-4), ~18%만 변동 → 1/p log 폭발 아님.**
- 상위 기여 상태(q,p): s958(0.008,1.2e-3) s1(0.016,7.2e-3) s7(0.009,3.0e-3) s805(0.004,5.4e-4) s808(0.159,1.5e-1).
  → 일부는 p 작으나 **ε 무관**(특이성 아님), s808는 q0.16 vs p0.15로 큰 모드도 미세 미스매치. **1/p 특이성보다 "소수 모드 미스매치"**.

## 8. self-check
- **pert0.1 best KL = 0.1389 → 직전 identity-block 0.139 정확 재현** ✓. 초기 max p pert↑에 **단조 감소**(0.854→0.008) ✓. NaN 없음.

## 9. 사실 관찰 (판정 아님, 단정 금지)
- **(i) KL 과민성은 "부분적" 병목**:
  - 시작점(near-δ₀)은 실제 장애 — pert↑(더 퍼진 시작)로 KL 0.139→0.087 개선(축1). 이 부분은 KL의 1/p 험한 시작 지형과 부합.
  - **그러나 최종 잔여 KL은 1/p 특이성 아님**(축2: ε 무관, 상위 상태 p 0 근처 아님) → 잔여 핵심은 **소수 모드 미스매치**.
- 종합: **KL 과민성만으론 잔여를 설명 못 함** — 시작점 완화(pert↑)로 일부 내려가나 바닥 미도달, 최종 잔여는 특이성 아닌 모드 미스매치 → **(ii)국소최적/(iii)표현력 경계로 넘어갈 근거**.
- ★ 이 ansatz(StronglyEntangling)·완화법(identity-block)·**KL loss 유지** 한정. loss 교체·우위 언급 없음(다음 단계). 단정 금지.

## 10. 생성 파일 / 실행 / tmux
신규: `tools/exp_kl_sensitivity.py`, `results2/kl_sensitivity/{kl_sensitivity.json,png,run.log,smoke.log,kl_sensitivity_smoke.json}`, 본 로그.
```
python -m py_compile tools/exp_kl_sensitivity.py       # OK
python tools/exp_kl_sensitivity.py --smoke             # 동작(축1 pert0.1/0.5, 축2)
tmux new -s hoc_kls -d 'python3 tools/exp_kl_sensitivity.py --perts 0.1,0.3,0.5,0.8 --seeds 2 --epochs 400 ...'
```
- **tmux 세션 hoc_kls**, 로그 `results2/kl_sensitivity/run.log`. 축1 pert당 ~19~22분(2seed 400ep), 축2 즉시. 총 ~80분. 자동 종료.
- 기존 코드 무수정(SBP.train_idblock/make_idb/synth_low, OD import만). **loss=KL 유지**. 합성만. dependency 무설치. 실패 없음.

## 11. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`. 기존 무수정.

## 12. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] cost/ε(1e-10)·pert 초기분포·상태별 KL기여 확인·기록.
- [x] 축1 pert별 초기확산+best KL(pert↑→KL↓ 확인, pert0.5 최저).
- [x] 축2 상위 상태 KL지배율(상위5 64%)+ε민감도(미미)+상태성격(특이성 아님).
- [x] self-check(pert0.1 0.139 재현, 초기 maxp 단조).
- [x] (i) 판정: **부분 병목**(시작점 O, 최종 잔여는 모드 미스매치→(ii)/(iii)). loss 무교체·우위 무언급. 기존 무변경·합성만·무설치.

**남은 불확실성**
- pert는 확산+gradient 동시 변경 → "시작점만"의 순수 효과 미분리. 단일 p 스냅샷(pert0.5 best) 의존.
- 잔여 모드 미스매치가 (ii)국소최적인지 (iii)표현력 경계인지 미분리(다음 단계).

**다음 추천 단계**
1. (ii) 국소최적 진단: 같은 pert0.5서 restart↑(예: 10 seed)·epoch↑로 best KL이 더 내려가는가(내려가면 국소최적, 정체면 표현력).
2. (iii) 표현력 진단: P↑(L↑)로 그 소수 미스매치 모드가 잡히는가.
3. (해석 입력) 웹 AI에: "KL 과민성은 시작점(near-δ₀)엔 해당(pert↑로 0.139→0.087)하나 최종 잔여는 1/p 특이성 아님(ε무관)=소수 모드 미스매치 → (ii)/(iii)로. 이 ansatz·완화법·KL 한정."
