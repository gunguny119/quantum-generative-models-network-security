# 강한 BP 완화 — identity-block(Grant V†V) + layerwise 로 n10 QCBM 바닥 도달 여부 (KL 유지)

- 작성: 2026-07-01 00:46 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). loss=KL 유지, lr 동일(loss-종속 미세튜닝 배제). 기존 코드 무수정(import). 단정 금지(이 ansatz·완화법 한정, 우위와 별개). 판정은 웹 AI.

## 1. 작업 목적
직전 near-identity(소각) 부분완화(n10 KL 0.643→0.391, 바닥 미도달). 더 강한 loss-독립 완화(정확 identity-block + layerwise)로
n10(+n8)이 같은 P로 바닥(MLP~1e-3) 도달하는지. BP 나는(고전 어려운) 영역서 양자가 학습 성공하는가.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (identity-block, layerwise) — ★핵심 발견 포함
- ansatz StronglyEntanglingLayers: 게이트순서 역전 불가 → 직접 항등 구성 불가 → **Grant V†V**: `SEL(wA)+qml.adjoint(SEL)(wB)`, 각 L/2층, 총 params=L·n·3(동일 예산). `qml.adjoint(qml.StronglyEntanglingLayers)` 유효 확인(wB=wA→p[0]=1=U=I).
- **★ exact identity-block(wB=wA)은 확률/KL 손실의 임계점(p₀=1 최대)이라 1차 gradient=0(측정 |g|~1e-16=함정).** Grant는 선형 관측량 ⟨O⟩엔 유효하나 확률(2차)엔 부적합 → **pert 필수**.
- gradient 측정(n=8, ⟨Z₀⟩ 중간층 RY, R=30): uniform Var 1.0e-3 / near-id σ0.1 5.5e-5(cold, 더 작음) / **idblock pert0.1 6.9e-3(uniform의 ~6.6배)** / pert0 ~0. → **idblock(pert~0.1)이 uniform보다 grad 큼 = 강한 완화**. M3 pert=0.1 확정.
- layerwise: full SEL L=2→4→6 warm-start(학습층 복사+새 층 소각 append). 단 새 층 CNOT링이 상태를 바꿔 완전 no-op은 아님(warm-start 근사).
- 재사용(무수정): `E.make_circuit/wshape`, `SY.make_terms/synth_dist`, `OD.kl_div`, qml/pnp.

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: Grant V†V 유효·exact=grad0·idblock(pert0.1) grad≈6.6×uniform. 재사용 n-generic. 직전 baseline n10 uniform 0.643/near-id 0.391.
- 추정: 큰 grad(idblock)이 학습 수렴 도와 바닥 근접 가능(보장X). layerwise 추가 도움 가능.
- 확인 불가: 정확 Grant는 선형관측량용 — 확률 손실선 pert-idblock이 실질 대안(문서/측정으로 확정).

## 5. 설계
loss=KL(−Σ p_true·log(p+ε)), lr=0.1 동일. 4방법(같은 P=L·n·3, 같은 epoch, seeds best): M1 uniform / M2 near-id(N(0,0.1)) /
M3 identity-block(V†V, wB=wA+N(0,0.1)) / M4 layerwise(L2→4→6 warm-start). n=10(주)+n=8. + gradient 재측정(uniform vs idblock).

## 6. n10(+8) 4방법 비교 (P=L·n·3=180/144, L=6, epochs=400, seeds=3, lr=0.1, best KL)
**n=10 (핵심):**
| 방법 | best KL | mean | seed spread |
|---|---|---|---|
| uniform | 0.6794 | 1.161 | 0.807 |
| near_identity | 0.3928 | 0.697 | 0.678 |
| **identity_block(Grant V†V,pert0.1)** | **0.1389** | 0.173 | **0.068** |
| layerwise | 0.5583 | 0.952 | 0.689 |

**n=8 (충분학습이면 대체로 바닥):**
| 방법 | best KL | spread |
|---|---|---|
| uniform | 0.0064 | 0.0003 |
| near_identity | 0.0097 | 0.007 |
| **identity_block** | **0.0022** | 0.015 |
| layerwise | 0.0146 | 0.038 |

- **identity-block이 4방법 중 최강**: n=10 uniform 0.679 → **0.139**(~5배↓), seed 분산도 0.807→**0.068**(대폭↓). n=8도 최저(0.0022).
- **layerwise(현 구현)는 도움 안 됨**(n10 0.558, near-id보다 나쁨) — 새 층 CNOT링 교란 + 스테이지당 epoch 부족. 정직히 기록.

## 7. 바닥 도달 판정
- **n=10: 미도달(부분개선)**. 최강(identity-block) KL=**0.139**도 바닥(MLP~1e-3)보다 **~140배 높음**. → 강한 loss-독립 완화로도 n=10은 바닥 미도달.
- n=8: 도달(모든 방법 ~0.002~0.015, 충분학습시 BP 비제약).

## 8. gradient 보존 (idblock vs uniform) + BP 재확인
- ⟨Z₀⟩ grad 분산: **n=10 uniform 3.35e-4 → identity-block 6.24e-3 (18.7배↑)**, n=8 5.5배↑. → **완화가 gradient를 크게 보존/증대**(BP 자체는 실제로 완화됨).
- ★결정적: **gradient는 보존됐는데(18.7×) KL은 바닥 미도달(0.139)** → **BP가 유일 병목이 아님**. gradient가 살아도 못 내려간 잔여 = BP 외 요인(다음 §10).
- (exact identity-block은 확률/KL 손실의 임계점이라 grad0=함정 — pert 필수, §3 확인.)

## 9. self-check
- baseline 재현: uniform n10 **0.6794**(직전 0.643, seed 분산 큼 감안 일치), near-identity n10 **0.3928**(직전 0.391 거의 정확 재현) ✓.
- NaN/발산 없음. identity-block(pert) 안정.

## 10. 사실 관찰 (판정 아님, 단정 금지)
- **강한 완화로 바닥 도달? — 아니오(부분)**: identity-block이 n=10을 크게 개선(0.68→0.14)·분산 급감·gradient 18.7배 보존했으나 바닥(1e-3) 미도달.
- **BP가 유일 병목? — 아니오**: gradient가 보존됐는데도 KL이 0.14에서 멈춤 → **BP는 한 병목이고 완화되나, 잔여 병목이 별도로 존재**.
  원인 후보(§작업목적 구분): (b) **BP 외 요인** — (i) KL의 희소상태 과민성(near-δ₀ 시작에서 1/p 지형이 험함), (ii) 국소최적, (iii) 이 예산(P=180)의 표현력 경계. gradient가 살아있음은 (a)완화부족을 배제 → (b)로 좁혀짐.
- **"고전 어려운(BP) 영역서 양자 학습 성공?" — 부분 성공/미완**: BP는 완화(gradient 보존)됐으나 목표분포 학습 완료(바닥)엔 미달 → 이 영역서 양자가 학습에 "완전 성공"했다고 볼 수 없음.
- ★ 모두 **이 ansatz(StronglyEntangling)·이 완화법·KL 손실·lr=0.1 한정**. **우위와 별개**(바닥 도달해도 고전 대비 우위는 별개 문제이고, 여기선 도달조차 못 함).

## 11. 생성 파일 / 실행 / tmux
신규: `tools/exp_strong_bp_mitigation.py`, `results2/strong_bp_mitigation/{strong_bp.json,strong_bp.png,run.log,smoke.log,strong_bp_smoke.json}`, 본 로그.
```
python -m py_compile tools/exp_strong_bp_mitigation.py                 # OK
python tools/exp_strong_bp_mitigation.py --smoke                       # 동작·안정성(identity-block best 0.0055)
tmux new -s hoc_sbp -d 'python3 tools/exp_strong_bp_mitigation.py --ns 10,8 --L 6 --epochs 400 --seeds 3 --R 40 ...'
```
- **tmux 세션 hoc_sbp**, 로그 `results2/strong_bp_mitigation/run.log`. n=10 방법당 ~14~32분(3seed), n=8 ~9~20분. 총 ~2.4h. 자동 종료.
- 기존 코드 무수정(E.make_circuit/wshape, SY, OD import만; identity-block(qml.adjoint)·layerwise·grad는 새 파일). 합성만. dependency 무설치. 실패/NaN 없음.

## 12. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 13. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] identity-block(Grant V†V)·layerwise·wshape 확인·기록 + ★exact=임계점(grad0) 발견.
- [x] n10(+8) 4방법 best KL+분산 표 json/png.
- [x] 바닥 도달 판정: **n=10 미도달(부분)**, 최강 0.139 vs 1e-3.
- [x] gradient 보존(idblock 18.7×uniform) + BP 재확인.
- [x] self-check baseline 재현(uniform 0.68/near-id 0.393).
- [x] 미도달 원인 후보 구분: gradient 보존됨 → (b)BP외 요인(KL과민성/국소최적/표현력)으로 좁힘.
- [x] 해석 "이 ansatz·완화법·KL·lr 한정·우위와 별개". 기존 무변경·합성만·무설치.

**남은 불확실성**
- 잔여 병목(BP 외)의 정체 미분리: KL 과민성 vs 국소최적 vs 표현력 경계. (다음 단계서 loss 교체/더 큰 P/재시작↑로 분리.)
- layerwise 저조가 구현(스테이지 epoch·CNOT append) 탓인지 방법 자체 탓인지 미분리.
- n=10 seed 분산 여전(3 seed). pert(0.1) 스윕 미실시(더 큰 pert가 더 나을지 미검).

**다음 추천 단계**
1. **잔여 병목 분리**: (i) KL 과민성 — near-δ₀ 대신 균등 근처 시작 또는 KL 대신 MMD/역KL로 바닥 도달 바뀌는지(단 loss 교체는 별도). (ii) 재시작·epoch↑로 국소최적 여부. (iii) P↑(L↑)로 표현력 경계 여부.
2. identity-block **pert 스윕**(0.1/0.3/0.5)·seeds↑로 n=10 최소 KL 추가 하강 여지.
3. (해석 입력) 웹 AI에: "강한 BP 완화(identity-block)로 n10 대폭 개선·gradient 18.7×보존하나 바닥 미도달 → BP는 한 병목이나 잔여(KL과민성/국소최적/표현력)가 별도. 이 ansatz·완화법 한정, 우위와 별개."
