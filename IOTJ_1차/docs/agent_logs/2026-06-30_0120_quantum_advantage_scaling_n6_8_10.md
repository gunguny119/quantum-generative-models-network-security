# 양자 우위 스케일링 — n=6,8,10 깊이 곡선으로 "고전 대비 QCBM 효율 격차의 n-추세"

- 작성: 2026-06-30 01:20 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). 기존 코드 무수정(import). 단일 프로세스. 단정 금지(추세 기록). 판정은 웹 AI.

## 1. 작업 목적
직전 n=6: QCBM 깊이↑(L≥4)로 동일 P MLP 동급 도달(둘 다 바닥) — 우위 판정 불가(64상태 쉬움). 양자 우위는 "한 분포
승리"가 아니라 n↑에서 고전 대비 효율 격차가 벌어지는 추세. n=6,8,10 동일 깊이 곡선으로 교차점 P*·격차의 n-이동 측정.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (n 파라미터화 등)
- 재사용(모두 n-generic): `KLV.train_kl_one(q,nq,L,epochs,seed)`, `SY.synth_dist/make_terms/ge3_share`,
  `CL.nmf_reconstruct(side=2^(n//2))/mlp_energy_fit(exact Z over 2^n)/fit_marginal/eff_rank`, `E.wshape/empirical_dist`, `OD.kl_div`.
- `exp_depth_expressivity.py`는 NQ=6 하드코딩 → 새 스크립트 `exp_quantum_advantage_scaling.py`에서 n 파라미터화.
- 고전 P맞춤: NMF rank=round(P/(2·side)), MLP hidden=round(P/(n+2)). QCBM params=L·n·3.
- N=500·2^n → 샘플바닥=1/(2·500)=1e-3 (n 무관). 공정성: 복잡도(ge3_share) n별 기록, MLP/QCBM 다seed best.

## 4. ★ 시간 게이트 (n=10 리스크)
QCBM 비용 ∝ params×2^n. n=6 실측 (L8,700ep,4restart)=2486s. n=10 1점 수 시간 우려 → `--time-only`로 n8/n10-L4 단일점
실측 후 n=10 포함/축소/생략 결정. (아래 §6에 실측 기록.)

## 5. 확실한 사실 / 추정 / 확인 불가
- 확실: 재사용 함수 n-generic, 고전 P맞춤·N스케일. n=6 직전: L≥4 QCBM≈MLP≈바닥, L=3 anomaly.
- 추정: n↑→고정 P로 고전(특히 NMF) 바닥도달 어려워질 수 있음; QCBM 교차점 이동이 핵심.
- 확인 불가(시간): n=10 전체(게이트로 부분/생략 가능).

## 6. 시간 게이트 실측 + 최종 스코프
`--time-only`(L=4, 200ep, 1seed): n=6 52.7s / n=8 87.0s / n=10 **133.8s** → 500ep·3seed 환산 n=10≈17분/점. **감당 가능 → n=6,8,10 전부 포함.**
스코프: ns=6,8,10 × L=2,3,4,6 × {low, highrank}, seeds=3(QCBM/MLP/NMF best), epochs=500, N=500·2ⁿ. 모드 분리 실행(각 JSON 저장).
총 wall ≈ 4.5시간(low 01:28→03:42, highrank 03:42→05:57). n=10 점당 17~36분.

## 7. n별 복잡도(ge3_share/eff_rank/floor)
- low(t=0): ge3_share≈0(전 n), 샘플바닥 n6 0.0024 / n8 0.0012 / n10 0.0010 (≈1e-3 일정). **복잡도 잘 통제됨**.
- highrank(t=0.6 random): ge3_share n6 0.178 / n8 0.020 / n10 0.139, rank99 4/3/6 — **random seed 의존으로 n간 정확히 못 맞춰짐(실값 기록, 해석 주의)**. low가 깨끗한 비교축.

## 8. n별 깊이 곡선 — KL(p_true‖model), QCBM=3seed best (괄호=seed min~max)
**low (t=0, ge3≈0):**
| n | L=2 (P) | L=3 | L=4 | L=6 (best) | MLP(L=6) | NMF(L=6) |
|---|---|---|---|---|---|---|
| 6 | 0.054(36) | 0.197(54) | 0.0034(72) | **0.0006**(108) | 0.0005 | 0.0036 |
| 8 | 0.911(48) | 0.440(72) | 0.014(96) | 0.0036(144) | 0.0003 | 0.0233 |
| 10 | 1.686(60) | 1.427(90) | 0.654(120) | **0.421**(180) | 0.0010 | 0.0922 |
**highrank (random):**
| n | L=2 | L=3 | L=4 | L=6 | MLP(L=6) | NMF(L=6) |
|---|---|---|---|---|---|---|
| 6 | 0.501 | 0.320 | 0.0225 | 0.0093 | 0.0005 | 0.2547 |
| 8 | 1.117 | 0.609 | 0.452 | 0.0208 | 0.0009 | 0.0580 |
| 10 | 2.371 | 1.455 | 0.797 | **0.636**(min0.64~max1.38) | 0.0029 | 5.1484 |
- **MLP는 모든 n·모드에서 바닥 근처(~1e-3)**. NMF(저랭크)는 고랭크·큰 n에서 붕괴(n10 highrank 5~15).
- QCBM seed 분산 큼(n8 low L4 0.014~0.69, n10 hr L6 0.64~1.38) → 최적화 landscape/바렌플래토 정황.

## 9. 교차점 P* 추세 (핵심)
- **P*(QCBM≤MLP 최소 P) = None — 모든 n(6,8,10), 모든 모드.** QCBM이 어떤 깊이/P에서도 동일 P MLP를 못 따라잡음.
- **격차가 n↑에서 벌어짐(QCBM 불리)**: 같은 best 깊이 L=6에서 QCBM(low) **n6 0.0006 → n8 0.0036 → n10 0.421**(단조 악화), MLP는 n과 무관 ~1e-3.
- **바닥 도달에 필요한 깊이가 n보다 빠르게 증가**: n6는 L=4~6서 바닥, n8은 L=6서 근접, **n10은 L=6(P=180)서도 0.42로 미도달**.
- → "n vs P*"는 그릴 수 없음(P* 부재). 대신 "n vs (QCBM−MLP 격차)"가 **n↑일수록 확대** = 양자 우위 점근의 **반대** 추세.

## 10. self-check
- n=6 low 재현: L=3 anomaly **0.197**(직전 0.197), L=4 0.0034, L=6 0.0006 — 직전 깊이곡선과 일치 ✓.
- marginal은 구조 있는 곳에서 KL 큼(하한 역할, 2.4~4.2). NMF/MLP/QCBM 모두 marginal보다 낮음(저차는 정상).

## 11. 사실 관찰 (판정 아님, n=6~10 한정, 단정 금지)
- 이 통제 합성 스케일링(StronglyEntangling·KL목적·Adam, n=6→8→10)에서 **QCBM의 고전(MLP) 대비 효율 격차는 n↑일수록 확대**
  — 양자 우위 점근 신호가 **아니라 그 반대**. 교차점 P* 부재(모든 n).
- 원인 정황: 큰 n에서 깊은 회로의 **최적화 실패**(seed 분산 폭증, 같은 깊이로 바닥 미도달) = 바렌플래토류 trainability 벽으로 보임
  (표현력 자체가 아니라 학습). 단 **이 ansatz·이 옵티마이저 한정**, "양자 일반의 한계" 아님.
- 고전 비교의 실질 기준은 MLP(전 구간 바닥); NMF 저랭크는 고랭크서 붕괴하나 MLP가 메워 양자가 비집을 틈 없음.
- 한계: highrank의 ge3_share가 n간 불일치(0.178/0.020/0.139) → 고랭크 모드 n-비교는 약함. **low 모드(복잡도 통제)가 주 결론축.**

## 12. 생성 파일 / 실행 / tmux
신규: `tools/exp_quantum_advantage_scaling.py`,
`results2/qa_scaling/{qa_scaling_low,qa_scaling_highrank}.{json,png}`, `run.log`, `time_gate.log`, 본 로그.
```
python -m py_compile tools/exp_quantum_advantage_scaling.py      # OK
python tools/exp_quantum_advantage_scaling.py --time-only --ns 6,8,10   # 게이트(n10 17분/점)
tmux new -s hoc_qa -d '... --modes low --tag low; ... --modes highrank --tag highrank'  # 본 실행
```
- **tmux 세션 hoc_qa**(+게이트 hoc_qa_time), 로그 `results2/qa_scaling/run.log`. 모드 분리(각 JSON). 단일 프로세스. 약 4.5h. 자동 종료.
- 기존 코드 무수정(KLV/SY/CL/experiment2 import만, 모두 n-generic). 합성만(실데이터 미사용). dependency 무설치. 실패/크래시 없음.

## 13. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 14. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] n 파라미터화·임의 n 동작·n별 시간(게이트) 확인·기록.
- [x] n=6,8,10 P-vs-KL 곡선(QCBM/MLP/NMF)+샘플바닥, json/png(두 모드).
- [x] 복잡도(ge3_share) n별 기록(low 통제 OK, highrank 불일치 명시).
- [x] 교차점 P*(=None 전 n)·"n vs 격차 확대" 추세. MLP 다seed·QCBM 다seed(분산 기록).
- [x] n=6 재현. n=10 전부 수행(게이트 통과). 해석 n-추세 한정·양자우위 단정 없음. 기존 무변경·합성만·무설치.

**남은 불확실성**
- highrank ge3_share n간 불일치 → 고랭크 n-비교 약함(low가 주축).
- QCBM 비단조(L=3)·대분산 잔존 → 더 많은 restart/seed·다른 init이면 일부 완화 가능(추세 방향은 견고).
- 결과는 StronglyEntangling·Adam·KL목적 한정. 다른 ansatz/옵티마이저/바렌플래토 완화는 미검.

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "n=6→10 통제 스케일링서 QCBM-고전 격차가 n↑일수록 확대(교차점 부재); 원인은 trainability(바렌플래토)
   정황. 이 ansatz·옵티마이저 한정." 양자 정당화 최종 정리.
2. **trainability가 병목임을 직접 입증**: identity-block init·layerwise 학습으로 큰 n QCBM이 바닥 도달하는지(같은 P) 재시험.
3. highrank를 ge3_share·eff_rank 일치하도록 재설계(n간 공정), 또는 다른 ansatz(문제맞춤/얕은-넓은)로 격차 추세 재확인.
