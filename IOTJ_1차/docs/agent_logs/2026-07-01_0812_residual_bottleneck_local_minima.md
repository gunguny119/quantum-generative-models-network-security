# 잔여 병목 분리 (ii) — 국소최적 진단 (n10 0.087 정체가 restart↑/epoch↑로 내려가나, P·loss=KL 고정)

- 작성: 2026-07-01 08:12 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만. **P(=180,L6)·loss(KL)·pert(0.5) 고정, restart·epoch만 키움**. 기존 코드 무수정(import). 단정 금지(이 ansatz·완화법·KL·P=180 한정). 판정은 웹 AI.

## 1. 작업 목적
직전 (i): pert↑로 n10 0.139→0.087, 최종 잔여는 1/p 특이성 아닌 소수 모드 미스매치(상위5=64%). 이번 (ii): 그 정체가 **국소최적**인가.
표현력(P=180) 고정한 채 찾는 능력(restart↑/epoch↑)만 키워 best KL 내려가면 국소최적, 정체+모드 일관이면 아님(→(iii)표현력).

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 구조 (seed/epoch, P 고정, 시간게이트)
- 재사용(무수정): `SBP.make_idb`(Grant V†V), `SBP.adam_step`, `SBP.synth_low`, `SBP.EPS=1e-10`; `KLS.kl_contributions`(지배모드); `OD.kl_div`.
  seed=rng(2000+seed), wA0=uniform, wB0=wA0+N(0,pert). P=180(L6) 고정. **checkpoint 트레이너** 새로 작성(동일 cost/Adam, ckpt에서 KL 기록).
- 축1 restart(seed 다수 @400ep), 축2 epoch(대표 seed @1200ep ckpt{400,800,1200}), 축3 지배모드 seed 겹침(Jaccard).

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: 재사용 함수·seed/epoch·P=180. 직전 pert0.5 best 0.087(2seed 400ep).
- 추정: 국소최적이면 restart↑로 best↓·모드 seed마다 다름; 구조적이면 정체·모드 일관.
- 확인 불가: restart/epoch로도 정체+모드 일관이면 국소최적 아님이나 "표현력 경계"는 다음 (iii).

## 5. 설계 / 시간게이트
S(축1 seed)·축2 epoch는 `--time-only`(1seed 400/800ep 시간)로 확정. 목표 총 ~2.5~3h.

## 6. 시간게이트 + 스코프
`--time-only`: n10 pert0.5 400ep 1seed = **564s(9.4분)**. → 축1 S=12(~113분), 축2 2seed×1200ep(~56분), 총 ~2.8h. 확정 스코프대로 실행.

## 7. 축1 restart↑ (12 seed @ 400ep, pert0.5, P=180)
KL(seed0..11): 0.0867, 0.494, 0.138, 0.137, 0.150, 0.114, 0.136, 0.103, 0.247, 0.189, 0.203, 0.168.
- **best=0.0867 / median=0.144 / max=0.494.** best는 직전(2seed)과 **동일한 0.0867**(스크립트 "내려감"은 0.0867<0.0869 근소차 판정일 뿐, 실질 개선 없음).
- → **restart를 6배(2→12) 늘려도 0.087 아래로 안 내려감** = "운 좋은 골짜기 하나 찾으면 되는" 국소최적 시나리오 아님.

## 8. 축2 epoch↑ 수렴곡선 (2 seed @ 400/800/1200ep)
| seed | 400 | 800 | 1200 |
|---|---|---|---|
| 0 (좋은 seed) | 0.0867 | 0.0748 | **0.0683** |
| 1 (나쁜 seed) | 0.4938 | 0.4821 | 0.4807 |
- 좋은 seed: 3배 epoch에 0.087→0.068(**21%↓, 점감**) — 계속 내려가나 매우 느리고 바닥(1e-3)엔 여전히 ~68배 멈. 나쁜 seed: **거의 평탄**(0.49 갇힘).
- → epoch↑도 바닥 근처로 못 감. 좋은 seed의 완만한 하락은 있으나 "덜 돌려서"만은 아님(점감).

## 9. 축3 지배모드 seed 일관성 (상위5 KL기여, 12 seed)
- 평균 Jaccard(상위5 겹침율) = **0.401**. **반복 상태(등장 seed수)**: s958(10/12), s808(9), s1018(9), s517(8), s1(5).
- → **같은 소수 상태(958,808,1018,517)가 seed 무관하게 반복적으로 못 맞춰짐** = **구조적 미스매치**(국소최적이면 seed마다 다른 상태여야 함).

## 10. self-check
- pert0.5 2seed 400ep best = **0.0867 → 직전 0.087 재현** ✓. NaN/발산 없음.

## 11. 사실 관찰 (판정 아님, 단정 금지)
- **(ii) 국소최적 = 대체로 기각(약함)**:
  - 축1: restart↑(12개)로 best 개선 없음(0.087 정체) → 골짜기 운 문제 아님.
  - 축3: 지배 미스매치 모드가 **seed 간 일관**(958/808/1018/517 반복, Jaccard 0.40) → 구조적, 국소최적 아님.
  - 단 **seed 분산(0.087~0.49)은 국소최적 성격**(나쁜 seed 0.48 갇힘) — 즉 국소최적은 "분산"을 만들지만 "최선 달성치(0.068~0.087)"는 못 낮춤.
- **종합**: 잔여 정체의 *바닥*(best~0.068-0.087)은 restart/epoch로 안 내려가고 특정 모드가 구조적으로 안 맞음 →
  **국소최적보다 (iii) 표현력 경계(이 P=180·이 ansatz가 그 소수 모드를 못 표현)** 를 강하게 시사. (국소최적은 seed 분산엔 기여.)
- ★ 이 ansatz(StronglyEntangling)·완화법(identity-block)·**KL·P=180 한정**. loss/P 무변경·우위 언급 없음. 단정 금지.

## 12. 생성 파일 / 실행 / tmux
신규: `tools/exp_local_minima.py`, `results2/local_minima/{local_minima.json,png,run.log,time.log,smoke.log,local_minima_smoke.json}`, 본 로그.
```
python -m py_compile tools/exp_local_minima.py            # OK
python tools/exp_local_minima.py --time-only             # 400ep 1seed=564s
python tools/exp_local_minima.py --smoke                 # 파이프라인 검증
tmux new -s hoc_lm -d 'python3 tools/exp_local_minima.py --S 12 --ep-axis1 400 --ep-axis2 1200 --seeds-axis2 2 ...'
```
- **tmux 세션 hoc_lm**, 로그 `results2/local_minima/run.log`. 축1 seed당 ~9.5분(12개), 축2 2seed×1200ep, 총 ~2.5h. 자동 종료.
- 기존 코드 무수정(SBP.make_idb/adam_step/synth_low, KLS.kl_contributions, OD import만). **loss=KL·P=180 고정**. 합성만. 무설치. 실패/NaN 없음.

## 13. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`. 기존 무수정.

## 14. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] seed/epoch·P=180 고정·지배모드 계산·시간게이트(564s) 기록.
- [x] 축1 restart 분포(12 seed, best 0.087 정체=미개선).
- [x] 축2 epoch 곡선(좋은seed 점감 0.068, 나쁜seed 평탄).
- [x] 축3 지배모드 seed 일관(Jaccard 0.40, 958/808/1018/517 반복).
- [x] self-check(0.087 재현).
- [x] (ii) 판정: **국소최적 기각(약함)** — best는 안 내려가고 모드 구조적; 분산엔 기여. → (iii)표현력 시사. 해석 한정·무변경·합성·무설치.

**남은 불확실성**
- 좋은 seed는 epoch↑에 아직 완만히 하락 → "완전 수렴"인지 "극도로 느림"인지 미분리(더 긴 학습 필요하나 점감).
- best 바닥(0.068~0.087)이 (iii)표현력 경계인지 확정은 다음 단계(P↑로 그 모드 잡히나).

**다음 추천 단계**
1. **(iii) 표현력 진단**: P↑(L↑, 예 L=8/10)로 재학습 → 반복 미스매치 모드(958/808/1018/517)가 잡히고 best KL이 바닥에 근접하나. 잡히면 (iii)표현력 확정, 안 잡히면 더 근본적(인코딩/분포·회로 표현 한계).
2. 그 반복 모드(958/808/1018/517)의 비트/구조 분석 — 어떤 feature 조합이라 이 회로가 못 맞추나(고차/특정 상관?).
3. (해석 입력) 웹 AI에: "잔여 바닥은 restart/epoch로 안 내려가고 특정 소수 모드가 seed 무관 반복 미스매치 → 국소최적 아닌 (iii)표현력 경계 시사. 이 ansatz·완화법·KL·P=180 한정."
