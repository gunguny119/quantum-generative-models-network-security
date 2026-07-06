# QCBM-KL 학습 포화 확인 — epoch/restart/depth↑로 저차 KL 바닥 (유한학습 vs 표현력 포화)

- 작성: 2026-06-29 08:00 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 합성만(실데이터/체크포인트 미사용). 한 점 집중. 기존 코드 무수정(import/재사용). 단일 프로세스. 판정은 웹 AI.

## 1. 작업 목적
직전 QCBM-KL 저차 KL=0.197(MLP 0.001, NMF 0.005 대비 큼). 이게 (a)유한 학습(250ep/3restart) 탓인지 (b)이 ansatz
(StronglyEntangling)·예산(P=54)·옵티마이저 표현력 포화인지 미분리. 학습량(epoch·restart, +참고 depth)을 키워
저차 KL 포화 바닥을 측정. **단정 금지**: "이 ansatz·예산·옵티마이저 포화점"으로만 한정.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip` 전부 untracked. 보존: 기존 모든 파일(import/재사용).

## 3. 확인한 학습 루프 구조
- `tools/qcbm_kl_variant.py:37 train_kl_one(q_target,nq,layers,epochs,lr,seed,log_every,eps)` → **(p_final, final_CE, loss_hist)**.
  loss_hist=epoch별 cross-entropy → KL곡선=CE−H(q). epoch/restart/L 모두 인자. (포화·곡선 측정 충분.)
- 재사용: SY.make_terms/synth_dist/ge3_share(동일 분포), E.make_circuit/wshape/empirical_dist, OD.kl_div.
- 기준선(직전): MLP 0.001, NMF 0.005, QCBM-KL(250,3)=0.197. 샘플링 KL 바닥 ≈ #states/(2N)=64/160000≈**4e-4**.

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: train_kl_one 곡선 반환, 학습량 인자화. 평가=KL(p_true‖model)(직전 지표).
- 추정: epoch/restart↑로 KL 더 내려갈 수 있음(유한학습). 포화 후 MLP보다 높으면 이 설정 표현력.
- 확인 불가: 다른 ansatz/큰 n은 별도(이 작업 범위 아님).

## 5. 설계
저차 t=0 고정(make_terms seed0 disjoint, N=80000, 샘플 seed12345 → q_emp; p_true=synth_dist(t=0)).
ladder(P=54, L=3): (250,3)→(750,5)→(1500,10), early-stop(개선<5%면 중단). 참고 L=6(P=108) 1점. 고랭크 random t=0.6 1점.
평가 KL(p_true‖model), 곡선 KL(q_emp‖model)=CE−H(q_emp). self-check: 학습량↑→KL 비증가/포화.

## 6. 저차(t=0) 포화 ladder (P=54, L=3) — KL(p_true‖model)
| epoch | restart | params | best KL | restart 분포(min~max) | train_KL | 개선 |
|---|---|---|---|---|---|---|
| 250 | 3 | 54 | **0.1967** | 0.1967~0.2174 | 0.1966 | — (직전 0.197 재현) |
| 750 | 5 | 54 | **0.1946** | 0.1946~0.2172 | 0.1946 | **1.0%** → 포화, ladder 중단 |
- restart 분포가 모두 0.19~0.22로 좁음 → 국소최적 산포가 아니라 **진짜 바닥**. ep 250→750·restart 3→5에 1%만 변화.

## 7. 포화점 / 포화 KL vs 기준
- **포화점**: ep=750, restart=5 (그 이상은 <5% 변화로 중단).
- **포화 KL = 0.1946.** 샘플링 바닥(p_true‖q_emp) = **0.0009**, MLP = 0.001, NMF = 0.005.
- 즉 L=3/P=54 포화 KL(0.195)은 샘플바닥/MLP보다 **~200배 높은 곳에서 멈춤** → 이 예산·깊이의 **표현력 포화**(유한학습 아님).

## 8. 참고 L=6 (P=108, 예산↑)
- L=6, ep=1000, restart=4 → best KL = **0.0002** (샘플바닥 0.0009·MLP 0.001 이하, 사실상 완전 적합).
- 깊이/예산을 2배로 키우면 QCBM-KL이 **바닥(고전 수준)에 도달** → L=3 포화는 "회로 자체 불능"이 아니라 **이 예산·깊이의 한계**.

## 9. 고랭크 random t=0.6
- L=3, ep=1200, restart=8 → KL = **0.2772** (share 0.178, rank99=4). smoke(150ep/2restart) 0.500 → 학습 늘려 0.28로 내려갔으나
  여전히 MLP(0.003)보다 큼. L=3에선 고랭크도 표현력/학습 제약(완전 포화 미도달, 더 학습/깊이 여지).

## 10. self-check
- ladder best KL **비증가 = True**, 학습 곡선 **감소 = True**. 학습 불안정/발산 없음.
- 직전 결과 재현: (250,3)에서 0.1967 = 직전 QCBM-KL 0.197 일치 ✓.

## 11. 사실 관찰 (판정 아님, 단정 금지)
- **저차 KL 0.197은 유한학습이 아님**: L=3/P=54에서 학습량을 키워도 0.195에서 포화(restart 분포도 좁음).
  → "이 ansatz(StronglyEntangling)·이 예산(P=54)·이 옵티마이저(Adam)"의 표현력 포화로 한정.
- **그 포화는 예산/깊이를 늘리면 해소**: L=6(P=108)에서 KL 0.0002(바닥) → QCBM이 분포를 못 배우는 게 아니라 **P=54가 부족**했던 것.
- 고랭크(random t=0.6)는 L=3에서 0.277로 더 큼(저차보다 어려움), 완전 포화 미도달.
- ★ "QCBM 전체 한계" 아님. 동일예산(P=54) 비교에선 고전 MLP가 우세했으나, 그건 **예산당 표현 효율** 차이이지 양자가 분포를
  학습 불가라는 뜻이 아님(L=6에서 도달). 양자 우위(어떤 고전도 못 하는 영역) 여부는 별개(미입증).

## 12. 생성 파일 / 실행 / tmux
신규: `tools/exp_qcbm_kl_saturation.py`, `results2/qcbm_kl_saturation/{saturation.json,saturation.png,run.log}`, 본 로그.
```
python -m py_compile tools/exp_qcbm_kl_saturation.py     # OK
python tools/exp_qcbm_kl_saturation.py --smoke           # 동작 확인(L=3 0.198 포화, L=6 0.0019)
tmux new -s hoc_sat -d 'python3 tools/exp_qcbm_kl_saturation.py 2>&1 | tee results2/qcbm_kl_saturation/run.log'  # 본 실행
```
- **tmux 세션 hoc_sat**, 로그 `results2/qcbm_kl_saturation/run.log`. 단일 프로세스, ladder early-stop. 총 ~70분. 자동 종료.
- experiment2/qcbm_kl_variant 무수정(import만). 합성만(실데이터 미사용). dependency 무설치. 실패 없음.

## 13. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규는 그 안). 기존 무수정.

## 14. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] qcbm_kl_variant 학습 루프 구조(epoch/restart/L 인자, loss_hist 곡선) 확인·기록.
- [x] 저차 학습량별 best KL + 곡선(json/png). 포화점(750,5)·포화 KL 0.195 식별.
- [x] 포화 KL vs MLP(0.001)/NMF(0.005)/샘플바닥(0.0009) 비교.
- [x] 참고 L=6(P=108) 결과 0.0002. 고랭크 random 0.277.
- [x] 해석 "이 ansatz·예산·옵티마이저 포화"로 한정(QCBM 전체 단정 없음). 기존 무변경·합성만·무설치.

**남은 불확실성**
- 고랭크(random t=0.6)는 L=3에서 미포화(0.277) — 더 학습/깊이면 더 내려갈지 미측정.
- 포화는 단일 분포(t=0)·단일 seed 기준. L=4,5 중간 깊이의 포화 곡선(예산-표현력 관계)은 미측정.
- L=6 효과는 예산 2배라 "동일예산 양자우위"와는 별개(참고용).

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "저차 0.197은 유한학습 아님(L=3/P=54 표현력 포화); L=6/P=108은 바닥(0.0002) 도달.
   즉 동일예산 고전우위는 '예산당 표현효율' 문제이지 QCBM 학습불능 아님." 양자 정당화 정리.
2. L=4,5,6,8 깊이별 저차 포화 KL 곡선 → "예산(깊이) vs 표현력" 관계 정량화(고전 MLP 동급 도달 깊이 찾기).
3. 고랭크/고차 분포에서 깊이↑ QCBM이 NMF뿐 아니라 MLP까지 넘는 영역이 생기는지(동일예산) 재탐색.
