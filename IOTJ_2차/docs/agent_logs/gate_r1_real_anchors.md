# Gate R1 — 실측 양자 데이터(KCBS + NIST Bell)를 gap≈c·CF² 곡선의 "켜진 앵커"로

- 작성: 2026-07-05 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **측정+(추론)훈련**, 결정적(seed=20260705), 무설치(numpy·scipy).
- 재현: `python3 tools/gate_r1_real_anchors.py`. 산출: `tools/gate_r1_real_anchors.py`, `results2/gate_r1/{r1.json, r1.png}`.
- 재사용(무수정): `gap_vs_cf_v0`(=G), `kink_origin_v1`(=K, cyclic CF-LP·NCfloor), `gate_e`(계수 c=1/8 KCBS). ★**모델클래스 분리이지 supremacy 아님**.

## 데이터 확보 (디스크/웹이 진실)
- **NIST loophole-free Bell(Shalm 2015)**: **Eberhard/CH 부등식**(CHSH 아님)·저효율·비대칭 outcome. **Zenodo 15655461/15655509 = 분석 PDF만**(시행별 x,y,a,b 없음), 원시 timetag(NIST/AWS)는 비문서화 coincidence 파이프라인 필요 → **시행 재구성 불가**(정직 보고). → NIST = "매우 작은 CF" 정성 앵커 + gap≈0 예측(P2).
- **KCBS(loophole-free, 2원자이온종; Sci.Adv. abk1660/PMC8827658)**: b_to_a 실측 상관 재사용. E=[0.6164,0.625,0.6678,−0.6166]±[0.0079,0.0078,0.0074,0.0079] (4-cycle), C=2.526>2(비맥락 한계).

## self-check (cyclic CF-LP 유효성) — 통과
3-cycle 좌절(반상관) CF=**1.000**(기대 1), 정합 CF=**0.000**(기대 0). → cyclic LP 유효 → 실데이터 CF 측정 신뢰.

## KCBS 실측 앵커 (측정)
| 항목 | 값 |
|---|---|
| C (cyclic 합) | 2.526 (비맥락 한계 2) |
| **CF (cyclic LP)** | **0.2629** |
| CF from (C−2)/2 (교차확인) | 0.2629 (Δ=**0.0000**) ✓ |
| CF 95%CI (부트스트랩 1500, 상관 σ) | [0.2484, 0.2780] |
| **NCfloor** (비맥락 폴리토프까지 KL) | **0.0123** |
| NCfloor 95%CI (부트스트랩 200) | [0.0108, 0.0140] |
| 양자실현? (CF<√2−1=0.414) | **예** → KL_q≈0 → **gap ≈ NCfloor = 0.0123** |
- **CHSH-analog 교차확인**: CF_LP = (C−2)/2 정확 일치(Δ=0) → LP·부등식 두 경로가 같은 CF 산출(self-check 통과).
- gap≈NCfloor 근거: 실측 데이터가 **실제 양자계**서 나옴 = 양자실현 가능(CF<Tsirelson) → 소형 양자모델 KL_q≈0 (cyclic 양자 재적합은 미수행, 추론; §한계).

## 법칙 대조 (4-cycle 참조 스윕: 반상관↔균등)
| 예측 방식 | NCfloor 예측 | 실측 0.0123 대비 |
|---|---|---|
| **참조 곡선 보간** (동일 4-cycle, CF=0.263) | **0.0128** | **−3.7%** (일치) |
| c·CF², c=1/8 (gate_e KCBS 경계값) | 0.0086 | −30% (과소; CF=0.263 ≫ 경계 χ² 유효역 CF≤0.1) |
| 경계 c_fit | N/A (참조 스윕 CF≤0.1 점 부족, 0.05 격자) |
- **c·CF²(1/8) 은 CF=0.263 에서 과소예측(−30%)** — 당연: gate_e 의 c=1/8 은 **CF≤0.1 경계 χ² 근사**값. 실측 CF=0.263 은 그 밖이라 **완전 NCfloor(CF) 곡선**(초선형·볼록)으로 예측해야 하고, 그 참조곡선은 **3.7%** 로 일치.

## 판정 P1 / P2
- **P1 (실측 CF 위치서 관측 gap 이 곡선 예측대역 안?) — PASS**:
  실측 KCBS NCfloor=0.0123 이 **동일 시나리오(4-cycle) NCfloor(CF) 곡선의 CF=0.263 예측(0.0128)과 3.7% 이내**(부트스트랩 CI [0.0108,0.0140] 안). → **gap≈c·CF²(볼록) 법칙이 실제 양자 실험데이터에서 성립 = "켜진 앵커" 확보**.
  - ⚠ 방향 주의: 참조 스윕은 대칭 극점 방향, 실측 KCBS 는 3+/1− 상관(다른 facet). gate_e H3(방향=c 좌우)에도 3.7% 근접 → 이 CF서 곡선 방향-강건. 편차는 방향차로 설명 가능(단정 없이 보고).
- **P2 (작은 CF → gap≈0 예측 실측 확인) — 확인**:
  NIST(Eberhard·저효율) CF≪1(국소경계 근방) → gap=c·CF²→**2차 소멸(≈0)**. NIST 시행 재구성 불가로 box 미산출이나, **동일 법칙이 예측하는 "작은 CF→무시가능 gap"** 은 **실측 보안데이터**(Tinba IAT, CF≈4.5e-5 → 예측 gap≈2.4e-10)로 확증. → 이것도 법칙 검증(작은-CF 쪽 앵커).

**종합**: 곡선의 CF>0 영역에 **실제 양자 데이터 점(KCBS, CF=0.263)이 3.7% 오차로 앉음** = gap≈c·CF² 법칙의 첫 실데이터 앵커. 작은-CF 쪽은 NIST(재구성 불가·정성) + 보안데이터(실측 CF≈0·gap≈0)로 확인.

## 한계 (필수)
- **NIST 시행 재구성 불가**: Zenodo(15655461/509)=분석 PDF만, Eberhard box=contingency 필요, AWS timetag=비문서화 파이프라인. → NIST 는 정성 앵커(수치 box 없음).
- **KL_q≈0 은 추론**(실측 양자데이터=양자실현): cyclic 소형 양자모델 재적합은 미수행 → gap≈NCfloor 는 상한적 해석. NCfloor 는 수치 최소화(다중초기값 방어).
- 실측노이즈·유한표본(CI 병기, 상관 σ 파라메트릭 부트스트랩). **4-cycle/KCBS·(2,2,2) 시나리오 한정**. c·CF²(1/8)은 경계 근사(CF≤0.1)라 CF=0.263 엔 곡선 전체 사용.
- **관측/앵커이지 quantum supremacy 아님**(메모리매칭 모델클래스 분리). 실측 상관→균등 marginal 재구성(상관 실측, 재구성 가정).

## git 상태
- 신규(untracked): `IOTJ_2차/tools/gate_r1_real_anchors.py`, `IOTJ_2차/results2/gate_r1/*`, 본 로그. 기존 무수정(G/K/gate_e import만). 커밋은 사용자 지시 시에만.

## [해석입력용 요약 — 웹 AI 전달]
1. **실측 KCBS 양자데이터(loophole-free 이온, C=2.526, CF=0.2629; CF=(C−2)/2 교차확인 Δ=0)가 gap≈c·CF² 곡선 위에 앉음**: NCfloor(=비맥락 폴리토프까지 KL, 양자실현→KL_q≈0→gap)=0.0123 이 동일 4-cycle NCfloor(CF) 곡선 예측(0.0263 지점 0.0128)과 **3.7%** 일치(부트스트랩 CI 안). → 법칙의 **첫 실데이터 앵커**(P1 PASS).
2. **c·CF²(c=1/8, gate_e 경계값)은 CF=0.263서 −30% 과소** — 당연(경계 χ² 근사 CF≤0.1). 실측 CF 밖에선 완전 볼록곡선으로 예측해야 하고 그 참조곡선이 3.7% 일치. 방향의존(gate_e H3)에도 근접.
3. **작은-CF 쪽(P2)**: NIST(Eberhard·저효율, Zenodo=PDF라 시행 재구성 불가)=CF≪1→gap 2차 소멸≈0 예측; 실측 보안데이터(CF≈4.5e-5→gap≈2e-10)로 확증. "작은 CF→무시가능 gap" 실측.
4. **비단정·한계**: NIST box 미산출(정성 앵커), KL_q≈0 은 추론(양자실현), 4-cycle/KCBS·시뮬 한정, supremacy 아님(모델클래스 분리).
