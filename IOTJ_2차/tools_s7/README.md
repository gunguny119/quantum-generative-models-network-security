# tools_s7 — Gate S7 신규 코드

`results2/gate_s7/*.json` 을 생산한 신규 코드다. **`tools/` 는 무수정**이며(원칙 유지),
이 디렉터리는 별도 위치에 추가한 것이지 기존 파일을 고친 것이 아니다.

## 왜 여기 있는가

원래 이 코드는 스크래치패드(`/tmp/.../scratchpad/`)에서 실행했다. 그런데 세션 재시작 때
**스크래치패드가 비워졌고**, 그 시점에 이미 `s7.json` 등이 `provenance.new_code =
"scratchpad/s7_finite_template.py"` 로 존재하지 않는 경로를 출처로 인용하고 있었다.
논문이 "JSON 만으로 독립 재현 가능"을 주장하려면 적합기 코드가 저장소에 있어야 하므로 옮겼다.

**JSON 안의 `scratchpad/xxx.py` 경로는 이 디렉터리의 같은 이름 파일로 읽으면 된다.**
저장된 결과 JSON 은 사후 편집하지 않았다(저장값 무결성 유지).

## 파일

| 파일 | 역할 | 사용처 |
|---|---|---|
| `s7_finite_template.py` | (2,3,2) 유한표본 템플릿 적합기 — `fit_quantum_params_23` / `fit_nc_params_23` / `sample_emp_23` / `verify_wrappers_23` | Phase 3-0, 1, B, C, F 전부 |
| `s7_phase2b.py` | 원시점수 래퍼 `raw_scores_23`, `marginal_template_23`, `empirical_template_23`, `fast_auc`, `verify_raw` | Phase B, C, F |
| `s7_phaseF.py` | Phase F 드라이버 (교정 예산 창) | Phase F |

### 원본 대응 (로직 복제 시 출처 명시 규칙)

| 신규 | 원본 |
|---|---|
| `sample_emp_23` | `tools/gap_vs_cf_v0.py:193 sample_emp` 의 (2,3,2) 대응 (= `detection_auc_23` 내부 `boxes()` 와 동일 샘플링·Laplace) |
| `_q_inits_23` | `tools/gap_vs_cf_v0.py:160 _q_inits` (9→13 파라, `CANON_23`) |
| `fit_quantum_params_23` | `tools/gap_vs_cf_v0.py:224 fit_quantum` = `tools/gate_s2_detection.py:67 fit_quantum_params` |
| `fit_nc_params_23` | `tools/gate_s2_detection.py:80 fit_nc_params` (원본 `nc_floor_23` 이 이미 emp 에 CE 최소화 적합) |
| `verify_wrappers_23` | `tools/gate_s2_detection.py:85 verify_wrappers` |
| `raw_scores_23` | `tools/gate_s5_multifacet.py:320 detection_auc_23` 의 원시점수 반환판 |
| `marginal_template_23`, `empirical_template_23` | 신규 (원본 없음) |

## 검증

- `raw_scores_23` vs 원본 `detection_auc_23`: N ∈ {245, 2000, 4900} 전부 **max_abs_diff = 0.0**
- 복원 후 재검증: 세 템플릿 훈련오차(seed 0) 가 Phase B 저장값과 **차이 0.00e+00**
- `fast_auc` (동점 평균순위) 와 `s7_phaseF.auc_pair` (벡터화) 최대차 **0.00e+00**

## 실행

```bash
pip install numpy scipy matplotlib     # matplotlib 은 tools/gap_vs_cf_v0.py:44 하드 임포트
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python3 s7_finite_template.py 6 200000        # Phase 3-0 검증 배터리
python3 s7_phase2b.py                         # raw_scores_23 원본 대조
python3 s7_phaseF.py                          # Phase F 재실행
```

**`fastq` 설치는 필수다.** 없이 실행하면 `quantum_behavior_23` 이 ~1e-16 달라지고
그 차이가 Powell 을 거쳐 NCfloor 에서 ~2.4e-12 로 증폭돼 저장값과 비트 일치하지 않는다.
각 모듈의 `install_fast()` / `_fast()` 가 이를 처리한다.

## 아직 복원되지 않은 것

아래 드라이버는 스크래치패드 소실로 **현재 저장소에 없다**. 산출 JSON 은 모두 남아 있으나,
그 JSON 을 처음부터 재생성하려면 재작성이 필요하다.

| 잃어버린 드라이버 | 생산한 산출물 |
|---|---|
| `s7_phase3_sweep.py`, `s7_finalize.py` | `s7.json` (Phase 3: N 스윕, 참값 vs 유한표본 템플릿) |
| `s7_phase1.py` | `s7_phase1.json` (Phase 1: KL_max 전역탐색, τ 스윕, N*(KL)) |
| `s7_phase2b.py` 의 `main()` | `s7_phase2b.json` (Phase B: 네 템플릿 대조) — 라이브러리 함수는 위 파일에 보존 |
| `s7_phase2c.py` | `s7_phase2c.json` (Phase C: CF 스윕) |

**복원 가능하다** — 요청 시 작성한다. 우선순위가 낮다고 판단해 라이브러리 모듈부터 옮겼다.
