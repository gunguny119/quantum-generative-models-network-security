# CF² 논문용 4개 값 추출 (조사 전용, 코드/결과 무수정)

- 작성: 2026-07-05 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 파일 수정/재실행/재계산 없음. 본 .md 1개만 신규 생성.
- 목적: 이미 산출·저장된 값에서 논문 플레이스홀더 4개를 **실제 파일 근거와 함께** 추출. 추측·보간 금지.

## 시작 git status
`git -C /home/elicer/IOTJ status --short` = **깨끗**(변경 없음; 직전 커밋 dca4ddb push 완료 상태). 본 조사로 인한 변경 없음(본 로그 파일만 추가).

## 실제로 연 파일
- `results2/gate_e/gate_e.json` (값 ①·④)
- `results2/gap_cf/gap_cf.json` (값 ②)
- `tools/gate_e_coefficient.py` (값 ③: chsh_setup/kcbs_setup/mix), `tools/kink_origin_v1.py`(kcbs_extremal), `tools/gap_vs_cf_v0.py`(box 정의)
- results2 구조: `gap_cf/{gap_cf.json,gate0.json,run_full.log}`, `gate_e/gate_e.json`, `kink/{kink.json,run.log}`, `gate_r1..r3/*.json` 등.

---

## [값 ①] CF² fit 의 결정계수 R²  — **있음**
- 출처: `results2/gate_e/gate_e.json` → `results.<시나리오>.<방향>.r2_fit` (방향별 저장). 코드: `tools/gate_e_coefficient.py` `process_direction`(원점통과 `c·CF²` 적합).
- **CF² 레짐(매끄러운 facet) 4방향 R² ≈ 1.000**:

| 방향 | r2_fit (출처: gate_e.json) |
|---|---|
| CHSH/uniform | **0.9999744** |
| CHSH/det_vertex_0101 | **0.9999882** |
| KCBS5/uniform | **0.9999684** |
| KCBS5/det_vertex_all0 | **0.9999697** |
| CHSH/det_vertex_0000 (선형레짐) | 0.7839172 |
| CHSH/local_perfect_corr (선형레짐) | 0.7915730 |

- ※ `gap_cf.json` 의 곡선형태 판정(`verdicts.P3_shape`)은 **AIC 기반**이라 R² 없음(그쪽엔 R² 미저장). "CF² fit R²"는 **gate_e 의 r2_fit** 가 해당 값.

## [값 ②] HMM(hidden-state 잠재사다리) H별 KL + NCfloor  — **있음**
- 출처: `results2/gap_cf/gap_cf.json` → `sweeps.inf[i].per_model["H{h}"]["mean"]`(각 H의 held-out KL 평균, seed 5), NCfloor=`sweeps.inf[i]["kl_nc_floor"]`, 양자=`kl_quantum`.
- ★ 실제 사다리는 **H∈{1,2,3,4,6,8}** (코드 `gap_vs_cf_v0.py` `H_LADDER=[1,2,3,4,6,8]`) — **H5·H7 은 미실행**(파라미터 5H−1). 추측 금지 원칙상 H5/H7 은 "없음".
- N=∞(모집단) 스윕 값 (KL, nats):

| CF | λ | H1 | H2 | H3 | H4 | H6 | H8 | **NCfloor** | Quantum |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.500 | 0.1308 | 0.0654 | 0.0163 | 0.0040 | 0.0014 | 0.0000 | 0.0000 | 0.0000 |
| 0.050 | 0.525 | 0.1450 | 0.0725 | 0.0157 | 0.0049 | 0.0004 | 0.0004 | 0.0004 | 0.0000 |
| 0.100 | 0.550 | 0.1600 | 0.0800 | 0.0282 | 0.0075 | 0.0026 | 0.0017 | 0.0017 | 0.0000 |
| 0.150 | 0.575 | 0.1759 | 0.0879 | 0.0192 | 0.0154 | 0.0084 | 0.0051 | 0.0039 | 0.0000 |
| 0.200 | 0.600 | 0.1927 | 0.0964 | 0.0367 | 0.0363 | 0.0113 | 0.0070 | 0.0070 | 0.0000 |
| 0.400 | 0.700 | 0.2704 | 0.1352 | 0.0702 | 0.0502 | 0.0358 | 0.0298 | 0.0298 | 0.0000 |
| **0.600** | **0.800** | **0.3681** | **0.1840** | **0.1129** | **0.1079** | **0.0823** | **0.0777** | **0.0725** | **0.0095** |
| 0.800 | 0.900 | 0.4946 | 0.2473 | 0.1689 | 0.1759 | 0.1483 | 0.1509 | 0.1441 | 0.0480 |
| 1.000 | 1.000 | 0.6931 | 0.3466 | 0.3144 | 0.2969 | 0.2954 | 0.2892 | 0.2877 | 0.1583 |

- (요청된 **CF=0.6 지점 포함**. λ=0.0~0.5의 CF=0 점들은 H8까지 KL→0=암기가능 확인, 대표로 λ=0.5만 표기.)
- 유한표본판(N=5000)은 `sweeps["5000"]` 에 별도 저장(위는 N=∞).

## [값 ③] 6개 mixing 방향의 수학적 정의 (코드 위치)
공통 혼합: `e(λ)=λ·e_ext + (1−λ)·e_dir` (`gate_e_coefficient.py:122` `mix`). λ=1=극점(맥락), λ=0=방향(비맥락 endpoint). CHSH 컨텍스트 순서 `CTX=[(0,0),(0,1),(1,0),(1,1)]`.

**CHSH 시나리오** (`gate_e_coefficient.py:91-101` `chsh_setup`; e_ext=`G.pr_box()` = PR상자, `gap_vs_cf_v0.py:75-81`, e[ci,a,b]=0.5·[a⊕b=x∧y]):
| 방향 | 정의 (분포/좌표) | 코드 위치 |
|---|---|---|
| **uniform** | 균등: e[ci,a,b]=0.25 전부 | `chsh_setup`:95 → `G.uniform_box()` `gap_vs_cf_v0.py:84` |
| **det_vertex_0000** | 국소 결정론 꼭짓점 g=(A0,A1,B0,B1)=(0,0,0,0): 모든 컨텍스트서 (a,b)=(0,0)에 질량1 | `chsh_setup`:96 → `Vt[0]` (`G._nc_vertices` `gap_vs_cf_v0.py:257-268`, D[ci,g[x],g[2+y]]=1) |
| **det_vertex_0101** | 꼭짓점 g=(0,1,0,1) → A_x=x, B_y=y: 컨텍스트(x,y)서 (a,b)=(x,y)에 질량1 | `chsh_setup`:97 → `Vt[int("0101",2)]=Vt[5]` |
| **local_perfect_corr** | 완전상관(설정무관 a=b): e[ci,0,0]=e[ci,1,1]=0.5 | `chsh_setup`:98 → `G.perfect_corr_box()` `gap_vs_cf_v0.py:88-94` |

**KCBS5 시나리오** (`gate_e_coefficient.py:104-119` `kcbs_setup`, n=5; e_ext=`K.kcbs_extremal(5)` = 반상관 5-cycle, `kink_origin_v1.py:249-253`, e[i,0,1]=e[i,1,0]=0.5):
| 방향 | 정의 | 코드 위치 |
|---|---|---|
| **uniform** | 균등: e[i,a,b]=0.25 (5 컨텍스트) | `kcbs_setup`:113 `unif=np.full((n,2,2),0.25)` |
| **det_vertex_all0** | 결정론 v=(0,0,…,0): 각 컨텍스트 i 서 (0,0)에 질량1 | `kcbs_setup`:114-116 `dv[i,0,0]=1` |

→ 논문의 "6개 방향" = CHSH 4 + KCBS 2. 선형레짐 2방향 = CHSH/det_vertex_0000, CHSH/local_perfect_corr.

## [값 ④] Gate E 4방향 대조 (배경 "확실한 사실 3" vs 파일)
출처: `results2/gate_e/gate_e.json` (`c_pred_limit`=CF→0 극한 ½χ², `c_fit`=적합, `alpha`, `rel_err_pred_vs_fit`). **전부 일치**.

| 방향 | 배경값(c_pred / c_fit / 오차) | 파일값 c_pred_limit / c_fit / α / rel_err | 일치? |
|---|---|---|---|
| CHSH/uniform | 1/6 / ≈0.170 / 1.9% | 0.166673 / 0.169949 / 2.0057 / 0.01927 | **일치** (1/6=0.16667) |
| CHSH/det_vertex_0101 | 0.199 / 0.202 / 1.3% | 0.199214 / 0.201912 / 2.0033 / 0.01337 | **일치** |
| KCBS/uniform | 1/8 / ≈0.128 / 2.2% | 0.125003 / 0.127762 / 2.0066 / 0.02160 | **일치** (1/8=0.125) |
| KCBS/det_vertex_all0 | 0.125 / 0.128 / 2.1% | 0.125001 / 0.127720 / 2.0079 / 0.02129 | **일치** |
| (선형) CHSH/det_vertex_0000 | α≈1 | α=1.0021, regime=linear(χ² invalid), r2=0.784 | **일치**(α≈1) |
| (선형) CHSH/local_perfect_corr | α≈1 | α=1.0065, regime=linear, r2=0.792 | **일치**(α≈1) |

## 확인 불가한 값과 이유
- **HMM H5·H7 KL**: 실행 안 됨(`H_LADDER=[1,2,3,4,6,8]`) → 저장 없음. 추측 금지로 "없음".
- **gap_cf 곡선형태 R²**: `gap_cf.json` 은 AIC만 저장(R² 없음). CF² fit R²는 gate_e 값(값 ①)으로 대체.
- 그 외 4개 값은 모두 파일에서 확인됨(확인 불가 없음).

## 실행한 명령 (읽기/확인 수준만)
- `git status`(깨끗) / `find results2 -type f`(구조) / `python3 -c "json.load..."`(gate_e.json·gap_cf.json key·값 읽기) / `grep -n`(gate_e_coefficient.py 방향 정의) / `Read`(gate_e_coefficient.py:104-131). **값 변경/재계산 명령 없음.**

## 다음 추천 단계
1. 논문 TODO 4곳에 위 값 삽입: ① R²(CF²레짐 0.99997~0.99999), ② HMM 표(CF=0.6: H1 0.368 … H8 0.078, NCfloor 0.073, Q 0.0095), ③ 6방향 정의, ④ 대조표(전부 일치).
2. (선택) 유한표본판 N=5000 HMM 값이 본문에 필요하면 `gap_cf.json sweeps["5000"]` 에서 동일 방식 추출.
3. H5/H7 이 논문에 필요하면 별도 실험 필요(이번 조사 범위 밖, 재계산 금지 준수).
