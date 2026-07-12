# CF² 논문 방향 구조 3개 추출 — 6방향 정의·선형행·방향별 gap (조사 전용, 무수정)

- 작성: 2026-07-08 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 무수정·무재계산. 본 .md 1개만 신규.
- 목적: 논문 Appendix B/C·Table I 채울 (A) 6방향 정의, (B) 선형 2방향 귀속·계수, (C) 방향별 gap.

## 시작 git status
`git status --short` 변경항목 7(최근 R3/R4/R5 untracked; 본 조사 무관). 본 조사로 파일 변경 없음(로그 1개만).

## 확인한 파일
`tools/gate_e_coefficient.py`(chsh_setup:91–101, kcbs_setup:104–119, mix:122), `tools/gap_vs_cf_v0.py`(pr_box:75, uniform_box:84, perfect_corr_box:88, _nc_vertices:257–268), `tools/kink_origin_v1.py`(kcbs_extremal:249–253), `results2/gate_e/gate_e.json`, `results2/kink/kink.json`.

---

## [값 A] 6방향 수학적 정의
공통 혼합: **$e(\lambda)=\lambda\,e_{\text{ext}}+(1-\lambda)\,e_{\text{dir}}$** (`gate_e_coefficient.py:122–123` `mix`). $\lambda{=}1$=극점(맥락), $\lambda{=}0$=방향(비맥락). CHSH 컨텍스트 순서 `CTX=[(0,0),(0,1),(1,0),(1,1)]`, 결과 $(a,b)\in\{0,1\}^2$.

### CHSH 시나리오 (`chsh_setup:91–101`) — $e_{\text{ext}}=$ **PR-box** ✓ (`gap_vs_cf_v0.py:75–81`, $e[ci,a,b]=0.5$ if $a\oplus b=x\wedge y$ else $0$)
| 방향 | $e_{\text{dir}}$ 정의 (셀별 확률) | 코드 |
|---|---|---|
| **uniform** | $e[ci,a,b]=0.25$ (전 컨텍스트·전 결과 균등) | `:95` → `G.uniform_box()` (`:84–85`) |
| **det_vertex_0000** | 국소 결정론 꼭짓점 $g=(A_0,A_1,B_0,B_1)=(0,0,0,0)$: **모든 컨텍스트서 $(a,b)=(0,0)$에 질량 1** | `:96` → `Vt[0]` (`_nc_vertices:257–268`, $D[ci,g[x],g[2{+}y]]=1$) |
| **det_vertex_0101** | 꼭짓점 $g=(A_0,A_1,B_0,B_1)=(0,1,0,1)$ ⇒ $A_x=x,\ B_y=y$: **컨텍스트 $(x,y)$서 $(a,b)=(x,y)$에 질량 1** → ctx(0,0)→(0,0), (0,1)→(0,1), (1,0)→(1,0), (1,1)→(1,1) | `:97` → `Vt[int("0101",2)]=Vt[5]` (인덱스 5 = 튜플 (0,1,0,1), 코드로 확정) |
| **local_perfect_corr** | 완전상관(설정 무관 $a=b$): $e[ci,0,0]=e[ci,1,1]=0.5$ | `:98` → `G.perfect_corr_box()` (`:88–94`) |

### KCBS5 시나리오 (`kcbs_setup:104–119`, n=5) — $e_{\text{ext}}=$ **반상관 5-cycle** ✓ (`kink_origin_v1.py:249–253` `kcbs_extremal`, $e[i,0,1]=e[i,1,0]=0.5$)
| 방향 | $e_{\text{dir}}$ 정의 | 코드 |
|---|---|---|
| **uniform** | $e[i,a,b]=0.25$ (5 컨텍스트 균등) | `:113` `unif=np.full((n,2,2),0.25)` |
| **det_vertex_all0** | 결정론 $v=(0,0,\dots,0)$: **각 컨텍스트 $i$서 $(0,0)$에 질량 1** | `:114–116` `dv[i,0,0]=1` |

★ **det_vertex_0101 좌표 코드 확정**: `int("0101",2)=5`, `itertools.product((0,1),repeat=4)[5]=(0,1,0,1)` → $(A_0{=}0,A_1{=}1,B_0{=}0,B_1{=}1)$. 상상 아님(파이썬 인덱스로 확인).

## [값 B] 선형 레짐 2방향 — 둘 다 **CHSH scenario**
출처 `gate_e.json → results.CHSH.<dir>` (둘 다 `chsh_setup`의 directions, 즉 **CHSH 소속** 확정). `verdict.boundary_anchored.directions=["CHSH/det_vertex_0000","CHSH/local_perfect_corr"]`.
| 방향 | scenario | alpha | c_fit | c_pred_limit | qmin_ref | r2_fit | regime |
|---|---|---|---|---|---|---|---|
| **det_vertex_0000** | **CHSH** | **1.0021** | 2.4826 | 48.76(발산) | 2.62e-37 | 0.7839 | linear(α≈1, χ² invalid) |
| **local_perfect_corr** | **CHSH** | **1.0065** | 2.1582 | 48.80(발산) | 3.68e-34 | 0.7916 | linear(α≈1, χ² invalid) |
- **alpha 매칭 확정**: 1.0021 = det_vertex_0000, 1.0065 = local_perfect_corr. (선형 median 1.0043)
- **계수**: `c_fit` 저장돼 있음(2.48 / 2.16)이나 **c·CF² 강제적합이라 R²≈0.78로 부적합**(선형이라 CF² 계수 의미 없음). `c_pred_limit`(½χ²)은 **발산(48.8)** → 선형레짐서 χ² 무효.
- **왜 vertex 접근인가**(코드/기하): `qmin_ref` = **2.6e-37 / 3.7e-34 ≈ 0** — 즉 비맥락 기준상자가 폴리토프 **꼭짓점/모서리에 정착**해 혼합 즉시 맥락적($\lambda_0\approx0$), KL 투영 $q^*$가 $e$-support 셀에서 **0**이 되어 $\chi^2=\sum\Delta^2/q$ **발산** → 경계지수 $\alpha{=}1$(선형). (`gate_e.json verdict.boundary_anchored.reading`.)

## [값 C] 방향별 gap ("같은 CF서 최대 ~2배 차이" 근거)
출처 `results2/kink/kink.json → gateB`. Gate B = PR-box를 4방향과 혼합해 NCfloor(CF) 수집(경계서 $KL_q\approx0$이라 gap=NCfloor). 방향: uniform, det_vertex_0000, det_vertex_0101, local_perfect_corr.

**핵심 지표**: `gateB.collapse.max_rel_spread = 1.7866` ((max−min)/mean, CF격자 최댓값), `mean_rel_spread=0.7540`, `collapse=False` → **CF로 붕괴 안 됨 = 방향이 gap 좌우**.

**저장 원값(보간 아님, nearest row) — 한 CF서 4방향 NCfloor 나란히:**
| CF≈ | uniform | det_vertex_0000 | det_vertex_0101 | local_perfect_corr | max/min |
|---|---|---|---|---|---|
| 0.2 | 0.0070 | 0.0414 | 0.0130 (cf=0.25) | 0.0367 | ~5.9× |
| 0.4 | 0.0298 | 0.0854 | 0.0346 | 0.0786 | ~2.9× |
- 방향간 NCfloor 차이가 **CF² 매끄러운 방향쌍(uniform vs det_0101)** 에선 CF=0.2서 0.0070 vs 0.0130 = **≈1.9×(≈2배)**, **vertex/선형 방향 포함**시 CF=0.2서 최대 ~5.9×, CF=0.4서 ~2.9×. → "방향별 gap 최대 ~2배(또는 그 이상)" 주장의 원값. (요약 지표 max_rel_spread=1.79.)
- ※ 정확히 동일 CF서 4방향 나란히 비교는 collapse_metric이 내부 보간으로 수행(요약만 저장). 위 표는 **각 방향의 저장 원 row 중 목표 CF 최근접값**(추가 재계산 아님).

## 확인 불가한 값과 이유
- 방향별 **정확히 같은 CF에서의 보간된 NCfloor 4값**: 저장 안 됨(collapse는 max_rel_spread 요약만 저장). → 위 표는 최근접 원값으로 대체(보간 미수행, 정직). 
- 그 외 값 A·B·C는 코드/저장값으로 모두 확인됨.

## 실행한 명령 (읽기 수준만)
`git status` / `grep -n`(방향 정의 라인) / `python3 -c`(det_0101 튜플 인덱스 확인·json 값 읽기) — 값 변경·모델 재실행 없음.

## 생성한 .md 로그 경로
`docs/agent_logs/2026-07-08_1120_extract_paperV_batch2.md` (본 파일, 유일한 신규).
