# CF² 논문 배치3 — 스윕설정·하이퍼파라미터·앵커출처·N-BaIoT셀수 (조사 전용, 무수정)

- 작성: 2026-07-08 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 무수정·무재계산. 본 .md 1개만 신규.
- 목적: 논문 Methods(IV-A)·Appendix C·앵커 각주 채울 (D)스윕/학습 설정, (E)KCBS 앵커 출처·4-cycle 재구성, (F)N-BaIoT 셀수, (G)Delft 전체 MD5.

## 시작 git status
`git status --short` 변경항목 8(최근 R3/R4/R5 untracked; 본 조사 무관). 본 조사로 파일 변경 없음(로그 1개만).

---

## [값 D] 스윕/학습 설정 (출처 `tools/gap_vs_cf_v0.py`)
- **seed**: `SEED = 20260703` (`:52`). 셀별 rng = `SEED + 1000·s + round(λ·1e4)` (`:300`).
- **λ 그리드** (`:455–457`): base = `np.arange(0.0, 1.0001, 0.1)` = {0.0,0.1,…,1.0} **+** dense = `[0.45,0.475,0.5,0.525,0.55,0.575,0.6]`, `sorted(set(...))` → **16개 고유 λ**: {0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.7, 0.8, 0.9, 1.0}.
- **seed 수**: **5** (`--seeds` 기본 `:423`; 각 λ에서 seed 5회).
- **표본 N**: `Ns` 기본 = **{∞(모집단), 5000}** (`:463`, `--full` 시 500 추가). CF는 λ마다 우리 CF-LP로 실측(이론 max(0,2λ−1)와 대조).
- **restart 수**: `rq, rc = (3, 2)` (`:465`) → QCBM restart **3**, 고전 restart **2**, NCfloor restart **rc+1=3**.
- **QCBM 학습** (`fit_quantum :224–228`): optimizer = **Powell**, `maxiter=2000, xtol=1e-6, ftol=1e-9`. 초기값 = `_q_inits` (`:160–167`) = **표준 CHSH-최적각 1개 + 랜덤 2개** (총 restart 3). ※경사無(gradient-free).
- **HMM/고전 학습** (`fit_classical :235–241`): optimizer = **Powell**(동일 설정), 초기값 = `rng.normal(0,1,size=5H)` 랜덤, restart **2**. → **EM 미사용**. **"EM 반복 수" = 해당 없음(EM 아님, 직접 MLE)**.
- **NCfloor** (`nc_floor :271–284`): Powell, restart 3, `maxiter=6000, xtol=1e-7, ftol=1e-10` (국소 폴리토프 16 꼭짓점 가중치 최적화).
- (`kink_origin_v1.py`·`gate_e_coefficient.py`는 이 G의 함수를 무수정 import; 자체 seed `20260704`.)

## [값 E] KCBS 앵커 실험 출처 + 4-cycle 재구성
- **출처(published 실험)**: **Sci. Adv. abk1660 / PMC8827658** — *loophole-free Kochen–Specker contextuality test with **two atomic ion species***. (`gate_r2 :46,279` 주석 `# ...(PMC8827658)`, `data_sources.kcbs="PMC8827658/Sci.Adv. abk1660 (Table1 실측 상관)"`; `gate_r1 :15,47`.) → **코드/주석에 명시된 이 논문만 인용**(그 외 추측 안 함).
- **실측 상관 (Table 1)**: `KCBS_E = [0.6164, 0.625, 0.6678, −0.6166]` = ⟨Ô₀Ô₁⟩,⟨Ô₁Ô₂⟩,⟨Ô₂Ô₃⟩,⟨Ô₃Ô₀⟩, ±[0.0079,0.0078,0.0074,0.0079] (`gate_r1 :48–49`, `gate_r2 :46`).
- **★ 시나리오 구조**: 코드/데이터는 **4개 관측량 Ô₀…Ô₃ = 4-cycle(짝수 사이클, CHSH-동치) 4상관**. **"5측정 KCBS 중 4개 선택"이 아님** — 파일엔 4상관만 있고 5→4 서브셋 매핑 근거 없음(추측 금지, "4-cycle chained inequality"로만 서술).
- **4-cycle → CHSH 재구성** (`gate_r2 kcbs_chsh_box :67–69`): 각 상관 E를 `pair_from_corr(E)`(균등 marginal: $P(a,b)=(1{+}E)/4$ if $a{=}b$ else $(1{-}E)/4$, `:63–65`)로 2×2 분포화 → **CHSH 컨텍스트에 배치: ctx(0,0)=E01, (0,1)=E12, (1,0)=E23, (1,1)=E30** (S=C 보존). 
- **witness/CF 재확인**: $C=E_{01}+E_{12}+E_{23}-E_{30}=$ **2.526** (비맥락 한계 2), CF-LP = **0.2629** = $(C-2)/2=0.2629$ (교차확인 Δ=0). `r2.json track_A_kcbs.cf=0.2629, C=2.526`. → **이 데이터에서 나온 값 맞음**.

## [값 F] N-BaIoT 감사 셀 수 (gate_r3)
출처 `results2/gate_r3/r3.json → cells`.
- **총 18 셀** = **def-a 15 + def-b 3** (요청 "18 맞나?" → **맞음**).
- 곱셈 구조: **기기 3** (Danmini_Doorbell, Ecobee_Thermostat, Ennio_Doorbell) × **부분집합 5** (highvar, targeted, random1, random2, random3) = **def-a 15**; **def-b = 기기 3 × 1(시간창) = 3**. → (정의 2종 중 def-a만 5부분집합, def-b는 기기당 1) → 3×5 + 3×1 = 18.
- **def-b CF ≤ 8e-5 재확인**: def-b CF 목록 = {2e-5, 7.6e-5, 0.0}, **최대 7.627e-05** (≤8e-5 ✓).
- 파라미터: rows=500000, k=8, boot=500 (`r3.json params`).
- ※ 참고: 후속 **gate_r5**는 **9기기 × 6 = 54 셀**(def-a 45 + def-b 9). 논문이 "3기기 18셀"이면 gate_r3, "9기기 54셀"이면 gate_r5 — **둘 다 존재**(본 값은 요청대로 gate_r3=18).

## [값 G] Delft 전체 MD5
출처 `results2/gate_r2/r2.json → track_B_delft.zip_md5`.
- **전체 MD5 = `342f29f8288c46575818acd2acebd535`** (Appendix C의 prefix `342f29f8`와 일치, `zip_md5_ok=True`).
- DOI = `10.4121/uuid:6E19E9B2-4A2D-40B5-8DD3-A660BF3C0A31` (Hensen 2015 loophole-free Bell, 4TU.ResearchData), n_trials = 245.

## 확인 불가한 값과 이유
- **KCBS "5측정 중 4개" 매핑 상세**: 파일엔 4상관(4-cycle)만 저장 — 5-cycle 원본에서 4개를 골랐는지 여부는 **코드/데이터에 없음** → "확인 불가"(4-cycle chained로만 서술, 추측 금지).
- **HMM EM 반복 수**: **해당 없음**(EM 미사용, Powell 직접 MLE) — "없음"이 정답.
- 그 외 D·E·F·G 값은 코드/저장값으로 모두 확인.

## 실행한 명령 (읽기 수준만)
`git status` / `grep -n`(seed·λ·optimizer·KCBS·MD5 라인) / `sed -n`(kcbs_chsh_box 본문) / `python3 -c`(r1/r2/r3.json 값 읽기). 값 변경·모델 재실행 없음.

## 생성한 .md 로그 경로
`docs/agent_logs/2026-07-08_1210_extract_paperV_batch3.md` (본 파일, 유일한 신규).
