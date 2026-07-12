# CF² 논문 Figure 3개 생성 (results2 저장값만, 데이터 조작·재계산 없음)

- 작성: 2026-07-08 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. results2 **읽기 전용**; `figures/` PNG 3개 + 본 로그 + 임시 스크립트 `tools/_make_figs.py` 만 신규(기존 무수정).
- 목적: 논문 Fig 1/2/3 을 **이미 저장된 실험결과값**으로만 생성(300 dpi). 지어낸 값·보간 없음.

## 시작 git status
`git status --short` 변경항목 9(최근 R3/R4/R5 untracked; 무관). 본 작업으로 results2/코드 무변경.

## 사용한 데이터 출처 (파일 + key + 추출 배열)
### `results2/gap_cf/gap_cf.json → sweeps.inf` (16 λ 지점, N=∞)
- `lam`, `cf`, `gap_contextual`, `gap_matched`, `kl_quantum`, `kl_class_matched` (셀별 저장 — **모두 존재 확인**).
- **CF 정렬**: [0×8, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 1.00] (λ≤0.5는 CF=0 중복 8점)
- **gap_contextual (CF순)**: [0,0,0,0,0,0,0,0, 0.0004, 0.0017, 0.0039, 0.0070, 0.0298, 0.0629, 0.0961, 0.1293]
- **gap_matched (CF순)**: (Fig1 좌 점선) — 저장값 사용.
- **KL_quantum(λ)** (λ순): [0,0,0,0,0,0,0,0,0,0,0,0,0, 0.0095, 0.0480, 0.1583]
- **KL_class_matched(λ)** (λ순): [0, 0.0025, 0.0101, 0.0229, 0.0411, 0.0525, 0.0587, 0.0654, 0.0725, 0.0800, 0.0879, 0.0964, 0.1352, 0.1840, 0.2473, 0.3466]

### `results2/gate_e/gate_e.json → results.CHSH.uniform` (Fig 2 주석/곡선)
- `c_fit = 0.16995`, `c_pred_limit = 0.16667 (=1/6)`, `alpha = 2.0057`, `r2_fit = 0.9999744`.

### `results2/gate_r2/r2.json` (Fig 3 앵커)
- KCBS `track_A_kcbs`: cf=**0.2629**, gap_measured=**0.0123**, cf_ci=[0.2482, 0.2795], rel_dev(`anchor_placement.kcbs`)=**0.037**.
- Delft `track_B_delft`: cf=**0.2112**, gap_measured=**0.0085**, cf_ci=[0.1151, 0.4185], rel_dev(`anchor_placement.delft`)=**0.053**.
- N-BaIoT: 원점 (CF≈0, gap≈0) 참조 마커(Gate R3 결과, 값 자체는 0 근방으로 표기).

## 생성 그림
### `figures/fig1_gap_vs_cf.png` — 2패널
- **(a) gap vs CF**: (cf, gap_contextual) 산점+선(실선), gap_matched(점선). 음영 CF≤√2−1(파랑, quantum-realizable) / >√2−1(빨강, super-quantum).
- **(b) CF self-check + KL**: λ vs CF(LP, 파랑) + CF 이론 max(0,2λ−1)(빨강 점선, **이론 닫힌형 — 허용**) + λ=1/√2 Tsirelson 수직선. 우축 = **KL_quantum(λ)·KL_class_matched(λ) (저장값)**.
- **★ Fig1 우패널 KL 곡선 = 저장값 있음 → 포함**(kl_quantum·kl_class_matched, gap_cf.json). 생략 없음.

### `figures/fig2_cf2_law.png` — 2패널
- **(a) linear**: (cf, gap_contextual) 산점 + `gap=c·CF²` 곡선, **c=c_fit=0.16995 (gate_e CHSH/uniform c_fit)**(캡션 명시). 제목에 α=2.006, R²=0.99997.
- **(b) log-log**: 같은 산점 + slope-2 기준선(CF² 확인) + slope-1 점선(vertex/linear 참조).

### `figures/fig3_anchors.png` — 단일
- 예측 곡선 `gap=CF²/6` (c=1/6) + 앵커 3점: KCBS(★, x-CI, dev 3.7%), Delft(◆, x-CI, dev 5.3%), N-BaIoT(▽, 원점≈0). 저장값 사용.

## Fig 1 우패널 KL 곡선 판정
**저장돼 있었고(그렸음)**: `kl_quantum`·`kl_class_matched`가 gap_cf.json sweeps.inf 셀별로 존재 → 두 곡선 모두 실제 저장값으로 그림. (임의 곡선·생략 없음.)

## 저장값이 없어 생략한 요소
- 없음. Fig 1·2·3 모든 점/곡선이 results2 저장값(또는 이론 닫힌형 CF=max(0,2λ−1) 1개)에서 옴.
- 참고: c는 두 값 저장(c_fit 0.16995 / c_pred 1/6=0.16667) — Fig2는 c_fit, Fig3은 명세대로 1/6 사용(캡션 명시). 지어낸 값 아님.

## 검증
- `ls -la figures/*.png`: fig1(439KB)·fig2(290KB)·fig3(220KB) 300 dpi 생성 확인.
- Fig1 렌더 육안 확인(음영·gap 2곡선·CF LP=이론 일치·KL 2곡선·Tsirelson선 정상).

## 생성한 .md 로그 경로
`docs/agent_logs/2026-07-08_1250_make_paper_figures.md` (본 파일). 스크립트: `tools/_make_figs.py`(재현용).
