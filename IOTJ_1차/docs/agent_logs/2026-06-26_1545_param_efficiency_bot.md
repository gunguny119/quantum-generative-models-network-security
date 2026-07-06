# Bot 파라미터 효율 비교 — QCBM(216p) vs 고전(저랭크/NMF/비트그룹), AUC·TV (재학습 없음)

- **작성일시**: 2026-06-26 15:45 (로컬)
- **작성 주체**: 로컬 Claude Code agent (plan→default)
- **작업 성격**: 저장된 QCBM 결과(npz) + 고전 저랭크/NMF/비트그룹 효율 비교. **QCBM 재학습 없음. 기존 소스/데이터/결과 무수정.**

## 1. 작업 목적
"왜 고전 말고 양자인가"의 후보 = **표현 효율**(같은 품질을 더 적은 파라미터로). HOIC(단순·저랭크)에선
고전 압승이었음. Bot(고랭크 r=7, 강한 joint 의존)에서 QCBM(216p, best AUC 0.746)과 고전 모델의
파라미터 효율을 비교한다. 핵심: **"QCBM AUC 0.746을 고전이 몇 params로 내는가? 216p 동일 예산에서
누가 높은가?"** 후보 A(양자 효율 우위 재탐색)의 최종 사실 측정. 판정은 웹 AI.

## 2. 시작 git status / 보존 대상
```
?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip  (추적 파일 변경 0)
```
- 새 생성만: `tools/diagnose_param_efficiency_bot.py`, `results2/param_efficiency_bot.json/.png`, 본 `.md`.
- 무수정: 모든 기존 소스/데이터/results2 기존 결과.

## 3. 저장물 확인 + 정합 검증 (재현 경로)
### npz 내용 (`results2/qcbm_bot_restarts.npz`)
| key | shape | 비고 |
|-----|-------|------|
| p_finals | (16, 4096) | 16 restart 학습 분포 |
| q_train | (4096,) | Bot benign train 경험분포 |
| aucs / mmd2 / tv | (16,) | 학습 시 저장 스칼라 |
| weights | (16,6,12,3) | 216 params/restart |
- **states(st_te/st_atk) 미저장** → `train_qcbm_bot_resumable.build_bot()`로 재현(동일 seed=7/0.7/경계).
### 정합 검증 (전부 통과)
- 재현 q_train vs npz q_train: **allclose=True**, 점유 **361** ✓
- npz p_finals 16개를 재현 states로 AUC 재계산 → npz aucs와 **max abs diff = 0.0** ✓
- best(최소 MMD²)=**seed 7**, 재계산 AUC=**0.7458**, TV=0.4933 ✓
- **emp_joint AUC=0.8749 / marginal AUC=0.2870** 재계산 일치 ✓
→ 재현 states가 학습 때 split과 동일. 고전 분포 AUC도 동일 states로 공정 평가 가능.

## 4. 확실한 사실 / 추정 / 확인 불가
- **확실**: 위 정합값, npz 키, occ=361, params=216(L6×12×3), 평가=raw MWU(방향반전 없음).
- **추정(측정으로 확정)**: Bot 고랭크라 고전 저랭크가 AUC 0.746 도달에 큰 r 필요할 것.
- **확인 불가**: 없음(정합 모두 통과).

## 5. 재사용 경로 (코드 확인 완료)
- `experiment2`(E): `tv_dist`, `empirical_dist`, `auc_score`(raw MWU). (build_kernel/MMD²는 본 작업 불필요 — TV·AUC만.)
- `diagnose_param_efficiency_b2`(PB2): `normalize_clip`(clip+정규화), `nmf_factorize`(numpy 곱셈업데이트),
  `group_product_dist`/`group_index`(비트그룹). 파라미터 정의(저랭크 128r, 비트그룹 sum(2^b-1)) 그대로.
- `train_qcbm_bot_resumable`(TBOT): `build_bot()` → q_train/st_tr/st_te/st_atk/occ/edges 재현.
- AUC 평가: 분포 p에 대해 s=-log p(states), `E.auc_score`(import, 재구현 안 함). HOIC 비교: `param_efficiency_b2.json`.

## 6. 분석 설계 (스크립트)
- (A) SVD 저랭크: q_train 64×64 → r=1..40, 근사 normalize_clip → (params=128r, TV, AUC).
- (B) NMF: r=1..20, nmf_factorize → (128r, TV, AUC).
- (C) 비트그룹: marginal(12p)~full, group_product_dist → (sum(2^b-1), TV, AUC).
- QCBM 16점: npz p_finals → (216, TV, AUC), best 강조. 끝점: emp_joint(occ=361, TV0, AUC0.875), marginal(12p, AUC0.287).
- 두 기준: (1) AUC≥0.7458 도달 최소 고전 params(A/B/C, 비단조면 명시); (2) params≤216 예산에서 고전 AUC(A/B/C).

## 7. 고전 곡선 (Bot) — (params, TV, AUC)
| 모델 | params | TV | AUC |
|------|--------|----|-----|
| marginal (1×12) | 12 | 0.882 | 0.2870 |
| 2bit×6 | 18 | 0.876 | 0.2444 |
| 3bit×4 | 28 | 0.839 | 0.6390 |
| 4bit×3 | 45 | 0.737 | 0.6264 |
| **6+6** | **126** | 0.655 | **0.7064** |
| SVD r=1 / NMF r=1 | 128 | 0.669 | 0.5927 |
| SVD r=2 | 256 | 0.439 | 0.5556 |
| **8+4 (비트그룹)** | **270** | 0.612 | **0.7957** |
| SVD r=3 | 384 | 0.242 | 0.7403 |
| **SVD r=4 / NMF r=4** | **512** | 0.188 | **0.8055 / 0.8027** |
| SVD r=6 | 768 | 0.076 | 0.8497 |
| SVD r=9 | 1152 | 0.004 | 0.8733 |
| 9+3 (비트그룹) | 518 | 0.422 | 0.7761 |
| full(1×12) | 4095 | 0.000 | 0.8749 |
| **QCBM best (216p, seed7)** | **216** | **0.493** | **0.7458** |
| QCBM 16 restart 범위 | 216 | — | 0.524 ~ 0.817 |
| emp_joint | 361(점유) | 0.0 | 0.8749 |
- **AUC 비단조(TV-AUC 해리)**: 예) SVD r=2(0.556)<r=1(0.593); 비트그룹 8+4 270p(0.796)>9+3 518p(0.776). 기록만(이전 tv_auc_dissociation 발견과 일치).

## 8. 두 기준 결과
### (기준1) QCBM AUC 0.7458 도달에 필요한 최소 고전 params
| 고전 | 최소 params | (지점, AUC) | 216 대비 |
|------|-------------|-------------|----------|
| SVD | **512** | r=4, 0.8055 | **2.4× (>216)** |
| NMF | **512** | r=4, 0.8027 | **2.4× (>216)** |
| 비트그룹 | **270** | 8+4, 0.7957 | **1.25× (>216)** |
→ **세 방법 모두 216보다 많은 params가 필요**(1.25×~2.4×). (216 이하 구간엔 0.746 도달 행 없음.)

### (기준2) params≤216 동일 예산에서 고전 최대 AUC vs QCBM 0.7458
| 고전 | 최대 AUC(≤216p) | 지점 | 판정 |
|------|-----------------|------|------|
| SVD | 0.5927 | r=1 (128p) | **QCBM 우위** |
| NMF | 0.5927 | r=1 (128p) | **QCBM 우위** |
| 비트그룹 | 0.7064 | 6+6 (126p) | **QCBM 우위** |
→ **216p 동일 예산에서 QCBM(0.746)이 모든 고전을 앞섬.** (128r 이산성상 정확히 216p는 없음 —
  바로 위 256p의 SVD r=2도 0.556로 더 낮고, 270p 비트그룹 8+4가 0.796으로 216 살짝 초과서 추월.)

## 9. QCBM 16점 분포 / 끝점
- QCBM 16 restart AUC(216p): 0.524 / 0.594 / 0.604 / 0.613 / 0.614 / 0.620 / 0.635 / 0.656 / 0.674 / 0.677 / 0.706 / 0.721 / 0.726 / 0.736 / **0.746(best-MMD²)** / **0.817(max)**.
  → 16점 중앙값 ~0.665, best(MMD²선택) 0.746. **216p 고전 최고(비트그룹 6+6 0.706)는 QCBM 중앙값 부근**, QCBM best/상위권엔 못 미침.
- 끝점: marginal(12p, AUC 0.287, 역방향) / emp_joint(361p, TV0, AUC 0.875).

## 10. HOIC vs Bot 비교
| | HOIC(단순·저랭크) | **Bot(고랭크·강 joint)** |
|---|---|---|
| 분포 | 점유 2.3%, SVD r=2면 충분 | 점유 8.81%, SVD r≈9까지 필요 |
| 같은 예산서 우위 | **고전 압승**(SVD r=1 128p가 QCBM TV/AUC 도달) | **QCBM 우위**(216p서 고전 다 아래) |
| QCBM 품질 고전도달 비용 | 고전이 더 적은 params | **고전이 216의 1.25~2.4× 필요** |
→ **Bot은 HOIC와 정반대 방향** — 고랭크 분포에서 QCBM이 동일 예산 AUC 우위 + 고전이 같은 AUC에 더 많은 params 소요.
  (HOIC 참고: param_efficiency_b2.json — QCBM 216p TV=0.093을 SVD r=1(128p)이 이미 도달.)

## 11. 사실 관찰 (판정 아님 — 웹 AI 몫)
1. **동일 216p 예산에서 QCBM(0.746) > 고전 최고(비트그룹 6+6 0.706, SVD/NMF r=1 0.593).** QCBM이 같은 파라미터로 더 높은 AUC.
2. **QCBM AUC 0.746 도달에 고전은 216보다 1.25×(비트그룹 270p)~2.4×(SVD/NMF 512p) 더 많은 params 필요.**
3. **Bot은 HOIC와 정반대**: HOIC는 고전 압승, Bot은 QCBM 효율 우위 방향. 복잡도 진단(고랭크 r=7)이 예측한 대로.
4. **단, TV 기준선 QCBM 열위**: QCBM best TV=0.49는 같은/적은 params 고전보다 나쁨(SVD r=4 512p가 TV 0.19). AUC↔TV 해리 — QCBM은 "분포 적합도(TV)"가 아니라 "공격 분리(AUC)"에서 효율 우위(정직 기록).
5. emp_joint(0.875)는 고전 SVD r≈9(1152p)/비트그룹 full(4095p)에서 도달 — 상한 자체는 큰 고전 비용.

## 12. 생성 파일 / 실행 명령 / git
### 생성(새 파일만)
- `tools/diagnose_param_efficiency_bot.py`, `results2/param_efficiency_bot.json`(13KB), `results2/param_efficiency_bot.png`(141KB), 본 `.md`.
### 실행 명령과 결과
```bash
python3 -m py_compile tools/diagnose_param_efficiency_bot.py        # PY_COMPILE_OK
python3 tools/diagnose_param_efficiency_bot.py                      # 9.5s, DONE
#  정합 게이트 통과: emp_joint 0.8749 / marginal 0.2870 / QCBM best(seed7) 0.7458 / occ 361 (재계산 diff 0.0)
```
- 실패/우회 없음. QCBM 재학습 없음. 방향반전 없음(raw MWU, 0.5 미만도 그대로 기록). 새 dependency 없음.
### git diff --stat (기존 무변경)
```
git diff --stat → (빈 출력 = 추적 파일 변경 0)
```
- 기존 소스/데이터/기존 results2 무수정. 새 untracked만 추가.

## 13. 성공 기준 충족 / 남은 불확실성 / 다음 추천
- 충족: 정합(0.746/0.875/0.287) ✓ / SVD·NMF·비트그룹 (params,TV,AUC) 곡선 ✓ / 두 기준 답 ✓ /
  끝점·QCBM 16점 ✓ / HOIC 비교 ✓ / json·png 저장 ✓ / 무수정·무재학습·무설치 ✓.
- 남은 불확실성:
  - **AUC 비단조(TV-AUC 해리)**로 "도달 params"가 곡선 들쭉날쭉에 의존 — 최소 params 기준으로 보고(보수적).
  - best를 "최소 MMD²"로 선택(0.746). max-AUC restart는 0.817 — 선택 기준에 따라 비교 수치 달라짐(둘 다 기록).
  - 128r 이산성으로 정확히 216p SVD/NMF 부재 — ≤216(r=1) 및 인접점 함께 명시.
  - 파라미터 카운팅은 기존 보수적 정의(고전 유리). 정의 바꾸면 경계 이동 가능.
- 다음 추천:
  1. (양자 우위 방향 확인됨) **재현성/견고성**: 다른 seed 집합·다른 day(다른 Bot 캡처)·다른 인코딩 해상도에서도 216p 우위 유지되는지.
  2. (선택) L=8~10 재학습으로 QCBM best AUC를 0.82+로 끌어올리면 고전 도달 params가 더 커지는지(효율 격차 확대 여부).
  3. 웹 AI 최종 판정(후보 A: 양자 효율 우위 성립 여부).
