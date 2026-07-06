# Bot 양자 효율 우위의 견고성 검증 — 216p QCBM 우위가 best 운인지 분포적인지

- 작성: 2026-06-26 16:53 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- QCBM 재학습 없음 (저장된 16 restart npz 재사용). raw MWU 유지. 판정은 웹 AI.

## 1. 작업 목적
Bot(고랭크 r=7 복잡 분포)에서 QCBM(216 params)의 파라미터 효율 우위가 "best 1개의 운"인지,
아니면 16 restart 분포적으로도 성립하는지를 저장된 결과만으로 정밀 규명한다. QCBM 16 restart의
AUC 분포를 고전(SVD/NMF/비트그룹) 곡선과 비교해 견고성을 **사실로** 기술한다(판정 금지).

## 2. 시작 git status, 보존 대상
시작 시 `git status --short`:
```
?? docs/
?? qcbm/
?? qcbm_share_20260626.zip
```
`git diff --stat`(tracked) = 비어 있음. 초기 커밋은 top-level 분석 스크립트만 추적하며 `qcbm/`,
`docs/` 전체가 untracked. 보존 대상: 모든 기존 소스/데이터/결과 파일(수정·삭제·이동 금지, import/읽기만).

**중요 발견**: 타깃 산출물 3개가 이미 16:04에 존재했다 —
`tools/verify_bot_efficiency_robustness.py`, `results2/bot_efficiency_robustness.json`,
`results2/bot_efficiency_robustness.png`. 그러나 `docs/agent_logs/`는 비어 있었다(이전 에이전트가
분석은 완료했으나 .md 로그를 남기지 않음). 본 실행에서 스크립트를 **재실행해 재현성/정합을 직접
검증**하고, 누락된 이 로그를 작성했다. 기존 소스/데이터 파일은 일절 수정하지 않았다.

## 3. 재현/정합: 16 AUC 재계산 vs qcbm_bot.json
스크립트는 `train_qcbm_bot_resumable.build_bot()`로 인코딩(test/attack states)을 재현하고,
`experiment2.auc_score`(raw MWU, s=−log p, 방향반전 없음)로 16 p_final의 AUC를 재계산한다.
재실행 결과(약 7초):
```
점유 361/4096 (8.81%)  -> qcbm_bot.json occupied_train_states=361 와 일치
q_train 재현 일치 (np.allclose)
[정합] 16 AUC 재계산 max|Δ| = 0.00e+00
       median 재=0.6650 (json 0.6650), max 재=0.8167 (json 0.8167) -> OK
```
**정합 완전 통과**: npz의 `aucs`와 p_final 재계산이 비트 단위로 일치(Δ=0), 저장 분포와도 일치.
인코딩/평가가 정확히 재현됨.

## 4. 고전 비교 기준 (216p-이하 최고 = 비트그룹 6+6)
`param_efficiency_bot.json`에서 고전 곡선 로드(재생성 안 함). params ≤ 216 인 고전 점들:
- 비트그룹: 12p=0.287, 18p=0.244, 28p=0.639, 45p=0.626, **126p(6+6)=0.7064**
- SVD/NMF: 128p(r=1)=0.593, (256p r=2=0.556 — 비단조 하락)
- → **216p-이하 고전 최고 = 비트그룹 6+6 (126p, AUC 0.7064)**. 정확히 216p 점은 고전에 없으며,
  바로 위 고전 점은 비트그룹 8+4 (270p, 0.7957).

## 5. 확실한 사실 / 추정 / 확인 불가
**확실(코드·재계산으로 확인)**
- QCBM 216p 16 restart AUC: min 0.5235 / median 0.6650 / max 0.8167 / std 0.0709 (재계산=저장값, Δ=0).
- 고전 216p-이하 최고 = 0.7064 (비트그룹 6+6, 126p).
- 고전 최고(0.7064) 초과 QCBM restart = **5/16 (31%)**.
- 선택규칙 min-MMD²(라벨 무사용)는 seed7(AUC 0.7458, 상위 2/16)을 고름.
- corr(MMD², AUC) = +0.077, corr(TV, AUC) = −0.014 (둘 다 0 근방).

**추정(확인됨)**
- "median QCBM(0.665) > 고전 최고" 가설은 **성립 안 함** — median이 고전 최고보다 아래(0.665 < 0.706).
  사전 추정대로였다.

**확인 불가/한계**
- npz에 test/attack states가 직접 저장돼 있지 않아 인코딩을 CSV에서 재현했으나, occ=361·q_train·
  16 AUC가 모두 정합하여 재현 신뢰도는 높음.
- 고전 곡선은 이산 점들이라 "QCBM median 도달 params"는 점 사이 보간이 아닌 **실제 점 중 최소
  params**로 보수적으로 산출(아래 6 참조).

## 6. 견고성 지표
- **고전 최고(0.7064) 초과 QCBM restart: 5/16 (31%)**
- QCBM AUC 분포 vs 고전선(0.7064):
  - min 0.5235(아래) / p25 0.6135(아래) / median 0.6650(**아래**) / p75 0.7220(위) /
    p90 0.7409(위) / max 0.8167(위) / std 0.0709
- 우위 성립 여부:
  - **best(0.7458, min-MMD² 선택): 위 ✅**
  - **p90(0.7409): 위 ✅**, **p75(0.7220): 위 ✅**
  - **median(0.6650): 아래 ❌**
- 고전이 QCBM **median(0.665)** 도달 필요 params: **126p (비트그룹 6+6, 0.706)** — 즉 216p보다
  **적은** 예산으로 도달. median 수준에서는 고전이 오히려 더 효율적.
  - 가족별: SVD 384p, NMF 384p, 비트그룹 126p.
- (참고) 고전이 QCBM **best(0.746)** 도달 필요 params: **270p (비트그룹 8+4, 0.796)** — 216p의 1.25배.
- **선택 견고성(중요 caveat)**: min-MMD²가 좋은 restart를 골랐지만 corr(MMD²,AUC)=0.077 ≈ 0.
  즉 라벨 없는 선택규칙이 고AUC restart를 **보장하지 못함**; seed7이 상위에 온 것은 사실상 운에 가깝다.
  무상관 가정 시 기대 선택 AUC ≈ median(0.665).

## 7. TV-AUC 해리 (restart 전반)
- corr(TV, AUC) = −0.014 ≈ 0. QCBM TV 범위 0.493~0.703로, 같은/적은 params 고전보다 적합도(TV)는
  나쁘지만 AUC는 상위권. **분포 적합도(TV) ≠ 탐지 성능(AUC)** 해리가 restart 전반에서 확인됨.
  (per-restart tv/auc는 `bot_efficiency_robustness.json`의 `tv_auc_dissociation`에 저장.)

## 8. 사실 관찰 (판정 아님)
- **216p 우위는 "분포적으로 일부" 성립, 중앙값에서는 미성립.**
  - best/p90/p75 (상위 약 25%, 5/16)는 고전 최고를 넘지만, median 및 하위 절반은 고전 최고 아래.
- 우위의 견고성 등급(사실 기술): **16개 중 5개(31%)가 고전 216p-이하 최고(0.706)를 초과.
  median(0.665)은 고전 최고 아래. best 우위는 확실하나 "전형적(분포 중앙값)" 우위는 아님.**
- 추가 caveat: 라벨 없는 실제 선택규칙(min MMD²)은 AUC와 거의 무상관이라, "best를 안정적으로 고를 수
  있는가"까지 따지면 우위의 실효 견고성은 더 약함.
- 한줄 요약: **"216p QCBM 우위"는 best 운만은 아니지만(상위 31%에서 성립), 분포 중앙값에서는
  고전이 더 효율적 — 즉 부분적·상위분위 한정 우위이며 라벨없는 모델선택까지 고려하면 신뢰도는 제한적.**

## 9. 생성 파일, 실행 명령과 결과
**생성/갱신(본 작업의 지정 산출물 + 로그)**
- `qcbm/results2/bot_efficiency_robustness.json` (재실행으로 결정적 재생성, 내용 동일)
- `qcbm/results2/bot_efficiency_robustness.png` (재생성)
- `qcbm/tools/verify_bot_efficiency_robustness.py` (이전 16:04 산출물 — 내용 검토·재실행, 미수정)
- `docs/agent_logs/2026-06-26_1653_bot_efficiency_robustness.md` (본 로그, 신규)

**실행 명령/결과**
```
python3 tools/verify_bot_efficiency_robustness.py   # real 0m6.7s, 정합 OK, DONE
# [정합] max|Δ|=0.00e+00 ; 고전최고 0.7064 ; 초과 5/16 ; median 아래 ; corr(MMD²,AUC)=0.077
```
실패/에러 없음. 환경: python3 /usr/bin/python3, OMP_NUM_THREADS=1.

## 10. git diff --stat (기존 무변경)
```
$ git diff --stat        # tracked 파일 변경: 없음 (비어 있음)
$ git status --short
?? docs/
?? qcbm/
?? qcbm_share_20260626.zip
```
기존 소스/데이터/결과(tracked) 무수정. `qcbm/`·`docs/`는 본래 전체 untracked이므로 신규 로그 추가가
새 tracked 변경을 만들지 않음. QCBM 무재학습, 새 dependency 무설치.

## 11. 성공 기준 충족 여부 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] 16 p_final AUC 재계산 = qcbm_bot.json 분포 정합 (Δ=0).
- [x] 216p 예산 고전 최고(0.706) 초과 QCBM restart = 5/16 명확 보고.
- [x] QCBM 분포(min/median/max/std/p75/p90)와 고전선 관계 기술.
- [x] best 전용 vs 상위 일부 vs 중앙값 구분 (best/p90/p75 위, median 아래).
- [x] 고전이 QCBM median(0.665) 도달 params 역산 = 126p 보고.
- [x] json/png 저장.
- [x] 기존 파일 무수정, 무재학습, 무설치.

**남은 불확실성**
- 우위가 "상위 31%"에 한정 — 다른 split seed/데이터셋에서도 같은 분위 패턴인지 미검증.
- min-MMD² 선택의 무상관성(0.077)은 본 16개 표본 한정; 더 많은 restart에서 안정성 미확인.

**다음 추천 단계**
1. 일반화 검증: 다른 benign split seed(예: 5개)로 QCBM 분포의 상위분위 우위가 재현되는지.
2. 라벨 없는 모델선택 대안(min TV, min KL 등)이 min-MMD²보다 고AUC restart를 더 잘 고르는지 비교.
3. 복잡도(랭크)와 "상위분위 우위" 폭의 관계: HOIC(단순)~Bot(복잡) 사이 분포 우위가 단조 증가하는지.
