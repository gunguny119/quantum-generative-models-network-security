# UNSW QCBM(MMD) 학습 + 견고성 + 180p 효율 — 양자 우위가 UNSW에서 재현되는가

- 작성: 2026-06-27 03:41 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 실제 QCBM 학습 수행(restart 체크포인트). raw MWU. ★전수. 판정은 웹 AI.

## 1. 작업 목적
독립 데이터셋 UNSW-NB15의 복잡+joint의존 공격(ALL)에서 QCBM(180p, 10bit)을 학습해, CIC Bot의
결과("best 운, median 미달, min-MMD² 선택 불가")가 재현되는지, 아니면 UNSW에선 견고한 양자
우위가 나오는지를 사실로 규명한다(양자 정당화의 최종 판결). 16 restart 분포 + 180p 효율 + 견고성/
선택을 통합 측정.

## 2. 시작 git status, which tmux, 절전방지
`git status --short`: `?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip` (전체 untracked).
`which tmux` → `/usr/bin/tmux` 있음. 16코어 / 32GB. 절전: 시스템 설정 변경 없음(체크포인트가 안전망).
보존 대상: 모든 기존 소스/데이터/결과(import/재사용만). 기존 `tools/diagnose_*`·`train_qcbm_*` 무수정.

## 3. UNSW 10비트 인코딩(M2 flow_iat), 점유, 정합 검증
공통 10bit(syn/psh 제외): proto2 + size4 + fwd_iat2 + flow_iat2 = 1024상태. 매핑:
proto→proto(tcp=6/udp=17/else=−1), size→smean, fwd_iat→sinpkt, **flow_iat→M2=(sinpkt+dinpkt)/2**.
경계는 UNSW benign train(seed=7, 0.7)에서. **점유 174/1024 (16.99%)**. benign train 65,099 / test 27,901 /
attack(ALL,label1) 164,673 (★전수, dropna 0).
**정합 게이트 통과**: emp_joint AUC=**0.8705**(기준 0.8705), marginal AUC=**0.7027**(기준 0.7027) — 정확 일치.
교차 진단(M2)과 인코딩 일치 확인 후 학습 진행.

## 4. 체크포인트/재개 메커니즘
- 각 restart(seed) 완료 즉시 `results2/unsw_ckpt/restart_<seed>.npz` 원자적 저장(p_final/weights/hist/AUC).
- `Pool.imap_unordered` → 완료마다 저장. 시작 시 `scan_done()`이 완료 seed 스캔→건너뜀. EPOCHS는
  `unsw_ckpt/decision.json`에 1회 저장(재개 시 동일). 최종 16/16 체크포인트 생성 확인.

## 5. smoke 시간, 전체 추정, 확정 EPOCHS
smoke(seed=9999, 50epoch): wall 61.3s = **1.226s/epoch**. NWORKERS=8 → waves=2. 추정 full(300ep)=
846s=**0.24h**(5h 이내). **확정 EPOCHS=300**.

## 6. 본 학습 실행 + 재개 방법
권장(tmux):
```bash
tmux new -s qcbm_unsw
python3 -u tools/train_qcbm_unsw_resumable.py 2>&1 | tee -a results2/qcbm_unsw_train.log
# detach Ctrl+b d / 재연결 tmux attach -t qcbm_unsw / 중단돼도 재실행 시 완료 restart 건너뜀
```
실제 실행: nohup 백그라운드(detach) + restart 체크포인트로 수행(체크포인트가 동일 안전망).
03:22 시작 → 03:39 완료, **wall ≈ 12분**(2 waves × ~6분). 16/16 정상 완료. gradient 정상 수렴
(아래), barren plateau 징후 없음. (tmux 미사용했으나 체크포인트로 동등 보호; 사용자 재현 시 위 tmux 권장.)

## 7. 분석1 — 학습 결과
- best(min-MMD² 선택) seed=6: MMD²=2.37e-03, TV=0.3724, **AUC=0.7929**, gnorm 8.38e-03→2.06e-05
  (정상 수렴, barren plateau 아님). best의 AUC 순위 = 상위 **2/16**.
- **16 AUC 분포: min 0.6327 / median 0.7578 / p75 0.7781 / p90 0.7899 / max 0.8025 / std 0.0464.**
- 비교: marginal 0.7027 < QCBM median 0.758 < QCBM best 0.793 < **emp_joint 0.8705(상한)**.
  QCBM best는 상한의 **91.1%**. marginal 초과 **13/16**, emp_joint 초과 **0/16**.
- QCBM은 ALL(혼합공격)을 잘 학습함(상한근접 91%) — "QCBM이 혼합공격을 못 잡음"은 아님.

| 모델 | AUC | 비고 |
|---|---|---|
| marginal | 0.7027 | 독립곱 |
| QCBM median (180p) | 0.7578 | 16 restart 중앙값 |
| QCBM best (180p, min-MMD²) | 0.7929 | 상한근접 91.1% |
| emp_joint | 0.8705 | 상한 |

## 8. 분석2 — 180p 효율 (QCBM vs 고전)
고전 곡선(10bit, low-rank params=64r):
- SVD: r=1(64p)=0.8110, **r=2(128p)=0.8340**, r=3(192p)=0.8157(비단조).
- NMF: **r=2(128p)=0.8351**. 비트그룹: marginal(10p)=0.703, 5+5(62p)=0.808, 8+2(258p)=0.835, full(1023p)=0.871.
- **고전 180p-이하 최고 = NMF r=2 (128p, AUC 0.8351).**

| 기준 | 결과 |
|---|---|
| QCBM best(0.793) 도달 고전 params | SVD/NMF **64p**(r=1, 0.811), 비트그룹 **62p**(5+5, 0.808) — QCBM 180p의 **약 1/3** |
| params≤180 동일예산 고전 최대 AUC | **0.8351**(NMF r=2) — QCBM best 0.793 대비 **고전 우위** |

→ **고전이 QCBM AUC를 ~62–64p로 도달(QCBM은 180p), 동일 180p 예산에서 0.835 vs 0.793으로 고전 압도.**
   UNSW에서 양자 파라미터 효율 우위 **없음**(오히려 고전이 더 효율적).

## 9. 분석3 — 견고성 + 선택
- **고전 최고(0.8351) 초과 QCBM restart = 0/16.** best/p90/p75/median **전부 고전 아래**.
- 고전이 QCBM median(0.758) 도달 params: SVD/NMF r=1(64p)로 이미 0.811 — median도 고전이 더 적은 params로 초과.
- 선택: **corr(MMD²,AUC) = −0.375 (Spearman −0.432)**, corr(TV,AUC)=−0.240. min-MMD²가 상위 2위 모델 선택
  (음의 상관 = 적합 좋을수록 AUC 높음, 선택이 어느 정도 **작동**). 단 best조차 고전 아래라 무의미.

## 10. 핵심 비교 (Bot vs UNSW)
| 항목 | Bot (216p, 12bit) | UNSW (180p, 10bit) |
|---|---|---|
| QCBM best vs 동일예산 고전최고 | **우위** 0.746 > 0.706 | **열위** 0.793 < 0.835 |
| median 우위 | 아니오 (0.665<0.706) | 아니오 (0.758<0.835) |
| 고전최고 초과 restart | 5/16 | **0/16** |
| 고전이 QCBM-best 도달 params | 270p (216의 1.25배, 양자 유리) | 62–64p (180의 0.35배, **고전 유리**) |
| min-MMD² 선택 corr(MMD²,AUC) | 0.077 (무상관, 선택불가) | −0.375 (음, 선택 다소 작동) |
| 상한근접(best/emp_joint) | 85% | 91% |

## 11. 사실 관찰 (판정 아님)
- **Bot 패턴(best 운으로 고전 초과) 재현?**: **재현 안 됨.** UNSW에선 QCBM best조차 동일예산 고전을 못
  넘고(0/16), 고전이 QCBM AUC를 1/3 params로 달성 — 즉 **양자 우위 부재, 고전 효율 우위**(HOIC형 결과에 가까움).
- **median 미달**: Bot과 동일하게 UNSW에서도 median은 고전 아래(더 심함 — best도 아래).
- **선택 가능성**: Bot(corr 0.077)보다 UNSW(−0.375)에서 min-MMD²↔AUC 상관이 더 뚜렷해 선택은 나음.
  그러나 best조차 고전 아래라 선택 개선이 결론을 못 바꿈.
- **QCBM 학습 자체는 정상**: 상한 91% 근접, 13/16 marginal 초과, gradient 정상 수렴(barren plateau 아님).
  즉 "QCBM이 학습 실패"가 아니라 "잘 학습해도 고전 저랭크가 더 파라미터 효율적"인 것.
- 원인 단서(교차 진단과 정합): UNSW ALL은 sp(TV,AUC)≈−1.0로 저랭크가 곧 고AUC(detection 신호가
  저랭크에 집중) → 고전 SVD r=2(128p)가 이미 emp_joint(0.871)의 96%(0.835)에 도달. 양자가 비집고
  들어갈 "고랭크에서만 탐지되는" 여지가 작음.
- 한줄: **독립 데이터셋 UNSW에서 양자 파라미터 효율 우위는 재현되지 않았고, 고전 저랭크가 더 효율적이다.**

## 12. 실행 명령과 결과 (실패 포함)
```
python3 -m py_compile tools/train_qcbm_unsw_resumable.py     # OK
UNSW_SMOKE_ONLY=1 python3 tools/train_qcbm_unsw_resumable.py  # 정합 OK(0.8705/0.7027), 1.226s/ep, EPOCHS=300
nohup python3 -u tools/train_qcbm_unsw_resumable.py >> results2/qcbm_unsw_train.log   # 16/16, wall~12min
python3 tools/diagnose_param_efficiency_unsw.py             # 정합 OK, 효율/견고성/선택 저장
```
실패/에러 없음. 5h 이내(12분). 샘플링 없음(전수). 새 dependency 없음(scipy 기존).
주의: 최초 백그라운드 런처는 즉시 반환됐고(`&`), 실제 학습은 detached python(PID 37875)이 수행 →
체크포인트 16개와 train.log로 완료 확인(실행 안 한 것을 했다고 하지 않음).

## 13. git diff --stat (기존 무변경)
```
$ git diff --stat    # tracked 변경: 없음
$ git status --short
?? docs/   ?? qcbm/   ?? qcbm_share_20260626.zip
```
기존 소스/데이터/결과(tracked) 무수정. 기존 CIC/UNSW 데이터·Bot 결과 보존. dependency 무설치.
신규: tools/{train_qcbm_unsw_resumable, diagnose_param_efficiency_unsw}.py, results2/qcbm_unsw.json,
qcbm_unsw_restarts.npz, unsw_ckpt/, param_efficiency_unsw.json, unsw_robustness.json,
{train,dist,anomaly}_qcbm_unsw.png, param_eff_unsw.png, qcbm_unsw_train.log, 본 로그.

## 14. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] UNSW 10bit emp/marginal(0.8705/0.7027) 정합.
- [x] restart 체크포인트 생성(16/16), 재개 로직(scan_done) 동작.
- [x] 16 restart 학습 완료, 각 AUC 저장(★전수, 샘플 수 명시).
- [x] 분석1(best 0.793, 분포, 상한근접 91%), 분석2(고전 64p로 도달/180p 0.835), 분석3(0/16, corr −0.375).
- [x] Bot 패턴 재현 여부 사실 정리(재현 안 됨 — 고전 우위).
- [x] json/npz/png 저장, 기존 무수정·무설치·5h 이내.

**남은 불확실성**
- ALL(혼합) 기준. 단일 공격(Exploits/Reconnaissance)에서는 다를 수 있음(미측정) — 단 ALL에서 QCBM은
  잘 학습됐고 고전이 압도하므로 단일 공격이 뒤집을 가능성은 낮아 보임(미검증).
- 10bit/L=6(180p) 고정. 더 깊은 회로(L↑)나 다른 인코딩이 양자에 유리할지는 별도 학습 필요.
- 고전 SVD r=2 비단조(r=3에서 하락) — 곡선 노이즈 가능, 단 r=2가 best라 결론 영향 작음.

**다음 추천 단계**
1. (판정 입력용) 웹 AI에 "Bot 우위는 best운·UNSW 미재현(고전 우위)" 사실 전달 — 양자 정당화 재검토.
2. 필요 시 UNSW 단일 공격(Exploits)·더 깊은 L로 보강 학습해 ALL 결론의 일반성 확인.
3. "왜 UNSW에서 고전이 효율적인가"(detection 신호가 저랭크 집중, sp(TV,AUC)≈−1) 정량 정리 →
   양자가 유리한 분포 조건(고랭크+고랭크에서만 탐지) 가설 재정의.
