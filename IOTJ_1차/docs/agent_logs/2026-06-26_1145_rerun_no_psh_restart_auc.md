# psh 제거 11비트 — 16 restart 각각의 AUC 계산·저장 재실행 (결정론적 재현)

> **전제: 기존 소스 무수정(import + 모듈속성 대입). 학습은 결정론적 재현(seed 고정), tmux 보호.**

- **작성일시**: 2026-06-26 11:45 (로컬), 본 학습 tmux 진행 중
- **작성 주체**: 로컬 Claude Code agent (default mode)

---

## 1. 작업 목적

best AUC 0.8023 > emp_joint 0.7773 이 **견고**인가 **best 운**인가? 이전 학습을 동일 설정으로 결정론 재현하되, **16 restart 각각의 p_final로 AUC를 계산**해 "emp_joint(0.7773) 초과 restart 수/16"을 직접 측정. 16 p_final도 npz로 저장(향후 재실행 불필요). 판정은 웹 AI.

---

## 2. 시작 git status / tmux / 이전 설정

```
?? docs/ ?? qcbm/ ?? qcbm_share_20260626.zip (전부 미추적; 추적 변경 0)
which tmux: /usr/bin/tmux (3.2a), 잔여 세션 없음
이전 본학습 설정(qcbm_iat_no_psh.json): RESTARTS=16, EPOCHS=300, qcbm_auc 0.8023, best_tv 0.0622
```
- 추가 파일: `tools/rerun_no_psh_restart_auc.py`, `results2/no_psh_restart_auc.{json,png,_train.log}`, `results2/qcbm_iat_no_psh_restarts.npz`, 본 `.md`.

---

## 3. 재사용 경로 (실제 확인)

- `import train_qcbm_iat_no_psh as NP`: `build_nopsh()`(q_train/st_te/st_atk/occ 재현), `SPEC`(nq=11), `run_train(d, R, EP)`(E.LAYERS/EPOCHS/RESTARTS 설정 후 run_config 호출).
- **train_one이 각 restart dict에 `p_final` 반환**([experiment2.py:212-224](../../qcbm/experiment2.py#L212-L224)) → `run_config`의 `cfg["results"][i]["p_final"]`로 16개 모두 접근.
- 각 restart AUC: `E.anomaly_eval`에 restart p_final을 best 자리로 감싸 적용(평가 동일: s=-log p, auc_score raw MWU).
- 동일 설정: nq=11, L=6, LR=0.1, RESTARTS=16, EPOCHS=300 → best 0.8023 재현 기대(결정론 검증 지표).

---

## 4. 인코딩 재현 검증

- `build_nopsh()` 점유 = 90 기대(불일치 시 중단). (실행 로그에서 확인)

---

## 5. 본 학습 실행 방식 (tmux 보호)

```bash
# Claude가 detached 기동 (cwd=qcbm)
tmux new-session -d -s qcbm_rerun "bash -lc '...; python3 -u tools/rerun_no_psh_restart_auc.py \
  2>&1 | tee results2/no_psh_restart_auc_train.log; echo EXIT_CODE=$?'"
```
사용자 표준 절차:
```bash
cd /home/elicer/IOTJ/qcbm
tmux new -s qcbm_rerun
python3 -u tools/rerun_no_psh_restart_auc.py 2>&1 | tee results2/no_psh_restart_auc_train.log
# Ctrl+b 후 d / tmux attach -t qcbm_rerun / tmux kill-session -t qcbm_rerun
```

---

## 6. 결정론 검증 (완료)

- best(최소 MMD², seed15) AUC = **0.8022966** → 기대 0.8023과 **일치(±0.001 OK)**. 결정론 재현 확인. best_tv 0.0622, MMD² 6.66e-5도 이전과 동일.

---

## 7. 16 restart AUC 분포 (완료) — ★ 핵심

| 지표 | 값 |
|------|-----|
| **emp_joint(0.7773) 초과** | **4/16** (seed 13/10/15/6) |
| **marginal(0.5136) 초과** | **14/16** |
| AUC min / median / max / std | 0.344 / **0.715** / 0.818 / 0.143 |
| best by MMD²(seed15) AUC | 0.8023 (재현) |
| max AUC | 0.8179 (seed13, best-MMD²와 다른 restart) |

### restart별 (AUC 내림차순)
| seed | AUC | TV | 비고 |
|------|-----|-----|------|
| 13 | 0.8179 | 0.107 | >emp |
| 10 | 0.8152 | 0.106 | >emp |
| 15 | 0.8023 | 0.062 | >emp (best by MMD²) |
| 6 | 0.7851 | 0.111 | >emp |
| 7 | 0.7666 | 0.111 | >marg |
| 3 | 0.7652 | 0.110 | >marg |
| 1 | 0.7619 | 0.120 | >marg |
| 2 | 0.7287 | 0.114 | >marg |
| 5 | 0.7012 | 0.267 | >marg |
| 14 | 0.6843 | 0.105 | >marg |
| 8 | 0.6324 | 0.109 | >marg |
| 11 | 0.5817 | 0.102 | >marg |
| 0 | 0.5755 | 0.101 | >marg |
| 9 | 0.5658 | 0.109 | >marg |
| 12 | 0.3784 | 0.112 | <marg (AUC 실패) |
| 4 | 0.3440 | 0.100 | <marg (AUC 실패) |

---

## 8. 사실 관찰 (판정 아님 — 웹 AI)

1. **결정론 재현 OK** (best 0.8023 일치).
2. **"QCBM > marginal(0.5136)"은 견고: 14/16**. 순수 상호작용 학습이 다수 restart에서 marginal을 넘음. (예외 2개 seed4/12는 TV는 좋은데 AUC 0.34/0.38로 붕괴 — 아래 4번.)
3. **"QCBM > emp_joint(0.7773)"은 견고하지 않음: 4/16(25%)**. **median AUC 0.715 < emp_joint 0.777**. 즉 전형적 restart는 emp_joint 상한을 **넘지 못함**. best(0.8023)가 상한을 넘은 것은 **유리한 선택의 영향**이 큼 → 직전 보고의 "QCBM이 상한을 일반화로 초과(103%)"는 **단일 best 기준이었고, 견고한 효과로 일반화하기 어렵다**(정정).
4. **TV(학습목표 적합)와 탐지 AUC의 해리(dissociation)**: 거의 모든 restart가 TV~0.10로 분포는 비슷하게 잘 학습했으나 **AUC는 0.34~0.82로 폭넓게 산포(std 0.143)**. seed4는 TV 0.100(좋은 적합)인데 AUC 0.344(무작위 이하). → **MMD²/TV 기반 모델선택(best=최소 MMD²)이 탐지 AUC를 최적화하지 못함**. best-MMD²(seed15, 0.802)는 AUC 3위, 최고 AUC는 seed13(0.818).

종합(사실): psh 통제 후 **"joint>marginal을 QCBM이 학습"은 견고(14/16)**. 그러나 **"QCBM이 emp_joint 상한을 일반화로 넘는다"는 견고하지 않음(4/16, median<상한)** — 직전의 강한 주장은 약화되어야 함. 또한 **분포 적합(TV)과 탐지(AUC)가 약상관**이라 현재 MMD 학습+MMD-best 선택이 탐지에 최적이 아님.

---

## 9. 저장 파일 (16 p_final 포함)

- `results2/qcbm_iat_no_psh_restarts.npz` (692KB): **16 p_final(16×2048)** + seeds + aucs + tvs + mmd2 + q_train + st_te + st_atk → 향후 AUC 재분석 시 재학습 불필요.
- `results2/no_psh_restart_auc.json`: determinism, restart_auc(16), auc_stats, n_exceed_emp_joint/marginal.
- `results2/no_psh_restart_auc.png`: 16 restart AUC 막대(emp_joint/marginal/best 기준선).
- `results2/no_psh_restart_auc_train.log`.
- `tools/rerun_no_psh_restart_auc.py` (신규).

---

## 10. git diff --stat (기존 소스 무변경)

- `qcbm/` 미추적 → 비어 있음. experiment2.py(mtime Jun 24)/train_qcbm_iat_no_psh.py 등 **무수정**(import + 모듈속성 대입만).

---

## 11. 성공 기준 / 남은 불확실성 / 다음 추천

충족: 결정론 best 0.8023 재현 ✅, 16 AUC 계산 + emp_joint/marginal 초과 수 ✅, 분포 통계·best 위치 ✅, 16 p_final npz 저장 ✅, json/png ✅, 무수정·무설치 ✅.

불확실/주의:
1. **TV↔AUC 해리**의 원인 미규명(왜 같은 TV에서 AUC가 0.34~0.82인지) — 분포의 어느 부분(공격이 떨어지는 상태)의 미세 차이가 AUC를 좌우하는 듯.
2. emp_joint 초과가 4/16라 "일반화 우위"는 약함. best 선택을 MMD²가 아니라 (held-out 일부) AUC로 하면 달라지나, 그건 평가 누수 위험.

다음 추천:
1. (서사 정정) 논문 주장을 **"QCBM이 순수 상호작용을 학습해 marginal을 견고히(14/16) 넘는다"** 로 한정. "emp_joint 상한 초과"는 best 사례로만 언급하고 **일반화 우위로 과대주장하지 말 것**(median<상한).
2. (분석) TV↔AUC 해리를 파고들기: npz의 16 p_final로 "공격이 떨어지는 상태들의 확률"이 AUC와 어떻게 연동되는지(재학습 없이 npz만으로) 분석.
3. (방법) 탐지가 목적이면 MMD-best 선택보다 탐지 지향 선택/목표가 필요할 수 있음(연구 프레임 논의).

---

## 최종 보고

```markdown
## 작업 요약
- psh 제거 11비트를 결정론 재현(best 0.8023 일치 ✅)하고 16 restart 각각의 AUC를 계산·저장(npz).
  결과: QCBM>marginal(0.5136)은 14/16로 견고. 그러나 QCBM>emp_joint(0.7773)은 4/16뿐, median AUC 0.715<상한
  -> "상한 일반화 초과"는 견고하지 않음(직전 best기준 주장 약화). TV~0.10 고정인데 AUC 0.34~0.82 산포(MMD-best 선택이 AUC 최적 아님).
  기존 소스 무수정.

## 본 학습 실행 방법 (tmux)
cd /home/elicer/IOTJ/qcbm
tmux new -s qcbm_rerun
python3 -u tools/rerun_no_psh_restart_auc.py 2>&1 | tee results2/no_psh_restart_auc_train.log

## 결정론 검증
- best(seed15) AUC=0.80230 = 0.8023 일치(±0.001). 재현 OK.

## 16 restart AUC 분포
- emp_joint(0.7773) 초과: 4/16 (seed 13,10,15,6)
- marginal(0.5136) 초과: 14/16
- AUC min/median/max/std = 0.344 / 0.715 / 0.818 / 0.143
- best by MMD²(seed15)=0.8023 (AUC 3위), max AUC=0.818(seed13)

## 사실 관찰 (판정 아님)
- QCBM>marginal: 견고(14/16). QCBM>emp_joint: 비견고(4/16, median<상한) -> 일반화 초과 주장 약화.
- TV(적합)와 AUC(탐지) 해리: 같은 TV~0.10에서 AUC 0.34~0.82. MMD-best 선택이 탐지를 최적화 못함.

## 저장 파일 (16 p_final 포함)
- results2/qcbm_iat_no_psh_restarts.npz(16 p_final+states), no_psh_restart_auc.{json,png}, tools/rerun_no_psh_restart_auc.py

## git diff --stat
- 비어 있음(미추적). 기존 소스 무수정.

## 다음 추천 단계
1. 서사 정정: "QCBM이 순수 상호작용 학습으로 marginal을 견고히(14/16) 넘는다"로 한정. emp_joint 초과는 과대주장 금지.
2. npz로 TV↔AUC 해리 원인 분석(재학습 없이). 탐지 지향 선택/목표 논의.
```
