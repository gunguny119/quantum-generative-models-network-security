# 고전 baseline 이상탐지 AUC 진단 (emp_joint + marginal vs QCBM)

- **작성일시**: 2026-06-26 07:10 (로컬), 실행 07:12
- **작성 주체**: 로컬 Claude Code agent (default mode)
- **작업 성격**: 새 진단 스크립트 1개 추가 + 결과 json/png 생성. **QCBM 새 학습 없음. experiment2.py 무수정.**

---

## 1. 작업 목적

QCBM(8q, MMD, joint 256상태)의 Infiltration AUC ≈ 0.43(역신호, Youden J 0.03)이
**"양자가 못해서"인지, "이 데이터·이 feature로는 분포 기반 탐지가 원래 안 되는지"** 를 가리기 위해,
**새 학습 없이** 동일한 데이터·인코딩·train/test 분할·평가로 고전 baseline 2종의 AUC를 계산해 QCBM과 비교.

- (a) **emp_joint**: benign train의 256상태 경험적 분포 — 분포 기반 탐지가 낼 수 있는 **사실상의 상한**
- (b) **marginal**: 각 비트의 P(bit=1)만으로 만든 독립(곱) 분포 — CMC 논문 방식의 핵심
- (c) **QCBM L3/L6**: 기존 `results2/anomaly_redo.json` 수치를 표에 함께 표시 (새 학습 X)

---

## 2. 시작 시점 git status 요약

```
?? docs/      (이전 조사 로그 + 본 로그)
?? qcbm/      (QCBM 코드/결과 전체가 여전히 미추적)
git diff --stat: (비어 있음 — 추적 파일 변경 없음)
```
- `qcbm/` 전체가 git 미추적이므로 `git diff --stat`은 비어 있음. experiment2.py를 포함해 **기존 파일 무수정**.
- 본 작업이 추가한 파일: `qcbm/tools/diagnose_classical_baseline.py`, `qcbm/results2/classical_baseline.json`, `qcbm/results2/classical_baseline.png`, 본 `.md`.

---

## 3. experiment2.py에서 재사용한 함수 (실제 시그니처)

코드에서 직접 확인 (가정 아님):

| 함수 | 시그니처 | 반환/동작 |
|------|----------|-----------|
| `load_frames()` | `() -> (benign_df, attack_df)` | DATA="data/cic_thursday.csv" 로드. 라벨 소문자화 후 `benign`/`infilteration` 분리 ([L76-87](../../qcbm/experiment2.py#L76-L87)) |
| `size_edges_from_benign(benign, nbuckets)` | `-> inner edges (np.ndarray)` | benign `Pkt Len Mean` 분위수 경계 ([L90-95](../../qcbm/experiment2.py#L90-L95)) |
| `discretize(df, spec, size_edges)` | `-> states (int array)` | feature를 MSB→LSB로 비트패킹 ([L98-116](../../qcbm/experiment2.py#L98-L116)) |
| `empirical_dist(states, nstates)` | `-> q (np.ndarray, 합=1)` | `bincount/합` ([L119-121](../../qcbm/experiment2.py#L119-L121)) |
| `auc_score(scores, labels)` | `-> float` | raw Mann-Whitney U, label 1=attack=positive, **방향반전 없음** ([L266-270](../../qcbm/experiment2.py#L266-L270)) |
| `roc_curve(scores, labels, npts=200)` | `-> (fpr, tpr)` | ([L273-282](../../qcbm/experiment2.py#L273-L282)) |
| `anomaly_eval(cfg_out, benign_test_states, attack_states, tag)` | `-> dict(auc, mean_p_*, mean_logp_*, youden_j, s_ben, s_atk, fpr, tpr)` | **오직 `cfg_out["best"]["p_final"]` 만 사용**. `s(x)=-log p(x)` ([L285-310](../../qcbm/experiment2.py#L285-L310)) |

> **확인된 사실**: `anomaly_eval`은 `final_mmd2`/`tv`를 참조하지 **않음**. 따라서 baseline 분포를
> `{"best": {"p_final": q.tolist()}}` 로만 감싸도 NaN 키 에러 없이 동일 평가가 됨 → 실행으로 검증됨(에러 없음).
> auc_score/roc_curve 직접 호출로 우회할 필요 없었음.

---

## 4. marginal 비트 매핑

`discretize`의 패킹은 features 순서 `[("proto",2),("syn",1),("psh",1),("size",4)]` 대로 MSB→LSB:
```
state = ((((proto<<1)|syn)<<1|psh)<<4)|size  =  proto*64 + syn*32 + psh*16 + size
```
→ 8비트 위치(bit0=LSB … bit7=MSB):

| bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|-----|---|---|---|---|---|---|---|---|
| feature | proto[hi] | proto[lo] | syn | psh | size b3 | size b2 | size b1 | size b0 |

marginal 분포: benign train 상태를 위 8비트로 분해 → 각 비트 `P(bit=1)` 추정 →
`P(state)=∏_b [p_b if bit_b=1 else 1-p_b]` → 합=1 정규화.

실측 `P(bit=1)` [bit7→bit0]:
`[0.0253, 0.3554, 0.0653, 0.3961, 0.3127, 0.3124, 0.3789, 0.6450]`
(proto[hi] 거의 0 = proto 코드가 대부분 0/1(TCP/UDP)임과 일치; psh(bit4)=0.40, syn(bit5)=0.065.)

---

## 5. AUC 비교표 (실행 결과)

동일 데이터·인코딩·분할(seed 7, benign 0.7/0.3)·평가(`s=-log p`, attack=positive, raw MWU AUC):

| 모델 | AUC(raw) | 1-AUC | Youden J | mean logp ben | mean logp atk | 격차(ben-atk) |
|------|----------|-------|----------|---------------|---------------|---------------|
| **emp_joint** | **0.4283** | 0.5717 | 0.0304 | -2.955 | -2.794 | -0.161 |
| **marginal** | **0.4383** | 0.5617 | 0.0304 | -4.240 | -3.997 | -0.243 |
| QCBM L=3 | 0.4333 | 0.5667 | 0.0304 | -3.895 | -3.582 | -0.313 |
| QCBM L=6 | 0.4289 | 0.5711 | 0.0304 | -3.145 | -2.936 | -0.208 |

샘플 수: benign train=166,625 / test=71,412 / attack(Infilteration)=93,063. 256상태 중 50개 점유.

---

## 6. 실행한 명령어와 결과

```bash
python3 -m py_compile tools/diagnose_classical_baseline.py   # PY_COMPILE_OK
# (experiment2.py 는 건드리지 않았으므로 컴파일/실행 불필요)
python3 -u tools/diagnose_classical_baseline.py              # 정상 완료(DONE), 에러 없음
```
- 환경: `python3` (3.10.12), `~/.local/bin` PATH, OMP/OPENBLAS/MKL/NUMEXPR=1 (기존 런처와 동일).
- 원시 CSV(`data/cic_thursday.csv`, 102MB) 존재 → 전체 실행 성공.
- 출력: `results2/classical_baseline.json`, `results2/classical_baseline.png` 생성.
- **실패/우회 없음.** anomaly_eval 재사용이 NaN 키 문제 없이 동작 → 직접 auc_score 호출 불필요.

---

## 7. git diff --stat 요약 (experiment2.py 무수정 확인)

- `git diff --stat`: **비어 있음** (추적 파일 변경 0). `qcbm/` 전체가 미추적이라 추적상 diff 자체가 없음.
- 본 작업이 만든/만지 파일: **신규 3개만** (`tools/diagnose_classical_baseline.py`, `results2/classical_baseline.json`, `results2/classical_baseline.png`) + 본 `.md`.
- **experiment2.py는 읽기만 했고 1바이트도 수정하지 않음** (mtime Jun 24 유지). 손실/회로/학습/anomaly_eval/auc_score 모두 그대로.

---

## 8. 해석에 대한 사실만 (단정적 결론은 웹 AI가)

- 네 모델의 AUC가 모두 **0.428~0.438** 의 좁은 구간에 몰림. **emp_joint(0.4283)** 가 가장 낮고(즉 가장 역신호), marginal(0.4383)이 약간 높음.
- emp_joint는 "benign train 빈도를 그대로 외운" 분포 = **분포 기반 탐지의 사실상 상한**인데도 AUC 0.43 → 0.5 미만.
- 즉 QCBM(L=3 0.4333, L=6 0.4289)은 emp_joint 상한과 **사실상 동일** (±0.005). QCBM이 상한보다 못한 게 아님.
- marginal(0.4383)이 joint(0.4283)보다 약간 **덜 역신호** → 이 데이터에선 joint 의존성을 살릴수록 오히려 공격이 더 "정상스럽게" 보임(격차 음수가 더 커짐). "joint 가설"의 이득이 (적어도 이 4 feature에선) 음(−)으로 나타남.
- 네 모델 모두 Youden J=0.0304로 동일 → 어떤 분포든 임계값 기반 분리력이 거의 없음(무작위 근처).
- 부호: 모든 모델에서 `mean logp(attack) > mean logp(benign)` → **공격이 학습된 정상분포의 고확률 영역에 더 몰림** (AUC<0.5의 일관된 원인). 이는 모델 종류와 무관한 **데이터·feature 차원의 현상**.

(위는 수치가 말하는 사실만 기록. "한계 지도화" 최종 판정은 웹 AI 몫.)

---

## 9. 아직 불확실한 점

1. **4 feature(proto/syn/psh/size)의 한계인지, 분위수 이산화의 한계인지** 미분리. 다른 feature/이산화에서 AUC가 0.5를 넘는지는 미검증.
2. **emp_joint의 256상태 중 50개만 점유** → train에 없던 상태(공격이 떨어질 수 있는)는 모두 eps 확률로 동률 처리됨. 이 희소성이 AUC를 0.5 쪽으로 끌었을 가능성 미정량.
3. **Infiltration 단일 공격**만 평가. 타 공격(DDoS, Bruteforce 등)에서 baseline AUC가 다른지 미확인.
4. QCBM ROC 좌표가 json에 저장돼 있지 않아 `classical_baseline.png`에는 QCBM ROC 곡선은 없고 AUC 텍스트만 표기(emp_joint/marginal ROC만 곡선).

---

## 10. 다음 추천 작업

1. (웹 AI 판단용) 이 표를 "한계 지도"의 1차 근거로 사용: 분포 기반 탐지의 상한(emp_joint)도 0.43이므로 **문제 자체의 한계** 가설이 강하게 지지됨.
2. (선택) feature 확장(예: flow duration, byte counts 등) 또는 이산화 변경으로 baseline AUC가 0.5를 넘는지 탐색 → "feature 한계 vs 방법 한계" 분리.
3. (선택) Infiltration 외 공격 유형으로 동일 baseline 진단 반복 → 공격 유형 의존성 지도화.
4. (선택) train 미점유 상태 처리(스무딩)에 따른 AUC 민감도 점검.

---

## 최종 보고

```markdown
## 작업 요약
- 새 학습 없이 동일 데이터·인코딩·분할·평가로 고전 baseline 2종(emp_joint, marginal) AUC를 계산해 QCBM과 비교.
  네 모델 모두 AUC≈0.43(0.428~0.438)로 동일. emp_joint(분포기반 상한)도 0.428 → QCBM이 상한보다 못한 게 아님.
  experiment2.py 무수정, QCBM 무학습.

## 재사용한 experiment2.py 함수 (시그니처)
- load_frames() -> (benign, attack)
- size_edges_from_benign(benign, nbuckets) -> inner edges
- discretize(df, spec, size_edges) -> states(int)
- empirical_dist(states, nstates) -> q(합1)
- anomaly_eval(cfg_out, benign_test_states, attack_states, tag) -> dict  (오직 best.p_final 사용)
- auc_score(scores, labels) -> raw MWU AUC (방향반전 없음)

## marginal 비트 매핑
- state = proto*64 + syn*32 + psh*16 + size ; bit7,6=proto / bit5=syn / bit4=psh / bit3..0=size
- P(bit=1)[7..0]=[0.0253,0.3554,0.0653,0.3961,0.3127,0.3124,0.3789,0.6450]

## AUC 비교표
| 모델 | AUC(raw) | 1-AUC | Youden J | mean logp ben | mean logp atk | 격차 |
|------|----------|-------|----------|---------------|---------------|------|
| emp_joint | 0.4283 | 0.5717 | 0.0304 | -2.955 | -2.794 | -0.161 |
| marginal  | 0.4383 | 0.5617 | 0.0304 | -4.240 | -3.997 | -0.243 |
| QCBM L=3  | 0.4333 | 0.5667 | 0.0304 | -3.895 | -3.582 | -0.313 |
| QCBM L=6  | 0.4289 | 0.5711 | 0.0304 | -3.145 | -2.936 | -0.208 |

## 실행한 검증
- python3 -m py_compile tools/diagnose_classical_baseline.py  -> OK
- python3 -u tools/diagnose_classical_baseline.py             -> DONE (에러 없음, CSV 존재)

## 생성/수정한 파일
- tools/diagnose_classical_baseline.py (신규)
- results2/classical_baseline.json (신규), results2/classical_baseline.png (신규)
- experiment2.py 변경 여부: 변경 없음 (읽기만)

## 확인된 사실
- 분포기반 탐지 상한(emp_joint)도 AUC 0.428 -> QCBM(0.43)은 상한과 동일, 양자 열위 아님.
- marginal(0.438) > joint(0.428): 이 4 feature에선 joint 의존성이 분리에 도움 안 됨(오히려 -).
- 모든 모델 공통: attack이 정상분포 고확률 영역에 몰림 -> AUC<0.5는 데이터·feature 차원 현상.

## 아직 불확실한 점
- feature 한계 vs 이산화 한계 미분리 / 미점유 상태 희소성 영향 / Infiltration 단일 공격만 평가

## 다음 추천 단계
1. 이 표를 '한계 지도' 근거로 웹 AI에 전달 (문제 자체의 한계 가설 지지)
2. feature/이산화 확장 또는 타 공격유형으로 baseline AUC 재측정
```
