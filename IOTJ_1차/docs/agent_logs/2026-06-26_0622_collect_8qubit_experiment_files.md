# 8큐빗 QCBM 실험 핵심 파일 수집 및 셋업 확인 (조사 전용)

- **작성일시**: 2026-06-26 06:22 (로컬)
- **작성 주체**: 로컬 Claude Code agent (조사 전용 모드)
- **작업 성격**: 코드 수정 없음. 본 `.md` 로그 1개만 생성.

---

## 1. 작업 목적

웹 AI(연구 총괄)가 외부 비교 논문(CMC, IEEE/ACM PDE) 검증을 위해 다음 3가지를 코드 근거로 확정 요청:

1. 기존 8큐빗 실험의 **손실 함수가 MMD(joint 기반)인지, marginal MSE인지**
2. 8큐빗 **AUC ≈ 0.43**이 어떤 셋업·어떤 공격 대상이었는지
3. AUC 계산에서 **score direction selection(방향 자동 반전)** 을 하는지 여부 (0.43이 0.5 미만인 원인 후보)

→ 결론: **3가지 모두 확정됨** (아래 §4~§5). 손실 = MMD²(joint), AUC 0.43 = 8q/L=3/Infiltration, direction selection 없음.

---

## 2. 시작 시점 git status 요약

```
On branch main (up to date with origin/main)
Untracked files:
	qcbm/          <- QCBM 실험 코드/결과 전체가 아직 미추적 상태
nothing added to commit but untracked files present
```

- `qcbm/` 디렉토리 전체가 git에 미추적(untracked). 추적된 변경/스테이징 없음.
- 본 작업 종료 후 git diff는 **이 `.md` 로그 파일 생성 외에 비어 있어야 함** (성공 기준 충족).
- 기존 변경사항 보존됨 (건드린 소스 없음).

---

## 3. 실제로 확인한 파일과 경로

### 저장소 구조 (상위)
```
/home/elicer/IOTJ
├── analyze_csv.py / analyze_iscx.py / analyze_pcap.py   (ISCX 데이터 분석, QCBM과 무관)
├── ISCX-VPN-Dataset/ , samples/ , figs/                 (별도 데이터/그림)
└── qcbm/                                                <- QCBM 실험 전체
    ├── experiment.py            (6큐빗 기준 실험)
    ├── experiment2.py           (6 vs 8큐빗 확장 + 이상탐지)  ★8q 핵심
    ├── depth_test.py            (8q L=3/6/9 깊이 비교)
    ├── anomaly_redo.py          (8q L=3 vs L=6 이상탐지 재평가) ★AUC 재검증
    ├── run_experiment.sh / run_experiment2.sh / run_anomaly_redo.sh
    ├── run.log / run2.log / depth.log / anomaly_redo.log
    ├── data/cic_thursday.csv    (102MB, CSE-CIC-IDS2018 Thursday)
    ├── results/                 (6큐빗 experiment.py 결과)
    └── results2/                (8큐빗 + 이상탐지 결과)  ★
```

### "확인 필요" 1~6 매핑

| # | 항목 | 파일·위치 |
|---|------|-----------|
| 1 | QCBM 회로/모델 정의 | [experiment2.py:140-151](../../qcbm/experiment2.py#L140-L151) `make_circuit()`, `wshape()` |
| 2 | 학습 손실 정의 | [experiment2.py:188-190](../../qcbm/experiment2.py#L188-L190) `cost(ww)` (+ 커널 [127-134](../../qcbm/experiment2.py#L127-L134)) |
| 3 | 학습 루프 | [experiment2.py:175-227](../../qcbm/experiment2.py#L175-L227) `train_one()`, [230-260](../../qcbm/experiment2.py#L230-L260) `run_config()` |
| 4 | 데이터 로딩/인코딩 | [experiment2.py:76-116](../../qcbm/experiment2.py#L76-L116) `load_frames()`, `discretize()`, `size_edges_from_benign()` |
| 5 | 평가/AUC 계산 | [experiment2.py:266-310](../../qcbm/experiment2.py#L266-L310) `auc_score()`, `roc_curve()`, `anomaly_eval()` |
| 6 | 8큐빗 결과 파일 | `qcbm/results2/` (§7 목록) |

---

## 4. 손실 함수 종류 + 코드 근거 ★

**결론: MMD² (joint 결합분포 기반), multi-bandwidth Gaussian kernel. marginal MSE 아님.**

### 코드 근거
- 비용 함수 [experiment2.py:188-190](../../qcbm/experiment2.py#L188-L190):
  ```python
  def cost(ww):
      p = circuit(ww)
      return p @ (K @ p) - 2.0 * (p @ qK) + qKq
  ```
  여기서 `qK = K @ q`, `qKq = q @ K @ q` (상수). 전개하면
  **`p·Kp − 2 p·Kq + q·Kq = (p − q)ᵀ K (p − q) = MMD²`**.
  → `p`, `q` 는 **256차원 전체 결합분포 확률벡터** (`qml.probs(wires=range(nq))`, [L146](../../qcbm/experiment2.py#L146)).
  → marginal(개별 feature별 분포)에 대한 MSE가 **아님**. 전체 joint 분포 간 MMD.

- 커널 [experiment2.py:127-134](../../qcbm/experiment2.py#L127-L134) `build_kernel()`:
  상태를 비트벡터로 펼친 뒤 제곱유클리드(=해밍) 거리 `d2`에 대해
  σ ∈ {0.5, 1, 2, 4} 가우시안 합/평균 → multi-bandwidth Gaussian MMD 커널.

- `experiment.py`(6큐빗)도 동일 구조 ([experiment.py:148-150](../../qcbm/experiment.py#L148-L150), [211-213](../../qcbm/experiment.py#L211-L213)). docstring에도 "손실: MMD^2 ... 전체 확률벡터로 정확 계산" 명시.

> **웹 AI 추정과의 차이**: "marginal MSE 가능성" 추정이 있었으나, 실제 코드는 **joint MMD²** 임. marginal MSE는 어디에도 없음. (marginal은 [experiment.py:248-254](../../qcbm/experiment.py#L248-L254) `decode_marginals()`에서 **시각화용**으로만 쓰이고 손실에는 안 들어감.)

---

## 5. AUC 계산 방식 + direction selection 여부 + 코드 근거 ★

### AUC 계산 방식
[experiment2.py:266-270](../../qcbm/experiment2.py#L266-L270) `auc_score()`:
```python
def auc_score(scores, labels):
    """labels: 1=공격(positive). Mann-Whitney U 기반 AUC."""
    r = rankdata(scores)
    n1 = labels.sum(); n0 = len(labels) - n1
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
```
- 점수: [anomaly_eval, L287-292](../../qcbm/experiment2.py#L287-L292) **`s(x) = -log p(x)`** (이상점수; 높을수록 비정상).
- positive = **공격(Infiltration), label=1**. 정상(benign test) = 0.
- Mann–Whitney U 통계량 그대로. **고정 방향** (공격이 높은 점수=낮은 확률을 받을 것이라는 가설).

### Direction selection(방향 자동 반전) 여부
**결론: 없음(NO). 방향 자동 반전·`max(auc, 1-auc)`·`abs` 같은 보정이 전혀 없음.**

- `auc_score()`는 raw Mann–Whitney U만 반환. `1-auc` 비교나 score 부호 자동선택 코드 없음.
- 따라서 **AUC 0.43 < 0.5 는 진짜(real) 역신호**: 공격 샘플이 오히려 **더 높은 확률(더 정상스러운) 상태**에 떨어짐.
  - 근거 수치 (results2/summary2.json, 8q): `mean_p_atk=0.0452 > mean_p_ben=0.0353`, `mean_logp_atk=-3.58 > mean_logp_ben=-3.89`.
  - 즉 Infiltration 공격이 benign과 분리 안 될 뿐 아니라 학습된 정상분포의 고확률 영역에 더 몰림 → AUC가 0.5 미만.

> **0.43이 0.5 미만인 원인**: direction selection 누락 때문이 **아님**(애초에 그런 로직이 없음). 실제로 공격이 정상분포 고확률 영역에 위치하는 데이터 특성 때문. (방향 반전을 넣었다면 0.57로 보일 수 있으나, 현재 코드는 정직하게 0.43으로 보고.)

---

## 6. 데이터셋 / 인코딩 / 큐빗 매핑 요약

- **데이터셋**: CSE-CIC-IDS2018, `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`
  - 로컬: `qcbm/data/cic_thursday.csv` (≈102MB, 실데이터 존재).
  - 라벨: `benign` (학습/정상), `infilteration` (공격/이상). [experiment2.py:83-86](../../qcbm/experiment2.py#L83-L86)
  - 분할: seed=7, benign 70% train / 30% test. 공격 전체는 평가용. [experiment2.py:376-381](../../qcbm/experiment2.py#L376-L381)
- **8큐빗(256상태) 인코딩** [experiment2.py:54-59](../../qcbm/experiment2.py#L54-L59):
  - `proto`(2bit: TCP=0/UDP=1/proto0=2/기타=3) + `syn`(1bit: SYN Flag Cnt>0) + `psh`(1bit: PSH Flag Cnt>0) + `size`(4bit: Pkt Len Mean 16버킷)
  - 비트 패킹: 나열 순서대로 MSB→LSB, `state = (state<<bits)|v` [experiment2.py:113-116](../../qcbm/experiment2.py#L113-L116)
  - size 버킷 경계: **benign train의 분위수**로 생성 (이산화기를 train에 적합, 양쪽 공유) [experiment2.py:90-95](../../qcbm/experiment2.py#L90-L95)
  - 6큐빗(64상태)은 size 3bit(8버킷), psh 없음.
- **모델**: `lightning.qubit`, `qml.StronglyEntanglingLayers` (회전층+CNOT 얽힘), `diff_method="parameter-shift"`, `qml.probs`. L(layers)=3 기본. [experiment2.py:140-151](../../qcbm/experiment2.py#L140-L151)
- **학습**: 수동 Adam(b1=.9,b2=.999), lr=0.1, epochs=300, 16 무작위 재시작 프로세스 병렬, best=최소 MMD². epoch별 gradient norm 추적(바렌플래토 진단). [experiment2.py:175-227](../../qcbm/experiment2.py#L175-L227)

### 8큐빗 핵심 수치 (확정)
| 셋업 | best MMD² | best TV | AUC (Infil.) | 출처 |
|------|-----------|---------|--------------|------|
| 8q L=3 | 1.09e-2 | 0.498 (underfit) | **0.4333** | results2/summary2.json, anomaly_redo.json("3") |
| 8q L=6 | 6.74e-4 | 0.141 (잘학습) | **0.4289** | results2/anomaly_redo.json("6") |
| 8q L=9 | 1.31e-3 | 0.186 | (이상탐지 미평가) | results2/depth_test.json |
| (6q L=3) | 4.5e-4 | 0.065~0.093 | 0.4104 | results/summary.json, summary2.json |

- **AUC 0.43의 정체**: experiment2.py 8q **L=3** 셋업, 공격대상 = **Infiltration**. anomaly_redo.py가 L=6(제대로 학습)로 재평가해도 **AUC=0.4289**로 여전히 0.5 미만 → ΔAUC=−0.0044. 즉 "얕아서 안 된 것"이 아니라 분포를 잘 학습해도(TV 0.14) Infiltration은 분리 안 됨(코드 결론도 동일, [anomaly_redo.py:175-178](../../qcbm/anomaly_redo.py#L175-L178)).

---

## 7. 8큐빗 실험 결과 파일 경로 목록

### `qcbm/results2/` (8큐빗 + 이상탐지)
- `summary2.json` — 6q/8q(L=3) 학습+이상탐지 요약 (AUC 0.4333 기록)
- `anomaly_redo.json` — 8q L=3 vs L=6 이상탐지 재평가 (AUC 0.4333 vs 0.4289, ΔAUC)
- `depth_test.json` — 8q L=3/6/9 깊이별 MMD²/TV/gradient
- `train_8q.png`, `train_8q_L3.png`, `train_8q_L6.png` — 손실곡선 + gradient-norm(바렌플래토)
- `dist_8q.png`, `dist_8q_L3.png`, `dist_8q_L6.png` — 256상태 joint 분포 real vs QCBM
- `anomaly_8q.png`, `anomaly_8q_L3.png`, `anomaly_8q_L6.png` — 이상점수 히스토그램 + ROC
- `anomaly_L3_vs_L6.png` — L=3 vs L=6 이상탐지 종합 비교
- (6q 대응물: `train_6q.png`, `dist_6q.png`, `anomaly_6q.png`)

### `qcbm/results/` (6큐빗 experiment.py)
- `summary.json`, `loss_curve.png`, `dist_compare.png`, `marginals.png`, `best_weights.npy`, `best_p.npy`, `target_q.npy`

### 로그
- `qcbm/run2.log` (experiment2 실행 로그), `qcbm/depth.log`, `qcbm/anomaly_redo.log`, `qcbm/run.log`(6q)

---

## 8. 웹 AI 공유 후보 파일 목록

| 경로 | 왜 필요한지 | 민감정보 포함 |
|------|-------------|----------------|
| `qcbm/experiment2.py` | 8q 회로/손실(MMD²)/학습/이산화/AUC 전부 정의. 비교논문 검증의 1차 근거 | 없음 (코드만) |
| `qcbm/anomaly_redo.py` | L=3 vs L=6 이상탐지 재평가 로직. AUC<0.5 정직성 확인 | 없음 |
| `qcbm/depth_test.py` | 8q 깊이별(L=3/6/9) 학습품질. underfit 원인 규명 | 없음 |
| `qcbm/results2/summary2.json` | 8q L=3 AUC 0.4333 등 핵심 수치 | 없음 (집계 통계만) |
| `qcbm/results2/anomaly_redo.json` | 8q L=6 AUC 0.4289, ΔAUC | 없음 |
| `qcbm/results2/depth_test.json` | 깊이별 MMD²/TV/gradient | 없음 |
| `qcbm/results2/*.png` (train/dist/anomaly/ROC) | 그래프(MMD², gradient norm, ROC) 시각 근거 | 없음 (집계 그림) |
| `qcbm/run2.log`, `depth.log`, `anomaly_redo.log` | 실행 시점 정확 수치/환경 재현 | 없음 (트래픽 통계 로그) |
| `qcbm/experiment.py` + `qcbm/results/*` | 6q 기준선(비교용) | 없음 |
| ~~`qcbm/data/cic_thursday.csv`~~ | **공유 비권장** | **있음**: 원시 네트워크 트래픽(102MB). 공개 데이터셋이나 대용량+원시 플로우 → 코드/집계만 공유 권장 |

> 권장: **코드(.py) + 집계 결과(.json/.png/.log)** 만 공유. 원시 CSV는 제외(용량+원시 트래픽). CSV는 코드의 `DATA_URL`로 누구나 재다운로드 가능.

---

## 9. 아직 불확실한 점

1. **8q L=9의 이상탐지 AUC 미측정**: depth_test는 학습품질(MMD²/TV)만 보고, AUC는 안 구함. L=9에서 이상탐지가 바뀌는지는 미확인.
2. **AUC<0.5의 데이터 차원 원인 미규명**: 공격이 고확률 영역에 몰리는 게 (a) 4개 feature(proto/syn/psh/size)만으로는 Infiltration이 benign과 구분 안 되어서인지, (b) 분위수 이산화가 공격 특성을 뭉개서인지 코드만으로는 단정 불가.
3. **공격 표본 수/점유 상태 수**: 정확한 attack 샘플 개수는 `run2.log`에 출력되나 본 조사에서 로그 본문 수치까지는 대조 안 함(필요 시 grep).
4. **다른 공격 유형 미검토**: Infiltration만 평가. CSE-CIC-IDS2018의 타 공격(DDoS 등)에서는 AUC가 다를 수 있음.

---

## 10. 다음 추천 작업

1. 웹 AI에게 §8의 **코드+집계 결과**를 GitHub push 또는 Google Drive 업로드로 공유 (원시 CSV 제외).
2. (선택) 8q L=9 또는 더 깊은 모델에서 **이상탐지 AUC 추가 측정** — 깊이-이상탐지 관계 완결.
3. (선택) Infiltration 외 **다른 공격 유형**으로 AUC 비교 → "QCBM 이상탐지가 공격 유형에 의존하는가" 검증.
4. (선택) AUC 방향성 정직 보고를 위해, 논문 비교 시 **direction selection 미적용**을 명시 (0.43은 raw, 반전 시 0.57로 표기 가능하나 현재 코드 기준은 0.43).

---

## 최종 보고 (요청 형식)

```markdown
## 작업 요약
- 8큐빗 QCBM 실험의 손실=MMD²(joint), AUC 0.43=8q/L=3/Infiltration, direction selection 없음을 코드 근거로 확정. 소스 무수정, 본 .md 1개만 생성.

## 확인한 핵심 파일 (경로)
- QCBM 모델 정의: qcbm/experiment2.py:140-151 (make_circuit, StronglyEntanglingLayers, lightning.qubit)
- 손실 정의 (MMD/marginal MSE + 근거): qcbm/experiment2.py:188-190 cost = (p-q)ᵀK(p-q) = MMD²(joint). marginal MSE 아님. 커널 L127-134.
- 학습 루프: qcbm/experiment2.py:175-227 train_one / 230-260 run_config (Adam lr=0.1, 300ep, 16재시작, L=3)
- 데이터 로딩/인코딩: qcbm/experiment2.py:76-116 (CSE-CIC-IDS2018, proto2+syn1+psh1+size4=8bit/256상태, benign분위수 이산화)
- 평가/AUC 계산 (direction selection 여부 + 근거): qcbm/experiment2.py:266-310. s(x)=-log p(x), attack=positive, raw Mann-Whitney U. 방향 자동반전 없음 → 0.43<0.5는 진짜 역신호.

## 8큐빗 실험 결과 파일 경로
- qcbm/results2/summary2.json (8q L=3 AUC 0.4333), anomaly_redo.json (L=6 AUC 0.4289), depth_test.json
- qcbm/results2/{train,dist,anomaly}_8q*.png, anomaly_L3_vs_L6.png
- qcbm/run2.log, depth.log, anomaly_redo.log

## 웹 AI 공유 후보 파일 목록
- experiment2.py / anomaly_redo.py / depth_test.py — 코드 근거, 민감정보 없음
- results2/*.json, *.png, *.log — 집계 결과, 민감정보 없음
- data/cic_thursday.csv — 공유 비권장(원시 트래픽 102MB, DATA_URL로 재다운로드 가능)

## 아직 불확실한 점
- 8q L=9 이상탐지 AUC 미측정 / AUC<0.5의 데이터 차원 원인 미규명 / Infiltration 외 공격 미검토

## 다음 추천 단계
1. 코드+집계결과만 GitHub/Drive로 웹 AI에 공유 (원시 CSV 제외)
2. 깊은 모델 이상탐지 AUC 추가 측정, 타 공격유형 비교
```
