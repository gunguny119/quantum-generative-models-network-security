# 프로젝트 현황 점검 + 웹 AI 공유용 코드 zip 패키징

> **전제: 이 작업은 기존 파일을 일절 수정·이동·삭제·이름변경하지 않았다.** 새 파일(이 snapshot.md, zip_manifest.txt, data_headers.txt, 공유 zip)만 생성함. 학습/평가 실행 없음.

- **작성일시**: 2026-06-26 10:12 (로컬)
- **작성 주체**: 로컬 Claude Code agent (default mode, 조사·패키징)

---

## 1. 작업 목적

QCBM 연구의 다음 단계("고전 생성모델 vs QCBM 파라미터 효율 비교") 설계를 위해 웹 AI(연구 총괄)가 전체 코드 구조를 파악하도록:
(1) 프로젝트 현황을 사실 기반 요약, (2) 코드+작은 메타데이터만 담은 공유 zip 생성(대용량 데이터/체크포인트 제외, 경로·크기는 manifest에 기록). **파일 재구성 아님.**

---

## 2. 시작 시점 git status

```
?? docs/   ?? qcbm/   (둘 다 미추적; 추적 파일 변경 0)
```
- 보존 대상 무관한 변경 없음. 이번 작업 후에도 기존 파일 무변경(§13).

---

## 3. 프로젝트 루트 / git 여부

- 루트: **`/home/elicer/IOTJ`**, **git 저장소(.git 존재)**, 브랜치 main. `qcbm/`와 `docs/`는 아직 untracked.

---

## 4. 디렉토리 구조 + 용량

```
/home/elicer/IOTJ/                         (git repo)
├── analyze_csv.py / analyze_iscx.py / analyze_pcap.py   (ISCX 분석 코드, QCBM과 별개)
├── .gitignore
├── docs/agent_logs/        116K   작업 로그(.md) 10개 + 본 작업 산출물
├── figs/                   1.1M   분석 그림(PNG)
├── " ISCX-VPN-Dataset"/    17M    원시 pcap/zip (앞에 공백 있는 디렉토리명; QCBM 무관) [zip 제외]
├── samples/                17M    원시 pcap 샘플 (QCBM 미사용) [zip 제외]
└── qcbm/                   420M   ★ QCBM 연구 본체
    ├── experiment.py            6q 기준 실험 (StronglyEntanglingLayers, MMD)
    ├── experiment2.py           ★핵심: 6/8q 확장 + 이상탐지 (run_config/train_one/anomaly_eval 등)
    ├── anomaly_redo.py          8q L3 vs L6 이상탐지 재평가
    ├── depth_test.py            8q 깊이(L=3/6/9) 비교
    ├── run_experiment*.sh / run_anomaly_redo.sh   tmux 런처
    ├── run.log/run2.log/depth.log/anomaly_redo.log  실행 로그(텍스트, 수십 KB)
    ├── data/               417M  cic_thursday.csv(103M) + cic_0221_ddos.csv(314M) + dl.log(0) [zip 제외]
    ├── results/                  6q 결과(summary.json, *.png, 작은 *.npy)
    ├── results2/                 ★8q~12q 결과: JSON·PNG·train.log 다수
    └── tools/                    ★진단/학습 스크립트 8개 (이번 연구의 핵심 산출)
        ├── inspect_csv_columns.py        컬럼 분리력 조사
        ├── check_attack_labels.py        라벨 종류 확인
        ├── diagnose_classical_baseline.py  emp_joint vs marginal (Infiltration)
        ├── diagnose_ddos_baseline.py     DDoS emp_joint vs marginal
        ├── diagnose_iat_joint.py         IAT 추가 joint 이득(상한) A/B1/B2
        ├── train_qcbm_iat_b1.py          ★10q QCBM 학습
        ├── train_qcbm_iat_b2.py          ★12q QCBM 학습
        └── rescue_b2_from_log.py         B2 결과 로그 박제
```

---

## 5. 파일 분류 (zip 포함 vs 제외)

- **포함(코드+작은 메타데이터, 94개, 압축전 3.6MB)**: 모든 `.py`, `.sh`, `.md`, `.gitignore`, `results/`·`results2/` 의 JSON·PNG·train.log·smoke.json·작은 .npy, `csv_columns_report.csv`, qcbm root 실행 로그, `figs/` PNG, `data_headers.txt`.
- **제외(대용량/원시)**: `qcbm/data/*.csv`(103M+314M), `qcbm/data/dl.log`, `" ISCX-VPN-Dataset"/`(17M), `samples/`(17M, pcap). → manifest에 경로·크기·사유 기록. (5MB 초과 개별 파일은 모두 이 제외군에 속함.)

---

## 6. train/eval entrypoint (실제 확인)

- **6q**: `qcbm/experiment.py` (`main()`), 런처 `run_experiment.sh`.
- **6q/8q + 이상탐지**: `qcbm/experiment2.py` (`main()`), 런처 `run_experiment2.sh`.
- **8q 깊이/재평가**: `depth_test.py`, `anomaly_redo.py` (둘 다 `import experiment2 as E`).
- **10q/12q QCBM(IAT)**: `tools/train_qcbm_iat_b1.py`, `tools/train_qcbm_iat_b2.py` (smoke→full, tmux 실행).
- **진단(학습 없음)**: `tools/diagnose_*.py`, `inspect_csv_columns.py`, `check_attack_labels.py`.
- 모든 학습 스크립트는 `experiment2`의 함수를 import 재사용(중복 학습 로직 없음).

---

## 7. QCBM 핵심 함수 위치 (experiment2.py, 실제 확인)

| 기능 | 위치 |
|------|------|
| 회로 정의 | `make_circuit(nq, n_layers)` [experiment2.py:140-147](../../qcbm/experiment2.py#L140-L147) (lightning.qubit, StronglyEntanglingLayers, parameter-shift) |
| MMD 커널 | `build_kernel(nq, sigmas)` [experiment2.py:127-134](../../qcbm/experiment2.py#L127-L134) |
| 1개 재시작 학습 | `train_one(seed)` [experiment2.py:175-227](../../qcbm/experiment2.py#L175-L227) (수동 Adam, epoch별 gnorm) |
| 학습 오케스트레이션 | `run_config(spec, q_train, size_edges)` [experiment2.py:230-260](../../qcbm/experiment2.py#L230-L260) (16재시작 Pool 병렬, best=최소 MMD²) |
| 이상탐지/AUC | `anomaly_eval(cfg_out, ben_test, atk, tag)` [experiment2.py:285-310](../../qcbm/experiment2.py#L285-L310), `auc_score` [266-270](../../qcbm/experiment2.py#L266-L270)(raw MWU), `roc_curve` [273-282](../../qcbm/experiment2.py#L273-L282) |

---

## 8. TV / 파라미터 수 / q_train (실제 확인)

- **TV 계산**: `tv_dist(p, q) = 0.5*Σ|p−q|` [experiment2.py:160-161](../../qcbm/experiment2.py#L160-L161). `train_one`에서 best 분포 vs q_train에 적용. (KL은 `kl_div` [154-157](../../qcbm/experiment2.py#L154-L157).)
- **파라미터 수 결정**: `wshape(nq, n_layers)=StronglyEntanglingLayers.shape` [experiment2.py:150-151](../../qcbm/experiment2.py#L150-L151); `run_config`에서 `nparams=prod(wshape(nq,LAYERS))` [233]. (예: 10q/L6=180, 12q/L6=216.)
- **q_train(목표 분포) 생성**: `empirical_dist(states, nstates)` [experiment2.py:119-121](../../qcbm/experiment2.py#L119-L121). 상태 인코딩은 `discretize(df, spec, size_edges)` [98-116](../../qcbm/experiment2.py#L98-L116)(MSB→LSB 비트팩); IAT 비트는 학습 스크립트(`tools/train_qcbm_iat_b*.py`)가 `discretize` 결과 뒤에 자체 append(`log1p`+분위수). run_config는 q_train을 직접 받음(내부 재이산화 없음).

---

## 9. results2/ JSON 목록과 실험 의미

| 파일 | 의미 |
|------|------|
| `summary2.json` | 6q vs 8q(L3) 학습+이상탐지 (8q AUC 0.433) |
| `depth_test.json` | 8q L=3/6/9 깊이별 MMD²/TV/gradient |
| `anomaly_redo.json` | 8q L3 vs L6 이상탐지 재평가 (0.4333 vs 0.4289) |
| `csv_columns_report.json` (+ .csv) | CSV 80컬럼 benign vs attack 분리력 |
| `classical_baseline.json` | Infiltration emp_joint(0.428) vs marginal(0.438) |
| `ddos_baseline.json` | DDoS(HOIC/LOIC-UDP) emp_joint vs marginal |
| `iat_joint.json` | IAT 추가 joint 이득(상한) A/B1/B2 (+0.003→+0.013→+0.052) |
| `qcbm_iat_b1.json` (+ `_smoke`) | ★10q QCBM: AUC 0.8811 > marginal 0.8726, 상한 99.5% |
| `qcbm_iat_b2.json` (+ `_smoke`) | ★12q QCBM: AUC 0.9185 > marginal 0.8741(+0.0445), 상한 99.2% |
| `qcbm_iat_b1_train.log`, `qcbm_iat_b2_train.log` | 두 학습의 tee 로그(텍스트) |

---

## 10. 사용 라이브러리 (import에서 확인)

`numpy`, `pandas`, `matplotlib`, `pennylane`(+`pennylane.numpy`), `scipy.stats`(rankdata), 표준 `multiprocessing`/`json`/`argparse`/`os`. **torch 미사용**(고전 생성모델 비교 시 추가 필요할 수 있음).

---

## 11. 생성한 zip / 제외 항목

- **경로**: `~/qcbm_share_20260626.zip` (`/home/elicer/qcbm_share_20260626.zip`)
- **크기**: 2.9MB (압축전 3.6MB), **94개 파일**. 개별 최대 = `train_qcbm_iat_b1.png` 207KB(전부 코드/그림/JSON).
- **포함 확인**: experiment2.py, tools/*.py 8개, results2/*.json(b1/b2 포함), docs/*.md, data_headers.txt.
- **제외(대용량/원시)**: `qcbm/data/cic_0221_ddos.csv`(314M), `cic_thursday.csv`(103M), `dl.log`(0), `" ISCX-VPN-Dataset"/`(17M), `samples/`(17M). 사유: 원시 트래픽/QCBM 무관, DATA_URL로 재다운로드 가능. **헤더는 `data_headers.txt`로 포함.**
- **비밀정보**: .env/key/token/credential/.pem **없음**(스캔 확인) → 제외 대상 없음.
- 상세 목록: `docs/agent_logs/zip_manifest.txt`.

---

## 12. 실행한 명령과 결과

```bash
git status                                    # ?? docs ?? qcbm
du -sh /home/elicer/IOTJ/*/                   # qcbm 420M(data 417M), ISCX 17M, samples 17M, figs 1.1M
find qcbm -maxdepth 2 -type f ...             # 파일 목록 확보
find . -type f -size +5M ...                  # 대용량: 두 CSV + facebook pcap
head -1 <csv> > docs/agent_logs/data_headers.txt   # 헤더만(본문 미독)
zip -rq ~/qcbm_share_20260626.zip . -x <대용량 제외>   # 2.9MB, 94 files
unzip -l / grep 검증                          # 대용량/비밀 없음, 핵심 코드 포함 확인
```
- 실패: 없음. (manifest 작성 중 `du`가 앞-공백 디렉토리명 `" ISCX-VPN-Dataset"`을 못 읽어 크기 공란 → 본 snapshot에 17M 명기. zip 제외는 정상 적용됨.)
- 학습/평가 실행 없음.

---

## 13. 기존 파일 무변경 확인

- 작업 후 `git status`: `?? docs/  ?? qcbm/` 그대로(추적 파일 변경 0). qcbm 미추적이라 `git diff`는 비어 있음.
- 신규 생성만: `docs/agent_logs/2026-06-26_1012_project_snapshot.md`(본 파일), `zip_manifest.txt`, `data_headers.txt`, `~/qcbm_share_20260626.zip`.
- 기존 .py/.json/.log/.csv 일절 수정·이동·삭제·이름변경 없음.

---

## 14. 남은 불확실성 / 다음 추천

불확실:
1. 고전 생성모델 비교에는 **torch 등 추가 의존성**이 필요할 수 있음(현재 미설치 여부 미확인 — requirements 파일 없음).
2. B2 분포/ROC 그림은 미복원 상태(모델 배열 미저장; rescue 로그 참조). 논문 figure 필요 시 재학습 1회 필요.

다음 추천:
1. `~/qcbm_share_20260626.zip`를 웹 AI에 업로드 → 전체 코드 구조 공유. (원시 CSV는 DATA_URL로 재다운로드 안내.)
2. 고전 baseline(예: 동일 256/1024/4096상태에 대한 작은 신경망/혼합모델) **파라미터 수 vs AUC** 비교 설계 — QCBM nparams(8q 72, 10q 180, 12q 216)와 동급 파라미터의 고전 모델 AUC를 같은 anomaly_eval로 측정.
3. (위생) 향후 학습 스크립트가 best p_final을 `.npy`로 저장하도록 개선(재현/그림 복원 용이) — 별도 작업.

---

## 최종 보고

```markdown
## 작업 요약
- 프로젝트 현황을 사실 기반 점검하고 코드+작은 메타데이터만 담은 공유 zip 생성. 기존 파일 무수정, 학습 실행 없음.
  zip 2.9MB/94파일, 대용량 CSV(417M)·원시 pcap(34M)은 제외하고 헤더만 포함.

## 프로젝트 루트 / git 여부
- /home/elicer/IOTJ, git repo(main). qcbm/·docs/는 untracked.

## 디렉토리 구조 (요약)
- 루트: analyze_*.py, figs(1.1M), ISCX(17M)/samples(17M)[원시,제외], docs/agent_logs, qcbm(420M).
- qcbm: experiment*.py, anomaly_redo/depth_test, data(417M,제외), results/results2(JSON·PNG·log), tools(8 스크립트).

## 핵심 파일 위치
- train/eval entrypoint: experiment.py/experiment2.py main(); tools/train_qcbm_iat_b1.py·b2.py; 진단 tools/diagnose_*.py
- QCBM 핵심 함수: experiment2.py — make_circuit:140, build_kernel:127, train_one:175, run_config:230, anomaly_eval:285, auc_score:266
- TV 계산: experiment2.py tv_dist:160 (0.5*Σ|p-q|)
- 파라미터 수 결정: wshape:150 + run_config:233 (nparams=prod(wshape(nq,LAYERS)); 10q/L6=180,12q/L6=216)
- q_train 생성: empirical_dist:119 (+ discretize:98, IAT는 train 스크립트가 자체 append)

## results2 JSON 목록 (실험별)
- summary2/depth_test/anomaly_redo(8q), csv_columns_report, classical_baseline(Infil), ddos_baseline,
  iat_joint(상한 A/B1/B2), qcbm_iat_b1(10q 0.8811), qcbm_iat_b2(12q 0.9185) + 각 _smoke/_train.log

## 생성한 zip
- 경로/크기/파일수: ~/qcbm_share_20260626.zip / 2.9MB / 94 files (개별최대 207KB PNG)
- 제외 대용량: qcbm/data/cic_0221_ddos.csv 314M, cic_thursday.csv 103M, dl.log 0, " ISCX-VPN-Dataset" 17M, samples 17M
  (사유: 원시 트래픽/QCBM 무관, DATA_URL 재다운로드 가능; 헤더는 data_headers.txt로 포함). 비밀정보 없음.

## 기존 파일 무변경 확인
- git status: ?? docs/ ?? qcbm/ 그대로. 신규 4개(snapshot.md, zip_manifest.txt, data_headers.txt, zip)만 생성.

## 다음 추천 단계
1. zip을 웹 AI에 업로드(코드 구조 공유). 원시 CSV는 DATA_URL 재다운로드 안내.
2. QCBM nparams(72/180/216) 대비 동급 파라미터 고전 생성모델의 AUC를 동일 anomaly_eval로 비교 설계.
```
