# Bot(+Infiltration) 분포 복잡도 진단 — 양자가 이길 가능성이 있는 복잡도인가 (학습 없음)

- **작성일시**: 2026-06-26 15:00 (로컬)
- **작성 주체**: 로컬 Claude Code agent (plan→default)
- **작업 성격**: 빈도/SVD 기반 분포 복잡도 진단만. **QCBM 학습 없음. 기존 소스/데이터/결과 무수정.**

---

## 1. 작업 목적
HOIC(기존, 단순: 점유 2.3%, psh 단독 AUC 0.878 지배, 저랭크)와 비교해 **Bot(필수)·Infiltration**
분포가 더 고랭크·조밀하고 단일 feature 지배가 약한지(=양자가 이길 가능성 있는 복잡도인지)를
**학습 없이** 진단한다. 판정은 웹 AI. 본 작업은 사실 측정만.

## 2. 시작 git status / 보존 대상
```
On branch main. Untracked: docs/, qcbm/, qcbm_share_20260626.zip. (추적 파일 변경 0)
```
- 새로 생성만: `tools/diagnose_attack_complexity.py`, `results2/attack_complexity.json/.png`, 본 `.md`.
- 무수정: 모든 기존 소스/데이터/`results2` 기존 결과.

## 3. 재사용 경로 (코드 확인 완료)
- `experiment2.py`: `proto_code`, `size_edges_from_benign`, `discretize`(SPEC8 = proto2+syn1+psh1+size4),
  `empirical_dist`, `tv_dist`, `auc_score`, `anomaly_eval`(s=-log p, attack=positive, raw MWU).
- `tools/train_qcbm_iat_b2.py`: `log1p_quantile_edges`, `iat_to_bits`, `append_bits`(B2 12bit 패킹),
  `marginal_dist`(1bit×12 독립곱), `eval_dist`(fake cfg로 anomaly_eval 호출). B2 인코딩 구조:
  proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2 = 12bit/4096상태.
- `tools/diagnose_encoding_info_loss.py`: feature별 엔트로피/단일 feature AUC 로직 참고.
- `tools/diagnose_param_efficiency_b2.py`: q_train 64×64 reshape → SVD 저랭크 TV 곡선(params=128r) 참고.

## 4. 새 데이터 Label/컬럼 확인 결과 (실측)
| 파일 | benign | attack(라벨문자열) | 오염/NaN |
|------|--------|--------------------|----------|
| `data/cic_bot_0203.csv` | 762,384 | **bot** 286,191 | 7컬럼 전부 NaN/inf **0** (깨끗) |
| `data/cic_thursday.csv` | 238,037 | **infilteration** 93,063 | "label" 25행 = 헤더오염 → 7컬럼 NaN 25 (dropna로 제거됨) |
| `data/cic_0221_ddos.csv`(HOIC, 기준) | 360,833 | **ddos attack-hoic** 686,012 | (기존) |
- 7개 핵심 컬럼 모두 존재(이전 작업서 검증). IAT 결측: bot 0 / thursday 25(오염행, 기존 `dropna` 처리).
- → **기존 인코딩 함수 그대로 적용 가능.** 단 size/IAT 분위수 경계는 각 데이터 benign train에서 새로 생성
  (HOIC 경계 재사용 금지 — 금지사항 4). 인코딩 "구조"(비트배분/log1p)는 동일.

## 5. HOIC 기준값 (기존 결과 json에서 확인 — 재계산 정합 대상)
| 항목 | 기존 값 (출처) |
|------|----------------|
| 점유 | 94/4096 = **2.29%** (`qcbm_iat_b2.json`) |
| marginal AUC | **0.8741** |
| emp_joint AUC | **0.9258** (joint이득 +0.0517) |
| psh 단독 AUC (최대 단일) | **0.8779** (`encoding_info_loss.json`) |
| SVD 저랭크 TV | r=1 0.0794 / r=2 0.0496 / r=3 0.0092 (`param_efficiency_b2.json`, params=128r) |
- 주의: 태스크 배경의 "저랭크 SVD r=2(128p)로 TV 0.0055"는 코드의 실제 param_def(128r ⇒ r=2=256p)
  및 실측(r=2 TV 0.0496)과 **불일치**. 본 진단은 코드 실측값으로 재계산해 기록(가정 안 맞추고 보고).

## 6. 확실한 사실 / 추정 / 확인 불가
- **확실**: 재사용 함수 시그니처, 새 데이터 라벨 문자열(bot/infilteration), 결측 구조, HOIC 기준값.
- **추정(측정으로 확정 예정)**: Bot이 HOIC보다 고랭크/조밀/단일지배 약한가.
- **확인 불가(현재)**: 없음. (진단 실행으로 채움.)

---
## 7. HOIC 재계산 정합 검증 — **전부 OK (PASS)**
| 지표 | 재계산 | 기존 | 판정 |
|------|--------|------|------|
| 점유 | 94 | 94 | OK |
| marginal AUC | 0.8741 | 0.8741 | OK |
| emp_joint AUC | 0.9258 | 0.9258 | OK |
| psh 단독 AUC | 0.8779 | 0.8779 | OK |
→ 인코딩/분할/평가 경로가 기존과 동일하게 재현됨. **분석 신뢰성 전제 충족.**

## 8. 복잡도 비교표 (5개 지표) — HOIC vs Bot vs Infiltration
| 지표 | HOIC(기존,단순) | **Bot** | Infiltration |
|------|------|------|------|
| 점유율 (지표3) | 2.29% (94/4096) | **8.81% (361)** | 6.18% (253) |
| emp_joint AUC | 0.9258 | 0.8749 | 0.4633 |
| marginal AUC | 0.8741 | **0.2870** | 0.4333 |
| **joint 이득 (지표1)** = emp−marg | +0.0518 | **+0.5879** | +0.0300 |
| 최대 단일 feature AUC (지표4) | **psh 0.878** (강지배) | psh **0.529** (지배없음) | psh 0.561 (지배없음) |
| 저랭크 비용: TV≤0.05 도달 r (지표2) | r=2 (256p) | **r=7 (896p)** | r=9 (1152p) |
| 저랭크 비용: TV≤0.01 도달 r | r=3 | r=9 | r=11 |
| SVD 누적에너지 r=3 | 0.999 | 0.884 | 0.883 |

### 지표5: feature별 단독 AUC / 엔트로피 활용률
- **HOIC**: psh **0.878**(활용 5%) 외 전부 ≤0.64. → psh 단일 feature가 분리를 지배.
- **Bot**: 모든 feature 단독 AUC ≤0.53, 다수가 **0.5 미만**(fwd_iat 0.125, flow_iat 0.125, size 0.349,
  proto 0.360). → **어떤 단일 feature도 Bot을 못 가른다(오히려 정상보다 더 정상처럼 보임).**
- **Infiltration**: 전부 ≤0.56, 다수 0.5 미만. 단일 지배 없음.

## 9. SVD 특이값 곡선 (저랭크 vs 고랭크) — `results2/attack_complexity.png` 패널1·2
- **HOIC**: 특이값이 index ~5에서 바닥. 누적에너지 r=3에 **0.999** → 사실상 rank-3 저랭크.
  저랭크 근사가 r=2(256p)만에 TV≤0.05 도달.
- **Bot**: 특이값이 index ~18까지 완만히 감소. 누적 r=3 0.884, r=5 0.950. TV≤0.05 도달에 **r=7** 필요.
- **Infiltration**: Bot과 유사하게 완만(누적 r=3 0.883), TV≤0.05 도달 r=9 (HOIC의 4.5배 랭크).

## 10. 사실 관찰 (판정 아님 — 양자 베팅 판단은 웹 AI)
1. **Bot은 HOIC보다 모든 축에서 더 복잡하다(측정값 기준):**
   - (a) 더 고랭크: TV≤0.05 도달 r=7 vs HOIC r=2 (3.5배); SVD 누적E(r3) 0.884 vs 0.999.
   - (b) 더 조밀: 점유 8.81% vs 2.29% (3.8배).
   - (c) 단일 feature 지배 없음: 최대 단독 AUC 0.529 vs HOIC psh 0.878.
   - (d) 고전 저랭크 비용 더 큼: emp_joint TV 도달에 약 3.5배 랭크(=128/r 파라미터) 필요.
2. **Bot의 분리력은 거의 전적으로 joint(상호작용) 구조에 있다:**
   marginal(독립곱) AUC=**0.287**(0.5 미만 = 방향이 반대로 틀림)인데 emp_joint AUC=**0.875**.
   → joint 이득 **+0.588** (HOIC +0.052의 11배). 개별 feature는 Bot을 못 가르나 "조합"이 가른다
   (정상 흉내 공격의 전형적 시그니처).
3. **Infiltration은 "복잡하지만 신호 약함":** 고랭크(r=9)·조밀(6.18%)·단일지배 없음이지만,
   emp_joint AUC=**0.463**(≤랜덤) → 이 12bit 인코딩에서는 **완벽한 경험적 joint조차 분리 거의 못 함**.
   복잡도는 높으나 분리 가능한 신호 자체가 희박(joint 이득도 +0.030로 작음).

## 11. 생성 파일 / 실행 명령 / git
### 생성(새 파일만)
- `tools/diagnose_attack_complexity.py` (진단 스크립트)
- `results2/attack_complexity.json` (21KB), `results2/attack_complexity.png` (181KB)
- 본 `.md`
### 실행 명령과 결과
```bash
python3 -m py_compile tools/diagnose_attack_complexity.py        # PY_COMPILE_OK
python3 tools/diagnose_attack_complexity.py                      # 13.9s, DONE
#  -> HOIC 정합 4/4 OK, 비교표 출력, json/png 저장. (Korean glyph 경고는 plot title 수정 후 해소)
```
- 실패/우회 없음. QCBM 학습 없음. 새 dependency 없음. benign 샘플링 없음(전수).
### git (기존 무변경)
```
git diff --stat  -> (빈 출력 = 추적 파일 변경 0)
git status --porcelain -> ?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip
  새 untracked: qcbm/tools/diagnose_attack_complexity.py, qcbm/results2/attack_complexity.{json,png}
```
기존 소스/데이터/results2 기존 결과 **무수정 확인.**

## 12. 성공 기준 충족 / 남은 불확실성 / 다음 추천
- 성공 기준: Bot 5개 지표 측정+HOIC 나란히 비교 ✓ / Infiltration 포함 ✓ / HOIC 정합 ✓ /
  핵심 비교(고랭크·조밀·단일지배약함·저랭크비용) 사실 기록 ✓ / json·png 저장 ✓ / 무변경·무학습·무설치 ✓.
- 남은 불확실성:
  - 본 진단은 **고정 12bit B2 인코딩** 기준. 해상도를 더 주면(특히 IAT/size 버킷↑) Bot/Infiltration의
    랭크·신호가 달라질 수 있음(현 인코딩에서 Bot fwd/flow_iat 단독 AUC 0.125로 낮음 — 정보가 다른 곳).
  - "고랭크=양자 우위"는 **상관일 뿐 보장 아님**(QCBM이 그 고랭크 분포를 실제로 적은 파라미터로
    표현/학습하는지는 별도 검증 필요). 판정은 웹 AI.
- 다음 추천:
  1. **Bot**은 "복잡한 분포 + 강한 joint 의존 + 분리 가능한 신호(emp_joint 0.875)"를 모두 갖춘
     **유일한 후보** → QCBM 학습 1순위 대상(marginal 0.287을 얼마나 넘어 emp_joint 0.875에 근접하는지).
  2. Infiltration은 현 인코딩에서 신호 부족 → QCBM 학습 가치 낮음(인코딩 feature 재설계가 선행돼야).
  3. (선택) Bot 인코딩 해상도/feature 민감도 진단 후 QCBM 베팅 확정.
