# flow_iat 매핑 민감도 점검 — 교차 데이터 재현이 flow_iat 컬럼 선택에 흔들리는가

- 작성: 2026-06-27 03:13 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 빈도/SVD 기반(QCBM 학습 없음). raw MWU. ★전수(샘플링 없음). 판정은 웹 AI.

## 1. 작업 목적
교차 데이터 진단에서 joint>marginal이 12/12 재현됐으나 UNSW flow_iat→dinpkt 매핑이 "불확실"이었다.
flow_iat을 UNSW의 다른 실제 컬럼으로 바꿔 재계산해, 핵심 재현 결과(특히 joint>marginal 12/12)가
flow_iat 매핑 선택에 흔들리는지(강건성)를 무거운 QCBM 학습 전에 싸게 점검한다.

## 2. 시작 git status, 보존 대상
`git status --short`: `?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip` (전체 untracked).
`git diff --stat`(tracked)=비어 있음. 보존 대상: 모든 기존 소스/데이터/결과(import/읽기만).
생성: `tools/sens_flow_iat_mapping.py`, `results2/flow_iat_sensitivity.{json,png}`, 본 로그.
**기존 `tools/diagnose_cross_dataset.py`는 수정하지 않고 import 재사용**(diagnose_dataset/load_cic/CANON).

## 3. flow_iat 대안 매핑 후보 (UNSW 실제 컬럼, 억지 없음)
UNSW 시간/속도 컬럼 확인(모두 nonneg): dinpkt(med 0.007, 0값 120,308), rate(med 2955, 0값 3,968),
sinpkt(med 0.382), dur, sjit/djit(jitter, 0값 과다라 부적합). 선정:
- **M0 = dinpkt** (기존 기준선; Dest interpacket arrival)
- **M1 = rate** (흐름 속도 pkts/s; 시간역수 의미 — 흐름수준 timing 신호)
- **M2 = (sinpkt+dinpkt)/2** (양방향 평균 interpacket; "전체 흐름"에 의미상 가장 근접)
CIC는 **Flow IAT Mean 고정**(CIC는 정확 대응 존재 — 매핑 무관). proto/size/fwd_iat도 고정.
인코딩 구조 동일(proto2+size4+fwd_iat2+flow_iat2=10bit, log1p 분위 4버킷, seed7 0.7 split).
sjit/djit은 0값이 절반 이상이라 이산화 퇴화 위험 → 후보 제외(억지 매핑 회피).

## 4. 샘플 수 (★전수)
- UNSW(매핑 무관): benign(Normal) **93,000** / 공격 9종 164,673 (Generic 58,871·Exploits 44,525·
  Fuzzers 24,246·DoS 16,353·Reconnaissance 13,987·Analysis 2,677·Backdoor 2,329·Shellcode 1,511·
  Worms 174) / ALL 164,673. dropna 0(M0/M1/M2 동일 — flow 컬럼 결측 없음).
- CIC(고정): HOIC benign 360,833 / attack 686,012; Bot benign 762,384 / attack 286,191.

## 5. 확실한 사실 / 추정 / 확인 불가
**확실(재계산)**: §6 표·§7 변동. 17초 내 전수, dropna 0.
**추정(확인됨)**: "flow_iat이 약신호라 매핑 바꿔도 큰 영향 없을 것" → **대체로 사실**(joint>marginal
완전 강건). 단 benign 복잡도 랭크와 max-single-feature는 매핑에 따라 일부 변동(아래).
**확인 불가/한계**: 어느 매핑이 "참" flow_iat인지 단정 불가(UNSW에 전체흐름 IAT 단일 컬럼 부재).
세 매핑은 모두 합리적 근사이며, 결론이 셋 다에서 같으면 강건으로 본다.

## 6. 매핑별 결과 (★전수)
| 매핑 | joint>marginal | UNSW/ALL joint이득 | emp_joint | marginal | benign reach(TV≤0.05) | benign 점유 |
|---|---|---|---|---|---|---|
| M0 dinpkt(기준) | **12/12** | +0.1612 | 0.8624 | 0.7012 | 12 | 16.9% |
| M1 rate | **12/12** | +0.1492 | 0.8527 | 0.7035 | 11 | 19.6% |
| M2 (sinpkt+dinpkt)/2 | **12/12** | +0.1678 | 0.8705 | 0.7027 | 7 | 17.0% |

공격별 joint이득(매핑 M0/M1/M2)과 변동폭:
| 공격 | M0 | M1 | M2 | 변동폭 | 항상>0 |
|---|---|---|---|---|---|
| Analysis | +0.074 | +0.089 | +0.099 | 0.025 | ✅ |
| Backdoor | +0.080 | +0.079 | +0.079 | 0.001 | ✅ |
| DoS | +0.083 | +0.077 | +0.085 | 0.008 | ✅ |
| Exploits | +0.254 | +0.227 | +0.260 | 0.033 | ✅ |
| Fuzzers | +0.122 | +0.102 | +0.150 | 0.048 | ✅ |
| Generic | +0.112 | +0.112 | +0.113 | 0.001 | ✅ |
| Reconnaissance | +0.264 | +0.242 | +0.258 | 0.022 | ✅ |
| Shellcode | +0.141 | +0.174 | +0.165 | 0.033 | ✅ |
| Worms (n=174) | +0.393 | +0.468 | +0.481 | 0.088 | ✅ |
| ALL | +0.161 | +0.149 | +0.168 | 0.019 | ✅ |

## 7. 매핑 간 변화
- **joint>marginal 깨진 공격: 없음.** 30개 측정(공격10×매핑3) 전부 joint이득>0.
- **joint이득 최대 변동폭 = 0.088**(Worms, n=174 소표본). ALL은 0.019, 대부분 공격 ≤0.05.
  부호 뒤집힘·결론 역전 전무.
- **공격 분류 뒤집힘 3개**(고정 임계 emp<0.75 / gain≥0.15 기준):
  - Shellcode: M0 inadequate(emp 0.739) → M1/M2 complex(0.769/0.756). emp가 0.75 문턱을 살짝 넘음.
  - Worms: M0 inadequate(emp 0.696) → M1/M2 complex(0.756/0.769). 0.75 문턱 통과(n=174 노이즈).
  - ALL: M0 complex(gain 0.161) → M1 simple(0.149) → M2 complex(0.168). gain이 0.15 문턱 바로 위/아래.
  → 셋 다 **임계 경계 근처의 분류 경계 효과**일 뿐, joint이득 값 자체는 거의 안 움직임(실질 변화 아님).
- **부수 변동(결론과 무관)**: max-single-feature가 M0에선 flow_iat(dinpkt)이던 공격들이 M1/M2에선
  size/proto로 바뀜(dinpkt가 단일 신호를 더 들고 있었음). benign reach 랭크도 M2(7)가 M0(12)보다 낮음
  — flow_iat 컬럼이 benign joint 복잡도에 영향. 단 joint>marginal 결론에는 영향 없음.

## 8. 사실 관찰 (판정 아님)
- **재현 결과가 flow_iat 매핑에 강건한가**: **핵심 결론(joint>marginal)은 완전 강건** —
  dinpkt/rate/(sinpkt+dinpkt)/2 세 매핑 모두 12/12, 모든 공격에서 joint이득>0. joint이득 크기도
  대부분 ±0.03 이내 변동(ALL ±0.02).
- **흔들리는 항목**: (i) benign 복잡도 랭크(reach TV05 = 12/11/7)와 점유율이 매핑에 따라 변동,
  (ii) max-single-feature 정체성, (iii) 경계 공격(Shellcode/Worms/ALL)의 임계 기반 분류 라벨.
  이들은 모두 2차 지표이며 결론(joint>marginal)을 뒤집지 않음.
- **Fuzzers는 세 매핑 모두 feature_inadequate**(emp 0.64~0.69) — 가장 안정적인 "feature 부적합" 사례.
- 한줄: **flow_iat 매핑 불확실성은 핵심 재현(joint>marginal)을 위협하지 않음. 강건.**

## 9. 생성 파일, 실행 명령과 결과
- `qcbm/tools/sens_flow_iat_mapping.py` (신규; diagnose_cross_dataset import 재사용)
- `qcbm/results2/flow_iat_sensitivity.json` (23KB), `.png` (93KB) (신규)
- `docs/agent_logs/2026-06-27_0313_flow_iat_sensitivity.md` (본 로그)
```
python3 -m py_compile tools/sens_flow_iat_mapping.py     # OK
python3 tools/sens_flow_iat_mapping.py                   # real 0m17.1s, DONE
# M0/M1/M2 각각 joint>marg 12/12; 최대 변동폭 0.088; joint 깨짐 False; 분류뒤집힘 3(경계)
```
실패/에러 없음. 메모리 문제 없음(usecols). 샘플링 없음(전수). 새 dependency 없음.

## 10. git diff --stat (기존 무변경)
```
$ git diff --stat        # tracked 변경: 없음
$ git status --short
?? docs/   ?? qcbm/   ?? qcbm_share_20260626.zip
```
기존 소스(diagnose_cross_dataset.py 포함)/데이터/결과 무수정. QCBM 무학습, dependency 무설치.

## 11. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] flow_iat 대안 3개(dinpkt/rate/(sinpkt+dinpkt)/2, UNSW 실제 컬럼, 억지 없음)로 진단.
- [x] 각 매핑 joint>marginal 카운트·joint이득·복잡도·공격분류 측정(★전수, 샘플 수 명시).
- [x] 매핑 간 변화(최대 변동폭 0.088, 결론 뒤집힘 없음, 분류 경계효과 3건) 정리.
- [x] "강건한가" 사실 보고: 핵심 결론 강건.
- [x] json/png 저장, 기존 무수정·무학습·무설치.

**남은 불확실성**
- 어느 매핑이 의미상 "참"인지 여전히 미상(UNSW 한계). 단 결론이 셋 다 동일이라 실무 영향 적음.
- benign 복잡도 랭크의 매핑 민감성(7~12)은 "복잡도-joint의존" 정량 비교에 영향 가능 — 절대 랭크
  비교 시 매핑 명시 필요.
- Worms(n=174) 변동폭 최대 — 소표본 노이즈, 결론엔 무영향.

**다음 추천 단계**
1. (강건 확인됨 → 양자 단계 진입 가능) UNSW 복잡+joint의존 공격(Exploits/Reconnaissance/ALL,
   세 매핑 모두 큰 joint이득)에서 QCBM 학습 후 restart 분포·216p 효율·해리가 CIC Bot처럼 나오는지.
   flow_iat 매핑은 의미 근접한 M2(sinpkt+dinpkt 평균) 권장(전체흐름에 가장 가깝고 결과도 강건).
2. Fuzzers/Shellcode "feature 부적합"의 빠진 신호를 UNSW 고유 컬럼(sttl/state/swin)에서 탐색.
