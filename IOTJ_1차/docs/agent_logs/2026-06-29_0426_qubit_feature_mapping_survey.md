# 큐빗-feature 할당 구조 및 marginal/joint 정의 조사 (조사 전용, 코드 무수정)

- 작성: 2026-06-29 04:26 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 순수 조사. 코드 일절 무수정(읽기 + 무해한 py_compile만). 판정은 웹 AI.

## 1. 작업 목적
"joint"가 feature 간 고차 상관을 실제 표현했는지, 아니면 저차에 머물렀는지를 나중에 판단하려면
먼저 (1) 큐빗이 각 feature에 몇 비트씩 할당되는가, (2) marginal/joint가 코드상 정확히 무엇인가를
확정해야 한다. 직전 차수 분해는 feature 경계를 무시하고 **비트 단위**로만 차수를 쟀기에 feature
내부 상관과 feature 간 상관이 섞였다. 이 조사는 그 분리를 위한 사전 정보 수집이다.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/  ?? qcbm/  ?? *.zip` — 전부 untracked. 보존: 모든 기존 파일(읽기만).
이번 작업 산출물: 본 .md 하나(코드 무수정).

## 3. 확인한 인코딩/binning 함수 (함수명·위치·시그니처)
- `experiment2.py`
  - `proto_code(p)` @ L69 — TCP(6)→0, UDP(17)→1, p==0→2, else→3.
  - `size_edges_from_benign(benign, nbuckets)` @ L90 — `np.quantile`(**원값, 선형 분위수**)로 size 경계. log 아님.
  - `discretize(df, spec, size_edges)` @ L98 — spec["features"]=[(name,bits)...] 순서대로 MSB→LSB 패킹: `state=(state<<bits)|v`.
    proto=proto_code, syn=`SYN Flag Cnt>0`, psh=`PSH Flag Cnt>0`, size=`np.digitize(Pkt Len Mean, size_edges)`→clip(0,2^bits−1).
  - `empirical_dist(states, nstates)` @ L119 — `np.bincount/합` = **전체 비트 결합(joint) 경험적 분포**.
- `tools/train_qcbm_iat_b2.py`
  - `log1p_quantile_edges(values, nbuckets)` @ L63 — **log1p 후** `np.quantile` 분위수 경계(IAT용).
  - `iat_to_bits(values, edges)` @ L69 — log1p→`np.digitize`→clip(0, IAT_BUCKETS−1). IAT_BUCKETS=4(=2비트).
  - `append_bits(state, add_val, add_bits)` @ L74 — `(state<<add_bits)|add_val`(LSB 쪽에 추가).
  - `marginal_dist(states, nq)` @ L78 — **비트 단위 독립** 곱(아래 §6).
- build 함수: `build_b2()`(HOIC) @ train_qcbm_iat_b2.py:94, `build_bot()` @ train_qcbm_bot_resumable.py:86,
  `build_unsw()` @ train_qcbm_unsw_resumable.py:79. 모두 benign train(seed=7, 0.7)에서 경계 적합, `q_train=empirical_dist`(joint) 반환.

## 4. 큐빗-feature 할당 표 (데이터셋별)
공통: 비트 패킹 MSB→LSB, `(state>>i)&1`로 비트 i 추출(i=0=LSB). feature별 비트 수와 **비트 위치**:

| 데이터셋 | 총 비트 | feature 개수 | feature(비트수) MSB→LSB | 비트 위치(i) | 대응 원본 컬럼 |
|---|---|---|---|---|---|
| **HOIC** | 12 | 6 | proto(2) · syn(1) · psh(1) · size(4) · fwd_iat(2) · flow_iat(2) | proto=10–11, syn=9, psh=8, size=4–7, fwd_iat=2–3, flow_iat=0–1 | Protocol, SYN Flag Cnt, PSH Flag Cnt, Pkt Len Mean, Fwd IAT Mean, Flow IAT Mean |
| **Bot** | 12 | 6 | (HOIC와 동일) | (HOIC와 동일) | (HOIC와 동일; 단 CSV=cic_bot_0203.csv, 공격="bot") |
| **UNSW** | 10 | 4 | proto(2) · size(4) · fwd_iat(2) · flow_iat(2) | proto=8–9, size=4–7, fwd_iat=2–3, flow_iat=0–1 | proto, smean, sinpkt, (sinpkt+dinpkt)/2 |

- proto: 2비트지만 실제 코드값은 HOIC/Bot 0–3, UNSW는 TCP/UDP/기타=0/1/3(코드2 미사용).
- size: 4비트=16버킷(SIZE_BUCKETS=16) 완전 사용. fwd/flow_iat: 2비트=4버킷 각각.
- **UNSW는 syn/psh(플래그 2비트) 제외** → 10비트. SPEC_PS=proto2+size4(6비트 MSB) 후 fwd/flow_iat 추가.
- UNSW 컬럼 매핑(build_unsw): Protocol←proto(tcp=6/udp=17/else=−1), Pkt Len Mean←smean, Fwd IAT Mean←sinpkt,
  **Flow IAT Mean←M2=(sinpkt+dinpkt)/2**.

## 5. binning 방식
- **proto**: 범주형 `proto_code`(TCP/UDP/0/기타). 2비트.
- **syn/psh**: 이진 임계 `값>0`. 1비트(HOIC/Bot만).
- **size(Pkt Len Mean)**: **원값 선형 분위수**(`size_edges_from_benign`=np.quantile, 16버킷) → digitize. log 아님.
- **fwd_iat/flow_iat**: **log1p 후 분위수**(`log1p_quantile_edges`, 4버킷) → digitize.
- 모든 경계는 **benign train(seed=7, 0.7 split)** 에서 적합, train/test/attack에 공유 적용.

## 6. marginal 정의 (코드 기준)
`marginal_dist(states, nq)` (train_qcbm_iat_b2.py:78):
- `p_bit[i] = mean_samples( bit_i )` (각 비트의 1 확률), `q[s] = ∏_i (bit_i==1 ? p_bit[i] : 1−p_bit[i])`, 정규화.
- 즉 **모든 비트가 서로 독립이라 가정한 완전 비트-인수분해 분포**.
- ★중요: 독립 단위가 **feature가 아니라 비트**다. 같은 feature 내부 비트들(예: size의 4비트, IAT의 2비트)
  사이의 상관조차 marginal은 보존하지 않는다. → "joint gain"과 직전 차수분해의 "고차"에는 **feature 내부 비트 상관**도 포함된다.

## 7. joint 정의 (코드 기준)
`empirical_dist(states, nstates)` (experiment2.py:119):
- 전체 비트열(state index 0..2ⁿ−1)의 **bincount 정규화** = **모든 비트의 결합 경험적 분포**.
- feature 간·feature 내부 모든 상관을 그대로 담음. QCBM 학습 타깃(q_train)이 이 joint.
- (marginal vs joint 차이 = 모든 비트 독립가정 대비 실제 결합. feature 단위 분해는 코드에 별도 정의 없음.)

## 8. 세 데이터셋 간 차이
- HOIC ↔ Bot: 인코딩 **완전 동일**(12비트, 동일 SPEC8+IAT, 동일 binning). 데이터/공격 라벨만 다름.
- UNSW: 10비트, **syn/psh 제외**(4 feature), flow_iat=M2 매핑. proto/size/IAT binning 규칙은 동일 계열.
- 공통: MSB→LSB 패킹, benign-train 분위수 경계, size=선형분위수·IAT=log1p분위수, marginal=비트독립곱, joint=전체 결합.

## 9. 확실한 사실 / 추정 / 확인 불가
- 확실(코드 확인): §4–§7 전부(함수 위치·시그니처·비트 위치·binning·marginal/joint 정의).
- 추정 없음(모두 코드로 확정).
- 확인 불가/유의: 비트 위치는 패킹 순서(MSB→LSB)로 도출 — 본 .md의 비트 i 매핑은 그 규칙의 직접 귀결(별도 테스트 미실행).
  feature 단위 marginal/joint(비트가 아닌 feature 독립)는 코드에 **존재하지 않음**(현 marginal은 비트 단위).

## 10. git diff (무변화 확인)
`git diff --stat` = 변경 없음(tracked). `py_compile`(experiment2 + 3개 train 파일) OK — 읽은 파일 정상, 무수정.
`git status --short` = `?? docs/  ?? qcbm/  ?? *.zip`(신규는 본 .md뿐).

## 11. 남은 불확실성 / 다음 추천
**남은 불확실성**
- 현 marginal/joint·직전 차수분해는 모두 **비트 단위**라, feature 내부 상관과 feature 간 상관이 섞임(목적에서 지적한 그대로).
- proto 2비트의 코드 분포(어느 코드가 실제 점유)는 데이터별로 다를 수 있음(미집계, 본 조사 범위 밖).

**다음 추천 단계**
1. 비트→feature 매핑(§4)을 이용해 차수분해를 **feature-인지**로 재정의: (a) feature 내부 비트 상관, (b) feature 간 상관 분리 측정.
   예: KL 분해에서 부분집합 S를 "단일 feature 내부" vs "2+ feature 걸침"으로 그룹화.
2. feature 단위 marginal(각 feature는 내부적으로 joint, feature끼리만 독립) 분포를 새로 정의해 현 비트단위 marginal과 대조.
3. proto/size 등 feature별 점유 코드 분포를 집계해 "비트 일부가 거의 고정(저엔트로피)"인지 확인(고차 confound 원인 탐색).
