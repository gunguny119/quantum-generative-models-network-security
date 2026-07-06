# Drebin binary feature 고차 상관 측정 — flow(2차 73~99%) 대비 정말 고차 지배인가 (측정 전용)

- 작성: 2026-07-02 05:30 UTC (KST 14:30)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- **순수 차수 측정.** QCBM 학습·탐지 없음, 기존 코드 무수정(import), 무설치. 단정 금지(측정값만, 우위·QCBM 언급 없음).

## 1. 작업 목적
확보한 Drebin binary static feature(0/1)가 flow(연속, 2차 지배)보다 **고차 상관이 강한지** 측정. flow는 ≥3차 비중이 작았음(UNSW 27%, Bot 19%, HOIC 1.2%). malware는 악성 행동이 여러 feature 조합이라 고차가 클 수 있다는 가설을 우리 기존 잣대(차수분해)로 측정한다. 결과에 따라 malware 축 존폐 판단 재료.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip`(전부 untracked). 보존: 기존 모든 파일·results2 기존 json·Drebin 원본(읽기만).

## 3. 확인한 flow 측정 규모 / 도구 재사용 / Drebin 로드 / 계산 한계
- **도구 재사용**: `tools/diagnose_order_decomposition.py`의 `decompose(p, nq, kmax=4)`가 **2^nq 상태 경험분포 p + nq만** 받아 1/2/3/4/≥5차 KL 기여(c2/c3/c4/resid5) 분해. flow 전용 전처리에 안 묶임 → **import만, 무수정**. `subset_features`/`maxent_upto`(L-BFGS)/`kl_div`/`self_check`(parity·ising 합성검증) 포함.
- **flow 규모/기준값**(`results2/order_decomposition.json`): UNSW 10bit occ174, Bot 12bit occ361, HOIC 12bit occ94, kmax=4. 지표 KL1=KL(p‖1차 maxent). **≥3차 비중: UNSW 27.0% / Bot 18.8% / HOIC 1.2%** (2차 73/81/99%).
- **Drebin 로드**: `data/malware/CADE/data/drebin_new_N.npz`(X_train/y_train/X_test/y_test, 0/1 확인). feature명 `.../drebin_new_N/drebin_newN_train_selected_features.txt`(열 1:1). **X = vstack(X_train,X_test)로 pooling**(표본 3315, 모멘트 안정 — 측정 전용·학습 아님이라 train/test leakage 무관). 전부 비상수(1376/1376).
- **계산 한계**: kmax=4 열수 C(k,≤4). k=10→386열, k=12→793열, 2^12=4096행 → 수 초~수십 초. **k∈{10,12}로 flow(UNSW10/Bot·HOIC12)와 동일 규모** 유지. k>12는 표본(3315) 대비 상태수 폭증=희소·불공정 → 미채택.

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: 도구 재사용 범위·flow 기준값·Drebin 0/1·로드 경로·k별 열수. feasibility 예비(참고): high-var k10 ge3≈15%, high-freq k10 ge3≈2%.
- 추정: malware 조합성으로 Drebin ge3가 flow보다 클 수 있음(미검증). 선정 방식/split에 따라 값 달라질 수 있음 → 3방식×3split.
- 확인 불가(본실행서): 방식·split 일관성, flow 대비 유의차, 고차 집중 조합. Bot/HOIC flow 재현은 350MB CSV라 저장 json 인용(UNSW만 재빌드 재현).

## 5. Drebin 부분집합 구성(방식별, split0 기준; split1/2도 동일 절차 재선정)
1376(split0) feature 전부 비상수. k∈{10,12}. 방식:
- **highfreq**: p(1) 최대 top-k. → feature/permission/intent 등 거의 항상 1(p 0.99~0.73) → **매우 편중**(occ 39~44/1024).
- **highvar**: p(1-p) 최대 top-k (p≈0.5). → permission/api_call/real_permission/call/feature 혼재, 정보량 최대(occ 162/1024).
- **category**: permission/api_call/intent/activity/real_permission/url/call 라운드로빈+카테고리 내 고분산. → 카테고리 다양 혼합(occ 110~112/1024).

## 6. 각 부분집합 차수 분해 (지표 = 총상관 KL1=KL(p‖1차 maxent) 중 비중)
※ **headline(2차 vs ≥3차)** = KL1·KL2(1·2차 maxent)만으로 계산 — 값싸고 수렴 견고, 3 split 일관.
※ **≥3차 내부 세부(3/4/≥5차)** = k≤10만 OD.decompose(kmax=4)로 수행. **12bit는 3·4차 maxent(298·793 params)가 희소 4096상태서 비수렴·수 분 소요** → headline만(세부 생략, 사유 명시). (원래 kmax=4 전면 실행은 highfreq k12에서 L-BFGS가 200k func-eval 한계로 11분+ 정체 → headline 방식으로 전환.)

**≥3차 비중 (3 split 평균 [min,max])**
| 방식 | k=10 | k=12 |
|---|---|---|
| highfreq | 2.3% [1.8,2.6] | 11.6% [11.1,12.4] |
| **highvar** | **15.2% [15.2,15.2]** | **21.8%** (split2 headline conv=False*) |
| category | 13.7% [13.2,14.7] | 12.3% [11.8,13.2] |
\* highvar k12 split2 2차 maxent 모멘트 수렴 플래그 False나 값(21.8%)은 split0/1과 동일 → 수치 안정.

**k=10 세부(총 KL1 대비, split0)**: highvar c2=85% / 3rd=12% / 4th=3% / ≥5th=0%; category c2=87% / 3rd=12% / 4th=1%; highfreq c2=98% / 3rd=2%. → **≥3차의 대부분이 3차**, 4차↓ 급감(5차 이상 ≈0).

## 7. flow 대비표
| 데이터 | 2차 | ≥3차 |
|---|---|---|
| flow UNSW (10bit) | 73.0% | **27.0%** |
| flow Bot (12bit) | 81.2% | 18.8% |
| flow HOIC (12bit) | 98.8% | 1.2% |
| Drebin highvar k12 | 78.2% | **21.8%** |
| Drebin highvar k10 | 84.8% | 15.2% |
| Drebin category k10/k12 | 86~88% | 12~14% |
| Drebin highfreq k10/k12 | 88~98% | 2~12% |
→ Drebin ≥3차 최대(highvar k12)=21.8%로 **flow UNSW(27%)보다 낮고 Bot(19%)과 비슷**. HOIC(1%)보다는 큼. **flow 범위(1~27%) 안**.

## 8. 고차 집중 feature 조합 (highvar k12 split2, ≥3차 최대 부분집합; 경험적 connected 3-body 상위)
- permission + real_permission + call (|conn|=0.395), permission + api_call + call (0.393), permission + real_permission + feature (0.300), call + permission + feature (0.297).
- → 고차 상관은 **permission·api_call·call 조합**에 집중(악성 행동 조합과 부합하는 방향; 단 그 KL 기여 총량은 위 표처럼 flow 대비 크지 않음). 향후 QCBM feature 선정 시 참고.

## 9. self-check (잣대 일관성)
- (i) 합성: parity3 c3=0.131, parity4 c4=0.131, ising2 c3=4.1e-17, 단조 all=True → **도구 유효**.
- (ii) **flow UNSW 재현**: OD.decompose 재실행 ≥3차 비중 현재=0.270 = 저장 0.270 (**Δ=0.0**). Bot/HOIC은 350MB CSV라 저장 json 인용(사유 기록). → **잣대 일관 확인**.

## 10. 사실 관찰 (판정 아님, 단정 금지)
- **Drebin이 flow보다 고차 지배인가 → (측정 범위 내) 지지되지 않음(대체로 기각 쪽)**:
  - 3방식·3split 모두 **2차가 지배(78~98%)**, ≥3차 최대=21.8%(highvar k12)로 **flow UNSW(27%)보다 낮고 Bot(19%) 수준**. flow 범위(1~27%)를 벗어나 "뚜렷이 더 고차"인 증거 없음.
  - 오히려 highvar k12(78% 2차)는 flow UNSW(73% 2차)보다 **더 2차-지배적**.
  - 방식 의존: highfreq(편중)=거의 2차, highvar(정보량 최대)=≥3차 최대. → 부분집합 선정에 민감하나, 어느 방식도 flow를 크게 넘지 않음.
  - k↑(10→12)로 ≥3차 소폭↑(highfreq/highvar)나 상한은 여전히 ~22%.
- **의미**: "malware static feature가 조합적이라 flow보다 고차 상관이 강하다"(추론1 전제)는 **이 측정(k≤12 부분집합·이 3방식·차수분해 잣대)에서는 뒷받침되지 않음**. 고차 상관은 permission·api·call 조합에 존재하되 총 기여 비중은 flow와 같은 범위.
- ★ **측정값 기준. 우위·QCBM 성능 언급 없음. 단정 금지** — 이 잣대·이 부분집합 규모 한정(전체 1376/5760 feature나 다른 인코딩에선 다를 수 있음).

## 11. 생성 파일 / 실행 명령 / 시간
- 신규: `tools/diagnose_drebin_order.py`, `results2/drebin_order/{drebin_order.json,drebin_order.png,run.log}`, 본 로그.
```
python3 -m py_compile tools/diagnose_drebin_order.py                  # OK
python3 tools/diagnose_drebin_order.py --smoke                        # OK
python3 tools/diagnose_drebin_order.py --splits 0,1,2 --ks 10,12 --methods highfreq,highvar,category  # exit0
```
- 시간: 본실행 ~2분(self-check UNSW 재빌드 포함; k10 fine 각 ~10s, k12 headline 각 ~2s). **단일 프로세스**(병렬 없음).
- 계산 이슈/전환: 초기 kmax=4 전면 실행이 highfreq k12서 L-BFGS 200k func-eval 한계로 11분+ 정체(state=R) → **headline(KL1·KL2)+k10만 세부**로 전환(기존 도구 무수정, decompose에 max_iter/tol·부분 호출만). 12bit 3·4차 maxent 비수렴은 기록.

## 12. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규 파일만). 기존 소스·results2 기존 json·Drebin 원본 무수정.

## 13. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] flow 규모·도구 재사용·Drebin 로드·계산 한계 확인·기록.
- [x] Drebin 3방식(고빈도/고분산/카테고리) 각 2차/≥3차 비중(+k10 3/4/≥5차 세부), 3split 일관.
- [x] flow(2차 73~99%) 대비표 — Drebin ≥3차 최대 21.8% < UNSW 27%.
- [x] 고차 집중 조합(permission·api·call).
- [x] self-check: 합성 통과 + flow UNSW 27% 재현(Δ=0).
- [x] 고차 지배 여부 판정 재료(복수 방식·split 일관): **flow보다 고차 지배 미지지**. 단정 금지.
- [x] 학습 없음·기존 무변경·무설치.

**남은 불확실성**
- 표본 3315 vs 상태 2^12=4096 → 희소·모멘트 표본오차(pooling으로 완화, flow도 유사). 12bit ≥3차 세부(3/4차 분리)는 maxent 비수렴으로 미측정(headline만).
- 부분집합=1376/5760 중 10~12개만. 다른 부분집합 규모(더 큰 k)·다른 인코딩·per-family(mixture 제거) 측정은 미수행 — 결과 달라질 여지.
- 가족 pooling이 mixture로 ≥3차를 일부 유도했을 수 있음(측정 대상이자 한계).

**다음 추천 단계**
1. per-family로 분해(mixture 효과 분리) — 가족 내부 feature 조합의 고차성이 pooling과 다른지.
2. ≥3차가 최대인 highvar 방식으로 QCBM feature(10~12개) 선정 시 permission·api·call 조합 포함 검토(단, 고차 총량은 flow 수준임을 전제).
3. (해석 입력) 웹 AI에: "Drebin binary feature 차수측정 결과 ≥3차 비중 최대 21.8%(highvar 12bit) < flow UNSW 27%, 대부분 2차 지배(78~98%) — malware가 flow보다 고차라는 전제는 이 잣대에서 미지지. 이 부분집합·차수분해 한정, 우위 언급 없음."
