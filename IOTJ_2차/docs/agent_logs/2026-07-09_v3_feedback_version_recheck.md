# 논문 "피드백 반영 버전(v3)" 재점검 — 실제 실험과의 최종 대조

- 작성: 2026-07-09 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 무수정. 본 .md 1개만 신규.
- 대상: 세 번째 판본("논문 초안 보고.pdf", 제목이 *Contextual Quantitative Analysis of Quantum Generative Models in IoT Security*로 변경, Algorithm 1 의사코드화, **Appendix A에 명시적 계산 단계 추가**).
- 앞선 점검(`2026-07-09_paper_vs_experiment_verification.md`, F-1~F-9)의 **후속**. 여기서는 (1) 이전 9건 반영 여부, (2) **새 판본에서 새로 드러난 오류**를 코드 검산으로 확정.
- 대조 원본: `tools/gap_vs_cf_v0.py`, `tools/kink_origin_v1.py`, `results2/gate_e/gate_e.json`, `results2/gate_r3/r3.json`.

---

## 0. 총평 (한 줄)
**major 재작성 불필요.** 실험·결과·결론(gap≈c·CF², c=1/6·1/8, CF=0 필연, 앵커) 전부 저장값과 일치. **국소 수정 4곳**만 필요하며, 그중 **Appendix A Step 4/5의 유도 산술이 실제로 틀림(F-10, 치명)** — 단 최종 계수는 옳음.

---

## 1. 🔴 F-10 [치명·신규] Appendix A Step 4/5 유도 산술 오류 (결과는 맞음)

### 논문 서술 (Step 4, CHSH)
> "the crossing point $e^\partial$ is the uniform behavior ($e^\partial_i=1/4$) … $u_i=\pm 1/2$ … each cell contributes $u_i^2/e^\partial_i=(1/2)^2/(1/4)=1$ … $c_{\text{CHSH}}=\tfrac12\chi^2_F(u)=1/6$."

### 코드 검산 (결정적)
`e(\lambda)=\lambda\,\text{PR}+(1-\lambda)\,\text{uniform}`, `CF=max(0,2\lambda-1)` → **경계(CF=0⁺)는 $\lambda=1/2$** (uniform은 $\lambda=0$ = 폴리토프 **깊은 내부**, 경계 아님).

| 항목 | 논문 주장 | **실제(검산)** |
|---|---|---|
| 경계 교차점 $e^\partial=e(\lambda{=}1/2)$ | uniform = 1/4 | **셀 3/8, 1/8** (uniform 아님) |
| 방향 $u=(\text{PR}-\text{uniform})/2$ | $\pm1/2$ | **$\pm1/8$** |
| $\chi^2_F(u)=\sum_i u_i^2/e^\partial_i$ (컨텍스트 평균) | (암시) | **1/3** |
| $c=\tfrac12\chi^2_F$ | 1/6 | **½·(1/3)=1/6** ✅ |

**논문 숫자(e=1/4, u=±1/2)를 그대로 대입하면 c=½·4=2** → 1/6이 아님. 즉 **중간 단계가 자기모순**(대입값과 결론이 불일치). 정답 c=1/6은 `gate_e.json c_pred=0.16667`과 일치하므로 **결론은 옳고 유도만 틀림**.

### KCBS(Step 5)도 동일 구조 오류
5-cycle 경계는 **$\lambda=3/5$**, $e^\partial$ 셀 = **0.4/0.1**, $u_i=\pm1/10$ → $\chi^2=1/4$ → **c=1/8** ✅. 논문의 "$e^\partial$ = uniform" 전제는 여기서도 틀림(검산으로 확인).

### 권장 수정 (Step 4, 2~3줄 교체)
$$e(\text{CF})=e^\partial+\text{CF}\cdot u,\quad e^\partial=\tfrac{\text{PR}+U}{2}\ (\text{셀 }3/8,1/8),\quad u=\tfrac{\text{PR}-U}{2}\ (u_i=\pm1/8)$$
$$\chi^2_{\text{ctx}}=2\cdot\frac{(1/8)^2}{3/8}+2\cdot\frac{(1/8)^2}{1/8}=\frac{1}{12}+\frac14=\frac13,\qquad c=\tfrac12\cdot\tfrac13=1/6$$
KCBS도 $e^\partial$ 셀 0.4/0.1, $u_i=\pm1/10$ → $\chi^2=1/4$ → $c=1/8$ 로 병행 수정.

**★ 리뷰어가 산술을 따라가면 반드시 걸림.** 이전 판본은 "sum and normalize to 1/6"로 뭉개져 안 보였으나, 이번에 계산을 명시화하며 오류가 노출됨.

---

## 2. 🟠 F-11 [회귀·신규] Fig 2 캡션 — 이전 F-5 수정이 삭제됨
- 2판본 캡션엔 있던 문장("schematic shows one rotation per qubit; model applies a distinct U per setting — 2 settings/qubit, 4 settings, 8 params")이 **3판본에서 삭제**됨. 현재: *"The two learners: (a) the two-qubit QCBM and (b) the classical latent-class mixture ladder."* 뿐.
- 그림 2(a)엔 측정회전 박스가 큐빗당 1개(U(θ₁,φ₁), U(θ₂,φ₂))인데 본문 IV-B는 **4설정·8파라미터** → 시각 불일치 재발.
- **조치**: 삭제된 문장 복구.

---

## 3. 🟡 경미 3건 (신규)

### F-12 Algorithm 1 line 13 — `gap_max ← c·CF²`
실제 def-b 상한 **2.9×10⁻⁹**는 부트스트랩 **CI 상단** `cf_upper=1.525×10⁻⁴`로 계산됨(`r3.json`, `gap_upper=c·cf_upper²`, c=1/8). 점추정 CF(7.6×10⁻⁵)로는 7.3×10⁻¹⁰라 재현 안 됨. → line 13을 **`gap_max ← c·CF_upper²`**(또는 "CI 상단 사용" 명시)로.

### F-13 "verified to 1–2%" (Contributions·Conclusion)
Table II 실제 최대 rel.err = **2.2%**(KCBS uniform). 본문 V-C는 "1.3–2.2%"로 정확 → 초록/기여/결론도 **"1.3–2.2%"**로 통일.

### F-14 Algorithm 1 line 4 — `sign(X−m)∈{−1,+1}`
코드는 `(X > median)→{0,1}`(sign은 동점 시 0 반환). CF엔 무영향(표기만). 또 line 1–6의 "feature n개 선택" 골격은 def-a용 — **def-b(한 feature의 3 시간지연)**는 이 골격에 안 맞아 별도 분기 표기 권장.

---

## 4. ✅ 이전 9건(F-1~F-9) 반영 확인 — 전부 정확히 들어감
| 이전 발견 | 3판본 위치 | 상태 |
|---|---|---|
| F-1 def-a 상한 5.6×10⁻³¹ | Table III | ✅ |
| §7 감사 시나리오 n-cycle | IV-D, IV-E, Algorithm 1 | ✅ |
| F-8 marginal 미고정(V-E) | 각주 1 | ✅ |
| F-2 앵커 편차 기준 곡선 | VI-A(fitted c≈0.18 / CF²/6 ~6.7%) | ✅ |
| F-3 Tsirelson KLq 스코프 | V-A 본문(Delft KLq=0.0045) | ✅ |
| F-4 Fig4 캡션 c | Fig 4 캡션(예측1/6, 적합0.170) | ✅ |
| F-6 초록 상한 | "∼3×10⁻⁹" | ✅ |
| F-7 2×5 / gate r5 | IV-E(def-a 5전략), App C(gate_r3 18셀) | ✅ |
| F-9 c=1/8 차용 | 각주 3 | ✅ |
- Table I(8값)·Table II(α·c_pred·c_fit·rel_err·R²)·Table III(KCBS·Delft)·AIC·provenance(MD5·기기행수)·PennyLane 6.66e-16 = 저장값과 일치(재확인).

---

## 5. 여전히 미검증 (앞 판본 §8과 동일)
- **III-A "예비 QCBM 우위 없음"**: IOTJ_1차 결과와 미대조.
- Section II 문헌 특성화, 참고문헌 서지: 미확인(원논문 필요).

---

## 6. 수정 우선순위
1. **F-10 (🔴 필수)**: Appendix A Step 4/5 유도 2~3줄 교체(e∂=3/8·1/8, u=±1/8; KCBS 0.4·0.1, ±1/10). 결론 c=1/6·1/8 유지.
2. **F-11 (🟠)**: Fig 2 캡션 문장 복구.
3. **F-12 (🟡)**: Algorithm 1 line 13 → `c·CF_upper²`.
4. **F-13/F-14 (🟡)**: "1.3–2.2%" 통일, line 4 이진화 표기·def-b 분기.

→ **전부 국소 수정. major 재작성 불필요. 실험/결과/결론 무변경.**

## 7. 실행한 명령 (읽기 수준만)
`python3 -c`(gap_vs_cf_v0·kink_origin_v1 로 경계점 e∂·방향 u·χ² 검산, r3/gate_e.json 값 확인). 값 변경·모델 재실행 없음.

## 8. 생성한 .md 로그 경로
`docs/agent_logs/2026-07-09_v3_feedback_version_recheck.md` (본 파일, 유일한 신규).
