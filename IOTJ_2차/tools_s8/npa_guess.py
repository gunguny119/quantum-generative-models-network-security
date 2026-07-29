"""
Gate S8 — NPA 추측확률 SDP (Bancal-Sheridan-Scarani NJP 16,033011 (2014) 방식).

브랜치별 모멘트 행렬 Γ_e ⪰ 0 (e = Eve 의 추측값). 정규화되지 않은 표현:
    Γ_e[I,I] = λ_e (브랜치 가중치),  Γ_e[I,M] = λ_e·⟨M⟩_e
따라서 Σ_e Γ_e[I,I] = 1.

관측량은 ±1 이항. 단항은 (A-단어, B-단어) 쌍으로 표현하고
Ax²=I, By²=I, [Ax,By]=0 로 환원한다. 실수 모멘트 행렬을 가정하므로
단항 m 과 그 역순 m† 를 동일시한다(실수 대칭 NPA).

레벨:
  "1+AB" : {I} ∪ {Ax} ∪ {By} ∪ {AxBy}
  "2"    : 길이 ≤2 인 모든 단어 (AxAx', ByBy' 포함)

★ 이 파일은 tools/ 를 수정하지 않는다(원칙 유지). 신규 디렉터리다.
"""
import itertools

import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# 단항 대수
# ---------------------------------------------------------------------------
def _reduce_word(w):
    """인접 동일 문자 제거 (Ax·Ax = I)."""
    out = []
    for c in w:
        if out and out[-1] == c:
            out.pop()
        else:
            out.append(c)
    return tuple(out)


def _canon(mon):
    """실수 대칭 가정: m 과 m† 중 사전순으로 작은 것을 대표로."""
    a, b = mon
    a, b = _reduce_word(a), _reduce_word(b)
    m1 = (a, b)
    m2 = (a[::-1], b[::-1])
    return min(m1, m2)


def _dagger(mon):
    a, b = mon
    return (a[::-1], b[::-1])


def _mul(mi, mj):
    """S_i† · S_j.  A 와 B 는 가환이므로 단어별로 이어붙인다."""
    ai, bi = _dagger(mi)
    aj, bj = mj
    return _canon((ai + aj, bi + bj))


def build_monomials(nA, nB, level):
    """생성 단항 목록(연산자 기저). level ∈ {"1+AB", "2"}."""
    A = list(range(nA))
    B = list(range(nB))
    mons = [((), ())]
    mons += [((x,), ()) for x in A]
    mons += [((), (y,)) for y in B]
    if level == "1+AB":
        mons += [((x,), (y,)) for x in A for y in B]
    elif level == "2":
        mons += [((x, xp), ()) for x in A for xp in A if x != xp]
        mons += [((), (y, yp)) for y in B for yp in B if y != yp]
        mons += [((x,), (y,)) for x in A for y in B]
    else:
        raise ValueError("level must be '1+AB' or '2'")
    # 중복 제거(순서 보존)
    seen, out = set(), []
    for m in mons:
        c = _canon(m)
        if c not in seen:
            seen.add(c)
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# SDP 구성
# ---------------------------------------------------------------------------
class GuessSDP:
    """
    P_guess = max Σ_e P(a=e | x=x_star, branch e)  (단일 관측자 단일 설정)
    제약 모드:
      mode="full" : 36셀(=nA·nB·4) 등식
      mode="S"    : CHSH S 값 하나만
    """

    def __init__(self, nA, nB, level, x_star, branches=(+1, -1)):
        self.nA, self.nB, self.level, self.x_star = nA, nB, level, x_star
        self.branches = branches
        self.mons = build_monomials(nA, nB, level)
        self.n = len(self.mons)
        # 곱 테이블: (i,j) → 대표 단항
        self.prod = [[_mul(self.mons[i], self.mons[j]) for j in range(self.n)]
                     for i in range(self.n)]
        keys = sorted({self.prod[i][j] for i in range(self.n) for j in range(self.n)})
        self.key_index = {k: t for t, k in enumerate(keys)}
        self.n_keys = len(keys)

    # -- 모멘트 벡터 → 대칭 행렬 --------------------------------------------
    def _gamma(self, xvec):
        rows = []
        for i in range(self.n):
            rows.append(cp.hstack([xvec[self.key_index[self.prod[i][j]]]
                                   for j in range(self.n)]))
        return cp.vstack(rows)

    def _k(self, mon):
        return self.key_index[_canon(mon)]

    # -- 본체 ---------------------------------------------------------------
    def solve(self, p_obs=None, S_obs=None, mode="full", chsh_idx=None,
              solver="SCS", eps=1e-8, max_iters=200000, verbose=False):
        """
        p_obs[x, y, a, b] with a,b ∈ {0,1} ↔ 결과 ±1 (0 → +1, 1 → −1).
        chsh_idx: S = Σ_{(x,y)∈K} s_{xy}⟨AxBy⟩ 의 (x, y, sign) 목록.
        """
        xs = [cp.Variable(self.n_keys) for _ in self.branches]
        cons = []
        for xv in xs:
            cons.append(self._gamma(xv) >> 0)
            cons.append(xv[self._k(((), ()))] >= 0)
        # 정규화
        cons.append(cp.sum([xv[self._k(((), ()))] for xv in xs]) == 1)

        def sgn(o):                      # 0 → +1, 1 → −1
            return 1.0 if o == 0 else -1.0

        if mode == "full":
            assert p_obs is not None
            for x in range(self.nA):
                for y in range(self.nB):
                    for a in (0, 1):
                        for b in (0, 1):
                            expr = 0
                            for xv in xs:
                                expr = expr + (xv[self._k(((), ()))]
                                               + sgn(a) * xv[self._k(((x,), ()))]
                                               + sgn(b) * xv[self._k(((), (y,)))]
                                               + sgn(a) * sgn(b) * xv[self._k(((x,), (y,)))]) / 4.0
                            cons.append(expr == float(p_obs[x, y, a, b]))
        elif mode == "S":
            assert S_obs is not None and chsh_idx is not None
            expr = 0
            for (x, y, s) in chsh_idx:
                for xv in xs:
                    expr = expr + s * xv[self._k(((x,), (y,)))]
            cons.append(expr == float(S_obs))
        else:
            raise ValueError(mode)

        # 목적함수: Σ_e (λ_e + e·λ_e⟨A_{x*}⟩_e)/2
        obj = 0
        for e, xv in zip(self.branches, xs):
            obj = obj + (xv[self._k(((), ()))]
                         + e * xv[self._k(((self.x_star,), ()))]) / 2.0

        prob = cp.Problem(cp.Maximize(obj), cons)
        kw = dict(verbose=verbose)
        if solver == "SCS":
            kw.update(eps=eps, max_iters=max_iters)
        prob.solve(solver=solver, **kw)

        pg = None if prob.value is None else float(prob.value)
        hmin = None
        if pg is not None and pg > 0:
            hmin = float(-np.log2(min(max(pg, 1e-300), 1.0)))
        return dict(status=prob.status, p_guess=pg, h_min=hmin,
                    lam=[float(xv.value[self._k(((), ()))]) if xv.value is not None else None
                         for xv in xs],
                    n_mon=self.n, n_keys=self.n_keys, level=self.level, mode=mode)


# ---------------------------------------------------------------------------
# 해석식 앵커 (Pironio et al. 2010, CHSH)
# ---------------------------------------------------------------------------
def pironio_hmin(S):
    v = 2.0 - S ** 2 / 4.0
    if v < 0:
        v = 0.0
    h = 1.0 - np.log2(1.0 + np.sqrt(v))
    return max(h, 0.0)


def isotropic_222(V):
    """CHSH 등방 박스: E = V/√2, 주변확률 0. p[x,y,a,b]."""
    E = np.array([[V / np.sqrt(2), V / np.sqrt(2)],
                  [V / np.sqrt(2), -V / np.sqrt(2)]])
    p = np.zeros((2, 2, 2, 2))
    for x in range(2):
        for y in range(2):
            for a in (0, 1):
                for b in (0, 1):
                    sa = 1.0 if a == 0 else -1.0
                    sb = 1.0 if b == 0 else -1.0
                    p[x, y, a, b] = (1 + sa * sb * E[x, y]) / 4.0
    return p, float(E[0, 0] + E[0, 1] + E[1, 0] - E[1, 1])


CHSH_SIGNS_222 = [(0, 0, +1), (0, 1, +1), (1, 0, +1), (1, 1, -1)]
