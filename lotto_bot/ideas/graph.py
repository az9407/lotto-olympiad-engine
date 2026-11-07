"""그래프·조합론·회차 구조 공식.

모두 df 마지막 회차 바로 다음 회차 점수를 df 안의 데이터만으로 계산합니다. 점수가 높을수록 위.

번호 단위, scores_next(df) → {이름: 길이 46 배열, 0번 무시}
- 그래프:직전보완      직전 회차 번호들과 과거에 기대보다 자주 함께 나온 번호, 동시출현 표준잔차 합
- 그래프:개인화페이지랭크 양의 동시출현 그래프에서 직전 회차 6개로 출발한 확산량 ÷ 전체 페이지랭크
- 그래프:삼각폐쇄      직전 회차 두 번호와 같은 회차에 함께 나왔던 삼중 조합 수, 기대값 차감
- 그래프:유사회차다음   직전 회차와 3개 이상 겹친 과거 회차들의 바로 다음 회차 번호
- 그래프:저역통과      최근 출현 편차를 동시출현 그래프 라플라시안으로 퍼뜨린 이웃 몫
- 그래프:매개중심성     최근 회차들을 겹친 그래프에서 회차와 회차를 잇는 다리 번호
- 그래프:커뮤니티순환   루뱅 커뮤니티로 나눈 뒤 같은 커뮤니티의 최근 출현 편차
- 그래프:앙상블        위 7개의 표준점수 평균
조합 단위, combo_scores_next(df) → {이름: 8,145,060 길이 배열, lotto_engine.space()["COMB"] 사전순}
- 구조:마르코프        합계구간·홀수·저번호·연번·번호대최대 상태의 전이 확률 ÷ 조합론 기본 확률
- 자리:시계열          정렬된 자리 x1~x6 각각 직전 값 근처였던 과거의 다음 값 분포 ÷ 자리 기본 확률
- 삼중:재출현          과거 회차와 공유하는 삼중 조합 수의 감쇠 합

매개변수 FINAL은 300~999회 성적만으로 골랐습니다.
"""
from itertools import chain, combinations
from math import comb

import numpy as np

P6 = 6 / 45
PAIR_P = 30 / 1980           # 두 번호가 한 회차에 같이 나올 확률
NCOMB = comb(45, 6)

# 300~999회로 고른 매개변수와 방향. 방향 -1은 점수를 뒤집어 씀
FINAL = {
    "그래프:직전보완": dict(W=0, sign=-1),
    "그래프:개인화페이지랭크": dict(W=200, alpha=0.85, sign=-1),
    "그래프:삼각폐쇄": dict(W=300, sign=-1),
    "그래프:유사회차다음": dict(lag=1, kind="exp", k=1.0, sign=-1),
    "그래프:저역통과": dict(w=5, tau=10.0, sign=-1),
    "그래프:매개중심성": dict(W=4, sign=1),
    "그래프:커뮤니티순환": dict(Wl=200, w=5, sign=-1),
}
COMBO_FINAL = {
    "구조:마르코프": dict(alpha=200.0, order=2, sign=-1),
    "자리:시계열": dict(method="knn2", h=0.0, K=50, sign=-1),
    "삼중:재출현": dict(half=30, sign=1),
}
FAMILIES = list(FINAL)
NAMES = FAMILIES + ["그래프:앙상블"]
COMBO_NAMES = list(COMBO_FINAL)


# ---------------- 공통 ----------------
def hist_arrays(df):
    X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.intp), 1)
    A = np.zeros((len(X), 46))
    A[np.arange(len(X))[:, None], X] = 1
    return X, A


def _win(n, W):
    """끝에서 W개, 0이면 전체"""
    return 0 if W <= 0 else max(0, n - W)


def _lift(Ah):
    """동시출현 표준잔차 (관측 - 기대) / √기대, 대각 0"""
    O = Ah.T @ Ah
    E = max(len(Ah) * PAIR_P, 1e-9)
    Z = (O - E) / np.sqrt(E)
    np.fill_diagonal(Z, 0)
    Z[0, :] = 0
    Z[:, 0] = 0
    return Z


def _out(v45):
    s = np.zeros(46)
    s[1:] = v45
    return s


def _z(s):
    v = s[1:]
    sd = v.std()
    o = np.zeros(46)
    o[1:] = (v - v.mean()) / (sd if sd > 0 else 1)
    return o


# ---------------- 번호 단위 부품, 입력은 다음 회차 직전까지의 X, A ----------------
def f_prev_complement(X, A, W):
    n = len(X)
    Z = _lift(A[_win(n - 1, W):n - 1])        # 직전 회차는 빼고 그 전 W회
    return _out(Z[X[-1]].sum(0)[1:])


def f_ppr(X, A, W, alpha):
    n = len(X)
    Zp = np.maximum(_lift(A[_win(n - 1, W):n - 1]), 0)[1:, 1:]
    rs = Zp.sum(1, keepdims=True)
    P = np.where(rs > 0, Zp / np.where(rs > 0, rs, 1), 1 / 45)
    e = np.zeros(45)
    e[X[-1] - 1] = 1 / 6
    pr = np.full(45, 1 / 45)
    pp = e.copy()
    for _ in range(60):
        pr = (1 - alpha) / 45 + alpha * P.T @ pr
        pp = (1 - alpha) * e + alpha * P.T @ pp
    diff = P.T @ pp                            # 출발 질량을 뺀 확산 몫
    return _out(np.log(diff + 1e-12) - np.log(pr + 1e-12))


def f_triangle(X, A, W):
    n = len(X)
    lo = _win(n - 1, W)
    Ah = A[lo:n - 1]
    M = A[-1]
    m = Ah @ M
    c2 = lambda k: k * (k - 1) / 2
    s_out = Ah.T @ c2(m)                       # 직전 번호가 아닌 j: 함께 나온 직전 번호 쌍 수
    s_in = Ah.T @ c2(np.maximum(m - 1, 0))     # 직전 번호 j: 자기 빼고
    freq = Ah.sum(0)
    E = np.where(M > 0, 100 / 946, 150 / 946)  # 무작위일 때 기대 쌍 수
    s = np.where(M > 0, s_in, s_out) - E * freq
    return _out(s[1:])


def f_analog(X, A, lag, kind, k):
    n = len(X)
    M1 = A[-1]
    if lag == 1:
        m = A[:n - 1] @ M1                     # s = 0..n-2, 다음 회차 s+1 은 이미 있음
        nxt = A[1:n]
    else:
        M2 = A[-2]
        m = A[1:n - 1] @ M1 + A[0:n - 2] @ M2
        nxt = A[2:n]
    w = (m >= k).astype(float) if kind == "ge" else np.exp(k * m)
    if w.sum() <= 0:
        return np.zeros(46)
    s = w @ (nxt - P6) / w.sum()
    s[0] = 0
    return s


def f_lowpass(X, A, w, tau):
    Zp = np.maximum(_lift(A), 0)[1:, 1:]
    d = Zp.sum(1)
    di = 1 / np.sqrt(np.where(d > 0, d, 1))
    L = np.eye(45) - di[:, None] * Zp * di[None, :]
    Mx = np.linalg.inv(np.eye(45) + tau * L)
    r = A[-w:, 1:].sum(0) - w * P6
    s = Mx @ r - np.diag(Mx) * r                # 자기 몫을 뺀 이웃 몫
    return _out(s)


def f_betweenness(X, A, W):
    import networkx as nx
    G = nx.Graph()
    for row in X[-W:]:
        G.add_edges_from(combinations(row.tolist(), 2))
    bc = nx.betweenness_centrality(G)
    s = np.zeros(46)
    for k, v in bc.items():
        s[k] = v
    return s


_COMM = {}


def _louvain(X, A, Wl):
    import networkx as nx
    n = len(X)
    key = (n, Wl, X[-1].tobytes())
    if key not in _COMM:
        if len(_COMM) > 64:
            _COMM.clear()
        Zp = np.maximum(_lift(A[_win(n, Wl):n]), 0)
        G = nx.Graph()
        G.add_nodes_from(range(1, 46))
        iu = np.argwhere(np.triu(Zp[1:, 1:], 1) > 0) + 1
        G.add_weighted_edges_from((int(a), int(b), float(Zp[a, b])) for a, b in iu)
        _COMM[key] = nx.community.louvain_communities(G, weight="weight", seed=0)
    return _COMM[key]


def f_community(X, A, Wl, w):
    comms = _louvain(X, A, Wl)
    cnt = A[-w:].sum(0)
    s = np.zeros(46)
    for C in comms:
        C = list(C)
        tot = cnt[C].sum()
        k = len(C) - 1
        for j in C:
            s[j] = 0 if k == 0 else ((tot - cnt[j]) - w * P6 * k) / k
    return s


FUNCS = {
    "그래프:직전보완": f_prev_complement,
    "그래프:개인화페이지랭크": f_ppr,
    "그래프:삼각폐쇄": f_triangle,
    "그래프:유사회차다음": f_analog,
    "그래프:저역통과": f_lowpass,
    "그래프:매개중심성": f_betweenness,
    "그래프:커뮤니티순환": f_community,
}


def family_score(name, X, A, params):
    p = {k: v for k, v in params.items() if k != "sign"}
    return params.get("sign", 1) * FUNCS[name](X, A, **p)


def scores_from_hist(X, A):
    out = {nm: family_score(nm, X, A, FINAL[nm]) for nm in FAMILIES}
    out["그래프:앙상블"] = np.mean([_z(out[nm]) for nm in FAMILIES], 0)
    return out


def scores_next(df):
    X, A = hist_arrays(df)
    return scores_from_hist(X, A)


# ---------------- 조합 단위 공용 ----------------
_SP = {}


def comb_space():
    """lotto_engine.space()["COMB"] 와 같은 사전순 8,145,060×6 int8"""
    if "COMB" not in _SP:
        _SP["COMB"] = np.fromiter(chain.from_iterable(combinations(range(1, 46), 6)), dtype=np.int8,
                                  count=NCOMB * 6).reshape(-1, 6)
    return _SP["COMB"]


_LEXCUM = np.zeros((6, 46), np.int64)
for _i in range(6):
    _LEXCUM[_i, 1:] = np.cumsum([comb(45 - v, 5 - _i) for v in range(1, 46)])


def lex_index(C):
    """정렬된 조합(…×6, 1~45)의 사전순 번호"""
    C = np.asarray(C, np.int64)
    prev = np.concatenate([np.zeros(C.shape[:-1] + (1,), np.int64), C[..., :-1]], -1)
    r = np.zeros(C.shape[:-1], np.int64)
    for i in range(6):
        r += _LEXCUM[i, C[..., i] - 1] - _LEXCUM[i, prev[..., i]]
    return r


# ---- 구조 상태 ----
STRUCT_SIZES = (7, 7, 7, 4, 4)     # 합계구간, 홀수개수, 1~22개수, 연번쌍(3+), 번호대 최대개수(1~4+)
_DEC = np.minimum(np.arange(46) // 10, 4)


def _sum_edges():
    if "edges" not in _SP:
        C = comb_space()
        s = C.astype(np.int16).sum(1)
        cnt = np.bincount(s, minlength=256).cumsum() / NCOMB
        _SP["edges"] = np.array([int(np.searchsorted(cnt, q)) for q in np.arange(1, 7) / 7])
    return _SP["edges"]


def struct_feats(C):
    """조합 배열 → (…, 5) 상태값"""
    C = np.asarray(C)
    Ci = C.astype(np.int16)
    sb = np.searchsorted(_sum_edges(), Ci.sum(-1), side="left")
    odd = (Ci % 2).sum(-1)
    low = (Ci <= 22).sum(-1)
    con = np.minimum((np.diff(Ci, axis=-1) == 1).sum(-1), 3)
    D = _DEC[Ci]
    dm = np.max(np.stack([(D == d).sum(-1) for d in range(5)], -1), -1)
    dm = np.minimum(dm, 4) - 1
    return np.stack([sb, odd, low, con, dm], -1).astype(np.int16)


def struct_space():
    if "cls" not in _SP:
        C = comb_space()
        F = np.concatenate([struct_feats(C[i:i + 1_000_000]) for i in range(0, NCOMB, 1_000_000)])
        cls = np.ravel_multi_index(tuple(F.T), STRUCT_SIZES).astype(np.int32)
        _SP["cls"] = cls
        _SP["cls_cnt"] = np.bincount(cls, minlength=int(np.prod(STRUCT_SIZES))).astype(np.int64)
        _SP["base"] = [np.bincount(F[:, f], minlength=STRUCT_SIZES[f]) / NCOMB for f in range(5)]
    return _SP


def struct_class_scores(X, alpha, order):
    """클래스 5,488개 각각의 점수: 특징별 마르코프 전이 확률 ÷ 기본 확률의 로그 합"""
    sp = struct_space()
    F = struct_feats(X)
    grid = np.array(np.unravel_index(np.arange(int(np.prod(STRUCT_SIZES))), STRUCT_SIZES)).T
    sc = np.zeros(len(grid))
    n = len(F)
    for f, V in enumerate(STRUCT_SIZES):
        pi = sp["base"][f]
        if order == 1:
            prev, cur, u = F[:n - 1, f], F[1:, f], F[-1, f]
            mask = prev == u
        else:
            mask = (F[:n - 2, f] == F[-2, f]) & (F[1:n - 1, f] == F[-1, f])
            cur = F[2:, f]
        cnt = np.bincount(cur[mask], minlength=V)
        p = (cnt + alpha * pi) / (cnt.sum() + alpha)
        lift = np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(pi, 1e-300))
        sc += lift[grid[:, f]]
    return sc


def rank_by_class(class_scores, actual):
    sp = struct_space()
    a = int(np.ravel_multi_index(tuple(struct_feats(np.sort(actual))), STRUCT_SIZES))
    s = class_scores[a]
    cnt = sp["cls_cnt"]
    return float((cnt[class_scores > s].sum() + 0.5 * cnt[class_scores == s].sum()) / NCOMB * 100)


# ---- 자리 ----
_PI_POS = np.zeros((6, 46))
for _i in range(6):
    for _v in range(1, 46):
        _PI_POS[_i, _v] = comb(_v - 1, _i) * comb(45 - _v, 5 - _i) / NCOMB


def position_scores(X, method, h, K, alpha=30.0):
    """(6, 46): 자리 i 에 값 v 가 올 조건부 확률 ÷ 자리 기본 확률의 로그"""
    n = len(X)
    f = np.zeros((6, 46))
    if method == "kern":
        Wt = np.exp(-((X[:n - 1] - X[-1]) ** 2) / (2 * h * h))   # (n-1, 6) 자리별 가중
        nxt = X[1:]
    else:
        lag = 2 if method == "knn2" else 1
        if lag == 1:
            d = ((X[:n - 1] - X[-1]) ** 2).sum(1)
            nxt = X[1:]
        else:
            d = ((X[1:n - 1] - X[-1]) ** 2).sum(1) + ((X[:n - 2] - X[-2]) ** 2).sum(1)
            nxt = X[2:]
        kk = min(K, len(d))
        nn = np.argpartition(d, kk - 1)[:kk]
        Wt = np.zeros((len(d), 6))
        Wt[nn] = 1
    for i in range(6):
        cnt = np.bincount(nxt[:, i], weights=Wt[:, i], minlength=46)
        p = (cnt + alpha * _PI_POS[i]) / (Wt[:, i].sum() + alpha)
        valid = _PI_POS[i] > 0
        f[i, valid] = np.log(p[valid]) - np.log(_PI_POS[i, valid])
    return f


def rank_position_dp(f, actual, L=3000):
    """자리 점수 합의 정확한 순위, 증가 수열 동적계획법. explore.rank_pct 와 같은 양자화"""
    valid = _PI_POS > 0
    vals = f[valid]
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return 50.0
    q = np.where(valid, np.rint((f - lo) / (hi - lo) * L), 0).astype(int)
    Qmax = 6 * L + 1
    D = np.zeros((46, Qmax))
    for v in range(1, 46):
        D[v, q[0, v]] = 1
    for i in range(1, 6):
        cum = np.cumsum(D, 0)
        Dn = np.zeros_like(D)
        for w in range(i + 1, 46):
            x = q[i, w]
            Dn[w, x:] = cum[w - 1, :Qmax - x]
        D = Dn
    row = D.sum(0)
    a = sorted(int(v) for v in actual)
    qa = int(sum(q[i, a[i]] for i in range(6)))
    return float((row[qa + 1:].sum() + 0.5 * row[qa]) / NCOMB * 100)


# ---- 삼중 ----
_T3 = np.array(list(combinations(range(42), 3)), np.intp)
_CNT3 = np.stack([(_T3 < r).sum(1) for r in range(43)])       # 남은 42개 중 앞 r개보다 작은 원소 수
_BIN = np.array([[comb(a, b) for b in range(7)] for a in range(46)], np.int64)
_ALLV = np.arange(1, 46)


def _triple_combos(row):
    """한 회차 6개 번호의 삼중 20개 각각을 포함하는 조합 11,480개 사전순 번호 (20, 11480)
    사전순 번호 = 전체-1 - Σ C(45 - x_j, 7 - j), j 는 정렬 위치. 정렬 없이 위치를 표로 셈"""
    row = sorted(int(v) for v in row)
    out = np.empty((20, len(_T3)), np.int64)
    for j, tri in enumerate(combinations(row, 3)):
        tau = np.array(tri)
        rest = _ALLV[~np.isin(_ALLV, tau)]
        ct = np.searchsorted(tau, rest)            # 각 남은 번호보다 작은 삼중 원소 수
        s = np.zeros(len(_T3), np.int64)
        for k in range(3):
            s += _BIN[45 - rest, 6 - k - ct][_T3[:, k]]
        r = np.searchsorted(rest, tau)             # 각 삼중 원소보다 작은 남은 번호 수
        for m in range(3):
            s += _BIN[45 - tau[m]][6 - m - _CNT3[r[m]]]
        out[j] = NCOMB - 1 - s
    return out


class TripleState:
    """과거 회차와 공유하는 삼중 조합 수의 감쇠 합을 조합 전체에 누적. half=0 이면 감쇠 없음"""

    def __init__(self, half):
        self.half = half
        self.S = np.zeros(NCOMB)
        self.t = 0

    def add(self, row, idx=None):
        w = 1.0 if self.half <= 0 else 2.0 ** (self.t / self.half)
        for ix in (_triple_combos(row) if idx is None else idx):
            self.S[ix] += w
        self.t += 1
        if self.half > 0 and self.t % 400 == 0:   # 넘침 방지, 비율은 그대로
            sc = 2.0 ** (-self.t / self.half)
            self.S *= sc
            self.t = 0

    def rank(self, actual):
        s = self.S[int(lex_index(np.sort(actual)))]
        return float((np.count_nonzero(self.S > s) + 0.5 * np.count_nonzero(self.S == s)) / NCOMB * 100)


def combo_scores_next(df):
    X, A = hist_arrays(df)
    C = comb_space()
    out = {}
    p = COMBO_FINAL["구조:마르코프"]
    sp = struct_space()
    out["구조:마르코프"] = (p["sign"] * struct_class_scores(X, p["alpha"], p["order"]))[sp["cls"]].astype(np.float32)
    p = COMBO_FINAL["자리:시계열"]
    f = p["sign"] * position_scores(X, p["method"], p["h"], p["K"])
    s = np.zeros(NCOMB, np.float32)
    for i in range(6):
        s += f[i].astype(np.float32)[C[:, i]]
    out["자리:시계열"] = s
    p = COMBO_FINAL["삼중:재출현"]
    st = TripleState(p["half"])
    for row in X:
        st.add(row)
    out["삼중:재출현"] = p["sign"] * st.S
    return out
