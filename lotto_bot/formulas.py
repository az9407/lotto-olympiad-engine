"""로또 공식 후보군.

모두 그 회차 이전 데이터만 보고 8,145,060개 조합 전체에 점수를 매깁니다. 점수가 높을수록 위.
- 번호별 공식: 번호마다 점수를 매기고 6개를 더함
- 조합 공식: 쌍, 자리, 구조, 간격처럼 조합 전체를 보고 매김
- 섞은 공식: 여러 공식의 표준점수 평균
"""
import numpy as np

import lotto_engine as LE

BASE_P = 6 / 45
_CACHE = {}


def _space():
    if not _CACHE:
        sp = LE.space()
        C = sp["COMB"]
        F = sp["BASE"]
        bins = {
            "sum": F["sum"] // 10, "tail_sum": F["tail_sum"] // 4, "span": F["span"] // 3,
            "odd": F["odd"], "low": F["low"], "ac": F["ac"], "consec": F["consec"],
            "max_gap": np.minimum(F["max_gap"], 25), "same_tail_pairs": F["same_tail_pairs"],
            "prime": F["prime"], "gcd_pairs": F["gcd_pairs"], "mult3": F["mult3"],
            **{f"dec{d}": F[f"dec{d}"] for d in range(5)},
        }
        _CACHE["C"] = C
        _CACHE["bins"] = {k: v.astype(np.int16) for k, v in bins.items()}
        _CACHE["pos"] = {f"x{i+1}": C[:, i].astype(np.int16) for i in range(6)}
        _CACHE["gaps"] = {f"gap{i+1}": F[f"gap{i+1}"] for i in range(5)}
        sz = {}
        for group in ("bins", "pos", "gaps"):
            for k, v in _CACHE[group].items():
                sz[k] = np.bincount(v, minlength=int(v.max()) + 1) / len(v)
        _CACHE["pspace"] = sz
    return _CACHE


def hist_arrays(df):
    X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.intp), 1)
    B = df["bonus"].to_numpy(np.intp)
    return X, B


def membership(X):
    M = np.zeros((len(X), 46), bool)
    M[np.arange(len(X))[:, None], X] = True
    return M


def ew_weights(n, half):
    return 0.5 ** (np.arange(n)[::-1] / half)


# ---------------- 번호별 공식 ----------------
def num_hot(X, half):
    M = membership(X)
    w = ew_weights(len(X), half) if half else np.ones(len(X))
    c = (M * w[:, None]).sum(0)[1:]
    p = (c + 1) / (c.sum() + 45)
    s = np.zeros(46)
    s[1:] = np.log(p * 45)
    return s


def num_cold(X):
    last = np.full(46, -1)
    for t, row in enumerate(X):
        last[row] = t
    ab = len(X) - last
    s = np.zeros(46)
    s[1:] = np.log(ab[1:] + 1.0)
    return s


def num_due(X):
    last = np.full(46, -1)
    M = membership(X)
    cnt = M.sum(0)
    for t, row in enumerate(X):
        last[row] = t
    ab = len(X) - last
    gap = len(X) / np.maximum(cnt, 1)
    s = np.zeros(46)
    s[1:] = np.log((ab[1:] + 1.0) / gap[1:])
    return s


def num_markov(X, lag):
    M = membership(X).astype(float)
    A, Bm = M[:-lag], M[lag:]
    co = A.T @ Bm
    occ = A.sum(0)
    P = (co + 10 * BASE_P) / (occ[:, None] + 10)
    last = X[-lag]
    s = np.zeros(46)
    s[1:] = np.log(P[last][:, 1:].mean(0) / BASE_P)
    return s


def num_carry(X, B):
    M = membership(X)
    prev, cur = M[:-1], M[1:]
    nb = np.zeros_like(prev)
    nb[:, 1:-1] = prev[:, :-2] | prev[:, 2:]
    nb[:, 45] |= prev[:, 44]
    nb &= ~prev
    nb[:, 0] = False
    bm = np.zeros_like(prev)
    bm[np.arange(len(prev)), B[:-1]] = True
    bm &= ~prev
    other = ~(prev | nb | bm)
    other[:, 0] = False

    def rate(mask):
        return (cur & mask).sum() / max(mask.sum(), 1)

    rc, rn, rb, ro = rate(prev), rate(nb), rate(bm), rate(other)
    last = M[-1]
    lnb = np.zeros(46, bool)
    lnb[1:-1] = last[:-2] | last[2:]
    lnb[45] |= last[44]
    lnb &= ~last
    s = np.full(46, np.log(ro / BASE_P))
    s[lnb] = np.log(rn / BASE_P)
    if not last[B[-1]]:
        s[B[-1]] = np.log(rb / BASE_P)
    s[last] = np.log(rc / BASE_P)
    s[0] = 0
    return s


def num_group_hot(X, key, half=30):
    M = membership(X)
    w = ew_weights(len(X), half)
    c = (M * w[:, None]).sum(0)
    groups = np.array([key(n) for n in range(46)])
    out = np.zeros(46)
    ids = sorted(set(groups[1:]))
    tot = {g: c[1:][groups[1:] == g].sum() for g in ids}
    size = {g: (groups[1:] == g).sum() for g in ids}
    allc = sum(tot.values())
    for n in range(1, 46):
        g = groups[n]
        out[n] = np.log(((tot[g] + 1) / (allc + len(ids))) / (size[g] / 45))
    return out


def combo_from_num(s):
    C = _space()["C"]
    return s.astype(np.float32)[C].sum(1)


# ---------------- 조합 공식 ----------------
def lr_score(group, hist_vals, W, k=20.0):
    sp = _space()
    S = np.zeros(len(sp["C"]), np.float32)
    for name, arr in sp[group].items():
        ps = sp["pspace"][name]
        hv = hist_vals[name][-W:] if W else hist_vals[name]
        hv = np.clip(hv, 0, len(ps) - 1)
        cnt = np.bincount(hv, minlength=len(ps)).astype(float)
        ph = (cnt + k * ps) / (len(hv) + k)
        tab = np.where(ps > 0, np.log(np.maximum(ph, 1e-12) / np.maximum(ps, 1e-12)), 0).astype(np.float32)
        S += tab[arr]
    return S


def hist_feature_values(X):
    f = LE.base_feats(X.astype(np.int16))
    vals = {
        "sum": f["sum"] // 10, "tail_sum": f["tail_sum"] // 4, "span": f["span"] // 3,
        "odd": f["odd"], "low": f["low"], "ac": f["ac"], "consec": f["consec"],
        "max_gap": np.minimum(f["max_gap"], 25), "same_tail_pairs": f["same_tail_pairs"],
        "prime": f["prime"], "gcd_pairs": f["gcd_pairs"], "mult3": f["mult3"],
        **{f"dec{d}": f[f"dec{d}"] for d in range(5)},
        **{f"x{i+1}": X[:, i] for i in range(6)},
        **{f"gap{i+1}": f[f"gap{i+1}"] for i in range(5)},
    }
    return {k: np.asarray(v, np.intp) for k, v in vals.items()}


def pair_score(X, W):
    sp = _space()
    C = sp["C"]
    Xw = X[-W:] if W else X
    cnt = np.zeros((46, 46))
    for row in Xw:
        for i in range(6):
            for j in range(i + 1, 6):
                cnt[row[i], row[j]] += 1
    E = len(Xw) * 15 / (45 * 44 / 2)
    tab = np.log((cnt + 1) / (E + 1)).astype(np.float32)
    S = np.zeros(len(C), np.float32)
    for i in range(6):
        for j in range(i + 1, 6):
            S += tab[C[:, i], C[:, j]]
    return S


# ---------------- 목록 ----------------
NUM_FORMULAS = {
    "빈도_최근10": lambda X, B: num_hot(X, 10),
    "빈도_최근30": lambda X, B: num_hot(X, 30),
    "빈도_최근100": lambda X, B: num_hot(X, 100),
    "빈도_전체": lambda X, B: num_hot(X, 0),
    "역빈도_최근30": lambda X, B: -num_hot(X, 30),
    "미출현": lambda X, B: num_cold(X),
    "미출현_대비주기": lambda X, B: num_due(X),
    "상승세": lambda X, B: num_hot(X, 10) - num_hot(X, 100),
    "전이_직전회": lambda X, B: num_markov(X, 1),
    "전이_2회전": lambda X, B: num_markov(X, 2),
    "이월이웃보너스": lambda X, B: num_carry(X, B),
    "끝수_빈도": lambda X, B: num_group_hot(X, lambda n: n % 10),
    "번호대_빈도": lambda X, B: num_group_hot(X, lambda n: n // 10),
}
COMBO_FORMULAS = {
    "쌍_최근300": lambda X, H: pair_score(X, 300),
    "쌍_전체": lambda X, H: pair_score(X, 0),
    "자리분포_최근200": lambda X, H: lr_score("pos", H, 200),
    "구조_최근100": lambda X, H: lr_score("bins", H, 100),
    "구조_최근300": lambda X, H: lr_score("bins", H, 300),
    "간격분포_최근200": lambda X, H: lr_score("gaps", H, 200),
}
BLENDS = {
    "섞기_번호": ["빈도_최근30", "미출현_대비주기", "전이_직전회", "이월이웃보너스"],
    "섞기_구조": ["구조_최근300", "자리분포_최근200", "간격분포_최근200", "쌍_최근300"],
    "섞기_전체": list(NUM_FORMULAS) + list(COMBO_FORMULAS),
}
ALL = list(NUM_FORMULAS) + list(COMBO_FORMULAS) + list(BLENDS)


def zscore(S):
    S = S.astype(np.float32)
    sd = S.std()
    return (S - S.mean()) / sd if sd > 0 else S * 0


def score_all(df_hist, names=None):
    """이름 → 8,145,060개 점수. names가 있으면 그것만, 섞기는 필요한 재료까지 계산"""
    names = list(names or ALL)
    need = set()
    for n in names:
        need.update(BLENDS.get(n, [n]))
    X, B = hist_arrays(df_hist)
    H = hist_feature_values(X) if need & set(COMBO_FORMULAS) else None
    Z = {}
    for n in need:
        if n in NUM_FORMULAS:
            Z[n] = zscore(combo_from_num(NUM_FORMULAS[n](X, B)))
        elif n in COMBO_FORMULAS:
            Z[n] = zscore(COMBO_FORMULAS[n](X, H))
    out = {}
    for n in names:
        if n in BLENDS:
            acc = np.zeros(len(next(iter(Z.values()))), np.float32)
            for m in BLENDS[n]:
                acc += Z[m]
            out[n] = acc / len(BLENDS[n])
        else:
            out[n] = Z[n]
    return out


def describe(name):
    if name in BLENDS:
        return f"{name}: " + "+".join(BLENDS[name])
    return name
