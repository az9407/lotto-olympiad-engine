"""로또 6/45 최근 국면 패턴 엔진.

build_model(draws)  : 주어진 회차까지로 다음 회차 점수를 8,145,060개 전체에 매김
pick(model, k)      : 겹침 3개 이하, 번호별 4회 이하로 상위 k개 선정
analyze(model, nums): 실제 당첨번호가 그 모델에서 몇 위였는지, 어떤 규칙이 맞았는지
"""
from itertools import chain, combinations
from math import gcd

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

CFG = dict(
    RECENT=250, SEGMENTS=[(150, 1.0), (50, 2.0), (50, 3.0)],
    MIN_N250=30, MIN_N100=12, MIN_N50=6, MIN_RATE=0.55,
    Q_LO=0.40, Q_HI=0.50, SHRINK=20.0, MAX_RULES=40, MAX_PER_FEAT=2,
    RHO_DUP=0.5, LIFT_CONFLICT=0.8, W_RECENT=0.8, W_POP=0.2,
    MAX_OVERLAP=3, MAX_USE=4, SAMPLE=1_000_000, SEED=645,
)

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
PR = np.array([n in PRIMES for n in range(46)])
SQ = np.array([n in {1, 4, 9, 16, 25, 36} for n in range(46)])
FIB = np.array([n in {1, 2, 3, 5, 8, 13, 21, 34} for n in range(46)])
GT = np.array([[gcd(a, b) > 1 for b in range(46)] for a in range(46)])
PAIRS = list(combinations(range(6), 2))

LABEL = {
    "sum": "합계", "span": "양끝차", "max_gap": "최대간격", "min_gap": "최소간격", "odd": "홀수개수",
    "low": "1~22 개수", "tail_sum": "끝수합", "prime": "소수개수", "composite": "합성수개수",
    "square": "제곱수개수", "fib": "피보나치개수", "mult3": "3배수개수", "mult5": "5배수개수",
    "same_tail_pairs": "동끝수쌍", "gcd_pairs": "공약수쌍", "ac": "AC값", "consec": "연번개수",
    "carry": "직전회 이월수", "neighbor": "직전회 이웃수", "bonus_in": "직전 보너스 포함", "bonus": "보너스",
}
for _d in range(5):
    LABEL[f"dec{_d}"] = f"{_d*10}번대 개수"
for _d in range(10):
    LABEL[f"tail_absent{_d}"] = f"끝수{_d} 미출현"
for _i in range(6):
    LABEL[f"x{_i+1}"] = f"x{_i+1}"
for _i in range(5):
    LABEL[f"gap{_i+1}"] = f"간격{_i+1}"


def lab(name):
    return LABEL.get(name, name)


def membership(rows):
    M = np.zeros((len(rows), 47), bool)
    for j in range(rows.shape[1]):
        M[np.arange(len(rows)), rows[:, j]] = True
    return M


def lookup(M, A):
    if M.ndim == 1:
        return M[A]
    return np.take_along_axis(M, A.astype(np.intp), 1)


def run_max(v):
    best = np.ones(len(v), np.int16)
    run = np.ones(len(v), np.int16)
    for i in range(1, v.shape[1]):
        run = np.where(v[:, i] == v[:, i - 1], run + 1, 1).astype(np.int16)
        best = np.maximum(best, run)
    return best


def base_feats(A):
    A = A.astype(np.int16)
    n = len(A)
    g = np.diff(A, axis=1)
    t = A % 10
    f = {f"x{i+1}": A[:, i] for i in range(6)}
    f["sum"] = A.sum(1, dtype=np.int16)
    f["span"] = A[:, 5] - A[:, 0]
    for i in range(5):
        f[f"gap{i+1}"] = g[:, i]
    f["max_gap"] = g.max(1)
    f["min_gap"] = g.min(1)
    f["odd"] = (A % 2).sum(1, dtype=np.int16)
    f["low"] = (A <= 22).sum(1, dtype=np.int16)
    f["tail_sum"] = t.sum(1, dtype=np.int16)
    f["prime"] = PR[A].sum(1, dtype=np.int16)
    f["composite"] = ((~PR[A]) & (A > 1)).sum(1, dtype=np.int16)
    f["square"] = SQ[A].sum(1, dtype=np.int16)
    f["fib"] = FIB[A].sum(1, dtype=np.int16)
    f["mult3"] = (A % 3 == 0).sum(1, dtype=np.int16)
    f["mult5"] = (A % 5 == 0).sum(1, dtype=np.int16)
    stp = np.zeros(n, np.int16)
    gp = np.zeros(n, np.int16)
    bits = np.zeros(n, np.uint64)
    one = np.uint64(1)
    for i, j in PAIRS:
        stp += t[:, i] == t[:, j]
        gp += GT[A[:, i], A[:, j]]
        bits |= np.left_shift(one, (A[:, j] - A[:, i]).astype(np.uint64))
    f["same_tail_pairs"] = stp
    f["gcd_pairs"] = gp
    f["ac"] = np.bitwise_count(bits).astype(np.int16) - 5
    f["consec"] = (g == 1).sum(1, dtype=np.int16)
    dec = A // 10
    for d in range(5):
        f[f"dec{d}"] = (dec == d).sum(1, dtype=np.int16)
    for d in range(10):
        f[f"tail_absent{d}"] = (~(t == d).any(1)).astype(np.int16)
    return f


def rel_feats(A, prev_main, prev_bonus):
    A = A.astype(np.int16)
    f = {"carry": lookup(prev_main, A).sum(1, dtype=np.int16)}
    if prev_main.ndim == 1:
        nb = np.zeros(47, bool)
        idx = np.where(prev_main)[0]
        nb[np.clip(idx - 1, 0, 46)] = True
        nb[np.clip(idx + 1, 0, 46)] = True
        nb &= ~prev_main
        nb[0] = False
    else:
        nb = np.zeros_like(prev_main)
        nb[:, :-1] |= prev_main[:, 1:]
        nb[:, 1:] |= prev_main[:, :-1]
        nb &= ~prev_main
        nb[:, 0] = False
    f["neighbor"] = lookup(nb, A).sum(1, dtype=np.int16)
    f["bonus_in"] = lookup(prev_bonus, A).sum(1, dtype=np.int16)
    return f


_SPACE = {}


def space():
    if not _SPACE:
        comb = np.fromiter(chain.from_iterable(combinations(range(1, 46), 6)), dtype=np.int8,
                           count=8145060 * 6).reshape(-1, 6)
        _SPACE["COMB"] = comb
        _SPACE["BASE"] = base_feats(comb)
        rng = np.random.default_rng(CFG["SEED"])
        _SPACE["SAMP"] = np.sort(rng.choice(len(comb), CFG["SAMPLE"], replace=False))
    return _SPACE


def make_events(FC, NC):
    events = []
    for name, v0 in FC.items():
        v = v0.astype(np.int32)
        lo, hi = int(v.min()), int(v.max())
        cnt = np.bincount(v - lo, minlength=hi - lo + 1).astype(float) / NC
        vals = np.arange(lo, hi + 1)
        cum = np.cumsum(cnt)
        flag = name.startswith("tail_absent")
        for i, c in enumerate(vals[:-1]):
            q = cum[i]
            if CFG["Q_LO"] <= q <= CFG["Q_HI"]:
                events.append(dict(feat=name, kind="le", c=int(c), q=q,
                                   text=f"끝수{name[-1]} 출현" if flag else f"{lab(name)} ≤ {c}"))
            q2 = 1 - cum[i]
            if CFG["Q_LO"] <= q2 <= CFG["Q_HI"]:
                events.append(dict(feat=name, kind="ge", c=int(vals[i + 1]), q=q2,
                                   text=f"끝수{name[-1]} 미출현" if flag else f"{lab(name)} ≥ {vals[i+1]}"))
        if hi - lo >= 4:
            seen = set()
            for m in range(2, 12):
                rc = np.bincount(vals % m, weights=cnt, minlength=m)
                for L in range(1, m):
                    for s in range(m):
                        S = tuple(sorted((s + j) % m for j in range(L)))
                        if (m, S) in seen:
                            continue
                        seen.add((m, S))
                        q = rc[list(S)].sum()
                        if CFG["Q_LO"] <= q <= CFG["Q_HI"]:
                            events.append(dict(feat=name, kind="mod", m=m, S=S, q=q,
                                               text=f"{lab(name)} mod {m} = {','.join(map(str, S))}"))
    return events


def ev_mask(e, F, idx=None):
    v = F[e["feat"]] if idx is None else F[e["feat"]][idx]
    if e["kind"] == "le":
        return v <= e["c"]
    if e["kind"] == "ge":
        return v >= e["c"]
    return np.isin(v % e["m"], e["S"])


def date_vals(rounds, dts):
    dts = pd.to_datetime(pd.Series(list(dts)))
    ds = dts.dt.strftime("%Y%m%d").apply(lambda s: sum(int(ch) for ch in s)).to_numpy()
    return {
        "회차": np.asarray(rounds), "연도": dts.dt.year.to_numpy(), "월": dts.dt.month.to_numpy(),
        "일": dts.dt.day.to_numpy(), "날짜숫자합": ds, "날짜 디지털루트": 1 + (ds - 1) % 9,
        "연중일": dts.dt.dayofyear.to_numpy(), "연중주차": dts.dt.isocalendar().week.to_numpy().astype(int),
    }


POP_NAMES = ["31이하 개수", "12이하 개수", "연번", "용지 같은 줄 최대", "용지 같은 열 최대",
             "5배수", "7배수", "합/10", "40번대", "직전회 이월", "홀수"]


def pop_feats(A, prevmask):
    A = A.astype(np.int16)
    rows = (A - 1) // 7
    cols = np.sort((A - 1) % 7, 1)
    g = np.diff(A, 1)
    return np.stack([
        (A <= 31).sum(1), (A <= 12).sum(1), (g == 1).sum(1),
        run_max(rows), run_max(cols), (A % 5 == 0).sum(1), (A % 7 == 0).sum(1),
        A.sum(1) / 10.0, (A >= 40).sum(1), lookup(prevmask, A).sum(1), (A % 2).sum(1),
    ], 1).astype(np.float32)


BASE_PARAMS = dict(WEIGHTS=[1.0, 2.0, 3.0], MIN_RATE=0.55, MAX_RULES=40, SHRINK=20.0, W_POP=0.2, PROFILE=True)


def apply_params(params):
    C = dict(CFG)
    p = {**BASE_PARAMS, **(params or {})}
    w = p["WEIGHTS"]
    C["SEGMENTS"] = [(150, float(w[0])), (50, float(w[1])), (50, float(w[2]))]
    C["MIN_RATE"] = float(p["MIN_RATE"])
    C["MAX_RULES"] = int(p["MAX_RULES"])
    C["SHRINK"] = float(p["SHRINK"])
    C["W_POP"] = float(p["W_POP"])
    C["W_RECENT"] = 1.0 - float(p["W_POP"])
    C["PROFILE"] = bool(p["PROFILE"])
    return C


def rank_pct(F, idx):
    """상위 몇 %인지, 동점은 가운데로"""
    f = F[idx]
    return float((np.count_nonzero(F > f) + 0.5 * np.count_nonzero(F == f)) / len(F) * 100)


def build_model(draws, params=None):
    """draws: round, date, n1..n6, bonus, w1, p1, w2, p2 열을 가진 회차 오름차순 DataFrame"""
    C = apply_params(params)
    sp = space()
    COMB, BASE, SAMP = sp["COMB"], sp["BASE"], sp["SAMP"]
    NC = len(COMB)
    df = draws.sort_values("round").reset_index(drop=True)
    N = len(df)
    T = int(df["round"].iloc[-1]) + 1
    X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int16), 1)
    B = df["bonus"].to_numpy(np.int16)
    DATES = pd.to_datetime(df["date"])
    TDATE = DATES.iloc[-1] + pd.Timedelta(days=7)
    W1, P1 = df["w1"].to_numpy(float), df["p1"].to_numpy(float)
    W2, P2 = df["w2"].to_numpy(float), df["p2"].to_numpy(float)
    REC = C["RECENT"]
    REC_ROWS = np.arange(N - REC, N)
    WT = np.concatenate([np.full(n, w) for n, w in C["SEGMENTS"]])
    L100 = np.arange(REC) >= REC - 100
    L50 = np.arange(REC) >= REC - 50

    prevM = np.zeros((N, 47), bool)
    prevM[1:] = membership(X[:-1])
    prevB = np.zeros((N, 47), bool)
    prevB[np.arange(1, N), B[:-1]] = True
    FH = base_feats(X)
    FH.update(rel_feats(X, prevM, prevB))
    FH["bonus"] = B

    absN = np.zeros((N + 1, 46), np.int16)
    absT = np.zeros((N + 1, 10), np.int16)
    lastN = np.zeros(46, int)
    lastT = np.zeros(10, int)
    for r in range(N + 1):
        absN[r] = (r + 1) - lastN
        absT[r] = (r + 1) - lastT
        if r < N:
            lastN[X[r]] = r + 1
            lastT[np.unique(X[r] % 10)] = r + 1

    lastM = np.zeros(47, bool)
    lastM[X[-1]] = True
    lastBv = np.zeros(47, bool)
    lastBv[B[-1]] = True
    FC = dict(BASE)
    FC.update(rel_feats(COMB, lastM, lastBv))

    events = make_events(FC, NC)
    EH = np.stack([ev_mask(e, FH, REC_ROWS) for e in events], 1).astype(np.float32)
    QV = np.array([e["q"] for e in events])

    conds = []

    def add(text, mask, src, info):
        mask = np.asarray(mask, bool)
        if mask.sum() >= C["MIN_N250"]:
            conds.append(dict(text=text, mask=mask, src=src, info=info))

    Q = [0.2, 0.33, 0.5, 0.67, 0.8]
    add("무조건", np.ones(REC, bool), "B", "")
    for k in (1, 2, 3):
        for name, arr in FH.items():
            h = arr[REC_ROWS - k].astype(np.int64)
            v = int(arr[N - k])
            info = f"{T-k}회 {lab(name)} {v}"
            if h.max() - h.min() >= 4:
                for m in range(2, 12):
                    add(f"{T-k}회 {lab(name)} mod {m} = {v % m}", h % m == v % m, f"lag{k}", info)
            for c in sorted(set(np.quantile(h, Q).round().astype(int))):
                if v >= c:
                    add(f"{T-k}회 {lab(name)} ≥ {c}", h >= c, f"lag{k}", info)
                if v <= c:
                    add(f"{T-k}회 {lab(name)} ≤ {c}", h <= c, f"lag{k}", info)
        for name, arr in (("1등 당첨자수", W1), ("2등 당첨자수", W2), ("1등 1인당 당첨금", P1)):
            h = arr[REC_ROWS - k]
            v = arr[N - k]
            info = f"{T-k}회 {name} {v:,.0f}"
            for c in sorted(set(np.quantile(h, Q))):
                if v >= c:
                    add(f"{T-k}회 {name} ≥ {c:,.0f}", h >= c, f"lag{k}", info)
                if v <= c:
                    add(f"{T-k}회 {name} ≤ {c:,.0f}", h <= c, f"lag{k}", info)
    DH = date_vals(REC_ROWS + 1, DATES.iloc[REC_ROWS])
    DT = date_vals([T], [TDATE])
    for name in DH:
        h = DH[name].astype(np.int64)
        v = int(DT[name][0])
        for m in range(2, 12):
            add(f"{T}회 {name} mod {m} = {v % m}", h % m == v % m, "date", f"{T}회 {name} {v}")
    for n_ in range(1, 46):
        h, v = absN[REC_ROWS, n_], absN[N, n_]
        for K in (5, 8, 10, 15, 20):
            if v >= K:
                add(f"{n_}번 {K}회 이상 미출현", h >= K, "absence", f"{n_}번 {v}회째 미출현")
        for K in (1, 2, 3):
            if v <= K:
                add(f"{n_}번 최근 {K}회 내 출현", h <= K, "absence", f"{n_}번 {v}회 전 출현")
    for d in range(10):
        h, v = absT[REC_ROWS, d], absT[N, d]
        for K in (2, 3, 4, 5):
            if v >= K:
                add(f"끝수{d} {K}회 이상 미출현", h >= K, "absence", f"끝수{d} {v}회째 미출현")
        if v <= 1:
            add(f"끝수{d} 직전회 출현", h <= 1, "absence", f"끝수{d} 직전회 출현")
    uniq, keep = set(), []
    for c in conds:
        key = np.packbits(c["mask"]).tobytes()
        if key not in uniq:
            uniq.add(key)
            keep.append(c)
    conds = keep
    CH = np.stack([c["mask"] for c in conds], 1).astype(np.float32)

    n250 = CH.sum(0)
    h250 = CH.T @ EH
    nw = (CH * WT[:, None]).sum(0)
    hw = (CH * WT[:, None]).T @ EH
    n100 = CH[L100].sum(0)
    h100 = CH[L100].T @ EH[L100]
    n50 = CH[L50].sum(0)
    h50 = CH[L50].T @ EH[L50]
    r250 = h250 / n250[:, None]
    rw = hw / nw[:, None]
    r100 = h100 / np.maximum(n100, 1)[:, None]
    r50 = h50 / np.maximum(n50, 1)[:, None]
    R = C["MIN_RATE"]
    ok = ((n250 >= C["MIN_N250"])[:, None] & (r250 >= R) & (rw >= R)
          & (n100 >= C["MIN_N100"])[:, None] & (r100 >= R)
          & (n50 >= C["MIN_N50"])[:, None] & (r50 >= R))
    ps = (hw + C["SHRINK"] * QV[None, :]) / (nw[:, None] + C["SHRINK"])
    logit = lambda p: np.log(p / (1 - p))
    WR = logit(np.clip(ps, 1e-6, 1 - 1e-6)) - logit(QV)[None, :]
    cand = np.argwhere(ok)
    cand = cand[np.argsort(-WR[ok], kind="stable")]

    cache = {}

    def smask(ei):
        if ei not in cache:
            cache[ei] = ev_mask(events[ei], FC, SAMP)
        return cache[ei]

    def emp_lift(ei, ej):
        a, b = EH[:, ei], EH[:, ej]
        W = WT.sum()
        pa, pb, pab = (WT * a).sum() / W, (WT * b).sum() / W, (WT * a * b).sum() / W
        return pab / (pa * pb) if pa * pb > 0 else 0.0

    selected, removed, used, fcount = [], [], set(), {}
    for ci, ei in cand:
        if len(selected) >= C["MAX_RULES"]:
            break
        if ei in used or fcount.get(events[ei]["feat"], 0) >= C["MAX_PER_FEAT"]:
            continue
        mi = smask(ei)
        qi = mi.mean()
        verdict = None
        for s in selected:
            mj = cache[s["ei"]]
            qj = mj.mean()
            J = (mi & mj).mean()
            rho = (J - qi * qj) / np.sqrt(qi * (1 - qi) * qj * (1 - qj))
            tl = J / (qi * qj)
            el = emp_lift(ei, s["ei"])
            if rho >= C["RHO_DUP"]:
                verdict = "중복"
            elif tl < C["LIFT_CONFLICT"] or el < C["LIFT_CONFLICT"]:
                verdict = "모순"
            if verdict:
                removed.append(dict(rule=f"{conds[ci]['text']} ⇒ {events[ei]['text']}", reason=verdict, vs=s["id"]))
                break
        used.add(ei)
        if verdict:
            continue
        fcount[events[ei]["feat"]] = fcount.get(events[ei]["feat"], 0) + 1
        selected.append(dict(
            id=f"R{len(selected)+1:02d}", ci=int(ci), ei=int(ei),
            rule=f"{conds[ci]['text']} ⇒ {events[ei]['text']}", cond=conds[ci]["text"],
            event=events[ei]["text"], basis=conds[ci]["info"], q=float(events[ei]["q"]),
            n250=int(n250[ci]), r250=float(r250[ci, ei]), rw=float(rw[ci, ei]),
            n50=int(n50[ci]), r50=float(r50[ci, ei]), w=float(WR[ci, ei])))
    if len(selected) < 15:
        raise RuntimeError("채택 규칙이 15개 미만입니다")

    S_rule = np.zeros(NC, np.float32)
    HIT = np.zeros(NC, np.int8)
    for s in selected:
        m = ev_mask(events[s["ei"]], FC)
        S_rule += np.float32(s["w"]) * m
        HIT += m
    z_recent = (S_rule - S_rule.mean()) / S_rule.std()
    del S_rule

    rows = np.arange(99, N)
    rows = rows[W2[rows] > 0]
    Fp = pop_feats(X[rows], prevM[rows])
    mu, sd = Fp.mean(0), Fp.std(0)
    expo = W2[rows] * P2[rows]
    expo = expo / expo.mean()
    pr = PoissonRegressor(alpha=1e-3, max_iter=1000).fit((Fp - mu) / sd, W1[rows] / expo, sample_weight=expo)
    beta = pr.coef_
    FPC = pop_feats(COMB, lastM)
    pop_lin = np.zeros(NC, np.float32)
    for j in range(len(beta)):
        pop_lin += np.float32(beta[j]) * ((FPC[:, j] - mu[j]) / sd[j])
    del FPC
    z_pop = -(pop_lin - pop_lin.mean()) / pop_lin.std()
    del pop_lin

    FINAL = C["W_RECENT"] * z_recent.astype(np.float64) + C["W_POP"] * z_pop.astype(np.float64)
    last50 = np.arange(N - 50, N)
    ok_prof = np.ones(NC, bool)
    for name in ("odd", "low", "carry", "ac"):
        vc = np.bincount(FH[name][last50].astype(int), minlength=20)
        ok_prof &= np.isin(FC[name], np.where(vc >= 3)[0])
    smin, smax = int(FH["sum"][last50].min()), int(FH["sum"][last50].max())
    ok_prof &= (FC["sum"] >= smin) & (FC["sum"] <= smax)
    if C["PROFILE"]:
        FINAL_F = FINAL.copy()
        FINAL_F[~ok_prof] = -np.inf
    else:
        FINAL_F = FINAL

    return dict(T=T, TDATE=TDATE, X=X, B=B, lastM=lastM, lastBv=lastBv, events=events,
                selected=selected, removed=removed, FINAL=FINAL, FINAL_F=FINAL_F, HIT=HIT,
                z_pop=z_pop, sum50=FH["sum"][last50].astype(float), odd50=FH["odd"][last50].astype(int),
                FC=FC)


def pick(model, k=10):
    C = CFG
    COMB = space()["COMB"]
    F = model["FINAL_F"]
    top = np.argpartition(-F, 400_000)[:400_000]
    top = top[np.lexsort((top, -F[top]))]
    picked, use = [], np.zeros(46, int)
    for idx in top:
        c = COMB[idx].astype(int)
        if (use[c] >= C["MAX_USE"]).any():
            continue
        if any(len(set(c) & set(COMB[p].astype(int))) > C["MAX_OVERLAP"] for p in picked):
            continue
        picked.append(int(idx))
        use[c] += 1
        if len(picked) == k:
            break
    out = []
    for r, idx in enumerate(picked, 1):
        c = COMB[idx].astype(int).tolist()
        out.append(dict(rank=r, numbers=c, score=round(float(model["FINAL"][idx]), 6),
                        hits=int(model["HIT"][idx]), rules=len(model["selected"])))
    return out


def comb_index(nums):
    COMB = space()["COMB"]
    a = np.array(sorted(nums), np.int8)
    return int(np.flatnonzero((COMB == a).all(1))[0])


def analyze(model, nums, bonus):
    """모델이 만든 순위에서 실제 당첨번호의 위치와 규칙 적중"""
    idx = comb_index(nums)
    F = model["FINAL_F"]
    rank = int((F > F[idx]).sum()) + 1
    A = np.array([sorted(nums)], np.int16)
    f = base_feats(A)
    f.update(rel_feats(A, model["lastM"], model["lastBv"]))
    res = []
    for s in model["selected"]:
        hit = bool(ev_mask(model["events"][s["ei"]], f)[0])
        res.append((s, hit))
    sum_all = model["FC"]["sum"]
    s = int(f["sum"][0])
    return dict(
        rank=rank, total=len(F), pct=rank / len(F) * 100,
        rules=res, n_hit=sum(h for _, h in res),
        sum=s, sum_pct_low=float((sum_all < s).mean() * 100), sum50=float(model["sum50"].mean()),
        odd=int(f["odd"][0]), low=int(f["low"][0]), ac=int(f["ac"][0]), consec=int(f["consec"][0]),
        carry=sorted(int(v) for v in A[0] if model["lastM"][v]),
        neighbor=int(f["neighbor"][0]), tail_sum=int(f["tail_sum"][0]),
        pop_top=float((model["z_pop"] > model["z_pop"][idx]).mean() * 100),
        in_profile=bool(np.isfinite(model["FINAL_F"][idx])),
    )
