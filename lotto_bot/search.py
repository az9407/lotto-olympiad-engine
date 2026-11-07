#!/usr/bin/env python3
"""로또 공식 자동 탐색기. 매일 깃허브에서 돌며 새 공식을 찾아 후보 풀에 넣습니다.

부품, 모두 t회 점수는 t-1회까지로만 계산하는 번호별 지표
- 다중 시간프레임: 단기·중기·장기 출현률의 정렬, 골든크로스·데드크로스
- 차트: 빈도, 지수이평, 이평차, 볼린저, 스토캐스틱, 윌리엄스%R, CCI, 돈치안 돌파, 일목균형 전환선·기준선, 추세기울기
- 카오스 유사패턴: 최근 상태와 가장 비슷했던 과거 시점들 바로 다음 회차의 번호
- 기타: 미출현, 주기 대비 미출현, 전이, 이월, 이웃, 직전 보너스
공식 = 부품 1~4개의 표준점수 가중합, 선택적으로 매크로 조건에 따라 방향 전환
탐색 = 유전 알고리즘: 무작위 생성 → 상위 보존 → 교배·변이, 시간 제한 안에서 반복
채점 = 300회~마지막 회차 중 3회마다 1회, 실제 1등 번호의 상위 %를 동적계획법으로 정확히 계산

python lotto_bot/search.py --minutes 20
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as EX  # noqa: E402

BASE = Path(__file__).resolve().parent
POOL = BASE / "state" / "search_pool.json"
POOL_CAP = 40
ADD_PER_RUN = 6
P6 = 6 / 45

WINDOWS = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
HALVES = [2, 3, 5, 8, 12, 20, 26, 40, 52, 80]
KS = [5, 10, 20, 30, 50, 80]
MACROS = ["직전1등당첨금", "직전1등당첨자수", "직전판매규모", "직전이월", "월"]


class Prims:
    """df 하나에 대한 부품 계산기, 결과는 캐시"""

    def __init__(self, df):
        X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.intp), 1)
        self.X, self.B = X, df["bonus"].to_numpy(np.intp)
        N = self.N = len(X)
        A = np.zeros((N, 46))
        A[np.arange(N)[:, None], X] = 1
        A[:, 0] = 0
        self.A = A
        self.csum = np.vstack([np.zeros(46), np.cumsum(A, 0)])
        self.Dp = np.vstack([np.zeros(46), np.cumsum(A - P6, 0)])[:N]
        self.mac = EX.macro(df)
        self.cache = {}

    def get(self, key):
        if key not in self.cache:
            self.cache[key] = self._make(key)
        return self.cache[key]

    def _rate(self, W):
        t = np.arange(self.N)
        lo = np.maximum(t - W, 0)
        return (self.csum[t] - self.csum[lo]) / np.maximum(t - lo, 1)[:, None]

    def _ema(self, h):
        a = 1 - 0.5 ** (1 / h)
        e = np.zeros((self.N, 46))
        cur = np.full(46, P6)
        for t in range(self.N):
            e[t] = cur
            cur = (1 - a) * cur + a * self.A[t]
        return e

    def _roll(self, K, fn):
        D = self.Dp
        pad = np.vstack([np.repeat(D[:1], K - 1, 0), D])
        return fn(swv(pad, K, axis=0))

    def _make(self, key):
        kind, *a = key
        N = self.N
        if kind == "빈도":
            return self._rate(a[0])
        if kind == "지수이평":
            return self._ema(a[0])
        if kind == "이평차":
            return self.get(("지수이평", a[0])) - self.get(("지수이평", a[1]))
        if kind == "볼린저":
            W = a[0]
            t = np.arange(N)
            n = np.maximum(np.minimum(t, W), 1)[:, None]
            lr = (self.csum[t] + 6) / (t[:, None] + 45)
            return (self._rate(W) * n - n * lr) / np.sqrt(n * lr * (1 - lr))
        if kind == "스토캐스틱":
            lo, hi = self._roll(a[0], lambda w: w.min(-1)), self._roll(a[0], lambda w: w.max(-1))
            return (self.Dp - lo) / np.maximum(hi - lo, 1e-9)
        if kind == "윌리엄스":
            return -self.get(("스토캐스틱", a[0]))
        if kind == "CCI":
            m = self._roll(a[0], lambda w: w.mean(-1))
            mad = self._roll(a[0], lambda w: np.abs(w - w.mean(-1, keepdims=True)).mean(-1))
            return (self.Dp - m) / (0.015 * np.maximum(mad, 1e-9))
        if kind == "돈치안돌파":
            K = a[0]
            hi = self._roll(K, lambda w: w[..., :-1].max(-1) if w.shape[-1] > 1 else w[..., 0])
            return (self.Dp >= hi).astype(float)
        if kind == "일목전환":
            K = a[0]
            mid = (self._roll(K, lambda w: w.max(-1)) + self._roll(K, lambda w: w.min(-1))) / 2
            return self.Dp - mid
        if kind == "추세기울기":
            K = a[0]
            x = np.arange(K) - (K - 1) / 2
            return self._roll(K, lambda w: w @ x / (x @ x))
        if kind == "다중프레임정렬":
            s, m, l = a
            rs, rm, rl = self._rate(s), self._rate(m), self._rate(l)
            return np.sign(rs - rm) + np.sign(rm - rl)
        if kind == "골든크로스":
            s, l = a
            d = self.get(("이평차", s, l))
            prev = np.vstack([d[:1], d[:-1]])
            return (np.sign(d) - np.sign(prev)) / 2
        if kind == "미출현":
            ab = np.zeros((N, 46))
            last = np.full(46, -1)
            for t in range(N):
                ab[t] = t - last
                last[self.X[t]] = t
            return np.log(ab + 1)
        if kind == "주기대비미출현":
            lr = (self.csum[np.arange(N)] + 6) / (np.arange(N)[:, None] + 45)
            return self.get(("미출현",)) + np.log(lr)
        if kind == "전이":
            co = np.zeros((46, 46))
            occ = np.zeros(46)
            mk = np.zeros((N, 46))
            for t in range(1, N):
                mk[t] = ((co + 10 * P6) / (occ[:, None] + 10))[self.X[t - 1]].mean(0)
                co[np.ix_(self.X[t - 1], self.X[t])] += 1
                occ[self.X[t - 1]] += 1
            return mk
        if kind == "이월":
            c = np.zeros((N, 46))
            c[1:] = self.A[:-1]
            return c
        if kind == "이웃":
            c = self.get(("이월",))
            nb = np.zeros((N, 46))
            nb[1:, 1:-1] = np.maximum(self.A[:-1, :-2], self.A[:-1, 2:])
            return nb * (1 - c)
        if kind == "직전보너스":
            b = np.zeros((N, 46))
            b[np.arange(1, N), self.B[:-1]] = 1
            return b
        if kind == "유사패턴":
            k, w1, w2 = a
            Z = np.hstack([self._rate(w1)[:, 1:], self._rate(w2)[:, 1:]])
            sq = (Z ** 2).sum(1)
            D2 = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
            out = np.zeros((N, 46))
            for t in range(120, N):
                cand = np.arange(100, t)
                nn = cand[np.argpartition(D2[t, cand], min(k, len(cand) - 1))[:k]]
                out[t] = self.A[nn].mean(0)
            return out
        raise KeyError(key)


def random_prim(rng):
    kinds = ["빈도", "지수이평", "이평차", "볼린저", "스토캐스틱", "윌리엄스", "CCI", "돈치안돌파", "일목전환",
             "추세기울기", "다중프레임정렬", "골든크로스", "미출현", "주기대비미출현", "전이", "이월", "이웃",
             "직전보너스", "유사패턴"]
    k = kinds[rng.integers(len(kinds))]
    if k in ("빈도", "볼린저"):
        return (k, int(rng.choice(WINDOWS)))
    if k == "지수이평":
        return (k, int(rng.choice(HALVES)))
    if k in ("이평차", "골든크로스"):
        a, b = sorted(rng.choice(HALVES, 2, replace=False))
        return (k, int(a), int(b))
    if k in ("스토캐스틱", "윌리엄스", "CCI", "돈치안돌파", "일목전환", "추세기울기"):
        return (k, int(rng.choice(KS)))
    if k == "다중프레임정렬":
        s, m, l = sorted(rng.choice(WINDOWS, 3, replace=False))
        return (k, int(s), int(m), int(l))
    if k == "유사패턴":
        return (k, int(rng.choice([10, 20, 40, 80])), int(rng.choice([5, 10, 20])), int(rng.choice([30, 50, 100])))
    return (k,)


def random_def(rng):
    terms = [[list(random_prim(rng)), float(rng.choice([-1, -0.5, 0.5, 1]))] for _ in range(rng.integers(1, 5))]
    sw = str(rng.choice(MACROS)) if rng.random() < 0.25 else None
    return dict(terms=terms, switch=sw)


def def_name(d):
    h = hashlib.sha1(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:6]
    main = max(d["terms"], key=lambda t: abs(t[1]))[0]
    label = main[0] + "".join(str(x) for x in main[1:])
    return f"탐색:{label}_{h}"


def describe(d):
    parts = []
    for p, w in d["terms"]:
        parts.append(f"{'+' if w > 0 else '-'}{abs(w):g}×{p[0]}{'/'.join(str(x) for x in p[1:])}")
    s = " ".join(parts)
    return s + (f" · {d['switch']} 높으면 그대로, 낮으면 반대" if d.get("switch") else "")


def _z(M):
    v = M[..., 1:]
    sd = v.std(-1, keepdims=True)
    out = np.zeros_like(M)
    out[..., 1:] = (v - v.mean(-1, keepdims=True)) / np.where(sd > 0, sd, 1)
    return out


def score_rows(d, pr, ts):
    S = np.zeros((len(ts), 46))
    for p, w in d["terms"]:
        S += w * _z(pr.get(tuple(p))[ts])
    if d.get("switch"):
        m = pr.mac[d["switch"]]
        if d["switch"] == "월":
            cond = m[ts] % 2 == 0
        else:
            cond = np.array([np.nan_to_num(m[t], nan=np.nanmedian(m[:t]) if t > 1 else 0) >= np.nanmedian(m[:t])
                             if t > 1 else True for t in ts])
        S = np.where(cond[:, None], S, -S)
    return S


def fitness(d, pr, ts):
    S = score_rows(d, pr, ts)
    v = np.array([EX.rank_pct(S[i], pr.X[t]) for i, t in enumerate(ts)])
    return float(v.mean()), v


def mutate(d, rng):
    d = json.loads(json.dumps(d))
    r = rng.random()
    if r < 0.35 and len(d["terms"]) < 4:
        d["terms"].append([list(random_prim(rng)), float(rng.choice([-1, -0.5, 0.5, 1]))])
    elif r < 0.55 and len(d["terms"]) > 1:
        d["terms"].pop(rng.integers(len(d["terms"])))
    elif r < 0.8:
        i = rng.integers(len(d["terms"]))
        d["terms"][i][0] = list(random_prim(rng))
    elif r < 0.9:
        i = rng.integers(len(d["terms"]))
        d["terms"][i][1] = -d["terms"][i][1]
    else:
        d["switch"] = None if d.get("switch") else str(rng.choice(MACROS))
    return d


def crossover(a, b, rng):
    terms = [t for t in a["terms"] + b["terms"] if rng.random() < 0.5][:4] or [a["terms"][0]]
    return dict(terms=json.loads(json.dumps(terms)), switch=a.get("switch") if rng.random() < 0.5 else b.get("switch"))


def load_pool():
    if POOL.exists():
        return json.loads(POOL.read_text(encoding="utf-8"))
    return {"defs": [], "tested": 0, "runs": []}


def run(df, minutes, seed, forward_hist=None, log=print):
    pr = Prims(df)
    N = pr.N
    ts = np.arange(299, N, 3)
    recent = np.arange(N - 100, N)
    rng = np.random.default_rng(seed)
    pool = load_pool()
    seen = {def_name(d["def"]) for d in pool["defs"]}
    t_end = time.time() + minutes * 60
    pop = [d["def"] for d in pool["defs"][:10]] + [random_def(rng) for _ in range(40)]
    scored = {}
    gen = 0
    while time.time() < t_end:
        for d in pop:
            n = def_name(d)
            if n not in scored:
                try:
                    scored[n] = (fitness(d, pr, ts)[0], d)
                except Exception as e:  # noqa: BLE001
                    log(f"실패 {describe(d)}: {e}")
                    scored[n] = (100.0, d)
            if time.time() >= t_end:
                break
        ranked = sorted(scored.values(), key=lambda x: x[0])
        elite = [d for _, d in ranked[:15]]
        children = []
        while len(children) < 35:
            if rng.random() < 0.3:
                children.append(random_def(rng))
            elif rng.random() < 0.5:
                children.append(crossover(elite[rng.integers(len(elite))], elite[rng.integers(len(elite))], rng))
            else:
                children.append(mutate(elite[rng.integers(len(elite))], rng))
        pop = elite + children
        gen += 1
        log(f"세대 {gen}: 시험 {len(scored)}개, 1위 {ranked[0][0]:.2f}% {describe(ranked[0][1])}")

    ranked = sorted(scored.values(), key=lambda x: x[0])
    added = []
    for fit, d in ranked:
        n = def_name(d)
        if n in seen:
            continue
        rec = float(fitness(d, pr, recent)[0])
        added.append({"name": n, "def": d, "desc": describe(d), "fit": round(fit, 3),
                      "recent100": round(rec, 3), "round": int(N)})
        seen.add(n)
        if len(added) >= ADD_PER_RUN:
            break
    fh = forward_hist or {}

    def keep_key(e):
        h = fh.get(e["name"], {})
        if len(h) >= 20:
            return (0, float(np.mean(list(h.values()))))
        return (1, e["fit"])

    pool["defs"] = sorted(pool["defs"] + added, key=keep_key)[:POOL_CAP]
    pool["tested"] = pool.get("tested", 0) + len(scored)
    pool["runs"] = (pool.get("runs", []) + [dict(round=int(N), date=time.strftime("%Y-%m-%d"), tested=len(scored),
                                                 generations=gen, added=[a["name"] for a in added],
                                                 best=ranked[0][0])])[-60:]
    POOL.parent.mkdir(exist_ok=True)
    POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    return pool, added, len(scored)


def pool_scores_next(df):
    """후보 풀 공식들의 다음 회차 번호별 점수"""
    import charts as CH
    pool = load_pool()
    if not pool["defs"]:
        return {}
    df2 = CH._with_next_row(df)
    pr = Prims(df2)
    t = np.array([len(df)])
    return {e["name"]: score_rows(e["def"], pr, t)[0] for e in pool["defs"]}


def pool_names():
    return [e["name"] for e in load_pool()["defs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    import run_lotto as R
    df = R.lotto_df(R.load_history(args.data))
    st = R.load_state()
    fh = st.get("tuner", {}).get("formulas", {})
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    pool, added, n = run(df, args.minutes, seed, fh)
    print(f"이번 탐색 {n}개 시험, 새 후보 {len(added)}개, 풀 {len(pool['defs'])}개, 누적 {pool['tested']}개")
    for a in added:
        print(f"  {a['name']} 전체 {a['fit']:.2f}% 최근100회 {a['recent100']:.2f}% {a['desc']}")


if __name__ == "__main__":
    main()
