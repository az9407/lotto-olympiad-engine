#!/usr/bin/env python3
"""번호별 지표를 대량으로 만들어 걸러냅니다.

- 차트 기법: 이동평균 빈도, 지수이동평균, MACD, 히스토그램, 볼린저 편차, 스토캐스틱, 추세 기울기
- 매크로 지표: 1등 당첨금, 당첨자 수, 판매 규모, 이월, 월
- 머신러닝: 위 지표 전부를 넣은 번호별 출현 확률 모델
고르는 구간 300~1149회, 처음 보는 채점 구간 1150~마지막 회차.
번호별 점수의 합으로 조합을 채점하므로 8,145,060개 중 순위를 동적계획법으로 정확히 셉니다.

python lotto_bot/explore.py --data history.json --out explore.json
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOTAL = comb(45, 6)
P = 6 / 45


def rank_pct(s, actual, L=3000):
    """s: 번호 1~45 점수(길이 46, 0번 무시). 높을수록 위. 실제 조합이 상위 몇 %인지"""
    v = np.asarray(s[1:], float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return 50.0
    q = np.rint((v - lo) / (hi - lo) * L).astype(int)
    dp = np.zeros((7, 6 * L + 1))
    dp[0, 0] = 1
    for x in q:
        for k in range(6, 0, -1):
            dp[k, x:] += dp[k - 1, :dp.shape[1] - x]
    a = int(q[np.asarray(actual) - 1].sum())
    row = dp[6]
    return float((row[a + 1:].sum() + 0.5 * row[a]) / TOTAL * 100)


def indicators(df):
    """회차 t의 점수는 t-1회까지로만 계산. 반환: 이름 → (N, 46) 배열"""
    X, B = R_hist(df)
    N = len(X)
    A = np.zeros((N, 46))
    A[np.arange(N)[:, None], X] = 1
    A[:, 0] = 0
    csum = np.vstack([np.zeros(46), np.cumsum(A, 0)])   # csum[t] = 0..t-1 합
    t_idx = np.arange(N)
    out = {}

    def win(W):
        lo = np.maximum(t_idx - W, 0)
        c = csum[t_idx] - csum[lo]
        n = (t_idx - lo)[:, None]
        return c, np.maximum(n, 1)

    for W in (5, 10, 20, 30, 50, 100, 200):
        c, n = win(W)
        out[f"빈도{W}"] = c / n
    out["빈도전체"] = csum[t_idx] / np.maximum(t_idx, 1)[:, None]
    longrate = (csum[t_idx] + 6) / (t_idx[:, None] + 45)

    # 지수이동평균, t-1회까지
    def ema(half):
        a = 1 - 0.5 ** (1 / half)
        e = np.zeros((N, 46))
        cur = np.full(46, P)
        for t in range(N):
            e[t] = cur
            cur = (1 - a) * cur + a * A[t]
        return e

    E = {h: ema(h) for h in (3, 6, 12, 26, 52)}
    for h, e in E.items():
        out[f"지수이평{h}"] = e
    macd = E[12] - E[26]
    out["MACD"] = macd
    sig = np.zeros_like(macd)
    a9 = 1 - 0.5 ** (1 / 9)
    cur = np.zeros(46)
    for t in range(N):
        cur = (1 - a9) * cur + a9 * macd[t]
        sig[t] = cur
    out["MACD히스토"] = macd - sig
    out["단기장기차"] = E[6] - E[52]

    # 볼린저: 최근 W회 출현이 자기 장기 평균에서 몇 표준편차인지
    for W in (10, 20, 50):
        c, n = win(W)
        mu = n * longrate
        sd = np.sqrt(n * longrate * (1 - longrate))
        out[f"볼린저{W}"] = (c - mu) / sd

    # 누적 편차 선, 스토캐스틱, 추세 기울기
    D = np.cumsum(A - P, 0)
    Dp = np.vstack([np.zeros(46), D])[:N]          # t-1까지
    for K in (20, 50):
        lo = np.zeros_like(Dp)
        hi = np.zeros_like(Dp)
        for t in range(N):
            s = Dp[max(0, t - K + 1):t + 1]
            lo[t], hi[t] = s.min(0), s.max(0)
        out[f"스토캐스틱{K}"] = (Dp - lo) / np.maximum(hi - lo, 1e-9)
    for K in (10, 30):
        x = np.arange(K) - (K - 1) / 2
        sl = np.zeros_like(Dp)
        for t in range(K, N):
            sl[t] = x @ Dp[t - K + 1:t + 1] / (x @ x)
        out[f"추세기울기{K}"] = sl

    # 미출현
    ab = np.zeros((N, 46))
    last = np.full(46, -1)
    for t in range(N):
        ab[t] = t - last
        last[X[t]] = t
    out["미출현"] = np.log(ab + 1)
    out["미출현대비주기"] = np.log((ab + 1) * longrate)

    # 전이·이월·이웃·보너스
    co = np.zeros((46, 46))
    occ = np.zeros(46)
    mk = np.zeros((N, 46))
    for t in range(1, N):
        Pm = (co + 10 * P) / (occ[:, None] + 10)
        mk[t] = Pm[X[t - 1]].mean(0)
        co[np.ix_(X[t - 1], X[t])] += 1
        occ[X[t - 1]] += 1
    out["전이"] = mk
    carry = np.zeros((N, 46))
    carry[1:] = A[:-1]
    nb = np.zeros((N, 46))
    nb[1:, 1:-1] = np.maximum(A[:-1, :-2], A[:-1, 2:])
    nb = nb * (1 - carry)
    bon = np.zeros((N, 46))
    bon[np.arange(1, N), B[:-1]] = 1
    out["이월"] = carry
    out["이웃"] = nb
    out["직전보너스"] = bon
    return out, A


def R_hist(df):
    X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.intp), 1)
    return X, df["bonus"].to_numpy(np.intp)


def macro(df):
    p1 = np.log1p(df["p1"].to_numpy(float))
    w1 = df["w1"].to_numpy(float)
    sales = np.log1p(df["w2"].to_numpy(float) * df["p2"].to_numpy(float))
    month = df["date"].dt.month.to_numpy()
    sh = lambda a: np.concatenate([[np.nan], a[:-1]])
    return {"직전1등당첨금": sh(p1), "직전1등당첨자수": sh(w1), "직전판매규모": sh(sales),
            "직전이월": sh((w1 == 0).astype(float)), "월": month.astype(float)}


def feature_rows(ind, mac, ts):
    names = list(ind)
    tens = np.arange(46) // 10
    tails = np.arange(46) % 10
    F = [np.stack([ind[n][ts] for n in names], -1)]
    F.append(np.broadcast_to(tens, (len(ts), 46))[..., None].astype(float))
    F.append(np.broadcast_to(tails, (len(ts), 46))[..., None].astype(float))
    for m in mac.values():
        F.append(np.repeat(np.nan_to_num(m[ts], nan=0)[:, None, None], 46, 1))
    F = np.concatenate(F, -1)[:, 1:, :]
    return F.reshape(-1, F.shape[-1])


def ml_models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    return {"머신러닝:로지스틱": lambda: LogisticRegression(C=0.05, max_iter=2000),
            "머신러닝:부스팅": lambda: HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.03, max_iter=150, l2_regularization=1.0)}


SWITCH_BASES = ("빈도30", "미출현", "전이", "MACD")


def evaluate(scores, X, rounds):
    return np.array([rank_pct(scores[t], X[t]) for t in rounds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=1150)
    ap.add_argument("--start", type=int, default=300)
    args = ap.parse_args()
    import run_lotto as R
    df = R.lotto_df(R.load_history(args.data))
    X, B = R_hist(df)
    N = len(X)
    rounds_in = np.arange(args.start - 1, args.split - 1)
    rounds_out = np.arange(args.split - 1, N)
    ind, A = indicators(df)
    mac = macro(df)
    res = {}

    def record(name, S):
        a = evaluate(S, X, rounds_in)
        b = evaluate(S, X, rounds_out)
        res[name] = dict(in_mean=float(a.mean()), in_top10=float((a <= 10).mean() * 100),
                         out_mean=float(b.mean()), out_se=float(b.std(ddof=1) / np.sqrt(len(b))),
                         out_top10=float((b <= 10).mean() * 100),
                         out_by_round={str(int(t) + 1): round(float(v), 4) for t, v in zip(rounds_out, b)})
        print(f"{name:24s} 고르는구간 {res[name]['in_mean']:5.1f}%  처음보는100회 {res[name]['out_mean']:5.1f}%"
              f" 상위10%안 {res[name]['out_top10']:4.0f}%", flush=True)

    for name, S in ind.items():
        record(f"차트:{name}", S)
        record(f"차트:역{name}", -S)

    # 매크로 조건부 전환: 조건이 참이면 추세, 거짓이면 역추세
    for mname, m in mac.items():
        if mname == "월":
            cond = (m % 2 == 0)
        else:
            med = np.nanmedian(m[:args.split - 1])
            cond = np.nan_to_num(m, nan=med) >= med
        for base in ("빈도30", "미출현", "전이", "MACD"):
            S = np.where(cond[:, None], ind[base], -ind[base])
            record(f"매크로전환:{mname}높으면{base}추세", S)

    # 머신러닝: 모든 지표 + 매크로로 번호별 출현 확률, 25회마다 다시 학습
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    names = list(ind)
    tens = np.arange(46) // 10
    tails = np.arange(46) % 10

    def feat_rows(ts):
        F = [np.stack([ind[n][ts] for n in names], -1)]
        F.append(np.broadcast_to(tens, (len(ts), 46))[..., None].astype(float))
        F.append(np.broadcast_to(tails, (len(ts), 46))[..., None].astype(float))
        for m in mac.values():
            F.append(np.repeat(np.nan_to_num(m[ts], nan=0)[:, None, None], 46, 1))
        F = np.concatenate(F, -1)[:, 1:, :]
        return F.reshape(-1, F.shape[-1])

    for label, make in (("머신러닝_로지스틱", lambda: LogisticRegression(C=0.05, max_iter=2000)),
                        ("머신러닝_부스팅", lambda: HistGradientBoostingClassifier(
                            max_depth=3, learning_rate=0.03, max_iter=150, l2_regularization=1.0))):
        S = np.zeros((N, 46))
        all_eval = np.concatenate([rounds_in, rounds_out])
        blocks = list(range(args.start - 1, N, 25))
        for b0 in blocks:
            tr = np.arange(100, max(b0, 101))
            if len(tr) < 150:
                continue
            ytr = A[tr][:, 1:].reshape(-1)
            Ftr = feat_rows(tr)
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
            mdl = make().fit((Ftr - mu) / sd, ytr)
            te = np.arange(b0, min(b0 + 25, N))
            pr = mdl.predict_proba((feat_rows(te) - mu) / sd)[:, 1].reshape(len(te), 45)
            S[te, 1:] = np.log(np.clip(pr, 1e-6, 1))
        keep = all_eval[all_eval >= blocks[0] + 25]
        a = evaluate(S, X, keep[keep < args.split - 1])
        b = evaluate(S, X, rounds_out)
        res[label] = dict(in_mean=float(a.mean()), in_top10=float((a <= 10).mean() * 100),
                          out_mean=float(b.mean()), out_se=float(b.std(ddof=1) / np.sqrt(len(b))),
                          out_top10=float((b <= 10).mean() * 100),
                          out_by_round={str(int(t) + 1): round(float(v), 4) for t, v in zip(rounds_out, b)})
        print(f"{label:24s} 고르는구간 {res[label]['in_mean']:5.1f}%  처음보는100회 {res[label]['out_mean']:5.1f}%"
              f" 상위10%안 {res[label]['out_top10']:4.0f}%", flush=True)

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
