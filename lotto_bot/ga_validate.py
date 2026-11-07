#!/usr/bin/env python3
"""3단계 검증 탐색.

1단계 탐색: 300~999회만 보고 유전 알고리즘으로 공식을 찾음
2단계 선별: 1단계 상위 후보를 1000~1149회로 다시 채점해 상위 몇 개만 남김
3단계 시험: 남은 공식을 한 번도 안 본 1150~마지막 회차로 최종 채점
목표 mean: 실제 1등 번호의 평균 상위 %를 낮춤, top10: 상위 10% 안에 든 회차 비율을 높임

python lotto_bot/ga_validate.py --minutes 60 --goal mean --seed 1 --out a.json
python lotto_bot/ga_validate.py --stage2 a.json b.json --out final.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search as SE  # noqa: E402

A_END, B_END = 1000, 1150   # 1단계 300~999, 2단계 1000~1149, 3단계 1150~


def load(path):
    import run_lotto as R
    return R.lotto_df(R.load_history(path))


def pcts(d, pr, ts):
    S = SE.score_rows(d, pr, ts)
    return np.array([SE.EX.rank_pct(S[i], pr.X[t]) for i, t in enumerate(ts)])


def parts(d, pr, ts):
    """회차별 평균 순위 %와 공정한 상위 10% 확률. 동점 묶음은 무작위로 나눈 것으로 계산"""
    S = SE.score_rows(d, pr, ts)
    pc, ft = [], []
    for i, t in enumerate(ts):
        gt, eq = SE.EX.rank_parts(S[i], pr.X[t])
        pc.append((gt + 0.5 * eq) * 100)
        ft.append(SE.EX.fair_top(gt, eq))
    return np.array(pc), np.array(ft)


def obj(v, goal):
    return float(v.mean()) if goal == "mean" else -float((v <= 10).mean() * 100)


def obj2(pc, ft, goal):
    return float(pc.mean()) if goal == "mean" else -float(ft.mean() * 100)


def stage1(df, minutes, goal, seed, out):
    pr = SE.Prims(df)
    ts = np.arange(299, A_END - 1)
    rng = np.random.default_rng(seed)
    t_end = time.time() + minutes * 60
    pop = [SE.random_def(rng) for _ in range(60)]
    scored = {}
    gen = 0
    while time.time() < t_end:
        for d in pop:
            n = SE.def_name(d)
            if n in scored:
                continue
            try:
                scored[n] = (obj(pcts(d, pr, ts), goal), d)
            except Exception as e:  # noqa: BLE001
                print("실패", SE.describe(d), e, flush=True)
                scored[n] = (1e9, d)
            if time.time() >= t_end:
                break
        ranked = sorted(scored.values(), key=lambda x: x[0])
        elite = [d for _, d in ranked[:20]]
        kids = []
        while len(kids) < 40:
            r = rng.random()
            if r < 0.25:
                kids.append(SE.random_def(rng))
            elif r < 0.5:
                kids.append(SE.crossover(elite[rng.integers(len(elite))], elite[rng.integers(len(elite))], rng))
            else:
                kids.append(SE.mutate(elite[rng.integers(len(elite))], rng))
        pop = elite + kids
        gen += 1
        if gen % 5 == 0:
            print(f"세대 {gen}: 시험 {len(scored)}개, 1위 {ranked[0][0]:.2f} {SE.describe(ranked[0][1])}", flush=True)
            Path(out).write_text(json.dumps(dict(goal=goal, tested=len(scored), generations=gen,
                                                 top=[{"fit": f, "def": d} for f, d in ranked[:200]]),
                                            ensure_ascii=False), encoding="utf-8")
    ranked = sorted(scored.values(), key=lambda x: x[0])
    Path(out).write_text(json.dumps(dict(goal=goal, tested=len(scored), generations=gen,
                                         top=[{"fit": f, "def": d} for f, d in ranked[:200]]),
                                    ensure_ascii=False), encoding="utf-8")
    print(f"1단계 끝: {len(scored)}개 시험, {gen}세대", flush=True)


def stage2(df, files, out, keep=10):
    pr = SE.Prims(df)
    N = pr.N
    tsA = np.arange(299, A_END - 1)
    tsB = np.arange(A_END - 1, B_END - 1)
    tsC = np.arange(B_END - 1, N)
    res = {}
    for f in files:
        D = json.loads(Path(f).read_text(encoding="utf-8"))
        goal = D["goal"]
        rows = []
        for e in D["top"]:
            pB, fB = parts(e["def"], pr, tsB)
            rows.append((obj2(pB, fB, goal), e, pB, fB))
        rows.sort(key=lambda x: x[0])
        chosen = []
        for oB, e, pB, fB in rows[:keep]:
            pA, fA = parts(e["def"], pr, tsA)
            pC, fC = parts(e["def"], pr, tsC)
            chosen.append(dict(name=SE.def_name(e["def"]), desc=SE.describe(e["def"]), def_=e["def"],
                               A_mean=float(pA.mean()), A_top10=float(fA.mean() * 100),
                               B_mean=float(pB.mean()), B_top10=float(fB.mean() * 100),
                               C_mean=float(pC.mean()), C_se=float(pC.std(ddof=1) / np.sqrt(len(pC))),
                               C_top10=float(fC.mean() * 100),
                               C_by_round={str(int(t) + 1): round(float(x), 4) for t, x in zip(tsC, pC)}))
        allB = np.array([r[0] for r in rows])
        res[f] = dict(goal=goal, tested=D["tested"], stage2_pool=len(rows), chosen=chosen,
                      B_spread=[float(allB.min()), float(np.median(allB)), float(allB.max())])
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for f, r in res.items():
        print(f"\n목표 {r['goal']}: 1단계 {r['tested']:,}개 시험 → 2단계 {r['stage2_pool']}개 → 3단계 {len(r['chosen'])}개")
        for c in r["chosen"]:
            print(f"  탐색 {c['A_mean']:5.1f}%/{c['A_top10']:4.1f}  선별 {c['B_mean']:5.1f}%/{c['B_top10']:4.1f}"
                  f"  최종 {c['C_mean']:5.1f}%±{c['C_se']:.1f}/{c['C_top10']:4.0f}  {c['desc']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--goal", choices=["mean", "top10"], default="mean")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage2", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load(args.data)
    if args.stage2:
        stage2(df, args.stage2, args.out)
    else:
        stage1(df, args.minutes, args.goal, args.seed, args.out)


if __name__ == "__main__":
    main()
