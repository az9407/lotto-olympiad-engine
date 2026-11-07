#!/usr/bin/env python3
"""공식 후보군을 지난 회차로 채점합니다. 각 회차는 그 이전 데이터만 씁니다.

python lotto_bot/backtest.py --data history.json --rounds 100 --mode formulas --out f.json
python lotto_bot/backtest.py --data history.json --rounds 40 --mode rules --configs A,S2 --out r.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formulas as FO      # noqa: E402
import lotto_engine as LE  # noqa: E402
import run_lotto as R      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--mode", choices=["formulas", "rules"], default="formulas")
    ap.add_argument("--configs", default="A")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = R.lotto_df(R.load_history(args.data))
    N = int(df["round"].max())
    rounds = list(range(N - args.rounds + 1, N + 1))
    out = {}
    st = R.load_state()
    params = {c["id"]: c["params"] for c in st.get("tuner", {}).get("configs", [])}
    params.setdefault("A", dict(LE.BASE_PARAMS))
    t0 = time.time()
    for r in rounds:
        row = df[df["round"] == r].iloc[0]
        act = LE.comb_index([int(row[f"n{i}"]) for i in range(1, 7)])
        sub = df[df["round"] < r]
        if args.mode == "formulas":
            for name, S in FO.score_all(sub).items():
                out.setdefault(name, {})[str(r)] = round(LE.rank_pct(S, act), 4)
        else:
            for cid in args.configs.split(","):
                m = LE.build_model(sub, params[cid])
                out.setdefault(f"규칙엔진_{cid}", {})[str(r)] = round(LE.rank_pct(m["FINAL_F"], act), 4)
                del m
        print(f"{r}회 {time.time()-t0:.0f}초", flush=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
