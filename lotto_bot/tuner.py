"""매주 스스로 설정을 고치는 자동 학습.

1. 새 회차가 나오면 설정 6개가 각각 그 회차 이전 데이터만으로 엔진을 돌려
   실제 당첨번호를 전체 중 상위 몇 %에 올렸는지 채점합니다. 작을수록 좋습니다.
2. 최근 12회 가중 평균으로 성적을 매겨 꼴찌 설정을 빼고,
   1등 설정을 조금 바꾼 새 설정을 만들어 최근 6회로 채점한 뒤 넣습니다.
3. 다음 회차에는 성적 1등 설정을 씁니다.
"""
import copy

import numpy as np

K_BACKFILL = 6
WINDOW = 12
HALF_LIFE = 6.0
MIN_EVAL = 4
MAX_CONFIGS = 6
LOG_KEEP = 40

INITIAL = {
    "lotto": [
        ("A", {}),
        ("B", {"WEIGHTS": [1.0, 1.5, 2.0]}),
        ("C", {"WEIGHTS": [1.0, 3.0, 5.0]}),
        ("D", {"MIN_RATE": 0.60}),
        ("E", {"MAX_RULES": 25}),
        ("F", {"PROFILE": False}),
    ],
}
MIX_KEY = {"lotto": "W_POP"}


def full_params(game, base, p):
    return {**copy.deepcopy(base), **copy.deepcopy(p)}


def describe(game, p):
    w = "/".join(f"{x:g}" for x in p["WEIGHTS"])
    return (f"가중치 옛→최근 {w} · 문턱 {p['MIN_RATE']*100:.1f}% · 규칙 {int(p['MAX_RULES'])}개 · 수축 {p['SHRINK']:g}"
            f" · 인기회피 {p['W_POP']*100:.0f}% · 구성필터 {'켬' if p['PROFILE'] else '끔'}")


def score(cfg, N):
    num = den = 0.0
    n = 0
    for r, pct in cfg["hist"].items():
        age = N - int(r)
        if 0 <= age < WINDOW:
            w = 0.5 ** (age / HALF_LIFE)
            num += w * pct
            den += w
            n += 1
    return (num / den if den else None), n


def mutate(game, p, seed, existing):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        q = copy.deepcopy(p)
        keys = ["WEIGHTS", "MIN_RATE", "MAX_RULES", "SHRINK", MIX_KEY[game], "PROFILE"]
        for k in rng.choice(keys, size=2, replace=False):
            if k == "WEIGHTS":
                w = np.clip(np.array(q["WEIGHTS"]) * np.exp(rng.normal(0, 0.3, 3)), 0.5, 6.0)
                w = np.sort(w)
                q["WEIGHTS"] = [round(float(x), 2) for x in w]
            elif k == "MIN_RATE":
                q[k] = round(float(np.clip(q[k] + rng.normal(0, 0.02), 0.55, 0.65)), 3)
            elif k == "MAX_RULES":
                q[k] = int(np.clip(q[k] + rng.integers(-8, 9), 20, 60))
            elif k == "SHRINK":
                q[k] = round(float(np.clip(q[k] * np.exp(rng.normal(0, 0.4)), 5, 60)), 1)
            elif k == "PROFILE":
                q[k] = not q[k]
            else:
                q[k] = round(float(np.clip(q[k] + rng.normal(0, 0.08), 0.0, 0.5)), 2)
        if all(q != e for e in existing):
            return q
    return q


def step(game, tuner, N, base, evaluate, used_id):
    """evaluate(params, r, keep) -> (pct, extra)
    반환: tuner, extra(사용 설정의 N회 분석), 보고 dict"""
    if not tuner or not tuner.get("configs"):
        tuner = {"configs": [dict(id=i, params=full_params(game, base, p), hist={}, born=N)
                             for i, p in INITIAL[game]], "next": 1, "log": []}
    ids = [c["id"] for c in tuner["configs"]]
    if used_id not in ids:
        used_id = "A" if "A" in ids else ids[0]
    extra = None

    def run(cfg, r, keep):
        try:
            return evaluate(cfg["params"], r, keep)
        except Exception as e:  # noqa: BLE001
            print(f"설정 {cfg['id']} {r}회 평가 실패: {e}")
            return 100.0, None

    for cfg in tuner["configs"]:
        todo = [N] if cfg["hist"] else list(range(N - K_BACKFILL + 1, N + 1))
        for r in todo:
            if str(r) in cfg["hist"] and not (cfg["id"] == used_id and r == N and extra is None):
                continue
            keep = cfg["id"] == used_id and r == N
            pct, ex = run(cfg, r, keep)
            cfg["hist"][str(r)] = round(pct, 4)
            if keep:
                extra = ex
    this_round = {c["id"]: c["hist"].get(str(N)) for c in tuner["configs"]}

    scored = []
    for c in tuner["configs"]:
        s, n = score(c, N)
        if s is not None and n >= MIN_EVAL:
            scored.append((s, c["id"]))
    scored.sort()
    added = removed = None
    evolved_already = any(c.get("born") == N and c["id"].startswith("S") for c in tuner["configs"])
    if len(scored) >= 3 and not evolved_already:
        best_id = scored[0][1]
        if len(tuner["configs"]) >= MAX_CONFIGS:
            for s, cid in reversed(scored):
                if cid not in ("A", best_id, tuner.get("champion")):
                    removed = cid
                    break
            if removed:
                tuner["configs"] = [c for c in tuner["configs"] if c["id"] != removed]
        best = next(c for c in tuner["configs"] if c["id"] == best_id)
        new_p = mutate(game, best["params"], seed=N * 10,
                       existing=[c["params"] for c in tuner["configs"]])
        new = dict(id=f"S{tuner['next']}", params=new_p, hist={}, born=N, parent=best_id)
        tuner["next"] += 1
        for r in range(N - K_BACKFILL + 1, N + 1):
            pct, _ = run(new, r, False)
            new["hist"][str(r)] = round(pct, 4)
        tuner["configs"].append(new)
        added = new

    ranking = []
    for c in tuner["configs"]:
        s, n = score(c, N)
        if s is not None and n >= MIN_EVAL:
            ranking.append((s, c["id"]))
    ranking.sort()
    chosen = ranking[0][1] if ranking else "A"
    chosen_cfg = next(c for c in tuner["configs"] if c["id"] == chosen)
    for c in tuner["configs"]:
        c["hist"] = {r: v for r, v in c["hist"].items() if N - int(r) < KEEP_ROUNDS}
    report = dict(round=N, used=used_id, this_round=this_round, ranking=[(cid, round(s, 2)) for s, cid in ranking],
                  chosen=chosen, chosen_score=ranking[0][0] if ranking else None,
                  chosen_desc=describe(game, chosen_cfg["params"]),
                  added=dict(id=added["id"], parent=added["parent"], params=added["params"],
                             score=score(added, N)[0]) if added else None,
                  removed=removed)
    tuner["log"] = (tuner.get("log", []) + [dict(round=N, used=used_id, chosen=chosen, this_round=this_round,
                                                   added=added["id"] if added else None, removed=removed)])[-LOG_KEEP:]
    return tuner, extra, report, chosen_cfg["params"]


# ---------------- 공식 후보군 + 챔피언 교체 ----------------
WINDOW_CHAMP = 60      # 최근 60회로 비교
KEEP_ROUNDS = 80
MIN_COMMON = 10        # 같은 회차가 10개 이상 겹쳐야 비교
MIN_GAIN = 2.0         # 평균 순위가 2%p 이상 앞서고
Z = 2.0                # 그 차이가 표준오차의 2배를 넘을 때만 교체
FORMULA_BACKFILL = 12


def formula_step(tuner, N, names, eval_round, no_backfill=()):
    """eval_round(r) -> {공식이름: pct}. 새 공식은 최근 12회를 채점해서 넣음, no_backfill은 앞으로만"""
    F = tuner.setdefault("formulas", {})
    for n in names:
        F.setdefault(n, {})
    todo = {N}
    for n in names:
        if len(F[n]) == 0 and n not in no_backfill:
            todo.update(range(N - FORMULA_BACKFILL + 1, N + 1))
    for r in sorted(todo):
        if all(str(r) in F[n] for n in names if not (n in no_backfill and r < N)):
            continue
        res = eval_round(r)
        for n in names:
            if n in res:
                F[n][str(r)] = round(res[n], 4)
    for n in list(F):
        if n not in names:
            del F[n]
            continue
        F[n] = {r: v for r, v in F[n].items() if N - int(r) < KEEP_ROUNDS}
    return {n: F[n].get(str(N)) for n in names}


def bonferroni_z(m, alpha=0.05, n=None):
    """도전자 m개를 동시에 겨룰 때 우연 교체를 막는 기준, 연금복권 쪽과 같은 방식.
    겹치는 회차 n이 적으면 정규분포가 너무 관대해서 t분포를 씀"""
    p = 1 - alpha / max(m, 1)
    if n is not None and n > 1:
        from scipy.stats import t as tdist
        return max(Z, float(tdist.ppf(p, n - 1)))
    from statistics import NormalDist
    return max(Z, NormalDist().inv_cdf(p))


def all_methods(tuner):
    d = {c["id"]: c["hist"] for c in tuner["configs"]}
    for n, h in tuner.get("formulas", {}).items():
        d[n if ":" in n else f"공식:{n}"] = h
    return d


def recent_mean(h, N):
    v = [x for r, x in h.items() if 0 <= N - int(r) < WINDOW_CHAMP]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def decide(tuner, N, default, start_after=None):
    """start_after: 이름 → 회차. 그 회차 이후 기록만 비교에 씀"""
    M = all_methods(tuner)
    for mid, r0 in (start_after or {}).items():
        if mid in M:
            M[mid] = {r: v for r, v in M[mid].items() if int(r) > r0}
    champ = tuner.get("champion") or default
    if champ not in M:
        champ = default if default in M else "A"
    best = None
    zcrit = bonferroni_z(len(M) - 1)
    for mid, h in M.items():
        if mid == champ:
            continue
        rs = [r for r in h if r in M[champ] and 0 <= N - int(r) < WINDOW_CHAMP]
        if len(rs) < MIN_COMMON:
            continue
        d = np.array([h[r] - M[champ][r] for r in rs])
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        if m <= -MIN_GAIN and (se == 0 or m / se <= -bonferroni_z(len(M) - 1, n=len(d))):
            if best is None or m < best[1]:
                best = (mid, m, se, len(d))
    old = champ
    if best:
        champ = best[0]
    tuner["champion"] = champ
    board = sorted(((recent_mean(h, N), mid) for mid, h in M.items()),
                   key=lambda t: (t[0][0] is None, t[0][0] if t[0][0] is not None else 0))
    top = [(mid, m, n) for (m, n), mid in board if m is not None and n >= MIN_COMMON][:5]
    cm, cn = recent_mean(M[champ], N)
    return dict(old=old, champion=champ, switched=bool(best), gain=best[1] if best else None,
                champ_mean=cm, champ_n=cn, top=top, n_methods=len(M))


def champion_lines(dec, this_round):
    L = ["", "공식 경쟁"]
    if this_round:
        tr = sorted(((v, n) for n, v in this_round.items() if v is not None))[:3]
        L.append("- 이번 회차 실제 번호를 가장 위에 올린 공식: " + " · ".join(f"{n} {v:.1f}%" for v, n in tr))
    L.append(f"- 최근 60회 평균 상위 {len(dec['top'])}개, 후보 {dec['n_methods']}개 중: "
             + " · ".join(f"{mid} {m:.1f}%" for mid, m, n in dec["top"]))
    if dec["switched"]:
        L.append(f"- 챔피언 교체 {dec['old']} → {dec['champion']}, 같은 회차 비교 평균 {abs(dec['gain']):.1f}%p 앞섬")
    else:
        L.append(f"- 챔피언 유지 {dec['champion']}, 최근 {dec['champ_n']}회 평균 상위 {dec['champ_mean']:.1f}%"
                 if dec["champ_mean"] is not None else f"- 챔피언 유지 {dec['champion']}")
    L.append(f"- 교체 기준: 같은 회차끼리 비교해 {MIN_GAIN:g}%p 이상 앞서고 표준오차의 {bonferroni_z(dec['n_methods'] - 1):.1f}배를 넘을 때만,"
             f" 후보가 많을수록 우연 교체를 막으려 기준을 높임")
    return L


def report_lines(game, rep):
    L = ["", "스스로 학습, 규칙 엔진 설정"]
    tr = " · ".join(f"{cid} {v:.1f}%" for cid, v in rep["this_round"].items() if v is not None)
    L.append(f"- 이번 회차 설정별 성적, 실제 번호가 상위 몇 %였나: {tr}")
    if rep["removed"]:
        L.append(f"- 최근 성적 꼴찌 {rep['removed']} 제외")
    if rep["added"]:
        a = rep["added"]
        L.append(f"- {a['parent']}를 바꾼 새 설정 {a['id']} 추가, 최근 6회 평균 상위 {a['score']:.1f}%")
    if rep["chosen_score"] is not None:
        L.append(f"- 규칙 엔진 설정 중 최근 12회 1등 {rep['chosen']}, 가중 평균 상위 {rep['chosen_score']:.1f}%")
    L.append(f"- {rep['chosen']} 설정: {rep['chosen_desc']}")
    return L
