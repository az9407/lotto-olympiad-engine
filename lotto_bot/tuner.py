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
                if cid not in ("A", best_id):
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
        c["hist"] = {r: v for r, v in c["hist"].items() if N - int(r) < WINDOW * 3}
    report = dict(round=N, used=used_id, this_round=this_round, ranking=[(cid, round(s, 2)) for s, cid in ranking],
                  chosen=chosen, chosen_score=ranking[0][0] if ranking else None,
                  chosen_desc=describe(game, chosen_cfg["params"]),
                  added=dict(id=added["id"], parent=added["parent"], params=added["params"],
                             score=score(added, N)[0]) if added else None,
                  removed=removed)
    tuner["log"] = (tuner.get("log", []) + [dict(round=N, used=used_id, chosen=chosen, this_round=this_round,
                                                   added=added["id"] if added else None, removed=removed)])[-LOG_KEEP:]
    return tuner, extra, report, chosen_cfg["params"]


def report_lines(game, rep):
    L = ["", "스스로 학습"]
    tr = " · ".join(f"{cid} {v:.1f}%" for cid, v in rep["this_round"].items() if v is not None)
    L.append(f"- 이번 회차 설정별 성적, 실제 번호가 상위 몇 %였나: {tr}")
    if rep["removed"]:
        L.append(f"- 최근 성적 꼴찌 {rep['removed']} 제외")
    if rep["added"]:
        a = rep["added"]
        L.append(f"- {a['parent']}를 바꾼 새 설정 {a['id']} 추가, 최근 6회 평균 상위 {a['score']:.1f}%")
    if rep["chosen_score"] is not None:
        L.append(f"- 다음 회차 사용 {rep['chosen']}, 최근 12회 가중 평균 상위 {rep['chosen_score']:.1f}%")
    L.append(f"- {rep['chosen']} 설정: {rep['chosen_desc']}")
    return L
