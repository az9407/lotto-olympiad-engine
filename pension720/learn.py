"""연금복권720+ 주간 학습.

지금 쓰는 방법(챔피언)과, 값을 하나나 둘 바꾼 방법(도전자) 23개를 최근 60회로 겨룬다.
각 회차는 그 회차 이전 자료로만 학습해 무작위 20만 후보 안에서 실제 1등의 순위를 잰다.
도전자가 챔피언보다 회차별로 짝지어 비교했을 때 평균 순위가 2,000 이상 낮고
그 차이가 표준오차의 3배를 넘을 때만 교체한다. 우연히 좋아 보이는 방법으로 매주 흔들리지 않게 하려는 기준이다.
매주 새 회차가 들어오므로 같은 비교도 자료가 바뀌고, 챔피언이 바뀌면 그 주변을 다시 찾는다.
"""
from __future__ import annotations

import random
from dataclasses import replace
from math import sqrt

try:
    from . import predict_pension15 as eng
except ImportError:
    import predict_pension15 as eng

LEARN_FOLDS = 60
OPTIONS = {
    'time_weights': [(1.0, 1.0, 1.0), (2.0, 1.5, 1.0), (3.0, 2.0, 1.0), (4.0, 2.0, 1.0)],
    'threshold': [0.55, 0.58, 0.60, 0.65],
    'shrink': [5, 10, 20, 40],
    'limit': [20, 30, 40, 60],
    'var_cap': [1, 2, 3],
    'place_boost': [True, False],
    'mix': [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)],
}
RANDOM_PAIRS = 4
Z = 3.0
MIN_GAIN = 0.01 * eng.RANK_SAMPLE
DEFAULT = eng.Setting((2.0, 1.5, 1.0), 0.55, 20, 40)


def challengers(champion, seed):
    """값 하나만 바꾼 방법 전부 + 값 두 개를 바꾼 방법 몇 개 (회차 번호로 고정된 무작위)."""
    out = [replace(champion, **{field: value})
           for field, values in OPTIONS.items() for value in values if value != getattr(champion, field)]
    rng = random.Random(seed)
    target = len(out) + RANDOM_PAIRS
    for _ in range(200):
        if len(out) >= target:
            break
        a, b = rng.sample(list(OPTIONS), 2)
        change = {f: rng.choice([v for v in OPTIONS[f] if v != getattr(champion, f)]) for f in (a, b)}
        candidate = replace(champion, **change)
        if candidate not in out:
            out.append(candidate)
    return out


def _share(rank):
    return 100 * rank / (eng.RANK_SAMPLE + 1)


def decide(methods, ranks):
    """챔피언(0번)과 회차별로 짝지은 순위 차이로 판정. (새 챔피언, 표 행들)"""
    base = ranks[0]
    rows, best = [], None
    for k in range(1, len(methods)):
        diff = ranks[k] - base
        mean = float(diff.mean())
        se = float(diff.std(ddof=1)) / sqrt(len(diff)) if len(diff) > 1 else 0.0
        passed = se > 0 and mean < -Z * se and -mean >= MIN_GAIN
        rows.append((methods[k], float(ranks[k].mean()), mean, se, passed))
        if passed and (best is None or mean < best[2]):
            best = rows[-1]
    rows.sort(key=lambda r: r[1])
    return (best[0] if best else methods[0]), rows


def learn(draws, champion, prepared, log_progress=True):
    """반환: (다음 주에 쓸 방법, 텔레그램용 줄들, 결과.md용 본문)."""
    catalogs, histories, space = prepared
    methods = [champion] + challengers(champion, seed=int(draws.rounds[-1]))
    ranks = eng.evaluate(draws, catalogs, histories, space, methods, LEARN_FOLDS, log_progress)
    chosen, rows = decide(methods, ranks)
    base = float(ranks[0].mean())
    best = rows[0]
    lines = [f'· 도전 {len(rows)}개를 최근 {LEARN_FOLDS}회로 시험: 지금 방법 평균 상위 {_share(base):.1f}%, '
             f'가장 나은 도전 {_share(best[1]):.1f}%']
    if chosen is champion:
        lines.append('· 확실한 차이가 없어 지금 방법 유지')
    else:
        win = next(r for r in rows if r[0] is chosen)
        lines.append(f'· 교체: {chosen.label()}')
        lines.append(f'  평균 상위 {_share(base):.1f}% → {_share(win[1]):.1f}%')
    table = '\n'.join(f"| {m.label()} | {_share(r):.1f}% | {d:+,.0f} | {se:,.0f} | {'교체 기준 통과' if ok else ''} |"
                      for m, r, d, se, ok in rows)
    md = (f'## 주간 학습\n\n'
          f'지금 방법: {champion.label()} · 최근 {LEARN_FOLDS}회 평균 상위 {_share(base):.1f}%\n\n'
          f'다음 주 방법: {chosen.label()}\n\n'
          f'평균 상위 %는 각 회차를 그 이전 자료로만 학습했을 때 무작위 20만 후보 안 실제 1등의 위치입니다. 무작위면 50%입니다. '
          f'차이는 회차별 순위 차이의 평균이며, 평균 2,000 이상 낮고 표준오차 3배를 넘어야 교체합니다.\n\n'
          f'| 도전 방법 | 평균 상위 | 순위 차이 | 표준오차 | 판정 |\n| --- | ---: | ---: | ---: | --- |\n{table}\n')
    return chosen, lines, md
