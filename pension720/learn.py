"""연금복권720+ 주간 학습.

1단계 엔진 다듬기: 지금 엔진 설정과, 값을 하나나 둘 바꾼 설정 23개를 최근 60회로 겨룬다.
2단계 공식 고르기: 다듬은 엔진, formulas.py·lab.py의 다른 공식 약 530개(차트 기법, 매크로·달력 국면,
시계열, 부스팅 포함), 앙상블 6개를 같은 60회로 겨룬다.
각 회차는 그 회차 이전 자료로만 학습해 같은 무작위 20만 후보 안에서 실제 1등의 순위를 잰다.
지금 쓰는 것보다 회차별 짝 비교에서 평균 순위가 2,000 이상 낮고 그 차이가 표준오차의 Z배를 넘을 때만 바꾼다.
Z는 도전자 수만큼 여러 번 시험하는 것을 감안해 우연히 통과할 확률이 5%가 되도록 정한다 (23개면 약 3.1, 560개면 약 3.9).
우연히 좋아 보이는 쪽으로 매주 흔들리지 않게 하려는 기준이다.
"""
from __future__ import annotations

import random
from dataclasses import replace
from math import sqrt
from statistics import NormalDist

import numpy as np

try:
    from . import formula_search as fs
    from . import formulas as fm
    from . import lab
    from . import predict_pension15 as eng
except ImportError:
    import formula_search as fs
    import formulas as fm
    import lab
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
MIN_GAIN = 0.01 * eng.RANK_SAMPLE
DEFAULT = eng.Setting((2.0, 1.5, 1.0), 0.55, 20, 40)


def pool(draws):
    """2단계에서 겨루는 공식들 (무작위 기준 제외)."""
    return [f for f in lab.creative_formulas(draws, lab.load_macro()) if f.key != 'random']


def threshold(challengers_count):
    return max(3.0, NormalDist().inv_cdf(1 - 0.05 / max(1, challengers_count)))


def challengers(champion, seed):
    """값 하나만 바꾼 설정 전부 + 값 두 개를 바꾼 설정 몇 개 (회차 번호로 고정된 무작위)."""
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
    """0번(지금 것)과 회차별로 짝지은 순위 차이로 판정. (고른 것, 표 행들 좋은 순)"""
    base = ranks[0]
    rows, best = [], None
    z = threshold(len(methods) - 1)
    for k in range(1, len(methods)):
        diff = ranks[k] - base
        mean = float(diff.mean())
        se = float(diff.std(ddof=1)) / sqrt(len(diff)) if len(diff) > 1 else 0.0
        passed = se > 0 and mean < -z * se and -mean >= MIN_GAIN
        rows.append((methods[k], float(ranks[k].mean()), mean, se, passed))
        if passed and (best is None or mean < best[2]):
            best = rows[-1]
    rows.sort(key=lambda r: r[1])
    return (best[0] if best else methods[0]), rows


def formula_label(key, setting=None):
    if key == 'engine':
        return '규칙 엔진' + (f' · {setting.label()}' if setting else '')
    return fs.label(key)


# ---------------------------------------------------------------- 500만 후보 점수

def tie_all(draws, ix, weights):
    logs = eng.frequency_logs(draws, ix, weights)
    out = np.empty(eng.N_CANDIDATES)
    for lo in range(0, eng.N_CANDIDATES, 250_000):
        d, g = eng.split_ids(np.arange(lo, lo + 250_000, dtype=np.int32))
        out[lo:lo + 250_000] = eng.tie_term(d, g, logs)
    return out


def engine_full(draws, ix, setting, prepared):
    """ix 시점 엔진 모델과 500만 후보 점수 (S, (1등 z, 보너스 z), 모델들, 시점)."""
    catalogs, histories, space = prepared
    context = eng.Context(draws, ix)
    models = [eng.RuleFold(c, h, context, space).fit(setting) for c, h in zip(catalogs, histories)]
    scores, parts = eng.score_all(draws, setting, models, index=ix)
    return scores, parts, models, context


def champion_full(draws, ix, key, setting, prepared, engine_scores=None):
    """공식 key로 ix 시점 500만 후보 점수. 엔진이 아니면 z 점수에 동점 정리용 빈도 항을 더한다."""
    if key == 'engine':
        return engine_scores if engine_scores is not None else engine_full(draws, ix, setting, prepared)[0]
    members = fs.ENSEMBLES[key][1] if key in fs.ENSEMBLES else (key,)
    registry = {f.key: f for f in lab.creative_formulas(draws, lab.load_macro())}
    parts = []
    for m in members:
        if m == 'engine':
            s = engine_scores if engine_scores is not None else engine_full(draws, ix, setting, prepared)[0]
        else:
            s = np.nan_to_num(fm.full_scores(registry[m], draws, ix), nan=-1e9)
        parts.append(fm.zscore(s))
    return np.mean(parts, axis=0) + tie_all(draws, ix, setting.time_weights)


# ---------------------------------------------------------------- 학습

def learn(draws, formula, setting, prepared, log_progress=True):
    """반환: (다음 주 공식 key, 다음 주 엔진 설정, 요약 줄들, 결과.md용 본문, 비교표 행들)."""
    catalogs, histories, space = prepared
    methods = [setting] + challengers(setting, seed=int(draws.rounds[-1]))
    ranks = eng.evaluate(draws, catalogs, histories, space, methods, LEARN_FOLDS, log_progress)
    new_setting, rows = decide(methods, ranks)

    names, franks = fs.walk(draws, pool(draws), LEARN_FOLDS, new_setting, prepared, log_progress)
    if formula not in names:
        formula = 'engine'
    order = [formula] + [n for n in names if n != formula]
    new_formula, frows = decide(order, franks[[names.index(n) for n in order]])
    current = float(franks[names.index(formula)].mean())
    best = frows[0]

    lines = [f'· 엔진 설정 {len(rows)}개를 최근 {LEARN_FOLDS}회로 시험: 지금 설정 평균 상위 {_share(ranks[0].mean()):.1f}%, '
             f'가장 나은 설정 {_share(rows[0][1]):.1f}%',
             '· 엔진 설정 유지' if new_setting is setting else f'· 엔진 설정 교체: {new_setting.label()}',
             f'· 다른 공식·앙상블 {len(frows)}개도 같은 {LEARN_FOLDS}회로 비교: 지금 공식 {formula_label(formula)} '
             f'상위 {_share(current):.1f}%, 가장 나은 {formula_label(best[0])} {_share(best[1]):.1f}%',
             '· 확실한 차이가 없어 지금 공식 유지' if new_formula == formula
             else f'· 공식 교체: {formula_label(new_formula)}']
    engine_table = '\n'.join(f"| {m.label()} | {_share(r):.1f}% | {d:+,.0f} | {se:,.0f} | {'교체 기준 통과' if ok else ''} |"
                             for m, r, d, se, ok in rows)
    formula_table = '\n'.join(f"| {formula_label(k)} | {_share(r):.1f}% | {d:+,.0f} | {se:,.0f} | {'교체 기준 통과' if ok else ''} |"
                              for k, r, d, se, ok in frows[:30])
    md = (f'## 주간 학습\n\n'
          f'지금 공식: {formula_label(formula, setting)} · 최근 {LEARN_FOLDS}회 평균 상위 {_share(current):.1f}%\n\n'
          f'다음 주 공식: {formula_label(new_formula, new_setting)}\n\n'
          f'평균 상위 %는 각 회차를 그 이전 자료로만 학습했을 때 무작위 20만 후보 안 실제 1등의 위치입니다. 무작위면 50%입니다. '
          f'차이는 지금 것과의 회차별 순위 차이 평균이며, 평균 2,000 이상 낮고 표준오차 3배를 넘어야 교체합니다.\n\n'
          f'### 엔진 설정\n\n| 설정 | 평균 상위 | 순위 차이 | 표준오차 | 판정 |\n| --- | ---: | ---: | ---: | --- |\n{engine_table}\n\n'
          f'### 다른 공식과 앙상블 {len(frows)}개 중 상위 30개 · 교체 기준 Z {threshold(len(frows)):.2f}\n\n| 공식 | 평균 상위 | 순위 차이 | 표준오차 | 판정 |\n| --- | ---: | ---: | ---: | --- |\n{formula_table}\n')
    unit = 100 / (eng.RANK_SAMPLE + 1)
    table = ([('엔진 설정', m.label(), _share(r), d * unit, se * unit, ok) for m, r, d, se, ok in rows]
             + [('공식', formula_label(k), _share(r), d * unit, se * unit, ok) for k, r, d, se, ok in frows])
    return new_formula, new_setting, lines, md, table
