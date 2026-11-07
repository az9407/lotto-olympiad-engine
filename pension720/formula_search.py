#!/usr/bin/env python3
"""공식 후보군 걷기 검증.

최근 F회 각각을 그 이전 자료로만 학습해, 같은 무작위 20만 후보 안에서 실제 1등이 몇 %에 오는지 잰다.
규칙 엔진, formulas.py의 공식들, 그리고 둘을 섞은 앙상블을 같은 회차·같은 후보로 비교한다.
--null R 을 주면 같은 길이의 가짜 무작위 자료 R개에서도 공식들을 돌려,
'공식을 이만큼 여러 개 시험하면 우연만으로 최고 공식이 몇 %까지 나오는지'를 함께 잰다.
"""
from __future__ import annotations

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from . import formulas as fm
    from . import predict_pension15 as eng
except ImportError:
    sys.path.insert(0, str(HERE))
    import formulas as fm
    import predict_pension15 as eng

ENSEMBLES = {
    'ens_engine_hot': ('엔진+빈도100', ('engine', 'hot_100')),
    'ens_engine_markov': ('엔진+마르코프', ('engine', 'markov_5')),
    'ens_engine_logit': ('엔진+로지스틱', ('engine', 'logit_100')),
    'ens_engine_feat': ('엔진+특징', ('engine', 'feat_all_strong')),
    'ens_simple': ('빈도+마르코프+자리쌍+특징', ('hot_100', 'markov_5', 'pairs_5', 'feat_all_strong')),
    'ens_all': ('전부 평균', ('engine', 'hot_100', 'cold_50', 'overdue', 'carry_main', 'markov_5', 'pairs_5',
                         'feat_all_strong', 'analog_30', 'logit_100')),
}


def _rank_last(s):
    return 1 + np.count_nonzero(s[:-1] > s[-1]) + 0.5 * np.count_nonzero(s[:-1] == s[-1])


def engine_joint(fold_pair, setting, draws, ix, ids):
    parts = [fold.fit(setting).score(ids) for fold in fold_pair]
    d, g = eng.split_ids(ids)
    return setting.mix[0] * parts[0] + setting.mix[1] * parts[1] + eng.tie_term(
        d, g, eng.frequency_logs(draws, ix, setting.time_weights))


LABELS = {}


def walk(draws, formulas, folds, setting=None, prepared=None, log_progress=True):
    """반환: (이름 목록, (공식 수, folds) 순위 배열). formulas는 Formula 목록이나 formulas.py key 목록.
    setting이 있으면 'engine'과 앙상블도 포함."""
    formulas = [fm.BY_KEY[f] if isinstance(f, str) else f for f in formulas]
    LABELS.update({f.key: f.label for f in formulas})
    keys = [f.key for f in formulas]
    n = len(draws.rounds)
    use_engine = setting is not None
    names = list(keys) + (['engine'] + list(ENSEMBLES) if use_engine else [])
    ranks = np.zeros((len(names), folds))
    if use_engine:
        catalogs, histories, space = prepared or eng.prepare(draws, folds)
    for f, ix in enumerate(range(n - folds, n)):
        ids = eng.rank_sample(draws, ix)
        d, g = eng.split_ids(ids)
        z = {}
        for k, formula in enumerate(formulas):
            s = np.nan_to_num(np.asarray(formula.fit(draws, ix)(d, g), dtype=float), nan=-1e9)
            ranks[k, f] = _rank_last(s)
            z[formula.key] = fm.zscore(s)
        if use_engine:
            context = eng.Context(draws, ix)
            pair = [eng.RuleFold(c, h, context, space) for c, h in zip(catalogs, histories)]
            s = engine_joint(pair, setting, draws, ix, ids)
            ranks[len(keys), f] = _rank_last(s)
            z['engine'] = fm.zscore(s)
            for e, (_, members) in enumerate(ENSEMBLES.values()):
                ranks[len(keys) + 1 + e, f] = _rank_last(np.mean([z[m] for m in members], axis=0))
        if log_progress and (f + 1) % 10 == 0:
            print(f'공식 검증 {f + 1}/{folds}회', file=sys.stderr, flush=True)
    return names, ranks


def label(name):
    if name == 'engine':
        return '규칙 엔진'
    if name in ENSEMBLES:
        return '앙상블 ' + ENSEMBLES[name][0]
    return LABELS.get(name) or fm.BY_KEY[name].label


def share(rank):
    return 100 * rank / (eng.RANK_SAMPLE + 1)


def table(names, ranks):
    """공식별 평균 상위 %, 표준오차, 50% 대비 z. 좋은 순."""
    pct = share(ranks)
    rows = []
    for name, p in zip(names, pct):
        se = p.std(ddof=1) / sqrt(len(p))
        rows.append((name, float(p.mean()), float(se), float((p.mean() - 50) / se) if se else 0.0))
    rows.sort(key=lambda r: r[1])
    return rows


def fake_draws(draws, seed):
    """같은 길이·같은 날짜의 완전 무작위 당첨번호."""
    rng = np.random.default_rng(seed)
    n = len(draws.rounds)
    return eng.Draws(draws.rounds, draws.dates, rng.integers(1, 6, n).astype(np.int8),
                     rng.integers(0, 10, (n, 6)).astype(np.int8), rng.integers(0, 10, (n, 6)).astype(np.int8))


def main():
    parser = argparse.ArgumentParser(description='공식 후보군 걷기 검증')
    parser.add_argument('data', nargs='?', type=Path, default=eng.DATA_CSV)
    parser.add_argument('--folds', type=int, default=100)
    parser.add_argument('--null', type=int, default=0, help='가짜 무작위 자료 개수')
    parser.add_argument('--out', type=Path, default=HERE / 'results' / '공식후보')
    args = parser.parse_args()
    draws = eng.load_draws(args.data)
    keys = [f.key for f in fm.FORMULAS]
    setting = eng.Setting((2.0, 1.5, 1.0), 0.55, 20, 40)
    names, ranks = walk(draws, keys, args.folds, setting)
    rows = table(names, ranks)
    nulls = []
    cheap = [k for k in keys if not k.startswith('logit')]
    for r in range(args.null):
        fake = fake_draws(draws, 1000 + r)
        _, fr = walk(fake, cheap, args.folds, log_progress=False)
        best = table(cheap, fr)[0]
        nulls.append((best[0], best[1]))
        print(f'가짜 자료 {r + 1}/{args.null}: 최고 {label(best[0])} {best[1]:.1f}%', file=sys.stderr, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [f'# 공식 후보군 걷기 검증 · {int(draws.rounds[-args.folds])}~{int(draws.rounds[-1])}회 {args.folds}회',
             '',
             '각 회차를 그 이전 자료로만 학습해 같은 무작위 20만 후보 안에서 실제 1등이 상위 몇 %에 왔는지의 평균입니다. '
             '낮을수록 좋고 무작위면 50%입니다. z는 50%와의 차이를 표준오차로 나눈 값으로, -2보다 작아야 우연이 아니라고 볼 여지가 생깁니다.',
             '', '| 순위 | 공식 | 평균 상위 % | 표준오차 | z |', '| ---: | --- | ---: | ---: | ---: |']
    lines += [f'| {i} | {label(n)} | {m:.1f}% | {se:.1f} | {z:+.2f} |' for i, (n, m, se, z) in enumerate(rows, 1)]
    if nulls:
        values = sorted(v for _, v in nulls)
        lines += ['', f'## 우연 기준: 가짜 무작위 자료 {len(nulls)}개',
                  f'로지스틱을 뺀 공식 {len(cheap)}개를 똑같이 돌렸을 때 최고 공식의 평균 상위 %: '
                  + ', '.join(f'{v:.1f}%' for v in values),
                  f'실제 자료의 최고 공식은 {rows[0][1]:.1f}% 입니다.']
    (args.out / '결과.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    np.savetxt(args.out / '회차별_상위퍼센트.csv', share(ranks).T, delimiter=',', fmt='%.3f',
               header=','.join(names), comments='')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
