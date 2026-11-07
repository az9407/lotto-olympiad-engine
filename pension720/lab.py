#!/usr/bin/env python3
"""연금복권720+ 창의 공식 실험실. 목표: 실제 1등을 평균 상위 10% 안에 놓는 공식 찾기.

공식 가족
- 차트 기법: 자리별 숫자·자리합·조·6자리 값·보너스를 가격 차트로 보고 이동평균, 지수평균, 모멘텀,
  되돌림, 볼린저, 피보나치, 일목 중간값, 중앙값, AR 회귀, 푸리에 외삽으로 다음 값을 겨냥하거나
  RSI·MACD·돌파 신호로 높은 숫자/낮은 숫자 쪽을 고른다
- 매크로 국면: 원·달러, 엔, 위안, 유로, WTI·브렌트 유가, S&P500, 금의 1주·4주·13주 변화와 52주 수준,
  주간 RSI, 이동평균 교차로 국면을 나누고 같은 국면 과거 회차의 자리별 숫자 빈도로 채점
- 달력·천체 국면: 회차 나머지 2~7, 계절, 월의 몇째 주, 달의 위상
- 복권 자체 국면: 직전 조·자리합·홀짝·고저·보너스 끝자리, 자리합 RSI, 이동평균 교차
- 시계열: 시차 2~52회 마르코프, 미출현 위험률, 닮은 회차 여러 방식
- 기계학습: 그래디언트 부스팅
- formulas.py의 기존 공식 27개
- 쌓기: 개발 구간 상위 공식들의 가중 합 (가중치는 개발 구간에서만 학습)

검증 규칙
- 개발 구간 100회: 여기서 공식을 고르고 가중치를 맞춘다
- 검증 구간 최근 40회: 고르는 데 전혀 쓰지 않은 회차. 진짜 실력은 이 숫자로만 본다
- 가짜 무작위 자료에서도 똑같이 돌려 우연만으로 개발 구간 최고가 얼마나 좋아 보이는지 함께 본다
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from . import formula_search as fsm
    from . import formulas as fm
    from . import predict_pension15 as eng
except ImportError:
    sys.path.insert(0, str(HERE))
    import formula_search as fsm
    import formulas as fm
    import predict_pension15 as eng

SAMPLE = 50_000
MACRO_CSV = HERE / 'data' / 'macro.csv'
MONTHLY = {'sp500', 'gold'}
POS = np.arange(6)


# ---------------------------------------------------------------- 매크로·달력 자료

def load_macro(path=MACRO_CSV):
    data = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            data[r['지표']].append((date.fromisoformat(r['날짜']).toordinal(), float(r['값'])))
    return {k: (np.array([d for d, _ in v]), np.array([x for _, x in v])) for k, v in sorted(data.items())}


MACRO_SOURCES = 'https://raw.githubusercontent.com/datasets/{}/main/data/{}'


def update_macro(path=MACRO_CSV):
    """datasets 공개 저장소에서 원·달러 등 매크로 값을 받아 macro.csv에 합친다. 실패한 출처는 기존 값 유지."""
    import io
    import urllib.request

    def get(repo, name):
        with urllib.request.urlopen(MACRO_SOURCES.format(repo, name), timeout=60) as r:
            return list(csv.reader(io.StringIO(r.read().decode('utf-8'))))

    rows = {}
    if path.exists():
        with path.open(encoding='utf-8') as f:
            rows = {(r['날짜'], r['지표']): r['값'] for r in csv.DictReader(f)}
    errors = []
    countries = {'South Korea': 'usdkrw', 'Japan': 'usdjpy', 'China': 'usdcny', 'Euro': 'usdeur'}
    jobs = (('exchange-rates', 'daily.csv', lambda t: [(d, countries[c], v) for d, c, v in t[1:] if c in countries]),
            ('oil-prices', 'wti-daily.csv', lambda t: [(d, 'wti', v) for d, v in t[1:]]),
            ('oil-prices', 'brent-daily.csv', lambda t: [(d, 'brent', v) for d, v in t[1:]]),
            ('s-and-p-500', 'data.csv', lambda t: [(r[0], 'sp500', r[1]) for r in t[1:]]),
            ('gold-prices', 'monthly.csv', lambda t: [(d + '-01' if len(d) == 7 else d, 'gold', v) for d, v in t[1:]]))
    for repo, name, parse in jobs:
        try:
            for d, key, v in parse(get(repo, name)):
                if d >= '2018-01-01' and v:
                    rows[(d, key)] = str(float(v))
        except Exception as error:   # 한 출처가 막혀도 나머지로 진행
            errors.append(f'{repo}/{name}: {type(error).__name__} {error}')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['날짜', '지표', '값'])
        writer.writerows((d, k, v) for (d, k), v in sorted(rows.items()))
    return errors


def _value_before(series, day, lag):
    """day - lag 이전에 이미 나와 있던 마지막 값. 월간 값은 그 달 평균이라 35일 늦춰 쓴다."""
    days, values = series
    i = int(np.searchsorted(days, day - lag, side='right')) - 1
    return values[i] if i >= 0 else np.nan


def moon_age(d):
    """2000-01-06 신월 기준 달 나이(일)."""
    return (d.toordinal() - date(2000, 1, 6).toordinal()) % 29.530588853


def draw_dates(draws):
    return list(draws.dates) + [draws.dates[-1] + timedelta(days=7)]


def _bins(x, edges):
    out = np.digitize(x, edges)
    out[np.isnan(x)] = -1
    return out


def _rsi(x, k=14):
    d = np.diff(x[-(k + 1):])
    up, down = d[d > 0].sum(), -d[d < 0].sum()
    return 50.0 if up + down == 0 else 100 * up / (up + down)


def build_states(draws, macro):
    """행 0..n(n은 목표 회차)마다 그 추첨 전에 알 수 있던 정보로 정한 국면 번호. 모를 때는 -1."""
    n = len(draws.rounds)
    dates = draw_dates(draws)
    rounds = np.append(draws.rounds, draws.rounds[-1] + 1)
    states = {}

    def prev_feature(f, edges):
        x = np.full(n + 1, np.nan)
        x[1:] = [f(t) for t in range(n)]
        return _bins(x, edges)

    main, bonus, groups = draws.main.astype(int), draws.bonus.astype(int), draws.groups.astype(int)
    states['직전 조'] = np.append(-1, groups - 1)
    states['직전 자리합'] = prev_feature(lambda t: main[t].sum(), [23, 32])
    states['직전 홀수 개수'] = prev_feature(lambda t: (main[t] % 2).sum(), [3, 4])
    states['직전 5 이상 개수'] = prev_feature(lambda t: (main[t] >= 5).sum(), [3, 4])
    states['직전 서로 다른 숫자'] = prev_feature(lambda t: len(set(main[t])), [5, 6])
    states['직전 끝자리 홀짝'] = prev_feature(lambda t: main[t, 5] % 2, [1])
    states['직전 보너스 끝자리 홀짝'] = prev_feature(lambda t: bonus[t, 5] % 2, [1])
    sums = main.sum(axis=1).astype(float)
    states['자리합 RSI'] = np.array([-1] + [-1 if t < 15 else int(np.digitize(_rsi(sums[:t]), [35, 65]))
                                          for t in range(1, n + 1)])
    states['자리합 이평 교차'] = np.array([-1 if t < 20 else int(sums[t - 5:t].mean() > sums[t - 20:t].mean())
                                     for t in range(n + 1)])
    for k in range(2, 8):
        states[f'회차 mod {k}'] = rounds % k
    states['계절'] = np.array([(d.month % 12) // 3 for d in dates])
    states['월의 몇째 주'] = np.array([min(3, (d.day - 1) // 7) for d in dates])
    ages = np.array([moon_age(d) for d in dates])
    states['달 위상 8'] = (ages / 29.530588853 * 8).astype(int) % 8
    states['달 위상 4'] = (ages / 29.530588853 * 4).astype(int) % 4
    for name, series in macro.items():
        lag = 35 if name in MONTHLY else 1
        step = 30 if name in MONTHLY else 7
        day = np.array([d.toordinal() for d in dates])

        def at(offset):
            return np.array([_value_before(series, x - offset, lag) for x in day])

        now = at(0)
        for weeks, label in ((1, '1주'), (4, '4주'), (13, '13주')):
            if name in MONTHLY and weeks == 1:
                continue
            past = at(step * weeks if name not in MONTHLY else 30 * max(1, weeks // 4))
            change = now / past - 1
            states[f'{name} {label} 방향'] = _bins(change, [0.0])
            states[f'{name} {label} 크기'] = _bins(change, [-0.01, 0.01])
        year = np.stack([at(7 * w) for w in range(52)])
        z = (now - np.nanmean(year, axis=0)) / (np.nanstd(year, axis=0) + 1e-12)
        states[f'{name} 52주 수준'] = _bins(z, [-0.5, 0.5])
        weekly = np.stack([at(7 * w) for w in range(15)][::-1])
        rsi = np.array([np.nan if np.isnan(weekly[:, i]).any() else _rsi(weekly[:, i]) for i in range(n + 1)])
        states[f'{name} 주간 RSI'] = _bins(rsi, [30, 70])
        ma4, ma13 = np.nanmean(year[:4], axis=0), np.nanmean(year[:13], axis=0)
        states[f'{name} 4주·13주 이평 교차'] = _bins(ma4 - ma13, [0.0])
    return states


def macro_vector(draws, macro):
    """기계학습용 숫자 특징: 지표별 1주·4주 변화율과 52주 수준, 달 나이. 행 0..n."""
    dates = draw_dates(draws)
    day = np.array([d.toordinal() for d in dates])
    cols = [np.array([moon_age(d) for d in dates]) / 29.53]
    for name, series in macro.items():
        lag = 35 if name in MONTHLY else 1
        now = np.array([_value_before(series, x, lag) for x in day])
        for off in (7, 28):
            past = np.array([_value_before(series, x - off, lag) for x in day])
            cols.append(now / past - 1)
        year = np.stack([np.array([_value_before(series, x - 7 * w, lag) for x in day]) for w in range(0, 52, 4)])
        cols.append((now - np.nanmean(year, 0)) / (np.nanstd(year, 0) + 1e-12))
    return np.nan_to_num(np.stack(cols, axis=1))


# ---------------------------------------------------------------- 국면별 빈도 공식

def conditional(state, strength, sign):
    def fit(draws, ix):
        s = state[ix]
        if s < 0:
            return lambda d, g: np.zeros(len(d))
        rows = draws.main[:ix]
        same = state[:ix] == s
        base = np.stack([np.bincount(rows[:, j], minlength=10) for j in range(6)]) + 1.0
        base /= base.sum(axis=1, keepdims=True)
        hit = np.stack([np.bincount(rows[same, j], minlength=10) for j in range(6)]).astype(float)
        lift = np.log((hit + strength * base) / (same.sum() + strength) / base)
        g_all = np.bincount(draws.groups[:ix] - 1, minlength=5) + 1.0
        g_all /= g_all.sum()
        g_hit = np.bincount(draws.groups[:ix][same] - 1, minlength=5).astype(float)
        g_lift = np.log((g_hit + strength * g_all) / (same.sum() + strength) / g_all)
        return lambda d, g: sign * (lift[POS, d].sum(axis=1) + g_lift[g])
    return fit


# ---------------------------------------------------------------- 차트 기법

def _ema(x, a):
    out = x[0].astype(float)
    for v in x[1:]:
        out = a * v + (1 - a) * out
    return out


def _ar(x, p):
    """열마다 최소제곱 AR(p) 한 걸음 예측."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = []
    for col in x.T:
        if len(col) <= p + 10:
            out.append(col.mean())
            continue
        a = np.column_stack([col[p - i - 1:len(col) - i - 1] for i in range(p)] + [np.ones(len(col) - p)])
        coef, *_ = np.linalg.lstsq(a, col[p:], rcond=None)
        out.append(float(np.r_[col[-1:-p - 1:-1], 1.0] @ coef))
    return np.array(out)


def _fourier(x, n, keep):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    seg = x[-n:]
    spec = np.fft.rfft(seg, axis=0)
    amp = np.abs(spec)
    amp[0] = np.inf   # 평균은 항상 남긴다
    out = []
    for c in range(seg.shape[1]):
        top = np.argsort(-amp[:, c])[:keep + 1]
        k = np.arange(spec.shape[0])[top]
        value = (spec[top, c] * np.exp(2j * np.pi * k * len(seg) / len(seg))).real
        weight = np.where(k == 0, 1, 2)
        if len(seg) % 2 == 0:
            weight = np.where(k == len(seg) // 2, 1, weight)
        out.append(float((weight * value).sum() / len(seg)))
    return np.array(out)


def technique(name):
    """시계열 x(T, k) → 다음 값 겨냥점(k,)."""
    if name == '직전값':
        return lambda x: x[-1].astype(float)
    if name.startswith('이평'):
        k = int(name[2:])
        return lambda x: x[-k:].mean(axis=0)
    if name.startswith('중앙'):
        k = int(name[2:])
        return lambda x: np.median(x[-k:], axis=0)
    if name.startswith('지수'):
        a = float(name[2:])
        return lambda x: _ema(x[-60:], a)
    if name.startswith('모멘텀'):
        k, m = name[3:].split('x')
        k, m = int(k), float(m)
        return lambda x: x[-1] + m * (x[-1] - x[-1 - k])
    if name.startswith('되돌림'):
        k = int(name[3:])
        return lambda x: 2 * x[-k:].mean(axis=0) - x[-1]
    if name.startswith('볼린저'):
        k = int(name[3:])

        def f(x):
            ma, sd = x[-k:].mean(axis=0), x[-k:].std(axis=0) + 1e-9
            return np.where(np.abs(x[-1] - ma) > 1.5 * sd, ma, x[-1])
        return f
    if name.startswith('피보'):
        r = float(name[2:])
        return lambda x: x[-20:].min(axis=0) + r * (x[-20:].max(axis=0) - x[-20:].min(axis=0))
    if name == '일목':
        return lambda x: ((x[-9:].max(0) + x[-9:].min(0)) / 2 + (x[-26:].max(0) + x[-26:].min(0)) / 2) / 2
    if name.startswith('AR'):
        p = int(name[2:])
        return lambda x: _ar(x, p)
    if name.startswith('푸리에'):
        n, keep = map(int, name[3:].split('-'))
        return lambda x: _fourier(x, n, keep)
    raise ValueError(name)


TECHNIQUES = ('직전값', '이평3', '이평5', '이평10', '이평20', '이평50', '중앙10', '지수0.2', '지수0.5',
              '모멘텀1x0.5', '모멘텀1x1', '모멘텀3x1', '되돌림5', '되돌림10', '되돌림20', '볼린저10', '볼린저20',
              '피보0.382', '피보0.618', '일목', 'AR1', 'AR2', 'AR3', '푸리에64-3', '푸리에128-5')


def chart(target_of, series):
    """series: 자리 숫자·자리합·조·6자리 값·보너스. 겨냥점과 가까울수록 높은 점수."""
    def fit(draws, ix):
        if series == '보너스→1등':
            x = draws.bonus[:ix].astype(float)
        elif series == '자리합':
            x = draws.main[:ix].sum(axis=1, keepdims=True).astype(float)
        elif series == '조':
            x = draws.groups[:ix, None].astype(float)
        elif series == '6자리 값':
            x = (draws.main[:ix] @ eng.POWERS)[:, None] / 1e5
        else:
            x = draws.main[:ix].astype(float)
        target = target_of(x)
        sd = np.maximum(x[-20:].std(axis=0), 0.5)
        if series in ('자리 숫자', '보너스→1등'):
            return lambda d, g: -(((d - target) / sd) ** 2).sum(axis=1)
        if series == '자리합':
            return lambda d, g: -((d.sum(axis=1) - target[0]) / sd[0]) ** 2
        if series == '조':
            return lambda d, g: -((g + 1 - target[0]) / sd[0]) ** 2
        return lambda d, g: -(((d @ eng.POWERS) / 1e5 - target[0]) / sd[0]) ** 2
    return fit


def signal(kind, sign):
    """자리별 RSI·MACD·돌파 신호로 높은 숫자 또는 낮은 숫자 쪽을 고른다."""
    def fit(draws, ix):
        x = draws.main[:ix].astype(float)
        if kind == 'RSI':
            r = np.array([_rsi(x[:, j]) for j in range(6)])
            direction = np.where(r > 70, -1, np.where(r < 30, 1, 0))
        elif kind == 'MACD':
            direction = np.sign(_ema(x[-60:], 2 / 13) - _ema(x[-60:], 2 / 27))
        else:
            prev = x[-21:-1]
            direction = np.where(x[-1] >= prev.max(0), 1, np.where(x[-1] <= prev.min(0), -1, 0))
        return lambda d, g: sign * ((d - 4.5) * direction).sum(axis=1)
    return fit


# ---------------------------------------------------------------- 시계열 기타

def lag_markov(lag, alpha=1.0):
    def fit(draws, ix):
        rows = draws.main[:ix]
        table = np.zeros((6, 10, 10))
        for j in range(6):
            np.add.at(table[j], (rows[:-lag, j], rows[lag:, j]), 1)
        logt = np.log((table + alpha) / (table.sum(axis=2, keepdims=True) + 10 * alpha))
        prev = rows[-lag]
        return lambda d, g: logt[POS, prev, d].sum(axis=1)
    return fit


def hazard(strength):
    """미출현 a회째에 나올 확률을 과거 간격 분포로 추정 (재생 과정)."""
    def fit(draws, ix):
        rows = draws.main[:ix]
        gaps = np.zeros(60)
        at_risk = np.zeros(60)
        for j in range(6):
            for v in range(10):
                hits = np.flatnonzero(rows[:, j] == v)
                for a, b in zip(hits, hits[1:]):
                    gap = min(b - a - 1, 59)
                    gaps[gap] += 1
                    at_risk[:gap + 1] += 1
        h = (gaps + strength * 0.1) / (at_risk + strength)
        ages = np.minimum(np.array(eng.Context(draws, ix).ages(0)), 59)
        logh = np.log(h[ages])
        return lambda d, g: logh[POS, d].sum(axis=1)
    return fit


def analog_by(kind, k):
    def fit(draws, ix):
        rows = draws.main[:ix].astype(int)
        prev, prev_b, prev_g = rows[-1], draws.bonus[ix - 1].astype(int), int(draws.groups[ix - 1])
        past = rows[:-2]
        if kind == '자리합':
            sim = -np.abs(past.sum(1) - prev.sum())
        elif kind == '조+자리합':
            sim = (draws.groups[:ix - 2] == prev_g) * 100 - np.abs(past.sum(1) - prev.sum())
        elif kind == '보너스':
            sim = (draws.bonus[:ix - 2].astype(int) == prev_b).sum(1)
        else:
            sim = (np.sort(past, 1) == np.sort(prev)).sum(1)   # 숫자 구성
        nearest = np.argsort(-(sim + 1e-3 * np.arange(len(past))))[:k] + 1
        dl = np.log((np.stack([np.bincount(rows[nearest, j], minlength=10) for j in range(6)]) + 1) / (k + 10))
        gl = np.log((np.bincount(draws.groups[nearest] - 1, minlength=5) + 1) / (k + 5))
        return lambda d, g: dl[POS, d].sum(axis=1) + gl[g]
    return fit


# ---------------------------------------------------------------- 기계학습

class Booster:
    """그래디언트 부스팅. 10회마다 그 시점 이전 행만으로 다시 학습한다."""

    def __init__(self, macro_x, depth, rounds):
        self.macro_x, self.depth, self.rounds = macro_x, depth, rounds
        self.models = {}

    def __call__(self, draws, ix):
        from sklearn.ensemble import HistGradientBoostingClassifier
        block = ix - ix % 10
        if block not in self.models:
            train = fm._TRAIN.setdefault(id(draws), fm._TrainRows(draws))
            xs, ys = [], []
            for t in range(10, block):
                x, y = train.get(t)
                xs.append(np.hstack((x, np.repeat(self.macro_x[t][None], len(x), 0))))
                ys.append(y)
            model = HistGradientBoostingClassifier(max_depth=self.depth, max_iter=self.rounds, learning_rate=0.05,
                                                   l2_regularization=1.0, random_state=0)
            model.fit(np.vstack(xs), np.concatenate(ys))
            self.models[block] = model
        model, ctx, mx = self.models[block], fm._context(draws, ix), self.macro_x[ix]

        def score(d, g):
            x = fm._phi(d, g, ctx)
            return model.predict_proba(np.hstack((x, np.repeat(mx[None], len(x), 0))))[:, 1]
        return score


# ---------------------------------------------------------------- 공식 목록

def creative_formulas(draws, macro, with_ml=True):
    out = [f for f in fm.FORMULAS]
    for series in ('자리 숫자', '자리합', '조', '6자리 값', '보너스→1등'):
        for name in TECHNIQUES:
            out.append(fm.Formula(f'chart|{series}|{name}', f'차트 {series} {name}', chart(technique(name), series)))
    for kind in ('RSI', 'MACD', '돌파'):
        for sign, word in ((1, '추세'), (-1, '역추세')):
            out.append(fm.Formula(f'signal|{kind}|{sign}', f'신호 {kind} {word}', signal(kind, sign)))
    for lag in (2, 3, 4, 5, 7, 10, 13, 26, 52):
        out.append(fm.Formula(f'lagmarkov|{lag}', f'{lag}회 전 같은 자리 전이', lag_markov(lag)))
    for s in (1.0, 10.0):
        out.append(fm.Formula(f'hazard|{s:g}', f'미출현 위험률 평활 {s:g}', hazard(s)))
    for kind in ('자리합', '조+자리합', '보너스', '숫자 구성'):
        for k in (10, 30, 60):
            out.append(fm.Formula(f'analog|{kind}|{k}', f'닮은 회차 {kind} {k}개', analog_by(kind, k)))
    for name, state in build_states(draws, macro).items():
        for strength in (10.0, 40.0):
            for sign, word in ((1, ''), (-1, ' 반대')):
                out.append(fm.Formula(f'cond|{name}|{strength:g}|{sign}', f'국면 {name} 수축{strength:g}{word}',
                                      conditional(state, strength, sign)))
    if with_ml:
        mx = macro_vector(draws, macro)
        for depth, rounds in ((2, 100), (3, 200)):
            out.append(fm.Formula(f'gbm|{depth}|{rounds}', f'부스팅 깊이{depth} {rounds}회', Booster(mx, depth, rounds)))
    return out


# ---------------------------------------------------------------- 걷기 검증

def fold_sample(draws, ix):
    """실제 1등(맨 끝) + 그 회차 번호로 고정한 무작위 후보 5만 개."""
    rng = np.random.default_rng(int(draws.rounds[ix]) * 31 + 5)
    answer = (int(draws.groups[ix]) - 1) * eng.N_NUMBERS + int(draws.main[ix] @ eng.POWERS)
    ids = rng.choice(eng.N_CANDIDATES - 1, SAMPLE, replace=False)
    ids += ids >= answer
    return np.append(ids, answer).astype(np.int64)


def pct_rank(s):
    return 100 * (1 + np.count_nonzero(s[:-1] > s[-1]) + 0.5 * np.count_nonzero(s[:-1] == s[-1])) / (SAMPLE + 1)


def walk(draws, formulas, folds, keep=(), keep_neg=3000, log=True):
    """(공식 수, folds) 상위 % 배열. keep에 든 공식은 쌓기용 z 점수를 (실제 + 음성 표본, 전체)로 보관."""
    n = len(draws.rounds)
    out = np.zeros((len(formulas), folds))
    kept = {}
    start = time.time()
    for f, ix in enumerate(range(n - folds, n)):
        ids = fold_sample(draws, ix)
        d, g = eng.split_ids(ids)
        for k, formula in enumerate(formulas):
            s = np.nan_to_num(np.asarray(formula.fit(draws, ix)(d, g), dtype=float), nan=-1e9)
            out[k, f] = pct_rank(s)
            if formula.key in keep:
                kept.setdefault(ix, {})[formula.key] = fm.zscore(s).astype(np.float32)
        if log and (f + 1) % 10 == 0:
            print(f'실험 {f + 1}/{folds}회 · {time.time() - start:.0f}초', file=sys.stderr, flush=True)
    return out, kept


def stack_weights(kept, keys, dev_rounds, neg=3000, l2=1.0):
    """개발 구간 회차에서 '실제 1등 대 무작위' 로지스틱으로 상위 공식 z 점수의 가중치를 학습."""
    xs, ys = [], []
    for ix in dev_rounds:
        z = np.stack([kept[ix][k] for k in keys], axis=1)
        pick = np.r_[np.arange(neg), len(z) - 1]
        xs.append(z[pick])
        ys.append(np.r_[np.zeros(neg), 1.0])
    x, y = np.vstack(xs).astype(float), np.concatenate(ys)
    w = np.zeros(x.shape[1] + 1)
    z = np.hstack((x, np.ones((len(x), 1))))
    reg = np.r_[np.full(x.shape[1], l2), 0]
    for _ in range(30):
        p = 1 / (1 + np.exp(-(z @ w)))
        step = np.linalg.solve((z * (p * (1 - p))[:, None]).T @ z + np.diag(reg + 1e-9), z.T @ (p - y) + reg * w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1]


def summarize(pct):
    m = pct.mean(axis=1)
    se = pct.std(axis=1, ddof=1) / sqrt(pct.shape[1])
    return m, se


def experiment(draws, macro, dev=100, test=40, top=20, with_ml=True, log=True):
    formulas = creative_formulas(draws, macro, with_ml)
    folds = dev + test
    pct, _ = walk(draws, formulas, folds, log=log)
    dev_m, dev_se = summarize(pct[:, :dev])
    test_m, test_se = summarize(pct[:, dev:])
    order = np.argsort(dev_m)
    chosen = [formulas[i] for i in order[:top]]
    keys = [f.key for f in chosen]
    _, kept = walk(draws, chosen, folds, keep=set(keys), log=False)
    n = len(draws.rounds)
    dev_rounds = list(range(n - folds, n - test))
    w = stack_weights(kept, keys, dev_rounds)
    stack = np.array([pct_rank(np.stack([kept[ix][k] for k in keys], 1).astype(float) @ w)
                      for ix in range(n - folds, n)])
    avg = np.array([pct_rank(np.mean([kept[ix][k] for k in keys[:10]], axis=0))
                    for ix in range(n - folds, n)])
    return {'formulas': formulas, 'pct': pct, 'dev': (dev_m, dev_se), 'test': (test_m, test_se), 'order': order,
            'stack': stack, 'avg10': avg, 'weights': dict(zip(keys, w)), 'dev_n': dev, 'test_n': test}


def report(real, nulls, draws):
    f, (dm, dse), (tm, tse), order = real['formulas'], real['dev'], real['test'], real['order']
    n, dev, test = len(draws.rounds), real['dev_n'], real['test_n']
    r0, r1, r2 = int(draws.rounds[n - dev - test]), int(draws.rounds[n - test - 1]), int(draws.rounds[n - test])
    st, av = real['stack'], real['avg10']
    lines = [f'# 창의 공식 실험 · 공식 {len(f)}개',
             '',
             f'개발 구간 {r0}~{r1}회 {dev}회에서 고르고, 검증 구간 {r2}~{int(draws.rounds[-1])}회 {test}회로 확인했습니다. '
             '숫자는 실제 1등이 무작위 5만 후보 안에서 상위 몇 %에 왔는지의 평균이며 무작위면 50%입니다.',
             '',
             '## 개발 구간 상위 20개와 검증 구간 성적',
             '', '| 순위 | 공식 | 개발 | 검증 |', '| ---: | --- | ---: | ---: |']
    lines += [f'| {i} | {f[k].label} | {dm[k]:.1f}% | {tm[k]:.1f}% |' for i, k in enumerate(order[:20], 1)]
    lines += ['', f'개발 상위 20개 쌓기: 개발 {st[:dev].mean():.1f}% · 검증 {st[dev:].mean():.1f}%',
              f'개발 상위 10개 평균: 개발 {av[:dev].mean():.1f}% · 검증 {av[dev:].mean():.1f}%',
              f'검증 구간 기준 최고 (뒤늦게 고른 것이라 참고만): {f[int(np.argmin(tm))].label} {tm.min():.1f}%',
              f'개발 구간에서 40% 아래였던 공식 {int((dm < 40).sum())}개, 그중 검증에서도 40% 아래 {int(((dm < 40) & (tm < 40)).sum())}개',
              f'상위 10% 목표: 검증 구간에서 평균 10% 이하인 방법 {"있음" if min(tm.min(), st[dev:].mean()) <= 10 else "없음"}']
    if nulls:
        lines += ['', f'## 같은 실험을 가짜 무작위 자료 {len(nulls)}개에서',
                  '| 자료 | 개발 최고 | 그 공식의 검증 | 개발 40% 아래 개수 | 쌓기 검증 |', '| --- | ---: | ---: | ---: | ---: |']
        for i, nu in enumerate(nulls, 1):
            ndm, ntm, b = nu['dev'][0], nu['test'][0], nu['order'][0]
            lines.append(f"| 가짜 {i} | {ndm[b]:.1f}% | {ntm[b]:.1f}% | {int((ndm < 40).sum())} | {nu['stack'][nu['dev_n']:].mean():.1f}% |")
        b = order[0]
        lines.append(f"| 실제 | {dm[b]:.1f}% | {tm[b]:.1f}% | {int((dm < 40).sum())} | {st[dev:].mean():.1f}% |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='창의 공식 실험실')
    parser.add_argument('data', nargs='?', type=Path, default=eng.DATA_CSV)
    parser.add_argument('--null', type=int, default=1)
    parser.add_argument('--no-ml', action='store_true')
    parser.add_argument('--out', type=Path, default=HERE / 'results' / '창의공식')
    args = parser.parse_args()
    draws = eng.load_draws(args.data)
    macro = load_macro()
    real = experiment(draws, macro, with_ml=not args.no_ml)
    nulls = []
    for r in range(args.null):
        fake = fsm.fake_draws(draws, 2000 + r)
        nulls.append(experiment(fake, macro, with_ml=not args.no_ml, log=False))
        print(f'가짜 자료 {r + 1}/{args.null} 끝', file=sys.stderr, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    text = report(real, nulls, draws)
    (args.out / '결과.md').write_text(text, encoding='utf-8')
    keys = [f.key for f in real['formulas']]
    with (args.out / '공식별.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(('공식', '설명', '개발 평균 %', '검증 평균 %'))
        for i, k in enumerate(keys):
            w.writerow((k, real['formulas'][i].label, f"{real['dev'][0][i]:.2f}", f"{real['test'][0][i]:.2f}"))
    print(text)


if __name__ == '__main__':
    main()
