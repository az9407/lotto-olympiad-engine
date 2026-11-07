"""연금복권720+ 공식 후보군.

규칙 엔진과 전혀 다른 방식의 점수 공식들. 모두 '그 회차 이전 자료만' 보고 후보(조, 6자리)에 점수를 준다.
fit(draws, ix) 는 ix번째 행을 맞히는 시점의 채점 함수 score(digits, groups)를 돌려준다.
digits는 (n, 6) 정수, groups는 0~4.

가족
- 빈도: 자리별 숫자·조가 최근 W회에 많이 나온 쪽
- 냉각: 빈도의 반대
- 미출현: 오래 안 나온 숫자·조 쪽 (또는 반대로 최근 나온 쪽)
- 이월: 직전 1등·보너스와 같은 자리 같은 숫자(또는 ±1) 개수
- 마르코프: 자리별로 직전 숫자 → 다음 숫자 전이 빈도, 조 전이
- 자리쌍: 같은 번호 안 이웃 자리 숫자쌍의 동반 출현(PMI)
- 특징: 합·홀짝·고저·서로 다른 숫자 수 등 특징값이 이론 확률보다 자주 나온 쪽 (나이브 베이즈)
- 유사회차: 직전 번호와 닮은 과거 회차들의 '다음 회차' 빈도
- 로지스틱: 위 특징들을 실제 1등 대 무작위 후보로 학습한 선형 모델
- 무작위: 비교 기준. 평균 50%가 나와야 정상
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from . import predict_pension15 as eng
except ImportError:
    import predict_pension15 as eng

POS = np.arange(6)


@dataclass(frozen=True)
class Formula:
    key: str
    label: str
    fit: Callable    # (draws, ix) -> score(digits, groups)


def _since(ix, window):
    return max(0, ix - window) if window else 0


def _digit_log(rows, alpha=1.0):
    counts = np.stack([np.bincount(rows[:, j], minlength=10) for j in range(6)]).astype(float)
    return np.log((counts + alpha) / (counts.sum(axis=1, keepdims=True) + 10 * alpha))


def _group_log(groups0, alpha=1.0):
    counts = np.bincount(groups0, minlength=5).astype(float)
    return np.log((counts + alpha) / (counts.sum() + 5 * alpha))


def hot(window, sign=1.0, bonus=False):
    def fit(draws, ix):
        lo = _since(ix, window)
        rows = draws.main[lo:ix]
        if bonus:
            rows = np.vstack((rows, draws.bonus[lo:ix]))
        dl, gl = _digit_log(rows), _group_log(draws.groups[lo:ix] - 1)
        return lambda d, g: sign * (dl[POS, d].sum(axis=1) + gl[g])
    return fit


def age(sign):
    def fit(draws, ix):
        c = eng.Context(draws, ix)
        da = np.log1p(np.array(c.ages(0), dtype=float))
        ga = np.log1p(np.array(c.ages(2)[0], dtype=float))
        return lambda d, g: sign * (da[POS, d].sum(axis=1) + ga[g])
    return fit


def carry(stream, distance, sign):
    def fit(draws, ix):
        prev = draws.digits(stream)[ix - 1].astype(np.int32)
        return lambda d, g: sign * (np.abs(d - prev) == distance).sum(axis=1).astype(float)
    return fit


def markov(alpha):
    def fit(draws, ix):
        rows = draws.main[:ix]
        table = np.zeros((6, 10, 10))
        for j in range(6):
            np.add.at(table[j], (rows[:-1, j], rows[1:, j]), 1)
        logt = np.log((table + alpha) / (table.sum(axis=2, keepdims=True) + 10 * alpha))
        g = draws.groups[:ix] - 1
        gt = np.zeros((5, 5))
        np.add.at(gt, (g[:-1], g[1:]), 1)
        loggt = np.log((gt + alpha) / (gt.sum(axis=1, keepdims=True) + 5 * alpha))
        prev, prev_g = rows[-1], g[-1]
        return lambda d, gg: logt[POS, prev, d].sum(axis=1) + loggt[prev_g, gg]
    return fit


def pairs(alpha):
    def fit(draws, ix):
        rows = draws.main[:ix]
        pmi = []
        for j in range(5):
            c = np.zeros((10, 10))
            np.add.at(c, (rows[:, j], rows[:, j + 1]), 1)
            n = c.sum()
            joint = np.log((c + alpha) / (n + 100 * alpha))
            left = np.log((c.sum(axis=1) + 10 * alpha) / (n + 100 * alpha))
            right = np.log((c.sum(axis=0) + 10 * alpha) / (n + 100 * alpha))
            pmi.append(joint - left[:, None] - right[None, :])
        pmi = np.stack(pmi)
        return lambda d, g: pmi[np.arange(5), d[:, :5], d[:, 1:]].sum(axis=1)
    return fit


FEATURE_KEYS = ('sum', 'odd-count', 'high-count', 'zero-count', 'prime-count', 'distinct-count',
                'max-repeat', 'adj-equal', 'adj-one', 'sum-mod10', 'front-minus-back', 'adj-difference-max')


def _metric(key):
    return next(m for m in eng.static_metrics() if m.key == (key,))


def features(window, strength):
    """특징값별 '실제 빈도 / 이론 확률'의 로그를 더한다. strength가 클수록 이론 쪽으로 강하게 당긴다."""
    def fit(draws, ix):
        lo = _since(ix, window)
        rows, g1 = draws.main[lo:ix], draws.groups[lo:ix]
        tables = []
        for key in FEATURE_KEYS:
            metric = _metric(key)
            values, counts = eng.static_counts(metric.key)
            base = counts / counts.sum()
            low = int(values.min())
            seen = np.bincount(metric.values(rows, g1, ()).astype(np.int64) - low, minlength=len(values))
            k = strength * len(values)
            tables.append((metric, low, np.log((seen + k * base) / (len(rows) + k) / base)))

        def score(d, g):
            out = np.zeros(len(d))
            for metric, low, lift in tables:
                out += lift[metric.values(d, g + 1, ()).astype(np.int64) - low]
            return out
        return score
    return fit


def analog(k):
    def fit(draws, ix):
        rows = draws.main[:ix]
        prev = rows[-1]
        sim = (rows[:-2] == prev).sum(axis=1) + 1e-3 * np.arange(len(rows) - 2)   # 같으면 최근 우선
        nearest = np.argsort(-sim)[:k] + 1                                           # 닮은 회차의 다음 회차
        dl, gl = _digit_log(rows[nearest]), _group_log(draws.groups[nearest] - 1)
        return lambda d, g: dl[POS, d].sum(axis=1) + gl[g]
    return fit


def _random_fit(draws, ix):
    rng = np.random.default_rng(int(draws.rounds[ix - 1]) * 7 + 1)
    salt = rng.integers(1, 2 ** 31)
    return lambda d, g: np.sin(((d @ eng.POWERS) + g * eng.N_NUMBERS + salt) * 12.9898) % 1.0


# ---------------------------------------------------------------- 로지스틱

NEGATIVES = 30


def _phi(d, g, ctx):
    """후보 특징 행렬. ctx = (직전 1등, 직전 보너스, 자리별 숫자 미출현, 조 미출현)."""
    prev, prev_b, ages, gages = ctx
    n = len(d)
    onehot = (d[:, :, None] == np.arange(10)).reshape(n, 60)
    group = g[:, None] == np.arange(5)
    a = ages[POS, d]
    age_bins = np.stack([(a == 0).sum(1), ((a >= 1) & (a <= 2)).sum(1), ((a >= 3) & (a <= 6)).sum(1),
                         ((a >= 7) & (a <= 14)).sum(1), (a >= 15).sum(1)], axis=1)
    s = d.sum(1)
    diff = np.abs(np.diff(d, axis=1))
    sorted_d = np.sort(d, axis=1)
    distinct = 1 + (np.diff(sorted_d, axis=1) != 0).sum(1)
    extra = np.stack([
        (d == prev).sum(1), (d == prev_b).sum(1), (np.abs(d - prev) == 1).sum(1),
        np.log1p(gages[g]), (s - 27) / 7.0, ((s - 27) / 7.0) ** 2, (d % 2).sum(1) - 3, (d >= 5).sum(1) - 3,
        distinct - 4.6, (diff == 0).sum(1), (diff == 1).sum(1)], axis=1)
    return np.hstack((onehot, group, age_bins, extra)).astype(np.float64)


def _context(draws, t):
    c = eng.Context(draws, t)
    return (draws.main[t - 1].astype(np.int32), draws.bonus[t - 1].astype(np.int32),
            np.array(c.ages(0)), np.array(c.ages(2)[0]))


class _TrainRows:
    """회차별 학습 행(실제 1등 1개 + 무작위 30개)을 한 번만 만든다. t번째 행은 t 이전 자료만 쓴다."""

    def __init__(self, draws):
        self.draws, self.rows = draws, {}

    def get(self, t):
        if t not in self.rows:
            rng = np.random.default_rng(t * 1_000_003 + 17)
            ids = np.append(rng.integers(0, eng.N_CANDIDATES, NEGATIVES),
                            (int(self.draws.groups[t]) - 1) * eng.N_NUMBERS + int(self.draws.main[t] @ eng.POWERS))
            d, g = eng.split_ids(ids)
            y = np.zeros(len(ids))
            y[-1] = 1
            self.rows[t] = (_phi(d, g, _context(self.draws, t)), y)
        return self.rows[t]


_TRAIN = {}


def logistic(l2, window=None):
    def fit(draws, ix):
        cache = _TRAIN.setdefault(id(draws), _TrainRows(draws))
        start = max(10, ix - window) if window else 10
        parts = [cache.get(t) for t in range(start, ix)]
        x = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        mu, sd = x.mean(0), x.std(0) + 1e-9
        z = np.hstack(((x - mu) / sd, np.ones((len(x), 1))))
        w = np.zeros(z.shape[1])
        reg = np.full(z.shape[1], l2)
        reg[-1] = 0
        for _ in range(25):   # 뉴턴법
            p = 1 / (1 + np.exp(-(z @ w)))
            grad = z.T @ (p - y) + reg * w
            hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(reg + 1e-9)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        ctx = _context(draws, ix)
        coef = w[:-1] / sd
        return lambda d, g: _phi(d, g, ctx) @ coef
    return fit


# ---------------------------------------------------------------- 목록

FORMULAS = (
    Formula('hot_all', '빈도·전체', hot(None)),
    Formula('hot_100', '빈도·최근100회', hot(100)),
    Formula('hot_50', '빈도·최근50회', hot(50)),
    Formula('hot_20', '빈도·최근20회', hot(20)),
    Formula('hot_bonus_100', '빈도·1등+보너스 최근100회', hot(100, bonus=True)),
    Formula('cold_100', '냉각·최근100회', hot(100, sign=-1.0)),
    Formula('cold_50', '냉각·최근50회', hot(50, sign=-1.0)),
    Formula('cold_20', '냉각·최근20회', hot(20, sign=-1.0)),
    Formula('overdue', '미출현 오래된 쪽', age(1.0)),
    Formula('recent', '최근 나온 쪽', age(-1.0)),
    Formula('carry_main', '직전 1등 같은 자리 반복', carry(eng.MAIN, 0, 1.0)),
    Formula('avoid_main', '직전 1등 같은 자리 피하기', carry(eng.MAIN, 0, -1.0)),
    Formula('neighbor_main', '직전 1등 ±1 이웃', carry(eng.MAIN, 1, 1.0)),
    Formula('carry_bonus', '직전 보너스 같은 자리 반복', carry(eng.BONUS, 0, 1.0)),
    Formula('markov_1', '마르코프 전이·약한 평활', markov(1.0)),
    Formula('markov_5', '마르코프 전이·강한 평활', markov(5.0)),
    Formula('pairs_1', '이웃 자리쌍 PMI', pairs(1.0)),
    Formula('pairs_5', '이웃 자리쌍 PMI·강한 평활', pairs(5.0)),
    Formula('feat_all', '특징 빈도·전체', features(None, 1.0)),
    Formula('feat_100', '특징 빈도·최근100회', features(100, 1.0)),
    Formula('feat_all_strong', '특징 빈도·전체·강한 수축', features(None, 5.0)),
    Formula('analog_10', '유사회차 10개', analog(10)),
    Formula('analog_30', '유사회차 30개', analog(30)),
    Formula('logit_10', '로지스틱·규제 10', logistic(10.0)),
    Formula('logit_100', '로지스틱·규제 100', logistic(100.0)),
    Formula('logit_100_w150', '로지스틱·규제 100·최근150회', logistic(100.0, 150)),
    Formula('random', '무작위 기준', _random_fit),
)
BY_KEY = {f.key: f for f in FORMULAS}


def zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def full_scores(formula, draws, ix=None):
    """ix 시점 채점 함수로 500만 후보 전체 점수 (조 우선 순서)."""
    ix = len(draws.rounds) if ix is None else ix
    score = formula.fit(draws, ix)
    grid = eng.number_grid().astype(np.int32)
    out = np.empty(eng.N_CANDIDATES)
    for g in range(5):
        for lo in range(0, eng.N_NUMBERS, 250_000):
            d = grid[lo:lo + 250_000]
            out[g * eng.N_NUMBERS + lo: g * eng.N_NUMBERS + lo + len(d)] = score(d, np.full(len(d), g))
    return out
