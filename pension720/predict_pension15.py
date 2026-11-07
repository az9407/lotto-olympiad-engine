#!/usr/bin/env python3
"""연금복권720+ 다음 회차 후보 15개 엔진.

흐름
1. 엑셀 당첨번호 시트를 읽는다.
2. 1등·보너스 각각 규칙 후보 목록을 만든다.
3. 최근 40회를 한 회차씩, 그 회차 이전 자료만으로 학습해 16개 설정을 비교한다.
4. 고른 설정으로 목표 회차 규칙을 학습하고 500만 후보를 전수 채점한다.
5. 다양성 조건을 지키며 점수순으로 15개를 고르고 결과를 저장한다.

Python 3.10+, numpy>=2.0, openpyxl 필요.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from itertools import combinations
from math import fsum, log, sqrt
from pathlib import Path
from typing import Callable

import numpy as np

N_NUMBERS = 1_000_000                # 000000~999999
N_GROUPS = 5                         # 1~5조
N_CANDIDATES = N_GROUPS * N_NUMBERS  # 후보 id = (조-1)*1,000,000 + 번호
POWERS = np.array([100_000, 10_000, 1_000, 100, 10, 1], dtype=np.int32)
HEADERS = ('회차', '추첨일', '1등 조', '1등 번호', '보너스 번호')
MAIN, BONUS = 0, 1
STREAM_NAMES = ('1등', '보너스')

MIN_COUNTS = np.array([30, 10, 4])   # 표본 하한: 전체 / 최근 100회 / 최근 40회
MIN_RATE = 0.55                      # 모든 구간 적중률 하한
PLACE_WEIGHTS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)  # 끝자리에 가까울수록 등수에 영향이 커서 배율을 높게
ROLLING_FOLDS = 40
RANK_SAMPLE = 200_000
TIE_SCALE = 1e-6
PICKS = 15


# ---------------------------------------------------------------- 입력

@dataclass(frozen=True)
class Draws:
    rounds: np.ndarray   # (n,)
    dates: tuple         # (n,) date
    groups: np.ndarray   # (n,) 1~5
    main: np.ndarray     # (n, 6) 1등 숫자
    bonus: np.ndarray    # (n, 6) 보너스 숫자

    def digits(self, stream):
        return self.main if stream == MAIN else self.bonus

    def truncated(self, n):
        return Draws(self.rounds[:n], self.dates[:n], self.groups[:n], self.main[:n], self.bonus[:n])


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip().rstrip('.').replace('.', '-').replace('/', '-').replace(' ', '')
    return date.fromisoformat('-'.join(part.zfill(2) for part in text.split('-')))


def _as_int(value, suffix):
    text = str(value).strip().removesuffix(suffix)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not text.isdigit():
        raise ValueError(f'숫자가 아닙니다: {value!r}')
    return int(text)


def _as_number(value):
    """엑셀이 숫자로 바꿔 앞자리 0이 사라진 경우도 6자리로 복원한다."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if not text.isdigit() or len(text) > 6:
        raise ValueError(f'번호는 6자리 이하 숫자여야 합니다: {value!r}')
    return text.zfill(6)


def _sheet_rows(path: Path):
    """엑셀 당첨번호 시트 또는 같은 열을 가진 CSV의 행들."""
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            yield from csv.reader(f)
        return
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    if '당첨번호' not in book.sheetnames:
        raise ValueError('엑셀에 당첨번호 시트가 없습니다.')
    yield from book['당첨번호'].iter_rows(values_only=True)
    book.close()


def load_draws(path: Path) -> Draws:
    rows = _sheet_rows(path)
    header = [str(v).strip() if v is not None else '' for v in next(rows)]
    if any(name not in header for name in HEADERS):
        raise ValueError('당첨번호 시트에 필수 열이 없습니다: ' + ', '.join(HEADERS))
    columns = [header.index(name) for name in HEADERS]
    records = []
    for line, row in enumerate(rows, 2):
        values = [row[c] if c < len(row) else None for c in columns]
        if all(v is None or str(v).strip() == '' for v in values):
            continue
        try:
            group = _as_int(values[2], '조')
            if not 1 <= group <= 5:
                raise ValueError('조는 1~5여야 합니다')
            records.append((_as_int(values[0], '회'), _as_date(values[1]), group,
                            _as_number(values[3]), _as_number(values[4])))
        except (ValueError, TypeError) as error:
            raise ValueError(f'당첨번호 시트 {line}행: {error}') from error
    records.sort(key=lambda r: r[0])
    if len(records) < ROLLING_FOLDS + 30:
        raise ValueError(f'최소 {ROLLING_FOLDS + 30}회가 필요합니다.')
    if any(b[0] != a[0] + 1 or b[1] <= a[1] for a, b in zip(records, records[1:])):
        raise ValueError('회차는 중복·누락 없이 이어지고 날짜는 증가해야 합니다.')
    rounds, dates, groups, main, bonus = zip(*records)

    def to_digits(strings):
        return np.array([[int(c) for c in s] for s in strings], dtype=np.int8)

    return Draws(np.array(rounds, dtype=np.int32), dates, np.array(groups, dtype=np.int8),
                 to_digits(main), to_digits(bonus))


# ---------------------------------------------------------------- 시점별 정보

DATE_VARIABLES = ('회차', '연', '월', '일', '날짜 숫자합', '날짜 디지털루트')


def digital_root(value):
    return np.where(value == 0, 0, 1 + (value - 1) % 9)


class Context:
    """index번째 행을 맞히는 시점. 그 행 이전 자료와 그 행의 회차·날짜만 쓴다."""

    def __init__(self, draws: Draws, index: int):
        self.draws = draws
        self.index = index
        if index < len(draws.rounds):
            self.round, self.date = int(draws.rounds[index]), draws.dates[index]
        else:  # 목표 회차: 마지막 회차 + 1, 마지막 추첨일 + 7일
            self.round, self.date = int(draws.rounds[-1]) + 1, draws.dates[-1] + timedelta(days=7)
        date_sum = sum(int(c) for c in self.date.isoformat() if c.isdigit())
        self.date_values = (self.round, self.date.year, self.date.month, self.date.day,
                            date_sum, int(digital_root(date_sum)))
        self._ages = {}

    def lag(self, stream, k):
        """k회 전 숫자 6개. 없으면 None."""
        if self.index < k:
            return None
        return tuple(map(int, self.draws.digits(stream)[self.index - k]))

    def lag_group(self, k):
        return None if self.index < k else int(self.draws.groups[self.index - k])

    def ages(self, stream):
        """자리별·숫자별 미출현 회차 수. 직전 회차에 나왔으면 0. stream 2는 조."""
        if self.index == 0:
            return None
        if stream not in self._ages:
            rounds = self.draws.rounds[:self.index]
            if stream == 2:
                history, domain = self.draws.groups[:self.index, None], range(1, 6)
            else:
                history, domain = self.draws.digits(stream)[:self.index], range(10)
            never = int(rounds[0]) - 1
            self._ages[stream] = tuple(
                tuple(self.round - 1 - (int(rounds[hit[-1]]) if len(hit) else never)
                      for value in domain for hit in (np.flatnonzero(column == value),))
                for column in history.T)
        return self._ages[stream]


# ---------------------------------------------------------------- 사건과 지표

@dataclass(frozen=True)
class Predicate:
    op: str        # eq, le, ge, between, in
    values: tuple

    def test(self, x):
        v = self.values
        if self.op == 'eq':
            return x == v[0]
        if self.op == 'le':
            return x <= v[0]
        if self.op == 'ge':
            return x >= v[0]
        if self.op == 'between':
            return (x >= v[0]) & (x <= v[1])
        mask = x == v[0]
        for value in v[1:]:
            mask = mask | (x == value)
        return mask

    def formula(self, label):
        v = self.values
        if self.op == 'between':
            return f'{v[0]} ≤ {label} ≤ {v[1]}'
        if self.op == 'in':
            return f"{label} ∈ {{{','.join(map(str, v))}}}"
        return f"{label} {dict(eq='=', le='≤', ge='≥')[self.op]} {v[0]}"


@dataclass(frozen=True, eq=False)
class Metric:
    """후보 6자리(와 조)에 대한 값. state가 있으면 시점마다 값이 달라지는 동적 지표."""
    key: tuple
    label: str                         # {p}는 1등 d, 보너스 b로 바뀐다
    variable: str                      # 같은 변수 규칙은 최대 2개
    evaluate: Callable                 # (digits, groups, state) -> values
    state: Callable | None = None      # context -> state, 쓸 수 없으면 None
    describe: Callable | None = None
    counts: Callable | None = None     # state -> (값, 500만 후보 중 개수)
    place: int | None = None
    group_sensitive: bool = False

    def get_state(self, context):
        return () if self.state is None else self.state(context)

    def values(self, digits, groups, state):
        return np.broadcast_to(self.evaluate(digits, groups, state), (len(digits),))

    def distribution(self, state):
        return static_counts(self.key) if self.counts is None else self.counts(state)

    def probability(self, predicate, state):
        values, counts = self.distribution(state)
        return float(counts[predicate.test(values)].sum()) / int(counts.sum())


def _range_counts(values):
    values = np.asarray(values)
    lo, hi = int(values.min()), int(values.max())
    return np.arange(lo, hi + 1), np.bincount(values.astype(np.int64) - lo, minlength=hi - lo + 1)


def _merge_counts(values, counts):
    total = Counter()
    for value, count in zip(values, counts):
        total[int(value)] += int(count)
    return np.array(tuple(total)), np.array(tuple(total.values()))


@lru_cache(maxsize=1)
def number_grid():
    """000000~999999의 자리 숫자 (1,000,000 x 6)."""
    return (np.arange(N_NUMBERS, dtype=np.int32)[:, None] // POWERS % 10).astype(np.int8)


def _digit_sum(d):
    return d.sum(axis=1, dtype=np.int16)


def _abs_diff(d):
    return np.abs(np.diff(d, axis=1))


def _max_repeat(d):
    out = np.zeros(len(d), dtype=np.int8)
    for v in range(10):
        np.maximum(out, (d == v).sum(axis=1, dtype=np.int8), out=out)
    return out


@lru_cache(maxsize=1)
def static_metrics():
    out = []

    def add(name, label, evaluate, place=None, group_sensitive=False):
        out.append(Metric((name,), label, name, evaluate, place=place, group_sensitive=group_sensitive))

    for j in range(6):
        p = f'{{p}}{j + 1}'
        add(f'digit-{j}', p, lambda d, g, s, j=j: d[:, j], j)
        add(f'high-{j}', f'I({p} ≥ 5)', lambda d, g, s, j=j: d[:, j] >= 5, j)
        for m in (2, 3, 4, 5):
            add(f'digit-mod{m}-{j}', f'{p} mod {m}', lambda d, g, s, j=j, m=m: d[:, j] % m, j)
    add('sum', '자리합', lambda d, g, s: _digit_sum(d))
    for m in (3, 4, 5, 7, 9, 10, 11):
        add(f'sum-mod{m}', f'자리합 mod {m}', lambda d, g, s, m=m: _digit_sum(d) % m)
    add('digital-root', '자리합 디지털루트', lambda d, g, s: digital_root(_digit_sum(d)))
    add('odd-count', '홀수 개수', lambda d, g, s: (d % 2).sum(axis=1))
    add('high-count', '5 이상 개수', lambda d, g, s: (d >= 5).sum(axis=1))
    add('zero-count', '0 개수', lambda d, g, s: (d == 0).sum(axis=1))
    add('prime-count', '{2,3,5,7} 개수', lambda d, g, s: np.isin(d, (2, 3, 5, 7)).sum(axis=1))
    add('distinct-count', '서로 다른 숫자 개수',
        lambda d, g, s: 1 + (np.diff(np.sort(d, axis=1), axis=1) != 0).sum(axis=1))
    add('max-repeat', '같은 숫자 최대 반복', lambda d, g, s: _max_repeat(d))
    add('adj-equal', '인접 같은 숫자 쌍', lambda d, g, s: (_abs_diff(d) == 0).sum(axis=1))
    add('adj-one', '인접 ±1 쌍', lambda d, g, s: (_abs_diff(d) == 1).sum(axis=1))
    for j in range(5):
        add(f'adj-difference-{j}', f'{{p}}{j + 2}−{{p}}{j + 1}', lambda d, g, s, j=j: d[:, j + 1] - d[:, j])
        add(f'adj-abs-difference-{j}', f'|{{p}}{j + 2}−{{p}}{j + 1}|',
            lambda d, g, s, j=j: np.abs(d[:, j + 1] - d[:, j]))
    add('adj-difference-sum', '인접 절대차 합', lambda d, g, s: _abs_diff(d).sum(axis=1))
    add('adj-difference-max', '인접 절대차 최댓값', lambda d, g, s: _abs_diff(d).max(axis=1))
    add('front-sum', '앞3자리 합', lambda d, g, s: d[:, :3].sum(axis=1))
    add('back-sum', '뒤3자리 합', lambda d, g, s: d[:, 3:].sum(axis=1))
    add('front-minus-back', '앞3자리 합−뒤3자리 합', lambda d, g, s: d[:, :3].sum(axis=1) - d[:, 3:].sum(axis=1))
    for m in (7, 9, 11, 13):
        weights = np.array([10 ** i % m for i in range(5, -1, -1)], dtype=np.int16)
        add(f'number-mod{m}', f'6자리 정수 mod {m}', lambda d, g, s, m=m, w=weights: (d @ w) % m)
    add('last-two', '끝2자리', lambda d, g, s: d[:, -2] * 10 + d[:, -1])
    add('last-three', '끝3자리', lambda d, g, s: d[:, -3].astype(np.int16) * 100 + d[:, -2] * 10 + d[:, -1])
    add('group', 'g', lambda d, g, s: g, group_sensitive=True)
    add('same-as-group', 'I(g와 같은 숫자 포함)', lambda d, g, s: (d == g[:, None]).any(axis=1),
        group_sensitive=True)
    add('sum-group-mod5', '(자리합+g) mod 5', lambda d, g, s: (_digit_sum(d) + g) % 5, group_sensitive=True)
    return tuple(out)


@lru_cache(maxsize=None)
def static_counts(key):
    """정적 지표 값의 500만 후보 전체 분포."""
    metric = next(m for m in static_metrics() if m.key == key)
    total = Counter()
    for group in (range(1, 6) if metric.group_sensitive else (1,)):
        values, counts = _range_counts(metric.values(number_grid(), np.array([group], dtype=np.int8), ()))
        total.update({int(v): int(c) for v, c in zip(values, counts)})
    scale = 1 if metric.group_sensitive else N_GROUPS
    return np.array(tuple(total)), np.array(tuple(total.values())) * scale


def _uniform_counts(values):
    """자리 하나(또는 조)만 보는 지표: 값마다 같은 수의 후보."""
    labels, counts = _range_counts(np.asarray(values, dtype=np.int16))
    return labels, counts * (N_CANDIDATES // len(values))


@lru_cache(maxsize=4096)
def _carry_counts(reference, distance):
    """직전 번호와 같은 자리 일치(또는 ±1) 개수의 정확 분포."""
    counts = np.array([1], dtype=np.int64)
    for digit in reference:
        hits = 1 if distance == 0 else 1 + int(0 < digit < 9)
        counts = np.convolve(counts, [10 - hits, hits])
    return np.arange(7), counts * N_GROUPS


def _lag_name(stream, k):
    return f'{k}회 전 {STREAM_NAMES[stream]}'


def _at(values, j):
    return None if values is None else values[j]


def _summary(context, stream, k, summarize):
    digits = context.lag(stream, k)
    return None if digits is None else summarize(digits)


def dynamic_metrics():
    """(지표, 사건 묶음 종류) 목록."""
    out = []
    for src in (MAIN, BONUS):
        for k in (1, 2, 3):
            name = _lag_name(src, k)
            for dist in (0, 1):
                key = ('carry-count', dist, src, k)
                out.append((Metric(
                    key, f"{name} 같은 자리 {'일치' if dist == 0 else '±1 이웃'} 개수", str(key),
                    lambda d, g, s, dist=dist: (np.abs(d - np.asarray(s, dtype=np.int8)) == dist).sum(axis=1),
                    lambda c, src=src, k=k: c.lag(src, k),
                    lambda c, s, name=name: f"{name} = {''.join(map(str, s))}",
                    lambda s, dist=dist: _carry_counts(s, dist)), 'carry'))
            for j in range(6):
                prefix = f'{{p}}{j + 1}−{name} {j + 1}자리'
                state = lambda c, src=src, k=k, j=j: _at(c.lag(src, k), j)
                describe = lambda c, s, name=name, j=j: f'{name} {j + 1}자리 = {s}'
                for dist in (0, 1):
                    out.append((Metric(
                        ('carry-digit', dist, src, k, j), f'|{prefix}|', str(('carry-digit', src, k, j)),
                        lambda d, g, s, j=j: np.abs(d[:, j] - s), state, describe,
                        lambda s: _uniform_counts(np.abs(np.arange(10) - s)), j),
                        'match' if dist == 0 else 'neighbor'))
                if k == 1:
                    key = ('lag-difference', src, j)
                    out.append((Metric(key, prefix, str(key), lambda d, g, s, j=j: d[:, j] - s, state, describe,
                                       lambda s: _uniform_counts(np.arange(10) - s), j), 'difference'))
        for j in range(6):
            key = ('digit-age', src, j)
            out.append((Metric(
                key, f"{STREAM_NAMES[src]} 이력의 {{p}}{j + 1} 미출현 주기{'·끝자리' if j == 5 else ''}", str(key),
                lambda d, g, s, j=j: np.take(s, d[:, j]),
                lambda c, src=src, j=j: _at(c.ages(src), j),
                lambda c, s: '숫자별 주기(0~9)=' + ','.join(map(str, s)),
                _uniform_counts, j), 'age'))
        # 직전 회차가 극단값이면 반대쪽으로 돌아오는지 보는 지표
        for base, label, low, high, top, thresholds, summarize, evaluate in (
                ('sum', '자리합', 18, 36, 54, (28, 29, 30, 32), sum, _digit_sum),
                ('odd-count', '홀수 개수', 1, 5, 6, (4, 5, 6),
                 lambda d: sum(x % 2 for x in d), lambda d: (d % 2).sum(axis=1)),
                ('high-count', '5 이상 개수', 1, 5, 6, (4, 5, 6),
                 lambda d: sum(x >= 5 for x in d), lambda d: (d >= 5).sum(axis=1))):
            previous = f'{_lag_name(src, 1)} {label}'

            def extreme(c, src=src, f=summarize, low=low, high=high):
                value = _summary(c, src, 1, f)
                return None if value is None or low < value < high else value

            def reflected(s, key=(base,), low=low, top=top):
                values, counts = static_counts(key)
                return (values, counts) if s <= low else (top - values, counts)

            key = ('reversion', src, base)
            out.append((Metric(
                key, f'({previous}≤{low}이면 현재 {label}, ≥{high}이면 {top}−현재 {label})', str(key),
                lambda d, g, s, f=evaluate, low=low, top=top: f(d) if s <= low else top - f(d),
                extreme, lambda c, s, previous=previous: f'{previous} = {s}', reflected),
                ('reversion', thresholds)))
    out.append((Metric(('group-age',), 'g 미출현 주기', 'group-age', lambda d, g, s: np.take(s, g - 1),
                       lambda c: _at(c.ages(2), 0), lambda c, s: '조별 주기(1~5)=' + ','.join(map(str, s)),
                       _uniform_counts, group_sensitive=True), 'age'))
    # 날짜·회차와의 나머지 관계
    for ref, ref_name in enumerate(DATE_VARIABLES):
        for j in (*range(6), None):
            for m in ((2, 3, 4, 5) if j is not None else (2, 3, 4, 5, 7, 9, 10, 11)):
                key = ('date', ref, j, m)
                base = ('sum',) if j is None else (f'digit-{j}',)
                if j is None:
                    evaluate = lambda d, g, s, m=m: (_digit_sum(d) - s) % m
                else:
                    evaluate = lambda d, g, s, j=j, m=m: (d[:, j] - s) % m
                out.append((Metric(
                    key, f"({'자리합' if j is None else f'{{p}}{j + 1}'}−{ref_name}) mod {m}", str(key), evaluate,
                    lambda c, ref=ref, m=m: c.date_values[ref] % m,
                    lambda c, s, ref=ref, name=ref_name, m=m: f'{name} = {c.date_values[ref]}, 나머지 mod {m} = {s}',
                    lambda s, base=base, m=m: _merge_counts((static_counts(base)[0] - s) % m, static_counts(base)[1]),
                    j), ('mod', m)))
    return tuple(out)


def _range_predicate(low, high, lo=None, hi=None):
    if low == high:
        return Predicate('eq', (low,))
    if low == lo:
        return Predicate('le', (high,))
    if high == hi:
        return Predicate('ge', (low,))
    return Predicate('between', (low, high))


def _mod_predicates(m, all_subsets):
    events = {Predicate('eq', (i,)) for i in range(m)}
    for size in range(2, (m + 1) // 2 + 1):
        if all_subsets and m <= 7:
            subsets = combinations(range(m), size)
        else:  # 큰 나머지는 등차(간격 1·2) 묶음만
            subsets = {tuple(sorted((start + i * step) % m for i in range(size)))
                       for start in range(m) for step in (1, 2)
                       if len({(start + i * step) % m for i in range(size)}) == size}
        events.update(Predicate('in', tuple(s)) for s in subsets)
    return tuple(sorted(events, key=lambda e: (e.op, e.values)))


def static_predicates(metric):
    values, counts = static_counts(metric.key)
    if 'mod' in metric.variable:
        return _mod_predicates(int(values.max()) + 1, all_subsets=True)
    if metric.variable == 'group':
        return (tuple(Predicate('eq', (g,)) for g in range(1, 6))
                + tuple(Predicate('in', pair) for pair in combinations(range(1, 6), 2)))
    lo, hi = int(values.min()), int(values.max())
    if (hi <= 14 and lo >= 0) or hi - lo <= 20:
        return tuple(_range_predicate(a, b, lo, hi) for a in range(lo, hi + 1) for b in range(a, hi + 1))
    # 값 범위가 넓으면 전체 후보의 35~50%를 덮는 구간만
    cumulative = np.r_[0, np.cumsum(counts[np.argsort(values)])]
    total = int(cumulative[-1])
    starts = range(lo, hi + 1, 25 if hi > 100 else 5) if hi >= 100 else range(lo, hi + 1)
    events = set()
    for a in starts:
        for mass in (.35, .40, .45, .50):
            edge = cumulative[a - lo] + mass * total
            b = int(np.searchsorted(cumulative, edge, side='right')) - 2 + lo
            if b >= a and edge <= total:
                events.add(_range_predicate(a, b, lo, hi))
    for mass in (.35, .40, .45, .50):
        index = int(np.searchsorted(cumulative, (1 - mass) * total, side='left'))
        if index <= hi - lo:
            events.add(Predicate('ge', (index + lo,)))
    return tuple(sorted(events, key=lambda e: (e.op, e.values)))


def dynamic_predicates(kind):
    if kind == 'match':
        return (Predicate('eq', (0,)),)
    if kind == 'neighbor':
        return (Predicate('eq', (1,)),)
    if kind == 'carry':
        return tuple(Predicate(op, (v,)) for op, domain in
                     (('eq', range(7)), ('ge', range(1, 7)), ('le', range(1, 6))) for v in domain)
    if kind == 'difference':
        return tuple(_range_predicate(a, b, -9, 9) for a in range(-9, 10) for b in range(a, 10)
                     if b - a <= 5 or a == -9 or b == 9)
    if kind == 'age':
        edges = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)
        return (tuple(Predicate('le', (v,)) for v in edges) + tuple(Predicate('ge', (v,)) for v in edges[1:])
                + tuple(Predicate('between', (a, b)) for a, b in zip(edges, edges[2:])))
    if kind[0] == 'reversion':
        return tuple(Predicate('ge', (t,)) for t in kind[1])
    return _mod_predicates(kind[1], all_subsets=False)


# ---------------------------------------------------------------- 규칙 후보 목록

@dataclass(frozen=True)
class Antecedent:
    label: str
    predicate: Predicate
    value: Callable     # context -> 값 또는 None
    active: np.ndarray  # 과거 각 행에서 성립 여부

    @property
    def formula(self):
        return self.predicate.formula(self.label)

    def holds(self, context):
        value = self.value(context)
        return value is not None and bool(self.predicate.test(value))


@dataclass(frozen=True, eq=False)
class Rule:
    """B: 항상(동적 지표는 상태가 있을 때) 적용. A: 선행조건이 성립할 때만 적용."""
    metric: Metric
    predicate: Predicate
    stream: int
    hist_hit: np.ndarray     # 과거 각 행에서 결론 성립 여부
    hist_active: np.ndarray  # 과거 각 행에서 규칙 적용 여부
    antecedent: Antecedent | None = None

    @property
    def kind(self):
        return 'B' if self.antecedent is None else 'A'

    @property
    def formula(self):
        consequence = self.predicate.formula(self.metric.label.replace('{p}', 'db'[self.stream]))
        return consequence if self.antecedent is None else f'{self.antecedent.formula} ⇒ {consequence}'

    @property
    def multiplier(self):
        return 1.0 if self.metric.place is None else PLACE_WEIGHTS[self.metric.place]

    def active(self, context):
        if self.metric.get_state(context) is None:
            return False
        return self.antecedent is None or self.antecedent.holds(context)

    def evidence(self, context):
        parts = []
        if self.antecedent is not None:
            parts.append(f'{self.antecedent.label} = {self.antecedent.value(context)}')
        if self.metric.describe is not None:
            parts.append(self.metric.describe(context, self.metric.get_state(context)))
        return ' · '.join(parts) if parts else '상시 적용'

    def baseline(self, context):
        return self.metric.probability(self.predicate, self.metric.get_state(context))

    def mask_key(self, context):
        return self.metric.key, self.predicate, self.metric.get_state(context)


def antecedents(contexts):
    out = []

    def add(label, value, predicates):
        states = [value(c) for c in contexts]
        available = np.array([s is not None for s in states])
        numbers = np.array([0 if s is None else s for s in states])
        for predicate in predicates:
            active = available & predicate.test(numbers)
            if int(active.sum()) >= MIN_COUNTS[0]:
                out.append(Antecedent(label, predicate, value, active))

    def low_high(lows, highs):
        return tuple(Predicate('le', (x,)) for x in lows) + tuple(Predicate('ge', (x,)) for x in highs)

    for src in (MAIN, BONUS):
        for k in (1, 2, 3):
            prefix = _lag_name(src, k)
            for label, f, events in (('자리합', sum, low_high((18, 22, 25), (29, 32, 36))),
                                     ('홀수 개수', lambda d: sum(x % 2 for x in d), low_high((1, 2), (4, 5))),
                                     ('5 이상 개수', lambda d: sum(x >= 5 for x in d), low_high((1, 2), (4, 5)))):
                add(f'{prefix} {label}', lambda c, s=src, k=k, f=f: _summary(c, s, k, f), events)
            for j in range(6):
                add(f'{prefix} {j + 1}자리', lambda c, s=src, k=k, j=j: _at(c.lag(s, k), j), low_high((2, 4), (5, 7)))
                add(f'{prefix} {j + 1}자리 mod 2', lambda c, s=src, k=k, j=j: _summary(c, s, k, lambda d: d[j] % 2),
                    (Predicate('eq', (0,)), Predicate('eq', (1,))))
    for k in (1, 2, 3):
        add(f'{k}회 전 g', lambda c, k=k: c.lag_group(k), tuple(Predicate('eq', (x,)) for x in range(1, 6)))
    for ref, label in enumerate(DATE_VARIABLES):
        for m in (2, 3, 4, 5):
            add(f'{label} mod {m}', lambda c, ref=ref, m=m: c.date_values[ref] % m,
                tuple(Predicate('eq', (x,)) for x in range(m)))
    return tuple(out)


def _prefix(flags):
    out = np.zeros((len(flags), flags.shape[1] + 1), dtype=np.int32)
    np.cumsum(flags, axis=1, dtype=np.int32, out=out[:, 1:])
    return out


def _windows(n, folds=ROLLING_FOLDS):
    """롤링 학습 시점 n-folds..n과 각 시점의 전체·최근100·최근40 시작 행."""
    ends = np.arange(max(0, n - folds), n + 1)
    starts = np.column_stack((np.zeros_like(ends), np.maximum(0, ends - 100), np.maximum(0, ends - 40)))
    return ends, starts


def _window_counts(flags, ends, starts):
    prefix = _prefix(flags)
    return prefix[:, ends, None] - prefix[:, starts]


def build_catalog(draws, stream, folds=ROLLING_FOLDS):
    """롤링 학습 시점 중 한 번이라도 채택 기준을 넘을 수 있는 규칙만 모은다.

    시점별 실제 채택은 RuleFold가 그 시점 이전 자료로 다시 판정하므로,
    이 목록은 넉넉한 상위집합일 뿐 이후 정보가 판정에 섞이지 않는다.
    """
    n = len(draws.rounds)
    contexts = [Context(draws, i) for i in range(n)]
    target = Context(draws, n)
    digits, groups = draws.digits(stream), draws.groups
    ends, starts = _windows(n, folds)
    ante = antecedents(contexts)
    ante_active = np.stack([a.active for a in ante])
    ante_now = np.column_stack((ante_active, [a.holds(target) for a in ante]))[:, ends]
    ante_counts = _window_counts(ante_active, ends, starts)
    ante_ok = ante_now & np.all(ante_counts >= MIN_COUNTS, axis=2)
    always = np.ones(n, dtype=bool)
    always_counts = _window_counts(always[None], ends, starts)
    always_ok = np.all(always_counts >= MIN_COUNTS, axis=2)
    catalog = []
    for metric in static_metrics():
        if stream == BONUS and metric.group_sensitive:
            continue  # 보너스는 조와 무관하다
        values = metric.values(digits, groups, ())
        for predicate in static_predicates(metric):
            p0 = metric.probability(predicate, ())
            if not 0 < p0 <= .5:
                continue
            hits = predicate.test(values)
            if hits.sum() < MIN_RATE * MIN_COUNTS[0]:
                continue
            hit_counts = _window_counts(hits[None], ends, starts)
            if np.any(always_ok & np.all(100 * hit_counts >= 55 * always_counts, axis=2)):
                catalog.append(Rule(metric, predicate, stream, hits, always))
            if p0 >= .4:
                cond = _window_counts(ante_active & hits, ends, starts)
                ok = ante_ok & np.all(100 * cond >= 55 * ante_counts, axis=2)
                catalog.extend(Rule(metric, predicate, stream, hits, ante[i].active, ante[i])
                               for i in np.flatnonzero(ok.any(axis=1)))
    for metric, kind in dynamic_metrics():
        if stream == BONUS and metric.group_sensitive:
            continue
        predicates = dynamic_predicates(kind)
        values, eligible = _dynamic_history(metric, predicates, contexts, digits, groups)
        counts = _window_counts(eligible, ends, starts)
        ok = np.all(counts >= MIN_COUNTS, axis=2)
        rows = np.flatnonzero(ok.any(axis=1))
        if not len(rows):
            continue
        hits = np.stack([predicates[i].test(values) for i in rows])
        hit_counts = _window_counts(hits & eligible[rows], ends, starts)
        ok = ok[rows] & np.all(100 * hit_counts >= 55 * counts[rows], axis=2)
        catalog.extend(Rule(metric, predicates[rows[r]], stream, hits[r], eligible[rows[r]])
                       for r in np.flatnonzero(ok.any(axis=1)))
    return catalog


def _dynamic_history(metric, predicates, contexts, digits, groups):
    """과거 각 행의 지표 값과, 그 행 시점에 사건 기저율이 (0, 50%]였는지."""
    values = np.zeros(len(digits), dtype=np.int32)
    eligible = np.zeros((len(predicates), len(digits)), dtype=bool)
    by_state = {}
    for i, c in enumerate(contexts):
        state = metric.get_state(c)
        if state is not None:
            by_state.setdefault(state, []).append(i)
    for state, rows in by_state.items():
        values[rows] = metric.values(digits[rows], groups[rows], state)
        support, counts = metric.distribution(state)
        total = int(counts.sum())
        assert total == N_CANDIDATES, (metric.key, total)
        for r, predicate in enumerate(predicates):
            if 0 < 2 * int(counts[predicate.test(support)].sum()) <= total:
                eligible[r, rows] = True
    return values, eligible


# ---------------------------------------------------------------- 500만 후보 공간

class Space:
    """규칙별 500만 후보 통과 여부를 비트로 압축해 보관한다."""

    def __init__(self, max_masks=2_000):
        self.max_masks = max_masks
        self.masks = OrderedDict()   # mask_key -> (1 또는 5, 125000) uint8

    def mask(self, rule, context, baseline):
        key = rule.mask_key(context)
        if key in self.masks:
            self.masks.move_to_end(key)
            return self.masks[key]
        state = rule.metric.get_state(context)
        grid = number_grid()
        rows = []
        for group in (range(1, 6) if rule.metric.group_sensitive else (1,)):
            flags = rule.predicate.test(rule.metric.values(grid, np.array([group], dtype=np.int8), state))
            rows.append(np.packbits(np.asarray(flags, dtype=bool), bitorder='little'))
        packed = np.stack(rows)
        passed = int(np.bitwise_count(packed).sum()) * (N_GROUPS // len(packed))
        if abs(passed / N_CANDIDATES - baseline) > 1e-12:
            raise AssertionError(f'{rule.formula}: 전수 통과율과 기저율 불일치')
        self.masks[key] = packed
        if len(self.masks) > self.max_masks:
            self.masks.popitem(last=False)
        return packed

    def joint(self, a, b):
        """두 압축 마스크를 동시에 통과하는 후보 수."""
        both = int(np.bitwise_count(a & b).sum())
        return both * N_GROUPS if len(a) == len(b) == 1 else both


# ---------------------------------------------------------------- 시점별 규칙 학습

@dataclass(frozen=True)
class Setting:
    """채점 방법. 앞 네 값은 기본 16개 조합, 뒤 세 값은 주간 학습이 바꿔 보는 값."""
    time_weights: tuple   # 최근 40회 / 그 앞 60회 / 나머지
    threshold: float
    shrink: int
    limit: int
    var_cap: int = 2              # 같은 변수 규칙 최대 개수
    place_boost: bool = True      # 끝자리 쪽 자리 배율 사용
    mix: tuple = (0.8, 0.2)       # 종합점수의 1등 z, 보너스 z 비율

    def label(self):
        weights = '/'.join(f'{v:g}' for v in self.time_weights)
        text = f'시간 가중치 {weights} · 적중률 문턱 {self.threshold:.0%} · 수축 {self.shrink} · 규칙 상한 {self.limit}개'
        if self.var_cap != 2:
            text += f' · 같은 변수 {self.var_cap}개'
        if not self.place_boost:
            text += ' · 자리 배율 끔'
        if self.mix != (0.8, 0.2):
            text += f' · 1등:보너스 {self.mix[0]:g}:{self.mix[1]:g}'
        return text

    def to_text(self):
        return '\n'.join(f'{k}={v}' for k, v in self.__dict__.items()) + '\n'

    @classmethod
    def from_text(cls, text):
        import ast
        fields = cls.__dataclass_fields__
        values = dict(line.split('=', 1) for line in text.strip().splitlines() if '=' in line)
        return cls(**{k: ast.literal_eval(v) for k, v in values.items() if k in fields})


SETTINGS = tuple(Setting(*args) for args in itertools.product(
    ((3.0, 2.0, 1.0), (2.0, 1.5, 1.0)), (0.55, 0.60), (20, 10), (40, 30)))


def _weighted(counts, weights):
    """(전체, 최근100, 최근40) 개수에 구간별 시간 가중치를 준 합."""
    total, last100, last40 = counts
    return last40 * weights[0] + (last100 - last40) * weights[1] + (total - last100) * weights[2]


def _logit(p):
    return log(p / (1 - p))


@dataclass(frozen=True, eq=False)
class Candidate:
    rule: Rule
    baseline: float
    counts: tuple   # 적용 표본: 전체, 최근100, 최근40
    hits: tuple     # 적중 수: 같은 순서
    mask: np.ndarray


@dataclass(frozen=True, eq=False)
class Fitted:
    rule: Rule
    baseline: float
    counts: tuple   # 전체, 가중, 최근100, 최근40
    rates: tuple    # 같은 순서
    weight: float
    adjusted: float  # weight x 자리 배율
    mask: np.ndarray


class HistoryIndex:
    def __init__(self, catalog):
        active = np.stack([r.hist_active for r in catalog])
        self.active = _prefix(active)
        self.hits = _prefix(np.stack([r.hist_hit for r in catalog]) & active)


class RuleFold:
    """context 시점(그 행 이전 자료)에서 채택 기준을 넘는 규칙들. 설정마다 fit으로 모델을 만든다."""

    def __init__(self, catalog, history, context, space):
        self.context, self.space = context, space
        end = context.index
        edges = [0, max(0, end - 100), max(0, end - 40)]
        counts = history.active[:, [end]] - history.active[:, edges]
        hits = history.hits[:, [end]] - history.hits[:, edges]
        passed = np.all((counts >= MIN_COUNTS) & (100 * hits >= 55 * counts), axis=1)
        self.candidates = []
        for i in np.flatnonzero(passed):
            rule = catalog[i]
            if not rule.active(context):
                continue
            p0 = rule.baseline(context)
            if not (0 < p0 <= .5 and (rule.kind == 'B' or p0 >= .4)):
                continue
            self.candidates.append(Candidate(rule, p0, tuple(map(int, counts[i])), tuple(map(int, hits[i])),
                                             space.mask(rule, context, p0)))
        self._space_pairs = {}
        self._history_pairs = {}

    def space_pair(self, a, b):
        """전수 공간에서 두 규칙의 (동시 통과 수, 공분산 분자, 분산곱 분자)."""
        key = (id(a.rule), id(b.rule)) if id(a.rule) < id(b.rule) else (id(b.rule), id(a.rule))
        if key not in self._space_pairs:
            n = N_CANDIDATES
            joint = self.space.joint(a.mask, b.mask)
            ca, cb = round(a.baseline * n), round(b.baseline * n)
            self._space_pairs[key] = (joint, joint * n - ca * cb, ca * cb * (n - ca) * (n - cb), ca * cb)
        return self._space_pairs[key]

    def history_pair(self, a, b, weights):
        """두 규칙이 함께 적용된 과거 회차에서 (공통 표본, a 적중, b 적중, 동시 적중) 가중 개수."""
        key = (id(a.rule), id(b.rule))
        if key not in self._history_pairs:
            end = self.context.index
            both = a.rule.hist_active[:end] & b.rule.hist_active[:end]
            hit_a, hit_b = both & a.rule.hist_hit[:end], both & b.rule.hist_hit[:end]
            self._history_pairs[key] = [
                tuple(int(np.count_nonzero(m[max(0, end - w):])) for w in (end, 100, 40))
                for m in (both, hit_a, hit_b, hit_a & hit_b)]
        return [_weighted(c, weights) for c in self._history_pairs[key]]

    def conflict(self, item, kept, weights):
        """중복(전수 상관 ≥ 0.5) 또는 모순(동시 성립률이 기대치의 80% 미만)이면 사유를 돌려준다."""
        joint, cov, var, expected = self.space_pair(item, kept)
        if cov >= 0 and 4 * cov * cov >= var:
            return '중복'
        if 5 * joint * N_CANDIDATES < 4 * expected:
            return '전수 모순'
        overlap, hit_a, hit_b, both = self.history_pair(item, kept, weights)
        if overlap and 5 * both * overlap < 4 * hit_a * hit_b:
            return '이력 모순'
        return None

    def fit(self, setting: Setting):
        w = setting.time_weights
        fitted = []
        for c in self.candidates:
            rates = [h / n for h, n in zip(c.hits, c.counts)]
            if min(rates) < setting.threshold:
                continue
            n_w, h_w = _weighted(c.counts, w), _weighted(c.hits, w)
            if h_w / n_w < setting.threshold:
                continue
            shrunk = (h_w + setting.shrink * c.baseline) / (n_w + setting.shrink)
            weight = _logit(shrunk) - _logit(c.baseline)
            multiplier = c.rule.multiplier if setting.place_boost else 1.0
            fitted.append(Fitted(c.rule, c.baseline, (c.counts[0], n_w, *c.counts[1:]),
                                 (rates[0], h_w / n_w, *rates[1:]), weight, weight * multiplier, c.mask))
        fitted.sort(key=lambda f: (-f.adjusted, f.rule.formula))
        kept, per_variable = [], Counter()
        for item in fitted:
            if per_variable[item.rule.metric.variable] >= setting.var_cap:
                continue
            if any(self.conflict(item, k, w) for k in kept):
                continue
            kept.append(item)
            per_variable[item.rule.metric.variable] += 1
            if len(kept) == setting.limit:
                break
        n2 = N_CANDIDATES ** 2
        mean = fsum(k.adjusted * k.baseline for k in kept)
        variance = fsum(k.adjusted ** 2 * k.baseline * (1 - k.baseline) for k in kept) + 2 * fsum(
            a.adjusted * b.adjusted * self.space_pair(a, b)[1] / n2 for i, a in enumerate(kept) for b in kept[:i])
        return Model(tuple(kept), mean, sqrt(max(0.0, variance)))


@dataclass(frozen=True)
class Model:
    """규칙 점수를 500만 후보 균등분포 기준으로 표준화한 z 점수."""
    rules: tuple
    mean: float
    stdev: float

    def passes(self, ids):
        groups, numbers = np.divmod(np.asarray(ids), N_NUMBERS)
        offsets, bits = numbers >> 3, (1 << (numbers & 7)).astype(np.uint8)
        return [(r.mask[0 if len(r.mask) == 1 else groups, offsets] & bits) != 0 for r in self.rules]

    def score(self, ids):
        out = np.zeros(len(ids))
        if not self.stdev:
            return out
        for rule, passed in zip(self.rules, self.passes(ids)):
            out += rule.adjusted * passed
        return (out - self.mean) / self.stdev

    def full_score(self):
        """조 우선 순서의 500만 후보 점수."""
        if not self.stdev:
            return np.zeros(N_CANDIDATES)
        common, by_group = np.zeros(N_NUMBERS), np.zeros((N_GROUPS, N_NUMBERS))
        for r in self.rules:
            passed = np.unpackbits(r.mask, axis=1, bitorder='little')
            if len(r.mask) == 1:
                common += r.adjusted * passed[0]
            else:
                by_group += r.adjusted * passed
        by_group += common
        by_group -= self.mean
        by_group /= self.stdev
        return by_group.ravel()


# ---------------------------------------------------------------- 점수

def split_ids(ids):
    ids = np.asarray(ids, dtype=np.int32)
    return (ids % N_NUMBERS)[:, None] // POWERS % 10, ids // N_NUMBERS


def frequency_logs(draws, index, weights):
    """동점 정리용: 시간 가중 자리별 숫자 빈도와 조 빈도의 로그 (라플라스 +1)."""
    w = np.full(index, weights[2])
    w[max(0, index - 100):] = weights[1]
    w[max(0, index - 40):] = weights[0]

    def digit_logs(values):
        counts = np.stack([np.bincount(values[:index, j], w, minlength=10) for j in range(6)]) + 1.0
        return np.log(counts / counts.sum(axis=1, keepdims=True))

    group = np.bincount(draws.groups[:index] - 1, w, minlength=5) + 1.0
    return 0.8 * digit_logs(draws.main) + 0.2 * digit_logs(draws.bonus), np.log(group / group.sum())


def tie_term(digits, groups, logs):
    digit_log, group_log = logs
    return TIE_SCALE * (digit_log[np.arange(6), digits].sum(axis=1) + group_log[groups])


def rank_sample(draws, index):
    """실제 1등 후보 1개 + 고정 시드 무작위 후보 20만 개 (실제 1등은 맨 끝)."""
    answer = (int(draws.groups[index]) - 1) * N_NUMBERS + int(draws.main[index] @ POWERS)
    rng = np.random.default_rng(np.random.SeedSequence([72015, int(draws.rounds[index])]))
    sample = rng.choice(N_CANDIDATES - 1, size=RANK_SAMPLE, replace=False, shuffle=False).astype(np.int32)
    sample += sample >= answer
    return np.append(sample, np.int32(answer))


WORKERS = int(os.environ.get('PENSION_WORKERS', os.cpu_count() or 1))
_PARALLEL = {}


def _parallel_call(ix):
    return _PARALLEL['fn'](ix, *_PARALLEL['args'])


def parallel_folds(fn, indices, args, workers=None):
    """회차마다 fn(ix, *args)를 돌린다. 리눅스에서는 프로세스를 나눠 동시에 돌리고 결과 순서는 그대로다."""
    import multiprocessing as mp
    workers = min(workers or WORKERS, len(indices))
    if workers <= 1 or 'fork' not in mp.get_all_start_methods():
        return [fn(ix, *args) for ix in indices]
    _PARALLEL.update(fn=fn, args=args)
    chunk = -(-len(indices) // workers)   # 연속 회차를 한 프로세스에 몰아 캐시를 살린다
    with mp.get_context('fork').Pool(workers) as pool:
        return pool.map(_parallel_call, indices, chunksize=chunk)


def _evaluate_fold(ix, draws, catalogs, histories, space, methods):
    context = Context(draws, ix)
    rule_folds = [RuleFold(cat, hist, context, space) for cat, hist in zip(catalogs, histories)]
    ids = rank_sample(draws, ix)
    digits, groups = split_ids(ids)
    ties = {w: tie_term(digits, groups, frequency_logs(draws, ix, w)) for w in {m.time_weights for m in methods}}
    cache = [{}, {}]
    out = np.zeros(len(methods))
    for s, method in enumerate(methods):
        parts = []
        for stream, fold in enumerate(rule_folds):
            model = fold.fit(method)
            key = tuple((id(r.rule), r.adjusted) for r in model.rules)
            if key not in cache[stream]:
                cache[stream][key] = model.score(ids)
            parts.append(cache[stream][key])
        joint = method.mix[0] * parts[0] + method.mix[1] * parts[1] + ties[method.time_weights]
        out[s] = 1 + np.count_nonzero(joint[:-1] > joint[-1]) + 0.5 * np.count_nonzero(joint[:-1] == joint[-1])
    return out


def evaluate(draws, catalogs, histories, space, methods, folds=ROLLING_FOLDS, log_progress=True):
    """최근 folds회 각각을 그 이전 자료로만 학습해, 무작위 20만 후보 안 실제 1등의 순위. (방법 수, folds) 배열."""
    n = len(draws.rounds)
    start = time.time()
    columns = parallel_folds(_evaluate_fold, list(range(n - folds, n)), (draws, catalogs, histories, space, methods))
    if log_progress:
        print(f'롤링 검증 {folds}회 · 방법 {len(methods)}개 · {time.time() - start:.0f}초', file=sys.stderr, flush=True)
    return np.stack(columns, axis=1)


def tune(draws, catalogs, histories, space, log_progress=True):
    """기본 16개 설정 중 최근 40회 평균 순위가 가장 좋은 것."""
    ranks = evaluate(draws, catalogs, histories, space, SETTINGS, ROLLING_FOLDS, log_progress)
    total = ranks.sum(axis=1)
    return SETTINGS[int(np.argmin(total))], total / ROLLING_FOLDS


def score_all(draws, setting, models, index=None):
    """index 시점(기본: 목표 회차) 모델로 500만 후보 종합점수 S와 (1등 z, 보너스 z)."""
    parts = np.stack([m.full_score() for m in models])
    logs = frequency_logs(draws, len(draws.rounds) if index is None else index, setting.time_weights)
    scores = np.empty(N_CANDIDATES)
    for start in range(0, N_CANDIDATES, 250_000):
        ids = np.arange(start, start + 250_000, dtype=np.int32)
        digits, groups = split_ids(ids)
        scores[start:start + 250_000] = (setting.mix[0] * parts[0, ids] + setting.mix[1] * parts[1, ids]
                                         + tie_term(digits, groups, logs))
    return scores, parts


# ---------------------------------------------------------------- 15개 선정

def shown(score):
    """결과표에 찍히는 소수 8자리 값."""
    return float(f'{score:.8f}')


def _skip_shown_tie(order, scores, start, value):
    """점수순 order에서 start부터, 표시 점수가 value 이상인 구간을 건너뛴 첫 위치."""
    lo, hi = start, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if shown(scores[order[mid]]) < value:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _next_pick(order, scores, start, chosen, picks=PICKS):
    """점수순으로 다음 조건 충족 후보의 위치.

    조건: 같은 자리 숫자 일치 3개 이하, 앞 5자리 자리별 같은 숫자 4회 이하,
    끝 2자리 중복 없음, 끝자리 최대 2회·조 최대 4회, 남은 칸이 빠진 끝자리(조) 수와
    같아지면 빠진 것만 허용, 표시 점수 동점 없음.
    """
    if chosen:
        start = _skip_shown_tie(order, scores, start, shown(scores[chosen[-1]]))
    slots = picks - len(chosen)
    old_groups, old_numbers = np.divmod(np.asarray(chosen, dtype=np.int32), N_NUMBERS)
    old_digits = (old_numbers[:, None] // POWERS % 10).astype(np.int8)
    place_counts = np.array([np.bincount(col, minlength=10) for col in old_digits.T]).reshape(6, 10)
    tails, group_counts = place_counts[5], np.bincount(old_groups, minlength=N_GROUPS)
    ok_tail = tails == 0 if slots == np.count_nonzero(tails == 0) else tails < 2
    ok_group = group_counts == 0 if slots == np.count_nonzero(group_counts == 0) else group_counts < 4
    ok_suffix = np.ones(100, dtype=bool)
    ok_suffix[old_numbers % 100] = False
    ok_digit = place_counts[:5] < 4
    for lo in range(start, len(order), 8_192):
        ids = order[lo:lo + 8_192]
        groups, numbers = np.divmod(ids, N_NUMBERS)
        at = np.flatnonzero(ok_group[groups] & ok_tail[numbers % 10] & ok_suffix[numbers % 100])
        if not at.size:
            continue
        digits = (numbers[at, None] // POWERS % 10).astype(np.int8)
        keep = ok_digit[np.arange(5), digits[:, :5]].all(axis=1)
        for other in old_digits:
            keep &= np.count_nonzero(digits == other, axis=1) <= 3
        if keep.any():
            return lo + int(at[np.argmax(keep)])
    return None


def select(scores, picks=PICKS):
    """점수 내림차순으로 조건을 만족하는 다음 후보를 picks개까지. 막히면 직전 선택을 한 칸씩 물린다.

    10개 이상이면 끝자리 0~9가 모두 들어가므로 7등 이상 하나는 반드시 맞고,
    끝 2·3자리도 모두 달라 6등·5등에 걸릴 확률도 그 개수에서 가장 높다.
    """
    order = np.argsort(-scores, kind='stable').astype(np.int32)
    chosen, ranks, start, backtracks = [], [], 0, 0
    while len(chosen) < picks:
        rank = _next_pick(order, scores, start, chosen, picks)
        if rank is None:
            if not ranks or backtracks == 256:
                raise RuntimeError(f'조건을 모두 만족하는 {picks}개를 찾지 못했습니다.')
            chosen.pop()
            start = ranks.pop() + 1
            backtracks += 1
            continue
        chosen.append(int(order[rank]))
        ranks.append(rank)
        start = rank + 1
    check_selection(chosen, scores)
    return chosen


def check_selection(ids, scores):
    """선정 조건을 결과만 보고 다시 확인한다."""
    picks = len(ids)
    groups = [i // N_NUMBERS + 1 for i in ids]
    numbers = [f'{i % N_NUMBERS:06d}' for i in ids]
    tails = [n[-1] for n in numbers]
    values = [float(scores[i]) for i in ids]
    problems = {
        '후보 중복': len(set(ids)) != picks,
        '점수 내림차순': any(a <= b for a, b in zip(values, values[1:])),
        '표시 점수 동점': len({shown(v) for v in values}) != picks,
        '끝 2자리 중복': len({n[-2:] for n in numbers}) != picks,
        '앞 5자리 같은 숫자 5회 이상': any(max(Counter(n[j] for n in numbers).values()) > 4 for j in range(5)),
        '같은 자리 일치 4개 이상': any(sum(x == y for x, y in zip(a, b)) > 3 for a, b in combinations(numbers, 2)),
        '끝자리 0~9 전부·최대 2회': len(set(tails)) != 10 or max(Counter(tails).values()) > 2,
        '1~5조 전부·최대 4회': len(set(groups)) != 5 or max(Counter(groups).values()) > 4,
    }
    seen_tails, seen_groups = set(), set()
    for i, (tail, group) in enumerate(zip(tails, groups)):
        left = picks - i
        problems['빠진 끝자리 우선'] = problems.get('빠진 끝자리 우선') or (
            left == 10 - len(seen_tails) and tail in seen_tails)
        problems['빠진 조 우선'] = problems.get('빠진 조 우선') or (
            left == 5 - len(seen_groups) and group in seen_groups)
        seen_tails.add(tail)
        seen_groups.add(group)
    broken = [name for name, bad in problems.items() if bad]
    if broken:
        raise RuntimeError('선정 조건 위반: ' + ', '.join(broken))


# ---------------------------------------------------------------- 실행과 저장

def run(draws, log_progress=True, method=None, prepared=None):
    """목표 회차 결과를 계산한다. 반환: (setting, 평균순위, context, models, selected, scores, parts).

    method를 주면 16개 설정 비교 없이 그 방법을 쓴다. prepared는 (catalogs, histories, space) 재사용.
    """
    catalogs, histories, space = prepared or prepare(draws)
    if method is None:
        setting, mean_ranks = tune(draws, catalogs, histories, space, log_progress)
    else:
        setting, mean_ranks = method, None
    context = Context(draws, len(draws.rounds))
    models = [RuleFold(cat, hist, context, space).fit(setting) for cat, hist in zip(catalogs, histories)]
    if log_progress:
        print('500만 후보 채점 중', file=sys.stderr, flush=True)
    scores, parts = score_all(draws, setting, models)
    selected = select(scores)
    return setting, mean_ranks, context, models, selected, scores, parts


def prepare(draws, folds=ROLLING_FOLDS):
    catalogs = [build_catalog(draws, stream, folds) for stream in (MAIN, BONUS)]
    return catalogs, [HistoryIndex(c) for c in catalogs], Space()


def summary(number):
    d = [int(c) for c in f'{number:06d}']
    return (f'합 {sum(d)} · 홀수 {sum(x % 2 for x in d)} · 5 이상 {sum(x >= 5 for x in d)} · '
            f'서로 다른 숫자 {len(set(d))} · 인접 ±1 {sum(abs(a - b) == 1 for a, b in zip(d, d[1:]))}')


CANDIDATE_COLUMNS = ('순위', '조', '6자리', '종합점수 S', '1등 규칙 점수', '보너스 점수', '충족 규칙 수', '특징 요약')
RULE_COLUMNS = ('구분', '유형', '수식', '목표 활성 근거', '전체 적중률', '가중 적중률', '최근 100회 적중률',
                '최근 40회 적중률', '표본 수 전체/최근100/최근40', '가중 표본 수', '기저 통과율', '가중치 w', '자리 배율')


def candidate_rows(selected, scores, parts, models):
    counts = [np.sum(m.passes(selected), axis=0) if m.rules else np.zeros(len(selected), int) for m in models]
    for rank, cid in enumerate(selected, 1):
        n_main, n_bonus = int(counts[0][rank - 1]), int(counts[1][rank - 1])
        yield (rank, cid // N_NUMBERS + 1, f'{cid % N_NUMBERS:06d}', f'{scores[cid]:.8f}', f'{parts[0, cid]:.8f}',
               f'{parts[1, cid]:.8f}', n_main + n_bonus,
               f'1등 {n_main}·보너스 {n_bonus} / {summary(cid % N_NUMBERS)}')


def rule_rows(context, models):
    for stream, model in zip(STREAM_NAMES, models):
        for f in model.rules:
            total, weighted, last100, last40 = f.counts
            yield (stream, f.rule.kind, f.rule.formula, f.rule.evidence(context), *(f'{r:.2%}' for r in f.rates),
                   f'{total}/{last100}/{last40}', f'{weighted:g}', f'{f.baseline:.4%}', f'{f.weight:.8f}',
                   f'{f.rule.multiplier:.1f}')


def _md_table(columns, rows):
    def cell(v):
        return str(v).replace('|', r'\|')
    return '\n'.join('| ' + ' | '.join(map(cell, row)) + ' |'
                     for row in (columns, ['---'] * len(columns), *rows))


def _write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def save(out_dir, context, setting, mean_ranks, models, selected, scores, parts, extra=''):
    """결과.md·후보15.csv·적용규칙.csv·방법.txt. mean_ranks가 있으면 16개 설정 비교표, extra는 덧붙일 본문."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(candidate_rows(selected, scores, parts, models))
    rules = list(rule_rows(context, models))
    _write_csv(out_dir / '후보15.csv', CANDIDATE_COLUMNS, candidates)
    _write_csv(out_dir / '적용규칙.csv', RULE_COLUMNS, rules)
    (out_dir / '방법.txt').write_text(setting.to_text(), encoding='utf-8')
    title = f'연금복권720+ {context.round}회 · {context.date.isoformat()}'
    report = (f'# {title}\n\n'
              f'선택 설정: {setting.label()}\n\n'
              f'500만 후보 전수 채점 · 15개 선정 · 적용 규칙 1등 {len(models[0].rules)}개 · 보너스 {len(models[1].rules)}개\n\n'
              f'## 후보 15개\n\n{_md_table(CANDIDATE_COLUMNS, candidates)}\n\n'
              f'## 적용 규칙\n\n{_md_table(RULE_COLUMNS, rules)}\n')
    if mean_ranks is not None:
        tuning = '\n'.join(f'| {s.label()} | {r:,.1f} |' for s, r in zip(SETTINGS, mean_ranks))
        report += ('\n## 설정 비교\n\n'
                   f'최근 {ROLLING_FOLDS}회를 그 이전 자료로만 학습했을 때, 무작위 20만 후보 안에서 실제 1등의 평균 순위입니다. '
                   '낮을수록 좋고, 무작위면 약 100,000입니다.\n\n'
                   f'| 설정 | 평균 순위 |\n| --- | ---: |\n{tuning}\n')
    if extra:
        report += '\n' + extra
    (out_dir / '결과.md').write_text(report, encoding='utf-8')
    return title


DATA_CSV = Path(__file__).resolve().parent / 'data' / '당첨번호.csv'


def find_workbook():
    if DATA_CSV.exists():
        return DATA_CSV
    here = Path(__file__).resolve().parent
    for folder in dict.fromkeys((Path.cwd(), here)):
        found = sorted(p for p in folder.glob('*.xlsx') if not p.name.startswith('~$'))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError('엑셀 파일이 여러 개입니다. 경로를 인자로 지정하세요.')
    raise ValueError('엑셀 파일을 스크립트 옆에 두거나 경로를 인자로 지정하세요.')


def main():
    parser = argparse.ArgumentParser(description='연금복권720+ 다음 회차 후보 15개')
    parser.add_argument('xlsx', nargs='?', type=Path, help='당첨번호 엑셀 또는 CSV (생략하면 data/당첨번호.csv)')
    parser.add_argument('--out', type=Path, help='결과 폴더 (기본: 스크립트 옆 results/round_<회차>)')
    args = parser.parse_args()
    xlsx = args.xlsx or find_workbook()
    draws = load_draws(xlsx)
    print(f'{xlsx.name}: {draws.rounds[0]}~{draws.rounds[-1]}회 {len(draws.rounds)}개 확인', file=sys.stderr)
    setting, mean_ranks, context, models, selected, scores, parts = run(draws)
    out = args.out or Path(__file__).resolve().parent / 'results' / f'round_{context.round}'
    title = save(out, context, setting, mean_ranks, models, selected, scores, parts)
    print(title)
    print(setting.label())
    for rank, cid in enumerate(selected, 1):
        print(f'{rank:2d}. {cid // N_NUMBERS + 1}조 {cid % N_NUMBERS:06d}  {scores[cid]:.8f}')
    print(out)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f'실행 오류: {error}') from error
