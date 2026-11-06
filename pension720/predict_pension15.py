#!/usr/bin/env python3
"""연금복권720+ 패턴 엔진. Python 3.10+와 NumPy가 필요하다."""
from __future__ import annotations

import argparse
import csv
import itertools
import posixpath
import re
import sys
from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import combinations
from math import fsum, log, sqrt
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


N_NUMBERS = 1_000_000
N_CANDIDATES = 5 * N_NUMBERS
DIGIT_POWERS = np.array([100_000, 10_000, 1_000, 100, 10, 1], dtype=np.int32)
XLS_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PKG_NS = '{http://schemas.openxmlformats.org/package/2006/relationships}'
HEADERS = ('회차', '추첨일', '1등 조', '1등 번호', '보너스 번호')


@dataclass(frozen=True)
class Draws:
    rounds: np.ndarray
    dates: tuple[date, ...]
    groups: np.ndarray
    main: np.ndarray
    bonus: np.ndarray


def sheet_rows(path: Path):
    """Read inline/shared text and raw Excel serials without losing identifier zeros."""
    ns = XLS_NS
    with ZipFile(path) as book:
        index = ET.fromstring(book.read('xl/workbook.xml'))
        sheet = next((s for s in index.findall(f'{ns}sheets/{ns}sheet') if s.get('name') == '당첨번호'), None)
        if sheet is None:
            raise ValueError('엑셀에 당첨번호 시트가 없습니다.')
        relations = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        targets = {r.get('Id'): r.get('Target') for r in relations.findall(f'{PKG_NS}Relationship')}
        target = targets[sheet.attrib[f'{REL_NS}id']]
        sheet_path = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        props = index.find(f'{ns}workbookPr')
        date_1904 = props is not None and props.get('date1904') in ('1', 'true')

        def read_text(el):
            if el is None:
                return ''
            direct = el.find(f'{ns}t')
            return direct.text or '' if direct is not None else ''.join((t.text or '') for t in el.findall(f'{ns}r/{ns}t'))

        shared = []
        if 'xl/sharedStrings.xml' in book.namelist():
            shared = [read_text(e) for e in ET.fromstring(book.read('xl/sharedStrings.xml')).findall(f'{ns}si')]
        with book.open(sheet_path) as source:
            for _, element in ET.iterparse(source, events=('end',)):
                if element.tag != f'{ns}row':
                    continue
                row = {}
                for cell in element.findall(f'{ns}c'):
                    ref = cell.get('r', '')
                    column = re.match(r'^[A-Z]+', ref)
                    if not column:
                        continue
                    kind = cell.get('t', 'n')
                    value = cell.findtext(f'{ns}v') or ''
                    if kind == 'inlineStr':
                        value = read_text(cell.find(f'{ns}is'))
                    elif kind == 's' and value:
                        value = shared[int(value)]
                    if value.strip():
                        row[column.group()] = value.strip()
                yield element.get('r', '?'), row, date_1904
                element.clear()


def excel_date(raw: str, date_1904: bool) -> date:
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', raw):
        serial = int(float(raw))
        epoch = date(1904, 1, 1) if date_1904 else date(1899, 12, 30)
        return epoch + timedelta(days=serial + int(not date_1904 and serial < 60))
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        match = re.fullmatch(r'(\d{4})[-./]\s*(\d{1,2})[-./]\s*(\d{1,2})\.?', raw)
        if match:
            return date(*(int(v) for v in match.groups()))
        raise


def load_draws(path: Path) -> Draws:
    columns = None
    records = []
    for row_number, row, date_1904 in sheet_rows(path):
        if not row:
            continue
        if columns is None:
            inverse = {name: column for column, name in row.items()}
            if any(name not in inverse for name in HEADERS):
                raise ValueError('당첨번호 시트에 필수 열이 없습니다: ' + ', '.join(HEADERS))
            columns = [inverse[name] for name in HEADERS]
            continue
        values = [row.get(col, '') for col in columns]
        if not any(values):
            continue
        try:
            raw_round, raw_date, raw_group, *numbers = values
            if not re.fullmatch(r'[0-9]+(?:회)?', raw_round) or not re.fullmatch(r'[1-5](?:조)?', raw_group):
                raise ValueError('회차 또는 조 형식 오류')
            if any(not re.fullmatch(r'[0-9]{1,6}', n) for n in numbers):
                raise ValueError('번호는 1~6자리의 숫자 문자열이어야 합니다')
            records.append((int(raw_round.removesuffix('회')), excel_date(raw_date, date_1904),
                            int(raw_group.removesuffix('조')), *(n.zfill(6) for n in numbers)))
        except ValueError as error:
            raise ValueError(f'당첨번호 시트 {row_number}행: {error}') from error
    records.sort(key=lambda r: r[0])
    if len(records) < 70:
        raise ValueError('롤링 40회와 최초 학습 표본 30회를 위해 최소 70회가 필요합니다.')
    if any(b[0] != a[0] + 1 or b[1] <= a[1] for a, b in zip(records, records[1:])):
        raise ValueError('회차는 중복·누락 없이 이어지고 날짜는 증가해야 합니다.')
    rounds, dates, groups, main, bonus = zip(*records)

    def digits(strings):
        return np.frombuffer(''.join(strings).encode('ascii'), dtype=np.uint8).reshape(-1, 6).astype(np.int8) - 48

    return Draws(np.asarray(rounds, dtype=np.int32), dates, np.asarray(groups, dtype=np.int8), digits(main), digits(bonus))


# 파생 변수와 규칙 탐색 공간

PLACE_WEIGHTS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
DATE_VARIABLES = ("회차", "연", "월", "일", "날짜 숫자합", "날짜 디지털루트")
UNAVAILABLE = object()


def digital_root(values):
    return np.where(values == 0, 0, 1 + (values - 1) % 9)


class FeatureContext:
    """A target row and data available strictly before that row."""

    def __init__(self, draws, index):
        self.draws = draws
        self.index = index
        if index == 0:
            self.round = int(draws.rounds[0])
            self.date = draws.dates[0]
        else:
            self.round = int(draws.rounds[index - 1]) + 1
            self.date = draws.dates[index - 1] + timedelta(days=7)
        date_sum = sum(int(char) for char in self.date.isoformat() if char.isdigit())
        self.date_values = (self.round, self.date.year, self.date.month, self.date.day,
                            date_sum, int(digital_root(date_sum)))
        self._ages = {}

    def lag(self, stream, lag):
        if self.index < lag:
            return UNAVAILABLE
        numbers = self.draws.main if stream == 0 else self.draws.bonus
        return tuple(map(int, numbers[self.index - lag]))

    def ages(self, stream):
        """Number of skipped rounds; immediately previous appearance has age 0."""
        if self.index == 0:
            return UNAVAILABLE
        if stream not in self._ages:
            if stream == 2:
                history = self.draws.groups[:self.index, None]
                domain = range(1, 6)
            else:
                numbers = self.draws.main if stream == 0 else self.draws.bonus
                history = numbers[:self.index]
                domain = range(10)
            rounds = self.draws.rounds[:self.index]
            fallback = int(rounds[0]) - 1
            self._ages[stream] = tuple(
                tuple(self.round - 1 - int(rounds[seen[-1]] if len(seen) else fallback)
                      for number in domain
                      for seen in (np.flatnonzero(column == number),))
                for column in history.T
            )
        return self._ages[stream]


@dataclass(frozen=True, slots=True)
class Metric:
    key: tuple
    label: str
    variable: str
    evaluate: object
    state: object = None
    describe: object = None
    counts: object = None
    place: int | None = None
    group_sensitive: bool = False

    def get_state(self, context):
        return () if self.state is None else self.state(context)

    def values(self, digits, groups, state):
        return np.broadcast_to(self.evaluate(digits, groups, state), (len(digits),))

    def probability(self, predicate, state):
        if state is UNAVAILABLE:
            return 0.0
        if self.counts is None:
            values, counts = _static_counts(self.key)
        else:
            values, counts = self.counts(state)
        return float(counts[predicate.test(values)].sum()) / int(counts.sum())


@dataclass(frozen=True, slots=True)
class Predicate:
    op: str
    values: tuple

    def test(self, observed):
        first = self.values[0]
        if self.op == "eq":
            return observed == first
        if self.op == "le":
            return observed <= first
        if self.op == "ge":
            return observed >= first
        if self.op == "between":
            return (observed >= first) & (observed <= self.values[1])
        mask = observed == first
        for value in self.values[1:]:
            mask = mask | (observed == value)
        return mask

    def formula(self, label):
        if self.op == "between":
            return f"{self.values[0]} ≤ {label} ≤ {self.values[1]}"
        if self.op == "in":
            return f"{label} ∈ {{{','.join(map(str, self.values))}}}"
        return f"{label} {dict(eq='=', le='≤', ge='≥')[self.op]} {self.values[0]}"


@dataclass(frozen=True, slots=True)
class Antecedent:
    label: str
    predicate: Predicate
    value: object
    active: np.ndarray

    @property
    def formula(self):
        return self.predicate.formula(self.label)


@dataclass(frozen=True, slots=True, eq=False)
class Rule:
    metric: Metric
    predicate: Predicate
    stream: int
    hist_hit: np.ndarray
    hist_active: np.ndarray
    antecedent: Antecedent | None = None

    @property
    def kind(self):
        return "B" if self.antecedent is None else "A"

    @property
    def key(self):
        return self.formula

    @property
    def variable(self):
        return self.metric.variable

    @property
    def multiplier(self):
        return 1.0 if self.metric.place is None else PLACE_WEIGHTS[self.metric.place]

    @property
    def group_sensitive(self):
        return self.metric.group_sensitive

    @property
    def formula(self):
        place_symbol = "d" if self.stream == 0 else "b"
        consequence = self.predicate.formula(self.metric.label.replace("{p}", place_symbol))
        return consequence if self.antecedent is None else f"{self.antecedent.formula} ⇒ {consequence}"

    def active(self, context):
        if self.metric.get_state(context) is UNAVAILABLE:
            return False
        if self.antecedent is None:
            return True
        value = self.antecedent.value(context)
        return value is not UNAVAILABLE and bool(self.antecedent.predicate.test(value))

    def activation(self, context):
        evidence = []
        if self.antecedent is not None:
            evidence.append(f"{self.antecedent.label} = {self.antecedent.value(context)}")
        if self.metric.describe is not None:
            evidence.append(self.metric.describe(context, self.metric.get_state(context)))
        return " · ".join(evidence) if evidence else "상시 적용"

    def p0(self, context):
        return self.metric.probability(self.predicate, self.metric.get_state(context))

    def mask(self, digits, groups, context):
        state = self.metric.get_state(context)
        if state is UNAVAILABLE:
            return np.zeros(len(digits), dtype=bool)
        return self.predicate.test(self.metric.values(digits, groups, state))

    def cache_key(self, context):
        return self.metric.key, self.predicate, self.metric.get_state(context)


def _counts(values):
    if values.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    start, end = int(values.min()), int(values.max())
    return (np.arange(start, end + 1),
            np.bincount(values.astype(np.int64, copy=False) - start, minlength=end - start + 1))


def _compress_counts(values, weights):
    combined = Counter()
    for value, count in zip(values, weights):
        combined[int(value)] += int(count)
    return np.array(tuple(combined)), np.array(tuple(combined.values()))


def _digit_sum(digits):
    return digits.sum(axis=1, dtype=np.int16)


def _abs_diff(digits):
    return np.abs(np.diff(digits, axis=1))


@lru_cache(maxsize=1)
def _feature_grid():
    numbers = np.arange(1_000_000, dtype=np.int32)
    powers = np.array([100_000, 10_000, 1_000, 100, 10, 1], dtype=np.int32)
    return ((numbers[:, None] // powers) % 10).astype(np.int8)


def _max_repeat(digits):
    answer = np.zeros(len(digits), dtype=np.int8)
    for value in range(10):
        np.maximum(answer, (digits == value).sum(axis=1, dtype=np.int8), out=answer)
    return answer


@lru_cache(maxsize=1)
def _static_metrics():
    metrics = []

    def add(name, label, evaluate, place=None, group_sensitive=False):
        metrics.append(Metric((name,), label, name, evaluate, place=place,
                              group_sensitive=group_sensitive))

    for place in range(6):
        digit = f"{{p}}{place + 1}"
        add(f"digit-{place}", digit, lambda d, g, s, j=place: d[:, j], place)
        add(f"high-{place}", f"I({digit} ≥ 5)", lambda d, g, s, j=place: d[:, j] >= 5, place)
        for modulus in (2, 3, 4, 5):
            add(f"digit-mod{modulus}-{place}", f"{digit} mod {modulus}",
                lambda d, g, s, j=place, m=modulus: d[:, j] % m, place)
    add("sum", "자리합", lambda d, g, s: _digit_sum(d))
    for modulus in (3, 4, 5, 7, 9, 10, 11):
        add(f"sum-mod{modulus}", f"자리합 mod {modulus}",
            lambda d, g, s, m=modulus: _digit_sum(d) % m)
    add("digital-root", "자리합 디지털루트", lambda d, g, s: digital_root(_digit_sum(d)))
    add("odd-count", "홀수 개수", lambda d, g, s: (d % 2).sum(axis=1))
    add("high-count", "5 이상 개수", lambda d, g, s: (d >= 5).sum(axis=1))
    add("zero-count", "0 개수", lambda d, g, s: (d == 0).sum(axis=1))
    add("prime-count", "{2,3,5,7} 개수", lambda d, g, s: np.isin(d, (2, 3, 5, 7)).sum(axis=1))
    add("distinct-count", "서로 다른 숫자 개수", lambda d, g, s: 1 +
        (np.diff(np.sort(d, axis=1), axis=1) != 0).sum(axis=1))
    add("max-repeat", "같은 숫자 최대 반복", lambda d, g, s: _max_repeat(d))
    add("adj-equal", "인접 같은 숫자 쌍", lambda d, g, s: (_abs_diff(d) == 0).sum(axis=1))
    add("adj-one", "인접 ±1 쌍", lambda d, g, s: (_abs_diff(d) == 1).sum(axis=1))
    for place in range(5):
        add(f"adj-difference-{place}", f"{{p}}{place + 2}−{{p}}{place + 1}",
            lambda d, g, s, j=place: d[:, j + 1] - d[:, j])
        add(f"adj-abs-difference-{place}", f"|{{p}}{place + 2}−{{p}}{place + 1}|",
            lambda d, g, s, j=place: np.abs(d[:, j + 1] - d[:, j]))
    add("adj-difference-sum", "인접 절대차 합", lambda d, g, s: _abs_diff(d).sum(axis=1))
    add("adj-difference-max", "인접 절대차 최댓값", lambda d, g, s: _abs_diff(d).max(axis=1))
    add("front-sum", "앞3자리 합", lambda d, g, s: d[:, :3].sum(axis=1))
    add("back-sum", "뒤3자리 합", lambda d, g, s: d[:, 3:].sum(axis=1))
    add("front-minus-back", "앞3자리 합−뒤3자리 합", lambda d, g, s: d[:, :3].sum(axis=1) -
        d[:, 3:].sum(axis=1))
    for modulus in (7, 9, 11, 13):
        powers = np.array([10 ** i % modulus for i in range(5, -1, -1)], dtype=np.int16)
        add(f"number-mod{modulus}", f"6자리 정수 mod {modulus}",
            lambda d, g, s, m=modulus, a=powers: (d @ a) % m)
    add("last-two", "끝2자리", lambda d, g, s: d[:, -2] * 10 + d[:, -1])
    add("last-three", "끝3자리", lambda d, g, s: d[:, -3].astype(np.int16) * 100 +
        d[:, -2] * 10 + d[:, -1])
    add("group", "g", lambda d, g, s: g, group_sensitive=True)
    add("same-as-group", "I(g와 같은 숫자 포함)", lambda d, g, s: (d == g[:, None]).any(axis=1),
        group_sensitive=True)
    add("sum-group-mod5", "(자리합+g) mod 5", lambda d, g, s: (_digit_sum(d) + g) % 5,
        group_sensitive=True)
    return tuple(metrics)


@lru_cache(maxsize=None)
def _static_counts(key):
    metric = next(metric for metric in _static_metrics() if metric.key == key)
    digits = _feature_grid()
    group_ids = range(1, 6) if metric.group_sensitive else (1,)
    totals = Counter()
    for group in group_ids:
        values, counts = _counts(metric.values(digits, np.array([group], dtype=np.int8), ()))
        totals.update({int(value): int(count) for value, count in zip(values, counts)})
    multiplier = 1 if metric.group_sensitive else 5
    return np.array(tuple(totals)), np.array(tuple(totals.values())) * multiplier


def _one_axis_counts(values):
    values = np.asarray(values, dtype=np.int16)
    labels, counts = _counts(values)
    return labels, counts * (5_000_000 // len(values))


@lru_cache(maxsize=4096)
def _carry_counts(reference, distance):
    counts = np.array([1], dtype=np.int64)
    for digit in reference:
        matches = 1 if distance == 0 else 1 + int(0 < digit < 9)
        counts = np.convolve(counts, [10 - matches, matches])
    return np.arange(7), counts * 5


def _lag_source(stream, lag):
    return f"{lag}회 전 {'1등' if stream == 0 else '보너스'}"


def _at_place(values, place):
    return UNAVAILABLE if values is UNAVAILABLE else values[place]


def _date_counts(base_key, state, modulus):
    values, counts = _static_counts(base_key)
    return _compress_counts((values - state) % modulus, counts)


def _extreme_lag(context, source, summarize, low, high):
    prior = _prior_value(context, source, 1, summarize)
    if prior is UNAVAILABLE or low < prior < high:
        return UNAVAILABLE
    return prior


def _dynamic_metrics():
    """Metric, candidate predicate family, and historical availability."""
    metrics = []
    for source in (0, 1):
        for lag in (1, 2, 3):
            name = _lag_source(source, lag)
            for distance in (0, 1):
                relation = "일치" if distance == 0 else "±1 이웃"
                key = ("carry-count", distance, source, lag)
                metric = Metric(key, f"{name} 같은 자리 {relation} 개수", str(key),
                                lambda d, g, s, distance=distance:
                                (np.abs(d - np.asarray(s, dtype=np.int8)) == distance).sum(axis=1),
                                lambda c, source=source, lag=lag: c.lag(source, lag),
                                lambda c, s, name=name: f"{name} = {''.join(map(str, s))}",
                                lambda s, distance=distance: _carry_counts(s, distance))
                metrics.append((metric, "carry"))
            for place in range(6):
                prefix = f"{{p}}{place + 1}−{name} {place + 1}자리"
                state = lambda c, source=source, lag=lag, j=place: _at_place(c.lag(source, lag), j)
                describe = lambda c, s, name=name, j=place: f"{name} {j + 1}자리 = {s}"
                for distance in (0, 1):
                    key = ("carry-digit", distance, source, lag, place)
                    variable = str(("carry-digit", source, lag, place))
                    metric = Metric(key, f"|{prefix}|", variable,
                                    lambda d, g, s, j=place: np.abs(d[:, j] - s),
                                    state, describe, lambda s: _one_axis_counts(np.abs(np.arange(10) - s)),
                                    place)
                    metrics.append((metric, "match" if distance == 0 else "neighbor"))
                if lag == 1:
                    key = ("lag-difference", source, place)
                    metric = Metric(key, prefix, str(key), lambda d, g, s, j=place: d[:, j] - s,
                                    state, describe, lambda s: _one_axis_counts(np.arange(10) - s), place)
                    metrics.append((metric, "difference"))

        for place in range(6):
            key = ("digit-age", source, place)
            source_label = "1등" if source == 0 else "보너스"
            suffix = "·끝자리" if place == 5 else ""
            metric = Metric(key, f"{source_label} 이력의 {{p}}{place + 1} 미출현 주기{suffix}", str(key),
                            lambda d, g, s, j=place: np.take(s, d[:, j]),
                            lambda c, source=source, j=place: _at_place(c.ages(source), j),
                            lambda c, s: "숫자별 주기(0~9)=" + ",".join(map(str, s)),
                            _one_axis_counts, place)
            metrics.append((metric, "age"))
        for base, label, low, high, maximum, thresholds, summarize, evaluate in (
            ("sum", "자리합", 18, 36, 54, (28, 29, 30, 32), sum, _digit_sum),
            ("odd-count", "홀수 개수", 1, 5, 6, (4, 5, 6),
             lambda d: sum(x % 2 for x in d), lambda d: (d % 2).sum(axis=1)),
            ("high-count", "5 이상 개수", 1, 5, 6, (4, 5, 6),
             lambda d: sum(x >= 5 for x in d), lambda d: (d >= 5).sum(axis=1)),
        ):
            key = ("reversion", source, base)
            previous = f"{_lag_source(source, 1)} {label}"
            formula = f"({previous}≤{low}이면 현재 {label}, ≥{high}이면 {maximum}−현재 {label})"
            metric = Metric(
                key, formula, str(key),
                lambda d, g, s, f=evaluate, low=low, maximum=maximum: f(d) if s <= low else maximum - f(d),
                lambda c, source=source, f=summarize, low=low, high=high: _extreme_lag(c, source, f, low, high),
                lambda c, s, previous=previous: f"{previous} = {s}",
                lambda s, key=(base,), low=low, maximum=maximum:
                _static_counts(key) if s <= low else (maximum - _static_counts(key)[0], _static_counts(key)[1]),
            )
            metrics.append((metric, ("reversion", thresholds)))
    metrics.append((Metric(("group-age",), "g 미출현 주기", "group-age",
                           lambda d, g, s: np.take(s, g - 1),
                           lambda c: _at_place(c.ages(2), 0),
                           lambda c, s: "조별 주기(1~5)=" + ",".join(map(str, s)),
                           _one_axis_counts, group_sensitive=True), "age"))

    for ref, ref_name in enumerate(DATE_VARIABLES):
        for place in (*range(6), None):
            moduli = (2, 3, 4, 5) if place is not None else (2, 3, 4, 5, 7, 9, 10, 11)
            for modulus in moduli:
                key = ("date", ref, place, modulus)
                base_key = ("sum",) if place is None else (f"digit-{place}",)
                left = "자리합" if place is None else f"{{p}}{place + 1}"
                if place is None:
                    evaluate = lambda d, g, s, m=modulus: (_digit_sum(d) - s) % m
                else:
                    evaluate = lambda d, g, s, j=place, m=modulus: (d[:, j] - s) % m
                metric = Metric(key, f"({left}−{ref_name}) mod {modulus}", str(key), evaluate,
                                lambda c, ref=ref, m=modulus: c.date_values[ref] % m,
                                lambda c, s, ref=ref, label=ref_name, m=modulus:
                                f"{label} = {c.date_values[ref]}, 나머지 mod {m} = {s}",
                                lambda s, k=base_key, m=modulus: _date_counts(k, s, m), place)
                metrics.append((metric, ("mod", modulus)))
    return tuple(metrics)


def _range_predicate(low, high, minimum=None, maximum=None):
    if low == high:
        return Predicate("eq", (low,))
    if low == minimum:
        return Predicate("le", (high,))
    if high == maximum:
        return Predicate("ge", (low,))
    return Predicate("between", (low, high))


def _mod_predicates(modulus, *, all_small):
    events = {Predicate("eq", (i,)) for i in range(modulus)}
    for size in range(2, (modulus + 1) // 2 + 1):
        if all_small and modulus <= 7:
            subsets = combinations(range(modulus), size)
        else:
            subsets = {tuple(sorted((start + offset * step) % modulus for offset in range(size)))
                       for start in range(modulus) for step in (1, 2)
                       if len({(start + offset * step) % modulus for offset in range(size)}) == size}
        events.update(Predicate("in", tuple(subset)) for subset in subsets)
    return tuple(sorted(events, key=lambda event: (event.op, event.values)))


def _static_predicates(metric):
    variable = metric.variable
    values, counts = _static_counts(metric.key)
    if "mod" in variable:
        return _mod_predicates(int(values.max()) + 1, all_small=True)
    if variable == "group":
        return tuple(Predicate("eq", (group,)) for group in range(1, 6)) + tuple(
            Predicate("in", pair) for pair in combinations(range(1, 6), 2))
    minimum, maximum = int(values.min()), int(values.max())
    if maximum <= 14 and minimum >= 0 or maximum - minimum <= 20:
        return tuple(_range_predicate(low, high, minimum, maximum)
                     for low in range(minimum, maximum + 1)
                     for high in range(low, maximum + 1))
    order = np.argsort(values)
    counts = counts[order]
    cumulative = np.r_[0, np.cumsum(counts)]
    total = int(cumulative[-1])
    if maximum >= 100:
        step = 25 if maximum > 100 else 5
        starts = range(minimum, maximum + 1, step)
    else:
        starts = range(minimum, maximum + 1)
    predicates = set()
    for low in starts:
        for mass in (.35, .40, .45, .50):
            edge = cumulative[low - minimum] + mass * total
            index = np.searchsorted(cumulative, edge, side="right") - 2
            if index + minimum >= low and edge <= total:
                predicates.add(_range_predicate(low, index + minimum, minimum, maximum))
    for mass in (.35, .40, .45, .50):
        index = np.searchsorted(cumulative, (1 - mass) * total, side="left")
        if index <= maximum - minimum:
            predicates.add(Predicate("ge", (int(index + minimum),)))
    return tuple(sorted(predicates, key=lambda event: (event.op, event.values)))


def _dynamic_predicates(kind):
    if kind == "match":
        return (Predicate("eq", (0,)),)
    if kind == "neighbor":
        return (Predicate("eq", (1,)),)
    if kind == "carry":
        return tuple(Predicate(op, (value,)) for op, domain in
                     (("eq", range(7)), ("ge", range(1, 7)), ("le", range(1, 6))) for value in domain)
    if kind == "difference":
        return tuple(_range_predicate(low, high, -9, 9)
                     for low in range(-9, 10) for high in range(low, 10)
                     if high - low <= 5 or low == -9 or high == 9)
    if kind == "age":
        boundaries = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)
        return tuple([Predicate("le", (value,)) for value in boundaries]
                     + [Predicate("ge", (value,)) for value in boundaries[1:]]
                     + [Predicate("between", (low, high)) for low, high in zip(boundaries, boundaries[2:])])
    if kind[0] == "reversion":
        return tuple(Predicate("ge", (threshold,)) for threshold in kind[1])
    return _mod_predicates(kind[1], all_small=False)


def _prior_value(context, stream, lag, summarize):
    digits = context.lag(stream, lag)
    return UNAVAILABLE if digits is UNAVAILABLE else summarize(digits)


def _antecedents(contexts):
    antecedents = []

    def add(label, value, predicates):
        states = [value(context) for context in contexts]
        available = np.array([state is not UNAVAILABLE for state in states])
        numbers = np.array([state if usable else 0 for state, usable in zip(states, available)])
        for predicate in predicates:
            active = available & predicate.test(numbers)
            if int(active.sum()) >= 30:
                antecedents.append(Antecedent(label, predicate, value, active))

    low_high = lambda lows, highs: tuple([Predicate("le", (x,)) for x in lows]
                                        + [Predicate("ge", (x,)) for x in highs])
    for stream in (0, 1):
        for lag in (1, 2, 3):
            prefix = _lag_source(stream, lag)
            summaries = (("자리합", sum, low_high((18, 22, 25), (29, 32, 36))),
                         ("홀수 개수", lambda d: sum(x % 2 for x in d), low_high((1, 2), (4, 5))),
                         ("5 이상 개수", lambda d: sum(x >= 5 for x in d), low_high((1, 2), (4, 5))))
            for label, summarize, predicates in summaries:
                add(f"{prefix} {label}",
                    lambda c, s=stream, l=lag, f=summarize: _prior_value(c, s, l, f), predicates)
            for place in range(6):
                add(f"{prefix} {place + 1}자리",
                    lambda c, s=stream, l=lag, j=place: _at_place(c.lag(s, l), j),
                    low_high((2, 4), (5, 7)))
                add(f"{prefix} {place + 1}자리 mod 2",
                    lambda c, s=stream, l=lag, j=place: _prior_value(c, s, l, lambda d: d[j] % 2),
                    (Predicate("eq", (0,)), Predicate("eq", (1,))))
    for lag in (1, 2, 3):
        add(f"{lag}회 전 g", lambda c, lag=lag: int(c.draws.groups[c.index - lag])
            if c.index >= lag else UNAVAILABLE, tuple(Predicate("eq", (x,)) for x in range(1, 6)))
    for ref, label in enumerate(DATE_VARIABLES):
        for modulus in (2, 3, 4, 5):
            add(f"{label} mod {modulus}", lambda c, ref=ref, m=modulus: c.date_values[ref] % m,
                tuple(Predicate("eq", (x,)) for x in range(modulus)))
    return tuple(antecedents)


def _dynamic_history(metric, predicates, contexts, digits, groups):
    states = [metric.get_state(context) for context in contexts]
    values = np.zeros(len(digits), dtype=np.int32)
    eligible = np.zeros((len(predicates), len(digits)), dtype=bool)
    by_state = {}
    for index, state in enumerate(states):
        if state is not UNAVAILABLE:
            by_state.setdefault(state, []).append(index)
    for state, indices in by_state.items():
        values[indices] = metric.values(digits[indices], groups[indices], state)
        support_values, counts = metric.counts(state)
        total = int(counts.sum())
        if total != 5_000_000:
            raise AssertionError(f"{metric.key} 분포의 후보 수가 500만이 아닙니다: {total}")
        for row, predicate in enumerate(predicates):
            passed = int(counts[predicate.test(support_values)].sum())
            if 0 < passed * 2 <= total:
                eligible[row, indices] = True
    return values, eligible


def _rolling_windows(row_count):
    ends = np.arange(max(0, row_count - 40), row_count + 1)
    starts = np.column_stack((np.zeros_like(ends), np.maximum(0, ends - 100), np.maximum(0, ends - 40)))
    return ends, starts


def _window_counts(flags, ends, starts):
    prefix = np.empty((len(flags), flags.shape[1] + 1), dtype=np.int32)
    prefix[:, 0] = 0
    np.cumsum(flags, axis=1, dtype=np.int32, out=prefix[:, 1:])
    return prefix[:, ends, None] - prefix[:, starts]


def build_catalog(draws, stream):
    row_count = len(draws.rounds)
    contexts = tuple(FeatureContext(draws, index) for index in range(row_count))
    target = FeatureContext(draws, row_count)
    digits = draws.main if stream == 0 else draws.bonus
    groups = draws.groups
    always_active = np.ones(len(digits), dtype=bool)
    antecedents = _antecedents(contexts)
    ends, starts = _rolling_windows(row_count)
    minimum = np.array([30, 10, 4])
    active_matrix = np.stack([antecedent.active for antecedent in antecedents])
    target_active = np.array([antecedent.predicate.test(antecedent.value(target))
                              if antecedent.value(target) is not UNAVAILABLE else False
                              for antecedent in antecedents])
    activates = np.column_stack((active_matrix, target_active))[:, ends]
    antecedent_counts = _window_counts(active_matrix, ends, starts)
    antecedent_possible = activates & np.all(antecedent_counts >= minimum, axis=2)
    minimum_antecedent_hits = (55 * antecedent_counts + 99) // 100
    always_counts = _window_counts(always_active[None, :], ends, starts)
    always_possible = np.all(always_counts >= minimum, axis=2)
    minimum_always_hits = (55 * always_counts + 99) // 100
    catalog = []
    for metric in _static_metrics():
        values = metric.values(digits, groups, ())
        for predicate in _static_predicates(metric):
            p0 = metric.probability(predicate, ())
            if not 0 < p0 <= .5:
                continue
            hits = predicate.test(values)
            if hits.sum() < 17:
                continue
            hit_counts = _window_counts(hits[None, :], ends, starts)
            if np.any(always_possible & np.all(hit_counts >= minimum_always_hits, axis=2)):
                catalog.append(Rule(metric, predicate, stream, hits, always_active))
            if p0 >= .4:
                conditional_hits = _window_counts(active_matrix & hits, ends, starts)
                possible = antecedent_possible & np.all(conditional_hits >= minimum_antecedent_hits, axis=2)
                for index in np.flatnonzero(possible.any(axis=1)):
                    antecedent = antecedents[index]
                    catalog.append(Rule(metric, predicate, stream, hits, antecedent.active, antecedent))
    for metric, kind in _dynamic_metrics():
        predicates = _dynamic_predicates(kind)
        values, eligible = _dynamic_history(metric, predicates, contexts, digits, groups)
        counts = _window_counts(eligible, ends, starts)
        possible = np.all(counts >= minimum, axis=2)
        indices = np.flatnonzero(possible.any(axis=1))
        if len(indices):
            hits = np.stack([predicates[index].test(values) for index in indices])
            hit_counts = _window_counts(hits & eligible[indices], ends, starts)
            possible = possible[indices] & np.all(100 * hit_counts >= 55 * counts[indices], axis=2)
            for row in np.flatnonzero(possible.any(axis=1)):
                index = indices[row]
                catalog.append(Rule(metric, predicates[index], stream, hits[row], eligible[index]))
    return catalog


# 규칙 채택과 전수 채점

NUMBER_SPACE = 1_000_000
GROUP_SPACE = 5
CANDIDATE_SPACE = GROUP_SPACE * NUMBER_SPACE


@dataclass(frozen=True)
class _SpaceMask:
    packed: np.ndarray
    integers: tuple
    cardinality: int


@dataclass(frozen=True)
class _SpacePair:
    joint: float
    expected: float
    covariance: float
    correlation: float
    duplicate: bool
    contradiction: bool


@dataclass(frozen=True)
class _HistoryPair:
    overlap: float
    joint: float | None
    expected: float | None
    contradiction: bool


class _HistoryCounts:
    """Prefix counts let every rolling fold use the same event history safely."""

    def __init__(self, catalog):
        active = np.stack([rule.hist_active for rule in catalog])
        hits = np.stack([rule.hist_hit for rule in catalog]) & active
        self.active = np.zeros((len(catalog), active.shape[1] + 1), dtype=np.int32)
        self.hits = np.zeros_like(self.active)
        np.cumsum(active, axis=1, dtype=np.int32, out=self.active[:, 1:])
        np.cumsum(hits, axis=1, dtype=np.int32, out=self.hits[:, 1:])


class ExactSpace:
    """Exact event sets over all five groups and all six-digit numbers."""

    def __init__(self, mask_cache_bytes=160 * 1024 * 1024, pair_cache_size=200_000):
        numbers = np.arange(NUMBER_SPACE, dtype=np.int32)
        self.digits = np.empty((NUMBER_SPACE, 6), dtype=np.int8)
        for pos, divisor in enumerate((100000, 10000, 1000, 100, 10, 1)):
            self.digits[:, pos] = (numbers // divisor) % 10
        self.mask_cache = OrderedDict()
        self.mask_cache_bytes = mask_cache_bytes
        self.cached_bytes = 0
        self.pair_cache = OrderedDict()
        self.pair_cache_size = pair_cache_size
        self.history_cache = {}

    def history(self, catalog):
        key = id(catalog)
        if key not in self.history_cache:
            self.history_cache[key] = (catalog, _HistoryCounts(catalog))
        return self.history_cache[key][1]

    def mask(self, candidate, context):
        key = candidate.mask_key
        if key in self.mask_cache:
            self.mask_cache.move_to_end(key)
            return self.mask_cache[key]

        group_count = GROUP_SPACE if candidate.rule.group_sensitive else 1
        packed = np.empty((group_count, NUMBER_SPACE // 8), dtype=np.uint8)
        integers = []
        for group_idx in range(group_count):
            groups = np.array([group_idx + 1], dtype=np.int8)
            flags = np.asarray(candidate.rule.mask(self.digits, groups, context), dtype=bool)
            if flags.shape != (NUMBER_SPACE,):
                flags = np.broadcast_to(flags, (NUMBER_SPACE,))
            packed[group_idx] = np.packbits(flags, bitorder="little")
            integers.append(int.from_bytes(packed[group_idx], "little"))
        cardinality = sum(value.bit_count() for value in integers)
        if group_count == 1:
            cardinality *= GROUP_SPACE
        if abs(cardinality / CANDIDATE_SPACE - candidate.baseline) > 1e-12:
            raise ValueError(
                f"규칙 {candidate.rule.key}의 전수 기저율과 계산값 불일치: "
                f"{cardinality / CANDIDATE_SPACE:.12f} vs {candidate.baseline:.12f}"
            )
        result = _SpaceMask(packed, tuple(integers), cardinality)
        self.mask_cache[key] = result
        self.cached_bytes += 2 * result.packed.nbytes
        while self.cached_bytes > self.mask_cache_bytes and len(self.mask_cache) > 1:
            _, old = self.mask_cache.popitem(last=False)
            self.cached_bytes -= 2 * old.packed.nbytes
        return result

    def joint_count(self, first, second, context):
        if first.mask_key == second.mask_key:
            return self.mask(first, context).cardinality
        pair_key = frozenset((first.mask_key, second.mask_key))
        if pair_key in self.pair_cache:
            self.pair_cache.move_to_end(pair_key)
            return self.pair_cache[pair_key]

        first_bits = self.mask(first, context).integers
        second_bits = self.mask(second, context).integers
        if len(first_bits) == len(second_bits) == 1:
            joint = GROUP_SPACE * (first_bits[0] & second_bits[0]).bit_count()
        else:
            if len(first_bits) == 1:
                first_bits *= GROUP_SPACE
            if len(second_bits) == 1:
                second_bits *= GROUP_SPACE
            joint = sum((a & b).bit_count() for a, b in zip(first_bits, second_bits))
        self.pair_cache[pair_key] = joint
        if len(self.pair_cache) > self.pair_cache_size:
            self.pair_cache.popitem(last=False)
        return joint


@dataclass(frozen=True)
class _EligibleRule:
    index: int
    rule: object
    mask_key: object
    baseline: float
    # Order: entire history, last 100 draws, last 40 draws.
    counts: tuple
    hits: tuple
    rates: tuple


@dataclass(frozen=True)
class FittedRule:
    index: int
    rule: object
    mask_key: object
    baseline: float
    # Order: entire history, weighted history, last 100 draws, last 40 draws.
    rates: tuple
    counts: tuple
    weight: float
    adjusted_weight: float


def _weighted_count(cumulative, time_weights):
    all_count, recent100, recent40 = cumulative
    recent, middle, older = time_weights
    return recent40 * recent + (recent100 - recent40) * middle + (all_count - recent100) * older


def _logit(value):
    return log(value / (1.0 - value))


class RuleFold:
    """One expanding-window fit, with shared exact and historical pair counts."""

    def __init__(self, catalog, context, space):
        self.context = context
        self.space = space
        self.history_pair_counts = {}
        self.space_pairs = {}
        self.eligible = []
        history = space.history(catalog)
        end = context.index
        boundaries = [0, max(0, end - 100), max(0, end - 40)]
        counts = history.active[:, [end]] - history.active[:, boundaries]
        hits = history.hits[:, [end]] - history.hits[:, boundaries]
        minimum = np.array([30, 10, 4], dtype=np.int32)
        empirical = np.all((counts >= minimum) & (100 * hits >= 55 * counts), axis=1)
        for index in np.flatnonzero(empirical):
            rule = catalog[index]
            if not rule.active(context):
                continue
            baseline = float(rule.p0(context))
            if not (0 < baseline <= 0.5 and (rule.kind == "B" or baseline >= 0.4)):
                continue
            counts_row = tuple(map(int, counts[index]))
            hits_row = tuple(map(int, hits[index]))
            rates = tuple(hit / count for hit, count in zip(hits_row, counts_row))
            self.eligible.append(_EligibleRule(
                int(index), rule, rule.cache_key(context), baseline, counts_row, hits_row, rates,
            ))

    def _space_pair(self, first, second):
        key = tuple(sorted((first.index, second.index)))
        if key not in self.space_pairs:
            total = CANDIDATE_SPACE
            joint_count = self.space.joint_count(first, second, self.context)
            first_count = round(first.baseline * total)
            second_count = round(second.baseline * total)
            expected_numerator = first_count * second_count
            covariance_numerator = joint_count * total - expected_numerator
            variance_numerator = expected_numerator * (total - first_count) * (total - second_count)
            # Integer comparisons preserve exact boundaries at 0.5 and 80%.
            self.space_pairs[key] = _SpacePair(
                joint_count / total,
                expected_numerator / total ** 2,
                covariance_numerator / total ** 2,
                covariance_numerator / sqrt(variance_numerator),
                covariance_numerator >= 0 and 4 * covariance_numerator ** 2 >= variance_numerator,
                5 * joint_count * total < 4 * expected_numerator,
            )
        return self.space_pairs[key]

    def _history_pair(self, first, second, time_weights):
        # For conditional rules, compare both RHS events on the intersection
        # of their historical antecedents. With no overlap there is no comparison.
        key = (first.index, second.index)
        if key not in self.history_pair_counts:
            end = self.context.index
            active = first.rule.hist_active[:end] & second.rule.hist_active[:end]
            first_hit = active & first.rule.hist_hit[:end]
            second_hit = active & second.rule.hist_hit[:end]
            joint = first_hit & second_hit
            self.history_pair_counts[key] = tuple(
                tuple(np.count_nonzero(mask[max(0, end - window):]) for window in (end, 100, 40))
                for mask in (active, first_hit, second_hit, joint)
            )
        overlap, first_hit, second_hit, joint = (
            _weighted_count(counts, time_weights) for counts in self.history_pair_counts[key]
        )
        if not overlap:
            return _HistoryPair(0.0, None, None, False)
        return _HistoryPair(
            overlap, joint / overlap, (first_hit * second_hit) / overlap ** 2,
            5 * joint * overlap < 4 * first_hit * second_hit,
        )

    def fit(self, time_weights, threshold, shrink, limit, audit=False):
        if threshold < 0.55:
            raise ValueError("적중률 문턱은 55% 이상이어야 합니다.")
        fitted = []
        for candidate in self.eligible:
            if min(candidate.rates) < threshold:
                continue
            weighted_count = _weighted_count(candidate.counts, time_weights)
            weighted_hit = _weighted_count(candidate.hits, time_weights)
            weighted_rate = weighted_hit / weighted_count
            if weighted_rate < threshold:
                continue
            shrunk_rate = (weighted_hit + shrink * candidate.baseline) / (weighted_count + shrink)
            weight = _logit(shrunk_rate) - _logit(candidate.baseline)
            fitted.append(FittedRule(
                candidate.index, candidate.rule, candidate.mask_key, candidate.baseline,
                (candidate.rates[0], weighted_rate, *candidate.rates[1:]),
                (candidate.counts[0], weighted_count, *candidate.counts[1:]),
                weight, weight * candidate.rule.multiplier,
            ))
        fitted.sort(key=lambda item: (-item.adjusted_weight, item.rule.key))

        kept = []
        rejected = []
        variables = {}
        for item in fitted:
            if variables.get(item.rule.variable, 0) >= 2:
                continue
            for previous in kept:
                space_pair = self._space_pair(item, previous)
                history_pair = self._history_pair(item, previous, time_weights)
                reasons = []
                if space_pair.duplicate:
                    reasons.append("중복: 전수 상관 ≥ 0.5")
                if space_pair.contradiction:
                    reasons.append("모순: 전수 동시 성립률 < 기대치의 80%")
                if history_pair.contradiction:
                    reasons.append("모순: 가중 이력 동시 성립률 < 기대치의 80%")
                if reasons:
                    if audit:
                        rejected.append({
                            "제외_규칙": item.rule.formula,
                            "제외_선행조건": item.rule.activation(self.context),
                            "유지_규칙": previous.rule.formula,
                            "유지_선행조건": previous.rule.activation(self.context),
                            "사유": " / ".join(reasons),
                            "전수_상관": space_pair.correlation,
                            "전수_동시성립률": space_pair.joint,
                            "전수_동시기대율": space_pair.expected,
                            "전수_동시성립_기대_비율": space_pair.joint / space_pair.expected,
                            "가중이력_공통표본": history_pair.overlap,
                            "가중이력_동시성립률": history_pair.joint,
                            "가중이력_동시기대율": history_pair.expected,
                            "가중이력_동시성립_기대_비율": (
                                history_pair.joint / history_pair.expected if history_pair.expected else None
                            ),
                        })
                    break
            else:
                kept.append(item)
                variables[item.rule.variable] = variables.get(item.rule.variable, 0) + 1
                if len(kept) == limit:
                    break

        mean = fsum(item.adjusted_weight * item.baseline for item in kept)
        variance = fsum(
            item.adjusted_weight ** 2 * item.baseline * (1 - item.baseline) for item in kept
        ) + 2 * fsum(
            first.adjusted_weight * second.adjusted_weight * self._space_pair(first, second).covariance
            for index, first in enumerate(kept) for second in kept[:index]
        )
        return RuleModel(self.space, self.context, tuple(kept), rejected, mean, sqrt(max(0.0, variance)))


@dataclass
class RuleModel:
    space: ExactSpace
    context: object
    rules: tuple
    rejections: list
    mean: float
    stdev: float

    @staticmethod
    def _packed_indices(ids):
        groups, numbers = np.divmod(np.asarray(ids), NUMBER_SPACE)
        return groups, numbers >> 3, (1 << (numbers & 7)).astype(np.uint8)

    def _passes(self, item, packed_indices, shared_masks=None):
        if shared_masks is not None and item.mask_key in shared_masks:
            return shared_masks[item.mask_key]
        groups, offsets, bit_flags = packed_indices
        packed = self.space.mask(item, self.context).packed
        group_indices = 0 if len(packed) == 1 else groups
        passed = (packed[group_indices, offsets] & bit_flags) != 0
        if shared_masks is not None:
            shared_masks[item.mask_key] = passed
        return passed

    def passes(self, ids):
        if not self.rules:
            return np.empty((0, len(ids)), dtype=bool)
        indices = self._packed_indices(ids)
        return np.stack([self._passes(item, indices) for item in self.rules])

    def score(self, ids, shared_masks=None):
        score = np.zeros(len(ids), dtype=np.float64)
        if not self.stdev:
            return score
        indices = self._packed_indices(ids)
        for item in self.rules:
            score += item.adjusted_weight * self._passes(item, indices, shared_masks)
        score -= self.mean
        score /= self.stdev
        return score

    def full_score(self):
        """Group-major scores over all candidates in their canonical order."""
        if not self.stdev:
            return np.zeros(CANDIDATE_SPACE, dtype=np.float64)
        common = np.zeros(NUMBER_SPACE, dtype=np.float64)
        group_specific = np.zeros((GROUP_SPACE, NUMBER_SPACE), dtype=np.float64)
        for item in self.rules:
            packed = self.space.mask(item, self.context).packed
            passed = np.unpackbits(packed, axis=1, bitorder="little")
            if len(packed) == 1:
                common += item.adjusted_weight * passed[0]
            else:
                group_specific += item.adjusted_weight * passed
        group_specific += common
        group_specific -= self.mean
        group_specific /= self.stdev
        return group_specific.ravel()


# 다양성 제약

SELECTION_NUMBERS = 1_000_000
SELECTION_GROUPS = 5
SELECTION_SIZE = 15
SELECTION_POWERS = np.array([100_000, 10_000, 1_000, 100, 10, 1], dtype=np.int32)


def candidate_group_and_number(candidate_id):
    group, number = divmod(int(candidate_id), SELECTION_NUMBERS)
    return group + 1, number


def _printed_score(score):
    return float(f"{score:.8f}")


def _next_distinct_score_rank(order, scores, start, previous):
    """Skip an entire band that would print as the previous selected score."""
    if start == len(order) or _printed_score(scores[order[start]]) < previous:
        return start
    lo, hi = start + 1, len(order)
    while lo < hi:
        middle = (lo + hi) // 2
        if _printed_score(scores[order[middle]]) < previous:
            hi = middle
        else:
            lo = middle + 1
    return lo


def _next_candidate_rank(order, scores, start, chosen):
    if chosen:
        start = _next_distinct_score_rank(order, scores, start, _printed_score(scores[chosen[-1]]))
    count = len(chosen)
    slots = SELECTION_SIZE - count
    prior = np.asarray(chosen, dtype=np.int32)
    old_groups, old_numbers = np.divmod(prior, SELECTION_NUMBERS)
    old_suffixes = old_numbers % 100
    old_digits = (old_numbers[:, None] // SELECTION_POWERS % 10).astype(np.uint8)
    position_counts = np.array([np.bincount(col, minlength=10) for col in old_digits.T])
    tail_counts = position_counts[5]
    group_counts = np.bincount(old_groups, minlength=SELECTION_GROUPS)
    allowed_tails = (tail_counts == 0) if slots == np.count_nonzero(tail_counts == 0) else tail_counts < 2
    allowed_groups = (group_counts == 0) if slots == np.count_nonzero(group_counts == 0) else group_counts < 4
    allowed_suffixes = np.ones(100, dtype=bool)
    allowed_suffixes[old_suffixes] = False
    allowed_digits = position_counts[:5] < 4
    for lo in range(start, len(order), 8_192):
        ids = order[lo:lo + 8_192]
        groups, numbers = np.divmod(ids, SELECTION_NUMBERS)
        offsets = np.flatnonzero(allowed_groups[groups] & allowed_tails[numbers % 10] & allowed_suffixes[numbers % 100])
        if not offsets.size:
            continue
        digits = (numbers[offsets, None] // SELECTION_POWERS % 10).astype(np.uint8)
        keep = allowed_digits[np.arange(5), digits[:, :5]].all(axis=1)
        for other in old_digits:
            keep &= np.count_nonzero(digits == other, axis=1) <= 3
        eligible = offsets[keep]
        if eligible.size:
            return lo + int(eligible[0])
    return None


def select_candidates(scores):
    """Take the next admissible score in descending order until 15 are filled."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (SELECTION_GROUPS * SELECTION_NUMBERS,) or not np.isfinite(scores).all():
        raise ValueError("유한한 후보 점수 500만 개가 필요합니다.")
    order = np.argsort(-scores, kind="stable").astype(np.int32)
    last_band = 0
    for _ in range(SELECTION_SIZE - 1):
        previous = _printed_score(scores[order[last_band]])
        last_band = _next_distinct_score_rank(order, scores, last_band + 1, previous)
        if last_band == len(order):
            raise RuntimeError("소수점 8자리에서 서로 다른 후보 점수가 15개 미만입니다.")
    chosen = []
    ranks = []
    start = 0
    backtracks = 0
    while len(chosen) < SELECTION_SIZE:
        rank = _next_candidate_rank(order, scores, start, chosen)
        if rank is None:
            # Keep the earliest viable prefix when printed-score collisions exhaust its suffix.
            if not ranks or backtracks == 256:
                raise RuntimeError("소수점 8자리에서 서로 다른 점수로 선정 조건을 모두 만족하는 15개를 찾지 못했습니다.")
            start = ranks.pop() + 1
            chosen.pop()
            backtracks += 1
            continue
        chosen.append(int(order[rank]))
        ranks.append(rank)
        start = rank + 1
    validate_selection(chosen, scores)
    return chosen


def validate_selection(ids, scores):
    """Independently check every cap, the greedy coverage gates, and displayed ties."""
    if len(ids) != SELECTION_SIZE or len(set(ids)) != SELECTION_SIZE:
        raise RuntimeError("서로 다른 후보가 15개여야 합니다.")
    if any(candidate_id < 0 or candidate_id >= SELECTION_GROUPS * SELECTION_NUMBERS for candidate_id in ids):
        raise RuntimeError("후보가 500만 개 공간을 벗어났습니다.")

    decoded = [candidate_group_and_number(candidate_id) for candidate_id in ids]
    groups = [group for group, _ in decoded]
    numbers = [f"{number:06d}" for _, number in decoded]
    tails = [number[-1] for number in numbers]
    score_values = [float(scores[candidate_id]) for candidate_id in ids]
    if any(left <= right for left, right in zip(score_values, score_values[1:])):
        raise RuntimeError("선정 후보의 점수 내림차순을 확인하지 못했습니다.")
    if len({_printed_score(score) for score in score_values}) != SELECTION_SIZE:
        raise RuntimeError("소수점 8자리로 출력한 종합점수에 동점이 있습니다.")
    if len(set(number[-2:] for number in numbers)) != SELECTION_SIZE:
        raise RuntimeError("선정 번호의 끝 2자리가 겹칩니다.")
    if any(max(Counter(number[pos] for number in numbers).values()) > 4 for pos in range(5)):
        raise RuntimeError("앞 5자리의 자리별 숫자 제한을 초과했습니다.")
    if any(sum(x == y for x, y in zip(left, right)) > 3 for left, right in combinations(numbers, 2)):
        raise RuntimeError("두 번호의 같은 자리 숫자가 4개 이상입니다.")
    if len(set(tails)) != 10 or max(Counter(tails).values()) > 2:
        raise RuntimeError("끝자리 커버리지 또는 최대 2회 제한을 위반했습니다.")
    if len(set(groups)) != SELECTION_GROUPS or max(Counter(groups).values()) > 4:
        raise RuntimeError("조 커버리지 또는 최대 4회 제한을 위반했습니다.")

    seen_tails = set()
    seen_groups = set()
    for index, (tail, group) in enumerate(zip(tails, groups)):
        remaining = SELECTION_SIZE - index
        if remaining == 10 - len(seen_tails) and tail in seen_tails:
            raise RuntimeError("빠진 끝자리 우선 조건을 위반했습니다.")
        if remaining == SELECTION_GROUPS - len(seen_groups) and group in seen_groups:
            raise RuntimeError("빠진 조 우선 조건을 위반했습니다.")
        seen_tails.add(tail)
        seen_groups.add(group)


def feature_summary(number):
    digits = tuple(map(int, f"{int(number):06d}"))
    odd = sum(digit % 2 for digit in digits)
    high = sum(digit >= 5 for digit in digits)
    neighbors = sum(abs(left - right) == 1 for left, right in zip(digits, digits[1:]))
    return f"합 {sum(digits)} · 홀수 {odd} · 5 이상 {high} · 서로 다른 숫자 {len(set(digits))} · 인접 ±1 {neighbors}"


# 롤링 평가와 결과 저장

@dataclass(frozen=True)
class Setting:
    time_weights: tuple[float, float, float]
    threshold: float
    shrink: int
    limit: int

    def fit(self, fold, audit=False):
        return fold.fit(self.time_weights, self.threshold, self.shrink, self.limit, audit=audit)

    def label(self):
        time = '/'.join(f'{v:g}' for v in self.time_weights)
        return f'시간 가중치 {time} · 적중률 문턱 {self.threshold:.0%} · 수축 {self.shrink} · 규칙 상한 {self.limit}개'


SETTINGS = tuple(Setting(*args) for args in itertools.product(
    ((3.0, 2.0, 1.0), (2.0, 1.5, 1.0)), (0.55, 0.60), (20, 10), (40, 30)))
ROLLING_FOLDS = 40
RANK_SAMPLE = 200_000
TIE_SCALE = 0.000001


def candidate_parts(ids):
    ids = np.asarray(ids, dtype=np.int32)
    return (ids % N_NUMBERS)[:, None] // DIGIT_POWERS % 10, ids // N_NUMBERS


def frequency_logs(draws, index, time_weights):
    weights = np.full(index, time_weights[2], dtype=np.float64)
    weights[max(0, index - 100):] = time_weights[1]
    weights[max(0, index - 40):] = time_weights[0]

    def digit_logs(values):
        counts = np.stack([np.bincount(values[:index, pos], weights, minlength=10) for pos in range(6)]) + 1.0
        return np.log(counts / counts.sum(axis=1, keepdims=True))

    group = np.bincount(draws.groups[:index] - 1, weights, minlength=5) + 1.0
    return 0.8 * digit_logs(draws.main) + 0.2 * digit_logs(draws.bonus), np.log(group / group.sum())


def tie_values(digits, groups, logs):
    digit_log, group_log = logs
    return TIE_SCALE * (digit_log[np.arange(6), digits].sum(axis=1) + group_log[groups])


def score_rank_sample(draws, index):
    held_out = (int(draws.groups[index]) - 1) * N_NUMBERS + int(draws.main[index] @ DIGIT_POWERS)
    rng = np.random.default_rng(np.random.SeedSequence([72015, int(draws.rounds[index])]))
    sample = rng.choice(N_CANDIDATES - 1, size=RANK_SAMPLE, replace=False, shuffle=False).astype(np.int32)
    sample += sample >= held_out
    return np.concatenate((sample, np.array([held_out], dtype=np.int32)))


def tune_settings(draws, catalogs, space):
    averages = np.zeros(len(SETTINGS))
    for ix in range(len(draws.rounds) - ROLLING_FOLDS, len(draws.rounds)):
        context = FeatureContext(draws, ix)
        folds = [RuleFold(catalog, context, space) for catalog in catalogs]
        ids = score_rank_sample(draws, ix)
        digits, groups = candidate_parts(ids)
        tie = {weights: tie_values(digits, groups, frequency_logs(draws, ix, weights))
               for weights in {setting.time_weights for setting in SETTINGS}}
        shared = [{}, {}]
        scores = [{}, {}]
        for n, setting in enumerate(SETTINGS):
            components = []
            for stream, fold in enumerate(folds):
                model = setting.fit(fold)
                key = tuple((row.rule.key, row.adjusted_weight) for row in model.rules)
                if key not in scores[stream]:
                    scores[stream][key] = model.score(ids, shared_masks=shared[stream])
                components.append(scores[stream][key])
            joint = 0.8 * components[0] + 0.2 * components[1] + tie[setting.time_weights]
            averages[n] += 1 + np.count_nonzero(joint[:-1] > joint[-1]) + 0.5 * np.count_nonzero(joint[:-1] == joint[-1])
        completed = ix - (len(draws.rounds) - ROLLING_FOLDS) + 1
        if completed % 10 == 0:
            print(f'롤링 검증 {completed}/{ROLLING_FOLDS}회 완료', file=sys.stderr, flush=True)
    return SETTINGS[int(np.argmin(averages))]


def score_space(draws, setting, models):
    components = np.empty((2, N_CANDIDATES), dtype=np.float64)
    scores = np.empty(N_CANDIDATES, dtype=np.float64)
    for i, model in enumerate(models):
        components[i] = model.full_score()
    logs = frequency_logs(draws, len(draws.rounds), setting.time_weights)
    for start in range(0, N_CANDIDATES, 250_000):
        stop = min(start + 250_000, N_CANDIDATES)
        ids = np.arange(start, stop, dtype=np.int32)
        digits, groups = candidate_parts(ids)
        scores[start:stop] = 0.8 * components[0, start:stop] + 0.2 * components[1, start:stop] + tie_values(digits, groups, logs)
    return scores, components


def csv_content(rows, columns):
    from io import StringIO

    out = StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(columns)
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def md_table(columns, rows):
    def cell(v):
        return str(v).replace('|', r'\|').replace('\n', ' ')

    return '\n'.join('|' + '|'.join(f' {cell(v)} ' for v in row) + '|'
                     for row in (columns, ['---'] * len(columns), *rows))


def default_workbook():
    for folder in dict.fromkeys((Path.cwd(), Path(__file__).resolve().parent)):
        matches = sorted(p for p in folder.glob('*.xlsx') if not p.name.startswith('~$'))
        if not matches:
            matches = sorted(p for p in (folder / '.codex-attachments').rglob('*.xlsx') if not p.name.startswith('~$'))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError('엑셀 파일이 여러 개입니다. 실행할 때 파일 경로를 인자로 지정하세요.')
    raise ValueError('엑셀 파일을 스크립트 옆에 두거나 경로를 인자로 지정하세요.')


CANDIDATE_COLUMNS = ('순위', '조', '6자리', '종합점수 S', '1등 규칙 점수', '보너스 점수', '충족 규칙 수', '특징 요약')
RULE_COLUMNS = ('구분', '유형', '수식', '목표 활성 근거', '전체 적중률', '가중 적중률', '최근 100회 적중률', '최근 40회 적중률',
                '표본 수 전체/최근100/최근40', '가중 표본 수', '기저 통과율', '가중치 w', '자리 배율')


def result_rows(selected, scores, components, models):
    selected = np.asarray(selected, dtype=np.int32)
    passes = [model.passes(selected).sum(axis=0) for model in models]
    for rank, id_ in enumerate(selected, 1):
        group, number = divmod(int(id_), N_NUMBERS)
        n_main, n_bonus = (int(arr[rank - 1]) for arr in passes)
        yield (rank, group + 1, f'{number:06d}', f'{scores[id_]:.8f}', f'{components[0, id_]:.8f}',
               f'{components[1, id_]:.8f}', n_main + n_bonus,
               f'1등 {n_main}·보너스 {n_bonus} / {feature_summary(number)}')


def rule_rows(context, models):
    for stream, model in zip(('1등', '보너스'), models):
        for row in model.rules:
            rule = row.rule
            a, b, c, d = row.counts
            yield (stream, rule.kind, rule.formula, rule.activation(context),
                   *(f'{rate:.2%}' for rate in row.rates), f'{a:g}/{c:g}/{d:g}', f'{b:g}',
                   f'{row.baseline:.4%}', f'{row.weight:.8f}', f'{rule.multiplier:.1f}')


def bundle_result(out, xlsx, draws, context, setting, models, selected, scores, components):
    target_round = int(draws.rounds[-1]) + 1
    target_date = draws.dates[-1] + timedelta(days=7)
    stem = f'연금복권720_{target_round}회'
    title = f'연금복권720+ {target_round}회 · {target_date.isoformat()}'
    candidates = list(result_rows(selected, scores, components, models))
    rules = list(rule_rows(context, models))
    rejected = [{'구분': stream, **item} for stream, model in zip(('1등', '보너스'), models) for item in model.rejections]
    reject_columns = tuple(dict.fromkeys(key for item in rejected for key in item)) if rejected else ('구분', '수식', '사유')
    counts = f'1등 {len(models[0].rules)}개 · 보너스 {len(models[1].rules)}개'
    rule_floor = (f'채택 기준을 통과한 양립 규칙: {counts}. 기준을 유지해 30개 미만인 모델은 통과분만 사용했습니다.\n\n'
                  if min(len(model.rules) for model in models) < 30 else '')
    report = (f'# {title}\n\n'
              f'{setting.label()}\n\n'
              f'5,000,000개 채점 · 15개 선정 · 소수 8자리 동점 0개 · 적용 규칙 {counts}\n\n'
              f'{rule_floor}'
              f'## 표1. 후보 15개\n\n{md_table(CANDIDATE_COLUMNS, candidates)}\n\n'
              f'## 표2. 적용 규칙\n\n{md_table(RULE_COLUMNS, rules)}\n\n'
              '재실행: 압축을 푼 폴더에서 `python predict_pension15.py` (Python 3.10+, NumPy). '
              '엑셀의 `당첨번호` 시트에 새 회차를 추가하면 다음 회차로 이동합니다.\n')
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f'{stem}.zip'
    temporary = destination.with_suffix('.zip.tmp')
    try:
        with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
            archive.write(Path(__file__).resolve(), 'predict_pension15.py')
            archive.write(xlsx, xlsx.name)
            archive.writestr('결과.md', report.encode('utf-8'))
            archive.writestr('표1_후보15.csv', csv_content(candidates, CANDIDATE_COLUMNS))
            archive.writestr('표2_적용규칙.csv', csv_content(rules, RULE_COLUMNS))
            archive.writestr('제외_중복_모순.csv', csv_content(([row.get(k, '') for k in reject_columns] for row in rejected), reject_columns))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(title)
    print(setting.label())
    print(f'5,000,000개 채점 · 15개 선정 · 동점 0개 · 규칙 {counts}')
    print(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description='연금복권720+ 다음 회차 패턴 후보 15개')
    parser.add_argument('xlsx', nargs='?', type=Path, help='입력 엑셀 (생략하면 스크립트 옆에서 검색)')
    parser.add_argument('--out', type=Path, default=Path.cwd() / 'outputs', help='ZIP 저장 폴더')
    args = parser.parse_args()
    start = perf_counter()
    xlsx = args.xlsx or default_workbook()
    draws = load_draws(xlsx)
    print(f'{len(draws.rounds)}회 입력 확인. 규칙 목록 준비 중.', file=sys.stderr, flush=True)
    catalogs = [build_catalog(draws, stream) for stream in range(2)]
    space = ExactSpace()
    setting = tune_settings(draws, catalogs, space)
    context = FeatureContext(draws, len(draws.rounds))
    models = [setting.fit(RuleFold(catalog, context, space), audit=True) for catalog in catalogs]
    print('목표 회차 5,000,000개 채점 중.', file=sys.stderr, flush=True)
    scores, components = score_space(draws, setting, models)
    selected = select_candidates(scores)
    validate_selection(selected, scores)
    bundle_result(args.out, xlsx, draws, context, setting, models, selected, scores, components)
    print(f'실행 시간 {perf_counter() - start:.1f}초', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f'실행 오류: {error}') from error
