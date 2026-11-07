#!/usr/bin/env python3
"""연금복권720+ 외부 지표·달력·다른 복권 공식 실험실.

lab.py에 이미 있는 것(원·달러 등 6개 매크로의 1·4·13주 방향과 크기, 52주 수준, 주간 RSI, 이평 교차,
회차 나머지 2~7, 계절, 월의 몇째 주, 달 위상, 복권 자체 국면)은 다시 만들지 않는다.

공식 가족
- 숫자 겨냥: 외부 값에서 6자리 숫자(와 조)를 뽑아 후보 숫자와 비교한다
  · 로또 6/45: 추첨 전 토요일 번호의 끝자리·십의 자리·이어 쓴 글자·곱·간격, 2주 전 로또, 1등 당첨금, 당첨자 수
  · 날짜 수비학: yymmdd, ddmmyy, 날짜 서수, 연중 날짜·주, 날짜 제곱
  · 음력·간지: 음력 yymmdd, ddmmyy, 년·월·일의 천간과 지지
  · 천체: 달 거리, 달 황경, 해 황경, 달 밝기, 서울 일출·일몰 시각
  · 매크로 값의 숫자: 원·달러, 엔, 위안, 유로, 유가, 비트코인, VIX, 천연가스, 대만달러, 홍콩달러, 파운드, 호주달러
  · 연금복권 직전 회차 6·7등 당첨자 수 (판매량)
  채점 네 가지: 같은 자리 일치 수, 일치 피하기, 과거에 자주 나온 '겨냥 숫자와의 차이'(자리별), 겨냥 숫자별 조건부 빈도
- 국면 빈도: 외부 값으로 나눈 국면에서 같은 국면 과거 회차의 자리별 숫자·조 빈도 (lab.conditional)
  · 로또 합·홀수·최대·최소·연번·보너스·1등 당첨자 수·판매 방향, 로또 끝자리와 직전 연금 번호 겹침
  · 간지 천간·지지, 음력 계절, 24절기 안 날짜, 해 황경 8분면, 공휴일까지·부터, 추첨 주 공휴일, 설·추석까지
  · 달 거리·적위·별자리, 추첨 시각 달 높이, 수성·금성 역행, 화성 별자리, 일·월식 철
  · 다른 나라 환율 17개 1주 방향, 교차 환율, 비트코인, VIX, 천연가스, 지구 기온, 원·달러 변동성
- 로또 번호 → (자리, 숫자) 지도 일치, 로또 끝자리 집합에 든 숫자 개수

자료
- data/external.csv: 날짜별 넓은 표. 다른 나라 환율, VIX, 천연가스, 비트코인, 지구 기온(월), 로또 6/45, 연금복권 당첨자 수.
  update_external()이 GitHub 공개 저장소에서 새로 받아 합친다. 받기 실패한 출처는 기존 값을 그대로 둔다.
- data/calendar_ext.csv: 2020~2030년 목요일마다 음력·간지·천체·공휴일 값. build_calendar()로 다시 만든다
  (ephem, korean-lunar-calendar, holidays 필요). 주간 실행은 만들어 둔 표만 읽으므로 추가 패키지가 필요 없다.

모든 외부 값은 추첨일 전날까지 날짜가 찍힌 것만 쓴다. 월간 값은 그 달 1일 날짜라서 35일 늦춰 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from . import formula_search as fsm
    from . import formulas as fm
    from . import lab
    from . import predict_pension15 as eng
except ImportError:
    sys.path.insert(0, str(HERE))
    import formula_search as fsm
    import formulas as fm
    import lab
    import predict_pension15 as eng

DATA = HERE / 'data'
EXT_CSV = DATA / 'external.csv'
CAL_CSV = DATA / 'calendar_ext.csv'
START = '2019-01-01'
POS = np.arange(6)

FX = {'Australia': 'fx_aud', 'Canada': 'fx_cad', 'Switzerland': 'fx_chf', 'United Kingdom': 'fx_gbp',
      'Sweden': 'fx_sek', 'Norway': 'fx_nok', 'New Zealand': 'fx_nzd', 'Malaysia': 'fx_myr', 'Denmark': 'fx_dkk',
      'India': 'fx_inr', 'South Africa': 'fx_zar', 'Thailand': 'fx_thb', 'Singapore': 'fx_sgd',
      'Hong Kong': 'fx_hkd', 'Taiwan': 'fx_twd', 'Mexico': 'fx_mxn', 'Brazil': 'fx_brl'}
FX_NAMES = {'fx_aud': '호주달러', 'fx_cad': '캐나다달러', 'fx_chf': '스위스프랑', 'fx_gbp': '파운드',
            'fx_sek': '스웨덴크로나', 'fx_nok': '노르웨이크로네', 'fx_nzd': '뉴질랜드달러', 'fx_myr': '링깃',
            'fx_dkk': '덴마크크로네', 'fx_inr': '루피', 'fx_zar': '랜드', 'fx_thb': '바트', 'fx_sgd': '싱가포르달러',
            'fx_hkd': '홍콩달러', 'fx_twd': '대만달러', 'fx_mxn': '페소', 'fx_brl': '헤알'}
LOTTO = ['lotto_n1', 'lotto_n2', 'lotto_n3', 'lotto_n4', 'lotto_n5', 'lotto_n6', 'lotto_bonus']
COLUMNS = (list(FX.values()) + ['vix', 'natgas', 'btc', 'gtemp'] + LOTTO
           + ['lotto_w1', 'lotto_p1', 'lotto_w4', 'lotto_w5', 'pen_w1', 'pen_w6', 'pen_w7', 'pen_wb'])
MONTHLY = {'gtemp'}
RAW = 'https://raw.githubusercontent.com/'
SOURCES = {
    'exchange-rates': RAW + 'datasets/exchange-rates/main/data/daily.csv',
    'vix': RAW + 'datasets/finance-vix/main/data/vix-daily.csv',
    'natgas': RAW + 'datasets/natural-gas/main/data/daily.csv',
    'btc': RAW + 'rcaljouw/btc-price-history/main/data/btc-usd-daily.csv',
    'gtemp': RAW + 'datasets/global-temp/main/data/monthly.csv',
    'nostra': RAW + 'nostra-team/nostra-data/main/history.json',
}


# ---------------------------------------------------------------- 자료 받기·읽기

def _fetch(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read().decode('utf-8')


def _parse(name, text):
    if name == 'nostra':
        h = json.loads(text)
        for r in h.get('lotto', []):
            d = r['date']
            d = f'{d[:4]}-{d[4:6]}-{d[6:8]}'
            for col, v in zip(LOTTO, list(r['numbers']) + [r['bonus']]):
                yield d, col, v
            p = r.get('prizes') or []
            if len(p) >= 5:
                yield d, 'lotto_w1', p[0][0]
                yield d, 'lotto_p1', p[0][1]
                yield d, 'lotto_w4', p[3][0]
                yield d, 'lotto_w5', p[4][0]
        for r in h.get('pension', []):
            d = r['date']
            d = f'{d[:4]}-{d[4:6]}-{d[6:8]}'
            p = r.get('prizes') or []
            if len(p) >= 8:
                for col, k in (('pen_w1', 0), ('pen_w6', 5), ('pen_w7', 6), ('pen_wb', 7)):
                    yield d, col, p[k][0]
        return
    table = list(csv.reader(io.StringIO(text)))[1:]
    if name == 'exchange-rates':
        for d, c, v in table:
            if c in FX:
                yield d, FX[c], v
    elif name == 'vix':
        for r in table:
            yield r[0], 'vix', r[4]
    elif name == 'natgas':
        for d, v in table:
            yield d, 'natgas', v
    elif name == 'btc':
        for r in table:
            yield r[0], 'btc', r[4]
    elif name == 'gtemp':
        for r in table:
            if r[0] == 'GCAG':
                yield r[1] + '-01', 'gtemp', r[2]


def _read_rows(path):
    rows = defaultdict(dict)
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                d = r.pop('날짜')
                rows[d].update({k: v for k, v in r.items() if v not in ('', None)})
    return rows


def update_external(path=EXT_CSV, fetch=_fetch):
    """GitHub 공개 저장소에서 외부 값을 받아 external.csv에 합친다. 실패한 출처는 기존 값 그대로. 실패 목록을 돌려준다."""
    rows = _read_rows(path)
    errors = []
    for name, url in SOURCES.items():
        try:
            got = [(d, c, v) for d, c, v in _parse(name, fetch(url)) if d >= START and str(v).strip() not in ('', '.')]
            if not got:
                raise ValueError('빈 자료')
            for d, c, v in got:
                rows[d][c] = f'{float(v):.12g}'
        except Exception as error:   # 한 출처가 막혀도 나머지로 진행
            errors.append(f'{name}: {type(error).__name__} {error}')
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['날짜'] + COLUMNS)
        for d in sorted(rows):
            if any(c in rows[d] for c in COLUMNS):
                w.writerow([d] + [rows[d].get(c, '') for c in COLUMNS])
    os.replace(tmp, path)
    return errors


def load_external(path=EXT_CSV):
    """지표 → (날짜 서수 배열, 값 배열, 소수 자릿수 최빈값)."""
    data, decimals = defaultdict(list), defaultdict(Counter)
    for d, r in sorted(_read_rows(path).items()):
        day = date.fromisoformat(d).toordinal()
        for k, v in r.items():
            data[k].append((day, float(v)))
            decimals[k][len(v.split('.')[1]) if '.' in v else 0] += 1
    return {k: (np.array([a for a, _ in v]), np.array([b for _, b in v]), decimals[k].most_common(1)[0][0])
            for k, v in data.items()}


STEMS = '갑을병정무기경신임계'
BRANCHES = '자축인묘진사오미신유술해'
CAL_COLUMNS = ('음력년', '음력월', '음력일', '윤달', '년천간', '년지지', '월천간', '월지지', '일천간', '일지지',
               '해황경', '달황경', '달적위', '달거리', '달밝기', '달고도', '일출', '일몰', '수성역행', '금성역행',
               '화성황경', '식철', '공휴일까지', '공휴일부터', '주중공휴일', '30일공휴일수', '명절까지')


def build_calendar(path=CAL_CSV, start=date(2020, 5, 7), end=date(2030, 12, 31)):
    """목요일마다 음력·간지·천체·공휴일 값을 계산해 저장한다. 날짜만으로 정해지는 값이라 미래 날짜도 미리 만든다."""
    import ephem
    import holidays
    from korean_lunar_calendar import KoreanLunarCalendar

    hol = holidays.KR(years=range(start.year - 1, end.year + 2))
    all_days = sorted(hol)
    planned = [d for d in all_days if not any(w in hol[d] for w in ('임시', '선거', '국군'))]   # 미리 정해진 공휴일
    festival = sorted(d for d in all_days if {'설날', '추석'} & set(hol[d].split('; ')))
    seoul = ephem.Observer()
    seoul.lat, seoul.lon, seoul.elevation = '37.5665', '126.9780', 38
    seoul.pressure = 0

    def lon_of(body, t):
        body.compute(t, epoch=t)
        return float(ephem.Ecliptic(body, epoch=t).lon) * 180 / np.pi

    def retro(body, t):
        a, b = lon_of(body, ephem.Date(t - 1)), lon_of(body, t)
        return int(((b - a + 180) % 360 - 180) < 0)

    def kst(t):
        x = ephem.Date(t).datetime() + timedelta(hours=9)
        return x.hour * 10000 + x.minute * 100 + x.second

    out = []
    d = start
    while d <= end:
        c = KoreanLunarCalendar()
        c.setSolarDate(d.year, d.month, d.day)
        gz = c.getGapJaString().split()
        t = ephem.Date(datetime(d.year, d.month, d.day, 10, 5))   # 추첨 19:05 KST
        moon = ephem.Moon()
        moon.compute(t, epoch=t)
        sun_lon, moon_lon = lon_of(ephem.Sun(), t), lon_of(ephem.Moon(), t)
        seoul.date = t
        topo = ephem.Moon(seoul)
        seoul.date = ephem.Date(datetime(d.year, d.month, d.day) - timedelta(hours=9))
        rise, sett = seoul.next_rising(ephem.Sun()), seoul.next_setting(ephem.Sun())
        jc = (float(t) + 2415020.0 - 2451545.0) / 36525   # ephem 날짜는 1899-12-31 12시 기준
        node = (125.04452 - 1934.136261 * jc) % 360
        gap = (sun_lon - node) % 180
        nxt = next(h for h in planned if h >= d)
        prv = max(h for h in all_days if h <= d)
        monday = d - timedelta(days=d.weekday())
        out.append([d.isoformat(), c.lunarYear, c.lunarMonth, c.lunarDay, int(c.isIntercalation),
                    STEMS.index(gz[0][0]), BRANCHES.index(gz[0][1]), STEMS.index(gz[1][0]), BRANCHES.index(gz[1][1]),
                    STEMS.index(gz[2][0]), BRANCHES.index(gz[2][1]),
                    f'{sun_lon:.4f}', f'{moon_lon:.4f}', f'{float(moon.dec) * 180 / np.pi:.4f}',
                    f'{moon.earth_distance * 149597870.7:.1f}', f'{moon.moon_phase:.5f}',
                    f'{float(topo.alt) * 180 / np.pi:.3f}', kst(rise), kst(sett),
                    retro(ephem.Mercury(), t), retro(ephem.Venus(), t), f'{lon_of(ephem.Mars(), t):.3f}',
                    int(min(gap, 180 - gap) < 17), (nxt - d).days, (d - prv).days,
                    int(any(monday <= h <= monday + timedelta(days=6) for h in all_days)),
                    sum(1 for h in all_days if d - timedelta(days=30) <= h < d),
                    (next(h for h in festival if h >= d) - d).days])
        d += timedelta(days=7)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(('날짜',) + CAL_COLUMNS)
        w.writerows(out)
    return len(out)


def load_calendar(path=CAL_CSV):
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as f:
        return {r['날짜']: {k: float(v) for k, v in r.items() if k != '날짜'} for r in csv.DictReader(f)}


# ---------------------------------------------------------------- 행별 값 (행 0..n, n은 목표 회차)

def _at(series, days, lag=1, offset=0, stale=10):
    """days - offset - lag 이전에 이미 나와 있던 마지막 값. stale일보다 오래된 값이면 모름(nan)."""
    if series is None:
        return np.full(len(days), np.nan)
    sd, sv = series[0], series[1]
    target = days - offset
    i = np.searchsorted(sd, target - lag, side='right') - 1
    v = np.where(i >= 0, sv[np.maximum(i, 0)], np.nan)
    age = target - sd[np.maximum(i, 0)]
    v[(i < 0) | (age > stale + lag)] = np.nan
    return v


def digits6(x):
    """정수 값의 뒤 6자리 숫자 (n, 6). 모르면 -1."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    v = np.where(ok, np.abs(np.nan_to_num(x)), 0).astype(np.int64) % 1_000_000
    out = (v[:, None] // eng.POWERS.astype(np.int64)) % 10
    out[~ok] = -1
    return out.astype(np.int64)


def _cal(cal, dates, key):
    return np.array([cal.get(d.isoformat(), {}).get(key, np.nan) for d in dates], dtype=float)


def _ival(x, mod=None):
    out = np.where(np.isfinite(x), np.nan_to_num(x), -1).astype(np.int64)
    if mod:
        out = np.where(out >= 0, out % mod, -1)
    return out


def _bins(x, edges):
    out = np.digitize(np.nan_to_num(x, nan=0.0), edges)
    out[~np.isfinite(x)] = -1
    return out


class Rows:
    """행마다 쓰는 외부 값 묶음."""

    def __init__(self, draws, macro, ext, cal):
        self.dates = lab.draw_dates(draws)
        self.day = np.array([d.toordinal() for d in self.dates])
        self.n1 = len(self.dates)
        self.ext, self.cal, self.macro = ext, cal, macro
        lotto = np.stack([_at(ext.get(c), self.day, 1, 0, 8) for c in LOTTO], 1)
        prev = np.stack([_at(ext.get(c), self.day, 1, 7, 8) for c in LOTTO], 1)
        self.lotto_ok = np.isfinite(lotto).all(1)
        self.lotto = np.where(self.lotto_ok[:, None], np.nan_to_num(lotto), 0).astype(np.int64)
        self.prev_ok = np.isfinite(prev).all(1)
        self.prev = np.where(self.prev_ok[:, None], np.nan_to_num(prev), 0).astype(np.int64)

    def series(self, name, offset=0, stale=16):
        """일간 값은 전날까지, 16일보다 오래되면 모름. 월간 값은 35일 늦추고 100일보다 오래되면 모름."""
        if name in self.macro:
            s = self.macro[name]
            lag, stale = (35, 100) if name in lab.MONTHLY else (1, stale)
        else:
            s = self.ext.get(name)
            lag, stale = (35, 100) if name in MONTHLY else (1, stale)
        return _at(s, self.day, lag, offset, stale)

    def scale(self, name):
        if name in self.macro:
            return {'usdcny': 4, 'usdeur': 4}.get(name, 2)
        return self.ext[name][2] if name in self.ext else 2

    def cal_col(self, key):
        return _cal(self.cal, self.dates, key)


def _vec(cols, ok=None):
    t = np.stack([np.asarray(c, dtype=np.int64) for c in cols], 1) % 10
    if ok is not None:
        t[~ok] = -1
    return t


def build_targets(draws, macro, ext, cal):
    """이름 → (6자리 겨냥 숫자 (n+1, 6), 겨냥 조 0~4 (n+1,)). 모르면 -1."""
    R = Rows(draws, macro, ext, cal)
    L, ok = R.lotto, R.lotto_ok
    main6, bonus = L[:, :6], L[:, 6]
    out = {}
    out['로또 끝자리'] = (_vec(main6.T, ok), np.where(ok, bonus % 5, -1))
    out['로또 십의 자리'] = (_vec((main6 // 10).T, ok), np.where(ok, main6.sum(1) % 5, -1))
    out['로또 앞 세 번호 글자'] = (_vec([main6[:, 0] // 10, main6[:, 0], main6[:, 1] // 10, main6[:, 1],
                                  main6[:, 2] // 10, main6[:, 2]], ok), np.where(ok, main6[:, 0] % 5, -1))
    out['로또 뒤 세 번호 글자'] = (_vec([main6[:, 3] // 10, main6[:, 3], main6[:, 4] // 10, main6[:, 4],
                                  main6[:, 5] // 10, main6[:, 5]], ok), np.where(ok, main6[:, 5] % 5, -1))
    prod = np.ones(R.n1, dtype=np.int64)
    for j in range(6):
        prod = prod * main6[:, j] % 1_000_000
    out['로또 번호 곱'] = (np.where(ok[:, None], digits6(prod), -1), np.where(ok, prod % 5, -1))
    gaps = np.diff(main6, axis=1)
    out['로또 번호 간격'] = (_vec(list(gaps.T) + [bonus], ok), np.where(ok, (main6[:, 5] - main6[:, 0]) % 5, -1))
    P, pok = R.prev, R.prev_ok
    out['2주 전 로또 끝자리'] = (_vec(P[:, :6].T, pok), np.where(pok, P[:, 6] % 5, -1))
    p1, w1 = R.series('lotto_p1', stale=8), R.series('lotto_w1', stale=8)
    head = np.array([int(str(int(v))[:6]) if np.isfinite(v) and v >= 1e5 else -1 for v in p1], dtype=float)
    head[head < 0] = np.nan
    gw1 = _ival(w1, 5)
    out['로또 1등 당첨금 앞 6자리'] = (digits6(head), gw1)
    out['로또 1등 당첨금 뒤 6자리'] = (digits6(p1), gw1)
    out['로또 5등 당첨자 수'] = (digits6(R.series('lotto_w5', stale=8)), _ival(R.series('lotto_w4', stale=8), 5))
    out['로또 4등 당첨자 수'] = (digits6(R.series('lotto_w4', stale=8)), _ival(R.series('lotto_w5', stale=8), 5))

    yy = np.array([d.year % 100 for d in R.dates])
    mm = np.array([d.month for d in R.dates])
    dd = np.array([d.day for d in R.dates])
    out['날짜 yymmdd'] = (_vec([yy // 10, yy, mm // 10, mm, dd // 10, dd]), dd % 5)
    out['날짜 ddmmyy'] = (_vec([dd // 10, dd, mm // 10, mm, yy // 10, yy]), mm % 5)
    out['날짜 서수'] = (digits6(R.day), R.day % 5)
    doy = np.array([d.timetuple().tm_yday for d in R.dates])
    wk = np.array([d.isocalendar()[1] for d in R.dates])
    out['연중 날짜·주'] = (_vec([doy // 100, doy // 10, doy, wk // 10, wk, yy]), wk % 5)
    ymd = np.array([d.year * 10000 + d.month * 100 + d.day for d in R.dates], dtype=np.int64)
    out['날짜 제곱 뒤 6자리'] = (digits6(ymd * ymd % 1_000_000), ymd * ymd % 5)

    if cal:
        ly, lm, ld = (_ival(R.cal_col(k)) for k in ('음력년', '음력월', '음력일'))
        cok = ly >= 0
        out['음력 yymmdd'] = (_vec([ly // 10, ly, lm // 10, lm, ld // 10, ld], cok), np.where(cok, ld % 5, -1))
        out['음력 ddmmyy'] = (_vec([ld // 10, ld, lm // 10, lm, ly // 10, ly], cok), np.where(cok, lm % 5, -1))
        gz = [_ival(R.cal_col(k)) for k in ('년천간', '월천간', '일천간', '년지지', '월지지', '일지지')]
        out['간지 천간·지지'] = (_vec(gz, cok), np.where(cok, gz[5] % 5, -1))
        dist = R.cal_col('달거리')
        out['달 거리 km'] = (digits6(dist), _ival(dist // 1000, 5))
        mlon = R.cal_col('달황경')
        out['달 황경'] = (digits6(mlon * 1000), _ival(mlon // 30, 5))
        slon = R.cal_col('해황경')
        out['해 황경'] = (digits6(slon * 1000), _ival(slon // 15, 5))
        illum = R.cal_col('달밝기')
        out['달 밝기'] = (digits6(illum * 1e6), _ival(illum * 10, 5))
        sset, srise = R.cal_col('일몰'), R.cal_col('일출')
        out['서울 일몰 시각'] = (digits6(sset), _ival(sset // 100, 5))
        out['서울 일출 시각'] = (digits6(srise), _ival(srise // 100, 5))

    names = {'usdkrw': '원·달러', 'usdjpy': '엔·달러', 'usdcny': '위안·달러', 'usdeur': '유로·달러', 'wti': 'WTI',
             'brent': '브렌트', 'btc': '비트코인', 'vix': 'VIX', 'natgas': '천연가스', 'fx_twd': '대만달러',
             'fx_hkd': '홍콩달러', 'fx_gbp': '파운드', 'fx_aud': '호주달러'}
    for key, label in names.items():
        if key not in macro and key not in ext:
            continue
        v = np.round(R.series(key) * 10 ** R.scale(key))
        out[f'{label} 값 숫자'] = (digits6(v), _ival(v, 5))
    w7, w6 = R.series('pen_w7', stale=8), R.series('pen_w6', stale=8)
    if np.isfinite(w7).any():
        out['연금 직전 7등 당첨자 수'] = (digits6(w7), _ival(R.series('pen_w1', stale=8), 5))
        out['연금 직전 6등 당첨자 수'] = (digits6(w6), _ival(R.series('pen_wb', stale=8), 5))
    if cal:   # 음력으로 같은 달·가까운 날이었던 지난해(2년 전) 회차의 1등 번호. 그 회차는 항상 목표보다 약 50회 앞
        index = {d.isoformat(): i for i, d in enumerate(R.dates)}
        lunar = defaultdict(list)
        for iso, row in cal.items():
            lunar[(int(row['음력년']), int(row['음력월']), int(row['윤달']))].append((row['음력일'], iso))
        main, groups = draws.main.astype(np.int64), draws.groups.astype(np.int64) - 1
        for back, word in ((1, '지난해'), (2, '2년 전')):
            t, tg = np.full((R.n1, 6), -1, dtype=np.int64), np.full(R.n1, -1, dtype=np.int64)
            for i, day in enumerate(R.dates):
                row = cal.get(day.isoformat())
                if not row:
                    continue
                cands = lunar.get((int(row['음력년']) - back, int(row['음력월']), 0), [])
                best = min(cands, key=lambda c: abs(c[0] - row['음력일']), default=None)
                j = index.get(best[1]) if best else None
                if j is not None and j < i:
                    t[i], tg[i] = main[j], groups[j]
            out[f'음력 같은 날 {word} 1등'] = (t, tg)
    return out, R


def _change(R, name, weeks, offset_step=7):
    now = R.series(name)
    past = R.series(name, offset=offset_step * weeks)
    return now / past - 1


def build_states(draws, macro, ext, cal, R=None):
    """이름 → 행별 국면 번호 (n+1,). 모르면 -1."""
    R = R or Rows(draws, macro, ext, cal)
    L, ok = R.lotto, R.lotto_ok
    main6, bonus = L[:, :6], L[:, 6]
    st = {}

    def lot(x, edges):
        x = np.where(ok, x, np.nan).astype(float)
        return _bins(x, edges)

    st['로또 합'] = lot(main6.sum(1), [125, 152])
    st['로또 홀수 개수'] = lot((main6 % 2).sum(1), [3, 4])
    st['로또 23 이상 개수'] = lot((main6 >= 23).sum(1), [3, 4])
    st['로또 최댓값'] = lot(main6.max(1), [41, 44])
    st['로또 최솟값'] = lot(main6.min(1), [4, 9])
    st['로또 연번 유무'] = lot((np.diff(main6, axis=1) == 1).any(1), [1])
    st['로또 보너스 홀짝'] = lot(bonus % 2, [1])
    st['로또 보너스 mod 5'] = lot(bonus % 5, [1, 2, 3, 4])
    st['로또 끝자리 중복'] = lot(6 - np.array([len(set(r % 10)) for r in main6]), [1, 2])
    st['로또 합 1주 방향'] = lot(np.where(R.prev_ok, main6.sum(1) - R.prev[:, :6].sum(1), np.nan), [0])
    st['로또 1등 당첨자 수'] = _bins(R.series('lotto_w1', stale=8), [9, 14])
    w5 = R.series('lotto_w5', stale=8)
    st['로또 판매 1주 방향'] = _bins(w5 / R.series('lotto_w5', offset=7, stale=8) - 1, [0.0])
    prev_main = np.vstack([np.full((1, 6), -1), draws.main.astype(np.int64)])[:R.n1]   # 행 t의 직전 연금 1등
    overlap = np.array([len(set(m % 10) & set(p)) if p[0] >= 0 else -1 for m, p in zip(main6, prev_main)], float)
    overlap[overlap < 0] = np.nan
    st['로또 끝자리와 직전 연금 숫자 겹침'] = lot(overlap, [3, 4])

    yy = np.array([d.year for d in R.dates])
    dd = np.array([d.day for d in R.dates])
    mm = np.array([d.month for d in R.dates])
    st['날짜 홀짝'] = dd % 2
    st['날짜 mod 3'] = dd % 3
    st['월 홀짝'] = mm % 2
    root = np.array([sum(map(int, d.strftime('%Y%m%d'))) for d in R.dates])
    st['날짜 수비학 1~9'] = (root - 1) % 9
    if R.cal:
        c = R.cal_col
        st['간지 일천간'] = _ival(c('일천간'))
        st['간지 일지지'] = _ival(c('일지지'))
        st['간지 월천간'] = _ival(c('월천간'))
        st['간지 월지지'] = _ival(c('월지지'))
        st['음력 계절'] = _ival((c('음력월') - 1) // 3)
        slon = c('해황경')
        st['24절기 안 날짜'] = _bins(slon % 15, [5, 10])
        st['해 황경 8분면'] = _ival(slon // 45)
        st['공휴일까지'] = _bins(c('공휴일까지'), [8, 22])
        st['공휴일부터'] = _bins(c('공휴일부터'), [8, 22])
        st['추첨 주 공휴일'] = _ival(c('주중공휴일'))
        st['지난 30일 공휴일 수'] = _bins(c('30일공휴일수'), [1, 2])
        st['설·추석까지'] = _bins(c('명절까지'), [29, 85])
        dist = c('달거리')
        st['달 거리'] = _bins(dist, [372000, 395000])
        st['달 거리 1주 방향'] = _bins(dist - np.r_[np.nan, dist[:-1]], [0.0])
        st['달 적위 남북'] = _bins(c('달적위'), [0.0])
        mlon = c('달황경')
        st['달 별자리 원소'] = _ival(mlon // 30, 4)
        st['달 별자리 성질'] = _ival(mlon // 30, 3)
        st['추첨 시각 달 높이'] = _bins(c('달고도'), [0.0])
        st['수성 역행'] = _ival(c('수성역행'))
        st['금성 역행'] = _ival(c('금성역행'))
        st['화성 별자리 원소'] = _ival(c('화성황경') // 30, 4)
        st['일·월식 철'] = _ival(c('식철'))

    for key, label in FX_NAMES.items():
        if key in R.ext:
            st[f'{label} 1주 방향'] = _bins(_change(R, key, 1), [0.0])
    for key in ('fx_gbp', 'fx_aud', 'fx_twd', 'fx_sgd'):
        if key in R.ext:
            st[f'{FX_NAMES[key]} 4주 방향'] = _bins(_change(R, key, 4), [0.0])
    if 'usdkrw' in macro:
        krw = R.series('usdkrw')
        for other, label in (('usdjpy', '원·엔'), ('usdcny', '원·위안'), ('usdeur', '원·유로')):
            if other in macro:
                x = krw / R.series(other)
                for weeks in (1, 4):
                    past = R.series('usdkrw', offset=7 * weeks) / R.series(other, offset=7 * weeks)
                    st[f'{label} 교차 {weeks}주 방향'] = _bins(x / past - 1, [0.0])
        daily = np.array([np.nan] * R.n1)
        s = macro['usdkrw']
        for i, day in enumerate(R.day):
            k = np.searchsorted(s[0], day - 1, side='right')
            v = s[1][max(0, k - 130):k]
            if len(v) >= 120:
                r = np.diff(np.log(v))
                daily[i] = r[-20:].std() / (r.std() + 1e-12)
        st['원·달러 변동성 국면'] = _bins(daily, [1.0])
    if 'wti' in macro and 'brent' in macro:
        spread = R.series('brent') - R.series('wti')
        st['브렌트-WTI 차이 1주 방향'] = _bins(spread - (R.series('brent', 7) - R.series('wti', 7)), [0.0])
    if 'btc' in R.ext:
        st['비트코인 1주 방향'] = _bins(_change(R, 'btc', 1), [0.0])
        st['비트코인 4주 방향'] = _bins(_change(R, 'btc', 4), [0.0])
        st['비트코인 1주 크기'] = _bins(_change(R, 'btc', 1), [-0.05, 0.05])
        year = np.stack([R.series('btc', 7 * w) for w in range(52)])
        st['비트코인 52주 수준'] = _bins((year[0] - np.nanmean(year, 0)) / (np.nanstd(year, 0) + 1e-12), [-0.5, 0.5])
    if 'vix' in R.ext:
        st['VIX 수준'] = _bins(R.series('vix'), [15, 22])
        st['VIX 1주 방향'] = _bins(_change(R, 'vix', 1), [0.0])
    if 'natgas' in R.ext:
        st['천연가스 1주 방향'] = _bins(_change(R, 'natgas', 1), [0.0])
        st['천연가스 4주 방향'] = _bins(_change(R, 'natgas', 4), [0.0])
    if 'gtemp' in R.ext:
        g = R.series('gtemp')
        st['지구 기온 1년 변화'] = _bins(g - R.series('gtemp', 365), [0.0])
    if 'pen_w7' in R.ext:
        st['연금 직전 1등 당첨자 수'] = _bins(R.series('pen_w1', stale=8), [1, 2])
        st['연금 판매 1주 방향'] = _bins(R.series('pen_w7', stale=8) / R.series('pen_w7', 7, stale=8) - 1, [0.0])
    return {k: np.asarray(v, dtype=np.int64) for k, v in st.items()}


# ---------------------------------------------------------------- 채점 방식

def _zero(d, g):
    return np.zeros(len(d))


def match(t, tg, sign):
    """겨냥 숫자와 같은 자리 같은 숫자 개수 (+ 조 일치)."""
    def fit(draws, ix):
        tt, gt = t[ix], int(tg[ix])
        if (tt < 0).all() and gt < 0:
            return _zero
        return lambda d, g: sign * ((d == tt).sum(axis=1) + (g == gt)).astype(float)
    return fit


def offset(t, tg, strength):
    """과거 회차에서 '실제 숫자 - 겨냥 숫자'(10으로 나눈 나머지)가 자리별로 얼마나 자주 나왔는지."""
    def fit(draws, ix):
        tt, gt = t[ix], int(tg[ix])
        if (tt < 0).any():
            return _zero
        past, rows = t[:ix], draws.main[:ix].astype(np.int64)
        ok = (past >= 0).all(axis=1)
        off = (rows[ok] - past[ok]) % 10
        cnt = np.stack([np.bincount(off[:, j], minlength=10) for j in range(6)]).astype(float)
        lift = np.log((cnt + strength * 0.1) / (ok.sum() + strength) / 0.1)
        gp = tg[:ix]
        okg = gp >= 0
        goff = (draws.groups[:ix][okg].astype(np.int64) - 1 - gp[okg]) % 5
        glift = np.log((np.bincount(goff, minlength=5) + strength * 0.2) / (okg.sum() + strength) / 0.2)
        if gt < 0:
            glift = np.zeros(5)
            gt = 0
        return lambda d, g: lift[POS, (d - tt) % 10].sum(axis=1) + glift[(g - gt) % 5]
    return fit


def poscond(t, tg, strength):
    """겨냥 숫자가 같았던 과거 회차의 자리별 숫자 빈도 (전체 빈도 쪽으로 수축)."""
    def fit(draws, ix):
        tt, gt = t[ix], int(tg[ix])
        if (tt < 0).any():
            return _zero
        past, rows = t[:ix], draws.main[:ix].astype(np.int64)
        base = np.stack([np.bincount(rows[:, j], minlength=10) for j in range(6)]) + 1.0
        base /= base.sum(axis=1, keepdims=True)
        same = past == tt                                  # (ix, 6) 자리별로 같은 겨냥 숫자였던 회차
        hit = np.stack([np.bincount(rows[same[:, j], j], minlength=10) for j in range(6)]).astype(float)
        lift = np.log((hit + strength * base) / (same.sum(axis=0)[:, None] + strength) / base)
        groups = draws.groups[:ix].astype(np.int64) - 1
        g_all = np.bincount(groups, minlength=5) + 1.0
        g_all /= g_all.sum()
        if gt >= 0:
            gs = tg[:ix] == gt
            g_lift = np.log((np.bincount(groups[gs], minlength=5) + strength * g_all) / (gs.sum() + strength) / g_all)
        else:
            g_lift = np.zeros(5)
        return lambda d, g: lift[POS, d].sum(axis=1) + g_lift[g]
    return fit


def digit_set(t, sign):
    """겨냥 숫자들의 집합에 든 후보 숫자 개수 (자리 무관)."""
    def fit(draws, ix):
        tt = t[ix]
        if (tt < 0).all():
            return _zero
        inside = np.isin(np.arange(10), tt[tt >= 0])
        return lambda d, g: sign * inside[d].sum(axis=1).astype(float)
    return fit


def board(stack, mode, strength=20.0):
    """여러 겨냥 숫자의 투표판. 'plus'·'minus'는 한 표씩, 'learn'은 자리별로 과거에 맞은 비율만큼 가중치(로그 비)."""
    def fit(draws, ix):
        cur = stack[:, ix]                                     # (K, 6)
        if mode == 'learn':
            past, rows = stack[:, :ix], draws.main[:ix].astype(np.int64)
            ok = past >= 0
            hits = ((past == rows[None]) & ok).sum(axis=1).astype(float)
            p = (hits + strength * 0.1) / (ok.sum(axis=1) + strength)
            w = np.log(p / 0.1) - np.log((1 - p) / 0.9)
        else:
            w = np.full(cur.shape, 1.0 if mode == 'plus' else -1.0)
        pan = np.zeros((6, 10))
        k, j = np.nonzero(cur >= 0)
        np.add.at(pan, (j, cur[k, j]), w[k, j])
        return lambda d, g: pan[POS, d].sum(axis=1)
    return fit


def lotto_map(R, kind, sign):
    """로또 번호 n(1~45)을 (자리, 숫자) 칸으로 바꿔 후보가 몇 칸을 밟는지."""
    def fit(draws, ix):
        if not R.lotto_ok[ix]:
            return _zero
        n = R.lotto[ix, :7] - 1
        if kind == 'A':
            pos, dig = n // 10, n % 10              # 1~10 → 1번째 자리 0~9, 11~20 → 2번째 자리 ...
        elif kind == 'B':
            pos, dig = n % 6, n // 6                # 6칸씩 돌아가며
        else:
            pos, dig = (n * 7) % 6, (n * 3) % 10    # 곱셈 섞기
        board = np.zeros((6, 10))
        np.add.at(board, (pos, dig), 1.0)
        return lambda d, g: sign * board[POS, d].sum(axis=1)
    return fit


# ---------------------------------------------------------------- 공식 목록

def formulas(draws, macro, ext=None, cal=None):
    ext = load_external() if ext is None else ext
    cal = load_calendar() if cal is None else cal
    targets, R = build_targets(draws, macro, ext, cal)
    out = []
    for name, (t, tg) in targets.items():
        out.append(fm.Formula(f'ext|겨냥|{name}|일치', f'외부 {name} 같은 자리 일치', match(t, tg, 1.0)))
        out.append(fm.Formula(f'ext|겨냥|{name}|피하기', f'외부 {name} 같은 자리 피하기', match(t, tg, -1.0)))
        out.append(fm.Formula(f'ext|겨냥|{name}|차이', f'외부 {name} 자주 나온 차이', offset(t, tg, 10.0)))
        out.append(fm.Formula(f'ext|겨냥|{name}|조건', f'외부 {name} 같은 겨냥 숫자 회차 빈도', poscond(t, tg, 20.0)))
    for name in ('로또 끝자리', '2주 전 로또 끝자리', '날짜 yymmdd', '음력 yymmdd'):
        if name in targets:
            t = targets[name][0]
            out.append(fm.Formula(f'ext|집합|{name}|1', f'외부 {name} 숫자 집합에 든 개수', digit_set(t, 1.0)))
            out.append(fm.Formula(f'ext|집합|{name}|-1', f'외부 {name} 숫자 집합 피하기', digit_set(t, -1.0)))
    groups = {'전체': list(targets),
              '로또': [k for k in targets if '로또' in k],
              '달력·천체': [k for k in targets if k.split()[0] in ('날짜', '연중', '음력', '간지', '달', '해', '서울')
                         and '1등' not in k],
              '매크로': [k for k in targets if k.endswith('값 숫자')]}
    for fam, names in groups.items():
        if not names:
            continue
        stack = np.stack([targets[k][0] for k in names])
        modes = (('learn', '학습 가중'), ('plus', '한 표씩'), ('minus', '한 표씩 피하기')) if fam == '전체' else (('learn', '학습 가중'),)
        for mode, word in modes:
            out.append(fm.Formula(f'ext|투표판|{fam}|{mode}', f'외부 {fam} 숫자 투표판 {word}', board(stack, mode)))
    for kind, word in (('A', '10칸씩'), ('B', '6칸씩'), ('C', '곱셈 섞기')):
        out.append(fm.Formula(f'ext|로또지도|{kind}|1', f'로또 번호 자리·숫자 지도 {word} 밟기', lotto_map(R, kind, 1.0)))
        out.append(fm.Formula(f'ext|로또지도|{kind}|-1', f'로또 번호 자리·숫자 지도 {word} 피하기', lotto_map(R, kind, -1.0)))
    for name, state in build_states(draws, macro, ext, cal, R).items():
        for sign, word in ((1, ''), (-1, ' 반대')):
            out.append(fm.Formula(f'ext|국면|{name}|20|{sign}', f'외부 국면 {name} 수축20{word}',
                                  lab.conditional(state, 20.0, sign)))
    return out


# ---------------------------------------------------------------- 실험

def check_lookahead(draws, formulas_, rows=(-3, -17, -38), macro=None):
    """같은 공식을 자른 자료에 쓴 점수, 그리고 macro를 주면 자른 자료로 처음부터 다시 만든 공식의 점수까지
    전체 자료 점수와 같은지 본다. 다른 (key, ix) 목록."""
    bad = []
    n = len(draws.rounds)
    for r in rows:
        ix = n + r
        ids = lab.fold_sample(draws, ix)[:2000]
        d, g = eng.split_ids(ids)
        cut = draws.truncated(ix)
        rebuilt = formulas(cut, macro) if macro is not None else [None] * len(formulas_)
        for f, f2 in zip(formulas_, rebuilt):
            ref = f.fit(draws, ix)(d, g)
            if not np.allclose(ref, f.fit(cut, ix)(d, g)):
                bad.append((f.key, ix))
            elif f2 is not None and (f2.key != f.key or not np.allclose(ref, f2.fit(cut, ix)(d, g))):
                bad.append((f.key, ix, '다시 만듦'))
    return bad


def main():
    parser = argparse.ArgumentParser(description='외부 지표·달력·다른 복권 공식 실험')
    parser.add_argument('--update', action='store_true', help='external.csv 새로 받기')
    parser.add_argument('--calendar', action='store_true', help='calendar_ext.csv 다시 만들기')
    parser.add_argument('--folds', type=int, default=140)
    parser.add_argument('--out', type=Path, default=HERE / 'results' / '창의공식')
    args = parser.parse_args()
    if args.update:
        print(update_external())
    if args.calendar:
        print(build_calendar())
    draws = eng.load_draws(eng.DATA_CSV)
    macro = lab.load_macro()
    fs = formulas(draws, macro)
    print(f'공식 {len(fs)}개', flush=True)
    start = time.time()
    pct, _ = lab.walk(draws, fs, args.folds)
    print(f'실제 걷기 {time.time() - start:.0f}초', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'external_pct.npy', pct)


if __name__ == '__main__':
    main()
