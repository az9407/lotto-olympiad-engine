#!/usr/bin/env python3
"""연금복권720+ 수학·정수론·조합 구조 공식 실험실.

후보 6자리 N(000000~999999)과 조 g를 정수·자리 숫자 구조로 보고 범주값 v를 만든 뒤,
과거 1등에서 v가 나온 횟수 O와 '후보가 완전 무작위일 때의 기대 횟수' E를 비교한다.
E는 1,000,000개 번호 격자에서 정확히 센 확률로 구한다. 직전 번호·회차에 따라 확률이 달라지는 값은
회차마다 그 시점 확률을 따로 더한다.
점수 = 부호 x log((O + 람다) / (E + 람다)), 람다 = 10 (감마-푸아송 수축). 부호 +1은 잘 나온 값 쪽, -1은 반대.

가족
- 정적 구조 65개: 6자리 정수 나머지 12개, 자리 숫자 구조, 2진·3진 표현, 소인수 구조, 행복수·지속수·콜라츠, 벤포드
- 직전 번호와의 자리별 차·합 변환 4개 x 구조 5개 (변환이 일대일이라 이론 확률은 정적 구조와 같다)
- 직전 번호·회차와의 정수 산술: (N - c) mod m, (N - c) mod 10^6 의 10만 단위 칸
- 상황별 확률: 자리합 변화, 회차로 나눈 나머지, 최대공약수, 비율, 숫자 겹침, 순서 패턴 일치, 정렬 거리
- 자리별 점화식 잔차: 등차, 2차, 피보나치, 트리보나치, 곱, 시차, 자리 이동, 뒤집기, 보너스 합
- 이웃이 아닌 자리쌍 동반 출현과 자리 차
- 조 산술: (조 - h) mod 5

모든 fit(draws, ix)는 ix 이전 행과 ix 행의 회차·날짜만 쓴다.
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache
from math import sqrt
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

M = eng.N_NUMBERS
POW = eng.POWERS.astype(np.int64)
POS = np.arange(6)
LAM = 10.0
T0 = 4                      # 점화식·직전 번호 공식은 5번째 회차부터 센다
PHI = (1 + 5 ** 0.5) / 2
GOLD = 618034               # 10^6 / 황금비의 소수부


def _lift(obs, exp, lam=LAM):
    return np.log((obs + lam) / (exp + lam))


# ---------------------------------------------------------------- 1,000,000 격자와 정수론 표

@lru_cache(maxsize=None)
def _grid():
    return eng.number_grid().astype(np.int16), np.arange(M, dtype=np.int64)


def _D():
    return _grid()[0]


def _N():
    return _grid()[1]


@lru_cache(maxsize=None)
def _counts():
    d = _D()
    return np.stack([(d == v).sum(1) for v in range(10)], axis=1).astype(np.uint8)


@lru_cache(maxsize=None)
def _sieve(limit=1_000_100):
    s = np.ones(limit + 1, bool)
    s[:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return s


@lru_cache(maxsize=None)
def _nt():
    """0~999999의 소인수 구조: Omega, omega, 약수 개수, 최대 지수, phi(N)/N, 두 제곱수 합 여부."""
    n = _N()
    spf = np.zeros(M, dtype=np.int64)
    for p in range(2, 1001):
        if spf[p] == 0:
            seg = spf[p * p::p]
            seg[seg == 0] = p
    rest = (spf == 0) & (n >= 2)
    spf[rest] = n[rest]
    spf[:2] = 1
    m = n.copy()
    m[0] = 1
    last = np.zeros(M, np.int64)
    e = np.zeros(M, np.int64)
    big_omega = np.zeros(M, np.int64)
    small_omega = np.zeros(M, np.int64)
    tau = np.ones(M, np.int64)
    max_e = np.zeros(M, np.int64)
    ratio = np.ones(M)
    two_sq = np.ones(M, bool)

    def close(ix):
        tau[ix] *= e[ix] + 1
        two_sq[ix] &= ~((last[ix] % 4 == 3) & (e[ix] % 2 == 1))

    while True:
        idx = np.flatnonzero(m > 1)
        if not idx.size:
            break
        p = spf[m[idx]]
        new = p != last[idx]
        close(idx[new & (last[idx] > 1)])
        nw = idx[new]
        last[nw] = p[new]
        e[nw] = 1
        small_omega[nw] += 1
        ratio[nw] *= 1 - 1 / p[new]
        e[idx[~new]] += 1
        max_e[idx] = np.maximum(max_e[idx], e[idx])
        big_omega[idx] += 1
        m[idx] //= p
    close(np.flatnonzero(last > 1))
    mobius = np.where(max_e >= 2, 0, np.where(small_omega % 2 == 0, 1, -1))
    mobius[0] = 2           # 0은 따로
    ratio[0] = -1.0
    tau[0] = 0
    return dict(spf=spf, Omega=big_omega, omega=small_omega, tau=tau, mobius=mobius, ratio=ratio, two_sq=two_sq)


@lru_cache(maxsize=None)
def _collatz():
    x = _N().copy()
    x[0] = 1
    steps = np.zeros(M, np.int64)
    act = np.flatnonzero(x > 1)
    while act.size:
        v = x[act]
        v = np.where(v % 2 == 0, v // 2, 3 * v + 1)
        x[act] = v
        steps[act] += 1
        act = act[v > 1]
    steps[0] = -1
    return steps


def _decile(x):
    edges = np.quantile(x, np.linspace(0.1, 0.9, 9))
    return np.searchsorted(edges, x, side='right')


def _longest(mask):
    best = np.zeros(len(mask), np.int64)
    cur = np.zeros(len(mask), np.int64)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _digit_product(x):
    """임의 자연수의 자리 곱 (앞자리 0은 세지 않는다)."""
    p = np.where(x == 0, 0, 1)
    y = x.copy()
    while (y > 0).any():
        a = y > 0
        p[a] *= y[a] % 10
        y[a] //= 10
    return p


def _isprime_small():
    return _sieve()[:1000]


def f_mod(m):
    return lambda: _N() % m


def f_dr():
    n = _N()
    return np.where(n == 0, 0, 1 + (n - 1) % 9)


def f_alt():
    d = _D()
    return d[:, 0] - d[:, 1] + d[:, 2] - d[:, 3] + d[:, 4] - d[:, 5]


def f_xor():
    return np.bitwise_xor.reduce(_D(), axis=1)


@lru_cache(maxsize=None)
def _prod():
    return _D().astype(np.int64).prod(1)


def f_prodcat():
    p = _prod()
    return np.where(p == 0, 0, 1 + np.floor(np.log2(np.maximum(p, 1))).astype(np.int64))


def f_prod10():
    return _prod() % 10


def f_mirror():
    d = _D()
    return (d[:, 0] == d[:, 5]).astype(int) + (d[:, 1] == d[:, 4]) + (d[:, 2] == d[:, 3])


def f_uprun():
    return _longest(np.diff(_D(), axis=1) == 1)


def f_downrun():
    return _longest(np.diff(_D(), axis=1) == -1)


def f_updown():
    df = np.diff(_D(), axis=1)
    return (df > 0).sum(1) * 6 + (df < 0).sum(1)


def f_shape():
    c = -np.sort(-_counts().astype(np.int64), axis=1)
    return (c * 7 ** np.arange(10)).sum(1)


def f_eqpart():
    d = _D()
    lab_ = np.zeros((M, 6), np.int64)
    nxt = np.ones(M, np.int64)
    for j in range(1, 6):
        lj = np.full(M, -1)
        for k in range(j - 1, -1, -1):
            lj = np.where(d[:, j] == d[:, k], lab_[:, k], lj)
        new = lj < 0
        lj = np.where(new, nxt, lj)
        nxt = nxt + new
        lab_[:, j] = lj
    return (lab_ * 6 ** np.arange(6)).sum(1)


def f_range():
    d = _D()
    return d.max(1) - d.min(1)


def f_consec():
    pres = _counts() > 0
    return (pres[:, :-1] & pres[:, 1:]).sum(1)


def f_maxgap():
    return np.diff(np.sort(_D(), axis=1), axis=1).max(1)


def f_dd0():
    d = _D().astype(np.int64)
    return ((d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]) % 10 == 0).sum(1)


def f_sq10():
    return (_D().astype(np.int64) ** 2).sum(1) % 10


def f_class(values):
    return lambda: np.isin(_D(), values).sum(1)


def f_ab_order():
    n = _N()
    return np.sign(n // 1000 - n % 1000)


def f_ab_gcd():
    n = _N()
    return np.digitize(np.gcd(n // 1000, n % 1000), [1, 2, 3, 4, 10, 100])


def _xyz():
    n = _N()
    return n // 10000, n // 100 % 100, n % 100


def f_xyz_order():
    x, y, z = _xyz()
    return (np.sign(x - y) + 1) * 9 + (np.sign(y - z) + 1) * 3 + np.sign(x - z) + 1


def f_xyz_primes():
    pr = _isprime_small()
    x, y, z = _xyz()
    return pr[x].astype(int) + pr[y] + pr[z]


def f_xyz_sum():
    x, y, z = _xyz()
    return (x + y + z) // 30


def f_pw(width):
    def f():
        d = _D().astype(np.int64)
        pr = _isprime_small()
        out = np.zeros(M, np.int64)
        for j in range(7 - width):
            w = np.zeros(M, np.int64)
            for k in range(width):
                w = w * 10 + d[:, j + k]
            out += pr[w]
        return out
    return f


def f_rev():
    rev = _D()[:, ::-1].astype(np.int64) @ POW
    return np.sign(rev - _N())


def f_kaprekar():
    s = np.sort(_D(), axis=1).astype(np.int64)
    return (s[:, ::-1] @ POW - s @ POW) // 100000


def f_mod7x13():
    n = _N()
    return n % 7 * 13 + n % 13


def _bits(n, k):
    return (n >> k) & 1


def f_popcount():
    n = _N()
    return sum(_bits(n, k) for k in range(20))


def f_v2():
    n = _N()
    out = np.zeros(M, np.int64)
    for k in range(1, 11):
        out += (n % (1 << k) == 0)
    return out


def f_v3():
    n = _N()
    return sum((n % 3 ** k == 0).astype(np.int64) for k in range(1, 7))


def f_b3sum():
    n = _N().copy()
    s = np.zeros(M, np.int64)
    while (n > 0).any():
        s += n % 3
        n //= 3
    return s


def f_run1():
    n = _N()
    return _longest(np.stack([_bits(n, k) == 1 for k in range(20)], axis=1))


def f_isprime():
    return _sieve()[:M].astype(int)


def f_nt(key):
    return lambda: _nt()[key]


def f_tau():
    t = _nt()['tau']
    return np.where(t == 0, -1, np.floor(np.log2(np.maximum(t, 1))).astype(np.int64))


def f_phiratio():
    r = _nt()['ratio']
    out = _decile(r[1:])
    return np.r_[-1, out]


def f_harshad():
    n = _N()
    s = _D().sum(1).astype(np.int64)
    return np.where(n == 0, 2, (n % np.maximum(s, 1) == 0).astype(int))


def f_nprime():
    n = _N()
    primes = np.flatnonzero(_sieve())
    i = np.searchsorted(primes, n)                 # primes[i] >= n
    up = primes[np.minimum(i, len(primes) - 1)] - n
    down = np.where(i > 0, n - primes[np.maximum(i - 1, 0)], 10 ** 6)
    return np.digitize(np.minimum(up, down), [1, 2, 3, 5, 9, 17])


def f_happy():
    def step(x):
        s = 0
        while x:
            s += (x % 10) ** 2
            x //= 10
        return s
    happy = np.zeros(487, int)
    for x in range(1, 487):
        seen, y = set(), x
        while y != 1 and y not in seen:
            seen.add(y)
            y = step(y)
        happy[x] = y == 1
    sq = (_D().astype(np.int64) ** 2).sum(1)
    out = happy[sq]
    out[0] = 2
    return out


def f_addpers():
    s = _D().sum(1).astype(np.int64)
    per = np.zeros(55, int)
    for x in range(10, 55):
        per[x] = 1 + per[sum(map(int, str(x)))]
    return np.where(_N() < 10, 0, 1 + per[s])


def f_mulpers():
    x = _N().copy()
    cnt = np.zeros(M, np.int64)
    while (x >= 10).any():
        a = x >= 10
        x[a] = _digit_product(x[a])
        cnt[a] += 1
    return cnt


def f_collatz():
    s = _collatz()
    return np.r_[-1, _decile(s[1:])]


def f_sum_prime():
    return _isprime_small()[_D().sum(1)].astype(int)


def f_lead():
    d = _D()
    i = np.argmax(d != 0, axis=1)
    return d[np.arange(M), i]


def f_mantissa():
    n = _N()
    x = np.log10(np.maximum(n, 1))
    return np.where(n == 0, -1, np.floor(10 * (x - np.floor(x))).astype(np.int64))


def f_phifrac():
    return np.floor(10 * ((_N() * PHI) % 1.0)).astype(np.int64)


def f_sqrtfrac():
    r = np.sqrt(_N().astype(float))
    return np.floor(10 * (r - np.floor(r))).astype(np.int64)


def f_nfib():
    fib = [0, 1]
    while fib[-1] < 3 * M:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[1:], np.int64)
    n = _N()
    i = np.searchsorted(fib, n)
    up = fib[np.minimum(i, len(fib) - 1)] - n
    down = np.where(i > 0, n - fib[np.maximum(i - 1, 0)], 10 ** 7)
    return np.floor(np.log10(np.minimum(up, down) + 1.0)).astype(np.int64)


STATIC = {}
for _m in (3, 4, 7, 8, 9, 11, 13, 16, 17, 19, 37, 101):
    STATIC[f'mod{_m}'] = (f'6자리 수 mod {_m}', f_mod(_m))
STATIC.update({
    'mod7x13': ('6자리 수 (mod 7, mod 13) 중국인 나머지 쌍', f_mod7x13),
    'dr': ('디지털 루트', f_dr),
    'alt': ('교대합 d1-d2+d3-d4+d5-d6', f_alt),
    'xor': ('자리 숫자 XOR', f_xor),
    'prodcat': ('자리 곱 크기 log2', f_prodcat),
    'prod10': ('자리 곱 끝자리', f_prod10),
    'mirror': ('좌우 대칭 자리 일치 수', f_mirror),
    'uprun': ('+1 오름 연속 길이', f_uprun),
    'downrun': ('-1 내림 연속 길이', f_downrun),
    'updown': ('오름·내림 걸음 수', f_updown),
    'shape': ('숫자 중복 모양 (분할)', f_shape),
    'eqpart': ('같은 숫자 자리 분할 (벨 수 203)', f_eqpart),
    'range': ('최대-최소 숫자', f_range),
    'consec': ('이어진 숫자 쌍 수', f_consec),
    'maxgap': ('정렬 숫자 최대 간격', f_maxgap),
    'dd0': ('2차 차분 0 (자리 등차) 수', f_dd0),
    'sq10': ('자리 제곱합 끝자리', f_sq10),
    'sqdig': ('제곱수 숫자 0,1,4,9 개수', f_class([0, 1, 4, 9])),
    'fibdig': ('피보나치 숫자 0,1,2,3,5,8 개수', f_class([0, 1, 2, 3, 5, 8])),
    'tridig': ('삼각수 숫자 0,1,3,6 개수', f_class([0, 1, 3, 6])),
    'm3dig': ('3의 배수 숫자 개수', f_class([0, 3, 6, 9])),
    'ab_order': ('앞 3자리 대 뒤 3자리 크기', f_ab_order),
    'ab_gcd': ('앞 3자리·뒤 3자리 최대공약수', f_ab_gcd),
    'xyz_order': ('두 자리 수 3개 순서 모양', f_xyz_order),
    'xyz_primes': ('두 자리 수 3개 중 소수 개수', f_xyz_primes),
    'xyz_sum': ('두 자리 수 3개 합 30 단위', f_xyz_sum),
    'pw2': ('연속 두 자리 창 소수 개수', f_pw(2)),
    'pw3': ('연속 세 자리 창 소수 개수', f_pw(3)),
    'rev': ('뒤집은 수 대 원래 수 크기', f_rev),
    'kaprekar': ('카프리카 차 10만 단위', f_kaprekar),
    'popcount': ('2진수 1의 개수', f_popcount),
    'v2': ('2로 나누어지는 횟수', f_v2),
    'v3': ('3으로 나누어지는 횟수', f_v3),
    'b3sum': ('3진수 자리합', f_b3sum),
    'run1': ('2진수 1 최장 연속', f_run1),
    'isprime': ('6자리 수 소수 여부', f_isprime),
    'Omega': ('소인수 개수 (중복 포함)', f_nt('Omega')),
    'omega': ('서로 다른 소인수 개수', f_nt('omega')),
    'mobius': ('뫼비우스 함수', f_nt('mobius')),
    'tau': ('약수 개수 log2', f_tau),
    'phiratio': ('오일러 phi(N)/N 10분위', f_phiratio),
    'two_sq': ('두 제곱수의 합 여부', f_nt('two_sq')),
    'harshad': ('하샤드 수 (자리합으로 나누어짐)', f_harshad),
    'nprime': ('가장 가까운 소수까지 거리', f_nprime),
    'happy': ('행복수 여부', f_happy),
    'addpers': ('덧셈 지속수', f_addpers),
    'mulpers': ('곱셈 지속수', f_mulpers),
    'collatz': ('콜라츠 정지 시간 10분위', f_collatz),
    'sum_prime': ('자리합 소수 여부', f_sum_prime),
    'lead': ('벤포드 첫 유효 숫자', f_lead),
    'mantissa': ('벤포드 log10 가수 10분위', f_mantissa),
    'phifrac': ('N x 황금비 소수부 10칸', f_phifrac),
    'sqrtfrac': ('제곱근 소수부 10칸', f_sqrtfrac),
    'nfib': ('가장 가까운 피보나치 수까지 거리 자릿수', f_nfib),
})
_HELPER_ONLY = {'sum10': ('자리합 끝자리', lambda: _D().sum(1) % 10)}


@lru_cache(maxsize=None)
def table(name):
    """격자 1,000,000개의 범주값 표, 범주 수, 정확한 확률."""
    fn = STATIC[name][1] if name in STATIC else _HELPER_ONLY[name][1]
    raw = np.asarray(fn())
    _, inv = np.unique(raw, return_inverse=True)
    t = inv.reshape(-1).astype(np.int16)
    k = int(t.max()) + 1
    return t, k, np.bincount(t, minlength=k) / M


def warm():
    for name in list(STATIC) + list(_HELPER_ONLY):
        table(name)


# ---------------------------------------------------------------- 시점 정보와 후보 캐시

_CX = {}


def _target_round(draws, ix):
    return int(draws.rounds[ix]) if ix < len(draws.rounds) else int(draws.rounds[ix - 1]) + 1


def _ctx(draws, ix):
    """행 t=0..ix마다 t 이전 정보: 1~4회 전 1등, 1회 전 보너스, 1·2회 전 조, 회차, 추첨일."""
    if _CX.get('draws') is draws and _CX.get('ix') == ix:
        return _CX['c']
    main = draws.main[:ix].astype(np.int64)
    bonus = draws.bonus[:ix].astype(np.int64)
    grp = draws.groups[:ix].astype(np.int64) - 1
    pad = np.vstack((np.zeros((T0, 6), np.int64), main))
    padb = np.vstack((np.zeros((T0, 6), np.int64), bonus))
    padg = np.r_[np.zeros(T0, np.int64), grp]

    def lag(a, k):
        return a[T0 - k:T0 - k + ix + 1]

    c = dict(main=main, grp=grp, N=main @ POW, S=main.sum(1),
             P1=lag(pad, 1), P2=lag(pad, 2), P3=lag(pad, 3), P4=lag(pad, 4), B1=lag(padb, 1),
             G1=lag(padg, 1), G2=lag(padg, 2),
             r=np.r_[draws.rounds[:ix].astype(np.int64), _target_round(draws, ix)],
             day=np.array([d.day for d in lab.draw_dates(draws)[:ix + 1]], np.int64))
    c['N1'], c['N2'], c['N3'] = c['P1'] @ POW, c['P2'] @ POW, c['P3'] @ POW
    c['NB1'] = c['B1'] @ POW
    c['S1'] = c['P1'].sum(1)
    _CX.update(draws=draws, ix=ix, c=c)
    return c


_DC = {}


def _cache(d, key, make):
    if _DC.get('d') is not d:
        _DC.clear()
        _DC['d'] = d
    if key not in _DC:
        _DC[key] = make()
    return _DC[key]


def _cd(d):
    return _cache(d, 'd64', lambda: np.asarray(d, dtype=np.int64))


def _cN(d):
    return _cache(d, 'N', lambda: _cd(d) @ POW)


def _cS(d):
    return _cache(d, 'S', lambda: _cd(d).sum(1))


def _cand_static(d, name):
    return _cache(d, ('s', name), lambda: table(name)[0][_cN(d)])


def _row_counts(x):
    return (x[:, :, None] == np.arange(10)).sum(1)


# ---------------------------------------------------------------- 가족 A: 정적 구조

def static_fit(name, sign):
    def fit(draws, ix):
        t, k, p = table(name)
        v = t[draws.main[:ix].astype(np.int64) @ POW]
        lift = sign * _lift(np.bincount(v, minlength=k), ix * p)
        return lambda d, g: lift[_cand_static(d, name)]
    return fit


# ---------------------------------------------------------------- 가족 B: 직전 번호와 자리별 차·합

TRANSFORMS = {'main': '직전 1등과 자리별 차', 'bonus': '직전 보너스와 자리별 차',
              'rev': '뒤집은 직전 1등과 자리별 차', 'sum': '직전 1등과 자리별 합'}
TRANSFORM_FEATS = ('sum10', 'mod7', 'mod11', 'shape', 'updown')


def _ref(c, kind):
    if kind == 'bonus':
        return c['B1']
    if kind == 'rev':
        return c['P1'][:, ::-1]
    return c['P1']


def _apply(x, ref, kind):
    return (x + ref) % 10 if kind == 'sum' else (x - ref) % 10


def transform_fit(kind, name, sign):
    def fit(draws, ix):
        c = _ctx(draws, ix)
        t, k, p = table(name)
        ref = _ref(c, kind)
        v = t[_apply(c['main'][T0:ix], ref[T0:ix], kind) @ POW]
        lift = sign * _lift(np.bincount(v, minlength=k), (ix - T0) * p)
        r = ref[ix].copy()

        def score(d, g):
            nd = _cache(d, ('t', kind, r.tobytes()), lambda: _apply(_cd(d), r, kind) @ POW)
            return lift[t[nd]]
        return score
    return fit


# ---------------------------------------------------------------- 가족 C: (N - c) 정수 산술

SHIFTS = {'prev': '직전 1등', 'ap': '등차 2N1-N2', 'fib': '피보나치 N1+N2', 'bonus': '직전 보너스',
          'gold': '직전 1등 x 황금비', 'rev': '뒤집은 직전 1등', 'sq': '직전 1등 제곱',
          'weyl': '회차 x 황금비 (바일 수열)', 'prod': '직전 두 1등 곱', 'quad': '2차 외삽 3N1-3N2+N3',
          'round': '회차'}


def _cvalue(c, name):
    n1, n2, n3 = c['N1'], c['N2'], c['N3']
    if name == 'prev':
        return n1
    if name == 'ap':
        return (2 * n1 - n2) % M
    if name == 'fib':
        return (n1 + n2) % M
    if name == 'bonus':
        return c['NB1']
    if name == 'gold':
        return n1 * GOLD % M
    if name == 'rev':
        return c['P1'][:, ::-1] @ POW
    if name == 'sq':
        return n1 * n1 % M
    if name == 'weyl':
        return c['r'] * GOLD % M
    if name == 'prod':
        return n1 * n2 % M
    if name == 'quad':
        return (3 * n1 - 3 * n2 + n3) % M
    if name == 'round':
        return c['r']
    raise ValueError(name)


@lru_cache(maxsize=None)
def _base_mod(m):
    return np.bincount(np.arange(M) % m, minlength=m) / M


def shift_fit(cname, m, bins, sign):
    """bins가 있으면 v = ((N - c) mod 10^6) x bins // 10^6, 없으면 v = (N - c) mod m."""
    def fit(draws, ix):
        c = _ctx(draws, ix)
        cv = _cvalue(c, cname)
        x, ch = c['N'][T0:ix], cv[T0:ix]
        if bins:
            v, k = (x - ch) % M * bins // M, bins
            exp = np.full(k, (ix - T0) / k)
        else:
            v, k = (x - ch) % m, m
            exp = (np.full(m, (ix - T0) / m) if M % m == 0
                   else _base_mod(m)[(np.arange(m)[None, :] + ch[:, None]) % m].sum(0))
        lift = sign * _lift(np.bincount(v, minlength=k), exp)
        ct = int(cv[ix])
        if bins:
            return lambda d, g: lift[(_cN(d) - ct) % M * bins // M]
        return lambda d, g: lift[(_cN(d) - ct) % m]
    return fit


# ---------------------------------------------------------------- 가족 D: 시점마다 확률이 다른 값

@lru_cache(maxsize=None)
def _p_sum():
    return np.bincount(_D().sum(1), minlength=55) / M


@lru_cache(maxsize=None)
def _p_modr(r):
    q, rem = divmod(M, r)
    cnt = np.full(r, float(q))
    cnt[:rem] += 1
    return np.bincount(np.arange(r) * 10 // r, weights=cnt, minlength=10) / M


@lru_cache(maxsize=None)
def _p_gcd(c):
    """[gcd(N,c)=1 확률, >1 확률]. 포함-배제로 정확히."""
    c = int(c)
    if c == 0:
        coprime = 1
    else:
        primes, x, p = [], c, 2
        while p * p <= x:
            if x % p == 0:
                primes.append(p)
                while x % p == 0:
                    x //= p
            p += 1
        if x > 1:
            primes.append(x)
        coprime = 0
        for mask in range(1 << len(primes)):
            e, bits = 1, 0
            for i, q in enumerate(primes):
                if mask >> i & 1:
                    e *= q
                    bits += 1
            coprime += (-1) ** bits * (-(-M // e))
    return np.array([coprime / M, 1 - coprime / M])


@lru_cache(maxsize=None)
def _L2():
    return np.log2(np.arange(M) + 1.0)


def _lc(c):
    return float(np.log2(np.array([float(c) + 1.0]))[0])


def _ratio_bin(l2n, c):
    return np.clip(np.floor(l2n - _lc(c)), -10, 9).astype(np.int64) + 10


@lru_cache(maxsize=None)
def _p_ratio(c):
    return np.bincount(_ratio_bin(_L2(), c), minlength=20) / M


@lru_cache(maxsize=None)
def _p_msov(shape):
    cp = np.zeros(10, np.uint8)
    cp[:len(shape)] = shape
    return np.bincount(np.minimum(_counts(), cp).sum(1), minlength=7) / M


@lru_cache(maxsize=None)
def _p_setov(k):
    return np.bincount((_counts()[:, :k] > 0).sum(1), minlength=7) / M


_PAIRS15 = [(a, b) for a in range(6) for b in range(a + 1, 6)]
_PA = np.array([a for a, _ in _PAIRS15])
_PB = np.array([b for _, b in _PAIRS15])


def _signs(x):
    return np.sign(x[:, _PB].astype(np.int64) - x[:, _PA]).astype(np.int8)


@lru_cache(maxsize=None)
def _grid_signs():
    return _signs(_D())


@lru_cache(maxsize=None)
def _p_concord(sp):
    sp = np.frombuffer(sp, np.int8)
    return np.bincount((_grid_signs() == sp).sum(1), minlength=16) / M


@lru_cache(maxsize=None)
def _grid_sorted():
    return np.sort(_D(), axis=1).astype(np.int16)


@lru_cache(maxsize=None)
def _p_sortl1(sp):
    sp = np.array(sp, np.int16)
    return np.bincount(np.abs(_grid_sorted() - sp).sum(1) // 3, minlength=19) / M


def _shape_of(counts_row):
    return tuple(sorted((int(v) for v in counts_row if v), reverse=True))


CTXFEATS = {
    'dsum': '자리합 변화 S - 직전 S',
    'modr': '6자리 수 mod 회차 10칸',
    'gcdr': '회차와 서로소 여부',
    'gcdprev': '직전 1등과 서로소 여부',
    'ratio': '직전 1등 대비 비율 log2',
    'msov': '직전 1등과 숫자 중복집합 겹침',
    'msovb': '직전 보너스와 숫자 중복집합 겹침',
    'setov': '직전 1등 숫자 집합에 든 서로 다른 숫자 수',
    'concord': '직전 1등과 자리 순서 패턴 일치 수 (15쌍)',
    'sortl1': '직전 1등과 정렬 숫자 거리',
}


def _ctxfeat(kind, c, ix):
    """(과거 값, 기대 횟수, 범주 수, 후보 함수)."""
    h = slice(T0, ix)
    x = c['main'][h]
    if kind == 'dsum':
        s1 = c['S1']
        ext = np.zeros(163)
        ext[54:109] = _p_sum()
        exp = ext[s1[h][:, None] + np.arange(109)].sum(0)
        st = int(s1[ix])
        return c['S'][h] - s1[h] + 54, exp, 109, lambda d: _cS(d) - st + 54
    if kind == 'modr':
        r = c['r']
        exp = sum(_p_modr(int(v)) for v in r[h])
        rt = int(r[ix])
        return c['N'][h] % r[h] * 10 // r[h], exp, 10, lambda d: _cN(d) % rt * 10 // rt
    if kind in ('gcdr', 'gcdprev'):
        cv = c['r'] if kind == 'gcdr' else c['N1']
        exp = sum(_p_gcd(int(v)) for v in cv[h])
        ct = int(cv[ix])
        return ((np.gcd(c['N'][h], cv[h]) != 1).astype(np.int64), exp, 2,
                lambda d: (np.gcd(_cN(d), ct) != 1).astype(np.int64))
    if kind == 'ratio':
        n1 = c['N1']
        exp = sum(_p_ratio(int(v)) for v in n1[h])
        v = np.array([_ratio_bin(_L2()[a], b) for a, b in zip(c['N'][h], n1[h])], np.int64)
        ct = int(n1[ix])
        return v, exp, 20, lambda d: _ratio_bin(_L2()[_cN(d)], ct)
    if kind in ('msov', 'msovb', 'setov'):
        ref = c['P1'] if kind != 'msovb' else c['B1']
        ch, cr = _row_counts(x), _row_counts(ref[h])
        crt = _row_counts(ref[ix:ix + 1])[0]

        def counts(d):
            return _cache(d, 'counts', lambda: _row_counts(_cd(d)))
        if kind == 'setov':
            exp = sum(_p_setov(int(k)) for k in (cr > 0).sum(1))
            v = ((ch > 0) & (cr > 0)).sum(1)
            mask = crt > 0
            return v, exp, 7, lambda d: (counts(d)[:, mask] > 0).sum(1)
        exp = sum(_p_msov(_shape_of(row)) for row in cr)
        v = np.minimum(ch, cr).sum(1)
        return v, exp, 7, lambda d: np.minimum(counts(d), crt).sum(1)
    if kind == 'concord':
        sh, sr = _signs(x), _signs(c['P1'][h])
        exp = sum(_p_concord(row.tobytes()) for row in sr)
        st = _signs(c['P1'][ix:ix + 1])[0]
        return ((sh == sr).sum(1), exp, 16,
                lambda d: (_cache(d, 'signs', lambda: _signs(_cd(d))) == st).sum(1))
    if kind == 'sortl1':
        xs, rs = np.sort(x, axis=1), np.sort(c['P1'][h], axis=1)
        exp = sum(_p_sortl1(tuple(int(a) for a in row)) for row in rs)
        st = np.sort(c['P1'][ix])
        return (np.abs(xs - rs).sum(1) // 3, exp, 19,
                lambda d: np.abs(_cache(d, 'sorted', lambda: np.sort(_cd(d), axis=1)) - st).sum(1) // 3)
    raise ValueError(kind)


def ctxfeat_fit(kind, sign):
    def fit(draws, ix):
        v, exp, k, cand = _ctxfeat(kind, _ctx(draws, ix), ix)
        lift = sign * _lift(np.bincount(v, minlength=k), exp)
        return lambda d, g: lift[cand(d)]
    return fit


# ---------------------------------------------------------------- 가족 E: 자리별 점화식 잔차

RECURS = {'ap': '등차 2x1-x2', 'quad': '2차 3x1-3x2+x3', 'fib': '피보나치 x1+x2', 'trib': '트리보나치 x1+x2+x3',
          'diff': '차 x1-x2', 'prod': '곱 x1·x2', 'mean3': '3회 평균', 'lag1': '1회 전', 'lag2': '2회 전',
          'lag3': '3회 전', 'lag4': '4회 전', 'shl': '1회 전 왼쪽 이웃 자리', 'shr': '1회 전 오른쪽 이웃 자리',
          'rev': '1회 전 뒤집기', 'mb': '1회 전 1등+보너스', 'sort': '1회 전 정렬'}


def _pred(c, kind, sl):
    p1, p2, p3, p4, b1 = (c[k][sl] for k in ('P1', 'P2', 'P3', 'P4', 'B1'))
    out = {'ap': lambda: 2 * p1 - p2, 'quad': lambda: 3 * p1 - 3 * p2 + p3, 'fib': lambda: p1 + p2,
           'trib': lambda: p1 + p2 + p3, 'diff': lambda: p1 - p2, 'prod': lambda: p1 * p2,
           'mean3': lambda: (p1 + p2 + p3) // 3, 'lag1': lambda: p1, 'lag2': lambda: p2, 'lag3': lambda: p3,
           'lag4': lambda: p4, 'shl': lambda: np.roll(p1, -1, axis=1), 'shr': lambda: np.roll(p1, 1, axis=1),
           'rev': lambda: p1[:, ::-1], 'mb': lambda: p1 + b1, 'sort': lambda: np.sort(p1, axis=1)}[kind]()
    return out % 10


def recur_fit(kind, sign):
    def fit(draws, ix):
        c = _ctx(draws, ix)
        e = (c['main'][T0:ix] - _pred(c, kind, slice(T0, ix))) % 10
        obs = np.stack([np.bincount(e[:, j], minlength=10) for j in range(6)])
        lift = sign * _lift(obs, (ix - T0) / 10)
        pt = _pred(c, kind, slice(ix, ix + 1))[0]
        return lambda d, g: lift[POS, (_cd(d) - pt) % 10].sum(1)
    return fit


# ---------------------------------------------------------------- 가족 F: 이웃 아닌 자리쌍

PAIRSETS = {'gap2': ('두 칸 떨어진 자리쌍', [(0, 2), (1, 3), (2, 4), (3, 5)]),
            'gap3': ('세 칸 떨어진 자리쌍', [(0, 3), (1, 4), (2, 5)]),
            'gap45': ('네·다섯 칸 떨어진 자리쌍', [(0, 4), (1, 5), (0, 5)]),
            'mirror': ('좌우 대칭 자리쌍', [(0, 5), (1, 4), (2, 3)])}


def cooc_fit(pairs, sign):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ar = np.arange(len(pairs))

    def fit(draws, ix):
        rows = draws.main[:ix].astype(np.int64)
        obs = np.zeros((len(pairs), 10, 10))
        for i in range(len(pairs)):
            np.add.at(obs[i], (rows[:, a[i]], rows[:, b[i]]), 1)
        lift = sign * _lift(obs, ix / 100)
        return lambda d, g: lift[ar, _cd(d)[:, a], _cd(d)[:, b]].sum(1)
    return fit


def pairdiff_fit(pairs, pooled, sign):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ar = np.arange(len(pairs))

    def fit(draws, ix):
        rows = draws.main[:ix].astype(np.int64)
        e = (rows[:, b] - rows[:, a]) % 10
        if pooled:
            lift = sign * _lift(np.bincount(e.ravel(), minlength=10), ix * len(pairs) / 10)
            return lambda d, g: lift[(_cd(d)[:, b] - _cd(d)[:, a]) % 10].sum(1)
        obs = np.stack([np.bincount(e[:, i], minlength=10) for i in ar])
        lift = sign * _lift(obs, ix / 10)
        return lambda d, g: lift[ar, (_cd(d)[:, b] - _cd(d)[:, a]) % 10].sum(1)
    return fit


# ---------------------------------------------------------------- 가족 G: 조 산술

GROUPH = {'S': '자리합', 'd6': '끝자리', 'd1': '첫자리', 'dr': '디지털 루트', 'r': '회차', 'g1': '직전 조',
          'g12': '직전 두 조 합', 'gap': '직전 두 조 등차 2g1-g2', 'S1': '직전 자리합', 'day': '추첨일 날짜'}


def _digit_h(name, x, n):
    if name == 'S':
        return x.sum(1)
    if name == 'd6':
        return x[:, 5]
    if name == 'd1':
        return x[:, 0]
    return np.where(n == 0, 0, 1 + (n - 1) % 9)


def _ctx_h(name, c):
    return {'r': c['r'], 'g1': c['G1'], 'g12': c['G1'] + c['G2'], 'gap': 2 * c['G1'] - c['G2'],
            'S1': c['S1'], 'day': c['day']}[name]


def group_fit(name, sign):
    digit = name in ('S', 'd6', 'd1', 'dr')

    def fit(draws, ix):
        c = _ctx(draws, ix)
        h = slice(T0, ix)
        hh = _digit_h(name, c['main'][h], c['N'][h]) if digit else _ctx_h(name, c)[h]
        lift = sign * _lift(np.bincount((c['grp'][h] - hh) % 5, minlength=5), (ix - T0) / 5)
        if digit:
            return lambda d, g: lift[(np.asarray(g, np.int64) - _digit_h(name, _cd(d), _cN(d))) % 5]
        ht = int(_ctx_h(name, c)[ix])
        return lambda d, g: lift[(np.asarray(g, np.int64) - ht) % 5]
    return fit


# ---------------------------------------------------------------- 공식 목록

SIGNS = ((1, '우세'), (-1, '반대'))


def formulas(draws=None, macro=None):
    """수학 구조 공식 목록. 자료에 따라 미리 계산하는 것이 없어 draws·macro는 쓰지 않는다."""
    out = []

    def add(key, label, make):
        for sign, word in SIGNS:
            out.append(fm.Formula(f'math|{key}|{sign:+d}', f'{label} · {word}', make(sign)))

    for name, (label, _) in STATIC.items():
        add(f'static|{name}', label, lambda s, n=name: static_fit(n, s))
    feat_label = {'sum10': '자리합 끝자리', 'mod7': 'mod 7', 'mod11': 'mod 11', 'shape': '중복 모양',
                  'updown': '오름·내림 걸음'}
    for kind, label in TRANSFORMS.items():
        for name in TRANSFORM_FEATS:
            if kind == 'rev' and name == 'sum10':   # 자리합 끝자리는 자리 순서와 무관해 'main'과 똑같다
                continue
            add(f'trans|{kind}|{name}', f'{label}의 {feat_label[name]}',
                lambda s, k=kind, n=name: transform_fit(k, n, s))
    for m in (7, 9, 11, 13, 37):
        add(f'shift|prev|mod{m}', f'(N - 직전 1등) mod {m}', lambda s, m=m: shift_fit('prev', m, None, s))
    for m in (7, 9, 11, 13):
        add(f'shift|round|mod{m}', f'(N - 회차) mod {m}', lambda s, m=m: shift_fit('round', m, None, s))
    for cname in ('prev', 'ap', 'fib', 'bonus', 'gold', 'rev', 'sq', 'weyl', 'prod', 'quad'):
        add(f'shift|{cname}|bin10', f'(N - {SHIFTS[cname]}) mod 10^6 의 10만 칸',
            lambda s, n=cname: shift_fit(n, M, 10, s))
    add('shift|prev|mod100', '(N - 직전 1등) mod 100 끝 두 자리', lambda s: shift_fit('prev', 100, None, s))
    for kind, label in CTXFEATS.items():
        add(f'ctx|{kind}', label, lambda s, k=kind: ctxfeat_fit(k, s))
    for kind, label in RECURS.items():
        add(f'recur|{kind}', f'자리별 점화식 잔차 {label}', lambda s, k=kind: recur_fit(k, s))
    for name, (label, pairs) in PAIRSETS.items():
        add(f'pair|{name}', f'{label} 동반 출현', lambda s, p=pairs: cooc_fit(p, s))
    add('pair|gap2diff', '두 칸 떨어진 자리 차 (합쳐 셈)', lambda s: pairdiff_fit(PAIRSETS['gap2'][1], True, s))
    add('pair|alldiff', '15개 자리쌍 각각의 차', lambda s: pairdiff_fit(_PAIRS15, False, s))
    for name, label in GROUPH.items():
        add(f'group|{name}', f'(조 - {label}) mod 5', lambda s, n=name: group_fit(n, s))
    return out


# ---------------------------------------------------------------- 실험

def lookahead_check(draws, formula_list, ixs):
    bad = []
    for ix in ixs:
        d, g = eng.split_ids(lab.fold_sample(draws, ix)[:2000])
        short = draws.truncated(ix)
        for f in formula_list:
            a = np.asarray(f.fit(draws, ix)(d, g), dtype=float)
            b = np.asarray(f.fit(short, ix)(d, g), dtype=float)
            if a.shape != b.shape or not np.allclose(a, b):
                bad.append((f.key, ix))
    return bad


def report(fs, pct, fake_pct, draws, info):
    dev, test = pct[:, :100].mean(1), pct[:, 100:].mean(1)
    se_dev = pct[:, :100].std(1, ddof=1) / sqrt(100)
    order = np.argsort(dev)
    n = len(draws.rounds)
    folds = pct.shape[1]
    r0, r1 = int(draws.rounds[n - folds]), int(draws.rounds[n - folds + 99])
    r2, r3 = int(draws.rounds[n - folds + 100]), int(draws.rounds[-1])
    lines = [f'# 수학·정수론·조합 구조 공식 · {len(fs)}개', '',
             f'개발 구간 {r0}~{r1}회 100회에서 고르고 검증 구간 {r2}~{r3}회 {folds - 100}회로 확인했습니다. '
             '숫자는 실제 1등이 무작위 5만 후보 안에서 상위 몇 %에 왔는지의 평균이며 무작위면 50%입니다. '
             '`lab.walk(draws, formulas, 140)` 그대로 쟀습니다.', '',
             f'공식 파일 `pension720/lab_math.py`, 회차별 상위 % 배열 `math_pct.npy` (공식 x 140회, 순서는 formulas() 순서), '
             f'가짜 자료 배열 `math_fake_pct.npy`, 공식 키 `math_keys.npy`. 모든 공식은 부호 +1(우세)·-1(반대) 쌍이라 '
             '한 쌍의 상위 %를 더하면 항상 100.002%이고, 전체 평균은 정확히 50.0%입니다.', '',
             '## 개발 구간 상위 20개와 검증 구간 성적', '',
             '| 순위 | 공식 | 개발 | 개발 표준오차 | 검증 |', '| ---: | --- | ---: | ---: | ---: |']
    lines += [f'| {i} | {fs[k].label} | {dev[k]:.1f}% | {se_dev[k]:.1f} | {test[k]:.1f}% |'
              for i, k in enumerate(order[:20], 1)]
    bt = int(np.argmin(test))
    top20 = order[:20]
    lines += ['',
              f'- 검증 구간 최고 (뒤늦게 고른 것이라 참고만): {fs[bt].label} {test[bt]:.1f}% (개발 {dev[bt]:.1f}%)',
              f'- 개발 상위 20개의 검증 평균: {test[top20].mean():.1f}%, 전체 {len(fs)}개의 검증 평균: {test.mean():.1f}%',
              f'- 개발·검증 성적 상관: {np.corrcoef(dev, test)[0, 1]:+.2f}',
              f'- 개발 45% 아래 공식 {int((dev < 45).sum())}개, 그중 검증도 45% 아래 {int(((dev < 45) & (test < 45)).sum())}개']
    if fake_pct is not None:
        fd, ft = fake_pct[:, :100].mean(1), fake_pct[:, 100:].mean(1)
        fb = int(np.argmin(fd))
        lines += ['', '## 가짜 무작위 자료 (fsm.fake_draws(draws, 3001))에서 같은 실험', '',
                  '| 자료 | 개발 최고 공식 | 개발 | 그 공식의 검증 | 개발 45% 아래 개수 | 검증 최고 |',
                  '| --- | --- | ---: | ---: | ---: | ---: |',
                  f'| 가짜 | {fs[fb].label} | {fd[fb]:.1f}% | {ft[fb]:.1f}% | {int((fd < 45).sum())} | {ft.min():.1f}% |',
                  f'| 실제 | {fs[order[0]].label} | {dev[order[0]]:.1f}% | {test[order[0]]:.1f}% | '
                  f'{int((dev < 45).sum())} | {test.min():.1f}% |']
    lines += ['', '## 미래 정보 누수 검사', '', info.get('lookahead', '아직 안 함'), '',
              '## 실행 시간', '', info.get('time', '')]
    return '\n'.join(lines) + '\n'


def main():
    out = HERE / 'results' / '창의공식'
    out.mkdir(parents=True, exist_ok=True)
    draws = eng.load_draws(eng.DATA_CSV)
    macro = lab.load_macro()
    t = time.time()
    warm()
    prep = time.time() - t
    fs = formulas(draws, macro)
    print(f'공식 {len(fs)}개, 격자 표 준비 {prep:.0f}초', file=sys.stderr, flush=True)
    info = {}
    t = time.time()
    pct, _ = lab.walk(draws, fs, 140)
    real_t = time.time() - t
    np.save(out / 'math_pct.npy', pct)
    np.save(out / 'math_keys.npy', np.array([f.key for f in fs]))
    info['time'] = f'격자 표 준비 {prep:.0f}초, 실제 자료 140회 {real_t:.0f}초 (회차당 {real_t / 140:.2f}초, 공식 {len(fs)}개 전부)'
    (out / 'math.md').write_text(report(fs, pct, None, draws, info), encoding='utf-8')
    n = len(draws.rounds)
    bad = lookahead_check(draws, fs, (n - 40, n - 17, n - 1))
    info['lookahead'] = (f'회차 행 {n - 40}, {n - 17}, {n - 1} 세 곳에서 공식 {len(fs)}개 모두 '
                         f'`fit(draws, ix)`와 `fit(draws.truncated(ix), ix)` 점수가 같은지 2,000개 후보로 비교: '
                         + ('모두 같음 (누수 없음)' if not bad else f'다른 것 {len(bad)}건: {bad[:10]}'))
    (out / 'math.md').write_text(report(fs, pct, None, draws, info), encoding='utf-8')
    fake = fsm.fake_draws(draws, 3001)
    ffs = formulas(fake, macro)
    t = time.time()
    fake_pct, _ = lab.walk(fake, ffs, 140)
    fake_t = time.time() - t
    np.save(out / 'math_fake_pct.npy', fake_pct)
    info['time'] += f', 가짜 자료 140회 {fake_t:.0f}초'
    text = report(fs, pct, fake_pct, draws, info)
    (out / 'math.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
