#!/usr/bin/env python3
"""연금복권720+ 기계학습·베이즈 공식 실험실.

lab.py 와 같은 걷기 검증으로 겨룰 공식들. 모두 '그 회차 이전 자료 + 그 회차의 회차 번호·날짜'만 쓴다.

가족
- 전문가 섞기: 빈도·감쇠·냉각·마르코프·미출현·이월·닮은 회차·변화점 전문가 25개(조 15개)를 과거 회차의
  예측 로그우도로 섞는다. 베이즈 모형평균, 헤지, 잊는 헤지, 고정 나눔, 선두 따라가기, EM 쌓기, 지수 경사,
  톰슨 표집, 반대 학습자, 국면별 메타 학습자, 로그 선형 풀링, 창 길이·감쇠율 모형평균
- 순위 보정: 전문가가 매긴 숫자 순위가 실제로 몇 번째였는지의 과거 분포로 확률을 다시 매김(컨포멀 방식)
- 베이즈: 위계 디리클레 경험적 베이즈, 베이즈 온라인 변화점, 은닉 마르코프, 위계 마르코프, 2단 수축,
  베이지안 릿지·가우스 과정 숫자 회귀
- 자리별 분류기: 자리마다 다음 숫자를 시차·미출현·매크로·달력·국면 특징으로 예측(소프트맥스, 랜덤
  포레스트, 엑스트라 트리, kNN, 나이브 베이즈, 다중 부스팅, MLP, 무작위 특징). 점수 = 자리별 로그확률 합 + 조
- 자리 합침 분류기: 여섯 자리를 한 모델로
- 후보 대 무작위: fm._phi 특징으로 MLP, 무작위 푸리에 특징, 쌍대 순위 학습, LDA, 가우스 나이브 베이즈, 엑스트라 트리
- 시계열 교차검증 쌓기: 분류기들의 표본 밖 예측을 EM으로 섞음

기계학습 모델은 10회마다 그 구간 시작 전 행만으로 다시 학습한다 (lab.Booster 와 같은 방식).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
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

warnings.filterwarnings('ignore')

POS = np.arange(6)
T0 = 20          # 학습·가중치 계산을 시작하는 행
EPS = 0.1        # 분류기 확률에 섞는 균등 분포 비율
BLOCK = 10       # 기계학습 재학습 간격
OUT = HERE / 'results' / '창의공식'


def _norm(p):
    return p / p.sum(axis=-1, keepdims=True)


def _additive(dl, gl):
    dl, gl = np.asarray(dl, float), np.asarray(gl, float)
    return lambda d, g: dl[POS, d].sum(axis=1) + gl[g]


def _from_probs(p, pg, eps=EPS):
    p = (1 - eps) * _norm(np.maximum(p, 1e-12)) + eps / 10
    pg = (1 - eps) * _norm(np.maximum(pg, 1e-12)) + eps / 5
    return _additive(np.log(p), np.log(pg))


def _lgamma(x):
    return np.frompyfunc(math.lgamma, 1, 1)(x).astype(float)


# ---------------------------------------------------------------- 날짜·매크로 (만들 때 한 번)

class _Static:
    """행 0..n 마다 그 추첨 날짜·회차 번호와 그 전에 알 수 있던 매크로·국면 값."""

    def __init__(self, draws, macro):
        dates = lab.draw_dates(draws)
        rounds = np.append(draws.rounds, draws.rounds[-1] + 1).astype(float)
        cols = [rounds % 2]
        for k in range(3, 8):
            cols += [np.sin(2 * np.pi * rounds / k), np.cos(2 * np.pi * rounds / k)]
        month = np.array([d.month for d in dates], float)
        cols += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
        cols.append(np.array([min(3, (d.day - 1) // 7) for d in dates], float))
        moon = np.array([lab.moon_age(d) for d in dates]) / 29.530588853
        cols += [np.sin(2 * np.pi * moon), np.cos(2 * np.pi * moon)]
        self.cal = np.stack(cols, axis=1)
        self.mac = lab.macro_vector(draws, macro)
        states = lab.build_states(draws, macro)
        self.state_names = list(states)
        self.states = np.stack([states[k] for k in self.state_names], axis=1).astype(np.int64)
        self.state = states


# ---------------------------------------------------------------- 자료별 이력 (fit 에 들어온 draws 로 만든다)

def _decay(lam, x):
    out = np.zeros((len(x) + 1,) + x.shape[1:])
    for t in range(1, len(x) + 1):
        out[t] = lam * out[t - 1] + x[t - 1]
    return out


def _bocpd(x, m, hazard, alpha=1.0):
    """베이즈 온라인 변화점(디리클레-다항). x (n, P) → 행 0..n 의 다음 기호 예측 (n+1, P, m). t행 예측은 t 이전만 쓴다."""
    n, p = x.shape
    rmax = n + 2
    r = np.zeros((p, rmax))
    r[:, 0] = 1
    c = np.zeros((p, rmax, m))
    out = np.zeros((n + 1, p, m))
    pi = np.arange(p)
    for t in range(n + 1):
        pr = (c + alpha) / (c.sum(-1, keepdims=True) + m * alpha)
        out[t] = np.einsum('pr,prm->pm', r, pr)
        if t == n:
            break
        v = x[t]
        like = pr[pi, :, v]
        grow = r * like
        new = np.zeros_like(r)
        new[:, 1:] = grow[:, :-1] * (1 - hazard)
        new[:, 0] = grow.sum(1) * hazard
        r = new / new.sum(1, keepdims=True)
        nc = np.zeros_like(c)
        nc[:, 1:] = c[:, :-1]
        nc[pi, 1:, v] += 1
        c = nc
    return out


DIGIT_EXPERTS = ('uni', 'hot_all', 'hot100', 'hot50', 'hot20', 'hot10', 'hot5', 'dec0.8', 'dec0.9', 'dec0.95',
                 'dec0.97', 'dec0.99', 'cold50', 'markov1', 'markov2', 'overdue', 'recent', 'carry', 'avoid',
                 'neighbor', 'bcarry', 'pool50', 'analog30', 'bocpd20', 'bocpd100')
GROUP_EXPERTS = ('uni', 'hot_all', 'hot100', 'hot50', 'hot20', 'hot10', 'dec0.9', 'dec0.97', 'cold50', 'markov1',
                 'overdue', 'recent', 'carry', 'avoid', 'bocpd20')


class _Hist:
    def __init__(self, draws):
        self.draws = draws
        n = self.n = len(draws.rounds)
        m = draws.main.astype(np.int64)
        gr = draws.groups.astype(np.int64) - 1
        b = draws.bonus.astype(np.int64)
        self.y, self.yg = m, gr
        ar = np.arange(n)
        oh = np.zeros((n, 6, 10))
        oh[ar[:, None], POS[None, :], m] = 1
        goh = np.zeros((n, 5))
        goh[ar, gr] = 1
        cum = np.zeros((n + 1, 6, 10))
        cum[1:] = oh.cumsum(0)
        gcum = np.zeros((n + 1, 5))
        gcum[1:] = goh.cumsum(0)
        tt = np.arange(n + 1)

        def win(c, w):
            return c - c[np.maximum(tt - w, 0)]

        age = np.zeros((n + 1, 6, 10))
        gage = np.zeros((n + 1, 5))
        age[0], gage[0] = 30, 10
        for t in range(1, n + 1):
            age[t] = age[t - 1] + 1
            age[t][POS, m[t - 1]] = 0
            gage[t] = gage[t - 1] + 1
            gage[t][gr[t - 1]] = 0
        age, gage = np.minimum(age, 60), np.minimum(gage, 60)
        prev = np.full((n + 1, 6), -1)
        prev[1:] = m
        prev2 = np.full((n + 1, 6), -1)
        prev2[2:] = m[:-1]
        prevb = np.full((n + 1, 6), -1)
        prevb[1:] = b
        prevg = np.full(n + 1, -1)
        prevg[1:] = gr
        self.prev, self.prev2, self.prevg = prev, prev2, prevg
        tr = np.zeros((n + 1, 6, 10, 10))
        tr2 = np.zeros((n + 1, 6, 10, 10))
        gtr = np.zeros((n + 1, 5, 5))
        for t in range(1, n + 1):
            tr[t], tr2[t], gtr[t] = tr[t - 1], tr2[t - 1], gtr[t - 1]
            if t >= 2:
                tr[t][POS, m[t - 2], m[t - 1]] += 1
                gtr[t][gr[t - 2], gr[t - 1]] += 1
            if t >= 3:
                tr2[t][POS, m[t - 3], m[t - 1]] += 1
        self.cum, self.gcum, self.tr, self.gtr = cum, gcum, tr, gtr

        # ---- 숫자 전문가 (n+1, 6, 10)
        ex = {'uni': np.full((n + 1, 6, 10), 0.1), 'hot_all': (cum + 1) / (tt[:, None, None] + 10)}
        for w in (100, 50, 20, 10, 5):
            c = win(cum, w)
            ex[f'hot{w}'] = (c + 1) / (c.sum(-1, keepdims=True) + 10)
        for lam in (0.8, 0.9, 0.95, 0.97, 0.99):
            dd = _decay(lam, oh)
            ex[f'dec{lam:g}'] = (dd + 0.5) / (dd.sum(-1, keepdims=True) + 5)
        ex['cold50'] = _norm(1 / (win(cum, 50) + 1))
        ti = tt[:, None]
        pc = np.maximum(prev, 0)
        mk = tr[ti, POS[None, :], pc]
        ex['markov1'] = np.where((prev >= 0)[..., None], (mk + 1) / (mk.sum(-1, keepdims=True) + 10), 0.1)
        mk2 = tr2[ti, POS[None, :], np.maximum(prev2, 0)]
        ex['markov2'] = np.where((prev2 >= 0)[..., None], (mk2 + 1) / (mk2.sum(-1, keepdims=True) + 10), 0.1)
        ex['overdue'] = _norm(np.log(2 + age))
        ex['recent'] = _norm(1 / np.sqrt(1 + age))
        is_prev = (prev[..., None] == np.arange(10))
        ex['carry'] = np.where(is_prev, 0.19, 0.09)
        ex['carry'][0] = 0.1
        ex['avoid'] = np.where(is_prev, 0.02, 0.98 / 9)
        ex['avoid'][0] = 0.1
        nb = 1.0 + (np.abs(prev[..., None] - np.arange(10)) == 1)
        nb[0] = 1
        ex['neighbor'] = _norm(nb)
        ex['bcarry'] = np.where(prevb[..., None] == np.arange(10), 0.19, 0.09)
        ex['bcarry'][0] = 0.1
        pool = win(cum, 50).sum(1, keepdims=True)
        ex['pool50'] = np.repeat((pool + 1) / (pool.sum(-1, keepdims=True) + 10), 6, axis=1)
        an = np.full((n + 1, 6, 10), 0.1)
        for t in range(40, n + 1):
            past = m[:t - 2]
            sim = (past == m[t - 1]).sum(1) + 1e-3 * np.arange(t - 2)
            near = np.argsort(-sim)[:30] + 1
            cnt = oh[near].sum(0)
            an[t] = (cnt + 1) / (30 + 10)
        ex['analog30'] = an
        ex['bocpd20'] = _bocpd(m, 10, 1 / 20)
        ex['bocpd100'] = _bocpd(m, 10, 1 / 100)
        self.P = np.stack([ex[k] for k in DIGIT_EXPERTS], axis=1)          # (n+1, E, 6, 10)

        # ---- 조 전문가 (n+1, 5)
        gx = {'uni': np.full((n + 1, 5), 0.2), 'hot_all': (gcum + 1) / (tt[:, None] + 5)}
        for w in (100, 50, 20, 10):
            c = win(gcum, w)
            gx[f'hot{w}'] = (c + 1) / (c.sum(-1, keepdims=True) + 5)
        for lam in (0.9, 0.97):
            dd = _decay(lam, goh)
            gx[f'dec{lam:g}'] = (dd + 0.5) / (dd.sum(-1, keepdims=True) + 2.5)
        gx['cold50'] = _norm(1 / (win(gcum, 50) + 1))
        gm = gtr[tt, np.maximum(prevg, 0)]
        gx['markov1'] = np.where((prevg >= 0)[:, None], (gm + 1) / (gm.sum(-1, keepdims=True) + 5), 0.2)
        gx['overdue'] = _norm(np.log(2 + gage))
        gx['recent'] = _norm(1 / np.sqrt(1 + gage))
        gp = prevg[:, None] == np.arange(5)
        gx['carry'] = np.where(gp, 0.36, 0.16)
        gx['carry'][0] = 0.2
        gx['avoid'] = np.where(gp, 0.05, 0.2375)
        gx['avoid'][0] = 0.2
        gx['bocpd20'] = _bocpd(gr[:, None], 5, 1 / 20)[:, 0]
        self.Pg = np.stack([gx[k] for k in GROUP_EXPERTS], axis=1)         # (n+1, Eg, 5)

        e, eg = len(DIGIT_EXPERTS), len(GROUP_EXPERTS)
        tn = np.arange(n)
        self.LL = np.log(self.P[tn[:, None, None], np.arange(e)[None, :, None], POS[None, None, :], m[:, None, :]])
        self.LLg = np.log(self.Pg[tn[:, None], np.arange(eg)[None, :], gr[:, None]])
        self.rank = np.argsort(np.argsort(-self.P, axis=-1, kind='stable'), axis=-1, kind='stable')
        self.rankg = np.argsort(np.argsort(-self.Pg, axis=-1, kind='stable'), axis=-1, kind='stable')
        self.RA = self.rank[tn[:, None, None], np.arange(e)[None, :, None], POS[None, None, :], m[:, None, :]]
        self.RAg = self.rankg[tn[:, None], np.arange(eg)[None, :], gr[:, None]]

        # ---- 분류기 특징
        sd = np.maximum(tt, 1)[:, None, None]
        w100 = win(cum, 100) / np.minimum(sd, 100)
        self.Xpos = np.concatenate([
            (prev[..., None] == np.arange(10)), (prev2[..., None] == np.arange(10)), np.log1p(age),
            win(cum, 10) / 10, win(cum, 30) / 30, w100, (prevb[..., None] == np.arange(10))], axis=-1).astype(float)
        prow = np.zeros((n + 1, 6))
        prow[1:] = m
        self.Xsh = np.hstack([
            (prevg[:, None] == np.arange(5)), np.log1p(gage), win(gcum, 30) / 30,
            ((prow.sum(1) - 27) / 8)[:, None], (prow % 2).sum(1, keepdims=True), (prow >= 5).sum(1, keepdims=True),
            np.array([len(set(r)) for r in prow])[:, None]]).astype(float)
        self.models, self.tables = {}, {}
        self._train = None
        self._phi = (None, None, None)

    @property
    def train(self):
        if self._train is None:
            self._train = fm._TrainRows(self.draws)
        return self._train

    def phi(self, d, g, ix):
        last_d, last_ix, x = self._phi
        if last_d is d and last_ix == ix:
            return x
        x = fm._phi(d, g, fm._context(self.draws, ix))
        self._phi = (d, ix, x)
        return x


_HIST = {}


def _hist(draws):
    h = _HIST.get(id(draws))
    if h is None or h.draws is not draws:
        h = _Hist(draws)
        _HIST[id(draws)] = h
    return h


def _idx(names, chosen):
    return [names.index(c) for c in chosen]


# ---------------------------------------------------------------- 전문가 섞기

def _softmax0(s):
    s = s - s.max(0, keepdims=True)
    w = np.exp(s)
    return w / w.sum(0, keepdims=True)


def _weights(kind, ll, rng, **kw):
    """ll (T, E, P) 과거 로그우도 → 전문가 가중치 (E, P)."""
    t, e, p = ll.shape
    if kind == 'bma':
        return _softmax0(kw.get('eta', 1.0) * ll.sum(0))
    if kind == 'fade':
        disc = kw['gamma'] ** np.arange(t)[::-1]
        return _softmax0((ll * disc[:, None, None]).sum(0))
    if kind == 'fixshare':
        w = np.full((e, p), 1 / e)
        a = kw['alpha']
        for row in ll:
            w = w * np.exp(row - row.max(0))
            w = w / w.sum(0)
            w = (1 - a) * w + a / e
        return w
    if kind in ('ftl', 'worst'):
        s = ll.sum(0)
        pick = s.argmax(0) if kind == 'ftl' else s.argmin(0)
        w = np.zeros((e, p))
        w[pick, np.arange(p)] = 1
        return w
    if kind == 'em':
        w = np.full((e, p), 1 / e)
        lik = np.exp(ll)
        for _ in range(60):
            r = w * lik
            r = r / r.sum(1, keepdims=True)
            w = r.mean(0)
        return w
    if kind == 'eg':
        w = np.full((e, p), 1 / e)
        for row in np.exp(ll):
            mix = (w * row).sum(0)
            w = w * np.exp(kw['eta'] * row / mix)
            w = w / w.sum(0)
        return w
    if kind == 'ts_beta':
        succ = (ll > np.log(1 / kw['m'])).sum(0)
        draw = rng.beta(1 + succ, 1 + t - succ)
        w = np.zeros((e, p))
        w[draw.argmax(0), np.arange(p)] = 1
        return w
    if kind == 'ts_dir':
        best = ll.argmax(1)
        cnt = np.stack([np.bincount(best[:, j], minlength=e) for j in range(p)], axis=1)
        return np.stack([rng.dirichlet(1 + cnt[:, j]) for j in range(p)], axis=1)
    raise ValueError(kind)


class _Mixer:
    def __init__(self, st, kind, window=None, shared=False, experts=None, gexperts=None, state=None, **kw):
        self.st, self.kind, self.window, self.shared, self.state, self.kw = st, kind, window, shared, state, kw
        self.ei = _idx(DIGIT_EXPERTS, experts) if experts else list(range(len(DIGIT_EXPERTS)))
        self.gi = _idx(GROUP_EXPERTS, gexperts) if gexperts else list(range(len(GROUP_EXPERTS)))

    def __call__(self, draws, ix):
        h = _hist(draws)
        lo = max(T0, ix - self.window) if self.window else T0
        rows = np.arange(lo, ix)
        if self.state is not None:
            s = self.st.state[self.state]
            if s[ix] >= 0:
                rows = rows[s[rows] == s[ix]]
        rng = np.random.default_rng(int(draws.rounds[ix - 1]) * 7919 + 13)
        ll = h.LL[rows][:, self.ei]
        llg = h.LLg[rows][:, self.gi][..., None]
        if self.shared:
            w = _weights(self.kind, ll.sum(2, keepdims=True), rng, m=10 ** 6, **self.kw)
            w = np.repeat(w, 6, axis=1)
        else:
            w = _weights(self.kind, ll, rng, m=10, **self.kw)
        wg = _weights(self.kind, llg, rng, m=5, **self.kw)[:, 0]
        p = np.einsum('ej,ejv->jv', w, h.P[ix, self.ei])
        pg = wg @ h.Pg[ix, self.gi]
        return _from_probs(p, pg, eps=0.0)


class _LogLinear:
    """로그 선형 풀링: 점수 = Σ θ_e log p_e, θ 는 과거 회차 조건부 로짓 최대우도(뉴턴법, L2)."""

    def __init__(self, window=None, l2=1.0):
        self.window, self.l2 = window, l2

    @staticmethod
    def _fit(logp, y, l2):
        # logp (N, E, M), y (N,)
        n, e, m = logp.shape
        theta = np.zeros(e)
        obs = logp[np.arange(n), :, y]                       # (N, E)
        for _ in range(20):
            z = np.einsum('e,nem->nm', theta, logp)
            z -= z.max(1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(1, keepdims=True)
            mean = np.einsum('nm,nem->ne', q, logp)
            grad = (obs - mean).sum(0) - l2 * theta
            flat = logp.transpose(1, 0, 2).reshape(e, -1)
            second = (flat * q.reshape(1, -1)) @ flat.T - mean.T @ mean
            step = np.linalg.solve(second + l2 * np.eye(e), grad)
            theta += step
            if np.abs(step).max() < 1e-6:
                break
        return theta

    def __call__(self, draws, ix):
        h = _hist(draws)
        lo = max(T0, ix - self.window) if self.window else T0
        lp = np.log(h.P[lo:ix])                              # (T, E, 6, 10)
        t = ix - lo
        logp = lp.transpose(0, 2, 1, 3).reshape(t * 6, lp.shape[1], 10)
        theta = self._fit(logp, h.y[lo:ix].reshape(-1), self.l2)
        lpg = np.log(h.Pg[lo:ix])
        tg = self._fit(lpg, h.yg[lo:ix], self.l2)
        dl = np.einsum('e,ejv->jv', theta, np.log(h.P[ix]))
        gl = tg @ np.log(h.Pg[ix])
        return _additive(dl, gl)


class _Conformal:
    """순위 보정: 전문가 순위 r 에 실제 숫자가 온 과거 비율 q(r)로 확률을 다시 매긴다. 여러 전문가면 순위 나이브 베이즈."""

    def __init__(self, experts, gexperts, per_position=False, window=None):
        self.ei, self.gi = _idx(DIGIT_EXPERTS, experts), _idx(GROUP_EXPERTS, gexperts)
        self.per_position, self.window = per_position, window

    def __call__(self, draws, ix):
        h = _hist(draws)
        lo = max(T0, ix - self.window) if self.window else T0
        dl = np.zeros((6, 10))
        for e in self.ei:
            ra = h.RA[lo:ix, e]                                  # (T, 6)
            if self.per_position:
                q = np.stack([np.bincount(ra[:, j], minlength=10) for j in range(6)]) + 1.0
                q = np.log(q / q.sum(1, keepdims=True))
                dl += q[POS[:, None], h.rank[ix, e]]
            else:
                q = np.bincount(ra.ravel(), minlength=10) + 1.0
                dl += np.log(q / q.sum())[h.rank[ix, e]]
        gl = np.zeros(5)
        for e in self.gi:
            q = np.bincount(h.RAg[lo:ix, e], minlength=5) + 1.0
            gl += np.log(q / q.sum())[h.rankg[ix, e]]
        return _additive(dl, gl)


# ---------------------------------------------------------------- 베이즈 모형

GRID = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000], float)


def _dirmult_best(counts, base):
    """counts (R, M), base (R, M) 사전 평균 → 증거 최대인 집중도 α."""
    best, best_a = -np.inf, GRID[0]
    nsum = counts.sum(1)
    for a in GRID:
        am = a * base
        ev = (_lgamma(am + counts) - _lgamma(am)).sum() + (_lgamma(np.full(len(counts), a)) - _lgamma(a + nsum)).sum()
        if ev > best:
            best, best_a = ev, a
    return best_a


def eb_dirichlet(window):
    """위계 디리클레: 자리별 빈도를 여섯 자리 합친 분포 쪽으로 수축, 수축 세기는 주변우도 최대."""
    def fit(draws, ix):
        h = _hist(draws)
        lo = max(0, ix - window) if window else 0
        c = h.cum[ix] - h.cum[lo]
        pooled = (c.sum(0) + 1) / (c.sum() + 10)
        base = np.repeat(pooled[None], 6, 0)
        a = _dirmult_best(c, base)
        p = (c + a * base) / (c.sum(1, keepdims=True) + a)
        cg = (h.gcum[ix] - h.gcum[lo])[None]
        ag = _dirmult_best(cg, np.full((1, 5), 0.2))
        pg = (cg[0] + ag * 0.2) / (cg.sum() + ag)
        return _from_probs(p, pg, eps=0.0)
    return fit


def eb_markov(draws, ix):
    """위계 마르코프: 전이 행을 그 자리 전체 분포 쪽으로 수축, 세기는 주변우도 최대."""
    h = _hist(draws)
    tr, cum = h.tr[ix], h.cum[ix]
    p = np.zeros((6, 10))
    for j in range(6):
        base = (cum[j] + 1) / (cum[j].sum() + 10)
        a = _dirmult_best(tr[j], np.repeat(base[None], 10, 0))
        prev = h.prev[ix, j]
        p[j] = (tr[j, prev] + a * base) / (tr[j, prev].sum() + a)
    gbase = (h.gcum[ix] + 1) / (h.gcum[ix].sum() + 5)
    ag = _dirmult_best(h.gtr[ix], np.repeat(gbase[None], 5, 0))
    pg = (h.gtr[ix, h.prevg[ix]] + ag * gbase) / (h.gtr[ix, h.prevg[ix]].sum() + ag)
    return _from_probs(p, pg, eps=0.0)


def shrink2(recent, a, b):
    """2단 수축: 최근 recent회 → 전체 → 균등."""
    def fit(draws, ix):
        h = _hist(draws)
        c = h.cum[ix] - h.cum[max(0, ix - recent)]
        q = (h.cum[ix] + b * 0.1) / (ix + b)
        p = (c + a * q) / (c.sum(1, keepdims=True) + a)
        cg = h.gcum[ix] - h.gcum[max(0, ix - recent)]
        qg = (h.gcum[ix] + b * 0.2) / (ix + b)
        pg = (cg + a * qg) / (cg.sum() + a)
        return _from_probs(p, pg, eps=0.0)
    return fit


def expert(name, gname):
    """전문가 하나 그대로 (변화점 모형 등)."""
    e, eg = DIGIT_EXPERTS.index(name), GROUP_EXPERTS.index(gname)

    def fit(draws, ix):
        h = _hist(draws)
        return _from_probs(h.P[ix, e], h.Pg[ix, eg], eps=0.0)
    return fit


def _hmm_fit(x, k, m, iters, seed):
    """여러 수열을 한꺼번에(Baum-Welch). x (T, P)."""
    t_len, p = x.shape
    rng = np.random.default_rng(seed)
    a = np.full((p, k, k), 0.2 / (k - 1))
    a[:, np.arange(k), np.arange(k)] = 0.8
    b = rng.dirichlet(np.full(m, 5.0), size=(p, k))
    pi = np.full((p, k), 1 / k)
    pidx = np.arange(p)[None, :]
    onehot = (x[..., None] == np.arange(m)).astype(float)    # (T, P, M)
    for _ in range(iters):
        em = b[pidx, :, x]                                     # (T, P, K)
        al = np.zeros((t_len, p, k))
        c = np.zeros((t_len, p))
        al[0] = pi * em[0]
        c[0] = al[0].sum(-1)
        al[0] /= c[0][:, None]
        for t in range(1, t_len):
            al[t] = np.einsum('pk,pkl->pl', al[t - 1], a) * em[t]
            c[t] = al[t].sum(-1)
            al[t] /= c[t][:, None]
        be = np.ones((t_len, p, k))
        for t in range(t_len - 2, -1, -1):
            be[t] = np.einsum('pkl,pl->pk', a, em[t + 1] * be[t + 1]) / c[t + 1][:, None]
        gam = al * be
        gam /= gam.sum(-1, keepdims=True)
        xi = np.einsum('tpk,pkl,tpl->pkl', al[:-1], a, em[1:] * be[1:] / c[1:, :, None])
        a = (xi + 0.1) / (xi + 0.1).sum(-1, keepdims=True)
        b = np.einsum('tpk,tpm->pkm', gam, onehot) + 1.0
        b /= b.sum(-1, keepdims=True)
        pi = (gam[0] + 0.1) / (gam[0] + 0.1).sum(-1, keepdims=True)
    return pi, a, b


def _hmm_next(x, params):
    pi, a, b = params
    p = x.shape[1]
    pidx = np.arange(p)[None, :]
    em = b[pidx, :, x]
    al = pi * em[0]
    al /= al.sum(-1, keepdims=True)
    for t in range(1, len(x)):
        al = np.einsum('pk,pkl->pl', al, a) * em[t]
        al /= al.sum(-1, keepdims=True)
    nxt = np.einsum('pk,pkl->pl', al, a)
    return np.einsum('pk,pkm->pm', nxt, b)


class _HMM:
    def __init__(self, k, iters=25):
        self.k, self.iters = k, iters

    def __call__(self, draws, ix):
        h = _hist(draws)
        block = ix - ix % BLOCK
        key = (self, block)
        if key not in h.models:
            h.models[key] = (_hmm_fit(h.y[:block], self.k, 10, self.iters, 1),
                             _hmm_fit(h.yg[:block, None], self.k, 5, self.iters, 2))
        pd, pg = h.models[key]
        p = _hmm_next(h.y[:ix], pd)
        q = _hmm_next(h.yg[:ix, None], pg)[0]
        return _from_probs(p, q, eps=0.0)


# ---------------------------------------------------------------- 자리별 분류기

class _PerPos:
    """자리마다(또는 합쳐서) 다음 숫자 분류기 + 조 분류기. 10회마다 그 구간 시작 전 행으로 학습."""

    def __init__(self, st, make, feats='lag', pooled=False, window=None, reg=False):
        self.st, self.make, self.feats, self.pooled, self.window, self.reg = st, make, feats, pooled, window, reg
        if feats == 'state':
            top = st.states.max(0) + 2
            self.cats_pos = np.r_[top, 11, 11]
            self.cats_grp = np.r_[top, 6]

    def _x(self, h, ts, j):
        if self.feats == 'state':
            parts = [self.st.states[ts] + 1]
            parts += [h.prev[ts, j, None] + 1, h.prev2[ts, j, None] + 1] if j is not None else [h.prevg[ts, None] + 1]
            return np.hstack(parts)
        parts = []
        if self.feats in ('lag', 'all'):
            parts += [h.Xpos[ts, j], h.Xsh[ts]] if j is not None else [h.Xsh[ts]]
        if self.feats in ('macro', 'all'):
            parts += [self.st.mac[ts], self.st.cal[ts]]
        return np.hstack(parts)

    def _model(self, j):
        if self.feats == 'state':
            return self.make(self.cats_pos if j is not None else self.cats_grp)
        return self.make()

    def _train(self, h, block):
        lo = max(T0, block - self.window) if self.window else T0
        ts = np.arange(lo, block)
        if self.pooled:
            x = np.vstack([np.hstack((self._x(h, ts, j), np.tile(np.eye(6)[j], (len(ts), 1)))) for j in range(6)])
            y = np.concatenate([h.y[ts, j] for j in range(6)])
            pos = [self._model(0).fit(x, y)]
        else:
            pos = [self._model(j).fit(self._x(h, ts, j), h.y[ts, j].astype(float) if self.reg else h.y[ts, j])
                   for j in range(6)]
        grp = self._model(None).fit(self._x(h, ts, None), (h.yg[ts] + 1.0) if self.reg else h.yg[ts])
        return pos, grp

    def _dist(self, model, x, m):
        if self.reg:
            mu, sd = model.predict(x, return_std=True)
            sd = np.maximum(sd, 0.5)
            v = np.arange(m) + (1 if m == 5 else 0)
            return _norm(np.exp(-0.5 * ((v[None] - mu[:, None]) / sd[:, None]) ** 2))
        out = np.zeros((len(x), m))
        out[:, np.asarray(model.classes_, int)] = model.predict_proba(x)
        return out

    def table(self, h, ix):
        key = (self, ix)
        if key in h.tables:
            return h.tables[key]
        block = ix - ix % BLOCK
        mk = (self, block)
        if mk not in h.models:
            h.models[mk] = self._train(h, block)
        pos, grp = h.models[mk]
        t = np.array([ix])
        if self.pooled:
            x = np.vstack([np.hstack((self._x(h, t, j), np.eye(6)[j][None])) for j in range(6)])
            p = self._dist(pos[0], x, 10)
        else:
            p = np.vstack([self._dist(pos[j], self._x(h, t, j), 10) for j in range(6)])
        pg = self._dist(grp, self._x(h, t, None), 5)[0]
        p = (1 - EPS) * _norm(np.maximum(p, 1e-12)) + EPS / 10
        pg = (1 - EPS) * _norm(np.maximum(pg, 1e-12)) + EPS / 5
        h.tables[key] = (p, pg)
        return p, pg

    def __call__(self, draws, ix):
        p, pg = self.table(_hist(draws), ix)
        return _additive(np.log(p), np.log(pg))


# ---------------------------------------------------------------- 후보 대 무작위

class _Cand:
    """fm._TrainRows(실제 1등 1개 + 무작위 30개) 로 '실제 대 무작위' 학습. pairwise면 같은 회차 안 쌍 차이로 순위 학습."""

    def __init__(self, st, make, with_macro=False, pairwise=False, window=None, decision=False):
        self.st, self.make, self.with_macro, self.pairwise = st, make, with_macro, pairwise
        self.window, self.decision = window, decision

    def _train(self, h, block):
        lo = max(10, block - self.window) if self.window else 10
        xs, ys = [], []
        for t in range(lo, block):
            x, y = h.train.get(t)
            if self.pairwise:
                diff = x[-1] - x[:-1]
                xs += [diff, -diff]
                ys += [np.ones(len(diff)), np.zeros(len(diff))]
            else:
                if self.with_macro:
                    x = np.hstack((x, np.repeat(self.st.mac[t][None], len(x), 0)))
                xs.append(x)
                ys.append(y)
        return self.make().fit(np.vstack(xs), np.concatenate(ys))

    def __call__(self, draws, ix):
        h = _hist(draws)
        block = ix - ix % BLOCK
        key = (self, block)
        if key not in h.models:
            h.models[key] = self._train(h, block)
        model, mx = h.models[key], self.st.mac[ix]

        def score(d, g):
            x = h.phi(d, g, ix)
            if self.with_macro:
                x = np.hstack((x, np.repeat(mx[None], len(x), 0)))
            if self.decision:
                return model.decision_function(x)
            return model.predict_proba(x)[:, 1]
        return score


# ---------------------------------------------------------------- 시계열 교차검증 쌓기

class _StackOOS:
    """분류기들의 표본 밖 예측(각 행은 그 행 이전 구간에서 학습한 모델)과 전문가 몇 개를 EM 으로 섞는다."""

    def __init__(self, bases, experts, gexperts, window=60, kind='em'):
        self.bases, self.window, self.kind = bases, window, kind
        self.ei, self.gi = _idx(DIGIT_EXPERTS, experts), _idx(GROUP_EXPERTS, gexperts)

    def _probs(self, h, t):
        ps, pgs = [], []
        for b in self.bases:
            p, pg = b.table(h, t)
            ps.append(p)
            pgs.append(pg)
        ps += list(h.P[t, self.ei])
        pgs += list(h.Pg[t, self.gi])
        return np.stack(ps), np.stack(pgs)          # (K, 6, 10), (K, 5)

    def __call__(self, draws, ix):
        h = _hist(draws)
        lo = max(100, ix - self.window)
        ll, llg = [], []
        for t in range(lo, ix):
            p, pg = self._probs(h, t)
            ll.append(np.log(p[:, POS, h.y[t]]))
            llg.append(np.log(pg[:, h.yg[t]]))
        rng = np.random.default_rng(0)
        w = _weights(self.kind, np.stack(ll), rng)
        wg = _weights(self.kind, np.stack(llg)[..., None], rng)[:, 0]
        p, pg = self._probs(h, ix)
        return _from_probs(np.einsum('kj,kjv->jv', w, p), wg @ pg, eps=0.0)


# ---------------------------------------------------------------- 공식 목록

def formulas(draws, macro):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import BayesianRidge, LogisticRegression
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    st = _Static(draws, macro)
    out = []

    def add(key, label, fit):
        out.append(fm.Formula('ml|' + key, label, fit))

    # ---- 전문가 섞기
    add('bma|pos', '베이즈 모형평균 자리별', _Mixer(st, 'bma'))
    add('bma|shared', '베이즈 모형평균 여섯 자리 공통', _Mixer(st, 'bma', shared=True))
    add('hedge|0.3', '헤지 학습률0.3', _Mixer(st, 'bma', eta=0.3))
    add('hedge|3', '헤지 학습률3', _Mixer(st, 'bma', eta=3.0))
    add('fade|0.9', '잊는 헤지 0.9', _Mixer(st, 'fade', gamma=0.9))
    add('fade|0.97', '잊는 헤지 0.97', _Mixer(st, 'fade', gamma=0.97))
    add('fixshare|0.05', '고정 나눔 헤지 0.05', _Mixer(st, 'fixshare', alpha=0.05))
    add('ftl|all', '선두 전문가 따라가기 전체', _Mixer(st, 'ftl'))
    add('ftl|30', '선두 전문가 따라가기 최근30회', _Mixer(st, 'ftl', window=30))
    add('em|all', 'EM 쌓기 전체', _Mixer(st, 'em'))
    add('em|60', 'EM 쌓기 최근60회', _Mixer(st, 'em', window=60))
    add('eg|0.05', '지수 경사 0.05', _Mixer(st, 'eg', eta=0.05))
    add('eg|0.5', '지수 경사 0.5', _Mixer(st, 'eg', eta=0.5))
    add('ts|beta', '톰슨 표집 베타', _Mixer(st, 'ts_beta'))
    add('ts|dir', '톰슨 표집 디리클레 가중치', _Mixer(st, 'ts_dir'))
    add('worst|all', '반대 학습자 꼴찌 전문가', _Mixer(st, 'worst'))
    for name in ('직전 조', '계절', '달 위상 4', '회차 mod 2', '자리합 이평 교차', 'usdkrw 4주 방향'):
        add(f'regime|{name}', f'국면별 메타 학습자 {name}', _Mixer(st, 'bma', state=name))
    add('loglin|all', '로그 선형 풀링 전체', _LogLinear())
    add('loglin|100', '로그 선형 풀링 최근100회', _LogLinear(window=100, l2=3.0))
    add('bmawin', '창 길이 베이즈 모형평균',
        _Mixer(st, 'bma', experts=['hot5', 'hot10', 'hot20', 'hot50', 'hot100', 'hot_all'],
               gexperts=['hot10', 'hot20', 'hot50', 'hot100', 'hot_all']))
    add('bmadecay', '감쇠율 베이즈 모형평균',
        _Mixer(st, 'bma', experts=['dec0.8', 'dec0.9', 'dec0.95', 'dec0.97', 'dec0.99', 'hot_all'],
               gexperts=['dec0.9', 'dec0.97', 'hot_all']))
    add('ebdecay', '감쇠율 경험적 베이즈 선택',
        _Mixer(st, 'ftl', experts=['dec0.8', 'dec0.9', 'dec0.95', 'dec0.97', 'dec0.99', 'hot_all'],
               gexperts=['dec0.9', 'dec0.97', 'hot_all']))

    # ---- 순위 보정
    add('conf|hot50', '순위 보정 빈도50', _Conformal(['hot50'], ['hot50']))
    add('conf|overdue', '순위 보정 미출현', _Conformal(['overdue'], ['overdue']))
    add('conf|markov1', '순위 보정 마르코프', _Conformal(['markov1'], ['markov1']))
    add('confnb', '순위 보정 나이브 베이즈 5전문가',
        _Conformal(['hot50', 'hot10', 'markov1', 'overdue', 'dec0.9'], ['hot50', 'hot10', 'markov1', 'overdue']))
    add('confpos|hot20', '순위 보정 빈도20 자리별', _Conformal(['hot20'], ['hot20'], per_position=True))

    # ---- 베이즈
    add('ebdir|all', '위계 디리클레 경험적 베이즈 전체', eb_dirichlet(None))
    add('ebdir|100', '위계 디리클레 경험적 베이즈 최근100회', eb_dirichlet(100))
    add('bocpd|20', '베이즈 변화점 위험1/20', expert('bocpd20', 'bocpd20'))
    add('bocpd|100', '베이즈 변화점 위험1/100', expert('bocpd100', 'bocpd20'))
    add('hmm|2', '은닉 마르코프 상태2', _HMM(2))
    add('hmm|3', '은닉 마르코프 상태3', _HMM(3))
    add('ebmarkov', '위계 마르코프 경험적 베이즈', eb_markov)
    add('shrink2', '2단 수축 최근20→전체→균등', shrink2(20, 20.0, 50.0))
    add('bridge|lag', '베이지안 릿지 숫자 회귀 시차',
        _PerPos(st, lambda: make_pipeline(StandardScaler(), BayesianRidge()), 'lag', reg=True))
    add('bridge|all', '베이지안 릿지 숫자 회귀 전체 특징',
        _PerPos(st, lambda: make_pipeline(StandardScaler(), BayesianRidge()), 'all', reg=True))
    add('gpr|macro', '가우스 과정 숫자 회귀 매크로',
        _PerPos(st, lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            ConstantKernel(0.1) * RBF(10.0) + WhiteKernel(1.0), normalize_y=True, random_state=0)), 'macro', reg=True))

    # ---- 자리별 분류기
    def softmax(c):
        return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300))

    def rf(leaf):
        return lambda: RandomForestClassifier(100, min_samples_leaf=leaf, max_features='sqrt', random_state=0)

    def knn(k):
        return lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(k))

    pp = {
        'softmax|lag|0.05': ('소프트맥스 자리별 시차 C0.05', softmax(0.05), 'lag', None),
        'softmax|all|0.05': ('소프트맥스 자리별 전체 특징 C0.05', softmax(0.05), 'all', None),
        'softmax|all|0.5': ('소프트맥스 자리별 전체 특징 C0.5', softmax(0.5), 'all', None),
        'softmax|macro|0.05': ('소프트맥스 자리별 매크로·달력 C0.05', softmax(0.05), 'macro', None),
        'softmax|lag|w100': ('소프트맥스 자리별 시차 최근100회 학습', softmax(0.05), 'lag', 100),
        'rf|all|5': ('랜덤 포레스트 자리별 전체 잎5', rf(5), 'all', None),
        'rf|lag|20': ('랜덤 포레스트 자리별 시차 잎20', rf(20), 'lag', None),
        'rf|all|w100': ('랜덤 포레스트 자리별 전체 최근100회 학습', rf(10), 'all', 100),
        'et|all|10': ('엑스트라 트리 자리별 전체 잎10',
                      lambda: ExtraTreesClassifier(100, min_samples_leaf=10, random_state=0), 'all', None),
        'knn|lag|20': ('kNN 자리별 시차 이웃20', knn(20), 'lag', None),
        'knn|macro|50': ('kNN 자리별 매크로·달력 이웃50', knn(50), 'macro', None),
        'gnb|macro': ('가우스 나이브 베이즈 자리별 매크로·달력', lambda: make_pipeline(StandardScaler(), GaussianNB()),
                      'macro', None),
        'hgb|all': ('다중 부스팅 자리별 전체 특징', lambda: HistGradientBoostingClassifier(
            max_iter=30, max_depth=2, learning_rate=0.05, l2_regularization=1.0, random_state=0), 'all', None),
        'mlp|lag': ('MLP 자리별 시차', lambda: make_pipeline(StandardScaler(), MLPClassifier(
            (16,), alpha=1.0, max_iter=300, random_state=0)), 'lag', None),
        'rff|all': ('무작위 푸리에 특징 소프트맥스 자리별', lambda: make_pipeline(
            StandardScaler(), RBFSampler(gamma=0.005, n_components=200, random_state=0),
            LogisticRegression(C=0.05, max_iter=300)), 'all', None),
    }
    bases = {}
    for key, (label, make, feats, window) in pp.items():
        obj = _PerPos(st, make, feats=feats, window=window)
        bases[key] = obj
        add('pp|' + key, label, obj)
    add('pp|cnb|state', '범주 나이브 베이즈 자리별 국면87개',
        _PerPos(st, lambda cats: CategoricalNB(alpha=1.0, min_categories=cats), feats='state'))

    # ---- 자리 합침
    add('pool|softmax|lag', '소프트맥스 여섯 자리 합침 시차', _PerPos(st, softmax(0.05), 'lag', pooled=True))
    add('pool|rf|lag', '랜덤 포레스트 여섯 자리 합침 시차', _PerPos(st, rf(20), 'lag', pooled=True))
    add('pool|knn|lag', 'kNN 여섯 자리 합침 시차 이웃60', _PerPos(st, knn(60), 'lag', pooled=True))
    add('pool|mlp|all', 'MLP 여섯 자리 합침 전체 특징', _PerPos(st, lambda: make_pipeline(
        StandardScaler(), MLPClassifier((32,), alpha=1.0, max_iter=200, random_state=0)), 'all', pooled=True))

    # ---- 후보 대 무작위
    add('cand|mlp', 'MLP 실제 대 무작위', _Cand(st, lambda: make_pipeline(
        StandardScaler(), MLPClassifier((32,), alpha=0.1, max_iter=100, random_state=0))))
    add('cand|mlp_macro', 'MLP 실제 대 무작위 매크로 포함', _Cand(st, lambda: make_pipeline(
        StandardScaler(), MLPClassifier((32, 8), alpha=0.1, max_iter=100, random_state=0)), with_macro=True))
    add('cand|rff', '무작위 푸리에 특징 로지스틱 실제 대 무작위', _Cand(st, lambda: make_pipeline(
        StandardScaler(), RBFSampler(gamma=0.01, n_components=300, random_state=0),
        LogisticRegression(C=0.1, max_iter=300))))
    add('cand|pair_logit', '쌍대 순위 로지스틱', _Cand(st, lambda: make_pipeline(
        StandardScaler(with_mean=False), LogisticRegression(C=0.01, fit_intercept=False, max_iter=300)),
        pairwise=True, decision=True))
    add('cand|pair_svm', '쌍대 순위 SVM', _Cand(st, lambda: make_pipeline(
        StandardScaler(with_mean=False), LinearSVC(C=0.001, fit_intercept=False)), pairwise=True, decision=True))
    add('cand|lda', '수축 LDA 실제 대 무작위', _Cand(st, lambda: LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto'), decision=True))
    add('cand|gnb', '가우스 나이브 베이즈 실제 대 무작위', _Cand(st, lambda: GaussianNB(var_smoothing=1e-3)))
    add('cand|et', '엑스트라 트리 실제 대 무작위 매크로 포함', _Cand(st, lambda: ExtraTreesClassifier(
        50, min_samples_leaf=50, max_features=0.3, random_state=0), with_macro=True))

    # ---- 쌓기
    add('stack|oos', '시계열 교차검증 쌓기 분류기3+전문가3',
        _StackOOS([bases['softmax|lag|0.05'], bases['knn|lag|20'], bases['gnb|macro']],
                  ['uni', 'hot50', 'markov1'], ['uni', 'hot50', 'markov1']))
    add('stack|bma', '시계열 교차검증 베이즈 모형평균 분류기3+전문가3',
        _StackOOS([bases['softmax|lag|0.05'], bases['knn|lag|20'], bases['gnb|macro']],
                  ['uni', 'hot50', 'markov1'], ['uni', 'hot50', 'markov1'], kind='bma'))
    add('stack|oos5', '시계열 교차검증 쌓기 분류기5 최근100회',
        _StackOOS([bases['softmax|lag|0.05'], bases['knn|lag|20'], bases['gnb|macro'], bases['rf|lag|20'],
                   bases['softmax|macro|0.05']], ['uni', 'hot_all'], ['uni', 'hot_all'], window=100))
    return out


# ---------------------------------------------------------------- 실험

def time_one_fold(draws, macro, ix=250):
    fs = formulas(draws, macro)
    ids = lab.fold_sample(draws, ix)
    d, g = eng.split_ids(ids)
    rows = []
    for f in fs:
        t = time.time()
        s = f.fit(draws, ix)(d, g)
        t1 = time.time() - t
        t = time.time()
        f.fit(draws, ix + 1)(d, g)
        rows.append((f.key, t1, time.time() - t, lab.pct_rank(np.nan_to_num(np.asarray(s, float), nan=-1e9))))
    return rows


def lookahead_check(draws, macro, ixs):
    fs = formulas(draws, macro)
    bad = []
    for ix in ixs:
        ids = lab.fold_sample(draws, ix)[:2000]
        d, g = eng.split_ids(ids)
        cut = draws.truncated(ix)
        for f in fs:
            a = np.asarray(f.fit(draws, ix)(d, g), float)
            b = np.asarray(f.fit(cut, ix)(d, g), float)
            if not np.allclose(a, b):
                bad.append((f.key, ix, float(np.abs(a - b).max())))
    return len(fs), bad


def main():
    parser = argparse.ArgumentParser(description='기계학습·베이즈 공식 걷기 검증')
    parser.add_argument('part', choices=('time', 'real', 'fake', 'check', 'report'))
    parser.add_argument('--folds', type=int, default=140)
    parser.add_argument('--seed', type=int, default=3003)
    args = parser.parse_args()
    draws = eng.load_draws(eng.DATA_CSV)
    macro = lab.load_macro()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == 'time':
        for key, t1, t2, p in time_one_fold(draws, macro):
            print(f'{key:32s} 첫 학습 {t1:6.2f}초 · 다음 {t2:6.3f}초 · {p:5.1f}%')
        return
    if args.part in ('real', 'fake'):
        data = draws if args.part == 'real' else fsm.fake_draws(draws, args.seed)
        fs = formulas(data, macro)
        start = time.time()
        pct, _ = lab.walk(data, fs, args.folds)
        sec = time.time() - start
        name = 'ml_pct.npy' if args.part == 'real' else 'ml_pct_fake.npy'
        np.save(OUT / name, pct)
        (OUT / (name + '.keys.txt')).write_text(
            '\n'.join(f'{f.key}\t{f.label}' for f in fs) + f'\n#seconds\t{sec:.1f}\n', encoding='utf-8')
        print(f'{args.part} 걷기 {sec:.0f}초, 모델 {len(fs)}개', file=sys.stderr)
        return
    if args.part == 'check':
        n = len(draws.rounds)
        count, bad = lookahead_check(draws, macro, (n - 35, n - 7))
        (OUT / 'ml_check.txt').write_text(f'{count}\t{n - 35},{n - 7}\t{len(bad)}\n' +
                                          '\n'.join(map(str, bad)), encoding='utf-8')
        print(count, bad)
        return
    report()


def report():
    pct = np.load(OUT / 'ml_pct.npy')
    lines = (OUT / 'ml_pct.npy.keys.txt').read_text(encoding='utf-8').splitlines()
    keys = [l.split('\t') for l in lines if not l.startswith('#')]
    sec = float([l for l in lines if l.startswith('#seconds')][0].split('\t')[1])
    dev, test = pct[:, :100].mean(1), pct[:, 100:].mean(1)
    order = np.argsort(dev)
    out = ['# 기계학습·베이즈 공식 실험', '',
           f'모델 {len(keys)}개. 개발 구간 201~300회 100회에서 고르고 검증 구간 301~340회 40회로 확인했습니다. '
           '숫자는 실제 1등이 무작위 5만 후보 안에서 상위 몇 %에 왔는지의 평균이며 무작위면 50%입니다.', '',
           f'실제 자료 걷기 140회 {sec:.0f}초, 회차당 {sec / pct.shape[1]:.1f}초.', '',
           '## 개발 구간 상위 20개와 검증 구간 성적', '', '| 순위 | 공식 | 키 | 개발 | 검증 |', '| ---: | --- | --- | ---: | ---: |']
    for r, k in enumerate(order[:20], 1):
        out.append(f'| {r} | {keys[k][1]} | {keys[k][0]} | {dev[k]:.1f}% | {test[k]:.1f}% |')
    bt = int(np.argmin(test))
    out += ['', f'검증 구간 최고, 뒤늦게 고른 것이라 참고만: {keys[bt][1]} {test[bt]:.1f}% · 그 공식의 개발 {dev[bt]:.1f}%',
            f'개발 구간 40% 아래 {int((dev < 40).sum())}개, 그중 검증에서도 40% 아래 {int(((dev < 40) & (test < 40)).sum())}개',
            f'전체 평균: 개발 {dev.mean():.1f}% · 검증 {test.mean():.1f}%',
            f'개발 성적과 검증 성적의 상관 {np.corrcoef(dev, test)[0, 1]:.2f}']
    fake_path = OUT / 'ml_pct_fake.npy'
    if fake_path.exists():
        fp = np.load(fake_path)
        fd, ft = fp[:, :100].mean(1), fp[:, 100:].mean(1)
        fb = int(np.argmin(fd))
        out += ['', '## 같은 실험을 가짜 무작위 자료에서', '',
                '가짜 자료는 fsm.fake_draws 씨앗 3003, 날짜는 같고 번호만 완전 무작위입니다. 모델은 가짜 자료로 다시 만들었습니다.', '',
                '| 자료 | 개발 최고 공식 | 개발 | 그 공식의 검증 | 개발 40% 아래 개수 | 전체 평균 검증 |',
                '| --- | --- | ---: | ---: | ---: | ---: |',
                f'| 가짜 | {keys[fb][1]} | {fd[fb]:.1f}% | {ft[fb]:.1f}% | {int((fd < 40).sum())} | {ft.mean():.1f}% |',
                f'| 실제 | {keys[order[0]][1]} | {dev[order[0]]:.1f}% | {test[order[0]]:.1f}% | {int((dev < 40).sum())} | {test.mean():.1f}% |']
        flines = (OUT / 'ml_pct_fake.npy.keys.txt').read_text(encoding='utf-8').splitlines()
        fsec = float([l for l in flines if l.startswith('#seconds')][0].split('\t')[1])
        out += ['', f'가짜 자료 개발 상위 5개, 걷기 {fsec:.0f}초:', '']
        for k in np.argsort(fd)[:5]:
            out.append(f'- {keys[k][1]}: 개발 {fd[k]:.1f}% · 검증 {ft[k]:.1f}%')
        out.append(f'- 가짜 자료 개발·검증 상관 {np.corrcoef(fd, ft)[0, 1]:.2f}')
    chk = OUT / 'ml_check.txt'
    if chk.exists():
        head = chk.read_text(encoding='utf-8').splitlines()
        count, ixs, nbad = head[0].split('\t')
        rows = ' · '.join(str(int(i) + 1) + '회' for i in ixs.split(','))
        out += ['', '## 미래 자료 누수 검사', '',
                f'{rows}를 맞히는 시점에서 모델 {count}개 모두 전체 자료로 학습한 점수와 그 회차 앞까지 자른 자료로 학습한 점수를 '
                f'후보 2,000개에서 비교했습니다. 다른 모델 {nbad}개.']
        out += head[1:]
    out += ['', '배열: ml_pct.npy 실제, ml_pct_fake.npy 가짜, 행 순서는 같은 이름의 .keys.txt.',
            '다시 만들기: python3 pension720/lab_ml.py real · fake · check · report']
    (OUT / 'ml.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
