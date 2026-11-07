#!/usr/bin/env python3
"""연금복권720+ 시계열·신호처리·동역학계 공식 실험실 (signal2).

자리별 숫자 6줄과 조 1줄을 '신호'로 보고 다음 값의 분포를 예측한다. 점수는 자리별 예측 확률 로그의 합
(+조 로그 확률), 또는 겨냥값과의 가우스 근접도.
lab.py에 이미 있는 이동평균·모멘텀·볼린저·피보나치·일목·AR(1~3)·단순 푸리에·RSI/MACD/돌파·시차 마르코프·
위험률·닮은 회차는 다시 만들지 않는다.

가족
- 지수감쇠 빈도(반감기 1~233회), 감쇠 전이표, 반감기 전문가 온라인 혼합(베이즈 모형 평균)
- 은닉 마르코프(자리별 2~4상태, 전체 공통 상태 2~4상태, EM)
- 칼만 국소 선형 추세(숫자 지표), 홀트 이중지수, 칼만 국소수준 최우추정(값)
- 하르 웨이블릿 다중 척도 외삽, 하르 잡음 제거
- 타켄스 지연 매장 + 최근접(심플렉스 투영), 전체 번호 지연 매장, S-맵, 귀환 사상 다항식
- 재귀 분석(대각선 길이), 감쇠 기억 유사 이력, PPM(가변 차수 마르코프)
- 변화점: 베이즈 온라인 변화점(BOCPD), 위험률 혼합, 이진 분할 후 마지막 국면 빈도
- 허스트 지수·분산비 국면, 엔트로피율·상호정보 시차 선택
- 그레인저식 원천 선택(1등·보너스 자리, 시차), 나이브 베이즈 결합, 능형 VAR, DMD(저계수 VAR)
- 자기상관 주기 검출 후 위상 접기, 미세 주파수 조화 회귀, SSA(특이 스펙트럼) 순환 예측
- ARIMA류: AIC AR, ARMA(1,1)·ARIMA(1,1,1) 해넌-리사넨, ARFIMA(GPH)
- 주(week-of-year) 계절: 같은 주 ±h주 빈도, 연 조화 회귀, 계절 분해 겨냥
- 기호 동역학: 순서 패턴 방향, 원형 증분(mod 10) 분포와 증분 마르코프

규칙: fit(draws, ix)는 ix 이전 행과 ix행의 회차·날짜만 쓴다. 공유 계산은 (draws 객체, ix) 단위로 캐시한다.
"""
from __future__ import annotations

import sys
import time
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from . import formulas as fm
    from . import lab
    from . import predict_pension15 as eng
except ImportError:
    sys.path.insert(0, str(HERE))
    import formulas as fm
    import lab
    import predict_pension15 as eng

POS = np.arange(6)
EPS = 1e-4

# ---------------------------------------------------------------- 공통 도구

_CACHE = {}


def _cached(draws, key, fn):
    """draws 객체별 캐시. 객체를 함께 붙잡아 id 재사용을 막는다."""
    k = (id(draws),) + key
    hit = _CACHE.get(k)
    if hit is not None and hit[0] is draws:
        return hit[1]
    value = fn()
    _CACHE[k] = (draws, value)
    return value


def clear_cache():
    _CACHE.clear()


_FLAT = [None, None]


def _flat(d):
    if _FLAT[0] is not d:
        _FLAT[0], _FLAT[1] = d, np.asarray(d, dtype=np.int64) + 10 * POS
    return _FLAT[1]


def _norm(p):
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.0), EPS, None)
    return p / p.sum(axis=-1, keepdims=True)


def _onehot(X, K):
    return (X[:, :, None] == np.arange(K)).astype(float)   # (T, c, K)


def _marg(X, K, a=1.0):
    oh = _onehot(X, K)
    return (oh.sum(0) + a / K) / (len(X) + a)


def _series(draws, ix):
    D = draws.main[:ix].astype(np.int64)
    G = (draws.groups[:ix].astype(np.int64) - 1)[:, None]
    return D, G


def _dist_score(pd, pg=None, sign=1.0, base_d=None, base_g=None):
    """pd (6,10), pg (1,5) 확률. base가 있으면 로그 비(향상)."""
    ld = np.log(_norm(pd))
    if base_d is not None:
        ld = ld - np.log(_norm(base_d))
    flat = (sign * ld).ravel()
    lg = None
    if pg is not None:
        lg = np.log(_norm(pg))[0]
        if base_g is not None:
            lg = lg - np.log(_norm(base_g))[0]
        lg = sign * lg

    def score(d, g):
        s = flat[_flat(d)].sum(axis=1)
        if lg is not None:
            s = s + lg[np.asarray(g, dtype=np.int64)]
        return s
    return score


def cat_formula(method, sign=1.0, lift=False, group=True):
    """method(X, K, draws, ix) -> (c, K) 확률. 자리 숫자(K=10)와 조(K=5)에 똑같이 쓴다."""
    def fit(draws, ix):
        D, G = _series(draws, ix)
        pd = method(D, 10, draws, ix)
        pg = method(G, 5, draws, ix) if group else None
        if lift:
            return _dist_score(pd, pg, sign, _marg(D, 10), _marg(G, 5) if group else None)
        return _dist_score(pd, pg, sign)
    return fit


def _value_series(draws, ix, kind):
    if kind == '합':
        return np.column_stack((draws.main[:ix].sum(axis=1), draws.groups[:ix])).astype(float)
    if kind == '값':
        return np.column_stack(((draws.main[:ix].astype(np.int64) @ eng.POWERS) / 1e5, draws.groups[:ix])).astype(float)
    return np.column_stack((draws.main[:ix], draws.groups[:ix])).astype(float)


def val_formula(method, kind='자리', sign=1.0):
    """method(x (T,c)) -> (겨냥값 (c,), 표준편차 (c,)). 마지막 열은 조(1~5)."""
    def fit(draws, ix):
        x = _value_series(draws, ix, kind)
        t, s = method(x)
        t = np.nan_to_num(np.asarray(t, dtype=float), nan=float(np.nanmean(x[-20:]))).reshape(-1)
        s = np.maximum(np.nan_to_num(np.asarray(s, dtype=float), nan=1.0).reshape(-1), 0.5)
        tg, sg = t[-1], s[-1]
        if kind == '자리':
            v = np.arange(10.0)
            ld = -0.5 * ((v[None, :] - t[:6, None]) / s[:6, None]) ** 2
            flat = (sign * ld).ravel()

            def score(d, g):
                return flat[_flat(d)].sum(axis=1) - sign * 0.5 * ((np.asarray(g) + 1 - tg) / sg) ** 2
            return score
        if kind == '합':
            return lambda d, g: sign * (-0.5 * ((d.sum(axis=1) - t[0]) / s[0]) ** 2
                                        - 0.5 * ((np.asarray(g) + 1 - tg) / sg) ** 2)
        return lambda d, g: sign * (-0.5 * (((d @ eng.POWERS) / 1e5 - t[0]) / s[0]) ** 2
                                    - 0.5 * ((np.asarray(g) + 1 - tg) / sg) ** 2)
    return fit


def _sd(x, n=60):
    return np.maximum(x[-n:].std(axis=0), 0.5)


# ---------------------------------------------------------------- 지수감쇠 빈도·전이·전문가 혼합

def _ew_w(T, h):
    return 0.5 ** ((T - 1 - np.arange(T)) / h) if h else np.ones(T)


def p_ew(h, a=1.0):
    def m(X, K, draws, ix):
        w = _ew_w(len(X), h)
        cnt = np.einsum('t,tck->ck', w, _onehot(X, K))
        return (cnt + a / K) / (w.sum() + a)
    return m


def p_ew_bonus(h, a=1.0):
    """1등과 보너스를 같은 감쇠로 합친 빈도 (조는 1등 조만)."""
    def m(X, K, draws, ix):
        w = _ew_w(len(X), h)
        cnt = np.einsum('t,tck->ck', w, _onehot(X, K))
        tot = w.sum()
        if K == 10:
            cnt = cnt + np.einsum('t,tck->ck', w, _onehot(draws.bonus[:ix].astype(np.int64), 10))
            tot *= 2
        return (cnt + a / K) / (tot + a)
    return m


def p_decay_markov(h, a=2.0):
    def m(X, K, draws, ix):
        T, c = X.shape
        w = _ew_w(T, h)[1:]
        tab = np.zeros((c, K, K))
        for j in range(c):
            np.add.at(tab[j], (X[:-1, j], X[1:, j]), w)
        marg = _marg(X, K)
        last = X[-1]
        row = tab[np.arange(c), last]
        return (row + a * marg) / (row.sum(1, keepdims=True) + a)
    return m


EXPERT_H = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 0)


def _expert_paths(X, K):
    """반감기 전문가마다 t번째 행 직전의 예측 확률 (E, T, c, K). t번째 예측은 t 이전 행만 쓴다."""
    T, c = X.shape
    oh = _onehot(X, K)
    lam = np.array([0.5 ** (1 / h) if h else 1.0 for h in EXPERT_H])
    cnt = np.zeros((len(lam), c, K))
    tot = np.zeros(len(lam))
    out = np.empty((len(lam), T + 1, c, K))
    for t in range(T + 1):
        out[:, t] = (cnt + 1.0 / K) / (tot + 1.0)[:, None, None]
        if t < T:
            cnt = lam[:, None, None] * cnt + oh[t][None]
            tot = lam * tot + 1
    return out


def p_expert_mix(gamma, mode='bma', burn=20):
    def m(X, K, draws, ix):
        paths = _expert_paths(X, K)               # (E, T+1, c, K)
        T, c = X.shape
        hit = np.take_along_axis(paths[:, :T], X[None, :, :, None], axis=3)[..., 0]   # (E, T, c)
        ll = np.log(hit[:, burn:])
        disc = gamma ** (np.arange(ll.shape[1])[::-1])
        score = np.einsum('etc,t->ec', ll, disc)  # (E, c)
        if mode == 'leader':
            w = (score == score.max(0, keepdims=True)).astype(float)
        else:
            w = np.exp(score - score.max(0, keepdims=True))
        w /= w.sum(0, keepdims=True)
        return np.einsum('ec,eck->ck', w, paths[:, T])
    return m


# ---------------------------------------------------------------- 은닉 마르코프

def _hmm_init(X, K, S):
    T, c = X.shape
    marg = _marg(X, K)
    s_i, k_i = np.arange(S)[:, None], np.arange(K)[None, :]
    pert = 1 + 0.6 * np.cos(2 * np.pi * (s_i + 1) * (k_i + 1) / K + 1.3 * s_i)
    B = marg[:, None, :] * pert[None]
    B /= B.sum(2, keepdims=True)
    A = np.full((c, S, S), 0.2 / max(S - 1, 1))
    A[:, np.arange(S), np.arange(S)] = 0.8
    return np.full((c, S), 1.0 / S), A, B


def _hmm_forward(pi, A, Bx):
    T, c, S = Bx.shape
    alpha = np.empty((T, c, S))
    scale = np.empty((T, c))
    a = pi * Bx[0]
    for t in range(T):
        if t:
            a = np.matmul(alpha[t - 1][:, None, :], A)[:, 0] * Bx[t]
        s = a.sum(1)
        alpha[t] = a / s[:, None]
        scale[t] = s
    return alpha, scale


def hmm_fit(X, K, S, iters=30):
    """자리별 독립 HMM (c개 사슬을 한꺼번에). 반환 pi (c,S), A (c,S,S), B (c,S,K)."""
    T, c = X.shape
    oh = _onehot(X, K)
    pi, A, B = _hmm_init(X, K, S)
    for _ in range(iters):
        Bx = np.einsum('csk,tck->tcs', B, oh)
        alpha, scale = _hmm_forward(pi, A, Bx)
        beta = np.empty((T, c, S))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = np.matmul(A, (Bx[t + 1] * beta[t + 1])[:, :, None])[:, :, 0] / scale[t + 1][:, None]
        gamma = alpha * beta
        gamma /= gamma.sum(2, keepdims=True)
        xi = np.einsum('tcs,csr,tcr->csr', alpha[:-1], A, Bx[1:] * beta[1:] / scale[1:, :, None])
        A = xi + 1e-2
        A /= A.sum(2, keepdims=True)
        B = np.einsum('tcs,tck->csk', gamma, oh) + 0.5
        B /= B.sum(2, keepdims=True)
        pi = gamma[0] + 1e-3
        pi /= pi.sum(1, keepdims=True)
    return pi, A, B


def hmm_predict(params, X, K):
    pi, A, B = params
    Bx = np.einsum('csk,tck->tcs', B, _onehot(X, K))
    alpha, _ = _hmm_forward(pi, A, Bx)
    state = np.matmul(alpha[-1][:, None, :], A)[:, 0]
    return np.einsum('cs,csk->ck', state, B)


def p_hmm(S, window=None, block=10):
    def m(X, K, draws, ix):
        b = ix - ix % block
        start = max(0, b - window) if window else 0
        params = _cached(draws, ('hmm', K, S, window, b), lambda: hmm_fit(X[start:b], K, S))
        return hmm_predict(params, X[start:], K)
    return m


def ghmm_fit(chans, Ks, S, iters=30):
    """모든 자리(+조)가 같은 숨은 상태를 공유하는 HMM. chans: (T,) 정수 배열 목록."""
    T = len(chans[0])
    ohs = [(ch[:, None] == np.arange(k)).astype(float) for ch, k in zip(chans, Ks)]
    Bs = []
    s_i = np.arange(S)[:, None]
    for n, (oh, k) in enumerate(zip(ohs, Ks)):
        marg = (oh.sum(0) + 1) / (T + k)
        pert = 1 + 0.6 * np.cos(2 * np.pi * (s_i + 1) * (np.arange(k)[None] + 1 + n) / k + 1.3 * s_i)
        b = marg[None] * pert
        Bs.append(b / b.sum(1, keepdims=True))
    A = np.full((S, S), 0.2 / max(S - 1, 1))
    A[np.arange(S), np.arange(S)] = 0.8
    pi = np.full(S, 1.0 / S)
    for _ in range(iters):
        logbx = sum(oh @ np.log(b).T for oh, b in zip(ohs, Bs))   # (T,S)
        Bx = np.exp(logbx - logbx.max(1, keepdims=True))
        alpha = np.empty((T, S))
        scale = np.empty(T)
        a = pi * Bx[0]
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ A) * Bx[t]
            s = a.sum()
            alpha[t], scale[t] = a / s, s
        beta = np.empty((T, S))
        beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (Bx[t + 1] * beta[t + 1]) / scale[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = np.einsum('ts,sr,tr->sr', alpha[:-1], A, Bx[1:] * beta[1:] / scale[1:, None])
        A = xi + 1e-2
        A /= A.sum(1, keepdims=True)
        Bs = [(gamma.T @ oh + 0.5) for oh in ohs]
        Bs = [b / b.sum(1, keepdims=True) for b in Bs]
        pi = gamma[0] + 1e-3
        pi /= pi.sum()
    return pi, A, Bs


def ghmm_predict(params, chans, Ks):
    pi, A, Bs = params
    ohs = [(ch[:, None] == np.arange(k)).astype(float) for ch, k in zip(chans, Ks)]
    logbx = sum(oh @ np.log(b).T for oh, b in zip(ohs, Bs))
    Bx = np.exp(logbx - logbx.max(1, keepdims=True))
    a = pi * Bx[0]
    a /= a.sum()
    for t in range(1, len(Bx)):
        a = (a @ A) * Bx[t]
        a /= a.sum()
    state = a @ A
    return [state @ b for b in Bs]


def ghmm_formula(S, sign=1.0, lift=False, block=10):
    def fit(draws, ix):
        D, G = _series(draws, ix)
        chans = [D[:, j] for j in range(6)] + [G[:, 0]]
        Ks = [10] * 6 + [5]
        b = ix - ix % block
        params = _cached(draws, ('ghmm', S, b), lambda: ghmm_fit([c[:b] for c in chans], Ks, S))
        pr = ghmm_predict(params, chans, Ks)
        pd, pg = np.stack(pr[:6]), pr[6][None]
        if lift:
            return _dist_score(pd, pg, sign, _marg(D, 10), _marg(G, 5))
        return _dist_score(pd, pg, sign)
    return fit


# ---------------------------------------------------------------- 칼만·홀트

def p_kalman_llt(ql, qs, r=0.09):
    def m(X, K, draws, ix):
        oh = _onehot(X, K).reshape(len(X), -1)
        lvl = np.full(oh.shape[1], 1.0 / K)
        slp = np.zeros(oh.shape[1])
        P = np.array([[0.01, 0.0], [0.0, 1e-4]])
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.diag([ql, qs])
        for y in oh:
            lvl, slp = lvl + slp, slp           # 예측
            P = F @ P @ F.T + Q
            k0, k1 = P[0, 0] / (P[0, 0] + r), P[1, 0] / (P[0, 0] + r)
            e = y - lvl
            lvl, slp = lvl + k0 * e, slp + k1 * e
            P = P - np.outer([k0, k1], P[0])
        return (lvl + slp).reshape(X.shape[1], K)
    return m


def v_holt(a, b):
    def m(x):
        lvl, tr = x[:10].mean(0), np.zeros(x.shape[1])
        for v in x[10:]:
            new = a * v + (1 - a) * (lvl + tr)
            tr = b * (new - lvl) + (1 - b) * tr
            lvl = new
        return lvl + tr, _sd(x)
    return m


def v_local_level_mle():
    def m(x):
        out = []
        for col in x.T:
            var = col.var() + 1e-9
            best = None
            for q in (1e-4, 1e-3, 1e-2, 0.05, 0.2, 1.0):
                lvl, P, ll = col[:5].mean(), var, 0.0
                for v in col[5:]:
                    P += q * var
                    S = P + var
                    e = v - lvl
                    ll += -0.5 * (log(S) + e * e / S)
                    k = P / S
                    lvl += k * e
                    P *= 1 - k
                if best is None or ll > best[0]:
                    best = (ll, lvl)
            out.append(best[1])
        return np.array(out), _sd(x)
    return m


# ---------------------------------------------------------------- 웨이블릿

def p_wave_extrap(kind, J=6):
    def m(X, K, draws, ix):
        I = _onehot(X, K)
        preds, ws = [], []
        for j in range(1, J + 1):
            s = 2 ** j
            if 2 * s > len(I):
                break
            A, B = I[-s:].mean(0), I[-2 * s:-s].mean(0)
            preds.append(A + (A - B) * (s + 1) / (2 * s))
            ws.append({'flat': 1.0, 'fine': 2.0 ** -j, 'coarse': 2.0 ** j}[kind])
        ws = np.array(ws) / sum(ws)
        return np.einsum('j,jck->ck', ws, np.array(preds))
    return m


def _haar_denoise_last(Y, mult):
    """Y (N, m), N=2^J. 직교 하르 변환 → 세부계수 연한 임계 → 복원 마지막 값."""
    N = len(Y)
    p = Y.mean(0)
    sigma = np.sqrt(np.clip(p * (1 - p), 1e-4, None))
    lam = mult * sigma * sqrt(2 * log(N))
    a = Y.copy()
    details = []
    while len(a) > 1:
        ev, od = a[0::2], a[1::2]
        details.append((ev - od) / sqrt(2))
        a = (ev + od) / sqrt(2)
    details = [np.sign(d) * np.maximum(np.abs(d) - lam, 0) for d in details]
    for d in reversed(details):
        up = np.empty((2 * len(a),) + a.shape[1:])
        up[0::2], up[1::2] = (a + d) / sqrt(2), (a - d) / sqrt(2)
        a = up
    return a[-1]


def p_haar_denoise(N, mult):
    def m(X, K, draws, ix):
        I = _onehot(X, K)[-N:]
        T, c, _ = I.shape
        return _haar_denoise_last(I.reshape(T, -1), mult).reshape(c, K)
    return m


def v_wave_extrap(J=5):
    def m(x):
        preds = []
        for j in range(1, J + 1):
            s = 2 ** j
            A, B = x[-s:].mean(0), x[-2 * s:-s].mean(0)
            preds.append(A + (A - B) * (s + 1) / (2 * s))
        return np.mean(preds, axis=0), _sd(x)
    return m


# ---------------------------------------------------------------- 지연 매장·카오스

def _embed(x, m, tau):
    """x (T,c) → 지연 벡터 (L, c, m) 와 시각 idx (L,)."""
    span = (m - 1) * tau
    idx = np.arange(span, len(x))
    return np.stack([x[idx - i * tau] for i in range(m)], axis=-1), idx


def p_simplex(m, tau, k, circ=False, a=2.0):
    def m_(X, K, draws, ix):
        E, idx = _embed(X, m, tau)
        q, lib = E[-1], E[:-1]
        y = X[idx[:-1] + 1]                               # (L, c)
        diff = np.abs(lib - q[None]).astype(float)
        if circ:
            diff = np.minimum(diff, K - diff)
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-6 * np.arange(len(lib))[::-1, None]
        order = np.argsort(dist, axis=0, kind='stable')[:k]
        dk = np.take_along_axis(dist, order, 0)
        yk = np.take_along_axis(y, order, 0)
        w = np.exp(-dk / np.maximum(dk[:1], 0.5))
        c = X.shape[1]
        cnt = np.zeros((c, K))
        np.add.at(cnt, (np.broadcast_to(np.arange(c), yk.shape), yk), w)
        return (cnt + a * _marg(X, K)) / (w.sum(0)[:, None] + a)
    return m_


def v_simplex(m, tau, k):
    def f(x):
        E, idx = _embed(x, m, tau)
        q, lib = E[-1], E[:-1]
        y = x[idx[:-1] + 1]
        sd = x.std(0) + 1e-9
        dist = np.sqrt((((lib - q[None]) / sd[None, :, None]) ** 2).sum(-1)) + 1e-6 * np.arange(len(lib))[::-1, None]
        order = np.argsort(dist, axis=0, kind='stable')[:k]
        dk = np.take_along_axis(dist, order, 0)
        w = np.exp(-dk / np.maximum(dk[:1], 1e-3))
        yk = np.take_along_axis(y, order, 0)
        t = (w * yk).sum(0) / w.sum(0)
        return t, _sd(x)
    return f


def v_smap(m, theta):
    def f(x):
        E, idx = _embed(x, m, 1)
        q, lib = E[-1], E[:-1]
        y = x[idx[:-1] + 1]
        out = []
        for c in range(x.shape[1]):
            L = lib[:, c]
            d = np.sqrt(((L - q[c]) ** 2).sum(1))
            w = np.exp(-theta * d / (d.mean() + 1e-9))
            Z = np.column_stack((np.ones(len(L)), L)) * np.sqrt(w)[:, None]
            coef = np.linalg.lstsq(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ (y[:, c] * np.sqrt(w)), rcond=None)[0]
            out.append(np.r_[1.0, q[c]] @ coef)
        return np.array(out), _sd(x)
    return f


def v_return_map(deg, two_d=False):
    def f(x):
        mu, sd = x.mean(0), x.std(0) + 1e-9
        z = (x - mu) / sd
        out = []
        for c in range(x.shape[1]):
            a, b = z[1:-1, c], z[:-2, c]
            if two_d:
                feats = lambda u, v: np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
                Z, y, qz = feats(a, b), z[2:, c], feats(z[-1:, c], z[-2:-1, c])
            else:
                feats = lambda u: np.column_stack([u ** p for p in range(deg + 1)])
                Z, y, qz = feats(z[:-1, c]), z[1:, c], feats(z[-1:, c])
            coef = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ y)
            out.append(float((qz @ coef)[0]) * sd[c] + mu[c])
        return np.array(out), _sd(x)
    return f


def vector_embed_formula(m, k, circ):
    """6자리 전체(+조)를 하나의 상태로 지연 매장, 최근접 이력의 다음 회차 분포."""
    def fit(draws, ix):
        D, G = _series(draws, ix)
        S = np.column_stack((D, G)).astype(float)
        E, idx = _embed(S, m, 1)                        # (L, 7, m)
        q, lib = E[-1], E[:-1]
        diff = np.abs(lib - q[None])
        if circ:
            diff[:, :6] = np.minimum(diff[:, :6], 10 - diff[:, :6])
        dist = np.sqrt((diff ** 2).sum((1, 2))) + 1e-6 * np.arange(len(lib))[::-1]
        order = np.argsort(dist, kind='stable')[:k]
        w = np.exp(-dist[order] / max(dist[order[0]], 0.5))
        nxt = idx[order] + 1
        cd = np.zeros((6, 10))
        for j in range(6):
            np.add.at(cd[j], D[nxt, j], w)
        cg = np.zeros((1, 5))
        np.add.at(cg[0], G[nxt, 0], w)
        pd = (cd + 2 * _marg(D, 10)) / (w.sum() + 2)
        pg = (cg + 2 * _marg(G, 5)) / (w.sum() + 2)
        return _dist_score(pd, pg)
    return fit


def recurrence_formula(mode, param, Lm=12, topk=40):
    """재귀 분석. mode 'diag': 같은 자리 일치 수 ≥ param 인 대각선 길이로 가중.
    mode 'mem': 감쇠 λ=param 로 과거 Lm회 일치 수를 합한 유사도 상위 topk."""
    def fit(draws, ix):
        D, G = _series(draws, ix)
        T = len(D)
        eq = (D[:, None, :] == D[None, T - Lm:, :]).sum(-1)   # (T, Lm): s행과 T-Lm+i행 일치 수
        s = np.arange(Lm, T - 1)                               # 이력 끝 s, 다음 행 s+1
        mat = np.stack([eq[s - l, Lm - 1 - l] for l in range(Lm)], axis=1)   # (len, Lm) l=0이 가장 최근
        if mode == 'diag':
            rec = mat >= param
            length = np.cumprod(rec, axis=1).sum(1)
            w = 2.0 ** length - 1
        else:
            sim = mat @ (param ** np.arange(Lm)) + 1e-6 * s
            top = np.argsort(-sim)[:topk]
            w = np.zeros(len(s))
            w[top] = 1.0
        nxt = s + 1
        cd = np.zeros((6, 10))
        for j in range(6):
            np.add.at(cd[j], D[nxt, j], w)
        cg = np.zeros((1, 5))
        np.add.at(cg[0], G[nxt, 0], w)
        pd = (cd + 3 * _marg(D, 10)) / (w.sum() + 3)
        pg = (cg + 3 * _marg(G, 5)) / (w.sum() + 3)
        return _dist_score(pd, pg)
    return fit


def p_ppm(maxo, beta):
    """보간 가변 차수 마르코프. beta='wb'면 위튼-벨(보인 기호 수)."""
    def m(X, K, draws, ix):
        T, c = X.shape
        out = np.empty((c, K))
        for j in range(c):
            x = X[:, j]
            p = np.full(K, 1.0 / K)
            for o in range(maxo + 1):
                if o == 0:
                    nxt = x
                else:
                    ctx = x[T - o:]
                    ok = np.ones(T - o, bool)
                    for i in range(o):
                        ok &= x[i:T - o + i] == ctx[i]
                    nxt = x[o:][ok]
                cnt = np.bincount(nxt, minlength=K).astype(float)
                b = max(1, np.count_nonzero(cnt)) if beta == 'wb' else beta
                p = (cnt + b * p) / (cnt.sum() + b)
            out[j] = p
        return out
    return m


# ---------------------------------------------------------------- 변화점

def bocpd(X, K, lam, a):
    """디리클레-다항 BOCPD. 반환 (예측 확률 (c,K), 로그 증거 (c,))."""
    T, c = X.shape
    H = 1.0 / lam
    R = np.ones((c, 1))
    N = np.zeros((c, 1, K))
    ev = np.zeros(c)
    ci = np.arange(c)[:, None]
    for t in range(T):
        x = X[t]
        n = N.sum(2)
        pred = (N[ci, np.arange(N.shape[1])[None, :], x[:, None]] + a) / (n + K * a)
        joint = R * pred
        tot = joint.sum(1)
        ev += np.log(tot)
        R = np.concatenate((joint.sum(1, keepdims=True) * H, joint * (1 - H)), axis=1) / tot[:, None]
        add = np.zeros((c, 1, K))
        add[np.arange(c), 0, x] = 1
        N = np.concatenate((np.zeros((c, 1, K)), N + add), axis=1)
        if R.shape[1] > 400:
            R, N = R[:, :400], N[:, :400]
            R /= R.sum(1, keepdims=True)
    n = N.sum(2)
    p = np.einsum('cr,crk->ck', R, (N + a) / (n + K * a)[:, :, None])
    return p, ev


def p_bocpd(lam, a):
    def m(X, K, draws, ix):
        return _cached(draws, ('bocpd', K, lam, a, ix), lambda: bocpd(X, K, lam, a))[0]
    return m


def p_bocpd_mix(lams=(10, 30, 100, 300, 1000), a=1.0):
    def m(X, K, draws, ix):
        res = [_cached(draws, ('bocpd', K, l, a, ix), lambda l=l: bocpd(X, K, l, a)) for l in lams]
        ev = np.stack([r[1] for r in res])
        w = np.exp(ev - ev.max(0))
        w /= w.sum(0)
        return np.einsum('lc,lck->ck', w, np.stack([r[0] for r in res]))
    return m


def p_binseg(minseg, pen_mult=1.0, s=10.0):
    """이진 분할로 마지막 국면 시작점을 찾고 그 국면 빈도(전체 쪽으로 수축)."""
    def m(X, K, draws, ix):
        T, c = X.shape
        out = np.empty((c, K))
        base = _marg(X, K)
        for j in range(c):
            start = 0
            while True:
                x = X[start:, j]
                L = len(x)
                if L < 2 * minseg:
                    break
                cum = np.cumsum(np.eye(K)[x], axis=0)
                tot = cum[-1]

                def ll(cnt, n):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return np.nansum(np.where(cnt > 0, cnt * np.log(cnt / n[..., None]), 0.0), axis=-1)
                taus = np.arange(minseg, L - minseg + 1)
                left = cum[taus - 1]
                right = tot - left
                gain = ll(left, taus.astype(float)) + ll(right, (L - taus).astype(float)) - ll(tot[None], np.array([float(L)]))
                b = int(np.argmax(gain))
                if gain[b] > pen_mult * (K - 1) * log(L):
                    start += int(taus[b])
                else:
                    break
            seg = X[start:, j]
            out[j] = (np.bincount(seg, minlength=K) + s * base[j]) / (len(seg) + s)
        return out
    return m


# ---------------------------------------------------------------- 허스트·분산비·엔트로피

def _agg_var_hurst(I):
    """I (T, n). 집계 분산법 허스트 지수 (n,)."""
    ms = [1, 2, 4, 8, 16]
    lv = []
    for m in ms:
        nb = len(I) // m
        blk = I[len(I) - nb * m:].reshape(nb, m, -1).mean(1)
        lv.append(np.log(blk.var(0, ddof=1) + 1e-9))
    slope = np.polyfit(np.log(ms), np.array(lv), 1)[0]
    return 1 + slope / 2


def p_hurst_regime(W, delta, h=8):
    def m(X, K, draws, ix):
        Xw = X[-W:] if W else X
        I = _onehot(Xw, K)
        T, c, _ = I.shape
        H = _agg_var_hurst(I.reshape(T, -1)).reshape(c, K)
        base = _marg(X, K)
        hot = p_ew(h)(X, K, draws, ix)
        cold = np.clip(2 * base - hot, EPS, None)
        return np.where(H > 0.5 + delta, hot, np.where(H < 0.5 - delta, cold, base))
    return m


def p_variance_ratio(W, q, h=8):
    def m(X, K, draws, ix):
        Xw = X[-W:] if W else X
        I = _onehot(Xw, K)
        mu = I.mean(0)
        v1 = I.var(0) + 1e-9
        cs = np.cumsum(I - mu, axis=0)
        sq = cs[q:] - cs[:-q]
        vr = (sq ** 2).mean(0) / (q * v1)
        base = _marg(X, K)
        hot = p_ew(h)(X, K, draws, ix)
        cold = np.clip(2 * base - hot, EPS, None)
        return np.where(vr > 1, hot, cold)
    return m


def v_hurst_regime(W):
    def f(x):
        xw = x[-W:]
        H = _agg_var_hurst(xw)
        ema = xw[-60:].copy()
        e = ema[0]
        for v in ema[1:]:
            e = 0.3 * v + 0.7 * e
        mean = xw[-20:].mean(0)
        rev = 2 * mean - xw[-1]
        t = np.where(H > 0.55, e, np.where(H < 0.45, rev, mean))
        return t, _sd(x)
    return f


def p_mi_lag(W, lag_max, mult=1.0, a=1.0):
    """상호정보(편향 보정)로 시차를 고른 전이표. 이득이 없으면 구간 빈도."""
    def m(X, K, draws, ix):
        Xw = X[-W:] if W else X
        T, c = Xw.shape
        out = np.empty((c, K))
        for j in range(c):
            x = Xw[:, j]
            marg = (np.bincount(x, minlength=K) + a) / (T + K * a)
            best, pick = 0.0, None
            for L in range(1, lag_max + 1):
                n = T - L
                tab = np.bincount(x[:-L] * K + x[L:], minlength=K * K).reshape(K, K).astype(float)
                pj = tab / n
                pa, pb = pj.sum(1), pj.sum(0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mi = np.nansum(np.where(pj > 0, pj * np.log(pj / np.outer(pa, pb)), 0.0))
                gain = mi - mult * (K - 1) ** 2 / (2 * n)
                if gain > best:
                    best, pick = gain, (L, tab)
            if pick is None:
                out[j] = marg
            else:
                L, tab = pick
                row = tab[x[T - L]]
                out[j] = (row + 2 * marg) / (row.sum() + 2)
        return out
    return m


# ---------------------------------------------------------------- 교차 상관·그레인저·VAR

def _loo_ll(cnt, alpha, K):
    """cnt (..., A, B) 분할표의 한 개 빼기 로그우도 합 (..)."""
    n_a = cnt.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(cnt > 0, cnt * np.log((cnt - 1 + alpha) / (n_a - 1 + K * alpha)), 0.0)
    return v.sum((-1, -2))


def granger_formula(sources, Lmax, topk=1, alpha=1.0):
    def fit(draws, ix):
        D = draws.main[:ix].astype(np.int64)
        T = len(D)
        Y = _onehot(D, 10).reshape(T, 60)
        srcs = []
        if sources in ('1등', '둘다'):
            srcs.append(D)
        if sources in ('보너스', '둘다'):
            srcs.append(draws.bonus[:ix].astype(np.int64))
        Sx = np.hstack(srcs)                                   # (T, ns)
        ns = Sx.shape[1]
        S1 = _onehot(Sx, 10).reshape(T, ns * 10)
        marg = _marg(D, 10)
        cands = []                                             # (gain (ns,6), L, tables)
        for L in range(1, Lmax + 1):
            C = (S1[:-L].T @ Y[L:]).reshape(ns, 10, 6, 10).transpose(0, 2, 1, 3)   # (ns, 6, a, b)
            ll = _loo_ll(C, alpha, 10)
            mc = C.sum(2)                                      # (ns, 6, b) 같은 행의 표적 빈도
            n = mc.sum(-1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                llm = np.where(mc > 0, mc * np.log((mc - 1 + alpha) / (n - 1 + 10 * alpha)), 0.0).sum(-1)
            cands.append((ll - llm, L, C))
        gains = np.stack([g for g, _, _ in cands])            # (Lmax, ns, 6)
        out = np.log(marg)
        for j in range(6):
            flat = gains[:, :, j].ravel()
            order = np.argsort(-flat)[:topk]
            acc = np.log(marg[j])
            for o in order:
                if flat[o] <= 0:
                    break
                li, si = divmod(int(o), ns)
                L, C = cands[li][1], cands[li][2]
                a_val = Sx[T - L, si]
                row = C[si, j, a_val]
                pc = (row + alpha) / (row.sum() + 10 * alpha)
                acc = acc + np.log(pc) - np.log(marg[j])
            out[j] = acc
        return _dist_score(np.exp(out))
    return fit


def var_formula(lam, lags=1, bonus=False, rank=None):
    """능형 VAR (rank가 있으면 저계수 = DMD 투영) on 숫자 지표 + 조 지표."""
    def fit(draws, ix):
        D, G = _series(draws, ix)
        T = len(D)
        parts = [_onehot(D, 10).reshape(T, 60), _onehot(G, 5).reshape(T, 5)]
        if bonus:
            parts.append(_onehot(draws.bonus[:ix].astype(np.int64), 10).reshape(T, 60))
        Z = np.hstack(parts)
        mu = Z.mean(0)
        Zc = Z - mu
        Xs = np.hstack([Zc[lags - 1 - l:T - 1 - l] for l in range(lags)])
        Y = Zc[lags:, :65]
        q = np.hstack([Zc[T - 1 - l] for l in range(lags)])
        if rank:
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            U, s, Vt = U[:, :rank], s[:rank], Vt[:rank]
            W = Vt.T @ np.diag(1 / s) @ U.T @ Y
        else:
            W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)
        pred = mu[:65] + q @ W
        return _dist_score(pred[:60].reshape(6, 10), pred[60:65][None])
    return fit


# ---------------------------------------------------------------- 주기성·스펙트럼·SSA

def p_acf_period(Pmax, W, s, thr=1.0):
    def m(X, K, draws, ix):
        Xw = X[-W:] if W else X
        I = _onehot(Xw, K)
        T = len(I)
        mu = I.mean(0)
        Z = I - mu
        var = (Z ** 2).sum(0) + 1e-9
        best = np.full(mu.shape, -np.inf)
        bestL = np.zeros(mu.shape, int)
        for L in range(2, Pmax + 1):
            acf = (Z[L:] * Z[:-L]).sum(0) / var
            up = acf > best
            best[up], bestL[up] = acf[up], L
        base = _marg(X, K)
        out = base.copy()
        sig = best > thr * 2 / sqrt(T)
        for L in range(2, Pmax + 1):
            sel = sig & (bestL == L)
            if not sel.any():
                continue
            idx = T - L * np.arange(1, T // L + 1)
            fold = I[idx].sum(0)
            p = (fold + s * base) / (len(idx) + s)
            out[sel] = p[sel]
        return out
    return m


def p_harmonic(W, Pmin=2.2, Pmax=60, shrink=0.5):
    def m(X, K, draws, ix):
        I = _onehot(X[-W:], K)
        T, c, _ = I.shape
        Z = (I - I.mean(0)).reshape(T, -1)
        periods = np.linspace(Pmin, min(Pmax, T / 2), 160)
        om = 2 * np.pi / periods
        t = np.arange(T)
        C, S = np.cos(np.outer(om, t)), np.sin(np.outer(om, t))
        a, b = C @ Z, S @ Z
        cc, ss = (C ** 2).sum(1)[:, None], (S ** 2).sum(1)[:, None]
        power = a ** 2 / cc + b ** 2 / ss
        k = np.argmax(power, axis=0)
        n = np.arange(Z.shape[1])
        fc = a[k, n] / cc[k, 0] * np.cos(om[k] * T) + b[k, n] / ss[k, 0] * np.sin(om[k] * T)
        return _marg(X, K) + shrink * fc.reshape(c, K)
    return m


def _ssa_forecast(Y, L, r):
    """Y (T, n) 열마다 SSA 순환 예측 (n,)."""
    win = np.lib.stride_tricks.sliding_window_view(Y, L, axis=0)       # (T-L+1, n, L)
    Cm = np.einsum('knl,knm->nlm', win, win)
    _, V = np.linalg.eigh(Cm)
    V = V[:, :, ::-1][:, :, :r]                                        # (n, L, r)
    q = Y[-L:].T                                                       # (n, L)
    qh = np.einsum('nlr,nr->nl', V, np.einsum('nlr,nl->nr', V, q))
    pi = V[:, -1, :]
    nu2 = np.clip((pi ** 2).sum(1), 0, 0.99)
    R = np.einsum('nr,nlr->nl', pi, V[:, :-1, :]) / (1 - nu2)[:, None]
    return (R * qh[:, 1:]).sum(1)


def p_ssa(L, r, W=None):
    def m(X, K, draws, ix):
        I = _onehot(X[-W:] if W else X, K)
        T, c, _ = I.shape
        return _ssa_forecast(I.reshape(T, -1), L, r).reshape(c, K)
    return m


def v_ssa(L, r):
    def f(x):
        return _ssa_forecast(x, L, r), _sd(x)
    return f


# ---------------------------------------------------------------- ARIMA류

def _ar_fit(col, p):
    a = np.column_stack([col[p - i - 1:len(col) - i - 1] for i in range(p)] + [np.ones(len(col) - p)])
    coef, *_ = np.linalg.lstsq(a, col[p:], rcond=None)
    res = col[p:] - a @ coef
    return coef, res


def v_ar_aic(pmax=12):
    def f(x):
        out = []
        for col in x.T:
            best = None
            n = len(col) - pmax
            for p in range(1, pmax + 1):
                coef, res = _ar_fit(col[pmax - p:], p)
                aic = n * log(res.var() + 1e-12) + 2 * p
                if best is None or aic < best[0]:
                    best = (aic, p, coef)
            _, p, coef = best
            out.append(float(np.r_[col[-1:-p - 1:-1], 1.0] @ coef))
        return np.array(out), _sd(x)
    return f


def _arma11(col):
    """해넌-리사넨 ARMA(1,1) 한 걸음 예측."""
    coef, res = _ar_fit(col, 10)
    e = np.r_[np.zeros(10), res]
    y, y1, e1 = col[11:], col[10:-1], e[10:-1]
    A = np.column_stack((y1, e1, np.ones(len(y))))
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi, theta = np.clip(c[0], -0.98, 0.98), np.clip(c[1], -0.98, 0.98)
    mu = c[2]
    eps, prev = 0.0, col[0]
    for v in col[1:]:
        pred = mu + phi * prev + theta * eps
        eps = v - pred
        prev = v
    return mu + phi * prev + theta * eps


def v_arma11(diff=False):
    def f(x):
        out = []
        for col in x.T:
            if diff:
                out.append(col[-1] + _arma11(np.diff(col)))
            else:
                out.append(_arma11(col))
        return np.array(out), _sd(x)
    return f


def v_arfima(nlag=100):
    def f(x):
        out = []
        for col in x.T:
            T = len(col)
            z = col - col.mean()
            I = np.abs(np.fft.rfft(z)) ** 2 / T
            m = int(sqrt(T))
            lam = 2 * np.pi * np.arange(1, m + 1) / T
            reg = np.log(4 * np.sin(lam / 2) ** 2)
            d = -np.polyfit(reg, np.log(I[1:m + 1] + 1e-12), 1)[0]
            d = float(np.clip(d, -0.45, 0.45))
            pi = np.empty(nlag + 1)
            pi[0] = 1
            for k in range(1, nlag + 1):
                pi[k] = pi[k - 1] * (k - 1 - d) / k
            L = min(nlag, T)
            out.append(col.mean() - (pi[1:L + 1] * z[::-1][:L]).sum())
        return np.array(out), _sd(x)
    return f


# ---------------------------------------------------------------- 계절 (연중 주)

def _doy(draws):
    return _cached(draws, ('doy',), lambda: np.array([d.timetuple().tm_yday for d in lab.draw_dates(draws)]))


def _circ_days(a, b):
    diff = np.abs(a - b) % 365.25
    return np.minimum(diff, 365.25 - diff)


def p_week_season(h, s):
    def m(X, K, draws, ix):
        doy = _doy(draws)
        near = _circ_days(doy[:ix], doy[ix]) <= 7 * h + 3
        near[max(0, ix - 26):] = False                   # 같은 해 최근 회차는 빼고 지난 해들만
        base = _marg(X, K)
        cnt = _onehot(X[near], K).sum(0)
        return (cnt + s * base) / (near.sum() + s)
    return m


def p_annual_harmonic(nh, shrink=1.0, ridge=5.0):
    def m(X, K, draws, ix):
        doy = _doy(draws)
        ang = 2 * np.pi * doy / 365.25
        feats = lambda a: np.column_stack([np.ones_like(a)] + [f(k * a) for k in range(1, nh + 1) for f in (np.cos, np.sin)])
        F = feats(ang[:ix])
        I = _onehot(X, K).reshape(len(X), -1)
        reg = ridge * np.eye(F.shape[1])
        reg[0, 0] = 0
        coef = np.linalg.solve(F.T @ F + reg, F.T @ I)
        pred = feats(ang[ix:ix + 1]) @ coef
        base = _marg(X, K).reshape(-1)
        return (base + shrink * (pred[0] - base)).reshape(X.shape[1], K)
    return m


def season_decomp_formula(kind, h=2, s=8.0):
    """추세(최근 26회 평균) + 같은 주 ±h주 지난 해들의 '추세 대비 잔차' 평균 → 겨냥값."""
    def fit(draws, ix):
        x = _value_series(draws, ix, kind)
        doy = _doy(draws)
        T = len(x)
        cs = np.cumsum(np.vstack((np.zeros(x.shape[1]), x)), axis=0)
        t = np.arange(26, T)
        trail = (cs[t] - cs[t - 26]) / 26               # t행 이전 26회 평균
        resid = x[t] - trail
        near = (_circ_days(doy[t], doy[ix]) <= 7 * h + 3) & (t < ix - 26)
        seas = resid[near].sum(0) / (near.sum() + s)
        target = x[-26:].mean(0) + seas
        method = lambda _x: (target, _sd(x))
        return val_formula(method, kind)(draws, ix)
    return fit


# ---------------------------------------------------------------- 기호 동역학

def p_ordinal(L, a=1.0):
    """직전 L-1개 방향(내림·같음·오름) 패턴 뒤 다음 방향 분포 → 그 방향 숫자들에 균등 배분."""
    def m(X, K, draws, ix):
        T, c = X.shape
        sgn = np.sign(np.diff(X, axis=0)) + 1                 # (T-1, c) 0,1,2
        out = np.empty((c, K))
        vals = np.arange(K)
        for j in range(c):
            s = sgn[:, j]
            ctx = s[len(s) - (L - 1):] if L > 1 else s[:0]
            n = len(s)
            ok = np.ones(n - (L - 1), bool)
            for i in range(L - 1):
                ok &= s[i:n - (L - 1) + i] == ctx[i]
            nxt = s[L - 1:][ok]
            pdir = (np.bincount(nxt, minlength=3) + a) / (len(nxt) + 3 * a)
            last = X[-1, j]
            dirs = np.sign(vals - last) + 1
            cnt = np.bincount(dirs, minlength=3).astype(float)
            out[j] = pdir[dirs] / cnt[dirs]
        return out
    return m


def p_increment(h, a=1.0):
    def m(X, K, draws, ix):
        dlt = (X[1:] - X[:-1]) % K
        w = _ew_w(len(dlt), h)
        pd = (np.einsum('t,tck->ck', w, _onehot(dlt, K)) + a / K) / (w.sum() + a)
        idx = (np.arange(K)[None, :] - X[-1][:, None]) % K
        return np.take_along_axis(pd, idx, 1)
    return m


def p_increment_markov(a=2.0):
    def m(X, K, draws, ix):
        dlt = (X[1:] - X[:-1]) % K
        T, c = dlt.shape
        out = np.empty((c, K))
        marg = _marg(dlt, K)
        for j in range(c):
            tab = np.bincount(dlt[:-1, j] * K + dlt[1:, j], minlength=K * K).reshape(K, K)
            row = tab[dlt[-1, j]]
            pd = (row + a * marg[j]) / (row.sum() + a)
            out[j] = pd[(np.arange(K) - X[-1, j]) % K]
        return out
    return m


# ---------------------------------------------------------------- 공식 목록

def formulas(draws=None, macro=None):
    F = fm.Formula
    out = []

    def add(key, label, fit):
        out.append(F('signal2|' + key, label, fit))

    # 지수감쇠 빈도
    for h in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        add(f'ew|{h}', f'지수감쇠 빈도 반감기 {h}회', cat_formula(p_ew(h)))
        add(f'ew|{h}|-', f'지수감쇠 빈도 반감기 {h}회 반대', cat_formula(p_ew(h), sign=-1.0))
    for h in (5, 21, 89):
        add(f'ewb|{h}', f'지수감쇠 1등+보너스 반감기 {h}회', cat_formula(p_ew_bonus(h)))
    for h in (15, 50):
        add(f'dmarkov|{h}', f'감쇠 전이표 반감기 {h}회', cat_formula(p_decay_markov(h)))
    add('mix|bma', '반감기 전문가 베이즈 혼합', cat_formula(p_expert_mix(1.0)))
    add('mix|bma97', '반감기 전문가 혼합 망각 0.97', cat_formula(p_expert_mix(0.97)))
    add('mix|leader', '반감기 전문가 최고 따라가기', cat_formula(p_expert_mix(0.97, 'leader')))

    # 은닉 마르코프
    for S in (2, 3, 4):
        add(f'hmm|{S}', f'은닉 마르코프 자리별 {S}상태', cat_formula(p_hmm(S)))
        add(f'hmm|{S}|lift', f'은닉 마르코프 자리별 {S}상태 향상분', cat_formula(p_hmm(S), lift=True))
        add(f'hmm|{S}|-lift', f'은닉 마르코프 자리별 {S}상태 향상분 반대', cat_formula(p_hmm(S), sign=-1.0, lift=True))
        add(f'hmm|{S}|w150', f'은닉 마르코프 자리별 {S}상태 최근150회', cat_formula(p_hmm(S, 150)))
        add(f'ghmm|{S}', f'은닉 마르코프 공통 {S}상태', ghmm_formula(S))
        add(f'ghmm|{S}|lift', f'은닉 마르코프 공통 {S}상태 향상분', ghmm_formula(S, lift=True))
        add(f'ghmm|{S}|-lift', f'은닉 마르코프 공통 {S}상태 향상분 반대', ghmm_formula(S, -1.0, True))

    # 칼만·홀트
    for ql, qs in ((1e-4, 1e-6), (1e-3, 1e-5), (1e-3, 0.0), (1e-2, 1e-4)):
        add(f'kalman|{ql:g}|{qs:g}', f'칼만 국소선형추세 수준{ql:g} 기울기{qs:g}', cat_formula(p_kalman_llt(ql, qs)))
    for a, b in ((0.1, 0.05), (0.3, 0.1), (0.5, 0.3)):
        for kind in ('자리', '합'):
            add(f'holt|{kind}|{a}|{b}', f'홀트 이중지수 {kind} α{a} β{b}', val_formula(v_holt(a, b), kind))
    for kind in ('자리', '합'):
        add(f'llmle|{kind}', f'칼만 국소수준 최우추정 {kind}', val_formula(v_local_level_mle(), kind))

    # 웨이블릿
    for kind, word in (('flat', '균등'), ('fine', '잔척도'), ('coarse', '큰척도')):
        add(f'wave|{kind}', f'하르 다중척도 외삽 {word}', cat_formula(p_wave_extrap(kind)))
        add(f'wave|{kind}|-', f'하르 다중척도 외삽 {word} 반대', cat_formula(p_wave_extrap(kind), sign=-1.0))
    for N in (64, 128):
        for mult in (0.5, 1.0):
            add(f'haar|{N}|{mult}', f'하르 잡음제거 {N}회 임계{mult}', cat_formula(p_haar_denoise(N, mult)))
    for kind in ('자리', '합'):
        add(f'wavev|{kind}', f'하르 다중척도 외삽 값 {kind}', val_formula(v_wave_extrap(), kind))

    # 지연 매장·카오스
    for m, tau, k in ((1, 1, 20), (2, 1, 5), (2, 1, 15), (2, 1, 40), (3, 1, 10), (3, 1, 30), (4, 1, 15),
                      (6, 1, 20), (2, 2, 15), (3, 2, 15), (2, 3, 15)):
        add(f'simplex|{m}|{tau}|{k}', f'지연매장 최근접 m{m} τ{tau} k{k}', cat_formula(p_simplex(m, tau, k)))
    for m, k in ((2, 15), (3, 15)):
        add(f'simplex|circ|{m}|{k}', f'지연매장 최근접 원형거리 m{m} k{k}', cat_formula(p_simplex(m, 1, k, circ=True)))
    for kind, m, k in (('합', 3, 10), ('합', 5, 20), ('자리', 3, 15), ('값', 3, 10)):
        add(f'simplexv|{kind}|{m}|{k}', f'지연매장 최근접 값 {kind} m{m} k{k}', val_formula(v_simplex(m, 1, k), kind))
    for m, k, circ in ((2, 10, False), (2, 30, False), (3, 20, False), (2, 20, True)):
        add(f'vembed|{m}|{k}|{int(circ)}', f'전체 번호 지연매장 m{m} k{k}{" 원형" if circ else ""}',
            vector_embed_formula(m, k, circ))
    for kind in ('자리', '합'):
        for m, th in ((3, 1.0), (3, 4.0), (6, 2.0)):
            add(f'smap|{kind}|{m}|{th:g}', f'S-맵 {kind} m{m} θ{th:g}', val_formula(v_smap(m, th), kind))
    for kind in ('자리', '합'):
        add(f'retmap|{kind}|2', f'귀환사상 2차 {kind}', val_formula(v_return_map(2), kind))
        add(f'retmap|{kind}|3', f'귀환사상 3차 {kind}', val_formula(v_return_map(3), kind))
        add(f'retmap|{kind}|2d', f'귀환사상 2차원 {kind}', val_formula(v_return_map(2, True), kind))

    # 재귀·PPM
    for th in (1, 2):
        add(f'rqa|diag|{th}', f'재귀 대각선 일치{th}개 이상', recurrence_formula('diag', th))
    for lam, k in ((0.5, 20), (0.5, 60), (0.8, 20), (0.8, 60)):
        add(f'rqa|mem|{lam}|{k}', f'감쇠 기억 유사이력 λ{lam} 상위{k}', recurrence_formula('mem', lam, topk=k))
    for o, b in ((2, 1.0), (3, 1.0), (3, 'wb'), (5, 'wb'), (2, 5.0)):
        add(f'ppm|{o}|{b}', f'가변차수 마르코프 최대{o}차 β{b}', cat_formula(p_ppm(o, b)))

    # 변화점
    for lam in (10, 30, 100, 300):
        for a in (0.5, 2.0):
            add(f'bocpd|{lam}|{a}', f'베이즈 온라인 변화점 λ{lam} α{a}', cat_formula(p_bocpd(lam, a)))
    add('bocpd|30|1|-', '베이즈 온라인 변화점 λ30 반대', cat_formula(p_bocpd(30, 1.0), sign=-1.0))
    add('bocpd|mix', '베이즈 온라인 변화점 위험률 혼합', cat_formula(p_bocpd_mix()))
    add('bocpd|mix|lift', '베이즈 온라인 변화점 혼합 향상분', cat_formula(p_bocpd_mix(), lift=True))
    for ms in (10, 20, 40):
        add(f'binseg|{ms}', f'이진분할 마지막 국면 최소{ms}회', cat_formula(p_binseg(ms)))
    add('binseg|10|-', '이진분할 마지막 국면 최소10회 반대', cat_formula(p_binseg(10), sign=-1.0))

    # 허스트·분산비·엔트로피
    for W, dl in ((128, 0.0), (128, 0.05), (None, 0.0), (None, 0.05)):
        add(f'hurst|{W}|{dl}', f'허스트 국면 {W or "전체"}회 폭{dl}', cat_formula(p_hurst_regime(W, dl)))
    for W, q in ((128, 4), (None, 8)):
        add(f'vr|{W}|{q}', f'분산비 국면 {W or "전체"}회 q{q}', cat_formula(p_variance_ratio(W, q)))
    for W in (64, 128):
        add(f'hurstv|{W}', f'허스트 국면 값 겨냥 {W}회', val_formula(v_hurst_regime(W), '자리'))
    for W, Lm in ((60, 1), (120, 1), (None, 1), (None, 13), (None, 52), (150, 26)):
        add(f'mi|{W}|{Lm}', f'상호정보 시차선택 {W or "전체"}회 최대{Lm}', cat_formula(p_mi_lag(W, Lm)))

    # 교차·그레인저·VAR·DMD
    for src in ('1등', '보너스', '둘다'):
        for Lm in (1, 3, 8):
            add(f'granger|{src}|{Lm}', f'그레인저 원천선택 {src} 시차≤{Lm}', granger_formula(src, Lm))
    for src in ('1등', '둘다'):
        add(f'granger|{src}|5|top3', f'그레인저 상위3 원천 결합 {src} 시차≤5', granger_formula(src, 5, topk=3))
    for lam in (30.0, 300.0, 3000.0):
        add(f'var|{lam:g}', f'능형 VAR 지표 λ{lam:g}', var_formula(lam))
    add('var|300|bonus', '능형 VAR 지표+보너스 λ300', var_formula(300.0, bonus=True))
    add('var|300|lag2', '능형 VAR(2) 지표 λ300', var_formula(300.0, lags=2))
    for r in (3, 8, 20):
        add(f'dmd|{r}', f'DMD 저계수 {r}', var_formula(0.0, rank=r))

    # 주기성·스펙트럼·SSA
    for P, W, s in ((12, None, 5.0), (12, None, 20.0), (26, None, 5.0), (26, None, 20.0), (52, None, 10.0),
                    (12, 150, 5.0), (26, 150, 10.0), (52, 260, 10.0)):
        add(f'acf|{P}|{W}|{s:g}', f'자기상관 주기 위상접기 ≤{P} {W or "전체"}회 수축{s:g}', cat_formula(p_acf_period(P, W, s)))
    for W in (64, 128, 256):
        add(f'harm|{W}', f'미세주파수 조화회귀 {W}회', cat_formula(p_harmonic(W)))
    for L, r, W in ((26, 1, None), (26, 3, None), (52, 3, None), (13, 2, 120)):
        add(f'ssa|{L}|{r}|{W}', f'SSA 지표 L{L} r{r} {W or "전체"}회', cat_formula(p_ssa(L, r, W)))
    for kind, L, r in (('자리', 26, 2), ('합', 26, 2), ('합', 52, 4)):
        add(f'ssav|{kind}|{L}|{r}', f'SSA 값 {kind} L{L} r{r}', val_formula(v_ssa(L, r), kind))

    # ARIMA류
    for kind in ('자리', '합'):
        add(f'ar_aic|{kind}', f'AIC 선택 AR(≤12) {kind}', val_formula(v_ar_aic(), kind))
        add(f'arma11|{kind}', f'ARMA(1,1) 해넌-리사넨 {kind}', val_formula(v_arma11(), kind))
        add(f'arima111|{kind}', f'ARIMA(1,1,1) {kind}', val_formula(v_arma11(True), kind))
        add(f'arfima|{kind}', f'ARFIMA GPH 장기기억 {kind}', val_formula(v_arfima(), kind))

    # 계절
    for h, s in ((0, 10.0), (0, 40.0), (1, 10.0), (1, 40.0), (2, 10.0), (2, 40.0), (4, 10.0), (4, 40.0)):
        add(f'woy|{h}|{s:g}', f'연중 같은 주 ±{h}주 수축{s:g}', cat_formula(p_week_season(h, s)))
    for h in (1, 2):
        add(f'woy|{h}|10|-', f'연중 같은 주 ±{h}주 반대', cat_formula(p_week_season(h, 10.0), sign=-1.0))
    for nh in (1, 2):
        add(f'annual|{nh}', f'연 조화회귀 {nh}차', cat_formula(p_annual_harmonic(nh)))
    for kind in ('자리', '합'):
        add(f'stl|{kind}', f'계절분해 겨냥 {kind}', season_decomp_formula(kind))

    # 기호 동역학
    for L in (2, 3, 4):
        add(f'ordinal|{L}', f'순서패턴 방향 길이{L}', cat_formula(p_ordinal(L)))
    for h in (10, 40, 0):
        add(f'incr|{h}', f'원형 증분 분포 반감기 {h or "∞"}', cat_formula(p_increment(h)))
    add('incr|markov', '원형 증분 마르코프', cat_formula(p_increment_markov()))
    return out


# ---------------------------------------------------------------- 실행

def lookahead_check(draws, fs, ixs):
    bad = []
    for ix in ixs:
        ids = lab.fold_sample(draws, ix)[:2000]
        d, g = eng.split_ids(ids)
        cut = draws.truncated(ix)
        for f in fs:
            a = np.asarray(f.fit(draws, ix)(d, g), dtype=float)
            b = np.asarray(f.fit(cut, ix)(d, g), dtype=float)
            if not np.allclose(a, b, equal_nan=True):
                bad.append((f.key, ix))
    return bad


if __name__ == '__main__':
    d = eng.load_draws(eng.DATA_CSV)
    fs = formulas(d, None)
    print(len(fs), 'formulas')
    ids = lab.fold_sample(d, 300)
    dd, gg = eng.split_ids(ids)
    for f in fs:
        t = time.time()
        s = np.asarray(f.fit(d, 300)(dd, gg), dtype=float)
        print(f'{time.time() - t:6.3f}s {lab.pct_rank(np.nan_to_num(s, nan=-1e9)):6.1f} {f.key}')
