#!/usr/bin/env python3
"""신호처리·물리·확률과정 관점의 번호별 점수 공식.

모든 계열 함수는 출현행렬 A(N, 46)를 받아 (N+1, 46) 점수 배열을 돌려줍니다.
행 t는 A[:t], 곧 앞 t개 회차만 써서 t+1번째 회차를 점친 점수이고 0번 열은 무시합니다.
그래서 S[N]이 df 마지막 회차 바로 다음 회차의 점수입니다. 점수가 높을수록 위.

계열
- 푸리에위상: 최근 W회 출현열에 비정수 주기 격자로 사인파를 맞추고 힘이 센 주기 K개를 다음 회차로 외삽
- 자기상관AR: 45개 번호를 합친 자기공분산으로 Yule-Walker AR(L) 계수를 구해 다음 값 예측
- 자기상관개별: 번호마다 시차별 자기상관을 그 시차 전 출현에 곱해 더함
- 허스트국면: 번호마다 집계분산법 허스트 지수, 0.5보다 크면 최근 편차가 이어지고 작으면 되돌아간다고 봄
- 변화점BOCPD: 베타-베르누이 베이지안 온라인 변화점 탐지의 예측 출현확률
- 은닉마르코프: 뜨거움·차가움 두 상태 전환 모형의 전방 필터
- 해저드공통/해저드조건부/해저드번호별: 미출현 기간별 경험적 위험률, 직전 간격 구간별 또는 번호별로 나눠 공통값으로 수축
- 갭리듬: 번호마다 간격의 로그 평균·분산을 지수가중으로 추적한 로그정규 재생과정의 위험률, 규칙적인 번호일수록 뾰족
- 문맥트리: 자기 출현열의 최근 D회 0/1 문맥으로 다음 출현확률, 짧은 문맥으로 계층 수축하는 가변차수 마르코프
- 용수철진동자: 위치=창 안 누적편차, 속도=지수이평 편차로 감쇠 조화진동자 계수를 과거로만 회귀해 다음 변화량 예측
- 적응국면전환: 최근 K회 동안 추세 신호가 맞았는지 상관을 보고 추세·역추세 방향과 세기를 바꿈
- SSA고유모드: 합친 시차공분산의 상위 고유벡터로 선형 반복식을 만들어 외삽
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.linalg import solve_toeplitz
from scipy.special import ndtr

P = 6 / 45


def occ(df):
    X = np.sort(df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.intp), 1)
    A = np.zeros((len(X), 46))
    A[np.arange(len(X))[:, None], X] = 1
    A[:, 0] = 0
    return A


def _csum(A):
    return np.vstack([np.zeros(A.shape[1]), np.cumsum(A, 0)])


def _ema_pred(x, half):
    """x(N, k) → (N+1, k), 행 t는 x[:t]의 지수이평"""
    a = 1 - 0.5 ** (1 / half)
    out = np.zeros((len(x) + 1, x.shape[1]))
    cur = np.full(x.shape[1], P)
    for t in range(len(x)):
        out[t] = cur
        cur = (1 - a) * cur + a * x[t]
    out[-1] = cur
    return out


# 1. 푸리에 위상 외삽
def fourier(A, W=128, K=3, pmin=2.0, chunk=150):
    N = len(A)
    S = np.zeros((N + 1, 46))
    if N < W:
        return S
    pmax = W / 4
    f = np.arange(1 / pmax, 1 / pmin + 1e-12, 1 / (2 * W))
    tau = np.arange(W)
    C = np.cos(2 * np.pi * np.outer(tau, f)) * (2 / W)
    Sn = np.sin(2 * np.pi * np.outer(tau, f)) * (2 / W)
    cn, sn = np.cos(2 * np.pi * f * W), np.sin(2 * np.pi * f * W)
    win = swv(A[:, 1:], W, axis=0)            # win[i] = A[i:i+W], (N-W+1, 45, W)
    out = np.zeros((len(win), 45))
    for i0 in range(0, len(win), chunk):
        x = win[i0:i0 + chunk]
        x = x - x.mean(-1, keepdims=True)
        a, b = x @ C, x @ Sn
        pw, pred = a * a + b * b, a * cn + b * sn
        if K:
            idx = np.argpartition(-pw, K - 1, axis=-1)[..., :K]
            out[i0:i0 + chunk] = np.take_along_axis(pred, idx, -1).sum(-1)
        else:
            out[i0:i0 + chunk] = (pred * pw).sum(-1) / np.maximum(pw.sum(-1), 1e-12)
    S[W:, 1:] = out
    return S


# 2. 합친 자기공분산 Yule-Walker AR(L)
def _lagprod(x, L):
    N = len(x)
    prod = np.zeros((N, L + 1))
    for l in range(L + 1):
        prod[l:, l] = (x[l:] * x[:N - l]).sum(1)
    return np.vstack([np.zeros(L + 1), np.cumsum(prod, 0)])


def ar_pooled(A, L=20, ridge=0.5, tmin=100):
    x = A[:, 1:] - P
    N = len(x)
    G = _lagprod(x, L)
    S = np.zeros((N + 1, 46))
    cnt = np.arange(L + 1)
    for t in range(max(tmin, 2 * L + 2), N + 1):
        g = G[t] / (45 * (t - cnt))
        c = g[:L].copy()
        c[0] *= 1 + ridge
        phi = solve_toeplitz(c, g[1:L + 1])
        S[t, 1:] = phi @ x[t - 1::-1][:L]
    return S


# 3. 번호별 시차 자기상관 메아리
def ar_each(A, L=20, W=None, tmin=100):
    x = A[:, 1:] - P
    N = len(x)
    prod = np.zeros((N, 45, L + 1))
    for l in range(L + 1):
        prod[l:, :, l] = x[l:] * x[:N - l]
    G = np.concatenate([np.zeros((1, 45, L + 1)), np.cumsum(prod, 0)])
    S = np.zeros((N + 1, 46))
    for t in range(max(tmin, L + 2), N + 1):
        g = G[t] - (G[t - W] if W and t > W else 0)
        rho = g[:, 1:] / np.maximum(g[:, :1], 1e-9)
        S[t, 1:] = (rho * x[t - 1::-1][:L].T).sum(1)
    return S


# 4. 허스트 지수 국면
def hurst(A, W=200, k=10, center=True, ms=(1, 2, 4, 8, 16)):
    N = len(A)
    c = _csum(A[:, 1:])                        # (N+1, 45)
    S = np.zeros((N + 1, 46))
    lm = np.log(ms)
    lmc = lm - lm.mean()
    logv = np.zeros((len(ms), N + 1, 45))
    for j, m in enumerate(ms):
        y = c[m:] - c[:-m]                     # y[τ] = A[τ:τ+m] 합, τ=0..N-m
        cy = _csum(y)
        cy2 = _csum(y * y)
        for t in range(W, N + 1):
            lo, hi = t - W, t - m + 1          # τ in [t-W, t-m]
            n = hi - lo
            mu = (cy[hi] - cy[lo]) / n
            v = (cy2[hi] - cy2[lo]) / n - mu * mu
            logv[j, t] = np.log(np.maximum(v, 1e-9))
    H = np.tensordot(lmc, logv, 1) / (lmc @ lmc) / 2      # (N+1, 45)
    t = np.arange(N + 1)
    lo = np.maximum(t - k, 0)
    kk = np.maximum(t - lo, 1)[:, None]
    dev = (c[t] - c[lo] - kk * P) / np.sqrt(kk * P * (1 - P))
    Hc = H - (H.mean(1, keepdims=True) if center else 0.5)
    S[:, 1:] = Hc * dev
    S[:W] = 0
    return S


# 5. 베타-베르누이 베이지안 온라인 변화점 탐지
def bocpd(A, lam=100, m=45, Rmax=400):
    x = A[:, 1:]
    N = len(x)
    Hz = 1 / lam
    a0, b0 = m * P, m * (1 - P)
    c = _csum(x)
    R = np.zeros((45, Rmax + 1))
    R[:, 0] = 1
    r = np.arange(Rmax + 1)
    S = np.zeros((N + 1, 46))
    for t in range(N + 1):
        rr = np.minimum(r, t)
        s = c[t][:, None] - c[t - rr].T
        pr = (a0 + s) / (a0 + b0 + rr)
        S[t, 1:] = (R * pr).sum(1)
        if t == N:
            break
        g = R * np.where(x[t][:, None] > 0, pr, 1 - pr)
        new = np.empty_like(R)
        new[:, 1:] = g[:, :-1] * (1 - Hz)
        new[:, -1] += g[:, -1] * (1 - Hz)
        new[:, 0] = g.sum(1) * Hz
        R = new / new.sum(1, keepdims=True)
    return S


# 6. 뜨거움·차가움 은닉 마르코프 전방 필터
def hmm2(A, delta=0.5, rho=0.05):
    x = A[:, 1:]
    N = len(x)
    ph, pc = P * (1 + delta), P * (1 - delta)
    pi = np.full(45, 0.5)
    S = np.zeros((N + 1, 46))
    for t in range(N + 1):
        S[t, 1:] = pi * ph + (1 - pi) * pc
        if t == N:
            break
        lh = np.where(x[t] > 0, ph, 1 - ph)
        lc = np.where(x[t] > 0, pc, 1 - pc)
        post = pi * lh / (pi * lh + (1 - pi) * lc)
        pi = post * (1 - rho) + (1 - post) * rho
    return S


# 7~9. 경험적 위험률(재생과정)
GAP_EDGES = np.array([2, 4, 8, 16])           # 직전 간격 구간: 1, 2-3, 4-7, 8-15, 16+


def hazard(A, cond="none", kappa=100.0, bw=2, Gmax=80):
    x = A[:, 1:]
    N = len(x)
    nb = {"none": 1, "prevgap": len(GAP_EDGES) + 1, "number": 45}[cond]
    risk0, ev0 = np.zeros(Gmax + 1), np.zeros(Gmax + 1)
    risk, ev = np.zeros((nb, Gmax + 1)), np.zeros((nb, Gmax + 1))
    gg = np.arange(Gmax + 1)
    Kz = np.exp(-0.5 * (np.subtract.outer(gg, gg) / bw) ** 2) if bw > 0 else np.eye(Gmax + 1)
    last = np.full(45, -1)
    prevgap = np.full(45, 8)
    rows = np.arange(45)
    S = np.zeros((N + 1, 46))
    for t in range(N + 1):
        g = np.minimum(t - last, Gmax)
        h0 = (ev0 @ Kz + kappa * P) / (risk0 @ Kz + kappa)
        if cond == "none":
            h = h0[g]
        else:
            b = np.searchsorted(GAP_EDGES, prevgap, side="right") if cond == "prevgap" else rows
            hb = (ev @ Kz + kappa * h0) / (risk @ Kz + kappa)
            h = hb[b, g]
        S[t, 1:] = np.log(h)
        if t == N:
            break
        np.add.at(risk0, g, 1)
        np.add.at(ev0, g, x[t])
        if cond != "none":
            b = np.searchsorted(GAP_EDGES, prevgap, side="right") if cond == "prevgap" else rows
            np.add.at(risk, (b, g), 1)
            np.add.at(ev, (b, g), x[t])
        hit = x[t] > 0
        prevgap[hit] = (t - last)[hit]
        last[hit] = t
    return S


def gap_rhythm(A, half=5, kappa=5.0):
    """번호별 로그 간격의 지수가중 평균·분산으로 만든 이산 로그정규 위험률. 분산은 공통값으로 수축"""
    x = A[:, 1:]
    N = len(x)
    a = 1 - 0.5 ** (1 / half)
    mu = np.full(45, 1.6)
    var = np.full(45, 0.8)
    wsum = np.zeros(45)
    pool_s, pool_n = 0.8 * 50, 50.0
    last = np.full(45, -1)
    S = np.zeros((N + 1, 46))
    for t in range(N + 1):
        g = (t - last).astype(float)
        vp = pool_s / pool_n
        sd = np.sqrt((wsum * var + kappa * vp) / (wsum + kappa))
        z1 = (np.log(g + 0.5) - mu) / sd
        z0 = np.where(g <= 1, -np.inf, (np.log(np.maximum(g - 0.5, 0.5)) - mu) / sd)
        pg = ndtr(z1) - ndtr(z0)
        surv = 1 - ndtr(z0)
        S[t, 1:] = np.log(np.maximum(pg, 1e-12) / np.maximum(surv, 1e-12))
        if t == N:
            break
        hit = x[t] > 0
        lg = np.log(t - last[hit])
        d = lg - mu[hit]
        pool_s += (d * d).sum()
        pool_n += hit.sum()
        mu[hit] += a * d
        var[hit] = (1 - a) * (var[hit] + a * d * d)
        wsum[hit] = np.minimum(wsum[hit] + 1, 1 / a)
        last[hit] = t
    return S


# 10. 자기 문맥 가변차수 마르코프, 계층 수축
def context(A, D=8, beta=10.0):
    x = A[:, 1:].astype(np.int64)
    N = len(x)
    n1 = [np.zeros(1 << k) for k in range(D + 1)]
    nn = [np.zeros(1 << k) for k in range(D + 1)]
    ctx = np.zeros(45, np.int64)
    mask = (1 << D) - 1
    S = np.zeros((N + 1, 46))
    for t in range(N + 1):
        p = np.full(45, P)
        cids = [ctx & ((1 << k) - 1) for k in range(D + 1)]
        for k in range(D + 1):
            p = (n1[k][cids[k]] + beta * p) / (nn[k][cids[k]] + beta)
        S[t, 1:] = np.log(p)
        if t == N:
            break
        for k in range(D + 1):
            np.add.at(n1[k], cids[k], x[t])
            np.add.at(nn[k], cids[k], 1)
        ctx = ((ctx << 1) | x[t]) & mask
    return S


# 11. 감쇠 조화진동자 회귀
def spring(A, W=50, half=5, ridge=50.0, tmin=100):
    x = A[:, 1:] - P
    N = len(x)
    c = _csum(x)
    t = np.arange(N + 1)
    lo = np.maximum(t - W, 0)
    pos = (c[t] - c[lo]) / np.maximum(t - lo, 1)[:, None]          # 창 안 평균 편차 = 위치
    vel = _ema_pred(A[:, 1:], half) - P                               # 속도
    F = np.stack([pos, vel], -1)                                      # (N+1, 45, 2)
    S = np.zeros((N + 1, 46))
    XtX = np.eye(2) * ridge * 1e-3
    Xty = np.zeros(2)
    for s in range(N + 1):
        if s >= tmin:
            beta = np.linalg.solve(XtX, Xty)
            S[s, 1:] = F[s] @ beta
        if s == N:
            break
        if s >= 20:
            XtX += F[s].T @ F[s]
            Xty += F[s].T @ x[s]
    return S


# 12. 성적 적응형 추세·역추세 전환
def adaptive(A, half=10, K=50):
    x = A[:, 1:] - P
    N = len(x)
    m = _ema_pred(A[:, 1:], half)
    z = (m - m.mean(1, keepdims=True)) / np.maximum(m.std(1, keepdims=True), 1e-9)
    hit = np.zeros(N)
    hit[:] = (z[:N] * x).sum(1)                                       # 회차별 추세 성적
    ch = np.concatenate([[0], np.cumsum(hit)])
    S = np.zeros((N + 1, 46))
    for t in range(K + 1, N + 1):
        perf = (ch[t] - ch[t - K]) / K
        S[t, 1:] = perf * z[t]
    return S


# 13. SSA 고유모드 선형 반복식
def ssa(A, L=30, r=3, tmin=150):
    x = A[:, 1:] - P
    N = len(x)
    G = _lagprod(x, L - 1)
    cnt = np.arange(L)
    S = np.zeros((N + 1, 46))
    for t in range(max(tmin, 2 * L + 2), N + 1):
        g = G[t] / (45 * (t - cnt))
        idx = np.abs(np.subtract.outer(cnt, cnt))
        w, U = np.linalg.eigh(g[idx])
        Ur = U[:, np.argsort(w)[::-1][:r]]
        pi = Ur[-1]
        nu2 = pi @ pi
        if nu2 >= 0.999:
            continue
        Rc = Ur[:-1] @ pi / (1 - nu2)                                  # 길이 L-1, 오래된 → 최근
        S[t, 1:] = Rc @ x[t - L + 1:t]
    return S


FAMILIES = {
    "푸리에위상": fourier, "자기상관AR": ar_pooled, "자기상관개별": ar_each, "허스트국면": hurst,
    "변화점BOCPD": bocpd, "은닉마르코프": hmm2, "해저드공통": lambda A, **k: hazard(A, "none", **k),
    "해저드조건부": lambda A, **k: hazard(A, "prevgap", **k), "해저드번호별": lambda A, **k: hazard(A, "number", **k),
    "갭리듬": gap_rhythm, "문맥트리": context, "용수철진동자": spring, "적응국면전환": adaptive, "SSA고유모드": ssa,
}


def zrow(S):
    v = S[:, 1:]
    out = np.zeros_like(S)
    out[:, 1:] = (v - v.mean(1, keepdims=True)) / np.maximum(v.std(1, keepdims=True), 1e-12)
    return out


# 300~999회로만 고른 계열별 최적 설정, 부호 -1은 역방향. 1000~1149회 성적은 선별에만 썼습니다.
FINAL = {
    '자기상관AR_L80_릿지2': ('자기상관AR', {'L': 80, 'ridge': 2.0}, 1),  # A 46.24
    'SSA고유모드_L50_r5': ('SSA고유모드', {'L': 50, 'r': 5}, 1),  # A 46.90
    '적응국면전환_반감20_K60': ('적응국면전환', {'half': 20, 'K': 60}, 1),  # A 47.00
    '자기상관개별_L30_W200': ('자기상관개별', {'L': 30, 'W': 200}, 1),  # A 47.93
    '푸리에위상_W128_K3': ('푸리에위상', {'W': 128, 'K': 3}, 1),  # A 48.35
    '은닉마르코프_δ0.3_ρ0.2': ('은닉마르코프', {'delta': 0.3, 'rho': 0.2}, 1),  # A 48.52
    '문맥트리_D4_β5': ('문맥트리', {'D': 4, 'beta': 5.0}, 1),  # A 48.68
    '해저드공통_κ500_bw0': ('해저드공통', {'kappa': 500, 'bw': 0}, 1),  # A 48.79
    '역해저드번호별_κ50_bw2': ('해저드번호별', {'kappa': 50, 'bw': 2}, -1),  # A 48.80
    '역허스트국면_W100_k5_중심고정': ('허스트국면', {'W': 100, 'k': 5, 'center': False}, -1),  # A 48.87
    '역해저드조건부_κ50_bw2': ('해저드조건부', {'kappa': 50, 'bw': 2}, -1),  # A 48.92
    '역갭리듬_반감20_κ10': ('갭리듬', {'half': 20, 'kappa': 10.0}, -1),  # A 49.18
    '변화점BOCPD_λ100_m100': ('변화점BOCPD', {'lam': 100, 'm': 100}, 1),  # A 49.34
    '역용수철진동자_W150_반감10': ('용수철진동자', {'W': 150, 'half': 10}, -1),  # A 49.49
}
ENSEMBLE = {
    '앙상블:A상위5': ['자기상관AR_L80_릿지2', 'SSA고유모드_L50_r5', '적응국면전환_반감20_K60', '자기상관개별_L30_W200', '푸리에위상_W128_K3'],
    '앙상블:전체14': ['자기상관AR_L80_릿지2', 'SSA고유모드_L50_r5', '적응국면전환_반감20_K60', '자기상관개별_L30_W200', '푸리에위상_W128_K3', '은닉마르코프_δ0.3_ρ0.2', '문맥트리_D4_β5', '해저드공통_κ500_bw0', '역해저드번호별_κ50_bw2', '역허스트국면_W100_k5_중심고정', '역해저드조건부_κ50_bw2', '역갭리듬_반감20_κ10', '변화점BOCPD_λ100_m100', '역용수철진동자_W150_반감10'],
}
NAMES = list(FINAL) + list(ENSEMBLE)
# 1000~1149회 평균 순위가 가장 낮아 고른 후보, 1150회 이후를 보기 전에 정함
PICK = "적응국면전환_반감20_K60"


def build(A, names=None):
    """이름 → (N+1, 46) 점수. 행 t는 앞 t개 회차만 사용"""
    names = names or NAMES
    cache, out = {}, {}

    def one(name):
        if name not in cache:
            fam, kw, sign = FINAL[name]
            cache[name] = sign * FAMILIES[fam](A, **kw)
        return cache[name]

    for name in names:
        if name in FINAL:
            out[name] = one(name)
        else:
            out[name] = sum(zrow(one(p)) for p in ENSEMBLE[name]) / len(ENSEMBLE[name])
    return out


def all_scores(df, names=None):
    return build(occ(df), names)


def scores_next(df):
    """df 마지막 회차 바로 다음 회차의 번호별 점수 {이름: 길이 46 배열}"""
    return {k: v[-1].copy() for k, v in all_scores(df).items()}
