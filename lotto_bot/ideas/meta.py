"""메타·앙상블·온라인 학습 공식.

번호를 직접 고르지 않고, 번호별 공식 여러 개(전문가)를 지난 성적에 따라 섞거나 고릅니다.
- 전문가: search.py Prims 부품 27개의 표준점수와 그 반대 방향, 그리고 기권(모두 0) 1개, 합 55개
- 전문가의 t회 성적 g = 실제 1등 번호 6개의 표준점수 합 / sqrt(6·39/44), 무작위면 평균 0 표준편차 1
- t회 가중치는 t-1회까지의 성적만으로 정함, 미래 누수 없음

가족
- 헤지: 지수가중 다수결, 할인 γ로 오래된 성적을 잊음
- 고정공유헤지: 헤지 + 매회 α만큼 균등으로 되돌려 1등 전문가가 바뀌는 것을 따라감
- k회1등: 최근 k회 성적 합이 가장 높은 전문가 하나만 씀
- 톰슨: 전문가 평균 성적의 정규 사후분포에서 표본을 뽑아 가장 높은 전문가, 시드는 회차 번호라 재현됨
- 톰슨확률: 톰슨 표본 64개로 각 전문가가 1등일 확률을 가중치로 씀
- 성적차트정렬: 전문가 성적을 주가처럼 보고 짧은·중간·긴 창 평균이 모두 양수인 전문가만, 5분봉·1시간봉·일봉 정렬 발상
- 성적차트MACD: 전문가 성적의 짧은 지수이평이 긴 지수이평보다 높은 만큼 가중
- 성적차트돌파: 누적 성적이 최근 K회 신고가인 전문가만
- 국면kNN: 회차 전체 특징 13개(직전 합·홀수·저번호·연속·이월·최대간격·쏠림·당첨금 등)가 비슷했던 과거 회차에서 잘한 전문가
- 순위학습: 45개 번호 소프트맥스 리스트와이즈 선형 순위 모델, 매회 온라인 경사 갱신
- 머신러닝: 번호별 다중 시간 단위 특징 + 회차 전체 특징으로 엑스트라트리·랜덤포레스트·MLP, 50회마다 다시 학습
- 스태킹: 위 온라인 가족 대표들을 다시 헤지로 섞거나 평균

scores_next(df): df 마지막 회차 다음 회차의 번호별 점수 {이름: 길이 46 배열}
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import search as SE  # noqa: E402

SQ = np.sqrt(6 * 39 / 44)
START = 100          # 이 회차 인덱스부터 성적을 쌓음
ML_EVERY = 50        # 머신러닝은 인덱스 50의 배수마다 다시 학습, 그 앞 회차까지만 학습

BASE = [("빈도", 5), ("빈도", 10), ("빈도", 30), ("빈도", 100), ("빈도", 300),
        ("지수이평", 3), ("지수이평", 12), ("지수이평", 52),
        ("이평차", 3, 26), ("이평차", 8, 80), ("볼린저", 10), ("볼린저", 50),
        ("스토캐스틱", 20), ("CCI", 30), ("일목전환", 20), ("추세기울기", 10), ("추세기울기", 30),
        ("다중프레임정렬", 5, 30, 200), ("골든크로스", 5, 40),
        ("미출현",), ("주기대비미출현",), ("전이",), ("이월",), ("이웃",), ("직전보너스",),
        ("유사패턴", 20, 10, 50), ("유사패턴", 80, 5, 100)]

GRID = {
    "헤지": [dict(eta=e, gamma=g) for e in (0.01, 0.03, 0.1, 0.3) for g in (1.0, 0.99, 0.95)],
    "고정공유헤지": [dict(eta=e, alpha=a) for e in (0.1, 0.3, 1.0) for a in (0.01, 0.05)],
    "k회1등": [dict(k=k) for k in (1, 3, 10, 30, 100, 300)],
    "톰슨": [dict(gamma=g, avg=False) for g in (1.0, 0.99, 0.97)],
    "톰슨확률": [dict(gamma=g, avg=True) for g in (1.0, 0.99, 0.97)],
    "성적차트정렬": [dict(ks=ks) for ks in ((5, 25, 100), (3, 13, 52), (10, 50, 200))],
    "성적차트MACD": [dict(s=s, l=l) for s, l in ((5, 26), (12, 52), (3, 20))],
    "성적차트돌파": [dict(K=K) for K in (13, 52, 104)],
    "국면kNN": [dict(K=K, tau=tau) for K in (20, 60, 200) for tau in (0.5, 2.0)],
    "순위학습": [dict(lr=lr, lam=lam) for lr in (0.003, 0.01, 0.03) for lam in (0.002, 0.02)],
    "머신러닝ET": [dict(model="et", leaf=100), dict(model="et", leaf=400)],
    "머신러닝RF": [dict(model="rf", leaf=200)],
    "머신러닝MLP": [dict(model="mlp", alpha=0.1), dict(model="mlp", alpha=3.0)],
}
ML_FAMILIES = ("머신러닝ET", "머신러닝RF", "머신러닝MLP")

# 가족마다 A 300~999회 평균 순위가 가장 낮은 설정 하나. B·C는 보지 않고 고름
_MEMBERS = [("헤지", dict(eta=0.1, gamma=0.95)), ("고정공유헤지", dict(eta=0.3, alpha=0.01)),
            ("k회1등", dict(k=100)), ("톰슨", dict(gamma=1.0, avg=False)),
            ("톰슨확률", dict(gamma=0.97, avg=True)), ("성적차트정렬", dict(ks=(3, 13, 52))),
            ("성적차트MACD", dict(s=3, l=20)), ("성적차트돌파", dict(K=104)),
            ("국면kNN", dict(K=20, tau=0.5)), ("순위학습", dict(lr=0.003, lam=0.002))]
CHOSEN = {
    "메타:헤지": _MEMBERS[0],
    "메타:고정공유헤지": _MEMBERS[1],
    "메타:최근100회1등따라가기": _MEMBERS[2],
    "메타:톰슨": _MEMBERS[3],
    "메타:톰슨확률": _MEMBERS[4],
    "메타:성적차트정렬3_13_52": _MEMBERS[5],
    "메타:성적차트MACD3_20": _MEMBERS[6],
    "메타:성적차트돌파104": _MEMBERS[7],
    "메타:국면kNN20": _MEMBERS[8],
    "메타:순위학습": _MEMBERS[9],
    "메타:엑스트라트리": ("머신러닝ET", dict(model="et", leaf=400)),
    "메타:랜덤포레스트": ("머신러닝RF", dict(model="rf", leaf=200)),
    "메타:MLP": ("머신러닝MLP", dict(model="mlp", alpha=3.0)),
    "메타:스태킹평균": ("스태킹평균", dict(members=_MEMBERS)),
    "메타:스태킹헤지": ("스태킹헤지", dict(members=_MEMBERS, eta=0.05)),
}


def with_next_row(df):
    last = df.iloc[-1].copy()
    last["round"] = int(last["round"]) + 1
    last["date"] = pd.Timestamp(last["date"]) + pd.Timedelta(days=7)
    return pd.concat([df, last.to_frame().T], ignore_index=True).astype(df.dtypes.to_dict())


def vname(fam, p):
    return fam + ":" + ",".join(f"{k}={v}" for k, v in p.items())


class Meta:
    """df2 하나(마지막 행은 다음 회차 자리)에 대한 전문가 은행과 학습기"""

    def __init__(self, df2):
        self.pr = pr = SE.Prims(df2)
        self.N = N = pr.N
        self.X = pr.X
        Zb = np.stack([SE._z(pr.get(k)) for k in BASE])                   # (27, N, 46)
        self.Zb = Zb
        self.Gb = Zb[:, np.arange(N)[:, None], pr.X].sum(-1) / SQ          # (27, N)
        nb = len(BASE)
        self.E = 2 * nb + 1
        self.G = np.vstack([self.Gb, -self.Gb, np.zeros((1, N))])          # (55, N)
        self.ctx = self._context(df2)
        self.cache = {}

    # 전문가 가중치 (N, E) → 번호별 점수 (N, 46)
    def combine(self, W):
        nb = len(BASE)
        Wb = W[:, :nb] - W[:, nb:2 * nb]
        return np.einsum("te,etn->tn", Wb, self.Zb)

    def _context(self, df2):
        pr, X, N, A = self.pr, self.pr.X, self.pr.N, self.pr.A
        c = np.zeros((N, 13))
        prev = X[:-1]
        c[1:, 0] = prev.sum(1)
        c[1:, 1] = (prev % 2).sum(1)
        c[1:, 2] = (prev <= 22).sum(1)
        c[1:, 3] = (np.diff(prev, axis=1) == 1).sum(1)
        c[2:, 4] = (A[1:-1] * A[:-2]).sum(1)
        c[1:, 5] = np.diff(np.c_[np.zeros(N - 1), prev, np.full(N - 1, 46)], axis=1).max(1)
        c[:, 6] = pr.get(("빈도", 30))[:, 1:].std(1)
        c[:, 7] = pr.get(("빈도", 10))[:, 1:].std(1)
        c[:, 8] = pr.get(("미출현",))[:, 1:].mean(1)
        for j, m in enumerate(("직전1등당첨금", "직전1등당첨자수", "직전이월")):
            v = pr.mac[m]
            c[:, 9 + j] = np.nan_to_num(v, nan=0.0)
        c[1:, 12] = [len(set(r % 10)) for r in prev]
        return c

    # ---------------- 온라인 학습기, 반환 (N, E) 가중치 ----------------
    def hedge(self, eta, gamma=1.0):
        G, E, N = self.G, self.E, self.N
        W = np.full((N, E), 1 / E)
        lw = np.zeros(E)
        for t in range(N):
            w = np.exp(lw - lw.max())
            W[t] = w / w.sum()
            if t >= START:
                lw = gamma * lw + eta * G[:, t]
        return W

    def fixed_share(self, eta, alpha):
        G, E, N = self.G, self.E, self.N
        W = np.full((N, E), 1 / E)
        w = np.full(E, 1 / E)
        for t in range(N):
            W[t] = w
            if t >= START:
                lw = np.log(w) + eta * G[:, t]
                w = np.exp(lw - lw.max())
                w /= w.sum()
                w = (1 - alpha) * w + alpha / E
        return W

    def _cum(self):
        if "cum" not in self.cache:
            G = self.G.copy()
            G[:, :START] = 0
            self.cache["cum"] = np.hstack([np.zeros((self.E, 1)), np.cumsum(G, 1)])  # cum[:, t] = t-1까지 합
        return self.cache["cum"]

    def follow_k(self, k):
        cum, E, N = self._cum(), self.E, self.N
        W = np.zeros((N, E))
        W[:, -1] = 1
        for t in range(START + 1, N):
            lo = max(START, t - k)
            s = cum[:, t] - cum[:, lo]
            W[t] = 0
            W[t, int(np.argmax(s))] = 1
        return W

    def thompson(self, gamma, avg, kappa=10.0, draws=64):
        G, E, N = self.G, self.E, self.N
        W = np.zeros((N, E))
        W[:, -1] = 1
        n = np.zeros(E)
        s = np.zeros(E)
        for t in range(N):
            if t > START:
                rng = np.random.default_rng(7919 * (t + 1) + 13)
                mu = s / (n + kappa)
                sd = 1 / np.sqrt(n + kappa)
                if avg:
                    th = mu + sd * rng.standard_normal((draws, E))
                    W[t] = np.bincount(th.argmax(1), minlength=E) / draws
                else:
                    th = mu + sd * rng.standard_normal(E)
                    W[t] = 0
                    W[t, int(th.argmax())] = 1
            if t >= START:
                n = gamma * n + 1
                s = gamma * s + G[:, t]
        return W

    def _winmean(self, k, t):
        cum = self._cum()
        lo = max(START, t - k)
        return (cum[:, t] - cum[:, lo]) / max(t - lo, 1)

    def chart_align(self, ks):
        E, N = self.E, self.N
        W = np.zeros((N, E))
        W[:, -1] = 1
        for t in range(START + max(ks[:1]) + 1, N):
            ms = np.stack([self._winmean(k, t) for k in ks])
            ok = (ms > 0).all(0)
            ok[-1] = False
            W[t] = 0
            if ok.any():
                W[t, ok] = ms[:, ok].mean(0)
            else:
                W[t, int(np.argmax(ms[-1]))] = 1
            W[t] /= W[t].sum()
        return W

    def chart_macd(self, s, l):
        G, E, N = self.G, self.E, self.N
        a_s, a_l = 1 - 0.5 ** (1 / s), 1 - 0.5 ** (1 / l)
        es, el = np.zeros(E), np.zeros(E)
        W = np.zeros((N, E))
        W[:, -1] = 1
        for t in range(N):
            if t > START + l:
                d = np.maximum(es - el, 0)
                d[-1] = 0
                if d.sum() > 0:
                    W[t] = d / d.sum()
            if t >= START:
                es = (1 - a_s) * es + a_s * G[:, t]
                el = (1 - a_l) * el + a_l * G[:, t]
        return W

    def chart_breakout(self, K):
        cum, E, N = self._cum(), self.E, self.N
        W = np.zeros((N, E))
        W[:, -1] = 1
        for t in range(START + K + 1, N):
            cur = cum[:, t]
            past = cum[:, t - K:t].max(1)
            ok = cur > past
            ok[-1] = False
            if ok.any():
                W[t] = 0
                W[t, ok] = 1 / ok.sum()
        return W

    def regime_knn(self, K, tau):
        G, E, N, C = self.G, self.E, self.N, self.ctx
        W = np.full((N, E), 1 / E)
        for t in range(START + K, N):
            hist = C[START:t + 1]
            mu, sd = hist.mean(0), hist.std(0) + 1e-9
            Z = (C[START:t] - mu) / sd
            q = (C[t] - mu) / sd
            d = ((Z - q) ** 2).sum(1)
            nn = np.argpartition(d, K - 1)[:K] + START
            mg = G[:, nn].mean(1) * np.sqrt(K)
            lw = mg / tau
            w = np.exp(lw - lw.max())
            W[t] = w / w.sum()
        return W

    # ---------------- 직접 번호 점수 (N, 46) ----------------
    def listwise(self, lr, lam):
        Zb, X, N = self.Zb, self.X, self.N
        nb = Zb.shape[0]
        th = np.zeros(nb)
        S = np.zeros((N, 46))
        for t in range(N):
            z = Zb[:, t, 1:]                       # (27, 45)
            sc = th @ z
            S[t, 1:] = sc
            if t >= START:
                p = np.exp(sc - sc.max())
                p /= p.sum()
                grad = z[:, X[t] - 1].sum(1) - 6 * z @ p
                th = (1 - lam) * th + lr * grad
        return S

    def _ml_features(self):
        if "mlF" not in self.cache:
            pr, N = self.pr, self.N
            parts = [np.transpose(self.Zb, (1, 2, 0))[:, 1:, :]]
            raw = [pr.get(("빈도", w)) for w in (5, 10, 30, 100, 300)] + [pr.get(("미출현",))]
            parts.append(np.stack(raw, -1)[:, 1:, :])
            num = np.arange(1, 46)
            idf = np.stack([num / 45, num // 10, num % 10], -1).astype(float)
            parts.append(np.broadcast_to(idf, (N, 45, 3)))
            parts.append(np.broadcast_to(self.ctx[:, None, :], (N, 45, self.ctx.shape[1])))
            self.cache["mlF"] = np.concatenate(parts, -1)
        return self.cache["mlF"]

    def ml(self, ts, model, leaf=200, alpha=1.0):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        F = self._ml_features()
        Y = self.pr.A[:, 1:]
        S = np.zeros((self.N, 46))
        ts = np.asarray(ts)
        for b in sorted(set((ts // ML_EVERY) * ML_EVERY)):
            te = ts[(ts // ML_EVERY) * ML_EVERY == b]
            tr = np.arange(START, b)
            if len(tr) < 100:
                continue
            Ftr = F[tr].reshape(-1, F.shape[-1])
            ytr = Y[tr].reshape(-1)
            Fte = F[te].reshape(-1, F.shape[-1])
            if model == "et":
                m = ExtraTreesClassifier(n_estimators=60, min_samples_leaf=leaf, max_features=0.3,
                                         random_state=0, n_jobs=1)
            elif model == "rf":
                m = RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_leaf=leaf,
                                           max_features=0.3, random_state=0, n_jobs=1)
            else:
                mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
                Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
                m = MLPClassifier(hidden_layer_sizes=(32,), alpha=alpha, max_iter=30, batch_size=512,
                                  learning_rate_init=1e-3, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Ftr, ytr)
            S[te, 1:] = np.log(np.clip(m.predict_proba(Fte)[:, 1], 1e-6, 1)).reshape(len(te), 45)
        return S

    # ---------------- 가족 이름 → 번호별 점수 (N, 46) ----------------
    def family(self, fam, p, ts=None):
        key = vname(fam, p)
        if key in self.cache:
            return self.cache[key]
        if fam == "헤지":
            S = self.combine(self.hedge(p["eta"], p["gamma"]))
        elif fam == "고정공유헤지":
            S = self.combine(self.fixed_share(p["eta"], p["alpha"]))
        elif fam == "k회1등":
            S = self.combine(self.follow_k(p["k"]))
        elif fam in ("톰슨", "톰슨확률"):
            S = self.combine(self.thompson(p["gamma"], p["avg"]))
        elif fam == "성적차트정렬":
            S = self.combine(self.chart_align(p["ks"]))
        elif fam == "성적차트MACD":
            S = self.combine(self.chart_macd(p["s"], p["l"]))
        elif fam == "성적차트돌파":
            S = self.combine(self.chart_breakout(p["K"]))
        elif fam == "국면kNN":
            S = self.combine(self.regime_knn(p["K"], p["tau"]))
        elif fam == "순위학습":
            S = self.listwise(p["lr"], p["lam"])
        elif fam in ML_FAMILIES:
            S = self.ml(np.arange(self.N) if ts is None else ts, p["model"], p.get("leaf", 200), p.get("alpha", 1.0))
            return S
        elif fam in ("스태킹헤지", "스태킹평균"):
            S = self.stack(p)
        else:
            raise KeyError(fam)
        self.cache[key] = S
        return S

    def stack(self, p):
        """p['members']: [(가족, 설정)] 온라인 가족 대표들, 표준점수로 맞춘 뒤 헤지 또는 평균"""
        mem = [SE._z(self.family(f, q)) for f, q in p["members"]]
        Zm = np.stack(mem)                                               # (M, N, 46)
        if "eta" not in p:
            return Zm.mean(0)
        Gm = Zm[:, np.arange(self.N)[:, None], self.X].sum(-1) / SQ      # (M, N)
        M = len(mem)
        W = np.full((self.N, M), 1 / M)
        lw = np.zeros(M)
        for t in range(self.N):
            w = np.exp(lw - lw.max())
            W[t] = w / w.sum()
            if t >= START:
                lw = lw + p["eta"] * Gm[:, t]
        return np.einsum("tm,mtn->tn", W, Zm)


def all_scores(df, names=None, ts=None):
    """df + 다음 회차 자리 1행에 대해 이름 → (len(df)+1, 46). t행은 t-1행까지만으로 계산"""
    df2 = with_next_row(df)
    mt = Meta(df2)
    out = {}
    for n in (names or NAMES):
        fam, p = CHOSEN[n]
        out[n] = mt.family(fam, p, ts=ts)
    return out


def scores_next(df):
    t = len(df)
    S = all_scores(df, ts=np.array([t]))
    return {n: v[t] for n, v in S.items()}


NAMES = list(CHOSEN)
