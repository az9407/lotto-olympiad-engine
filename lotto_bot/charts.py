"""차트 기법·매크로 전환·머신러닝 후보를 매주 경쟁에 넣기 위한 계산.

scores_next(df): df 마지막 회차 다음 회차에 대한 번호별 점수, 이름 → 길이 46 배열
"""
import numpy as np
import pandas as pd

import explore as EX


MACRO_CSV = __import__("pathlib").Path(__file__).resolve().parent.parent / "pension720" / "data" / "macro.csv"
REAL_MACROS = ["gold", "sp500", "brent", "wti", "usdkrw", "usdjpy", "usdcny", "usdeur"]
_MACRO = {}


def real_macro_up(dates):
    """추첨일 하루 전까지 알려진 값 기준, 4주 전보다 올랐으면 True, 자료가 없으면 None. 연금복권 쪽 macro.csv를 읽기만 함"""
    if "df" not in _MACRO:
        try:
            m = pd.read_csv(MACRO_CSV)
            m["날짜"] = pd.to_datetime(m["날짜"])
            _MACRO["df"] = {k: g.sort_values("날짜").set_index("날짜")["값"] for k, g in m.groupby("지표")}
        except Exception:  # noqa: BLE001
            _MACRO["df"] = {}
    out = {}
    for k in REAL_MACROS:
        s = _MACRO["df"].get(k)
        vals = []
        for d in pd.to_datetime(pd.Series(list(dates))):
            if s is None:
                vals.append(None)
                continue
            now = s[s.index < d - pd.Timedelta(days=0)]
            past = s[s.index < d - pd.Timedelta(days=28)]
            if len(now) == 0 or len(past) == 0 or (d - now.index[-1]).days > 45:
                vals.append(None)
            else:
                vals.append(bool(now.iloc[-1] >= past.iloc[-1]))
        out[k] = vals
    return out


def _with_next_row(df):
    last = df.iloc[-1].copy()
    last["round"] = int(last["round"]) + 1
    last["date"] = pd.Timestamp(last["date"]) + pd.Timedelta(days=7)
    return pd.concat([df, last.to_frame().T], ignore_index=True).astype(df.dtypes.to_dict())


def names():
    ind_names = ["빈도5", "빈도10", "빈도20", "빈도30", "빈도50", "빈도100", "빈도200", "빈도전체",
                 "지수이평3", "지수이평6", "지수이평12", "지수이평26", "지수이평52", "MACD", "MACD히스토",
                 "단기장기차", "볼린저10", "볼린저20", "볼린저50", "스토캐스틱20", "스토캐스틱50",
                 "추세기울기10", "추세기울기30", "미출현", "미출현대비주기", "전이", "이월", "이웃", "직전보너스"]
    out = []
    for n in ind_names:
        out += [f"차트:{n}", f"차트:역{n}"]
    for m in ("직전1등당첨금", "직전1등당첨자수", "직전판매규모", "직전이월", "월"):
        for b in EX.SWITCH_BASES:
            out.append(f"매크로전환:{m}높으면{b}추세")
    out += list(EX.ml_models())
    for k in REAL_MACROS:
        for b in EX.SWITCH_BASES:
            out.append(f"실물매크로:{k}오르면{b}추세")
    import search as SE
    return out + SE.pool_names()


def pool_start():
    """자동 탐색으로 찾은 공식은 발견한 회차 이후 성적만 교체 판단에 씀"""
    import search as SE
    return {e["name"]: e["round"] for e in SE.load_pool()["defs"]}


def scores_next(df, with_ml=True):
    df2 = _with_next_row(df)
    t = len(df)
    ind, A = EX.indicators(df2)
    mac = EX.macro(df2)
    out = {}
    for n, S in ind.items():
        out[f"차트:{n}"] = S[t]
        out[f"차트:역{n}"] = -S[t]
    for mname, m in mac.items():
        if mname == "월":
            cond = m[t] % 2 == 0
        else:
            med = np.nanmedian(m[:t])
            cond = (m[t] if not np.isnan(m[t]) else med) >= med
        for b in EX.SWITCH_BASES:
            out[f"매크로전환:{mname}높으면{b}추세"] = ind[b][t] if cond else -ind[b][t]
    if with_ml:
        tr = np.arange(100, t)
        ytr = A[tr][:, 1:].reshape(-1)
        Ftr = EX.feature_rows(ind, mac, tr)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        Fte = (EX.feature_rows(ind, mac, np.array([t])) - mu) / sd
        for name, make in EX.ml_models().items():
            mdl = make().fit((Ftr - mu) / sd, ytr)
            s = np.zeros(46)
            s[1:] = np.log(np.clip(mdl.predict_proba(Fte)[:, 1], 1e-6, 1))
            out[name] = s
    up = real_macro_up([df2["date"].iloc[t]])
    for k in REAL_MACROS:
        u = up[k][0]
        for b in EX.SWITCH_BASES:
            out[f"실물매크로:{k}오르면{b}추세"] = ind[b][t] if (u is None or u) else -ind[b][t]
    import search as SE
    out.update(SE.pool_scores_next(df))
    return out


def pct_for_round(df, r, with_ml=True):
    """r회 실제 번호가 r-1회까지로 만든 각 후보 점수에서 상위 몇 %였나"""
    sub = df[df["round"] < r]
    row = df[df["round"] == r].iloc[0]
    act = [int(row[f"n{i}"]) for i in range(1, 7)]
    return {n: EX.rank_pct(s, act) for n, s in scores_next(sub, with_ml).items()}
