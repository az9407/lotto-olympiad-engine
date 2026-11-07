#!/usr/bin/env python3
"""로또 새 회차가 올라오면 분석 + 자동 학습 + 다음 회차 후보 10개를 텔레그램으로 보냅니다.

python lotto_bot/run_lotto.py             새 회차만 처리
python lotto_bot/run_lotto.py --force     마지막 회차를 다시 처리
python lotto_bot/run_lotto.py --dry-run   보내지 않고 화면에만 출력
환경변수 TELEGRAM_BOT_TOKEN 필수, TELEGRAM_CHAT_ID 선택
대화방은 연금복권 봇이 저장한 pension720/data/telegram_chat.enc 를 같이 씁니다.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts as CH        # noqa: E402
import formulas as FO      # noqa: E402
import lotto_engine as LE  # noqa: E402
import tuner as TU         # noqa: E402

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
STATE = BASE / "state"
RESULTS = BASE / "results"
CHAT_FILES = [REPO / "pension720" / "data" / "telegram_chat.enc", STATE / "telegram_chat.enc"]
DATA_URLS = [
    "https://raw.githubusercontent.com/nostra-team/nostra-data/main/history.json",
    "https://software-lucas.github.io/nostra-data/history.json",
]
K = 10
LOTTO_RANK = {6: "1등", 5: "3등", 4: "4등", 3: "5등"}


def load_history(path=None):
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    last = None
    for url in DATA_URLS:
        for _ in range(3):
            try:
                req = urllib.request.Request(url + f"?t={int(time.time())}", headers={"User-Agent": "lotto-bot"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(5)
    raise RuntimeError(f"당첨번호 데이터를 못 받았습니다: {last}")


def lotto_df(h):
    rows = []
    for d in h["lotto"]:
        n = sorted(d["numbers"])
        pz = d["prizes"]
        rows.append(dict(round=d["round"], date=pd.to_datetime(d["date"], format="%Y%m%d"),
                         **{f"n{i+1}": n[i] for i in range(6)}, bonus=d["bonus"],
                         w1=pz[0][0], p1=pz[0][1], w2=pz[1][0], p2=pz[1][1]))
    df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    assert (df["round"].to_numpy() == range(1, len(df) + 1)).all(), "로또 회차가 연속이 아닙니다"
    return df


def load_state():
    p = STATE / "lotto.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(st):
    STATE.mkdir(exist_ok=True)
    (STATE / "lotto.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def token():
    t = (os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN") or "").strip()
    if not t:
        raise SystemExit("TELEGRAM_BOT_TOKEN이 없어 보내지 못했습니다.")
    return t


def tg(method, **params):
    req = urllib.request.Request(f"https://api.telegram.org/bot{token()}/{method}",
                                 data=urllib.parse.urlencode(params).encode())
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode())
    if not out.get("ok"):
        raise RuntimeError(f"텔레그램 {method} 실패: {out.get('description')}")
    return out["result"]


# 대화방 번호는 공개 저장소라 봇 토큰에서 만든 키로 암호화해 둠, 연금복권 봇과 같은 방식
def _key(tok):
    return hashlib.sha256(b"pension720-chat:" + tok.encode()).digest()


def _stream(key, nonce, n):
    out = b""
    for c in range((n + 31) // 32):
        out += hmac.new(key, nonce + c.to_bytes(4, "big"), hashlib.sha256).digest()
    return out[:n]


def seal(tok, text):
    key, nonce, data = _key(tok), secrets.token_bytes(16), text.encode()
    body = bytes(a ^ b for a, b in zip(data, _stream(key, nonce, len(data))))
    return base64.b64encode(nonce + body + hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]).decode()


def unseal(tok, blob):
    raw = base64.b64decode(blob)
    key, nonce, body, tag = _key(tok), raw[:16], raw[16:-16], raw[-16:]
    if not hmac.compare_digest(tag, hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]):
        return None
    return bytes(a ^ b for a, b in zip(body, _stream(key, nonce, len(body)))).decode()


def chat_ids():
    if os.environ.get("TELEGRAM_CHAT_ID"):
        return [c.strip() for c in os.environ["TELEGRAM_CHAT_ID"].split(",") if c.strip()]
    tok = token()
    for f in CHAT_FILES:
        if f.exists():
            text = unseal(tok, f.read_text().strip())
            if text:
                return text.split(",")
    found = set()
    for u in tg("getUpdates"):
        for k in ("message", "my_chat_member"):
            chat = u.get(k, {}).get("chat", {})
            if chat.get("type") == "private":
                found.add(str(chat["id"]))
    if not found:
        raise SystemExit("봇 대화방을 찾지 못했습니다. 텔레그램에서 @lotto_pension_bot 에 /start 를 보내 주세요.")
    STATE.mkdir(exist_ok=True)
    (STATE / "telegram_chat.enc").write_text(seal(tok, ",".join(sorted(found))) + "\n")
    return sorted(found)


def send(text, dry):
    if dry:
        print(text)
        return
    for cid in chat_ids():
        for i in range(0, len(text), 3900):
            tg("sendMessage", chat_id=cid, text=text[i:i + 3900], disable_web_page_preview="true")
            time.sleep(1)


def won(v):
    v = int(v)
    if v >= 100_000_000:
        eok, man = divmod(v, 100_000_000)
        man //= 10_000
        return f"{eok:,}억{man:,}만원" if man else f"{eok:,}억원"
    if v >= 10_000:
        return f"{v // 10_000:,}만원"
    return f"{v:,}원"


def ymd(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def rule_lines(rules, n=2):
    hit = sorted([s for s, h in rules if h], key=lambda s: -s["w"])[:n]
    miss = sorted([s for s, h in rules if not h], key=lambda s: -s["w"])[:n]
    return ([f"  맞음  {s['cond']} → {s['event']}" for s in hit]
            + [f"  빗나감 {s['cond']} → {s['event']}" for s in miss])


def lotto_rank(c, nums, bonus):
    m = len(set(c) & set(nums))
    if m == 5 and bonus in c:
        return m, "2등"
    return m, LOTTO_RANK.get(m, "")


def search_lines(N):
    import search as SE
    pool = SE.load_pool()
    if not pool["defs"]:
        return []
    week = [r for r in pool.get("runs", []) if r["round"] >= N - 1]
    L = [f"- 자동 탐색: 지금까지 {pool['tested']:,}개 시험, 후보 풀 {len(pool['defs'])}개, 이번 주 {sum(r['tested'] for r in week):,}개 시험"]
    L.append("  새 공식은 발견 다음 회차부터 10회 이상 쌓여야 교체 경쟁에 들어감")
    return L


def process(h, st):
    df = lotto_df(h)
    last = df.iloc[-1]
    N = int(last["round"])
    nums = [int(last[f"n{i}"]) for i in range(1, 7)]
    bonus = int(last["bonus"])

    def evaluate(params, r, keep):
        row = df[df["round"] == r].iloc[0]
        act = [int(row[f"n{i}"]) for i in range(1, 7)]
        m = LE.build_model(df[df["round"] < r], params)
        pct = LE.rank_pct(m["FINAL_F"], LE.comb_index(act))
        extra = (LE.analyze(m, act, int(row["bonus"])), LE.pick(m, K)) if keep else None
        del m
        return pct, extra

    lead = st.get("config_id", "A")
    tuner, extra, rep, params = TU.step("lotto", st.get("tuner"), N, LE.BASE_PARAMS, evaluate, lead)
    if extra is None:
        _, extra = evaluate(params, N, True)
    a, recomputed = extra
    if st.get("target") == N and st.get("candidates"):
        prev_c, label = [c["numbers"] for c in st["candidates"]], "보낸 후보 10개"
    else:
        prev_c, label = [c["numbers"] for c in recomputed], "엔진 재계산 후보 10개"

    def eval_formulas(r):
        row = df[df["round"] == r].iloc[0]
        idx = LE.comb_index([int(row[f"n{i}"]) for i in range(1, 7)])
        res = {n: LE.rank_pct(S, idx) for n, S in FO.score_all(df[df["round"] < r]).items()}
        res.update(CH.pct_for_round(df, r))
        res.update({n: LE.rank_pct(S, idx) for n, S in CH.combo_scores_next(df[df["round"] < r]).items()})
        return res

    starts = CH.pool_start()
    this_f = TU.formula_step(tuner, N, FO.ALL + CH.names() + CH.combo_names(), eval_formulas,
                             no_backfill=set(starts))
    this_all = {**{c["id"]: c["hist"].get(str(N)) for c in tuner["configs"]},
                **{(n if ":" in n else f"공식:{n}"): v for n, v in this_f.items()}}
    dec = TU.decide(tuner, N, default=st.get("champion") or lead, start_after=starts)
    champ = dec["champion"]
    T, TDATE = N + 1, pd.Timestamp(last["date"]) + pd.Timedelta(days=7)
    nxt = None
    if champ.startswith("공식:"):
        S = FO.score_all(df, [champ[3:]])[champ[3:]]
        picks = LE.pick_array(S, K)
        del S
        new_lead = rep["chosen"]
    elif champ in CH.combo_names():
        S = CH.combo_scores_next(df)[champ]
        picks = LE.pick_array(S, K)
        del S
        new_lead = rep["chosen"]
    elif champ in CH.names():
        S = FO.combo_from_num(CH.scores_next(df)[champ])
        picks = LE.pick_array(S, K)
        del S
        new_lead = rep["chosen"]
    else:
        cfg = next(c for c in tuner["configs"] if c["id"] == champ)
        nxt = LE.build_model(df, cfg["params"])
        picks = LE.pick(nxt, K)
        new_lead = champ

    L = [f"로또 {N}회 결과 {ymd(last['date'])}",
         f"당첨번호 {' '.join(f'{n:02d}' for n in nums)} + 보너스 {bonus:02d}",
         f"1등 {int(last['w1'])}명 · 1인당 {won(last['p1'])}", "",
         f"지난 회차 {label}"]
    for i, c in enumerate(prev_c, 1):
        m, rk = lotto_rank(c, nums, bonus)
        L.append(f"{i:>2}. {' '.join(f'{n:02d}' for n in c)} → {m}개{' ' + rk if rk else ''}")
    L += ["", "왜 이 번호였나",
          f"- 합계 {a['sum']}: 최근 50회 평균 {a['sum50']:.0f}, 전체 조합의 {a['sum_pct_low']:.0f}%가 이보다 작음",
          f"- 홀짝 {a['odd']}:{6-a['odd']} · 1~22 {a['low']}개 · AC {a['ac']} · 연번 {a['consec']}쌍 · 끝수합 {a['tail_sum']}",
          f"- 직전회 이월 {len(a['carry'])}개{' ' + ','.join(map(str, a['carry'])) if a['carry'] else ''} · 이웃수 {a['neighbor']}개",
          f"- 엔진 규칙 {len(a['rules'])}개 중 {a['n_hit']}개 적중",
          *rule_lines(a["rules"]),
          f"- 엔진 순위 {a['total']:,}개 중 {a['rank']:,}위, 상위 {a['pct']:.1f}%",
          f"- 인기회피 상위 {a['pop_top']:.0f}% · 최근 50회 구성 필터 {'통과' if a['in_profile'] else '밖'}",
          *TU.report_lines("lotto", rep),
          *TU.champion_lines(dec, this_all),
          *search_lines(N),
          "", f"로또 {T}회 패턴 상위 10개 {ymd(TDATE)} 추첨 · 챔피언 {champ}"]
    for c in picks:
        tail = f" · 규칙 {c['hits']}/{c['rules']}" if "hits" in c else ""
        L.append(f"{c['rank']:>2}. {' '.join(f'{n:02d}' for n in c['numbers'])}  점수 {c['score']:.3f}{tail}")
    st_new = dict(last_round=N, target=T, target_date=ymd(TDATE), candidates=picks,
                  config_id=new_lead, champion=champ, tuner=tuner,
                  rules=([{k: s[k] for k in ("id", "rule", "basis", "r250", "r50", "w")} for s in nxt["selected"]]
                         if nxt else []))
    return "\n".join(L), st_new, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="history.json 경로, 생략하면 인터넷에서 받음")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    h = load_history(args.data)
    st = load_state()
    latest = max(d["round"] for d in h["lotto"])
    if st.get("last_round", 0) >= latest and not args.force:
        print(f"로또 {latest}회 이미 처리함")
        return
    t0 = time.time()
    text, st_new, N = process(h, st)
    send(text, args.dry_run)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"lotto_{N:04d}.txt").write_text(text, encoding="utf-8")
    save_state(st_new)
    print(f"로또 {N}회 처리 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
