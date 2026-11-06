#!/usr/bin/env python3
"""연금복권720+ 주간 자동 갱신.

새 회차를 받아 data/당첨번호.csv에 추가 → 방금 나온 번호를 지난주 모델로 분석 →
다음 회차 후보를 계산 → 텔레그램으로 보낸다. GitHub Actions가 목요일 추첨 뒤 여러 번
실행하고, 이미 보낸 회차면 바로 끝난다.

환경변수
  TELEGRAM_BOT_TOKEN  보낼 때 필요
  TELEGRAM_CHAT_ID    선택. 없으면 봇에 /start를 보낸 대화방을 찾아 암호화해 저장한다.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import secrets
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict_pension15 as eng  # noqa: E402

DATA = eng.DATA_CSV
XLSX = HERE / 'data' / '연금복권720_역대당첨번호.xlsx'
RESULTS = HERE / 'results'
CHAT_FILE = HERE / 'data' / 'telegram_chat.enc'
SEND_COUNT = 10
RESULTS_URL = ('https://github.com/az9407/lotto-olympiad-engine/tree/'
               'feature/lotto-and-pension-engines/pension720/results')
DHLOTTERY = 'https://www.dhlottery.co.kr/pt720/selectPstPt720Info.do'
NOSTRA = 'https://raw.githubusercontent.com/nostra-team/nostra-data/main/history.json'
KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------- 자료

def read_rows():
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_rows(rows):
    with DATA.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=eng.HEADERS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = '당첨번호'
    sheet.append(list(eng.HEADERS))
    for r in rows:
        sheet.append([int(r['회차']), date.fromisoformat(r['추첨일']), int(r['1등 조']), r['1등 번호'], r['보너스 번호']])
        sheet.cell(sheet.max_row, 2).number_format = 'yyyy-mm-dd'
    for col, width in zip('ABCDE', (8, 13, 9, 12, 13)):
        sheet.column_dimensions[col].width = width
    book.save(XLSX)


def _get_json(url):
    request = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36',
        'Accept': 'application/json', 'Referer': 'https://www.dhlottery.co.kr/'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _record(no, ymd, group, number, bonus):
    ymd = str(ymd).replace('-', '')
    return {'회차': str(int(no)), '추첨일': f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}', '1등 조': str(int(group)),
            '1등 번호': str(number).zfill(6), '보너스 번호': str(bonus).zfill(6)}


def from_dhlottery(last):
    """동행복권 공식 API. 인자 없이 부르면 최근 여러 회차가 온다."""
    rows = _get_json(DHLOTTERY)['data']['result']
    by_round = {}
    for r in rows:
        by_round.setdefault(int(r['psltEpsd']), []).append(r)
    out = []
    for no, items in by_round.items():
        first = next((r for r in items if int(r['wnSqNo']) == 1), None)
        bonus = next((r for r in items if int(r['wnSqNo']) == 21), None)
        if no > last and first and bonus:
            out.append(_record(no, first['psltRflYmd'], first['wnBndNo'], first['wnRnkVl'], bonus['wnRnkVl']))
    return out


def from_nostra(last):
    """동행복권 결과를 모아 두는 공개 저장소 nostra-team/nostra-data."""
    return [_record(r['round'], r['date'], r['group'], r['number'], r['bonus'])
            for r in _get_json(NOSTRA)['pension'] if int(r['round']) > last]


def fetch_new(last_round):
    """두 출처에서 last_round 다음 회차부터 이어지는 결과. 출처끼리 다르면 멈춘다."""
    found, errors = {}, []
    for name, fetch in (('동행복권', from_dhlottery), ('nostra-data', from_nostra)):
        try:
            records = fetch(last_round)
        except Exception as error:  # 한 출처가 막혀도 다른 출처로 진행
            errors.append(f'{name}: {type(error).__name__} {error}')
            continue
        for r in records:
            no = int(r['회차'])
            if no in found and found[no] != r:
                raise SystemExit(f'{no}회 결과가 출처마다 다릅니다: {found[no]} / {r}')
            found.setdefault(no, r)
    new, no = [], last_round + 1
    while no in found:
        new.append(found[no])
        no += 1
    return new, errors


# ---------------------------------------------------------------- 분석

PRIZE_TAILS = ((5, '3등'), (4, '4등'), (3, '5등'), (2, '6등'), (1, '7등'))


def prizes(group, number, draw_group, draw_number, bonus):
    out = []
    if number == draw_number:
        out.append('1등' if group == draw_group else '2등')
    else:
        out.extend(name for k, name in PRIZE_TAILS if number[-k:] == draw_number[-k:])
        out = out[:1]
    if number == bonus:
        out.append('보너스')
    return out


def read_setting(folder):
    report = folder / '결과.md'
    if not report.exists():
        return None
    labels = {s.label(): s for s in eng.SETTINGS}
    for line in report.read_text(encoding='utf-8').splitlines():
        if line.startswith('선택 설정: '):
            return labels.get(line.removeprefix('선택 설정: ').strip())
    return None


def read_candidates(folder, count=SEND_COUNT):
    path = folder / '후보15.csv'
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [(int(r['순위']), int(r['조']), r['6자리'], r['종합점수 S']) for r in csv.DictReader(f)][:count]


def analyze(draws, setting, previous):
    """마지막 회차를, 그 회차 직전 자료로 학습한 모델(지난주 모델)로 본다."""
    i = len(draws.rounds) - 1
    context = eng.Context(draws, i)
    space = eng.Space()
    models = []
    for stream in (eng.MAIN, eng.BONUS):
        catalog = eng.build_catalog(draws, stream)
        models.append(eng.RuleFold(catalog, eng.HistoryIndex(catalog), context, space).fit(setting))
    scores, parts = eng.score_all(draws, setting, models, index=i)
    group = int(draws.groups[i])
    main = ''.join(map(str, draws.main[i]))
    bonus = ''.join(map(str, draws.bonus[i]))
    actual = (group - 1) * eng.N_NUMBERS + int(main)
    bonus_scores = parts[1, :eng.N_NUMBERS]   # 보너스 모델은 조를 쓰지 않는다
    age = context.ages(2)[0][group - 1]

    def rule_table(model, cid):
        passed = [bool(p[0]) for p in model.passes([cid])]
        return [(r.rule.formula, r.adjusted, ok) for r, ok in zip(model.rules, passed)]

    return {
        'round': int(draws.rounds[i]), 'date': draws.dates[i].isoformat(),
        'group': group, 'main': main, 'bonus': bonus, 'setting': setting,
        'rank': 1 + int(np.count_nonzero(scores > scores[actual])),
        'bonus_rank': 1 + int(np.count_nonzero(bonus_scores > bonus_scores[int(bonus)])),
        'main_rules': rule_table(models[0], actual),
        'bonus_rules': rule_table(models[1], int(bonus)),
        'group_streak': '직전 회차에 이어 다시' if age == 0 else f'{age + 1}회 만에',
        'summary': eng.summary(int(main)),
        'previous': [(rank, g, num, prizes(g, num, group, main, bonus)) for rank, g, num, _ in previous],
    }


def analysis_lines(a, top=3):
    main_hit = [r for r in a['main_rules'] if r[2]]
    main_miss = sorted((r for r in a['main_rules'] if not r[2]), key=lambda r: -r[1])
    bonus_hit = sum(r[2] for r in a['bonus_rules'])
    lines = [
        f"· 지난주 모델 점수 순위: 500만 중 {a['rank']:,}위, 상위 {100 * a['rank'] / eng.N_CANDIDATES:.1f}%",
        f"· 1등 규칙 {len(a['main_rules'])}개 중 {len(main_hit)}개 충족",
        *(f'  맞음 {f}' for f, _, _ in main_hit[:top]),
        *(f'  빗나감 {f}' for f, _, _ in main_miss[:top]),
        f"· 보너스 번호: 보너스 규칙 {len(a['bonus_rules'])}개 중 {bonus_hit}개 충족, "
        f"100만 중 상위 {100 * a['bonus_rank'] / eng.N_NUMBERS:.1f}%",
        f"· {a['group']}조는 {a['group_streak']} 1등 조",
        f"· 번호 특징: {a['summary']}",
    ]
    return lines


def previous_lines(a):
    if not a['previous']:
        return []
    wins = [(rank, g, num, p) for rank, g, num, p in a['previous'] if p]
    if not wins:
        return [f"지난주 후보 {len(a['previous'])}개 성적", '· 당첨 없음']
    return [f"지난주 후보 {len(a['previous'])}개 성적",
            *(f"· {rank}. {g}조 {num} → {'·'.join(p)}" for rank, g, num, p in wins)]


def write_analysis(folder, a):
    folder.mkdir(parents=True, exist_ok=True)
    def table(rows):
        body = '\n'.join(f"| {'O' if ok else 'X'} | {w:.4f} | {f.replace('|', chr(92) + '|')} |" for f, w, ok in rows)
        return f'| 충족 | 가중치 | 규칙 |\n| --- | ---: | --- |\n{body}'
    text = (f"# 연금복권720+ {a['round']}회 분석 · {a['date']}\n\n"
            f"1등 {a['group']}조 {a['main']} · 보너스 각조 {a['bonus']}\n\n"
            f"지난주 모델 설정: {a['setting'].label()}\n\n"
            + '\n'.join(analysis_lines(a, top=5)) + '\n\n'
            + '\n'.join(previous_lines(a)) + '\n\n'
            f"## 1등 규칙\n\n{table(a['main_rules'])}\n\n## 보너스 규칙\n\n{table(a['bonus_rules'])}\n")
    (folder / '분석.md').write_text(text, encoding='utf-8')


def compose(a, target, target_date, setting, candidates):
    lines = [f"연금복권720+ {a['round']}회 결과 · {a['date']}",
             f"1등 {a['group']}조 {a['main']}", f"보너스 각조 {a['bonus']}", '',
             '엔진으로 본 이번 번호', *analysis_lines(a)]
    previous = previous_lines(a)
    if previous:
        lines += ['', *previous]
    lines += ['', f'{target}회 후보 {len(candidates)}개 · {target_date}',
              *(f'{rank}. {g}조 {num}' for rank, g, num, _ in candidates),
              '', f'설정: {setting.label()}', f'{RESULTS_URL}/round_{target}']
    return '\n'.join(lines)


# ---------------------------------------------------------------- 텔레그램

def telegram(token, method, **params):
    request = urllib.request.Request(f'https://api.telegram.org/bot{token}/{method}',
                                     data=urllib.parse.urlencode(params).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        out = json.load(response)
    if not out.get('ok'):
        raise RuntimeError(f"텔레그램 {method} 실패: {out.get('description')}")
    return out['result']


def _key(token):
    return hashlib.sha256(b'pension720-chat:' + token.encode()).digest()


def _stream(key, nonce, n):
    out = b''
    for counter in range((n + 31) // 32):
        out += hmac.new(key, nonce + counter.to_bytes(4, 'big'), hashlib.sha256).digest()
    return out[:n]


def seal(token, text):
    """대화방 번호를 공개 저장소에 그대로 두지 않으려고 봇 토큰에서 만든 키로 암호화한다."""
    key, nonce, data = _key(token), secrets.token_bytes(16), text.encode()
    body = bytes(a ^ b for a, b in zip(data, _stream(key, nonce, len(data))))
    return base64.b64encode(nonce + body + hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]).decode()


def unseal(token, blob):
    raw = base64.b64decode(blob)
    key, nonce, body, tag = _key(token), raw[:16], raw[16:-16], raw[-16:]
    if not hmac.compare_digest(tag, hmac.new(key, nonce + body, hashlib.sha256).digest()[:16]):
        return None
    return bytes(a ^ b for a, b in zip(body, _stream(key, nonce, len(body)))).decode()


def chat_ids(token):
    if os.environ.get('TELEGRAM_CHAT_ID'):
        return [c.strip() for c in os.environ['TELEGRAM_CHAT_ID'].split(',') if c.strip()]
    if CHAT_FILE.exists():
        text = unseal(token, CHAT_FILE.read_text().strip())
        if text:
            return text.split(',')
    found = set()
    for update in telegram(token, 'getUpdates'):
        for key in ('message', 'my_chat_member'):
            chat = update.get(key, {}).get('chat', {})
            if chat.get('type') == 'private':
                found.add(str(chat['id']))
    if not found:
        raise SystemExit('봇 대화방을 찾지 못했습니다. 텔레그램에서 @lotto_pension_bot 에 /start 를 보내 주세요.')
    CHAT_FILE.write_text(seal(token, ','.join(sorted(found))) + '\n')
    return sorted(found)


def send(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    if not token:
        raise SystemExit('TELEGRAM_BOT_TOKEN이 없어 보내지 못했습니다.')
    for chat in chat_ids(token):
        telegram(token, 'sendMessage', chat_id=chat, text=text, disable_web_page_preview='true')


# ---------------------------------------------------------------- 실행

def main():
    parser = argparse.ArgumentParser(description='연금복권720+ 주간 자동 갱신')
    parser.add_argument('--no-fetch', action='store_true', help='새 회차를 받지 않는다')
    parser.add_argument('--no-send', action='store_true', help='텔레그램으로 보내지 않고 출력만')
    args = parser.parse_args()

    rows = read_rows()
    last_round, last_date = int(rows[-1]['회차']), date.fromisoformat(rows[-1]['추첨일'])
    new, errors = ([], []) if args.no_fetch else fetch_new(last_round)
    for error in errors:
        print('자료 받기 실패', error, file=sys.stderr)
    if new:
        write_rows(rows + new)
        print('추가:', ', '.join(f"{r['회차']}회 {r['1등 조']}조 {r['1등 번호']}" for r in new))

    draws = eng.load_draws(DATA)
    target = int(draws.rounds[-1]) + 1
    target_dir = RESULTS / f'round_{target}'
    if (target_dir / '텔레그램.txt').exists():
        overdue = datetime.now(KST).date() >= last_date + timedelta(days=8)
        if overdue and not args.no_send:   # 금요일 실행까지 새 회차가 없으면 한 번 알린다
            send(f'연금복권720+ {last_round + 1}회 결과를 아직 받지 못했습니다.\n' + '\n'.join(errors))
        print(f'{target}회 후보는 이미 보냈습니다. 새 회차 없음.')
        return

    if new or not (target_dir / '후보15.csv').exists():
        setting, mean_ranks, context, models, selected, scores, parts = eng.run(draws)
        eng.save(target_dir, context, setting, mean_ranks, models, selected, scores, parts)
    setting = read_setting(target_dir)
    previous_dir = RESULTS / f'round_{target - 1}'
    analysis = analyze(draws, read_setting(previous_dir) or setting, read_candidates(previous_dir))
    write_analysis(previous_dir, analysis)
    target_date = (draws.dates[-1] + timedelta(days=7)).isoformat()
    message = compose(analysis, target, target_date, setting, read_candidates(target_dir))
    print(message)
    if args.no_send:
        return
    send(message)
    (target_dir / '텔레그램.txt').write_text(message + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
