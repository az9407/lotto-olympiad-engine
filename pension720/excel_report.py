"""매주 텔레그램으로 보내는 엑셀: 무엇이 문제였는지, 무엇을 개선했는지, 다음 회차 후보."""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

FONT = Font(name='맑은 고딕', size=10)
HEAD = Font(name='맑은 고딕', size=10, bold=True, color='FFFFFF')
TITLE = Font(name='맑은 고딕', size=12, bold=True)
FILL = PatternFill('solid', fgColor='305496')
LINE = Side(style='thin', color='BFBFBF')
BOX = Border(left=LINE, right=LINE, top=LINE, bottom=LINE)
WRAP = Alignment(vertical='top', wrap_text=True)


def _sheet(book, title, header, rows, widths, first=False):
    ws = book.active if first else book.create_sheet()
    ws.title = title
    ws.append(list(header))
    for c in ws[1]:
        c.font, c.fill, c.border, c.alignment = HEAD, FILL, BOX, Alignment(horizontal='center', vertical='center')
    for row in rows:
        ws.append(list(row))
    for line in ws.iter_rows(min_row=2):
        for c in line:
            c.font, c.border, c.alignment = FONT, BOX, WRAP
            if isinstance(c.value, str) and c.value.isdigit() and len(c.value) == 6:
                c.number_format = '@'
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.freeze_panes = 'A2'
    return ws


def _pct(rank, total):
    return 100 * rank / total


def build(path: Path, a, target, target_date, formula_name, candidates, learning, table, live, scoreboard, notes,
          next_formula=''):
    """a: weekly.analyze 결과. candidates: (순위, 조, 6자리, 점수). learning: 학습 요약 줄. table: learn 비교표 행."""
    book = Workbook()
    rank_pct = _pct(a['rank'], 5_000_000)
    hits = [r for r in a['main_rules'] if r[2]]
    misses = sorted((r for r in a['main_rules'] if not r[2]), key=lambda r: -r[1])
    prev = a['previous']
    won = [(rk, g, n, p) for rk, g, n, p in prev if p]
    side = '높게' if rank_pct <= 30 else '낮게' if rank_pct >= 70 else '중간 정도로'

    summary = [
        ('회차', f"{a['round']}회 · {a['date']}"),
        ('1등', f"{a['group']}조 {a['main']}"),
        ('보너스', f"각조 {a['bonus']}"),
        ('지난주 공식', formula_name(a['formula'], a['setting'])),
        ('지난주 공식이 본 실제 1등', f"500만 중 {a['rank']:,}위 · 상위 {rank_pct:.1f}%"),
        ('지난주 보낸 10개', ('당첨 없음' if not won else ', '.join(f"{rk}번 {'·'.join(p)}" for rk, _, _, p in won))
         if prev else '기록 없음'),
        ('실전 누적', f'{live[0]}주 평균 상위 {live[1]:.1f}%' if live else '이번 주부터 기록'),
        ('다음 회차', f'{target}회 · {target_date}'),
        ('다음 회차 공식', next_formula),
    ]
    _sheet(book, '요약', ('항목', '내용'), summary, (24, 90), first=True)

    problems = [
        ('실제 1등 위치', f"지난주 공식은 이번 1등을 500만 중 상위 {rank_pct:.1f}%로 {side} 봤습니다."),
        ('빗나간 큰 규칙', '\n'.join(f'{f} · 가중치 {w:.3f}' for f, w, _ in misses[:5]) or '없음'),
        ('맞은 큰 규칙', '\n'.join(f'{f} · 가중치 {w:.3f}' for f, w, _ in hits[:5]) or '없음'),
        ('엔진 1등 규칙', f"{len(a['main_rules'])}개 중 {len(hits)}개 충족"),
        ('보너스 번호', f"보너스 규칙 {len(a['bonus_rules'])}개 중 {sum(r[2] for r in a['bonus_rules'])}개 충족 · "
                    f"100만 중 상위 {_pct(a['bonus_rank'], 1_000_000):.1f}%"),
        ('1등 조', f"{a['group']}조는 {a['group_streak']} 1등 조"),
        ('번호 특징', a['summary']),
    ]
    for rk, g, n, p in prev:
        problems.append((f'지난주 {rk}번 {g}조 {n}', '·'.join(p) if p else '낙첨'))
    problems.append(('판단', '한 회차 결과만으로는 공식의 좋고 나쁨을 가를 수 없어, 개선은 최근 60회 전체 성적으로만 판단합니다.'))
    _sheet(book, '문제점', ('항목', '내용'), problems, (28, 90))

    improved = [('이번 주 학습', line.removeprefix('· ').strip()) for line in learning]
    improved += [('자료', note) for note in notes]
    _sheet(book, '개선', ('항목', '내용'), improved, (16, 100))

    _sheet(book, f'{target}회 후보', ('순위', '조', '6자리', '종합점수'),
           [(int(rk), f'{g}조', n, float(s)) for rk, g, n, s in candidates], (8, 8, 12, 16))

    _sheet(book, '학습 상세', ('구분', '방법', '최근 60회 평균 상위 %', '지금 것과 차이 %p', '표준오차 %p', '교체 기준'),
           [(k, n, round(m, 2), round(d, 2), round(se, 2), '통과' if ok else '') for k, n, m, d, se, ok in table],
           (10, 70, 16, 16, 12, 10))

    rules = ([('1등', f, round(w, 4), 'O' if ok else 'X') for f, w, ok in a['main_rules']]
             + [('보너스', f, round(w, 4), 'O' if ok else 'X') for f, w, ok in a['bonus_rules']])
    _sheet(book, '규칙 충족', ('구분', '규칙', '가중치', '충족'), rules, (8, 80, 10, 8))

    if scoreboard:
        _sheet(book, '실전 성적', tuple(scoreboard[0].keys()), [tuple(r.values()) for r in scoreboard], (8, 60, 12, 16, 18))
    book.save(path)
    return path
