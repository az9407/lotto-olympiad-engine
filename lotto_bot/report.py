"""로또 결과마다 텔레그램으로 보내는 엑셀 보고서.

시트: 요약, 지난 후보 결과, 무엇이 문제였나, 무엇이 개선됐나, 다음 회차 후보, 공식 성적표, 새로 찾은 공식
수동 변경 기록은 state/changes.json 에 적어 두면 다음 보고서에 한 번 실리고 reported 로 표시됨
"""
import json
from pathlib import Path

import pandas as pd


def pending_changes(state_dir):
    p = Path(state_dir) / "changes.json"
    if not p.exists():
        return []
    return [c for c in json.loads(p.read_text(encoding="utf-8")) if not c.get("reported")]


def mark_reported(state_dir, changes):
    p = Path(state_dir) / "changes.json"
    if not p.exists() or not changes:
        return
    allc = json.loads(p.read_text(encoding="utf-8"))
    ids = {c["text"] for c in changes}
    for c in allc:
        if c["text"] in ids:
            c["reported"] = True
    p.write_text(json.dumps(allc, ensure_ascii=False, indent=1), encoding="utf-8")


def nums(c):
    return " ".join(f"{n:02d}" for n in c)


def build_excel(R, search_info, changes, path):
    a, dec, rep = R["analysis"], R["decision"], R["rule_report"]
    this = {k: v for k, v in R["this_all"].items() if v is not None}
    ranked_this = sorted(this.items(), key=lambda kv: kv[1])
    used = R["used_last"]
    used_pct = this.get(used)
    used_rank = next((i for i, (k, _) in enumerate(ranked_this, 1) if k == used), None)
    board = R["board"]
    champ_board_rank = next((i for i, b in enumerate(board, 1) if b["name"] == R["champion"]), None)

    summary = pd.DataFrame([
        ("회차", f"{R['N']}회 {R['date']}"),
        ("당첨번호", f"{nums(R['nums'])} + 보너스 {R['bonus']:02d}"),
        ("1등", f"{R['w1']}명, 1인당 {R['p1']:,}원"),
        ("지난 회차 후보", f"{R['prev_label']}, 최고 {max(r['match'] for r in R['prev_rows'])}개 일치"),
        ("지난 회차를 뽑은 방법", used),
        ("다음 회차", f"{R['T']}회 {R['TDATE']} 추첨"),
        ("다음 회차 챔피언", R["champion"]),
        ("챔피언 설명", R["champ_desc"]),
        ("후보 수", dec["n_methods"]),
    ], columns=["항목", "내용"])

    prev = pd.DataFrame([dict(순위=r["rank"], 번호=nums(r["numbers"]), 일치=r["match"], 등수=r["prize"] or "낙첨")
                         for r in R["prev_rows"]])

    prob = [("지난 방법의 성적", f"{used}이 실제 번호를 8,145,060개 중 상위 {used_pct:.1f}%에 둠"
             if used_pct is not None else f"{used} 기록 없음"),
            ("이번 회차 후보 중 순위", f"{len(ranked_this)}개 중 {used_rank}위" if used_rank else "-"),
            ("합계", f"{a['sum']}, 최근 50회 평균 {a['sum50']:.0f}, 전체 조합의 {a['sum_pct_low']:.0f}%가 이보다 작음"),
            ("홀짝·저번호", f"홀짝 {a['odd']}:{6-a['odd']}, 1~22 {a['low']}개"),
            ("AC·연번·끝수합", f"AC {a['ac']}, 연번 {a['consec']}쌍, 끝수합 {a['tail_sum']}"),
            ("직전회 이월·이웃", f"이월 {len(a['carry'])}개 {','.join(map(str, a['carry']))}, 이웃수 {a['neighbor']}개"),
            ("규칙 엔진 적중", f"{len(a['rules'])}개 중 {a['n_hit']}개, 엔진 순위 상위 {a['pct']:.1f}%"),
            ("인기회피·구성필터", f"인기회피 상위 {a['pop_top']:.0f}%, 최근 50회 구성 필터 {'통과' if a['in_profile'] else '밖'}")]
    miss = sorted([s for s, h in a["rules"] if not h], key=lambda s: -s["w"])[:8]
    for s in miss:
        prob.append(("빗나간 규칙", f"{s['cond']} → {s['event']}, 가중치 {s['w']:.2f}"))
    for k, v in ranked_this[:8]:
        prob.append(("이번 회차를 더 잘 맞힌 공식", f"{k} 상위 {v:.1f}%"))
    problem = pd.DataFrame(prob, columns=["항목", "내용"])

    imp = []
    if rep.get("removed"):
        imp.append(("규칙 엔진 설정", f"최근 성적 꼴찌 {rep['removed']} 제외"))
    if rep.get("added"):
        ad = rep["added"]
        imp.append(("규칙 엔진 설정", f"{ad['parent']}를 바꾼 새 설정 {ad['id']} 추가, 최근 6회 평균 상위 {ad['score']:.1f}%"))
    if dec["switched"]:
        imp.append(("챔피언", f"{dec['old']} → {dec['champion']} 교체, 같은 회차 비교 {abs(dec['gain']):.1f}%p 앞섬"))
    else:
        cl = dec.get("closest")
        imp.append(("챔피언", f"{dec['champion']} 유지, 최근 {dec['champ_n']}회 평균 상위 {dec['champ_mean']:.1f}%"
                    if dec["champ_mean"] is not None else f"{dec['champion']} 유지"))
        if cl:
            imp.append(("가장 가까운 도전자", f"{cl['name']}: {cl['n']}회 비교 평균 {abs(cl['gain']):.1f}%p 앞섬, "
                        f"표준오차 {abs(cl['t']):.1f}배, 교체에 필요한 기준 {cl['need']:.1f}배"))
    if champ_board_rank:
        imp.append(("챔피언의 최근 60회 순위", f"{len(board)}개 중 {champ_board_rank}위"))
    if search_info:
        imp.append(("새 공식 탐색", f"{search_info['minutes']:g}분 동안 {search_info['tested']:,}개 시험, 새 후보 "
                    f"{len(search_info['added'])}개 추가, 후보 풀 {search_info['pool']}개, 누적 {search_info['total']:,}개"))
        imp.append(("새 후보 규칙", "발견 다음 회차부터 10회 이상 쌓여야 챔피언 교체 경쟁에 들어감"))
    for c in changes:
        imp.append((f"코드 개선 {c.get('date', '')}", c["text"]))
    improve = pd.DataFrame(imp, columns=["항목", "내용"])

    picks = pd.DataFrame([dict(순위=c["rank"], 번호=nums(c["numbers"]), 점수=c["score"],
                               규칙=(f"{c['hits']}/{c['rules']}" if "hits" in c else "-")) for c in R["picks"]])

    boardf = pd.DataFrame([dict(순위=i, 후보=b["name"], 이번회차_상위퍼센트=None if b["this"] is None else round(b["this"], 2),
                                최근60회_평균=None if b["mean60"] is None else round(b["mean60"], 2),
                                회차수=b["n"], 발견회차=b["found"]) for i, b in enumerate(board, 1)])

    sheets = [("요약", summary), ("지난 후보 결과", prev), ("무엇이 문제였나", problem), ("무엇이 개선됐나", improve),
              ("다음 회차 후보", picks), ("공식 성적표", boardf)]
    if search_info and search_info["added"]:
        sheets.append(("새로 찾은 공식", pd.DataFrame([dict(이름=x["name"], 설명=x["desc"], 탐색구간_평균=x["fit"],
                                                       최근100회=x["recent100"]) for x in search_info["added"]])))
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, d in sheets:
            d.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.6), 110)
    return path
