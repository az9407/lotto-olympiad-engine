"""Next-draw composite log-odds ranking, constrained selection and GitHub push.

Run ``python predict_top15.py`` after appending the newest draw to the workbook.
Scores rank this fitted model; they are not calibrated joint winning odds.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
import time
from pathlib import Path

# The desktop's default Python may have no analysis libraries. Reuse the
# provisioned runtime without installing packages or modifying global settings.
if __name__ == '__main__':
    import importlib.util
    if any(importlib.util.find_spec(name) is None for name in ('numpy', 'pandas', 'openpyxl')):
        bundled = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
        if bundled.is_file() and Path(sys.executable).resolve() != bundled.resolve():
            import subprocess
            raise SystemExit(subprocess.call([str(bundled), str(Path(__file__).resolve()), *sys.argv[1:]]))

import numpy as np
import pandas as pd

from combinatorial import TOTAL, combination_chunks, static_features
from generate_candidates import evaluate_next
from number_model import fit_number_model
from rule_engine import load_context, discover_rules


def raw_log_odds_weight(p, q, n):
    """Requested validation-vs-reference log odds, clipped only at endpoints."""
    if not 0 < q < 1:
        raise ValueError('Theoretical coverage must be strictly between zero and one')
    epsilon = .5/(n+1)
    clipped = min(1-epsilon, max(epsilon, p)) if p in (0, 1) else p
    return math.log(clipped/(1-clipped))-math.log(q/(1-q))


def build_scoring_rules(registry, train, limit_a=100, training_x=None):
    """Discount resolved target predicates on training X; include eligible B rules.

    A conclusions resolving to the same future event contribute once. All
    eligible A antecedents are retained separately in the audit CSV. Every B
    record is retained, with zero weight when mathematically duplicated.
    """
    if training_x is None:
        raise ValueError('Training combinations are required for target-predicate correlation')
    training_features = static_features(training_x)
    def ordered(rows):
        prepared = []
        for r in rows:
            item = dict(r)
            item['raw_weight'] = raw_log_odds_weight(item['valid_rate'], item['q'], item['valid_n'])
            item['scoring_correlation_mask'] = evaluate_next(training_x, training_features, item['spec'])
            prepared.append(item)
        return sorted(prepared, key=lambda r: (-r['raw_weight'], -r['valid_n'], -r['train_n'], r['name']))
    selected = []
    definitions = set()
    chosen_a = 0
    diagnostics = {'duplicate_a_conclusions_removed': 0}
    # A is selected first to guarantee the requested minimum conditional rules.
    # B is appended in its entirety, with the same correlation discount.
    for category, source in [('A', ordered(registry['eligible_a'])), ('B', ordered(registry['eligible_b']))]:
        for r in source:
            if category == 'A' and chosen_a >= limit_a:
                break
            signature = json.dumps(r['spec'], sort_keys=True)
            if category == 'A' and signature in definitions:
                diagnostics['duplicate_a_conclusions_removed'] += 1
                continue
            rho = 0.0
            for old in selected:
                left, right = r['scoring_correlation_mask'], old['scoring_correlation_mask']
                if len(left) and left.std() > 0 and right.std() > 0:
                    value = abs(float(np.corrcoef(left.astype(float), right.astype(float))[0, 1]))
                    rho = max(rho, min(1.0, value))
            if signature in definitions:
                rho = 1.0
            discount = 1-rho if rho >= .5 else 1.0
            if discount < 1e-12: discount = 0.0
            if category == 'A' and discount == 0:
                diagnostics['duplicate_a_conclusions_removed'] += 1
                continue
            r.update({'max_abs_correlation': rho, 'correlation_discount': discount,
                      'weight': r['raw_weight']*discount, 'rule_id': f'{category}{len(selected)+1:03d}'})
            selected.append(r)
            definitions.add(signature)
            chosen_a += category == 'A'
    if chosen_a < 30:
        raise ValueError(f'Only {chosen_a} supported distinct active A rules; the minimum 30 was not met')
    diagnostics.update({'selected_a': chosen_a, 'included_b': sum(r['category']=='B' for r in selected),
                        'positive_weight_rules': sum(r['weight'] > 0 for r in selected),
                        'correlation_discounted_rules': sum(r['max_abs_correlation'] >= .5 for r in selected),
                        'correlation_data': 'resolved target-round predicates evaluated on identical training main-number combinations; threshold abs(rho)>=0.5'})
    return selected, diagnostics


def score_batch(x, rules, probabilities, number_weight=.05):
    features = static_features(x)
    rule_score = np.zeros(len(x), float)
    matched = np.zeros(len(x), np.uint16)
    for r in rules:
        hit = evaluate_next(x, features, r['spec'])
        matched += hit.astype(np.uint16)
        rule_score += r['weight']*hit
    number_score = number_weight*np.log(probabilities[x-1]).sum(axis=1)
    return rule_score, number_score, matched


def score_universe(rules, probabilities, number_weight=.05):
    x = np.empty((TOTAL, 6), np.uint8)
    rule_score = np.empty(TOTAL, float)
    number_score = np.empty(TOTAL, float)
    matched = np.empty(TOTAL, np.uint16)
    cursor = 0
    for batch in combination_chunks():
        end = cursor+len(batch)
        x[cursor:end] = batch
        rule_score[cursor:end], number_score[cursor:end], matched[cursor:end] = score_batch(
            batch, rules, probabilities, number_weight)
        cursor = end
        if cursor % 2_000_000 == 0:
            print(f'Scored {cursor:,}/{TOTAL:,}', flush=True)
    assert cursor == TOTAL and np.isfinite(rule_score+number_score).all()
    return x, rule_score, number_score, matched


def choose_hard_core(x, rules, requested=3, minimum_pool=10_000):
    """Choose strongest feasible conjuncts, never silently return the full space."""
    if requested < 1:
        raise ValueError('At least one genuine hard-filter rule is required')
    candidates = sorted([r for r in rules if r['weight'] > 0], key=lambda r:(-r['weight'], r['name']))
    keep = np.ones(len(x), bool)
    chosen, skipped = [], []
    for r in candidates:
        proposed = keep.copy()
        for start in range(0, len(x), 100_000):
            end = min(start+100_000, len(x))
            if not keep[start:end].any(): continue
            part = x[start:end]
            proposed[start:end] &= evaluate_next(part, static_features(part), r['spec'])
        n = int(proposed.sum())
        reason = None
        if n == int(keep.sum()): reason = 'no additional reduction'
        elif n < minimum_pool: reason = f'pool below {minimum_pool}'
        else:
            pool = x[proposed]
            numbers = np.bincount(pool.ravel(), minlength=46)[1:]
            if np.count_nonzero(numbers) < 23: reason = 'fewer than 23 usable numbers for exposure cap4'
            elif np.any(numbers == n): reason = 'a mandatory number would violate cap4 across 15 rows'
        if reason:
            skipped.append({'name': r['name'], 'reason': reason, 'proposed_count': n})
            continue
        keep = proposed
        chosen.append(r)
        if len(chosen) == requested: break
    if len(chosen) != requested or not (15 <= keep.sum() < len(x)):
        raise RuntimeError('Could not construct the requested feasible hard-filter intersection')
    return keep, chosen, skipped


def strict_order(scores):
    """Sort full floating scores descending, exact ties by number lexicography.

    combinations() enumerates tuples lexicographically. No random extraction,
    jitter or fabricated probability is introduced to break exact equal scores.
    """
    return np.argsort(-scores, kind='stable')


def diversity_select(x, order, hard_mask, count=15, max_shared=3, max_exposure=4):
    selected = []
    exposure = np.zeros(46, int)
    positions = []
    examined = 0
    for position, index in enumerate(order, 1):
        if not hard_mask[index]: continue
        examined += 1
        row = x[index]
        if np.any(exposure[row] >= max_exposure): continue
        if any(np.isin(row, x[old]).sum() > max_shared for old in selected): continue
        selected.append(int(index))
        positions.append(position)
        exposure[row] += 1
        if len(selected) == count:
            return selected, positions, {'examined_hard_pass_rows': examined, 'number_exposure': exposure[1:].tolist()}
    raise RuntimeError(f'Only {len(selected)} candidates meet hard filters and exact diversity constraints; none were relaxed')


def tie_audit(scores, order, selected):
    sorted_scores = scores[order]
    rounded = np.round(sorted_scores, 8)
    chosen = scores[selected]
    return {'exact_equal_adjacent_pairs_all_combinations': int(np.count_nonzero(np.diff(sorted_scores) == 0)),
            'display_8dp_equal_adjacent_pairs_all_combinations': int(np.count_nonzero(np.diff(rounded) == 0)),
            'selected_exact_ties': len(chosen)-len(np.unique(chosen)),
            'selected_display_8dp_ties': len(chosen)-len(np.unique(np.round(chosen, 8))),
            'policy': 'full float64 S descending; exact ties use ascending number tuple; no random choice',
            'interpretation': 'display rounding can create equal 8dp values; no probability advantage is inferred from a tie-break'}


def _clean_rule(r):
    return {k:v for k,v in r.items() if not isinstance(v, np.ndarray)}


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def rule_table(rules, hard_ids=()):
    records = []
    for r in rules:
        records.append({'규칙ID': r['rule_id'], '분류': r['category'], '규칙 수식': r['name'],
                        '목표회차 활성 여부': bool(r['condition_active']),
                        '활성 근거': json.dumps(r.get('activation_evidence', {'active': True, 'kind': 'unconditional B'}),
                                              ensure_ascii=False, sort_keys=True),
                        '학습 적중률': r['train_rate'], '학습 표본 수': r['train_n'],
                        '검증 적중률': r['valid_rate'], '검증 표본 수': r['valid_n'],
                        '이론확률 q': r['q'], '전체 Lift': r['lift'],
                        '원래 가중치 w_r': r['raw_weight'], '최대 절대상관': r['max_abs_correlation'],
                        '상관 할인계수': r['correlation_discount'], '적용 가중치': r['weight'],
                        '하드 필터': r['rule_id'] in hard_ids})
    return pd.DataFrame(records)


def save_results(out, context, rules, registry, model, x, sr, sn, matched, selected, ranks,
                 hard_mask, hard_rules, skipped, selection_diag, rule_diag, ties, number_weight, runtime):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (i, overall) in enumerate(zip(selected, ranks), 1):
        records.append({'선정 순위': rank, '전체 종합점수 순위': overall,
                        '후보 번호 6개': ' '.join(f'{int(n):02d}' for n in x[i]),
                        **{f'번호{j+1}': int(n) for j,n in enumerate(x[i])},
                        '종합 확률 점수 S(X)': float(sr[i]+sn[i]),
                        '규칙 점수 S_rule': float(sr[i]), '개별 번호 점수 S_num': float(sn[i]),
                        '충족한 활성 규칙 비율': int(matched[i])/len(rules),
                        '충족 규칙 수': int(matched[i]), '사용 규칙 수': len(rules)})
    candidates = pd.DataFrame(records)
    candidates.to_csv(out/'top15.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
    hard_ids = {r['rule_id'] for r in hard_rules}
    rt = rule_table(rules, hard_ids)
    rt.to_csv(out/'active_rules.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    eligible = []
    for r in registry['eligible_a']+registry['eligible_b']:
        eligible.append({k:r[k] for k in ('category','name','q','train_k','train_n','train_rate',
                         'valid_k','valid_n','valid_rate','total_rate','lift','condition_active')})
    pd.DataFrame(eligible).to_csv(out/'eligible_active_rules.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(model['table']).to_csv(out/'number_probabilities.csv', index=False, encoding='utf-8-sig', float_format='%.12f')
    summary = {'target_round': context['target_round'], 'target_date': str(context['target_date'].date()),
               'source_sha256': hashlib.sha256(Path(context['source']).read_bytes()).hexdigest(),
               'latest_main': context['x'][-1].tolist(), 'latest_bonus': int(context['b'][-1]),
               'registry': registry['diagnostics'], 'rule_selection': rule_diag,
               'score_formula': 'S=sum_r [logit(p_valid_r)-logit(q_r)]*correlation_discount_r*I_r(X) + lambda*sum_n_in_X log(p_number_n)',
               'number_weight_lambda': number_weight, 'rules': [_clean_rule(r) for r in rules],
               'hard_rule_ids': sorted(hard_ids), 'hard_filter_skipped': skipped,
               'scored_combinations': TOTAL, 'hard_pass_count': int(hard_mask.sum()),
               'hard_coverage': float(hard_mask.mean()), 'ties': ties,
               'diversity': selection_diag, 'max_pair_overlap': 3, 'max_number_exposure': 4,
               'number_model': model['diagnostics'], 'runtime_seconds': runtime,
               'calibration': 'S is an empirical composite ranking score, not a calibrated joint winning probability',
               'validation_use': 'latest250 validation outcomes are deliberately used for rule eligibility and rule weights; retrospective model selection',
               'candidates': records}
    (out/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
    display = candidates[['선정 순위','전체 종합점수 순위','후보 번호 6개','종합 확률 점수 S(X)',
                          '규칙 점수 S_rule','개별 번호 점수 S_num','충족한 활성 규칙 비율']].copy()
    display['충족한 활성 규칙 비율'] = display['충족한 활성 규칙 비율'].map(lambda v:f'{v:.2%}')
    rdisplay = rt.copy()
    for c in ('학습 적중률','검증 적중률','이론확률 q'):
        rdisplay[c] = rdisplay[c].map(lambda v:f'{v:.4%}')
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{context['target_round']}회 종합점수 상위15</title>
    <style>body{{font:14px/1.6 'Malgun Gothic',sans-serif;max-width:1800px;margin:36px auto;padding:0 20px;color:#17253b}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #dce1e8;text-align:right}}
    th{{background:#edf1f6}}td:nth-child(3){{white-space:nowrap}}.scroll{{overflow:auto}}pre{{background:#f4f6f9;padding:18px;overflow:auto}}</style>
    <h1>{context['target_round']}회 · {context['target_date'].date()} · 종합점수 후보 15개</h1>
    <p>직전 본번호 {', '.join(map(str, context['x'][-1]))}, 보너스 {context['b'][-1]}를 적용했습니다.
    모든 A 규칙은 현재 활성=True, 학습 표본≥100, 검증 표본≥30, 양 구간 적중률≥55%입니다.
    B는 전체 경험적 적중률/정확 이론통과율&gt;1.05 및 검증 적중률≥55%를 사용합니다.</p>
    <p>전체 {TOTAL:,}개를 점수화하고 핵심 {len(hard_rules)}개 규칙의 교집합으로 {int(hard_mask.sum()):,}개를 남겼습니다.
    A {rule_diag['selected_a']}개와 B {rule_diag['included_b']}개를 채점했습니다.
    두 후보 간 공통 번호≤3개, 번호별 전체 출현≤4회를 엄격히 적용했습니다.</p>
    <p>S_rule=Σ w_r·discount_r·I_r(X), w_r=logit(검증 적중률)−logit(이론확률),
    S_num={number_weight}·Σ ln P̂(번호), S=S_rule+S_num. |ρ|≥0.5이면 가중치에 (1−|ρ|)를 곱했습니다.
    검증구간을 규칙 선택과 가중치에 사용한 경험적 모델입니다. S는 실제 1등 당첨확률로 보정된 값이 아닙니다.</p>
    <p>전체 점수 순위와 다양성 제약 아래의 선정 순위를 함께 표시했습니다. 임의 추출이나 무작위 잡음을 사용하지 않습니다.
    실수 S를 내림차순으로 정렬하고 정확한 동점은 번호 사전순으로만 처리합니다.
    소수점 8자리 반올림 동점 감사 결과는 아래에 공개합니다.</p><div class="scroll">'''
    content += display.to_html(index=False, float_format=lambda v:f'{v:.8f}')+'</div>'
    content += '<h2>실제 적용 활성 규칙 명세</h2><div class="scroll">'+rdisplay.to_html(index=False, float_format=lambda v:f'{v:.8f}')+'</div>'
    content += '<h2>핵심 하드 필터</h2><ul>'+''.join('<li>'+html.escape(r['name'])+'</li>' for r in hard_rules)+'</ul>'
    content += '<h2>동점 감사</h2><pre>'+html.escape(json.dumps(ties, ensure_ascii=False, indent=2))+'</pre>'
    metrics = {k:model['diagnostics'][k] for k in ('validation','validation_constant_baseline','forecast_refit')}
    content += '<h2>번호 모델 검증 및 학습 방식</h2><pre>'+html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))+'</pre>'
    content += '<p>엑셀에 최신 행을 추가하면 다음 회차와 날짜를 자동 산출합니다. 최근250회는 검증, 이전 회차는 학습입니다. '
    content += '지지 표본이나 활성 규칙이 부족하거나 제약하에서 15개를 채우지 못하면 명시적인 오류를 냅니다. '
    content += '전체 조합 반환, 조건 무시, 무작위 대체로 성공을 가장하지 않습니다.</p></html>'
    (out/'report.html').write_text(content, encoding='utf-8')
    return summary


def find_input(specified=None):
    if specified: return Path(specified)
    candidates = [*Path('.').glob('*.xlsx'), *Path('.codex-attachments').rglob('*.xlsx')]
    if not candidates: raise FileNotFoundError('Place the source XLSX here or pass --input PATH')
    return max(candidates, key=lambda p:p.stat().st_mtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('results'))
    parser.add_argument('--validation-size', type=int, default=250)
    parser.add_argument('--active-rules', type=int, default=100, choices=range(30,101))
    parser.add_argument('--hard-core-count', type=int, default=3)
    parser.add_argument('--number-weight', type=float, default=.05)
    parser.add_argument('--target-date')
    parser.add_argument('--no-push', action='store_true', help='Compute local artifacts only; default also commits and pushes')
    args = parser.parse_args(argv)
    if not (0 < args.number_weight <= 1): parser.error('--number-weight must be in (0,1]')
    start = time.perf_counter()
    context = load_context(find_input(args.input), args.validation_size, args.target_date)
    out = args.output_root/f"round_{context['target_round']}"
    registry = discover_rules(context, Path('.cache'))
    rules, rule_diag = build_scoring_rules(registry, context['train'], args.active_rules,
                                         training_x=context['x'][context['train']])
    model = fit_number_model(context['x'], context['b'], context['train'], context['valid'])
    print(f"Target {context['target_round']} {context['target_date'].date()}; active eligible A={len(registry['eligible_a'])}, B={len(registry['eligible_b'])}; selected A={rule_diag['selected_a']}, B={rule_diag['included_b']}", flush=True)
    x, sr, sn, matched = score_universe(rules, model['probabilities'], args.number_weight)
    hard, core, skipped = choose_hard_core(x, rules, args.hard_core_count)
    total_score = sr+sn
    order = strict_order(total_score)
    selected, ranks, selection_diag = diversity_select(x, order, hard)
    ties = tie_audit(total_score, order, selected)
    summary = save_results(out, context, rules, registry, model, x, sr, sn, matched,
                           selected, ranks, hard, core, skipped, selection_diag, rule_diag,
                           ties, args.number_weight, round(time.perf_counter()-start, 2))
    print(pd.read_csv(out/'top15.csv').to_string(index=False, float_format=lambda v:f'{v:.8f}'), flush=True)
    print('TIES', json.dumps(ties, ensure_ascii=False), flush=True)
    print(f'Results: {out.resolve()}', flush=True)
    if not args.no_push:
        from publish_github import publish, PublishError
        try:
            result = publish(message=f"{context['target_round']}회 활성 규칙과 연속 종합점수 후보를 갱신")
        except (PublishError, OSError) as error:
            failure = {'status': 'failed', 'error': str(error), 'local_results_saved': str(out.resolve()),
                       'proposed_repository': 'https://github.com/az9407/lotto-olympiad-engine',
                       'remote_upload_verified': False}
            (out/'publish_status.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
            print('GITHUB', json.dumps(failure, ensure_ascii=False), flush=True)
            raise SystemExit(2)
        else:
            (out/'publish_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print('GITHUB', json.dumps(result, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    main()
