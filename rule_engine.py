"""Generalized next-round data context and eligible active rule registry.

The empirical selection uses the latest validation block intentionally; results
describe a fitted composite scoring model rather than an independent forecast
test. All retained conditional rules must be active at the requested next draw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from combinatorial import enumerate_null, evaluate_spec, search_b
from lotto_analysis import read_data, history_features, make_conditions, make_events


def load_context(path, validation_size=250, target_date=None):
    """Read all historical rows and predict exactly the next unobserved round.

    The newest ``validation_size`` draws form validation; preceding draws form
    training. Thus the current 1248-row input reproduces rounds 1..998 and
    999..1248, and an appended draw advances both boundaries by one.
    """
    df = read_data(Path(path))
    if validation_size < 30 or len(df)-validation_size < 100:
        raise ValueError('Need at least 100 training and 30 validation draws')
    x, b, features, gap_num, gap_tail, future = history_features(df, target_date)
    train = np.arange(len(df)) < len(df)-validation_size
    valid = ~train
    next_date = pd.Timestamp(target_date) if target_date is not None else pd.Timestamp(df['추첨일'].iloc[-1])+pd.Timedelta(days=7)
    return {'df': df, 'x': x, 'b': b, 'features': features,
            'gap_num': gap_num, 'gap_tail': gap_tail, 'future': future,
            'train': train, 'valid': valid, 'target_round': future['round'],
            'target_date': next_date, 'source': str(Path(path).resolve())}


def _conditional_rules(context, null):
    train, valid = context['train'], context['valid']
    features, future = context['features'], context['future']
    conditions = make_conditions(features, train, future)
    events = make_events(context['x'], context['b'], features,
                         context['gap_num'], context['gap_tail'], future, null, train)
    active = [c for c in conditions if c['next_active']]
    cm = np.array([c['mask'] for c in active], dtype=np.int32)
    rows = []
    support_pairs = hit_pass_pairs = bonus_pairs = 0

    def activation_evidence(condition):
        detail = condition['detail']
        source, lag = detail['source'], detail['lag']
        if lag:
            value = float(features[source][-lag])
        elif source in future:
            value = int(future[source])
        elif source.startswith('absence_number_'):
            number = int(source.rsplit('_', 1)[1])
            value = 0 if number in context['x'][-1] else int(features[source][-1])+1
        elif source.startswith('absence_tail_'):
            tail = int(source.rsplit('_', 1)[1])
            value = 0 if tail in context['x'][-1] % 10 else int(features[source][-1])+1
        else:
            raise ValueError(f'Cannot audit activation source {source}')
        result = {**detail, 'observed_value': value,
                  'source_round': context['target_round']-lag if lag else context['target_round']}
        if detail['op'] == 'mod':
            result['observed_remainder'] = value % detail['mod']
        result['active'] = True
        return result
    for ei, event in enumerate(events):
        # An inactive event validity condition (e.g. overdue pool size) prevents
        # use in the target draw just as an inactive explicit antecedent does.
        if event['next_spec'] is None:
            continue
        nt = cm @ (train & event['valid']).astype(np.int32)
        nv = cm @ (valid & event['valid']).astype(np.int32)
        supported = (nt >= 100) & (nv >= 30)
        support_pairs += int(supported.sum())
        outcome = event['mask'].astype(np.int32)
        kt = cm @ (outcome * (train & event['valid']))
        kv = cm @ (outcome * (valid & event['valid']))
        passed = supported & (100*kt >= 55*nt) & (100*kv >= 55*nv)
        hit_pass_pairs += int(passed.sum())
        if event['bonus_event']:
            bonus_pairs += int(passed.sum())
            continue
        for ci in np.flatnonzero(passed):
            c = active[ci]
            tm, vm = c['mask'] & train & event['valid'], c['mask'] & valid & event['valid']
            p_train = float(event['probabilities'][tm].mean())
            p_valid = float(event['probabilities'][vm].mean())
            q = float(event.get('next_probability', event['baseline']))
            total_rate = float((kt[ci]+kv[ci])/(nt[ci]+nv[ci]))
            rows.append({'category': 'A', 'name': c['name']+' ⇒ '+event['name'],
                         'spec': event['next_spec'], 'q': q, 'baseline': q,
                         'train_k': int(kt[ci]), 'train_n': int(nt[ci]),
                         'train_rate': float(kt[ci]/nt[ci]),
                         'valid_k': int(kv[ci]), 'valid_n': int(nv[ci]),
                         'valid_rate': float(kv[ci]/nv[ci]),
                         'total_k': int(kt[ci]+kv[ci]), 'total_n': int(nt[ci]+nv[ci]),
                         'total_rate': total_rate, 'lift': total_rate/q,
                         'condition_active': True, 'condition_detail': c['detail'],
                         'activation_evidence': activation_evidence(c),
                         'condition_name': c['name'], 'event_name': event['name'],
                         'history_mask': event['mask'].copy(),
                         'history_valid': event['valid'].copy(),
                         'condition_history_mask': c['mask'].copy(),
                         'baseline_train_conditional': p_train, 'baseline_valid_conditional': p_valid,
                         'event_definition': event['spec'], 'condition_index': int(ci), 'event_index': ei})
    diagnostics = {'condition_definitions': len(conditions), 'active_condition_definitions': len(active),
                   'event_definitions': len(events), 'active_supported_condition_event_pairs': support_pairs,
                   'active_both_hit_rate_pass_pairs': hit_pass_pairs,
                   'bonus_only_pairs_excluded_from_six_number_scoring': bonus_pairs,
                   'eligible_a': len(rows)}
    return rows, diagnostics


def discover_rules(context, cache_dir, b_coverage_limit=0.5):
    """Return every eligible A/B record before model correlation discounting."""
    cache_dir = Path(cache_dir)
    null = enumerate_null(cache_dir)
    a, diagnostics = _conditional_rules(context, null)
    raw_b, null = search_b(context['x'], context['train'], cache_dir, coverage_limit=b_coverage_limit)
    b = []
    for r in raw_b:
        b.append({**r, 'q': r['baseline'], 'condition_active': True,
                  'condition_detail': {'kind': 'unconditional_internal_filter'},
                  'history_mask': evaluate_spec(context['features'], r['spec']),
                  'history_valid': np.ones(len(context['x']), bool),
                  'condition_history_mask': np.ones(len(context['x']), bool)})
    diagnostics.update({'b_hypotheses': int(null['searched_count']), 'eligible_b': len(b),
                        'b_coverage_limit': b_coverage_limit,
                        'train_first_round': int(context['df']['회차'][context['train']].iloc[0]),
                        'train_last_round': int(context['df']['회차'][context['train']].iloc[-1]),
                        'valid_first_round': int(context['df']['회차'][context['valid']].iloc[0]),
                        'valid_last_round': int(context['df']['회차'][context['valid']].iloc[-1]),
                        'target_round': context['target_round'],
                        'target_date': str(context['target_date'].date())})
    return {'eligible_a': a, 'eligible_b': b, 'diagnostics': diagnostics}
