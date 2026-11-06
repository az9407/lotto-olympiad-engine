"""Reusable chronological features, condition definitions and event definitions.

Use predict_top15.py for the current next-round CLI.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from combinatorial import static_features, baseline_events

MODS = (2, 3, 4, 5, 7, 9, 10, 11)
KS = (3, 5, 7, 10, 15, 20, 30)
TOTAL = math.comb(45, 6)


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name='로또 당첨번호')
    df = df.sort_values('회차').reset_index(drop=True)
    expected = ['회차', '추첨일', *[f'번호{i}' for i in range(1, 7)], '보너스',
                '1등 당첨자수', '1등 1인당 당첨금', '2등 당첨자수', '2등 1인당 당첨금']
    assert list(df.columns) == expected, list(df.columns)
    assert len(df) >= 104 and not df.isna().any().any()
    assert np.array_equal(df['회차'], np.arange(1, len(df)+1))
    x = df[[f'번호{i}' for i in range(1, 7)]].to_numpy(int)
    b = df['보너스'].to_numpy(int)
    assert np.all((x >= 1) & (x <= 45)) and np.all(np.diff(x, axis=1) > 0)
    assert np.all((b >= 1) & (b <= 45)) and not np.any(x == b[:, None])
    assert df['추첨일'].iloc[0] == pd.Timestamp('2002-12-07')
    assert (df['추첨일'].diff().iloc[1:] == pd.Timedelta(days=7)).all()
    return df


def digit_sum(value: int) -> int:
    return sum(map(int, str(int(value))))


def context_arrays(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df['추첨일'])
    ds = np.array([digit_sum(int(d.strftime('%Y%m%d'))) for d in dates])
    return {'round': df['회차'].to_numpy(int), 'year': dates.dt.year.to_numpy(int),
            'month': dates.dt.month.to_numpy(int), 'day': dates.dt.day.to_numpy(int),
            'date_digit_sum': ds, 'date_digital_root': 1 + (ds - 1) % 9}


def history_features(df: pd.DataFrame, target_date=None):
    x = df[[f'번호{i}' for i in range(1, 7)]].to_numpy(int)
    b = df['보너스'].to_numpy(int)
    f = dict(static_features(x))
    f.update(context_arrays(df))
    f['bonus'] = b
    f['mean'] = x.mean(axis=1)
    f['variance'] = x.var(axis=1, ddof=0)
    f['even_count'] = 6 - f['odd_count']
    f['high_count'] = 6 - f['low_count']
    f['has_consecutive'] = (f['consecutive_count'] > 0).astype(int)
    f['sum_with_bonus'] = f['sum'] + b
    f['bonus_min_distance'] = np.abs(x-b[:, None]).min(axis=1)
    f['bonus_gcd_pairs'] = (np.gcd(x, b[:, None]) > 1).sum(axis=1)
    f['bonus_below_x3'] = (b < x[:, 2]).astype(int)
    f['bonus_above_x4'] = (b > x[:, 3]).astype(int)
    for col in [*[f'x{i}' for i in range(1, 7)], 'sum', 'bonus']:
        for m in MODS:
            f[f'{col}_mod{m}'] = f[col] % m
    for key in ('round', 'year', 'month', 'day', 'date_digit_sum', 'date_digital_root'):
        for m in MODS:
            f[f'{key}_matching_residue_count_mod{m}'] = (x % m == (f[key] % m)[:, None]).sum(axis=1)
            f[f'sum_minus_{key}_mod{m}'] = (f['sum'] - f[key]) % m
    f['first_prize_rollover'] = (df['1등 당첨자수'].to_numpy() == 0).astype(int)
    n = len(x)
    for lag in (1, 2, 3):
        for key in ('carry_main', 'carry_bonus', 'carry_main_bonus', 'bonus_to_main',
                    'bonus_to_bonus', 'main_to_bonus', 'neighbor_count'):
            vals = np.full(n, np.nan)
            for t in range(lag, n):
                old = set(x[t-lag])
                neighbors = {v + d for v in old for d in (-1, 1) if 1 <= v+d <= 45}
                # An old number may also be a neighbor of another old number; retain it.
                pool = old | {int(b[t-lag])}
                vals[t] = {'carry_main': len(set(x[t]) & old),
                           'carry_bonus': int(b[t-lag] in x[t]),
                           'carry_main_bonus': len(set(x[t]) & pool),
                           'bonus_to_main': int(b[t-lag] in x[t]),
                           'bonus_to_bonus': int(b[t] == b[t-lag]),
                           'main_to_bonus': int(b[t] in old),
                           'neighbor_count': len(set(x[t]) & neighbors)}[key]
            f[f'{key}_lag{lag}'] = vals
        for key in ('sum', 'odd_count', 'low_count'):
            vals = np.full(n, np.nan)
            vals[lag:] = f[key][lag:] - f[key][:-lag]
            f[f'{key}_change_lag{lag}'] = vals
            center = {'sum': 138, 'odd_count': 6*23/45, 'low_count': 6*22/45}[key]
            nearer = np.full(n, np.nan)
            nearer[lag:] = (np.abs(f[key][lag:]-center) < np.abs(f[key][:-lag]-center)).astype(int)
            f[f'{key}_nearer_center_lag{lag}'] = nearer
            opposite = np.full(n, np.nan)
            opposite[lag:] = ((f[key][lag:]-center)*(f[key][:-lag]-center) < 0).astype(int)
            f[f'{key}_opposite_side_lag{lag}'] = opposite
    gap_num = np.zeros((n, 45), int)
    gap_tail = np.zeros((n, 10), int)
    last_num = np.full(45, -1, int)
    last_tail = np.full(10, -1, int)
    for t, row in enumerate(x):
        gap_num[t] = t - last_num - 1
        gap_tail[t] = t - last_tail - 1
        last_num[row - 1] = t
        last_tail[np.unique(row % 10)] = t
    for j in range(45):
        f[f'absence_number_{j+1}'] = gap_num[:, j]
    for j in range(10):
        f[f'absence_tail_{j}'] = gap_tail[:, j]
    for k in KS:
        f[f'overdue_number_count_K{k}'] = (gap_num >= k).sum(axis=1)
        f[f'overdue_tail_count_K{k}'] = (gap_tail >= k).sum(axis=1)
    for col in ('1등 당첨자수', '1등 1인당 당첨금', '2등 당첨자수', '2등 1인당 당첨금'):
        f[col] = df[col].to_numpy(int)
    next_date = pd.Timestamp(target_date) if target_date is not None else pd.Timestamp(df['추첨일'].iloc[-1])+pd.Timedelta(days=7)
    if next_date <= pd.Timestamp(df['추첨일'].iloc[-1]):
        raise ValueError('Target date must be after the last observed draw')
    future = {'round': int(df['회차'].iloc[-1])+1, 'year': next_date.year, 'month': next_date.month,
              'day': next_date.day, 'date_digit_sum': digit_sum(int(next_date.strftime('%Y%m%d')))}
    future['date_digital_root'] = 1 + (future['date_digit_sum'] - 1) % 9
    return x, b, f, gap_num, gap_tail, future


def make_conditions(f: dict, train: np.ndarray, future: dict) -> list[dict]:
    """Create conditions solely from pre-draw information; quantiles fit on training."""
    n = len(train)
    conditions = []
    signatures = set()

    def add(name, mask, next_active, detail):
        mask = np.asarray(mask, bool)
        # A training hit rate must have a nonempty denominator. The user's
        # whole-sample support>=100 gate is checked without consulting E outcomes.
        if not (mask & train).any():
            return
        # Deduplicate mathematical definitions only. Neither holdout masks nor
        # the next-round activation may change the hypothesis registry.
        signature = json.dumps(detail, sort_keys=True, ensure_ascii=False)
        if signature in signatures:
            return
        signatures.add(signature)
        conditions.append({'name': name, 'mask': mask, 'next_active': bool(next_active),
                           'detail': detail})

    current = ['round', 'year', 'month', 'day', 'date_digit_sum', 'date_digital_root']
    for key in current:
        a = f[key]
        if key in ('month', 'day', 'date_digital_root'):
            for v in np.unique(a[train]):
                add(f'{key} = {v}', a == v, future[key] == v,
                    {'source': key, 'lag': 0, 'op': 'eq', 'value': int(v)})
        for m in MODS:
            for r in range(m):
                add(f'{key} mod {m} = {r}', a % m == r, future[key] % m == r,
                    {'source': key, 'lag': 0, 'op': 'mod', 'mod': m, 'value': r})
        if key in ('month', 'day', 'date_digit_sum', 'date_digital_root'):
            for q in (.2, .5, .8):
                v = int(np.quantile(a[train], q))
                for op in ('le', 'ge'):
                    add(f'{key} {"≤" if op=="le" else "≥"} {v}',
                        a <= v if op == 'le' else a >= v,
                        future[key] <= v if op == 'le' else future[key] >= v,
                        {'source': key, 'lag': 0, 'op': op, 'value': v})

    lagged = ['sum', 'odd_count', 'low_count', 'prime_count', 'composite_count',
              'square_count', 'fib_count', 'multiple3_count', 'multiple5_count',
              'tail_sum', 'same_tail_pairs', 'gcd_pairs', 'min_gap', 'max_gap',
              'span', 'consecutive_count', 'ac', 'variance', 'bonus',
              *[f'x{i}' for i in range(1, 7)], *[f'gap{i}' for i in range(1, 6)],
              *[f'decade{i}_count' for i in range(5)],
              'first_prize_rollover', 'bonus_min_distance', 'bonus_gcd_pairs',
              '1등 당첨자수', '1등 1인당 당첨금', '2등 당첨자수', '2등 1인당 당첨금']
    for lag in (1, 2, 3):
        for key in lagged:
            a = np.full(n, np.nan)
            a[lag:] = f[key][:-lag]
            nxt = float(f[key][-lag])
            train_a = a[train & np.isfinite(a)]
            thresholds = np.unique(np.quantile(train_a, [.1, .2, .35, .5, .65, .8, .9]))
            # All integer features use integer thresholds; variance remains exact.
            if key != 'variance':
                thresholds = np.unique(thresholds.astype(int))
            for v in thresholds:
                for op in ('le', 'ge'):
                    add(f'{key}(t−{lag}) {"≤" if op=="le" else "≥"} {v:g}',
                        a <= v if op == 'le' else a >= v,
                        nxt <= v if op == 'le' else nxt >= v,
                        {'source': key, 'lag': lag, 'op': op, 'value': float(v)})
            if len(np.unique(train_a)) <= 15:
                for v in np.unique(train_a):
                    add(f'{key}(t−{lag}) = {v:g}', a == v, nxt == v,
                        {'source': key, 'lag': lag, 'op': 'eq', 'value': float(v)})
            if key in ('sum', 'bonus', *[f'x{i}' for i in range(1, 7)]):
                for m in MODS:
                    for r in range(m):
                        add(f'{key}(t−{lag}) mod {m} = {r}', a % m == r, nxt % m == r,
                            {'source': key, 'lag': lag, 'op': 'mod', 'mod': m, 'value': r})
    for key in [k for k in f if k.startswith('absence_') or k.startswith('overdue_')]:
        a = f[key]
        if key.startswith('absence_number_'):
            number = int(key.rsplit('_', 1)[1])
            # The next gap uses only observations available after the last draw.
            nxt = 0 if any(f[f'x{i}'][-1] == number for i in range(1, 7)) else a[-1]+1
        elif key.startswith('absence_tail_'):
            tail = int(key.rsplit('_', 1)[1])
            nxt = 0 if any(f[f'x{i}'][-1] % 10 == tail for i in range(1, 7)) else a[-1]+1
        else:
            continue  # counts are derived and tested via the individual gaps below
        for k in KS:
            add(f'{key} ≥ {k}회(추첨 직전)', a >= k, nxt >= k,
                {'source': key, 'lag': 0, 'op': 'ge', 'value': k})
    return conditions


def make_events(x, b, f, gap_num, gap_tail, future, null, train_mask=None):
    events = []
    n = len(x)
    train_mask = np.arange(n) < n-250 if train_mask is None else np.asarray(train_mask, bool)
    for event in baseline_events(x, null):
        events.append({**event, 'valid': np.ones(n, bool),
                       'probabilities': np.full(n, event['baseline']),
                       'next_spec': event['spec'], 'bonus_event': False})
    # Fixed four/five member decimal-tail pools have nontrivial exact baselines.
    for r in range(10):
        pool = [v for v in range(1, 46) if v % 10 == r]
        present_p = 1 - math.comb(45-len(pool), 6)/TOTAL
        want_present = present_p <= .5
        p = present_p if want_present else 1-present_p
        if .4 <= p <= .5:
            mask = np.isin(x % 10, r).any(axis=1)
            if not want_present:
                mask = ~mask
            spec = {'kind': 'pool', 'pool': pool, 'op': 'ge' if want_present else 'eq',
                    'value': 1 if want_present else 0}
            events.append({'name': f'끝수 {r} {"출현" if want_present else "미출현"}',
                           'spec': spec, 'next_spec': spec, 'mask': mask,
                           'valid': np.ones(n, bool), 'probabilities': np.full(n, p),
                           'baseline': p, 'bonus_event': False})
    for lag in (1, 2, 3):
        valid = np.arange(n) >= lag
        p = math.comb(39, 6)/TOTAL
        spec = {'kind': 'pool', 'pool': list(map(int, x[-lag])), 'op': 'eq', 'value': 0}
        events.append({'name': f't−{lag} 본번호 이월수 = 0', 'spec': {'kind': 'carry', 'lag': lag},
                       'next_spec': spec, 'mask': f[f'carry_main_lag{lag}'] == 0,
                       'valid': valid, 'probabilities': np.full(n, p), 'baseline': p,
                       'bonus_event': False})
    for key in ('round', 'year', 'month', 'day', 'date_digit_sum', 'date_digital_root'):
        for m in MODS:
            half = m//2
            residues = list(range(half))
            hist = null['hist']['sum']
            sum_values = np.arange(len(hist))
            probs = np.array([hist[np.isin((sum_values-r) % m, residues)].sum()/TOTAL
                              for r in range(m)])
            baseline = probs[f[key] % m]
            # Eligibility depends on theoretical null probabilities and training contexts.
            if not (.4 <= baseline[train_mask].mean() <= .5):
                continue
            mask = np.isin((f['sum'] - f[key]) % m, residues)
            next_spec = {'terms': [{'feature': 'sum', 'op': 'modin', 'mod': m,
                                   'value': [(r+future[key]) % m for r in residues]}], 'logic': 'and'}
            events.append({'name': f'(sum − {key}) mod {m} ∈ {residues}',
                           'spec': {'kind': 'date_mod', 'source': key, 'mod': m, 'residues': residues},
                           'next_spec': next_spec, 'mask': mask, 'valid': np.ones(n, bool),
                           'probabilities': baseline, 'baseline': float(baseline[train_mask].mean()),
                           'next_probability': float(probs[future[key] % m]),
                           'bonus_event': False})
        tail = f[key] % 10
        pool_size = np.where((tail >= 1) & (tail <= 5), 5, 4)
        present_probability = np.array([1-math.comb(45-int(s), 6)/TOTAL for s in pool_size])
        want_present = present_probability <= .5
        observed_present = (x % 10 == tail[:, None]).any(axis=1)
        probability = np.where(want_present, present_probability, 1-present_probability)
        next_tail = future[key] % 10
        next_pool = [j for j in range(1, 46) if j % 10 == next_tail]
        next_present = len(next_pool) == 4
        events.append({'name': f'{key} 끝수: 4개 번호인 끝수는 출현, 5개 번호인 끝수는 미출현',
                       'spec': {'kind': 'date_tail', 'source': key},
                       'next_spec': {'kind': 'pool', 'pool': next_pool,
                                     'op': 'ge' if next_present else 'eq',
                                     'value': 1 if next_present else 0},
                       'mask': observed_present == want_present, 'valid': np.ones(n, bool),
                       'probabilities': probability, 'baseline': float(probability[train_mask].mean()),
                       'next_probability': float(1-math.comb(45-len(next_pool), 6)/TOTAL) if next_present else float(math.comb(45-len(next_pool), 6)/TOTAL),
                       'bonus_event': False})
    # Overdue pools are defined before observing the current draw; restrict their size
    # to four or five, giving an exact fixed 40--50% event probability.
    next_num_gaps = np.array([0 if j in x[-1] else gap_num[-1, j-1]+1 for j in range(1, 46)])
    next_tail_gaps = np.array([0 if r in x[-1] % 10 else gap_tail[-1, r]+1 for r in range(10)])
    for k in KS:
        for domain, gaps, next_gaps in [('number', gap_num, next_num_gaps), ('tail', gap_tail, next_tail_gaps)]:
            overdue = gaps >= k
            if domain == 'number':
                pool_masks = overdue
            else:
                pool_masks = overdue[:, np.arange(1, 46) % 10]
            size = pool_masks.sum(axis=1)
            for poolsize in (4, 5):
                valid = (size == poolsize) & (np.arange(n) >= k)
                if not valid[train_mask].any():
                    continue
                present = np.take_along_axis(pool_masks, x-1, axis=1).any(axis=1)
                p_present = 1-math.comb(45-poolsize, 6)/TOTAL
                want_present = p_present <= .5
                p = p_present if want_present else 1-p_present
                pool = [j for j in range(1, 46) if next_gaps[j-1 if domain == 'number' else j % 10] >= k]
                next_spec = {'kind': 'pool', 'pool': pool, 'op': 'ge' if want_present else 'eq',
                             'value': 1 if want_present else 0} if len(pool) == poolsize else None
                events.append({'name': f'{domain} 미출현≥{k}회 집합 크기={poolsize}: '
                               f'{"하나 이상 출현" if want_present else "전부 미출현"}',
                               'spec': {'kind': 'overdue', 'domain': domain, 'k': k, 'size': poolsize,
                                        'present': want_present},
                               'next_spec': next_spec, 'mask': present if want_present else ~present,
                               'valid': valid, 'probabilities': np.full(n, p), 'baseline': p,
                               'bonus_event': False})
    for feature, mask, p in [('bonus_even', b % 2 == 0, 22/45), ('bonus_low', b <= 22, 22/45)]:
        spec = {'kind': 'bonus', 'feature': feature}
        events.append({'name': '보너스 짝수' if feature == 'bonus_even' else '보너스 ≤ 22',
                       'spec': spec, 'next_spec': spec, 'mask': mask, 'valid': np.ones(n, bool),
                       'probabilities': np.full(n, p), 'baseline': p, 'bonus_event': True})
    for feature, mask in [('bonus_below_x3', b < x[:, 2]), ('bonus_above_x4', b > x[:, 3])]:
        spec = {'kind': 'bonus', 'feature': feature}
        events.append({'name': '보너스 < x3' if feature.endswith('x3') else '보너스 > x4',
                       'spec': spec, 'next_spec': spec, 'mask': mask, 'valid': np.ones(n, bool),
                       'probabilities': np.full(n, 3/7), 'baseline': 3/7, 'bonus_event': True})
    # Sum of main+bonus has the same distribution as a uniformly selected
    # seven-element subset; DP counts coefficients of product(1+z*q^j).
    dp = np.zeros((8, 316), dtype=np.int64)
    dp[0, 0] = 1
    for number in range(1, 46):
        for count in range(7, 0, -1):
            dp[count, number:] += dp[count-1, :-number]
    assert int(dp[7].sum()) == math.comb(45, 7)
    for m in MODS:
        residues = list(range(m//2))
        p = dp[7, np.isin(np.arange(316) % m, residues)].sum()/math.comb(45, 7)
        if not (.4 <= p <= .5): continue
        spec = {'kind': 'bonus_sum_mod', 'mod': m, 'residues': residues}
        events.append({'name': f'(sum + bonus) mod {m} ∈ {residues}', 'spec': spec, 'next_spec': spec,
                       'mask': np.isin((f['sum']+b) % m, residues), 'valid': np.ones(n, bool),
                       'probabilities': np.full(n, p), 'baseline': float(p), 'bonus_event': True})
    # Exact mathematical definitions, not observed historical agreement, deduplicate events.
    unique = {}
    for e in events:
        key = json.dumps(e['spec'], sort_keys=True, ensure_ascii=False)
        unique.setdefault(key, e)
    return list(unique.values())



def main():
    from predict_top15 import main as predict_main
    return predict_main()


if __name__ == '__main__':
    main()
