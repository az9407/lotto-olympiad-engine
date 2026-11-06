"""Smoothed, pre-draw empirical number-transition model for Lotto 6/45.

The returned values are model marginal inclusion estimates, not calibrated
joint winning probabilities.  All covariates for draw t use draws before t.
Validation predictions freeze coefficients fitted on the supplied train mask;
the next-draw forecast refits the same fixed specification on all known draws.
"""
from __future__ import annotations

import math

import numpy as np


BASELINE = 6.0 / 45.0
IDENTITY_STRENGTH = 200.0
TRANSITION_STRENGTH = 120.0
GAP_EDGES = np.array([1, 3, 5, 10, 15, 20, 30], dtype=int)
DISTANCE_EDGES = np.array([1, 2, 3, 5, 10, 20], dtype=int)
SIGNED_EDGES = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10], dtype=int)
WEIGHTS = {
    'identity': .25,
    'carry_lag1': .08,
    'carry_lag2': .04,
    'carry_lag3': .025,
    'previous_bonus': .025,
    'neighbor_lag1': .075,
    'minimum_distance_bin': .07,
    'number_gap_bin': .09,
    'tail_gap_bin': .045,
    'hot5': .045,
    'hot10': .055,
    'hot20': .085,
    'tail_hot10': .055,
    'signed_distance_bin': .045,
}
WEIGHT_TOTAL = sum(WEIGHTS.values())
WEIGHTS = {key: value / WEIGHT_TOTAL for key, value in WEIGHTS.items()}
CODE_SIZES = {
    'identity': 45,
    'carry_lag1': 3, 'carry_lag2': 3, 'carry_lag3': 3,
    'previous_bonus': 3, 'neighbor_lag1': 3,
    'minimum_distance_bin': len(DISTANCE_EDGES) + 2,
    'number_gap_bin': len(GAP_EDGES) + 1,
    'tail_gap_bin': len(GAP_EDGES) + 1,
    'hot5': 6, 'hot10': 11, 'hot20': 21,
    'tail_hot10': 11,
    'signed_distance_bin': len(SIGNED_EDGES) + 2,
}


def _validate_history(x, bonus):
    x = np.asarray(x, dtype=np.int16)
    bonus = np.asarray(bonus, dtype=np.int16)
    if x.ndim != 2 or x.shape[1] != 6 or bonus.shape != (len(x),):
        raise ValueError('Expected main numbers (n,6) and bonus (n,)')
    if not len(x):
        raise ValueError('At least one historical draw is required')
    if not np.all((x >= 1) & (x <= 45)) or not np.all(np.diff(x, axis=1) > 0):
        raise ValueError('Main numbers must be six sorted distinct integers in 1..45')
    if not np.all((bonus >= 1) & (bonus <= 45)) or np.any(x == bonus[:, None]):
        raise ValueError('Bonus must be in 1..45 and outside its main draw')
    return x, bonus


def build_transition_features(x, bonus):
    """Return fixed categorical codes for historical draws plus the next draw.

    Row t means the state immediately *before* zero-based historical draw t.
    The final row t=n means the upcoming draw.  Code 2 denotes unavailable lag
    information for binary lag covariates.  Gap 0 means present last draw.
    """
    x, bonus = _validate_history(x, bonus)
    n = len(x)
    numbers = np.arange(1, 46, dtype=np.int16)
    present = np.zeros((n, 45), dtype=np.int8)
    present[np.arange(n)[:, None], x-1] = 1
    tail_present = np.zeros((n, 10), dtype=np.int8)
    for t, row in enumerate(x):
        tail_present[t, np.unique(row % 10)] = 1
    codes = {'identity': np.broadcast_to(np.arange(45), (n+1, 45)).copy()}
    for lag in (1, 2, 3):
        a = np.full((n+1, 45), 2, dtype=np.int16)
        a[lag:] = present[:n+1-lag]
        codes[f'carry_lag{lag}'] = a
    for key in ('previous_bonus', 'neighbor_lag1'):
        codes[key] = np.full((n+1, 45), 2, dtype=np.int16)
    codes['previous_bonus'][1:] = 0
    codes['previous_bonus'][np.arange(1, n+1), bonus-1] = 1
    neighbor = np.zeros_like(present)
    neighbor[:, 1:] |= present[:, :-1]
    neighbor[:, :-1] |= present[:, 1:]
    codes['neighbor_lag1'][1:] = neighbor
    distances = np.full((n+1, 45), 45, dtype=np.int16)
    signed = np.full((n+1, 45), 45, dtype=np.int16)
    for t, row in enumerate(x, 1):
        delta = numbers[:, None] - row[None, :]
        nearest = np.argmin(np.abs(delta), axis=1)
        signed[t] = delta[np.arange(45), nearest]
        distances[t] = np.abs(signed[t])
    codes['minimum_distance_bin'] = np.searchsorted(DISTANCE_EDGES, distances, side='right')
    codes['minimum_distance_bin'][0] = CODE_SIZES['minimum_distance_bin']-1
    codes['signed_distance_bin'] = np.searchsorted(SIGNED_EDGES, signed, side='right')
    codes['signed_distance_bin'][0] = CODE_SIZES['signed_distance_bin']-1
    number_gap = np.empty((n+1, 45), dtype=int)
    tail_gap = np.empty((n+1, 10), dtype=int)
    last_number = np.full(45, -1, dtype=int)
    last_tail = np.full(10, -1, dtype=int)
    for t in range(n+1):
        number_gap[t] = t-last_number-1
        tail_gap[t] = t-last_tail-1
        if t < n:
            last_number[x[t]-1] = t
            last_tail[np.unique(x[t] % 10)] = t
    codes['number_gap_bin'] = np.searchsorted(GAP_EDGES, number_gap, side='right')
    codes['tail_gap_bin'] = np.searchsorted(GAP_EDGES, tail_gap[:, numbers % 10], side='right')
    prefix = np.vstack([np.zeros((1, 45), dtype=int), np.cumsum(present, axis=0)])
    tail_prefix = np.vstack([np.zeros((1, 10), dtype=int), np.cumsum(tail_present, axis=0)])
    times = np.arange(n+1)
    for window in (5, 10, 20):
        codes[f'hot{window}'] = prefix[times]-prefix[np.maximum(0, times-window)]
    codes['tail_hot10'] = (tail_prefix[times]-tail_prefix[np.maximum(0, times-10)])[:, numbers % 10]
    assert set(codes) == set(WEIGHTS)
    assert all(a.shape == (n+1, 45) for a in codes.values())
    assert all(np.all((a >= 0) & (a < CODE_SIZES[key])) for key, a in codes.items())
    return {'codes': codes, 'outcomes': present, 'number_gap': number_gap,
            'tail_gap': tail_gap[:, numbers % 10], 'minimum_distance': distances,
            'signed_distance': signed}


def _fit_posterior(codes, outcomes, mask):
    fitted = {}
    y = outcomes[mask].ravel().astype(float)
    for key, a in codes.items():
        observed_codes = a[:-1][mask].ravel()
        counts = np.bincount(observed_codes, minlength=CODE_SIZES[key])
        hits = np.bincount(observed_codes, weights=y, minlength=CODE_SIZES[key])
        strength = IDENTITY_STRENGTH if key == 'identity' else TRANSITION_STRENGTH
        probabilities = (hits + strength*BASELINE)/(counts+strength)
        fitted[key] = {'probabilities': probabilities, 'hits': hits, 'counts': counts}
    return fitted


def _sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z, -700, 700)))


def normalize_marginals(logits, target=6.0):
    """Preserve logit ordering while imposing exactly six expected inclusions."""
    logits = np.asarray(logits, dtype=float)
    one_row = logits.ndim == 1
    a = logits[None, :] if one_row else logits
    if a.ndim != 2 or a.shape[1] != 45 or not np.isfinite(a).all():
        raise ValueError('Expected finite logits with final dimension 45')
    lo = np.full(len(a), -50.0)
    hi = np.full(len(a), 50.0)
    for _ in range(64):
        middle = (lo+hi)/2
        sums = _sigmoid(a+middle[:, None]).sum(axis=1)
        lo = np.where(sums < target, middle, lo)
        hi = np.where(sums >= target, middle, hi)
    p = _sigmoid(a+((lo+hi)/2)[:, None])
    return p[0] if one_row else p


def _predict(codes, fitted):
    shape = next(iter(codes.values())).shape
    logits = np.zeros(shape, dtype=float)
    for key, a in codes.items():
        p = fitted[key]['probabilities'][a]
        logits += WEIGHTS[key]*(np.log(p)-np.log1p(-p))
    return normalize_marginals(logits)


def _diagnostic_metrics(y, p):
    if not len(y):
        return None
    clipped = np.clip(p, 1e-15, 1-1e-15)
    return {'number_draw_observations': int(y.size),
            'brier': float(np.mean((p-y)**2)),
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            'mean_predicted_marginal': float(p.mean()), 'empirical_marginal': float(y.mean()),
            'row_probability_sum_max_error': float(np.max(np.abs(p.sum(axis=1)-6)))}


def fit_number_model(x, bonus, trainmask, validmask):
    """Fit fixed smoothed empirical transitions and forecast next inclusion rates.

    Validation is used only for diagnostic evaluation in this component. Its
    coefficients, feature bins, shrinkage constants and mixture weights are
    fixed before evaluation. The final forecast explicitly refits on every known
    row, including previously held-out draws, after reporting frozen-fit scores.
    """
    x, bonus = _validate_history(x, bonus)
    n = len(x)
    train = np.asarray(trainmask, dtype=bool)
    valid = np.asarray(validmask, dtype=bool)
    if train.shape != (n,) or valid.shape != (n,) or np.any(train & valid) or not train.any():
        raise ValueError('Expected disjoint training/validation masks and nonempty training')
    state = build_transition_features(x, bonus)
    codes, y = state['codes'], state['outcomes']
    training_fit = _fit_posterior(codes, y, train)
    frozen = _predict(codes, training_fit)
    final_fit = _fit_posterior(codes, y, np.ones(n, dtype=bool))
    final = _predict({key: value[-1:] for key, value in codes.items()}, final_fit)[0]
    assert np.isfinite(final).all() and np.all((final > 0) & (final < 1))
    assert abs(float(final.sum())-6) < 1e-10
    table = []
    for number, p in enumerate(final, 1):
        j = number-1
        table.append({'number': number, 'probability': float(p),
                      'last_carry': int(codes['carry_lag1'][-1, j]),
                      'carry_lag2': int(codes['carry_lag2'][-1, j]),
                      'carry_lag3': int(codes['carry_lag3'][-1, j]),
                      'previous_bonus': int(codes['previous_bonus'][-1, j]),
                      'neighbor': int(codes['neighbor_lag1'][-1, j]),
                      'gap': int(state['number_gap'][-1, j]),
                      'tail_gap': int(state['tail_gap'][-1, j]),
                      'minimum_distance': int(state['minimum_distance'][-1, j]),
                      'signed_distance': int(state['signed_distance'][-1, j]),
                      'last5_hot': int(codes['hot5'][-1, j]),
                      'last10_hot': int(codes['hot10'][-1, j]),
                      'last20_hot': int(codes['hot20'][-1, j]),
                      'tail_last10_hot': int(codes['tail_hot10'][-1, j])})
    serializable_training_fit = {
        key: {field: array.tolist() for field, array in values.items()}
        for key, values in training_fit.items()}
    baseline_valid = np.full_like(frozen[:-1][valid], BASELINE)
    return {'probabilities': final, 'table': table,
            'history_probabilities': frozen[:-1],
            'diagnostics': {
                'baseline': BASELINE, 'identity_prior_strength': IDENTITY_STRENGTH,
                'transition_prior_strength': TRANSITION_STRENGTH,
                'weights': WEIGHTS,
                'conditional_estimate': '(hits + strength*(6/45))/(observations + strength)',
                'ensemble': 'weighted mean of conditional posterior logits; common intercept enforces sum(p)=6',
                'gap_bins_edges': GAP_EDGES.tolist(), 'minimum_distance_edges': DISTANCE_EDGES.tolist(),
                'signed_distance_edges': SIGNED_EDGES.tolist(),
                'validation': _diagnostic_metrics(y[valid], frozen[:-1][valid]),
                'validation_constant_baseline': _diagnostic_metrics(y[valid], baseline_valid),
                'training_in_sample': _diagnostic_metrics(y[train], frozen[:-1][train]),
                'forecast_refit_rows': n, 'frozen_validation_training_rows': int(train.sum()),
                'training_model': serializable_training_fit,
                'forecast_refit': 'same fixed specification refitted on all observed draws after frozen validation evaluation',
                'interpretation': 'empirical model marginal inclusion estimates; sum(p)=6; not a joint combination probability',
                'tie_note': 'No random jitter or artificial tie breaker; unequal marginals do not guarantee unequal six-number log sums',
            }}


if __name__ == '__main__':
    import json
    from pathlib import Path
    import pandas as pd
    source = next(Path('.codex-attachments').rglob('*.xlsx'))
    df = pd.read_excel(source, sheet_name='로또 당첨번호').sort_values('회차').reset_index(drop=True)
    main = df[[f'번호{i}' for i in range(1, 7)]].to_numpy(int)
    bonus = df['보너스'].to_numpy(int)
    train = np.arange(len(df)) < len(df)-250
    result = fit_number_model(main, bonus, train, ~train)
    print(json.dumps({'top_numbers': sorted(result['table'], key=lambda row: -row['probability'])[:10],
                      'validation': result['diagnostics']['validation'],
                      'baseline': result['diagnostics']['validation_constant_baseline'],
                      'sum': float(result['probabilities'].sum())}, ensure_ascii=False))
