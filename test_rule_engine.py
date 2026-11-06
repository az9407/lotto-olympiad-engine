"""Independent chronology, activation, sample-count and next-row checks.

These tests validate the mechanics of the composite scoring model. They make
no claim that the selected empirical rules establish future winning odds.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from combinatorial import enumerate_null
from lotto_analysis import history_features, make_conditions, make_events
from rule_engine import load_context, discover_rules


def scalar_term(value, term):
    threshold = term['value']
    return {'le': lambda: value <= threshold,
            'ge': lambda: value >= threshold,
            'eq': lambda: value == threshold,
            'between': lambda: threshold[0] <= value <= threshold[1],
            'modin': lambda: value % term['mod'] in threshold}[term['op']]()


def scalar_static_spec(features, spec, index):
    results = [scalar_term(features[t['feature']][index], t) for t in spec['terms']]
    return any(results) if spec.get('logic', 'and') == 'or' else all(results)


def condition_comparison(value, detail):
    threshold = detail['value']
    if detail['op'] == 'mod':
        return value % detail['mod'] == threshold
    return {'eq': lambda: value == threshold,
            'le': lambda: value <= threshold,
            'ge': lambda: value >= threshold}[detail['op']]()


def independent_condition_mask(context, detail):
    source, lag = detail['source'], detail['lag']
    result = np.zeros(len(context['x']), bool)
    for t in range(lag, len(result)):
        value = context['features'][source][t-lag]
        result[t] = condition_comparison(value, detail)
    return result


def independent_event_mask(context, definition):
    """Evaluate mathematical event definitions without the production evaluator."""
    x, f = context['x'], context['features']
    result = []
    for t, row in enumerate(x.tolist()):
        kind = definition.get('kind')
        if kind is None:
            result.append(scalar_static_spec(f, definition, t))
        elif kind == 'pool':
            size = len(set(row) & set(definition['pool']))
            result.append(size == definition['value'] if definition['op'] == 'eq' else size >= definition['value'])
        elif kind == 'carry':
            lag = definition['lag']
            result.append(t >= lag and not(set(row) & set(x[t-lag])))
        elif kind == 'date_mod':
            result.append((sum(row)-int(f[definition['source']][t])) % definition['mod'] in definition['residues'])
        elif kind == 'date_tail':
            tail = int(f[definition['source']][t]) % 10
            present = any(n % 10 == tail for n in row)
            result.append(not present if 1 <= tail <= 5 else present)
        elif kind == 'overdue':
            k = definition['k']
            gap = context['gap_num'][t] if definition['domain'] == 'number' else context['gap_tail'][t]
            pool = {n for n in range(1, 46) if gap[n-1 if definition['domain'] == 'number' else n % 10] >= k}
            present = bool(set(row) & pool)
            result.append(present if definition['present'] else not present)
        else:
            raise AssertionError(f'Unexpected six-number event: {definition}')
    return np.array(result, bool)


class RuleEngineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next(Path('.codex-attachments').rglob('*.xlsx'))
        cls.context = load_context(cls.source)
        cls.null = enumerate_null(Path('outputs/cache'))
        cls.registry = discover_rules(cls.context, Path('outputs/cache'))

    def test_latest_row_and_split_control_target_without_fixed_round_constants(self):
        c = self.context
        last_round = int(c['df']['회차'].iloc[-1])
        self.assertEqual(c['target_round'], last_round+1)
        self.assertEqual(c['target_date'], c['df']['추첨일'].iloc[-1]+pd.Timedelta(days=7))
        self.assertEqual(int(c['train'].sum()), len(c['df'])-250)
        self.assertEqual(int(c['valid'].sum()), 250)
        self.assertTrue(np.all(np.diff(c['df']['회차']) == 1))
        self.assertFalse(np.any(c['train'] & c['valid']))
        self.assertTrue(np.all(c['train'] | c['valid']))

    def test_active_conditions_are_true_for_actual_latest_lag_values(self):
        c = self.context
        self.assertGreaterEqual(len(self.registry['eligible_a']), 30)
        for rule in self.registry['eligible_a']:
            detail = rule['condition_detail']
            key, lag = detail['source'], detail['lag']
            if lag:
                actual = float(c['features'][key][-lag])
            elif key in c['future']:
                actual = c['future'][key]
            else:
                if key.startswith('absence_number_'):
                    number = int(key.rsplit('_', 1)[1])
                    positions = [i for i, row in enumerate(c['x'].tolist()) if number in row]
                elif key.startswith('absence_tail_'):
                    tail = int(key.rsplit('_', 1)[1])
                    positions = [i for i, row in enumerate(c['x'].tolist()) if any(n % 10 == tail for n in row)]
                else:
                    self.fail(key)
                actual = len(c['x'])-1-positions[-1] if positions else len(c['x'])
            with self.subTest(rule=rule['name']):
                self.assertEqual(actual, rule['activation_evidence']['observed_value'])
                self.assertTrue(condition_comparison(actual, detail))
                self.assertTrue(rule['condition_active'])

    def test_a_samples_and_successes_match_independent_predicates(self):
        c = self.context
        event_cache = {}
        for rule in self.registry['eligible_a']:
            key = str(rule['event_definition'])
            if key not in event_cache:
                event_cache[key] = independent_event_mask(c, rule['event_definition'])
            outcome = event_cache[key]
            condition = independent_condition_mask(c, rule['condition_detail'])
            eligible = condition & rule['history_valid']
            with self.subTest(rule=rule['name']):
                np.testing.assert_array_equal(condition, rule['condition_history_mask'])
                np.testing.assert_array_equal(outcome[rule['history_valid']], rule['history_mask'][rule['history_valid']])
                for name, block in [('train', c['train']), ('valid', c['valid'])]:
                    selected = eligible & block
                    self.assertEqual(int(selected.sum()), rule[f'{name}_n'])
                    self.assertEqual(int((outcome & selected).sum()), rule[f'{name}_k'])
                    self.assertGreaterEqual(100*rule[f'{name}_k'], 55*rule[f'{name}_n'])
                self.assertGreaterEqual(rule['train_n'], 100)
                self.assertGreaterEqual(rule['valid_n'], 30)

    def test_b_lift_and_validation_samples_match_scalar_filter(self):
        c = self.context
        self.assertGreater(len(self.registry['eligible_b']), 0)
        for rule in self.registry['eligible_b']:
            outcome = np.array([scalar_static_spec(c['features'], rule['spec'], i) for i in range(len(c['x']))])
            with self.subTest(rule=rule['name']):
                for name, block in [('train', c['train']), ('valid', c['valid'])]:
                    self.assertEqual(int((outcome & block).sum()), rule[f'{name}_k'])
                    self.assertEqual(int(block.sum()), rule[f'{name}_n'])
                expected_lift = float(outcome.mean())/rule['q']
                self.assertAlmostEqual(expected_lift, rule['lift'])
                self.assertGreater(expected_lift, 1.05)
                self.assertGreaterEqual(100*rule['valid_k'], 55*rule['valid_n'])
                self.assertTrue(0 < rule['q'] <= .5)

    def test_append_one_draw_advances_target_date_and_both_split_boundaries(self):
        c = self.context
        appended = c['df'].iloc[-1].copy()
        appended['회차'] = c['target_round']
        appended['추첨일'] = c['target_date']
        for i, number in enumerate([1, 7, 12, 23, 31, 44], 1):
            appended[f'번호{i}'] = number
        appended['보너스'] = 6
        df = pd.concat([c['df'], appended.to_frame().T], ignore_index=True)
        for col in df:
            df[col] = pd.to_datetime(df[col]) if col == '추첨일' else df[col].astype(int)
        with patch('rule_engine.read_data', return_value=df):
            extended = load_context(self.source)
        self.assertEqual(extended['target_round'], c['target_round']+1)
        self.assertEqual(extended['target_date'], c['target_date']+pd.Timedelta(days=7))
        self.assertEqual(int(extended['train'].sum()), int(c['train'].sum())+1)
        self.assertEqual(int(extended['valid'].sum()), 250)
        for key, mask in [('train', c['train']), ('valid', c['valid'])]:
            self.assertEqual(int(extended['df']['회차'][extended[key]].iloc[0]),
                             int(c['df']['회차'][mask].iloc[0])+(key == 'valid'))
            self.assertEqual(int(extended['df']['회차'][extended[key]].iloc[-1]),
                             int(c['df']['회차'][mask].iloc[-1])+1)
        if c['target_round'] == 1249:
            self.assertEqual(extended['target_round'], 1250)
            self.assertEqual(str(extended['target_date'].date()), '2026-11-14')

    def test_training_thresholds_and_event_definitions_ignore_validation_numbers(self):
        c = self.context
        altered = c['df'].copy()
        altered.loc[c['valid'], [f'번호{i}' for i in range(1, 7)]] = [1, 2, 3, 4, 5, 6]
        altered.loc[c['valid'], '보너스'] = 7
        x, b, f, g, h, future = history_features(altered)
        original_conditions = make_conditions(c['features'], c['train'], c['future'])
        altered_conditions = make_conditions(f, c['train'], future)
        self.assertEqual([r['detail'] for r in original_conditions], [r['detail'] for r in altered_conditions])
        for left, right in zip(original_conditions, altered_conditions):
            np.testing.assert_array_equal(left['mask'][c['train']], right['mask'][c['train']])
        original_events = make_events(c['x'], c['b'], c['features'], c['gap_num'], c['gap_tail'],
                                     c['future'], self.null, c['train'])
        altered_events = make_events(x, b, f, g, h, future, self.null, c['train'])
        self.assertEqual([(r['name'], r['spec']) for r in original_events],
                         [(r['name'], r['spec']) for r in altered_events])
        for left, right in zip(original_events, altered_events):
            np.testing.assert_array_equal(left['mask'][c['train']], right['mask'][c['train']])
            np.testing.assert_array_equal(left['valid'][c['train']], right['valid'][c['train']])
            np.testing.assert_allclose(left['probabilities'][c['train']], right['probabilities'][c['train']], rtol=0, atol=0)

    def test_invalid_sample_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_context(self.source, validation_size=29)
        with self.assertRaises(ValueError):
            load_context(self.source, validation_size=len(self.context['df'])-99)


if __name__ == '__main__':
    unittest.main()
