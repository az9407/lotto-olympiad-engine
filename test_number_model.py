"""Number-transition timing, training isolation and probability checks."""
from __future__ import annotations

import unittest

import numpy as np

from number_model import build_transition_features, fit_number_model, normalize_marginals


def history(seed=5, n=80):
    rng = np.random.default_rng(seed)
    rows = np.array([np.sort(rng.choice(np.arange(1, 46), 7, replace=False)) for _ in range(n)])
    return rows[:, :6], rows[:, 6]


class NumberModelTests(unittest.TestCase):
    def test_pre_draw_carry_bonus_neighbor_gap_and_hot(self):
        x = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [1, 13, 14, 15, 16, 17]])
        b = np.array([20, 21, 22])
        state = build_transition_features(x, b)
        codes = state['codes']
        self.assertTrue(np.all(codes['carry_lag1'][0] == 2))
        np.testing.assert_array_equal(np.flatnonzero(codes['carry_lag1'][1])+1, x[0])
        self.assertEqual(codes['previous_bonus'][1, 19], 1)
        self.assertEqual(codes['previous_bonus'][1, 20], 0)
        self.assertEqual(codes['neighbor_lag1'][1, 6], 1)  # 7 is adjacent to previous 6.
        self.assertEqual(codes['neighbor_lag1'][1, 7], 0)
        self.assertEqual(state['number_gap'][2, 0], 1)
        self.assertEqual(state['number_gap'][3, 0], 0)
        self.assertEqual(state['number_gap'][3, 6], 1)
        self.assertEqual(codes['hot10'][2, 0], 1)
        self.assertEqual(codes['hot10'][3, 0], 2)
        self.assertEqual(state['minimum_distance'][1, 6], 1)

    def test_covariates_do_not_read_current_or_later_outcomes(self):
        x, b = history()
        before = build_transition_features(x, b)
        changed_x, changed_b = x.copy(), b.copy()
        changed_x[50:] = np.array([1, 8, 16, 24, 32, 40])
        changed_b[50:] = 45
        after = build_transition_features(changed_x, changed_b)
        for key in before['codes']:
            np.testing.assert_array_equal(before['codes'][key][:51], after['codes'][key][:51])

    def test_training_fit_does_not_change_when_validation_outcomes_change(self):
        x, b = history()
        train = np.arange(len(x)) < 50
        original = fit_number_model(x, b, train, ~train)
        changed_x, changed_b = x.copy(), b.copy()
        changed_x[50:] = np.array([1, 8, 16, 24, 32, 40])
        changed_b[50:] = 45
        changed = fit_number_model(changed_x, changed_b, train, ~train)
        self.assertEqual(original['diagnostics']['training_model'], changed['diagnostics']['training_model'])
        np.testing.assert_array_equal(original['history_probabilities'][train], changed['history_probabilities'][train])
        # Prediction at the first held-out draw also has unchanged past information.
        np.testing.assert_array_equal(original['history_probabilities'][50], changed['history_probabilities'][50])

    def test_marginals_are_strictly_between_zero_and_one_and_sum_to_six(self):
        x, b = history()
        train = np.arange(len(x)) < 50
        result = fit_number_model(x, b, train, ~train)
        p = result['probabilities']
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertAlmostEqual(float(p.sum()), 6, places=11)
        self.assertEqual(len(result['table']), 45)
        np.testing.assert_allclose(result['history_probabilities'].sum(axis=1), 6, atol=1e-11)

    def test_normalization_preserves_order_and_flat_prior(self):
        logits = np.linspace(-4, 3, 45)
        p = normalize_marginals(logits)
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertAlmostEqual(float(p.sum()), 6, places=11)
        np.testing.assert_allclose(normalize_marginals(np.zeros(45)), np.full(45, 6/45), atol=1e-14)

    def test_smoothing_covers_unseen_numbers_without_zero_probabilities(self):
        x = np.tile([1, 2, 3, 4, 5, 6], (40, 1))
        b = np.full(40, 45)
        train = np.arange(40) < 30
        result = fit_number_model(x, b, train, ~train)
        self.assertTrue(np.all(result['probabilities'] > 0))
        self.assertTrue(np.all(result['probabilities'] < 1))
        self.assertTrue(np.isfinite(np.log(result['probabilities'])).all())


if __name__ == '__main__':
    unittest.main()
