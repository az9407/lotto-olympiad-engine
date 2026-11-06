"""Independent combinatorial and chronological feature checks."""
import itertools
import math
import unittest
from pathlib import Path

import numpy as np

from combinatorial import static_features, enumerate_null, self_test
from rule_engine import load_context


class ArithmeticAndChronology(unittest.TestCase):
    def test_scalar_feature_definitions(self):
        rows = np.array([[1,2,3,4,5,6],[2,11,21,31,41,45],[16,18,27,36,38,42]])
        f = static_features(rows)
        for i, row in enumerate(rows.tolist()):
            pairs = list(itertools.combinations(row, 2))
            self.assertEqual(f['ac'][i], len({b-a for a,b in pairs})-5)
            self.assertEqual(f['gcd_pairs'][i], sum(math.gcd(a,b)>1 for a,b in pairs))
            self.assertEqual(f['same_tail_pairs'][i], sum(a%10==b%10 for a,b in pairs))

    def test_exact_null_against_closed_forms(self):
        self.assertTrue(self_test(enumerate_null(Path('.cache')))['exact_null_checked'])

    def test_chronological_latest_and_absence(self):
        path = next(Path('.codex-attachments').rglob('*.xlsx'))
        c = load_context(path)
        self.assertEqual(c['target_round'], len(c['x'])+1)
        self.assertEqual(c['df']['회차'].iloc[0], 1)
        for t in (1, 2, 100, len(c['x'])-1):
            for v in (1,16,45):
                previous = [j for j,row in enumerate(c['x'][:t]) if v in row]
                gap = t-previous[-1]-1 if previous else t
                self.assertEqual(c['gap_num'][t,v-1], gap)


if __name__ == '__main__':
    unittest.main()
