"""Independent scoring, total ordering and selected-artifact checks."""
import itertools
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predict_top15 import raw_log_odds_weight, score_batch, strict_order, diversity_select, tie_audit
from generate_candidates import evaluate_next
from combinatorial import static_features
from rule_engine import load_context


class CompositeRankingChecks(unittest.TestCase):
    def test_requested_log_odds_formula(self):
        expected = math.log(.6/.4)-math.log(.45/.55)
        self.assertAlmostEqual(raw_log_odds_weight(.6,.45,100), expected)
        self.assertTrue(math.isfinite(raw_log_odds_weight(1,.45,30)))

    def test_weighted_rule_and_number_score_scalar(self):
        x = np.array([[1,2,3,4,5,6],[4,8,16,36,38,40]])
        p = np.linspace(.10,.17,45)
        rules = [{'spec': {'kind':'pool','pool':[16,18,27,36,38,42],'op':'eq','value':0}, 'weight':.4},
                 {'spec': {'terms':[{'feature':'sum','op':'ge','value':100}]}, 'weight':.7}]
        sr,sn,count = score_batch(x,rules,p,.05)
        for i,row in enumerate(x.tolist()):
            a = not bool(set(row)&{16,18,27,36,38,42})
            b = sum(row)>=100
            self.assertAlmostEqual(sr[i], .4*a+.7*b)
            self.assertAlmostEqual(sn[i], .05*sum(math.log(p[n-1]) for n in row))
            self.assertEqual(count[i], a+b)

    def test_exact_tie_total_order_without_randomness(self):
        scores = np.array([2.,1.,2.,2.+1e-10])
        order = strict_order(scores)
        self.assertEqual(order.tolist(), [3,0,2,1])
        audit = tie_audit(scores,order,[3,0])
        self.assertEqual(audit['exact_equal_adjacent_pairs_all_combinations'],1)
        self.assertGreater(audit['display_8dp_equal_adjacent_pairs_all_combinations'],0)

    def test_constrained_greedy_small_universe(self):
        x = np.array(list(itertools.combinations(range(1,10),3)),dtype=np.uint8)
        score = np.arange(len(x),dtype=float)
        selected,ranks,_ = diversity_select(x,strict_order(score),np.ones(len(x),bool),count=4,max_shared=1,max_exposure=2)
        self.assertEqual(selected[0],len(x)-1)
        self.assertTrue(all(a<b for a,b in zip(ranks,ranks[1:])))
        for a,b in itertools.combinations(selected,2): self.assertLessEqual(len(set(x[a])&set(x[b])),1)
        self.assertLessEqual(np.bincount(x[selected].ravel()).max(),2)

    def test_saved_actual_score_components_filters_and_diversity(self):
        files = sorted(Path('results').glob('round_*/run_summary.json'))
        self.assertTrue(files,'Run predict_top15.py --no-push before integration checks')
        summary = json.loads(files[-1].read_text(encoding='utf-8'))
        table = pd.read_csv(files[-1].parent/'top15.csv')
        self.assertEqual(len(table),15)
        x = table[[f'번호{i}' for i in range(1,7)]].to_numpy(int)
        self.assertEqual(len(set(map(tuple,x))),15)
        self.assertTrue(np.all(np.diff(x,axis=1)>0))
        self.assertTrue(np.all((x>=1)&(x<=45)))
        sr,sn,_ = score_batch(x,summary['rules'],
              pd.read_csv(files[-1].parent/'number_probabilities.csv')['probability'].to_numpy(),
              summary['number_weight_lambda'])
        np.testing.assert_allclose(sr,table['규칙 점수 S_rule'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(sn,table['개별 번호 점수 S_num'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(sr+sn,table['종합 확률 점수 S(X)'],atol=1e-8,rtol=0)
        self.assertTrue(np.all(np.diff(table['종합 확률 점수 S(X)'])<0))
        for a,b in itertools.combinations(x.tolist(),2): self.assertLessEqual(len(set(a)&set(b)),3)
        self.assertLessEqual(np.bincount(x.ravel(),minlength=46).max(),4)
        f = static_features(x)
        core = set(summary['hard_rule_ids'])
        for rule in summary['rules']:
            if rule['rule_id'] in core: self.assertTrue(evaluate_next(x,f,rule['spec']).all())
            if rule['category']=='A':
                self.assertTrue(rule['condition_active'])
                self.assertGreaterEqual(rule['train_n'],100)
                self.assertGreaterEqual(rule['valid_n'],30)
                self.assertGreaterEqual(100*rule['train_k'],55*rule['train_n'])
                self.assertGreaterEqual(100*rule['valid_k'],55*rule['valid_n'])
            else:
                self.assertGreater(rule['lift'],1.05)
                self.assertGreaterEqual(rule['valid_rate'],.55)
            expected_discount = 1-rule['max_abs_correlation'] if rule['max_abs_correlation']>=.5 else 1
            self.assertAlmostEqual(rule['weight'],rule['raw_weight']*expected_discount,places=10)
        self.assertEqual(summary['scored_combinations'],math.comb(45,6))
        self.assertGreater(summary['hard_pass_count'],15)
        self.assertLess(summary['hard_pass_count'],math.comb(45,6))

    def test_discount_correlations_use_resolved_target_predicates(self):
        summary_file = sorted(Path('results').glob('round_*/run_summary.json'))[-1]
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        c = load_context(next(Path('.codex-attachments').rglob('*.xlsx')))
        x = c['x'][c['train']]
        f = static_features(x)
        previous = []
        for r in summary['rules']:
            mask = evaluate_next(x,f,r['spec']).astype(float)
            correlations = [abs(float(np.corrcoef(mask,old)[0,1])) for old in previous
                            if mask.std()>0 and old.std()>0]
            expected = min(1.,max(correlations,default=0.))
            self.assertAlmostEqual(r['max_abs_correlation'],expected,places=10)
            previous.append(mask)


if __name__=='__main__':
    unittest.main()
