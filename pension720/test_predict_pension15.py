"""연금복권720+ 엔진 테스트. 저장소 루트에서: python -m unittest pension720.test_predict_pension15 -v

원본 엑셀 경로는 환경변수 PENSION_XLSX로 지정하거나 이 폴더에 둔다. 없으면 엑셀이 필요한 테스트는 건너뛴다.
"""
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

try:
    from . import predict_pension15 as eng
except ImportError:
    import predict_pension15 as eng

HERE = Path(__file__).resolve().parent


def _workbook():
    if os.environ.get('PENSION_XLSX'):
        return Path(os.environ['PENSION_XLSX'])
    found = sorted(HERE.glob('*.xlsx'))
    return found[0] if found else None


def _write_book(path, rows):
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = '당첨번호'
    sheet.append(list(eng.HEADERS))
    for row in rows:
        sheet.append(row)
    book.save(path)


def _fake_rows(n, number_as_int=False):
    rng = np.random.default_rng(1)
    start = date(2020, 5, 7)
    rows = []
    for i in range(n):
        main, bonus = (f'{rng.integers(0, 10 ** 6):06d}' for _ in range(2))
        rows.append([i + 1, start + timedelta(weeks=i), int(rng.integers(1, 6)),
                     int(main) if number_as_int else main, bonus])
    return rows


class LoadTest(unittest.TestCase):
    def test_leading_zero_kept_when_excel_stored_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _fake_rows(80, number_as_int=True)
            rows[0][3] = 1234          # 엑셀이 001234를 숫자로 바꾼 경우
            path = Path(tmp) / 'x.xlsx'
            _write_book(path, rows)
            draws = eng.load_draws(path)
            self.assertEqual(''.join(map(str, draws.main[0])), '001234')

    def test_next_round_follows_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.xlsx'
            _write_book(path, _fake_rows(80))
            target = eng.Context(eng.load_draws(path), 80)
            self.assertEqual((target.round, target.date), (81, date(2020, 5, 7) + timedelta(weeks=80)))

    def test_gap_in_rounds_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _fake_rows(80)
            del rows[10]
            path = Path(tmp) / 'x.xlsx'
            _write_book(path, rows)
            with self.assertRaises(ValueError):
                eng.load_draws(path)

    @unittest.skipUnless(_workbook(), '원본 엑셀 없음')
    def test_real_workbook_leading_zeros(self):
        draws = eng.load_draws(_workbook())
        self.assertEqual(int((draws.main[:, 0] == 0).sum()), 32)
        self.assertEqual(int((draws.bonus[:, 0] == 0).sum()), 36)


class WeeklyTest(unittest.TestCase):
    def test_master_csv_loads_with_leading_zeros(self):
        draws = eng.load_draws(eng.DATA_CSV)
        self.assertGreaterEqual(len(draws.rounds), 340)
        self.assertEqual(''.join(map(str, draws.main[23])), '002989')   # 24회 앞자리 0

    def test_prizes_and_chat_seal(self):
        try:
            from . import weekly
        except ImportError:
            import weekly
        self.assertEqual(weekly.prizes(4, '194892', 4, '194892', '288615'), ['1등'])
        self.assertEqual(weekly.prizes(2, '194892', 4, '194892', '288615'), ['2등'])
        self.assertEqual(weekly.prizes(2, '904892', 4, '194892', '288615'), ['4등'])
        self.assertEqual(weekly.prizes(2, '288615', 4, '194892', '288615'), ['보너스'])
        self.assertEqual(weekly.prizes(2, '000000', 4, '194892', '288615'), [])
        sealed = weekly.seal('1:x', '123,456')
        self.assertEqual(weekly.unseal('1:x', sealed), '123,456')
        self.assertIsNone(weekly.unseal('2:y', sealed))


class SelectTest(unittest.TestCase):
    def test_constraints_on_random_scores(self):
        scores = np.random.default_rng(7).normal(size=eng.N_CANDIDATES)
        chosen = eng.select(scores)
        eng.check_selection(chosen, scores)   # 위반이면 예외

    def test_group_heavy_scores_still_cover_all_groups(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=eng.N_CANDIDATES) * 1e-3
        scores[:eng.N_NUMBERS] += 10           # 1조만 압도적으로 높아도
        chosen = eng.select(scores)
        self.assertEqual(sorted({c // eng.N_NUMBERS for c in chosen}), [0, 1, 2, 3, 4])


@unittest.skipUnless(_workbook(), '원본 엑셀 없음')
class NoLookaheadTest(unittest.TestCase):
    """시점 k의 모델이 k 이후 행을 지워도 똑같은지 (이후 정보가 섞이지 않는지)."""

    @classmethod
    def setUpClass(cls):
        cls.draws = eng.load_draws(_workbook())
        cls.space = eng.Space()
        cls.k = len(cls.draws.rounds) - 23   # 롤링 구간 안쪽 한 시점

    def _models(self, draws, index, setting):
        out = []
        for stream in (eng.MAIN, eng.BONUS):
            catalog = eng.build_catalog(draws, stream)
            fold = eng.RuleFold(catalog, eng.HistoryIndex(catalog), eng.Context(draws, index), self.space)
            out.append(fold.fit(setting))
        return out

    def test_truncated_history_gives_same_model_and_scores(self):
        setting = eng.SETTINGS[0]
        full = self._models(self.draws, self.k, setting)
        cut = self._models(self.draws.truncated(self.k), self.k, setting)
        ids = eng.rank_sample(self.draws, self.k)
        for a, b in zip(full, cut):
            self.assertEqual([(r.rule.formula, r.adjusted) for r in a.rules],
                             [(r.rule.formula, r.adjusted) for r in b.rules])
            np.testing.assert_array_equal(a.score(ids), b.score(ids))
        # 표본 채점과 전수 채점이 같은 후보에 같은 점수를 주는지
        for model in full:
            np.testing.assert_allclose(model.score(ids), model.full_score()[ids], rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
