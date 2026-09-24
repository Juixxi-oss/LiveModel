"""Synthetic-only checks; no study data, checkpoints or outputs are opened."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedules = load_module('snap_schedules', 'trainer/schedulers.py')
readers = load_module('reader_icc', 'evaluation/reliability/reader_icc.py')
repeats = load_module('intra_rater', 'evaluation/reliability/intra_rater.py')
wide = load_module('wide_repeatability', 'evaluation/reliability/wide_repeatability.py')


class SnapScheduleTests(unittest.TestCase):
    @staticmethod
    def optimizer():
        return torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=5e-4)

    def test_default_keeps_constant_rate(self):
        optimizer = self.optimizer()
        self.assertIsNone(schedules.build_snap_scheduler(optimizer))
        for _ in range(50):
            optimizer.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], 5e-4)

    def test_warmup_then_cosine_and_restart(self):
        optimizer = self.optimizer()
        scheduler = schedules.build_snap_scheduler(optimizer, 'warmup_cosine')
        rates = [optimizer.param_groups[0]['lr']]
        for _ in range(22):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(rates[0], 5e-5)
        self.assertTrue(all(a < b for a, b in zip(rates[:5], rates[1:6])))
        self.assertAlmostEqual(rates[5], 5e-4)
        self.assertTrue(all(a > b for a, b in zip(rates[5:19], rates[6:20])))
        self.assertAlmostEqual(rates[20], 5e-4)
        self.assertTrue(all(1e-7 <= value <= 5e-4 * (1 + 1e-12) for value in rates))
        # A new fold starts with a new optimizer and scheduler.
        next_optimizer = self.optimizer()
        schedules.build_snap_scheduler(next_optimizer, 'warmup_cosine')
        self.assertAlmostEqual(next_optimizer.param_groups[0]['lr'], rates[0])

    def test_zero_warmup_and_invalid_config(self):
        optimizer = self.optimizer()
        scheduler = schedules.build_snap_scheduler(optimizer, 'warmup_cosine', warmup_epochs=0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-4)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]['lr'], 5e-4)
        for settings in [{'warmup_epochs': -1}, {'start_factor': 0}, {'T_0': 0},
                         {'T_mult': 0}, {'eta_min': -1}, {'eta_min': 1}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                schedules.build_snap_scheduler(self.optimizer(), 'warmup_cosine', **settings)


class ReliabilityTests(unittest.TestCase):
    @staticmethod
    def ratings():
        rows = []
        for case, score in enumerate([0.2, 1.1, 2.3, 3.0, 3.8]):
            for reader in ['r1', 'r2', 'r3']:
                rows.append(dict(task='Evaluation', outcome='C', method='HumanAI',
                                 case_id=f'c{case}', patient_id=f'p{case}',
                                 reader_id=reader, y_pred=score))
        return pd.DataFrame(rows)

    def test_identical_raters_have_perfect_agreement(self):
        result = readers.inter_rater_icc_table(self.ratings(), n_bootstrap=0).iloc[0]
        self.assertAlmostEqual(result['icc_absolute_agreement'], 1)
        self.assertAlmostEqual(result['sem_agreement'], 0)
        self.assertAlmostEqual(result['mdc95_agreement'], 0)
        self.assertEqual(result['n_cases'], 5)
        self.assertEqual(result['n_readers'], 3)

    def test_systematic_reader_shift_reduces_agreement_not_consistency(self):
        data = self.ratings()
        data.loc[data.reader_id.eq('r2'), 'y_pred'] += 2
        result = readers.inter_rater_icc_table(data, n_bootstrap=0).iloc[0]
        self.assertLess(result['icc_absolute_agreement'], 1)
        self.assertAlmostEqual(result['icc_consistency'], 1)
        self.assertGreater(result['sem_agreement'], 0)
        self.assertAlmostEqual(result['mdc95_agreement'],
                               1.96 * np.sqrt(2) * result['sem_agreement'])

    def test_fixed_reader_scope_changes_inference_not_point_estimate(self):
        data = self.ratings()
        data.loc[data.reader_id.eq('r2'), 'y_pred'] += 0.5
        random = readers.inter_rater_icc_table(data, n_bootstrap=30, seed=42).iloc[0]
        fixed = readers.inter_rater_icc_table(data, n_bootstrap=30, seed=42,
                                              reader_scope='fixed').iloc[0]
        self.assertAlmostEqual(random['icc_absolute_agreement'],
                               fixed['icc_absolute_agreement'])
        self.assertEqual(fixed['bootstrap_design'], 'patient_cluster_fixed_readers')
        self.assertIn('mixed-effects', fixed['icc_model'])

    def test_duplicate_sessions_and_unknown_reader_are_rejected(self):
        data = self.ratings()
        with self.assertRaises(ValueError):
            readers.inter_rater_icc_table(pd.concat([data, data.iloc[:1]]), 0)
        data.loc[0, 'reader_id'] = 'reader_unknown'
        with self.assertRaises(ValueError):
            readers.inter_rater_icc_table(data, 0)

    def test_multiple_cases_per_patient_are_rejected(self):
        data = self.ratings()
        data.loc[data.patient_id.eq('p1'), 'patient_id'] = 'p0'
        with self.assertRaises(ValueError):
            readers.inter_rater_icc_table(data, 0)

    def test_wide_identical_repeats_have_zero_measurement_error(self):
        data = pd.DataFrame({'a': [0.2, 1.1, 2.3, 3.0], 'b': [0.2, 1.1, 2.3, 3.0]})
        icc = wide.icc_a1_from_two_ratings(data, 'a', 'b')
        sem, mdc = wide.compute_sem_and_mdc(data, 'a', 'b', icc)
        self.assertAlmostEqual(icc, 1)
        self.assertAlmostEqual(sem, 0)
        self.assertAlmostEqual(mdc, 0)

    def test_wide_column_mapping_does_not_reuse_previous_unit(self):
        source_columns = ['grade_1', 'prediction_1.1', 'prediction_2.1',
                          'grade_2', 'prediction_1.2', 'prediction_2.2']
        self.assertEqual(wide.find_model_columns(source_columns, 2),
                         ('grade_2', 'prediction_1.2', 'prediction_2.2'))
        pandas_columns = ['grade_1', 'prediction_1', 'prediction_2',
                          'grade_2', 'prediction_1.1', 'prediction_2.1']
        self.assertEqual(wide.find_model_columns(pandas_columns, 2),
                         ('grade_2', 'prediction_1.1', 'prediction_2.1'))

    def test_wide_agreement_sem_includes_session_shift(self):
        data = pd.DataFrame({'a': [0., 1., 2., 3.], 'b': [2., 3., 4., 5.]})
        icc = wide.icc_a1_from_two_ratings(data, 'a', 'b')
        sem, mdc = wide.compute_sem_and_mdc(data, 'a', 'b', icc)
        self.assertLess(icc, 1)
        self.assertAlmostEqual(sem, np.sqrt(2))
        self.assertAlmostEqual(mdc, 1.96 * 2)

    def test_wide_cross_unit_alignment_requires_case_ids(self):
        source = pd.DataFrame({'case_id': ['c0', 'c1', 'c2'],
                               'a': [0., 1., 2.], 'b': [0., 1., 2.]})
        sub = source.loc[[0, 2]]
        aligned = wide._case_aligned_means(source, sub, 'a', 'b')
        self.assertEqual(list(aligned.index), ['c0', 'c2'])
        with self.assertRaisesRegex(ValueError, 'case_id is required'):
            wide._case_aligned_means(source.drop(columns='case_id'), sub, 'a', 'b')

    @staticmethod
    def repeated_ratings(offset=0):
        rows = []
        for case, score in enumerate([0., 1., 2., 3., 4.]):
            for session, value in [('first', score), ('second', score + offset)]:
                rows.append(dict(task='Evaluation', outcome='C', method='Human',
                                 case_id=f'c{case}', patient_id=f'p{case}',
                                 reader_id='r1', session_id=session, y_pred=value))
        return pd.DataFrame(rows)

    def test_intra_rater_identical_repeats_are_exact(self):
        result = repeats.intra_rater_repeatability_table(
            self.repeated_ratings(), n_bootstrap=0).iloc[0]
        self.assertAlmostEqual(result['icc_absolute_agreement'], 1)
        self.assertAlmostEqual(result['sem_agreement'], 0)
        self.assertAlmostEqual(result['mdc95_agreement'], 0)
        self.assertEqual(result['n_cases_paired'], 5)

    def test_intra_rater_shift_affects_agreement_and_mdc(self):
        result = repeats.intra_rater_repeatability_table(
            self.repeated_ratings(offset=2), n_bootstrap=0).iloc[0]
        self.assertLess(result['icc_absolute_agreement'], 1)
        self.assertAlmostEqual(result['icc_consistency'], 1)
        self.assertAlmostEqual(result['sem_agreement'], np.sqrt(2))
        self.assertAlmostEqual(result['mdc95_agreement'], 1.96 * 2)
        self.assertAlmostEqual(result['mean_session_2_minus_1'], 2)

    def test_intra_rater_rejects_condition_switch_as_retest(self):
        data = self.repeated_ratings()
        data.loc[data.session_id.eq('second'), 'method'] = 'HumanAI'
        with self.assertRaises(ValueError):
            repeats.intra_rater_repeatability_table(data, n_bootstrap=0)

    def test_intra_rater_rejects_duplicate_sessions(self):
        data = self.repeated_ratings()
        with self.assertRaises(ValueError):
            repeats.intra_rater_repeatability_table(pd.concat([data, data.iloc[:1]]), 0)

    def test_bootstrap_is_reproducible_on_synthetic_ratings(self):
        data = self.ratings()
        data.loc[data.reader_id.eq('r2'), 'y_pred'] += np.linspace(0.05, 0.25, 5)
        first = readers.inter_rater_icc_table(data, n_bootstrap=25, seed=42)
        second = readers.inter_rater_icc_table(data, n_bootstrap=25, seed=42)
        pd.testing.assert_frame_equal(first, second)


if __name__ == '__main__':
    unittest.main()
