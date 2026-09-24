"""Synthetic-only checks; no study data, checkpoints or outputs are opened."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))

import analysis_core
import analysis_human
import analysis_tasks
import analysis_tests


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedules = load_module('training_util', 'trainer/util.py')
readers = load_module('reader_icc', 'evaluation/reliability/reader_icc.py')
repeats = load_module('intra_rater', 'evaluation/reliability/intra_rater.py')
wide = load_module('wide_repeatability', 'evaluation/reliability/wide_repeatability.py')


class TrainingScheduleTests(unittest.TestCase):
    @staticmethod
    def optimizer():
        return torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=5e-4)

    def test_warmup_then_cosine_and_restart(self):
        optimizer = self.optimizer()
        scheduler = schedules.build_warmup_cosine_scheduler(optimizer)
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
        schedules.build_warmup_cosine_scheduler(next_optimizer)
        self.assertAlmostEqual(next_optimizer.param_groups[0]['lr'], rates[0])

    def test_zero_warmup_and_invalid_config(self):
        optimizer = self.optimizer()
        scheduler = schedules.build_warmup_cosine_scheduler(optimizer, warmup_epochs=0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-4)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]['lr'], 5e-4)
        for settings in [{'warmup_epochs': -1}, {'start_factor': 0}, {'T_0': 0},
                         {'T_mult': 0}, {'eta_min': -1}, {'eta_min': 1}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                schedules.build_warmup_cosine_scheduler(self.optimizer(), **settings)


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


class AnalysisTests(unittest.TestCase):
    def test_pooled_performance_preserves_each_center_size(self):
        rows = []
        for center, count, error in [('S01', 2, 0.), ('S09', 3, 10.)]:
            for patient in range(count):
                for visit in ['V1', 'V2']:
                    rows.append(dict(dataset_name='external', center=center,
                                     grade_type='C', model_name='LiveModel',
                                     patient_id=f'{center}{patient}',
                                     SeriesID=f'{center}{patient}_OD_{visit}',
                                     visit=visit, y_true=0., y_pred=error))
        result = analysis_core.performance_table(
            pd.DataFrame(rows), ['dataset_name', 'center', 'grade_type', 'model_name'],
            n_bootstrap=100, seed=42)
        self.assertEqual(set(result.center), {'S01', 'S09', 'pooled'})
        pooled = result.loc[result.center.eq('pooled') & result.metric.eq('MAE')].iloc[0]
        self.assertEqual((pooled['n'], pooled['n_clusters']), (10, 5))
        self.assertEqual((pooled['estimate'], pooled['ci_lower'], pooled['ci_upper']),
                         (6., 6., 6.))

    def test_human_ai_mse_uses_one_twelve_comparison_holm_family(self):
        rows = []
        for grade in ['C', 'N', 'P', 'BCVA']:
            for patient in range(4):
                base = dict(dataset_name='external', eval_mode='human_ai',
                            grade_type=grade, center='S01', patient_id=f'S01{patient}',
                            eye_id=f'S01{patient}_OD', SeriesID=f'S01{patient}_OD_V9',
                            visit='V9', target_visit='V9', y_true=float(patient))
                rows.append({**base, 'model_name': 'Model alone',
                             'reader_id': '', 'y_pred': patient + 0.1})
                for reader in ['R1', 'R2']:
                    rows.append({**base, 'model_name': 'Human alone',
                                 'reader_id': reader, 'y_pred': patient + 0.4})
                    rows.append({**base, 'model_name': 'Human+AI',
                                 'reader_id': reader, 'y_pred': patient + 0.2})
        result = analysis_tests.human_ai_pairwise_tests(pd.DataFrame(rows),
                                                        n_bootstrap=30, seed=42)
        self.assertEqual(len(result), 12)
        self.assertEqual(set(result.metric), {'MSE'})
        self.assertEqual(result.grade_type.value_counts().to_dict(),
                         {'C': 3, 'N': 3, 'P': 3, 'BCVA': 3})
        np.testing.assert_allclose(result.p_holm,
                                   analysis_tests.holm_adjust(result.p_value))

    def test_human_ai_outputs_are_separate_by_task_and_average_readers(self):
        rows = []
        for task in ['assessment', 'progression']:
            for grade in ['C', 'N', 'P', 'BCVA']:
                for patient in range(4):
                    center = 'S01' if patient < 2 else 'S09'
                    base = dict(dataset_name='external', eval_mode='human_ai',
                                study_task=task, grade_type=grade, center=center,
                                patient_id=f'{center}{patient}', eye_id=f'{center}{patient}_OD',
                                SeriesID=f'{center}{patient}_OD_V9', visit='V9',
                                target_visit='V9', y_true=0.)
                    rows.append({**base, 'model_name': 'AI alone',
                                 'reader_id': '', 'y_pred': 0.1})
                    rows.append({**base, 'model_name': 'Human alone',
                                 'reader_id': 'R1', 'y_pred': 1.})
                    rows.append({**base, 'model_name': 'Human+AI',
                                 'reader_id': 'R1', 'y_pred': 0.2})
                    if patient == 0:
                        rows.append({**base, 'model_name': 'Human alone',
                                     'reader_id': 'R2', 'y_pred': 3.})
                        rows.append({**base, 'model_name': 'Human+AI',
                                     'reader_id': 'R2', 'y_pred': 0.3})
        data = pd.DataFrame(rows)
        perf = analysis_tests.human_ai_performance_table(data, n_bootstrap=30, seed=42)
        self.assertEqual(len(perf), 216)
        self.assertEqual(set(perf.center), {'S01', 'S09', 'pooled'})
        human_mse = perf.loc[(perf.study_task == 'assessment') &
                             (perf.grade_type == 'C') &
                             (perf.model_name == 'Human alone') &
                             (perf.center == 'pooled') &
                             (perf.metric == 'MSE')].iloc[0]
        self.assertAlmostEqual(human_mse.estimate, 5.)
        self.assertEqual((human_mse.n_readers, human_mse.n_patients), (2, 4))
        self.assertTrue(np.isfinite(human_mse.ci_lower))
        ai_mse = perf.loc[(perf.study_task == 'assessment') &
                          (perf.grade_type == 'C') &
                          (perf.model_name == 'AI alone') &
                          (perf.center == 'pooled') &
                          (perf.metric == 'MSE')].iloc[0]
        self.assertEqual(ai_mse.bootstrap_design, 'patient_cluster')
        self.assertAlmostEqual(ai_mse.estimate, 0.01)
        pair = analysis_tests.human_ai_pairwise_tests(data, n_bootstrap=30, seed=42)
        self.assertEqual(pair.study_task.value_counts().to_dict(),
                         {'assessment': 12, 'progression': 12})
        for _, group in pair.groupby('study_task'):
            np.testing.assert_allclose(group.p_holm,
                                       analysis_tests.holm_adjust(group.p_value))

    def test_human_tables_reach_external_performance_output(self):
        truth = pd.DataFrame([
            dict(SeriesID='S01001_OD_V9', grade_type=grade, y_true=0.)
            for grade in ['C', 'N', 'P', 'BCVA']
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ratings.csv'
            pd.DataFrame([dict(SeriesID='S01001_OD_V9', C=1., N=1., P=1., BCVA=1.)]).to_csv(path, index=False)
            human = analysis_human.load_human_tables(
                [f'assessment:Human alone:R1:{path}'], truth
            )
            self.assertEqual(set(human.dataset_name), {'external'})
            self.assertEqual(set(human.study_task), {'assessment'})
            selected = analysis_tasks.subset_task(
                analysis_tasks.select_dataset(human, 'external'), 'human_ai'
            )
            self.assertEqual(len(selected), 4)
            analysis_tasks.run_task(
                human, 'human_ai', 'external', directory, 'sample',
                n_bootstrap=10, seed=42, run_gee=False, run_mixedlm=False,
            )
            output = pd.read_csv(Path(directory) / 'sample_human_ai_external_performance.csv')
            self.assertEqual(len(output), 12)
            self.assertEqual(set(output.model_name), {'Human alone'})

    def test_human_table_preserves_reader_ids_in_each_row(self):
        truth = pd.DataFrame([dict(SeriesID='S01001_OD_V9', grade_type='C', y_true=0.)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ratings.csv'
            pd.DataFrame([
                dict(SeriesID='S01001_OD_V9', study_task='assessment', reader_id='R1', C=0.),
                dict(SeriesID='S01001_OD_V9', study_task='assessment', reader_id='R2', C=2.),
            ]).to_csv(path, index=False)
            for spec in [f'Human alone:{path}', f'assessment:Human alone:R0:{path}']:
                with self.subTest(spec=spec):
                    human = analysis_human.load_human_tables([spec], truth)
                    self.assertEqual(set(human.reader_id), {'R1', 'R2'})
                    performance = analysis_tests.human_ai_performance_table(human, n_bootstrap=0)
                    mse = performance.loc[performance.metric.eq('MSE')].iloc[0]
                    self.assertEqual(mse.n_readers, 2)
                    self.assertAlmostEqual(mse.estimate, 2.)

    def test_short_human_specs_require_task_and_keep_task_families_separate(self):
        grades = ['C', 'N', 'P', 'BCVA']
        series_ids = [f'S0100{i}_OD_V9' for i in range(1, 5)]
        truth = pd.DataFrame([
            dict(SeriesID=sid, grade_type=grade, y_true=0.)
            for sid in series_ids for grade in grades
        ])
        with tempfile.TemporaryDirectory() as directory:
            specs = []
            for task in ['assessment', 'progression']:
                for mode, offset in [('Human alone', 0.4), ('Human+AI', 0.2), ('AI alone', 0.1)]:
                    path = Path(directory) / f'{task}_{mode.replace(" ", "_")}.csv'
                    pd.DataFrame([
                        dict(SeriesID=sid, study_task=task,
                             **{grade: offset + 0.05 * i for grade in grades})
                        for i, sid in enumerate(series_ids)
                    ]).to_csv(path, index=False)
                    specs.append(f'{mode}:R1:{path}')

            human = analysis_human.load_human_tables(specs, truth)
            pair = analysis_tests.human_ai_pairwise_tests(human, n_bootstrap=20)
            self.assertEqual(pair.study_task.value_counts().to_dict(),
                             {'assessment': 12, 'progression': 12})
            for _, group in pair.groupby('study_task'):
                np.testing.assert_allclose(group.p_holm,
                                           analysis_tests.holm_adjust(group.p_value))

            missing_task = Path(directory) / 'missing_task.csv'
            pd.DataFrame([dict(SeriesID=series_ids[0], reader_id='R1', C=1.)]
                         ).to_csv(missing_task, index=False)
            for spec in [f'Human alone:{missing_task}',
                         f'Human alone:R1:{missing_task}']:
                with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, 'study_task'):
                    analysis_human.load_human_tables([spec], truth)

    def test_regression_ci_and_adjustment_with_demographics(self):
        wide = pd.DataFrame([
            dict(center='S01' if i < 24 else 'S09', patient_id=f'p{i}',
                 subject_id=f'p{i}', eye_id=f'p{i}_OD', visit_num=i % 3,
                 age=50 + i % 15, sex=['F', 'M'][(i // 2) % 2],
                 y_true=float(i % 5), A=float(i % 5) + 0.3 + 0.05 * (i % 3),
                 B=float(i % 5) + 0.8 + 0.04 * (i % 4))
            for i in range(48)
        ])
        for test in [analysis_tests.gee_pairwise_test,
                     analysis_tests.mixedlm_pairwise_test]:
            result = test(wide, 'A', 'B', 'squared_error')
            self.assertEqual(result['status'], 'ok')
            self.assertLess(result['ci_lower'], result['diff_A_minus_B'])
            self.assertLess(result['diff_A_minus_B'], result['ci_upper'])
            self.assertAlmostEqual(result['ci_lower'], -result['coef_ci_upper'])
        formula = analysis_tests.build_regression_formula(
            wide.assign(model_label='A'), 'squared_error')
        self.assertIn('age', formula)
        self.assertIn('C(sex)', formula)
        with self.assertRaisesRegex(ValueError, 'age and sex'):
            analysis_tests.gee_pairwise_test(wide.drop(columns='age'), 'A', 'B', 'squared_error')
        with self.assertRaisesRegex(ValueError, 'age and sex'):
            analysis_tests.mixedlm_pairwise_test(
                wide.assign(age=np.nan), 'A', 'B', 'squared_error')
        comparisons = pd.DataFrame([
            dict(grade_type=grade, variant_a='A', variant_b='B', metric=metric,
                 p_value=p)
            for metric, values in [('MAE', [0.01, 0.02, 0.03, 0.04]),
                                   ('MSE', [0.05, 0.06, 0.07, 0.08])]
            for grade, p in zip(['C', 'N', 'P', 'BCVA'], values)
        ])
        adjusted = analysis_tests.holm_adjust_by_comparison(comparisons)
        for metric, group in adjusted.groupby('metric'):
            np.testing.assert_allclose(group.p_holm,
                                       analysis_tests.holm_adjust(group.p_value))

    def test_mixedlm_includes_subject_intercept_and_eye_component(self):
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(4)
        rows = []
        for subject in range(24):
            subject_effect = rng.normal(0, 0.4)
            for eye in ['OD', 'OS']:
                eye_effect = rng.normal(0, 0.2)
                for visit in [1, 2]:
                    rows.append(dict(
                        center='S01' if subject < 12 else 'S09',
                        patient_id=f'p{subject}', subject_id=f'p{subject}',
                        eye_id=f'p{subject}_{eye}', visit_num=visit,
                        age=50 + subject % 15, sex='F' if subject % 2 else 'M',
                        y_true=0., A=subject_effect + eye_effect + rng.normal(0, 0.2),
                        B=subject_effect + eye_effect + 0.6 + rng.normal(0, 0.2),
                    ))
        with patch('statsmodels.formula.api.mixedlm', wraps=smf.mixedlm) as fit:
            result = analysis_tests.mixedlm_pairwise_test(
                pd.DataFrame(rows), 'A', 'B', 'squared_error'
            )
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(fit.call_args.kwargs['re_formula'], '1')
        self.assertEqual(fit.call_args.kwargs['vc_formula'], {'eye': '0 + C(eye_id)'})

    def test_demographics_join_requires_complete_values(self):
        predictions = pd.DataFrame({'SeriesID': ['S01001_OD_V1', 'S01002_OD_V1'],
                                    'y_pred': [0.1, 0.2]})
        demographics = pd.DataFrame({'SeriesID': predictions.SeriesID,
                                     'age': [60, 65], 'sex': ['女', '男']})
        joined = analysis_core.attach_demographics(predictions, demographics)
        self.assertEqual(joined.age.tolist(), [60, 65])
        self.assertEqual(joined.sex.tolist(), ['女', '男'])
        with self.assertRaisesRegex(ValueError, 'Age and sex'):
            analysis_core.attach_demographics(predictions, demographics.assign(age=[60, np.nan]))


if __name__ == '__main__':
    unittest.main()
