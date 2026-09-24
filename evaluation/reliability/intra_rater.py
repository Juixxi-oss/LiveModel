"""Test-retest ICC and MDC for genuine repeated ratings by the same reader.

Each analysis stays within one task, outcome, condition, and reader. Sessions
are fixed occasions, so the primary coefficient is two-way mixed-effects,
absolute-agreement, single-measure ICC(A,1). Patient-cluster resampling is
used for confidence intervals; sessions are never resampled or relabeled.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def _repeat_statistics(matrix: np.ndarray) -> dict[str, float]:
    """ANOVA agreement and measurement error for two paired sessions."""
    matrix = np.asarray(matrix, dtype=float)
    n_cases, n_sessions = matrix.shape
    if n_cases < 2 or n_sessions != 2 or not np.isfinite(matrix).all():
        raise ValueError('At least two complete cases and exactly two finite sessions are required')

    grand = matrix.mean()
    case_means = matrix.mean(axis=1)
    session_means = matrix.mean(axis=0)
    ms_case = 2 * np.square(case_means - grand).sum() / (n_cases - 1)
    ms_session = n_cases * np.square(session_means - grand).sum()
    residual = matrix - case_means[:, None] - session_means[None, :] + grand
    ms_error = np.square(residual).sum() / (n_cases - 1)
    denominator = ms_case + ms_error + 2 * (ms_session - ms_error) / n_cases
    icc_agreement = ((ms_case - ms_error) / denominator
                     if denominator > 0 else np.nan)
    consistency_denominator = ms_case + ms_error
    icc_consistency = ((ms_case - ms_error) / consistency_denominator
                       if consistency_denominator > 0 else np.nan)

    # The agreement SEM includes systematic session variation. Using only the
    # SD of paired differences would instead estimate consistency error.
    variance_session = max(float((ms_session - ms_error) / n_cases), 0.0)
    variance_residual = max(float(ms_error), 0.0)
    sem_agreement = float(np.sqrt(variance_session + variance_residual))
    differences = matrix[:, 1] - matrix[:, 0]
    mean_bias = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    return {
        'icc_absolute_agreement': float(icc_agreement),
        'icc_consistency': float(icc_consistency),
        'sem_agreement': sem_agreement,
        'mdc95_agreement': float(1.96 * np.sqrt(2) * sem_agreement),
        'variance_session': variance_session,
        'variance_residual': variance_residual,
        'mean_session_2_minus_1': mean_bias,
        'sd_paired_difference': sd_difference,
        'loa95_lower': float(mean_bias - 1.96 * sd_difference),
        'loa95_upper': float(mean_bias + 1.96 * sd_difference),
    }


def _patient_bootstrap_intervals(matrix: np.ndarray, patient_ids: np.ndarray,
                                 n_bootstrap: int, seed: int) -> dict[str, tuple[float, float]]:
    names = ('icc_absolute_agreement', 'sem_agreement', 'mdc95_agreement')
    if n_bootstrap == 0:
        return {name: (np.nan, np.nan) for name in names}
    patients = pd.unique(patient_ids)
    case_indices = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([case_indices[patient] for patient in sampled])
        if len(indices) < 2:
            continue
        statistics = _repeat_statistics(matrix[indices])
        for name in names:
            values[name].append(statistics[name])
    intervals = {}
    for name in names:
        finite = np.asarray(values[name], dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[name] = (float(np.percentile(finite, 2.5)),
                           float(np.percentile(finite, 97.5))) if len(finite) else (np.nan, np.nan)
    return intervals


def intra_rater_repeatability_table(data: pd.DataFrame, n_bootstrap: int = 5000,
                                    seed: int = 42) -> pd.DataFrame:
    """Calculate separate same-reader, same-condition test-retest statistics.

    Required long columns: task, outcome, method, case_id, patient_id,
    reader_id, session_id, y_pred. The two session IDs must represent actual
    repeat measurements on stable cases, not Human versus HumanAI conditions.
    Cases with only one session are counted but excluded from the paired fit.
    """
    required = ['task', 'outcome', 'method', 'case_id', 'patient_id',
                'reader_id', 'session_id', 'y_pred']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing repeated-rating columns: {missing}')
    if not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError('n_bootstrap must be a non-negative integer')
    data = data.copy()
    if not data['method'].isin(['Human', 'HumanAI', 'AI']).all():
        raise ValueError('method must use the canonical labels Human, HumanAI or AI')
    data = data[data['method'].isin(['Human', 'HumanAI'])].copy()
    if data.empty:
        raise ValueError('Repeated-reading input must contain Human or HumanAI ratings')
    for column in required[:-1]:
        if data[column].isna().any():
            raise ValueError(f'{column} must not be missing')
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq('').any():
            raise ValueError(f'{column} must not be blank')
    if data['reader_id'].isin(['AI', 'reader_unknown']).any():
        raise ValueError('Repeated human ratings require actual reader identifiers')
    data['y_pred'] = pd.to_numeric(data['y_pred'], errors='raise')
    if not np.isfinite(data['y_pred'].to_numpy(dtype=float)).all():
        raise ValueError('Ratings must be finite')
    keys = ['task', 'outcome', 'method', 'reader_id', 'case_id', 'session_id']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate reader/case/session ratings are not independent repeats')

    rows = []
    group_keys = ['task', 'outcome', 'method', 'reader_id']
    for (task, outcome, method, reader_id), group in data.groupby(group_keys, sort=False):
        if group.groupby('case_id')['patient_id'].nunique().gt(1).any():
            raise ValueError('A case_id must refer to exactly one patient')
        sessions = sorted(group['session_id'].unique())
        if len(sessions) != 2:
            raise ValueError(f'{(task, outcome, method, reader_id)} requires exactly two actual sessions')
        wide = group.pivot(index='case_id', columns='session_id', values='y_pred')
        n_cases_total = len(wide)
        wide = wide[sessions].dropna()
        if len(wide) < 2:
            raise ValueError(f'{(task, outcome, method, reader_id)} has fewer than two paired cases')
        patient_by_case = group.groupby('case_id')['patient_id'].first()
        patient_ids = patient_by_case.reindex(wide.index).to_numpy()
        matrix = wide.to_numpy(dtype=float)
        statistics = _repeat_statistics(matrix)
        token = f'{seed}|{task}|{outcome}|{method}|{reader_id}'
        group_seed = (int(seed) + zlib.crc32(token.encode('utf-8'))) % (2**32 - 1)
        intervals = _patient_bootstrap_intervals(matrix, patient_ids, n_bootstrap, group_seed)
        row = dict(task=task, outcome=outcome, method=method, reader_id=reader_id,
                   session_1=sessions[0], session_2=sessions[1],
                   n_cases_total=n_cases_total, n_cases_paired=len(wide),
                   n_patients_paired=len(pd.unique(patient_ids)),
                   icc_model='two-way mixed-effects absolute-agreement single-measure ICC(A,1)',
                   bootstrap_design='patient-cluster; fixed sessions', **statistics)
        for name, (lower, upper) in intervals.items():
            row[f'{name}_ci_lower'] = lower
            row[f'{name}_ci_upper'] = upper
        rows.append(row)
    return pd.DataFrame(rows)
