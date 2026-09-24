"""Inter-rater ICC and measurement error for individual reader ratings.

Ratings are analyzed within each task, outcome, and reading condition.
Reader and patient identifiers determine the case-by-reader matrix and
bootstrap units.
"""
from __future__ import annotations

import zlib
import numpy as np
import pandas as pd


def inter_rater_icc_table(data, n_bootstrap=5000, seed=42, reader_scope='random'):
    """Analyze raw ratings separately by task, outcome and reading condition.

    Input is canonical long data: task, outcome, method (Human/HumanAI/AI),
    case_id, patient_id, reader_id, y_pred. AI rows are excluded because a
    single fixed model is not a sampled human reader. Duplicate reader/case
    rows are rejected: a repeated-rating session must be selected explicitly.
    Each patient contributes one case. Random-reader inference resamples
    patients and readers; fixed-reader inference resamples patients only.
    The ICC(A,1) point formula is the same for both reader scopes.
    """
    required = ['task', 'outcome', 'method', 'case_id', 'patient_id', 'reader_id', 'y_pred']
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing canonical columns: {sorted(missing)}')
    if not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError('n_bootstrap must be a non-negative integer')
    if reader_scope not in {'random', 'fixed'}:
        raise ValueError('reader_scope must be random or fixed')
    data = data.copy()
    if not data['method'].isin(['Human', 'HumanAI', 'AI']).all():
        raise ValueError('method must use the canonical labels Human, HumanAI or AI')
    data = data[data['method'].isin(['Human', 'HumanAI'])].copy()
    if data.empty:
        raise ValueError('Reader-study input must contain Human or HumanAI ratings')
    identifiers = required[:-1]
    if data[identifiers].isna().any().any():
        raise ValueError('Reader/case/group identifiers must not be missing')
    for column in identifiers:
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq('').any():
            raise ValueError(f'{column} must not be blank')
    if data['reader_id'].isin(['AI', 'reader_unknown']).any():
        raise ValueError('Human ratings require actual reader identifiers')
    data['y_pred'] = pd.to_numeric(data['y_pred'], errors='raise')
    if np.isinf(data['y_pred'].to_numpy(dtype=float)).any():
        raise ValueError('Ratings must not contain infinity')
    keys = ['task', 'outcome', 'method', 'case_id', 'reader_id']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate reader/case ratings: select a single session explicitly')
    rows = []
    for (task, outcome, method), group in data.groupby(['task', 'outcome', 'method'], sort=False):
        if group.groupby('case_id')['patient_id'].nunique().max() > 1:
            raise ValueError('A case_id must refer to exactly one patient')
        if group.groupby('patient_id')['case_id'].nunique().max() > 1:
            raise ValueError('Inter-rater bootstrap requires one case per patient')
        matrix = _icc_matrix(group, 'y_pred')
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError('Inter-rater ICC requires at least two complete cases and two readers')
        components = _two_way_random_components(matrix)
        intervals = _bootstrap_icc(
            matrix, n_bootstrap, stable_seed(seed, 'icc', task, outcome, method, 'prediction'),
            resample_readers=(reader_scope == 'random'),
        )
        effects = 'random-effects' if reader_scope == 'random' else 'mixed-effects'
        row = dict(task=task, outcome=outcome, method=method,
                   measurement='prediction', n_cases=matrix.shape[0], n_readers=matrix.shape[1],
                   icc_model=f'two-way {effects} absolute-agreement single-reader ICC(A,1)',
                   reader_scope=reader_scope,
                   bootstrap_design=('crossed_case_reader_bootstrap' if reader_scope == 'random'
                                     else 'patient_cluster_fixed_readers'), **components)
        for metric, (lower, upper) in intervals.items():
            row[f'{metric}_ci_lower'] = lower
            row[f'{metric}_ci_upper'] = upper
        rows.append(row)
    return pd.DataFrame(rows)

def stable_seed(seed: int, *parts: object) -> int:
    token = "|".join([str(seed), *[str(part) for part in parts]])
    return int((int(seed) + zlib.crc32(token.encode("utf-8"))) % (2**32 - 1))


def _two_way_random_components(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    n_cases, n_readers = matrix.shape
    if n_cases < 2 or n_readers < 2:
        return {
            "icc_absolute_agreement": np.nan,
            "icc_consistency": np.nan,
            "ms_case": np.nan,
            "ms_reader": np.nan,
            "ms_error": np.nan,
            "variance_case": np.nan,
            "variance_reader": np.nan,
            "variance_residual": np.nan,
            "total_variance": np.nan,
            "reader_variance_fraction": np.nan,
            "sem_agreement": np.nan,
            "mdc95_agreement": np.nan,
        }
    grand_mean = float(matrix.mean())
    case_means = matrix.mean(axis=1)
    reader_means = matrix.mean(axis=0)
    ss_case = float(n_readers * np.sum((case_means - grand_mean) ** 2))
    ss_reader = float(n_cases * np.sum((reader_means - grand_mean) ** 2))
    residual = matrix - case_means[:, None] - reader_means[None, :] + grand_mean
    ss_error = float(np.sum(residual ** 2))
    ms_case = ss_case / (n_cases - 1)
    ms_reader = ss_reader / (n_readers - 1)
    ms_error = ss_error / ((n_cases - 1) * (n_readers - 1))
    agreement_denominator = ms_case + (n_readers - 1) * ms_error + n_readers * (ms_reader - ms_error) / n_cases
    consistency_denominator = ms_case + (n_readers - 1) * ms_error
    icc_absolute = (ms_case - ms_error) / agreement_denominator if agreement_denominator != 0 else np.nan
    icc_consistency = (ms_case - ms_error) / consistency_denominator if consistency_denominator != 0 else np.nan
    variance_case = max((ms_case - ms_error) / n_readers, 0.0)
    variance_reader = max((ms_reader - ms_error) / n_cases, 0.0)
    variance_residual = max(ms_error, 0.0)
    total_variance = variance_case + variance_reader + variance_residual
    reader_fraction = variance_reader / total_variance if total_variance > 0 else np.nan
    # Error of one rating from a randomly selected reader. The reader component
    # includes systematic differences between readers; ICC(C,1) omits these.
    sem_agreement = float(np.sqrt(variance_reader + variance_residual))
    mdc95_agreement = float(1.96 * np.sqrt(2) * sem_agreement)
    return {
        "icc_absolute_agreement": float(icc_absolute),
        "icc_consistency": float(icc_consistency),
        "ms_case": float(ms_case),
        "ms_reader": float(ms_reader),
        "ms_error": float(ms_error),
        "variance_case": float(variance_case),
        "variance_reader": float(variance_reader),
        "variance_residual": float(variance_residual),
        "total_variance": float(total_variance),
        "reader_variance_fraction": float(reader_fraction),
        "sem_agreement": sem_agreement,
        "mdc95_agreement": mdc95_agreement,
    }


def _icc_matrix(data: pd.DataFrame, value_column: str) -> np.ndarray:
    wide = data.pivot(index="case_id", columns="reader_id", values=value_column).dropna(axis=0, how="any")
    return wide.to_numpy(dtype=float)


def _bootstrap_icc(
    matrix: np.ndarray,
    n_bootstrap: int,
    seed: int,
    resample_readers: bool = True,
) -> dict[str, tuple[float, float]]:
    n_cases, n_readers = matrix.shape
    if n_bootstrap <= 0 or n_cases < 2 or n_readers < 2:
        return {
            "icc_absolute_agreement": (np.nan, np.nan),
            "icc_consistency": (np.nan, np.nan),
            "reader_variance_fraction": (np.nan, np.nan),
            "sem_agreement": (np.nan, np.nan),
            "mdc95_agreement": (np.nan, np.nan),
        }
    rng = np.random.default_rng(seed)
    names = ["icc_absolute_agreement", "icc_consistency", "reader_variance_fraction",
             "sem_agreement", "mdc95_agreement"]
    values = np.full((n_bootstrap, len(names)), np.nan)
    for index in range(n_bootstrap):
        case_index = rng.integers(0, n_cases, n_cases)
        reader_index = (rng.integers(0, n_readers, n_readers)
                        if resample_readers else np.arange(n_readers))
        components = _two_way_random_components(matrix[np.ix_(case_index, reader_index)])
        values[index] = [components[name] for name in names]
    intervals = {}
    for column, name in enumerate(names):
        finite = values[np.isfinite(values[:, column]), column]
        intervals[name] = ((float(np.percentile(finite, 2.5)),
                            float(np.percentile(finite, 97.5)))
                           if len(finite) else (np.nan, np.nan))
    return intervals
