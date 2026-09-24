"""ICC, SEM, and MDC calculations for paired wide-format ratings.

Each numbered unit has a reference grade and two paired predictions. The
case_id column aligns distinct units for between-unit agreement.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def bootstrap(
        data: list, num_bootstrap_samples: int = 1000, confidence_level: float = 0.95
) -> tuple[float, float]:
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return np.nan, np.nan
    bootstrap_means = np.zeros(num_bootstrap_samples, dtype=float)
    for i in range(num_bootstrap_samples):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(resampled_data)
    alpha = (1 - confidence_level) / 2
    lower_bound, upper_bound = np.percentile(
        bootstrap_means, [alpha * 100, (1 - alpha) * 100]
    )
    return float(lower_bound), float(upper_bound)

def _two_session_mean_squares(df_pair, col1, col2):
    paired = df_pair[[col1, col2]].apply(pd.to_numeric, errors='raise').dropna()
    if len(paired) < 2:
        return len(paired), np.nan, np.nan, np.nan
    matrix = paired.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Repeated measurements must be finite')
    n = len(matrix)
    grand = matrix.mean()
    case_means = matrix.mean(axis=1)
    session_means = matrix.mean(axis=0)
    ms_case = 2 * np.square(case_means - grand).sum() / (n - 1)
    ms_session = n * np.square(session_means - grand).sum()
    residual = matrix - case_means[:, None] - session_means[None, :] + grand
    ms_error = np.square(residual).sum() / (n - 1)
    return n, float(ms_case), float(ms_session), float(ms_error)


def icc_a1_from_two_ratings(df_pair, col1, col2):
    """
    绝对一致 ICC(A,1)：两因素混合、绝对一致、单次评分
    适配同一模型的两次重复测量（组内一致性）
    McGraw & Wong (1996) ANOVA 公式：
      ICC(A,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)
    R=subjects, C=ratings, k=2
    """
    n, ms_r, ms_c, ms_e = _two_session_mean_squares(df_pair, col1, col2)
    if n < 2:
        return np.nan
    denom = ms_r + ms_e + 2 * (ms_c - ms_e) / n
    if denom <= 0:
        return np.nan
    return (ms_r - ms_e) / denom


def icc_21_from_matrix(Y):
    """
    绝对一致 ICC(2,1)：两因素随机、绝对一致、单次评分（组间一致性）
    输入 Y 为 (n_subjects x k_raters) 的评分矩阵。
    """
    Y = pd.DataFrame(Y).astype(float)
    Y = Y.dropna(axis=0, how="any")
    n, k = Y.shape
    if n < 2 or k < 2:
        return np.nan

    Yv = Y.to_numpy()
    grand_mean = Yv.mean()
    subj_means = Yv.mean(axis=1)
    rater_means = Yv.mean(axis=0)

    ss_subj  = k * ((subj_means - grand_mean) ** 2).sum()
    ss_rater = n * ((rater_means - grand_mean) ** 2).sum()
    residual = Yv - subj_means[:, None] - rater_means[None, :] + grand_mean
    ss_error = np.square(residual).sum()

    ms_r = ss_subj / (n - 1)
    ms_c = ss_rater / (k - 1)
    ms_e = ss_error / ((n - 1) * (k - 1))

    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if denom <= 0:
        return np.nan
    return (ms_r - ms_e) / denom


def icc_21_two_raters(x, y):
    """两个模型的 ICC(2,1)。"""
    D = pd.DataFrame({"r1": x, "r2": y}).dropna()
    if len(D) < 2:
        return np.nan
    return icc_21_from_matrix(D.to_numpy())


def find_model_columns(columns, i):
    """
    Match one unit using a single file-wide suffix convention.

    Input files may have suffixes .1, .2, ... for units 1, 2, ... . If
    pandas supplied unsuffixed first columns, later units use .1, .2, ... .
    One convention is chosen for the entire table so a unit cannot reuse
    another unit's prediction columns.
    """
    grade_col = f"grade_{i}"
    if grade_col not in columns:
        return None, None, None
    first_plain = 'prediction_1' in columns
    second_plain = 'prediction_2' in columns
    if first_plain != second_plain:
        raise ValueError('Incomplete unsuffixed prediction pair')
    suffix = ('' if i == 1 else f'.{i - 1}') if first_plain else f'.{i}'
    p1, p2 = f'prediction_1{suffix}', f'prediction_2{suffix}'
    if p1 not in columns or p2 not in columns:
        raise ValueError(f'Missing exact prediction pair for {grade_col}: {p1}, {p2}')
    return grade_col, p1, p2


def compute_sem_and_mdc(df_pair, col1, col2, icc=None, ci_z=1.96):
    """
    Agreement SEM from the session and residual variance components.

    Stacking both sessions into one SD is not the pooled-within-session SD.
    The supplied ICC is checked against the same paired data.
    """
    n, _, ms_session, ms_error = _two_session_mean_squares(df_pair, col1, col2)
    if n < 2:
        return np.nan, np.nan
    if icc is not None and not np.isclose(icc, icc_a1_from_two_ratings(df_pair, col1, col2), equal_nan=True):
        raise ValueError('ICC and repeated-measurement data do not match')
    sem = np.sqrt(max((ms_session - ms_error) / n, 0.0) + max(ms_error, 0.0))
    mdc95_icc = ci_z * np.sqrt(2) * sem
    return float(sem), float(mdc95_icc)


def _case_aligned_means(df, sub, col1, col2):
    """Align repeated ratings from distinct units by their shared case ID."""
    if 'case_id' not in df:
        raise ValueError('case_id is required to align ratings across units')
    case_ids = df.loc[sub.index, 'case_id']
    if case_ids.isna().any() or case_ids.astype(str).str.strip().eq('').any():
        raise ValueError('case_id must be populated for every wide-table row')
    case_ids = case_ids.astype(str).str.strip()
    if case_ids.duplicated().any():
        raise ValueError('case_id must be unique in a wide table')
    return pd.Series(sub[[col1, col2]].mean(axis=1).to_numpy(), index=case_ids)


def run_wide_repeatability(input_dir, output_dir, seed=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    expected = [input_dir / f"{prefix}_{grade}.xlsx"
                for prefix in ("machine_efficientnet_b0", "human")
                for grade in ("C", "N", "P", "BCVA")]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing wide-table input files: " + ", ".join(missing))
    # Never overwrite or append to an existing results directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    if seed is not None:
        np.random.seed(seed)
    _excel_path = str(input_dir)
    n_models = 10
    ci_z = 1.96
    _out_xlsx = str(output_dir)

    for grade_type in ['C', 'N', 'P', 'BCVA']:
        excel_path = f'{_excel_path}/machine_efficientnet_b0_{grade_type}.xlsx'
        out_xlsx = f"{_out_xlsx}/machine_{grade_type}.xlsx"
        df = pd.read_excel(excel_path)
        if 'case_id' not in df:
            raise ValueError(f'{excel_path}: case_id is required for between-unit ICC')
        per_model_rows = []
        model_mean_scores = {}  # 用于后续组间 ICC（每模型的“平均评分”）
        for i in range(1, n_models + 1):
            grade_col, p1_col, p2_col = find_model_columns(df.columns, i)
            model_name = f"model_{i}"
            if grade_col is None:
                per_model_rows.append({
                    "model": model_name, "n": 0,
                    "ICC_within_A1": np.nan,
                    "SEM": np.nan,
                    "MDC95_ICC": np.nan,
                    "MDC95_diff": np.nan,
                    "mean_session_2_minus_1": np.nan,
                    "loa95_lower": np.nan,
                    "loa95_upper": np.nan,
                })
                continue

            sub = df[[grade_col, p1_col, p2_col]].astype(float).dropna()
            n = len(sub)

            if n < 2:
                per_model_rows.append({
                    "model": model_name, "n": n,
                    "ICC_within_A1": np.nan,
                    "SEM": np.nan,
                    "MDC95_ICC": np.nan,
                    "MDC95_diff": np.nan,
                    "mean_session_2_minus_1": np.nan,
                    "loa95_lower": np.nan,
                    "loa95_upper": np.nan,
                })
                continue
            icc_within = icc_a1_from_two_ratings(sub, p1_col, p2_col)
            sem, mdc95_icc = compute_sem_and_mdc(
                sub, p1_col, p2_col, icc_within, ci_z=ci_z
            )
            diff = (sub[p2_col] - sub[p1_col]).to_numpy()
            sd_diff = diff.std(ddof=1)
            mean_diff = float(diff.mean())
            mdc95_diff = ci_z * sd_diff
            per_model_rows.append({
                "model": model_name, "n": n,
                "ICC_within_A1": icc_within,
                "SEM": sem,
                "MDC95_ICC": mdc95_icc,
                "MDC95_diff": mdc95_diff,
                "mean_session_2_minus_1": mean_diff,
                "loa95_lower": mean_diff - mdc95_diff,
                "loa95_upper": mean_diff + mdc95_diff,
            })
            # 供组间 ICC 用：该模型对每个样本的“单一评分”取两次的平均值
            aligned_means = _case_aligned_means(df, sub, p1_col, p2_col)
            model_mean_scores[model_name] = aligned_means
        per_model_df = pd.DataFrame(per_model_rows)
        per_model_df['between_ICC_avg_pairwise'] = np.nan
        aligned = pd.DataFrame(model_mean_scores)  # 每列是一个模型的均分
        overall_between_icc_21 = icc_21_from_matrix(aligned) if len(model_mean_scores) >= 2 else np.nan
        pairwise_rows = []
        models = list(model_mean_scores.keys())
        for i, mi in enumerate(models):
            xi = aligned[mi]
            iccs = []
            for j, mj in enumerate(models):
                if j == i:
                    continue
                xj = aligned[mj]
                icc_ij = icc_21_two_raters(xi, xj)
                if not pd.isna(icc_ij):
                    iccs.append(icc_ij)
                pairwise_rows.append({"model_i": mi, "model_j": mj, "ICC21_pair": icc_ij})
            avg_pair = np.mean(iccs) if len(iccs) > 0 else np.nan
            per_model_df.loc[per_model_df["model"] == mi, "between_ICC_avg_pairwise"] = avg_pair
        pairwise_df = pd.DataFrame(pairwise_rows)
        summary_rows = []
        numeric_cols = per_model_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            vals = per_model_df[col].dropna().to_numpy()
            n = len(vals)
            if n == 0:
                mean_val = std_val = ci_lo = ci_hi = np.nan
            else:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if n > 1 else np.nan
                ci_lo, ci_hi = bootstrap(vals.tolist(), num_bootstrap_samples=5000, confidence_level=0.95)
            summary_rows.append({
                "metric": col,
                "n": n,
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi
            })
        summary_df = pd.DataFrame(summary_rows)
        with pd.ExcelWriter(out_xlsx) as writer:
            per_model_df.to_excel(writer, sheet_name="per_model", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            pairwise_df.to_excel(writer, sheet_name="pairwise", index=False)
        print("overall_between_ICC21:", overall_between_icc_21)
        print(f"Saved all outputs to: {out_xlsx}")
        print()

    n_models = 3
    for grade_type in ['C', 'N', 'P', 'BCVA']:
        excel_path = f'{_excel_path}/human_{grade_type}.xlsx'
        out_xlsx = f"{_out_xlsx}/human_{grade_type}.xlsx"
        df = pd.read_excel(excel_path)
        if 'case_id' not in df:
            raise ValueError(f'{excel_path}: case_id is required for between-unit ICC')
        per_model_rows = []
        model_mean_scores = {}  # 用于后续组间 ICC（每模型的“平均评分”）
        for i in range(1, n_models + 1):
            grade_col, p1_col, p2_col = find_model_columns(df.columns, i)
            model_name = f"model_{i}"
            if grade_col is None:
                per_model_rows.append({
                    "model": model_name, "n": 0,
                    "ICC_within_A1": np.nan,
                    "SEM": np.nan,
                    "MDC95_ICC": np.nan,
                    "MDC95_diff": np.nan,
                    "mean_session_2_minus_1": np.nan,
                    "loa95_lower": np.nan,
                    "loa95_upper": np.nan,
                })
                continue

            sub = df[[grade_col, p1_col, p2_col]].astype(float).dropna()
            n = len(sub)

            if n < 2:
                per_model_rows.append({
                    "model": model_name, "n": n,
                    "ICC_within_A1": np.nan,
                    "SEM": np.nan,
                    "MDC95_ICC": np.nan,
                    "MDC95_diff": np.nan,
                    "mean_session_2_minus_1": np.nan,
                    "loa95_lower": np.nan,
                    "loa95_upper": np.nan,
                })
                continue
            icc_within = icc_a1_from_two_ratings(sub, p1_col, p2_col)
            sem, mdc95_icc = compute_sem_and_mdc(
                sub, p1_col, p2_col, icc_within, ci_z=ci_z
            )
            diff = (sub[p2_col] - sub[p1_col]).to_numpy()
            sd_diff = diff.std(ddof=1)
            mean_diff = float(diff.mean())
            mdc95_diff = ci_z * sd_diff
            per_model_rows.append({
                "model": model_name, "n": n,
                "ICC_within_A1": icc_within,
                "SEM": sem,
                "MDC95_ICC": mdc95_icc,
                "MDC95_diff": mdc95_diff,
                "mean_session_2_minus_1": mean_diff,
                "loa95_lower": mean_diff - mdc95_diff,
                "loa95_upper": mean_diff + mdc95_diff,
            })
            # 供组间 ICC 用：该模型对每个样本的“单一评分”取两次的平均值
            aligned_means = _case_aligned_means(df, sub, p1_col, p2_col)
            model_mean_scores[model_name] = aligned_means
        per_model_df = pd.DataFrame(per_model_rows)
        per_model_df['between_ICC_avg_pairwise'] = np.nan
        aligned = pd.DataFrame(model_mean_scores)  # 每列是一个模型的均分
        overall_between_icc_21 = icc_21_from_matrix(aligned) if len(model_mean_scores) >= 2 else np.nan
        pairwise_rows = []
        models = list(model_mean_scores.keys())
        for i, mi in enumerate(models):
            xi = aligned[mi]
            iccs = []
            for j, mj in enumerate(models):
                if j == i:
                    continue
                xj = aligned[mj]
                icc_ij = icc_21_two_raters(xi, xj)
                if not pd.isna(icc_ij):
                    iccs.append(icc_ij)
                pairwise_rows.append({"model_i": mi, "model_j": mj, "ICC21_pair": icc_ij})
            avg_pair = np.mean(iccs) if len(iccs) > 0 else np.nan
            per_model_df.loc[per_model_df["model"] == mi, "between_ICC_avg_pairwise"] = avg_pair
        pairwise_df = pd.DataFrame(pairwise_rows)
        summary_rows = []
        numeric_cols = per_model_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            vals = per_model_df[col].dropna().to_numpy()
            n = len(vals)
            if n == 0:
                mean_val = std_val = ci_lo = ci_hi = np.nan
            else:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if n > 1 else np.nan
                ci_lo, ci_hi = bootstrap(vals.tolist(), num_bootstrap_samples=5000, confidence_level=0.95)
            summary_rows.append({
                "metric": col,
                "n": n,
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi
            })
        summary_df = pd.DataFrame(summary_rows)
        with pd.ExcelWriter(out_xlsx) as writer:
            per_model_df.to_excel(writer, sheet_name="per_model", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            pairwise_df.to_excel(writer, sheet_name="pairwise", index=False)
        print("overall_between_ICC21:", overall_between_icc_21)
        print(f"Saved all outputs to: {out_xlsx}")
        print()
