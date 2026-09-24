import warnings
from typing import Optional

import numpy as np
import pandas as pd

from analysis_core import ensure_columns, metric_value, METRICS


def make_variant(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = ensure_columns(df).copy()
    # Variant is what gets paired/tested.
    variant = df["model_name"].astype(str)
    if task in {"predictor_length", "predictor_tag", "predictor_all"}:
        # Compare padding conditions within each model.
        parts = []
        for _, r in df.iterrows():
            if r.get("eval_mode") == "predictor_padding_length":
                parts.append(f"{r['model_name']}|length={int(r['padding_length']) if pd.notna(r['padding_length']) else 'NA'}")
            elif r.get("eval_mode") == "predictor_padding_tag":
                tag = r.get("padding_tag", "")
                av = r.get("available_visits", "")
                parts.append(f"{r['model_name']}|tag={tag}|{av}")
            else:
                parts.append(str(r["model_name"]))
        variant = pd.Series(parts, index=df.index)
    elif task in {"evaluator_padding", "single_vs_multi_evaluator"}:
        variant = df.apply(lambda r: f"{r['model_name']}|length={int(r['padding_length']) if pd.notna(r['padding_length']) else 'NA'}", axis=1)
    elif task in {"single_vs_multi_predictor"}:
        variant = df.apply(lambda r: f"{r['model_name']}|{r['eval_mode']}|length={int(r['padding_length']) if pd.notna(r['padding_length']) else 'NA'}|tag={r.get('padding_tag','')}|{r.get('available_visits','')}", axis=1)
    else:
        variant = df["model_name"].astype(str)
    df["variant"] = variant
    return df


def build_pairwise_wide(df: pd.DataFrame, id_cols: Optional[list[str]] = None) -> pd.DataFrame:
    df = ensure_columns(df).dropna(subset=["y_true", "y_pred"]).copy()
    if "variant" not in df.columns:
        df["variant"] = df["model_name"]
    if id_cols is None:
        id_cols = [
            "dataset_name",
            "grade_type",
            "SeriesID",
            "patient_id",
            "subject_id",
            "eye_id",
            "center",
            "visit",
            "visit_num",
            "target_visit",
            "target_visit_num",
            "age",
            "sex",
        ]
        # Include condition columns when present so comparisons are within same scenario.
        for c in [
            "padding_length", "padding_tag", "padding_mode", "padding_value",
            "available_visits", "mask_pattern", "repeat_idx"
        ]:
            if c in df.columns and df[c].notna().any() and not (df[c].astype(str).replace("nan", "").eq("").all()):
                id_cols.append(c)
    id_cols = [c for c in id_cols if c in df.columns]
    keep = id_cols + ["variant", "y_true", "y_pred"]
    tmp = df[keep].copy()
    # If duplicate rows exist for same variant/case, average predictions.
    tmp = tmp.groupby(id_cols + ["variant"], dropna=False).agg({"y_true": "first", "y_pred": "mean"}).reset_index()
    wide_pred = tmp.pivot_table(index=id_cols, columns="variant", values="y_pred", aggfunc="first")
    wide_true = tmp.groupby(id_cols, dropna=False)["y_true"].first()
    wide = wide_pred.join(wide_true).reset_index()
    return wide


def pairwise_cluster_bootstrap(
    wide: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    metric: str,
    cluster_col: str = "patient_id",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    cols = ["y_true", variant_a, variant_b]
    cols += [cluster_col] if cluster_col in wide.columns else []
    sub = wide.dropna(subset=["y_true", variant_a, variant_b]).copy()
    if len(sub) == 0:
        return {"variant_a": variant_a, "variant_b": variant_b, "metric": metric, "n": 0, "estimate_a": np.nan, "estimate_b": np.nan, "diff_a_minus_b": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan}
    est_a = metric_value(sub["y_true"], sub[variant_a], metric)
    est_b = metric_value(sub["y_true"], sub[variant_b], metric)
    diff = est_a - est_b
    if n_bootstrap <= 0:
        lo = hi = p = np.nan
    else:
        rng = np.random.default_rng(seed)
        if cluster_col not in sub.columns:
            sub[cluster_col] = np.arange(len(sub))
        clusters = [g.index.to_numpy() for _, g in sub.groupby(cluster_col, dropna=False)]
        vals = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            sampled = rng.integers(0, len(clusters), len(clusters))
            idx = np.concatenate([clusters[j] for j in sampled])
            boot = sub.loc[idx]
            vals[i] = metric_value(boot["y_true"], boot[variant_a], metric) - metric_value(boot["y_true"], boot[variant_b], metric)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
        p = min(float(p), 1.0)
    return {"variant_a": variant_a, "variant_b": variant_b, "metric": metric, "n": int(len(sub)), "estimate_a": est_a, "estimate_b": est_b, "diff_a_minus_b": diff, "ci_lower": float(lo) if np.isfinite(lo) else lo, "ci_upper": float(hi) if np.isfinite(hi) else hi, "p_value": p}


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    valid = np.isfinite(pvals)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(pvals[idx])]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min((m - rank) * pvals[i], 1.0)
        prev = max(prev, adj)
        out[i] = prev
    return out.tolist()

NON_VARIANT_COLS = {
    "dataset_name",
    "eval_mode",
    "grade_type",
    "SeriesID",
    "patient_id",
    "subject_id",
    "eye_id",
    "center",
    "eye",
    "visit",
    "visit_num",
    "target_visit",
    "target_visit_num",
    "padding_length",
    "padding_tag",
    "padding_mode",
    "padding_value",
    "available_visits",
    "mask_pattern",
    "repeat_idx",
    "age",
    "sex",
    "assistance_mode",
    "reader_id",
    "y_true",
}


HUMAN_AI_CASE_COLS = [
    "grade_type",
    "SeriesID",
    "patient_id",
    "eye_id",
    "center",
    "visit",
    "target_visit",
]


def _is_model_alone_variant(variant: str) -> bool:
    """Return whether a human-AI label denotes the fixed AI-only model."""
    normalized = "".join(ch for ch in str(variant).lower() if ch.isalnum())
    return normalized == "modelalone"


def _aggregate_human_ai_variant(
    df: pd.DataFrame,
    variant: str,
    include_reader: bool,
) -> pd.DataFrame:
    """
    Collapse only true duplicate records.

    Reader identity remains part of the key for human conditions, so two
    different readers are never averaged before the crossed bootstrap.  The
    model-alone condition has no reader dimension and is aggregated only over
    duplicate exports for the same case.
    """
    sub = df[df["model_name"].astype(str).eq(str(variant))].copy()
    sub = sub.dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        return sub

    group_cols = list(HUMAN_AI_CASE_COLS)
    if include_reader:
        group_cols.append("reader_id")

    return (
        sub.groupby(group_cols, dropna=False)
        .agg({"y_true": "first", "y_pred": "mean"})
        .reset_index()
    )


def _build_human_ai_pair_data(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
) -> pd.DataFrame:
    """
    Build reader-preserving paired data for one human-AI comparison.

    Human-vs-human comparisons are paired within reader and case.  For a
    human-vs-model comparison, the fixed model prediction is joined to each
    reader's score on the same case.  This repeats the model value only to
    form reader-specific paired losses; it does not treat the model as a
    randomly sampled reader.
    """
    a_is_model = _is_model_alone_variant(variant_a)
    b_is_model = _is_model_alone_variant(variant_b)

    if a_is_model and b_is_model:
        warnings.warn(
            "Skipping model-alone versus model-alone comparison in human_ai: "
            "the crossed reader-by-patient analysis needs a human condition."
        )
        return pd.DataFrame()

    if a_is_model or b_is_model:
        model_variant = variant_a if a_is_model else variant_b
        human_variant = variant_b if a_is_model else variant_a

        model = _aggregate_human_ai_variant(df, model_variant, include_reader=False)
        human = _aggregate_human_ai_variant(df, human_variant, include_reader=True)
        if model.empty or human.empty:
            return pd.DataFrame()

        pair = human.merge(
            model,
            on=HUMAN_AI_CASE_COLS,
            how="inner",
            suffixes=("_human", "_model"),
        )
        if pair.empty:
            return pd.DataFrame()

        # Keep the requested A/B orientation in every output statistic.
        if a_is_model:
            pred_a = pair["y_pred_model"]
            pred_b = pair["y_pred_human"]
            truth_a = pair["y_true_model"]
            truth_b = pair["y_true_human"]
        else:
            pred_a = pair["y_pred_human"]
            pred_b = pair["y_pred_model"]
            truth_a = pair["y_true_human"]
            truth_b = pair["y_true_model"]
    else:
        a = _aggregate_human_ai_variant(df, variant_a, include_reader=True)
        b = _aggregate_human_ai_variant(df, variant_b, include_reader=True)
        if a.empty or b.empty:
            return pd.DataFrame()

        pair = a.merge(
            b,
            on=HUMAN_AI_CASE_COLS + ["reader_id"],
            how="inner",
            suffixes=("_a", "_b"),
        )
        if pair.empty:
            return pd.DataFrame()
        pred_a = pair["y_pred_a"]
        pred_b = pair["y_pred_b"]
        truth_a = pair["y_true_a"]
        truth_b = pair["y_true_b"]

    y_true_a = pd.to_numeric(truth_a, errors="coerce")
    y_true_b = pd.to_numeric(truth_b, errors="coerce")
    disagree = y_true_a.notna() & y_true_b.notna() & ~np.isclose(y_true_a, y_true_b)
    if disagree.any():
        warnings.warn(
            f"{disagree.sum()} human-AI paired rows have inconsistent y_true; "
            "using the first comparison side as the reference truth."
        )

    out = pair[HUMAN_AI_CASE_COLS + ["reader_id"]].copy()
    out["y_true"] = y_true_a
    out["y_pred_a"] = pd.to_numeric(pred_a, errors="coerce")
    out["y_pred_b"] = pd.to_numeric(pred_b, errors="coerce")
    out = out.dropna(subset=["y_true", "y_pred_a", "y_pred_b"])

    # All patient records remain in a cluster.
    out["_bootstrap_patient_id"] = out["patient_id"].where(
        out["patient_id"].astype(str).str.strip().ne(""),
        out["SeriesID"],
    )
    out["reader_id"] = out["reader_id"].fillna("reader_unknown").astype(str)
    return out


def _weighted_metric(
    sub: pd.DataFrame,
    prediction_col: str,
    metric: str,
    patient_counts: dict,
) -> float:
    """Calculate a row-level metric with bootstrap patient multiplicities."""
    if sub.empty:
        return np.nan

    weights = sub["_bootstrap_patient_id"].map(patient_counts).fillna(0).to_numpy(dtype=float)
    valid = weights > 0
    if not valid.any():
        return np.nan

    y_true = sub.loc[valid, "y_true"].to_numpy(dtype=float)
    y_pred = sub.loc[valid, prediction_col].to_numpy(dtype=float)
    weights = weights[valid]
    err = y_pred - y_true
    denominator = weights.sum()

    if metric == "MAE":
        return float(np.dot(weights, np.abs(err)) / denominator)
    if metric == "MSE":
        return float(np.dot(weights, err ** 2) / denominator)
    if metric == "RMSE":
        mse = np.dot(weights, err ** 2) / denominator
        return float(np.sqrt(mse))
    raise ValueError(f"Unknown metric: {metric}")


def _reader_averaged_pair_metrics(
    paired: pd.DataFrame,
    metric: str,
    sampled_readers: np.ndarray,
    patient_counts: dict,
) -> tuple[float, float]:
    """Return reader-equal metric estimates for A and B in one replicate."""
    estimates_a = []
    estimates_b = []
    for reader_id in sampled_readers:
        sub = paired[paired["reader_id"].eq(reader_id)]
        metric_a = _weighted_metric(sub, "y_pred_a", metric, patient_counts)
        metric_b = _weighted_metric(sub, "y_pred_b", metric, patient_counts)
        if np.isfinite(metric_a) and np.isfinite(metric_b):
            estimates_a.append(metric_a)
            estimates_b.append(metric_b)

    if not estimates_a:
        return np.nan, np.nan
    return float(np.mean(estimates_a)), float(np.mean(estimates_b))


def crossed_reader_patient_bootstrap(
    paired: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Reader-averaged paired comparison with crossed reader/patient bootstrap.

    In every replicate readers and patient clusters are independently sampled
    with replacement.  The returned p value is a two-sided, null-centered
    bootstrap p value for H0: metric(A) - metric(B) = 0.
    """
    if paired.empty:
        return {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "n": 0,
            "n_patients": 0,
            "n_readers": 0,
            "estimate_a": np.nan,
            "estimate_b": np.nan,
            "diff_a_minus_b": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "bootstrap_design": "crossed_reader_patient",
        }

    reader_ids = paired["reader_id"].drop_duplicates().to_numpy()
    patient_ids = paired["_bootstrap_patient_id"].drop_duplicates().to_numpy()
    point_patient_counts = {patient_id: 1 for patient_id in patient_ids}
    estimate_a, estimate_b = _reader_averaged_pair_metrics(
        paired,
        metric,
        reader_ids,
        point_patient_counts,
    )
    diff = estimate_a - estimate_b

    if n_bootstrap <= 0 or not np.isfinite(diff):
        lo = hi = p_value = np.nan
    else:
        rng = np.random.default_rng(seed)
        boot_diffs = []
        for _ in range(n_bootstrap):
            sampled_readers = rng.choice(reader_ids, size=len(reader_ids), replace=True)
            sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            patient_counts = pd.Series(sampled_patients).value_counts().to_dict()
            boot_a, boot_b = _reader_averaged_pair_metrics(
                paired,
                metric,
                sampled_readers,
                patient_counts,
            )
            if np.isfinite(boot_a) and np.isfinite(boot_b):
                boot_diffs.append(boot_a - boot_b)

        boot_diffs = np.asarray(boot_diffs, dtype=float)
        if len(boot_diffs) == 0:
            lo = hi = p_value = np.nan
        else:
            lo, hi = np.nanpercentile(boot_diffs, [2.5, 97.5])
            p_value = (
                1 + np.sum(np.abs(boot_diffs - diff) >= abs(diff))
            ) / (len(boot_diffs) + 1)

    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "metric": metric,
        "n": int(len(paired)),
        "n_patients": int(len(patient_ids)),
        "n_readers": int(len(reader_ids)),
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "diff_a_minus_b": diff,
        "ci_lower": float(lo) if np.isfinite(lo) else lo,
        "ci_upper": float(hi) if np.isfinite(hi) else hi,
        "p_value": p_value,
        "bootstrap_design": "crossed_reader_patient",
    }


def human_ai_pairwise_tests(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
    variants: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run reader-preserving crossed bootstrap comparisons for human-AI data."""
    df = ensure_columns(df).dropna(subset=["y_true", "y_pred"]).copy()
    available = df["model_name"].astype(str).drop_duplicates().tolist()
    if variants is not None:
        available = [variant for variant in variants if variant in available]

    rows = []
    for i, variant_a in enumerate(available):
        for variant_b in available[i + 1:]:
            paired = _build_human_ai_pair_data(df, variant_a, variant_b)
            for metric in METRICS:
                rows.append(
                    crossed_reader_patient_bootstrap(
                        paired=paired,
                        variant_a=variant_a,
                        variant_b=variant_b,
                        metric=metric,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                )

    out = pd.DataFrame(rows)
    if len(out) and "p_value" in out:
        out["p_holm"] = holm_adjust(out["p_value"].tolist())
    return out


def pairwise_tests(
    df: pd.DataFrame,
    task: str,
    cluster_col: str = "patient_id",
    n_bootstrap: int = 2000,
    seed: int = 42,
    variants: Optional[list[str]] = None,
) -> pd.DataFrame:
    if task == "human_ai":
        return human_ai_pairwise_tests(
            df=df,
            n_bootstrap=n_bootstrap,
            seed=seed,
            variants=variants,
        )

    df = make_variant(df, task)
    id_cols = [
        "dataset_name",
        "grade_type",
        "SeriesID",
        "patient_id",
        "subject_id",
        "eye_id",
        "center",
        "visit",
        "visit_num",
        "target_visit",
        "target_visit_num",
        "age",
        "sex",
    ]

    if task in {"single_vs_multi_evaluator", "single_vs_multi_predictor"}:
        # 这类任务通常需要在相同 padding/tag 条件下比较 single vs multi
        for c in ["padding_length", "padding_tag", "padding_mode", "padding_value", "available_visits", "mask_pattern",
                  "repeat_idx"]:
            if c in df.columns:
                id_cols.append(c)

    elif task in {"predictor_length", "predictor_tag", "predictor_all"}:
        # 这类任务是比较 padding/tag 条件本身，所以不要把这些条件列放进 id_cols
        pass

    wide = build_pairwise_wide(df, id_cols=id_cols)
    all_variants = [c for c in wide.columns if c not in NON_VARIANT_COLS]
    if variants is not None:
        all_variants = [v for v in variants if v in all_variants]
    rows = []
    for i in range(len(all_variants)):
        for j in range(i + 1, len(all_variants)):
            for metric in METRICS:
                rows.append(pairwise_cluster_bootstrap(wide, all_variants[i], all_variants[j], metric, cluster_col, n_bootstrap, seed))
    out = pd.DataFrame(rows)
    if len(out) and "p_value" in out:
        out["p_holm"] = holm_adjust(out["p_value"].tolist())
    return out


def make_pair_long_for_regression(
    wide: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    cluster_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Convert pairwise wide table to long format for GEE/MixedLM.

    model_label:
        A = variant_a
        B = variant_b

    outcome:
        abs_error or squared_error
    """
    rows = []

    for _, row in wide.iterrows():
        y_true = float(row["y_true"])

        subject_id = row.get("subject_id", row.get(cluster_col, row.get("patient_id", "UNKNOWN")))
        if pd.isna(subject_id):
            subject_id = "UNKNOWN"

        eye_id = row.get("eye_id", "UNKNOWN")
        if pd.isna(eye_id):
            eye_id = "UNKNOWN"

        center = row.get("center", "UNKNOWN")
        if pd.isna(center):
            center = "UNKNOWN"

        visit_num = row.get("visit_num", row.get("target_visit", np.nan))
        age = row.get("age", np.nan)
        sex = row.get("sex", "UNKNOWN")

        for label, pred_col in [("A", variant_a), ("B", variant_b)]:
            pred = float(row[pred_col])

            rows.append({
                "target_series_id": row.get("SeriesID", row.get("target_series_id", "")),
                "model_label": label,
                "center": center,
                "subject_id": subject_id,
                "eye_id": eye_id,
                "visit_num": visit_num,
                "age": age,
                "sex": sex,
                "y_true": y_true,
                "y_pred": pred,
                "abs_error": abs(y_true - pred),
                "squared_error": (y_true - pred) ** 2,
            })

    return pd.DataFrame(rows)


def build_regression_formula(data: pd.DataFrame, outcome: str) -> str:
    """
    Build GEE/MixedLM formula.

    Main coefficient:
        C(model_label, Treatment(reference="A"))[T.B]

    Its meaning:
        error_B - error_A
    """
    terms = ['C(model_label, Treatment(reference="A"))']

    if "center" in data.columns and data["center"].nunique(dropna=True) > 1:
        terms.append("C(center)")

    if "visit_num" in data.columns and data["visit_num"].nunique(dropna=True) > 1:
        terms.append("C(visit_num)")

    if "age" in data.columns:
        age_numeric = pd.to_numeric(data["age"], errors="coerce")
        if age_numeric.notna().sum() > 0 and age_numeric.nunique(dropna=True) > 1:
            terms.append("age")

    if "sex" in data.columns and data["sex"].nunique(dropna=True) > 1:
        terms.append("C(sex)")

    return outcome + " ~ " + " + ".join(terms)


def extract_model_label_coef(result) -> tuple[float, float, str]:
    """
    Extract B-vs-A coefficient and p value.

    Coefficient meaning:
        error_B - error_A
    """
    params = result.params
    pvalues = result.pvalues

    target_name = None

    for name in params.index:
        if "model_label" in name and "T.B" in name:
            target_name = name
            break

    if target_name is None:
        return np.nan, np.nan, "model_label coefficient not found"

    return float(params[target_name]), float(pvalues[target_name]), ""


def gee_pairwise_test(
    wide: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    outcome: str,
    cluster_col: str = "patient_id",
) -> dict:
    """
    GEE pairwise test.

    outcome:
        abs_error      -> MAE
        squared_error  -> MSE

    Returned diff_A_minus_B is aligned with bootstrap:
        diff_A_minus_B = error_A - error_B
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as exc:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": f"statsmodels import failed: {exc}",
        }

    data = make_pair_long_for_regression(
        wide=wide,
        variant_a=variant_a,
        variant_b=variant_b,
        cluster_col=cluster_col,
    )

    data = data.dropna(subset=[outcome, "subject_id", "model_label"]).copy()
    data["center"] = data["center"].fillna("UNKNOWN").astype(str)
    data["subject_id"] = data["subject_id"].fillna("UNKNOWN").astype(str)
    data["eye_id"] = data["eye_id"].fillna("UNKNOWN").astype(str)
    data["sex"] = data["sex"].fillna("UNKNOWN").astype(str)
    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    if data["age"].notna().sum() > 0:
        data["age"] = data["age"].fillna(data["age"].mean())

    if data.empty or data["model_label"].nunique() < 2:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": "comparison input requires both model labels",
        }

    try:
        formula = build_regression_formula(data, outcome)

        model = smf.gee(
            formula=formula,
            groups="subject_id",
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit()

        coef_b_minus_a, p_value, err = extract_model_label_coef(result)

        return {
            "diff_A_minus_B": -coef_b_minus_a,
            "coef_B_minus_A": coef_b_minus_a,
            "p_value": p_value,
            "status": "ok" if err == "" else err,
        }

    except Exception as exc:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": f"gee failed: {exc}",
        }


def mixedlm_pairwise_test(
    wide: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    outcome: str,
    cluster_col: str = "patient_id",
) -> dict:
    """
    MixedLM pairwise test.

    Random structure:
        random intercept for subject_id
        variance component for eye_id when available

    Returned diff_A_minus_B is aligned with bootstrap:
        diff_A_minus_B = error_A - error_B
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": f"statsmodels import failed: {exc}",
        }

    data = make_pair_long_for_regression(
        wide=wide,
        variant_a=variant_a,
        variant_b=variant_b,
        cluster_col=cluster_col,
    )

    data = data.dropna(subset=[outcome, "subject_id", "model_label"]).copy()
    data["center"] = data["center"].fillna("UNKNOWN").astype(str)
    data["subject_id"] = data["subject_id"].fillna("UNKNOWN").astype(str)
    data["eye_id"] = data["eye_id"].fillna("UNKNOWN").astype(str)
    data["sex"] = data["sex"].fillna("UNKNOWN").astype(str)
    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    if data["age"].notna().sum() > 0:
        data["age"] = data["age"].fillna(data["age"].mean())

    if data.empty or data["model_label"].nunique() < 2:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": "comparison input requires both model labels",
        }

    try:
        formula = build_regression_formula(data, outcome)

        vc_formula = None
        if data["eye_id"].nunique(dropna=True) > 1:
            vc_formula = {"eye": "0 + C(eye_id)"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data["subject_id"],
                vc_formula=vc_formula,
            )
            result = model.fit(
                reml=False,
                method="lbfgs",
                maxiter=200,
                disp=False,
            )

        coef_b_minus_a, p_value, err = extract_model_label_coef(result)

        return {
            "diff_A_minus_B": -coef_b_minus_a,
            "coef_B_minus_A": coef_b_minus_a,
            "p_value": p_value,
            "status": "ok" if err == "" else err,
        }

    except Exception as exc:
        return {
            "diff_A_minus_B": np.nan,
            "coef_B_minus_A": np.nan,
            "p_value": np.nan,
            "status": f"mixedlm failed: {exc}",
        }


def regression_pairwise_tests(
    df: pd.DataFrame,
    task: str,
    cluster_col: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    GEE/MixedLM pairwise tests on absolute and squared error.

    A is the reference condition: coefficient = error_B - error_A and
    diff_A_minus_B = error_A - error_B. Subject ID defines repeated rows.
    """
    df = make_variant(df, task)
    id_cols = [
        "dataset_name",
        "grade_type",
        "SeriesID",
        "patient_id",
        "subject_id",
        "eye_id",
        "center",
        "visit",
        "visit_num",
        "target_visit",
        "target_visit_num",
        "age",
        "sex",
    ]

    if task in {"single_vs_multi_evaluator", "single_vs_multi_predictor"}:
        # 这类任务通常需要在相同 padding/tag 条件下比较 single vs multi
        for c in ["padding_length", "padding_tag", "padding_mode", "padding_value", "available_visits", "mask_pattern",
                  "repeat_idx"]:
            if c in df.columns:
                id_cols.append(c)

    elif task in {"predictor_length", "predictor_tag", "predictor_all"}:
        # 这类任务是比较 padding/tag 条件本身，所以不要把这些条件列放进 id_cols
        pass

    wide = build_pairwise_wide(df, id_cols=id_cols)
    variant_cols = [c for c in wide.columns if c not in NON_VARIANT_COLS]
    gee_rows = []
    mixed_rows = []

    metric_map = {
        "MAE": "abs_error",
        "MSE": "squared_error",
    }

    for i in range(len(variant_cols)):
        for j in range(i + 1, len(variant_cols)):
            a, b = variant_cols[i], variant_cols[j]

            sub = wide.dropna(subset=["y_true", a, b]).copy()

            if len(sub) < 5:
                continue

            for metric, outcome in metric_map.items():
                gee = gee_pairwise_test(
                    wide=sub,
                    variant_a=a,
                    variant_b=b,
                    outcome=outcome,
                    cluster_col=cluster_col,
                )

                gee_rows.append({
                    "variant_a": a,
                    "variant_b": b,
                    "metric": metric,
                    **gee,
                })

                mixed = mixedlm_pairwise_test(
                    wide=sub,
                    variant_a=a,
                    variant_b=b,
                    outcome=outcome,
                    cluster_col=cluster_col,
                )

                mixed_rows.append({
                    "variant_a": a,
                    "variant_b": b,
                    "metric": metric,
                    **mixed,
                })

    gee_df = pd.DataFrame(gee_rows)
    mixed_df = pd.DataFrame(mixed_rows)

    if len(gee_df) and "p_value" in gee_df:
        gee_df["p_holm"] = holm_adjust(gee_df["p_value"].tolist())

    if len(mixed_df) and "p_value" in mixed_df:
        mixed_df["p_holm"] = holm_adjust(mixed_df["p_value"].tolist())

    return gee_df, mixed_df
