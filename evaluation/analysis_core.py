import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

METRICS = ["MAE", "MSE", "RMSE"]


VIEW_ORDER = ["D", "C", "N", "P"]
VIEW_PRED_COLUMNS = {v: f"y_pred_{v}" for v in VIEW_ORDER}


def normalize_view_spec(view_spec: str) -> str:
    """
    Normalize a view specification.

    Accepted values:
        D/C/N/P/DC/DN/DP/CN/CP/NP/DCN/DCP/DNP/CNP/DCNP

    The returned string is always ordered as D/C/N/P.
    DCNP is the four-view mean.
    """
    if view_spec is None:
        return "DCNP"

    spec = str(view_spec).strip().upper()
    if spec == "":
        return "DCNP"

    valid_specs = {
        "D", "C", "N", "P",
        "DC", "DN", "DP", "CN", "CP", "NP",
        "DCN", "DCP", "DNP", "CNP", "DCNP",
    }

    if spec not in valid_specs:
        raise ValueError(
            f"Invalid view spec: {view_spec}. "
            "Use one of: D/C/N/P/DC/DN/DP/CN/CP/NP/DCN/DCP/DNP/CNP/DCNP."
        )

    return "".join([v for v in VIEW_ORDER if v in spec])

def _view_prediction_column(df: pd.DataFrame, view: str) -> str:
    """
    Return the best available prediction column for one view.

    Aggregated files may contain y_pred_D_mean / y_pred_C_mean / ...
    Fold-level files may contain y_pred_D / y_pred_C / ...
    This function supports both.
    """
    preferred = f"y_pred_{view}_mean"
    fallback = f"y_pred_{view}"

    if preferred in df.columns and pd.to_numeric(df[preferred], errors="coerce").notna().any():
        return preferred

    if fallback in df.columns and pd.to_numeric(df[fallback], errors="coerce").notna().any():
        return fallback

    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback

    raise KeyError(
        f"Missing prediction column for view {view}: expected {preferred} or {fallback}."
    )


def _has_view_prediction_columns(df: pd.DataFrame, view_spec: str) -> bool:
    spec = normalize_view_spec(view_spec)
    try:
        for v in spec:
            _view_prediction_column(df, v)
        return True
    except KeyError:
        return False


def _view_prediction_mean(df: pd.DataFrame, view_spec: str) -> pd.Series:
    spec = normalize_view_spec(view_spec)
    cols = [_view_prediction_column(df, v) for v in spec]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1)

def apply_view_selection(
    df: pd.DataFrame,
    snap_view: str = "DCNP",
    single_view: str = "DCNP",
    rename_models: bool = True,
) -> pd.DataFrame:
    """
    Build analysis-ready y_pred for single-view model families.

    Exported single-view model rows store predictions in:
        y_pred_D, y_pred_C, y_pred_N, y_pred_P

    This function selects one or multiple views and writes their mean into y_pred.

    Examples:
        snap_view="C"   -> y_pred = y_pred_C
        snap_view="CP"  -> y_pred = mean(y_pred_C, y_pred_P)
        snap_view="DCNP" -> four-view mean

    Multi-view model rows are left unchanged.
    """
    df = ensure_columns(df).copy()

    snap_spec = normalize_view_spec(snap_view)
    single_spec = normalize_view_spec(single_view)

    # Accept base names and view-qualified names such as SnapRegressor_C.
    names = df["model_name"].astype(str)
    snap_mask = (
        names.eq("SnapRegressor")
        | names.str.startswith("SnapRegressor_")
    )
    single_mask = (
        names.eq("SingleViewTimeFusionRegressor")
        | names.str.startswith("SingleViewTimeFusionRegressor_")
    )

    if snap_mask.any():
        if _has_view_prediction_columns(df.loc[snap_mask], snap_spec):
            df.loc[snap_mask, "y_pred"] = _view_prediction_mean(
                df.loc[snap_mask],
                snap_spec
            ).values
        else:
            # Use an existing prediction when view-specific columns are absent.
            if df.loc[snap_mask, "y_pred"].isna().all():
                missing = [
                    VIEW_PRED_COLUMNS[v]
                    for v in snap_spec
                    if VIEW_PRED_COLUMNS[v] not in df.columns
                ]
                raise KeyError(
                    "SnapRegressor rows need selected view predictions, but missing: "
                    + ", ".join(missing)
                )

        if rename_models:
            df.loc[snap_mask, "model_name"] = f"SnapRegressor_{snap_spec}"

    if single_mask.any():
        if _has_view_prediction_columns(df.loc[single_mask], single_spec):
            df.loc[single_mask, "y_pred"] = _view_prediction_mean(
                df.loc[single_mask],
                single_spec
            ).values
        else:
            if df.loc[single_mask, "y_pred"].isna().all():
                missing = [
                    VIEW_PRED_COLUMNS[v]
                    for v in single_spec
                    if VIEW_PRED_COLUMNS[v] not in df.columns
                ]
                raise KeyError(
                    "SingleViewTimeFusionRegressor rows need selected view predictions, but missing: "
                    + ", ".join(missing)
                )

        if rename_models:
            df.loc[single_mask, "model_name"] = f"SingleViewTimeFusionRegressor_{single_spec}"

    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2

    return df



def metric_value(y_true, y_pred, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    err = y_pred - y_true
    if metric == "MAE":
        return float(np.mean(np.abs(err)))
    if metric == "MSE":
        return float(np.mean(err ** 2))
    if metric == "RMSE":
        return float(np.sqrt(np.mean(err ** 2)))
    raise ValueError(f"Unknown metric: {metric}")


def all_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {m: np.nan for m in METRICS}
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def parse_series_id(series_id: str) -> dict:
    s = str(series_id)
    # Human table style: S01003_OD_V9 or S01003_OS_V9
    m = re.match(r"^(S\d{2}\d{3})[_-]?([A-Za-z]{2})[_-]?V?(\d)$", s)
    if m:
        patient = m.group(1)
        eye = m.group(2).upper()
        visit = f"V{m.group(3)}"
        return {
            "center": patient[:3],
            "patient_id": patient,
            "eye_id": f"{patient}_{eye}",
            "eye": eye,
            "visit": visit,
        }
    parts = re.split(r"[_-]+", s)
    if len(parts) >= 3 and parts[-1].upper().startswith("V"):
        patient = parts[0]
        eye = parts[1].upper()
        visit = parts[-1].upper()
        return {
            "center": patient[:3],
            "patient_id": patient,
            "eye_id": f"{patient}_{eye}",
            "eye": eye,
            "visit": visit,
        }
    center = s[:3] if len(s) >= 3 else "UNKNOWN"
    patient_id = s[:6] if len(s) >= 6 else s
    eye_id = s[:9] if len(s) >= 9 else s
    visit = f"V{s[-1]}" if len(s) > 0 and str(s[-1]).isdigit() else ""
    return {
        "center": center,
        "patient_id": patient_id,
        "eye_id": eye_id,
        "eye": "",
        "visit": visit,
    }


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "dataset_name": "",
        "eval_mode": "",
        "model_name": "",
        "reader_id": "",
        "assistance_mode": "",
        "study_task": "",
        "grade_type": "",
        "fold": np.nan,
        "SeriesID": "",
        "center": "",
        "patient_id": "",
        "eye_id": "",
        "eye": "",
        "visit": "",
        "target_visit": "",
        "padding_length": np.nan,
        "padding_tag": "",
        "available_visits": "",
        "mask_pattern": "",
        "repeat_idx": np.nan,
        "y_true": np.nan,
        "y_pred": np.nan,
        "y_pred_D": np.nan,
        "y_pred_C": np.nan,
        "y_pred_N": np.nan,
        "y_pred_P": np.nan,
        "y_pred_D_mean": np.nan,
        "y_pred_C_mean": np.nan,
        "y_pred_N_mean": np.nan,
        "y_pred_P_mean": np.nan,
        "y_pred_std": np.nan,
        "n_folds": np.nan,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    # Fill identity columns from SeriesID when missing.
    meta = df["SeriesID"].map(parse_series_id)
    for col in ["center", "patient_id", "eye_id", "eye", "visit"]:
        mask = df[col].isna() | (df[col].astype(str) == "")
        if mask.any():
            df.loc[mask, col] = meta[mask].map(lambda x: x[col])
    target_mask = df["target_visit"].isna() | (df["target_visit"].astype(str) == "")
    df.loc[target_mask, "target_visit"] = df.loc[target_mask, "visit"]
    # Support both fold-level view columns (y_pred_D) and final-level
    # aggregated columns (y_pred_D_mean).
    for v in VIEW_ORDER:
        raw_col = f"y_pred_{v}"
        mean_col = f"y_pred_{v}_mean"

        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
        df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")

        # Keep a normalized raw column as a convenience alias. This is important
        # for final_long_predictions files, where only y_pred_*_mean may be valid.
        df[raw_col] = df[raw_col].fillna(df[mean_col])

    if "y_pred_mean" in df.columns:
        df["y_pred"] = df["y_pred"].fillna(pd.to_numeric(df["y_pred_mean"], errors="coerce"))

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["padding_length"] = pd.to_numeric(df["padding_length"], errors="coerce")
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2
    for c in ["dataset_name", "eval_mode", "model_name", "reader_id", "assistance_mode", "study_task", "grade_type", "SeriesID", "center", "patient_id", "eye_id", "visit", "target_visit", "padding_tag", "available_visits", "mask_pattern"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def read_prediction_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_columns(df)


def read_many_prediction_csv(paths: Iterable[str]) -> pd.DataFrame:
    frames = [read_prediction_csv(p) for p in paths]
    if not frames:
        raise ValueError("At least one prediction CSV path is required.")
    return ensure_columns(pd.concat(frames, ignore_index=True))


def load_demographics(path: str) -> pd.DataFrame:
    source = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    age_col = next((name for name in ("age", "年龄") if name in source.columns), None)
    sex_col = next((name for name in ("sex", "性别") if name in source.columns), None)
    if "SeriesID" not in source.columns or age_col is None or sex_col is None:
        raise ValueError("The label table must contain SeriesID, age/年龄, and sex/性别 columns.")
    demographics = source[["SeriesID", age_col, sex_col]].rename(
        columns={age_col: "age", sex_col: "sex"}
    ).copy()
    demographics["SeriesID"] = demographics["SeriesID"].astype(str).str.strip()
    if demographics["SeriesID"].duplicated().any():
        raise ValueError("The label table must have one demographic record per SeriesID.")
    demographics["age"] = pd.to_numeric(demographics["age"], errors="coerce")
    demographics["sex"] = demographics["sex"].astype("string").str.strip().replace("", pd.NA)
    return demographics


def attach_demographics(predictions: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    if "SeriesID" not in predictions.columns:
        raise ValueError("Prediction rows must contain SeriesID to match demographics.")
    rows = predictions.copy()
    rows["SeriesID"] = rows["SeriesID"].astype(str).str.strip()
    rows = rows.merge(demographics, on="SeriesID", how="left", validate="many_to_one", suffixes=("", "_label"))
    for column in ("age", "sex"):
        label_column = f"{column}_label"
        if label_column not in rows.columns:
            continue
        existing = rows[column]
        label = rows[label_column]
        if column == "age":
            existing = pd.to_numeric(existing, errors="coerce")
            label = pd.to_numeric(label, errors="coerce")
        else:
            existing = existing.astype("string").str.strip().replace("", pd.NA)
            label = label.astype("string").str.strip().replace("", pd.NA)
        if (existing.notna() & label.notna() & existing.ne(label).fillna(False)).any():
            raise ValueError(f"Prediction {column} values conflict with the label table.")
        rows[column] = existing.fillna(label)
        rows = rows.drop(columns=label_column)
    if rows[["age", "sex"]].isna().any().any():
        raise ValueError("Age and sex are required for every prediction row.")
    return rows


def cluster_bootstrap_metric_ci(
    df: pd.DataFrame,
    metric: str,
    cluster_col: str = "patient_id",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    df = df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    if len(df) == 0:
        return np.nan, np.nan, np.nan, 0
    point = metric_value(df["y_true"].values, df["y_pred"].values, metric)
    if n_bootstrap <= 0:
        return point, np.nan, np.nan, len(df)
    rng = np.random.default_rng(seed)
    for col in ("center", cluster_col):
        if col not in df.columns or df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Center-stratified bootstrap requires a populated {col} column.")
    center_clusters = [
        [patient.index.to_numpy() for _, patient in center.groupby(cluster_col, sort=False)]
        for _, center in df.groupby("center", sort=False)
    ]
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = np.concatenate([
            clusters[i]
            for clusters in center_clusters
            for i in rng.integers(0, len(clusters), len(clusters))
        ])
        sub = df.iloc[idx]
        boot[b] = metric_value(sub["y_true"].values, sub["y_pred"].values, metric)
    return point, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), len(df)


def performance_table(
    df: pd.DataFrame,
    group_cols: list[str],
    cluster_col: str = "patient_id",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    df = ensure_columns(df).dropna(subset=["y_true", "y_pred"])
    group_cols = [c for c in group_cols if c in df.columns]
    rows = []

    def append_group_rows(columns, pooled=False):
        grouped = df.groupby(columns, dropna=False) if columns else [((), df)]
        for key, sub in grouped:
            if pooled and sub["center"].nunique() < 2:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(columns, key))
            if pooled:
                base["center"] = "pooled"
            n_clusters = sub[["center", cluster_col]].drop_duplicates().shape[0]
            for metric in METRICS:
                point, lo, hi, n = cluster_bootstrap_metric_ci(
                    sub, metric, cluster_col=cluster_col,
                    n_bootstrap=n_bootstrap, seed=seed,
                )
                rows.append({**base, "metric": metric, "estimate": point,
                             "ci_lower": lo, "ci_upper": hi, "n": n,
                             "n_clusters": n_clusters})

    append_group_rows(group_cols)
    if "center" in group_cols:
        append_group_rows([c for c in group_cols if c != "center"], pooled=True)
    return pd.DataFrame(rows)
