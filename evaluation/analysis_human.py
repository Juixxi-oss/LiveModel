import os
from typing import Iterable

import numpy as np
import pandas as pd

from analysis_core import ensure_columns, parse_series_id

GRADE_COLUMNS = ["C", "N", "P", "BCVA", "OSI", "MTF", "SR"]


def _parse_table_spec(spec: str) -> tuple[str, str, str]:
    """Parse mode[:reader_id]:path."""
    parts = spec.split(":", 2)
    if len(parts) == 2:
        mode, path = parts
        reader = "reader_unknown"
    elif len(parts) == 3:
        mode, reader, path = parts
    else:
        raise ValueError("Human table spec must be mode:path or mode:reader_id:path")
    return mode, reader, path


def human_table_to_long(path: str, assistance_mode: str, reader_id: str = "reader_unknown") -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    if "SeriesID" not in raw.columns:
        raise ValueError(f"Human table must contain SeriesID column: {path}")
    grades = [c for c in raw.columns if c in GRADE_COLUMNS]
    rows = []
    for _, r in raw.iterrows():
        sid = str(r["SeriesID"])
        meta = parse_series_id(sid)
        for g in grades:
            if pd.isna(r[g]):
                continue
            rows.append({
                "dataset_name": meta["center"],
                "eval_mode": "human_ai",
                "model_name": assistance_mode,
                "reader_id": reader_id,
                "assistance_mode": assistance_mode,
                "grade_type": g,
                "SeriesID": sid,
                "center": meta["center"],
                "patient_id": meta["patient_id"],
                "eye_id": meta["eye_id"],
                "eye": meta["eye"],
                "visit": meta["visit"],
                "target_visit": meta["visit"],
                "y_pred": float(r[g]),
            })
    return pd.DataFrame(rows)


def build_truth_map(model_df: pd.DataFrame) -> pd.DataFrame:
    model_df = ensure_columns(model_df)
    keys = ["grade_type", "SeriesID"]
    truth = model_df.dropna(subset=["y_true"]).groupby(keys, dropna=False)["y_true"].first().reset_index()
    # Match patient, eye, visit, and outcome when SeriesID formats differ.
    meta = truth["SeriesID"].map(parse_series_id)
    truth["norm_key"] = [f"{m['patient_id']}|{m['eye_id']}|{m['visit']}|{g}" for m, g in zip(meta, truth["grade_type"])]
    return truth


def attach_truth(human_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    truth = build_truth_map(model_df)
    human = human_df.copy()
    human = human.merge(truth[["grade_type", "SeriesID", "y_true"]], on=["grade_type", "SeriesID"], how="left")
    missing = human["y_true"].isna()
    if missing.any():
        truth_norm = truth[["norm_key", "y_true"]].drop_duplicates("norm_key")
        meta = human.loc[missing, "SeriesID"].map(parse_series_id)
        human.loc[missing, "norm_key"] = [
            f"{m['patient_id']}|{m['eye_id']}|{m['visit']}|{g}"
            for m, g in zip(meta, human.loc[missing, "grade_type"])
        ]
        human = human.merge(truth_norm.rename(columns={"y_true": "y_true_norm"}), on="norm_key", how="left")
        human["y_true"] = human["y_true"].fillna(human["y_true_norm"])
        human = human.drop(columns=[c for c in ["norm_key", "y_true_norm"] if c in human.columns])
    return ensure_columns(human)


def load_human_tables(specs: Iterable[str], truth_prediction_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spec in specs:
        mode, reader, path = _parse_table_spec(spec)
        frames.append(human_table_to_long(path, mode, reader))
    if not frames:
        return pd.DataFrame()
    human = pd.concat(frames, ignore_index=True)
    return attach_truth(human, truth_prediction_df)
