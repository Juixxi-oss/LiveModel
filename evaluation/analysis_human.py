import os
from typing import Iterable

import numpy as np
import pandas as pd

from analysis_core import ensure_columns, parse_series_id

GRADE_COLUMNS = ["C", "N", "P", "BCVA", "OSI", "MTF", "SR"]


def _parse_table_spec(spec: str) -> tuple[str, str, str, str]:
    """Parse mode:path, mode:reader_id:path, or study_task:mode:reader_id:path."""
    parts = spec.split(":", 3)
    if len(parts) == 2:
        mode, path = parts
        reader = ""
        study_task = ""
    elif len(parts) == 3:
        mode, reader, path = parts
        study_task = ""
    elif len(parts) == 4:
        study_task, mode, reader, path = parts
    else:
        raise ValueError("Human table spec must be mode:path, mode:reader_id:path, or study_task:mode:reader_id:path")
    if not mode or not path:
        raise ValueError("Human table spec requires a mode and path")
    return study_task, mode, reader, path


def human_table_to_long(
    path: str,
    assistance_mode: str,
    reader_id: str = "",
    study_task: str = "",
) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    if "SeriesID" not in raw.columns:
        raise ValueError(f"Human table must contain SeriesID column: {path}")
    grades = [c for c in raw.columns if c in GRADE_COLUMNS]
    has_reader_column = "reader_id" in raw.columns
    model_alone = "".join(ch for ch in assistance_mode.lower() if ch.isalnum()) in {
        "aialone", "modelalone", "livemodel"
    }
    rows = []
    for _, r in raw.iterrows():
        sid = str(r["SeriesID"])
        meta = parse_series_id(sid)
        row_task = r.get("study_task", "")
        row_task = "" if pd.isna(row_task) else str(row_task).strip()
        if study_task and row_task and row_task != study_task:
            raise ValueError(f"Conflicting study_task values in {path}")
        row_task = study_task or row_task
        if not row_task:
            raise ValueError(f"Human table requires study_task in the input spec or each row: {path}")

        row_reader = r.get("reader_id", "")
        row_reader = "" if pd.isna(row_reader) else str(row_reader).strip()
        if has_reader_column and not row_reader and not model_alone:
            raise ValueError(f"Human table has a missing reader_id: {path}")
        resolved_reader = row_reader or str(reader_id).strip()
        if not resolved_reader or resolved_reader == "reader_unknown":
            if not model_alone:
                raise ValueError(f"Human table requires reader_id in the input spec or each row: {path}")
            resolved_reader = "AI"
        for g in grades:
            if pd.isna(r[g]):
                continue
            rows.append({
                "dataset_name": "external" if meta["center"] in {"S01", "S09"} else "internal",
                "eval_mode": "human_ai",
                "model_name": assistance_mode,
                "reader_id": resolved_reader,
                "assistance_mode": assistance_mode,
                "study_task": row_task,
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
        study_task, mode, reader, path = _parse_table_spec(spec)
        frames.append(human_table_to_long(path, mode, reader, study_task))
    if not frames:
        return pd.DataFrame()
    human = pd.concat(frames, ignore_index=True)
    return attach_truth(human, truth_prediction_df)
