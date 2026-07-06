import os
from typing import Optional

import pandas as pd

from analysis_core import ensure_columns, performance_table
from analysis_tests import pairwise_tests, regression_pairwise_tests


def load_exported_predictions(input_root: str, grade_type: str, datasets: list[str]) -> pd.DataFrame:
    frames = []
    for d in datasets:
        path = os.path.join(input_root, grade_type, d, f"final_long_predictions_{grade_type}_{d}.csv")
        if not os.path.exists(path):
            print(f"[!] Missing prediction file, skipped: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No exported final_long_predictions CSV files found.")
    return ensure_columns(pd.concat(frames, ignore_index=True))


def select_dataset(df: pd.DataFrame, dataset_group: str) -> pd.DataFrame:
    df = ensure_columns(df)
    if dataset_group == "internal":
        return df[df["dataset_name"].eq("internal")].copy()
    if dataset_group in {"S01", "S09"}:
        # External CSV may have dataset_name=external, so split by center.
        return df[df["center"].eq(dataset_group)].copy()
    if dataset_group == "external":
        return df[df["dataset_name"].eq("external")].copy()
    if dataset_group == "all":
        return df.copy()
    return df[df["dataset_name"].eq(dataset_group)].copy()


def _family_mask(df: pd.DataFrame, family: str) -> pd.Series:
    names = df["model_name"].astype(str)
    return names.eq(family) | names.str.startswith(f"{family}_")


def subset_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = ensure_columns(df)

    snap = _family_mask(df, "SnapRegressor")
    view = _family_mask(df, "ViewFusionRegressor")
    single = _family_mask(df, "SingleViewTimeFusionRegressor")
    multi = _family_mask(df, "MultiViewTimeFusionRegressor")

    if task == "single_time":
        return df[
            (df["eval_mode"] == "single_time_all_visits")
            & (snap | view)
        ].copy()

    if task == "evaluator_padding":
        return df[
            (df["eval_mode"] == "evaluator_padding")
            & (single | multi)
        ].copy()

    if task == "predictor_length":
        return df[
            (df["eval_mode"] == "predictor_padding_length")
            & (single | multi)
            & (df["target_visit"] == "V9")
        ].copy()

    if task == "predictor_tag":
        return df[
            (df["eval_mode"] == "predictor_padding_tag")
            & (single | multi)
            & (df["target_visit"] == "V9")
        ].copy()

    if task == "v9_models":
        a = df[
            (df["eval_mode"] == "single_time_all_visits")
            & (df["target_visit"] == "V9")
            & (snap | view)
        ]
        b = df[
            (df["eval_mode"] == "evaluator_padding")
            & (df["padding_length"] == 0)
            & (df["target_visit"] == "V9")
            & (single | multi)
        ]
        c = df[
            (df["eval_mode"] == "predictor_padding_length")
            & (df["padding_length"] == 0)
            & (df["target_visit"] == "V9")
            & (single | multi)
        ]
        return pd.concat([a, b, c], ignore_index=True)

    if task == "single_vs_multi_evaluator":
        return df[
            (df["eval_mode"] == "evaluator_padding")
            & (single | multi)
        ].copy()

    if task == "single_vs_multi_predictor":
        return df[
            (df["target_visit"] == "V9")
            & df["eval_mode"].isin(["predictor_padding_length", "predictor_padding_tag"])
            & (single | multi)
        ].copy()

    if task == "generalization":
        return df.copy()

    if task == "human_ai":
        return df[
            df["eval_mode"].eq("human_ai")
            | df["model_name"].isin(["Human alone", "Model alone", "Human+AI", "human_alone", "model_alone", "human_ai"])
        ].copy()

    raise ValueError(f"Unknown task: {task}")


def performance_group_cols(task: str) -> list[str]:
    base = ["dataset_name", "center", "grade_type", "model_name"]
    if task == "single_time":
        return base + ["target_visit"]
    if task == "evaluator_padding":
        return base + ["padding_length", "target_visit", "available_visits"]
    if task == "predictor_length":
        return base + ["padding_length", "available_visits", "mask_pattern"]
    if task == "predictor_tag":
        return base + ["padding_tag", "available_visits", "mask_pattern"]
    if task == "v9_models":
        return base + ["eval_mode", "padding_length", "padding_tag", "available_visits"]
    if task in {"single_vs_multi_evaluator", "single_vs_multi_predictor"}:
        return base + ["eval_mode", "padding_length", "padding_tag", "target_visit", "available_visits"]
    if task == "generalization":
        return ["dataset_name", "center", "grade_type", "model_name", "eval_mode", "target_visit"]
    if task == "human_ai":
        return ["dataset_name", "center", "grade_type", "model_name", "assistance_mode", "reader_id", "target_visit"]
    return base


def should_pairwise(task: str) -> bool:
    return task in {
        "single_time", "predictor_length", "predictor_tag", "v9_models",
        "single_vs_multi_evaluator", "single_vs_multi_predictor", "human_ai"
    }


def run_task(
    df: pd.DataFrame,
    task: str,
    dataset_group: str,
    output_dir: str,
    prefix: str,
    n_bootstrap: int,
    seed: int,
    cluster_col: str = "patient_id",
    run_gee: bool = True,
    run_mixedlm: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    sub = select_dataset(df, dataset_group)
    sub = subset_task(sub, task)
    sub = ensure_columns(sub)
    if len(sub) == 0:
        print(f"[!] Empty subset: task={task}, dataset={dataset_group}")
        return
    sub.to_csv(os.path.join(output_dir, f"{prefix}_{task}_{dataset_group}_analysis_input.csv"), index=False)
    perf = performance_table(sub, performance_group_cols(task), cluster_col, n_bootstrap, seed)
    perf.to_csv(os.path.join(output_dir, f"{prefix}_{task}_{dataset_group}_performance.csv"), index=False)
    if should_pairwise(task):
        boot = pairwise_tests(sub, task=task if task not in {"single_time", "v9_models", "human_ai"} else "model", cluster_col=cluster_col, n_bootstrap=n_bootstrap, seed=seed)
        boot.to_csv(os.path.join(output_dir, f"{prefix}_{task}_{dataset_group}_pairwise_cluster_bootstrap.csv"), index=False)
        if run_gee or run_mixedlm:
            gee, mixed = regression_pairwise_tests(sub, task=task if task not in {"single_time", "v9_models", "human_ai"} else "model", cluster_col=cluster_col)
            if run_gee:
                gee.to_csv(os.path.join(output_dir, f"{prefix}_{task}_{dataset_group}_pairwise_gee.csv"), index=False)
            if run_mixedlm:
                mixed.to_csv(os.path.join(output_dir, f"{prefix}_{task}_{dataset_group}_pairwise_mixedlm.csv"), index=False)
