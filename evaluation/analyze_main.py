import argparse
import os

import pandas as pd

from analysis_core import ensure_columns, apply_view_selection, normalize_view_spec
from analysis_tasks import load_exported_predictions, run_task
from analysis_human import load_human_tables

TASKS = [
    "single_time",
    "evaluator_padding",
    "predictor_length",
    "predictor_tag",
    "v9_models",
    "single_vs_multi_evaluator",
    "single_vs_multi_predictor",
    "generalization",
]

VALID_VIEW_SPECS = [
    "D", "C", "N", "P",
    "DC", "DN", "DP", "CN", "CP", "NP",
    "DCN", "DCP", "DNP", "CNP", "DCNP",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze exported long prediction CSVs.")

    parser.add_argument("--input-root", type=str, default="file/long_predictions")
    parser.add_argument("--grade-type", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["internal", "external"])
    parser.add_argument("--dataset-groups", nargs="+", default=["internal", "S01", "S09"])
    parser.add_argument("--tasks", nargs="+", default=TASKS)

    parser.add_argument("--output-dir", type=str, default="file/stat_results")
    parser.add_argument("--output-prefix", type=str, default=None)

    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-col", type=str, default="patient_id")

    parser.add_argument("--skip-gee", action="store_true")
    parser.add_argument("--skip-mixedlm", action="store_true")

    parser.add_argument(
        "--snap-view",
        type=str,
        default="DCNP",
        choices=VALID_VIEW_SPECS,
        help=(
            "View selection for SnapRegressor. "
            "Use D/C/N/P or combinations such as CP, DCN, DCNP. "
            "If multiple views are given, their predictions are averaged. "
        ),
    )
    parser.add_argument(
        "--single-view",
        type=str,
        default="DCNP",
        choices=VALID_VIEW_SPECS,
        help=(
            "View selection for SingleViewTimeFusionRegressor. "
            "Use D/C/N/P or combinations such as CP, DCN, DCNP. "
            "If multiple views are given, their predictions are averaged. "
        ),
    )

    parser.add_argument("--extra-input", nargs="*", default=[], help="Extra prediction CSVs to append.")
    parser.add_argument(
        "--human-table",
        action="append",
        default=[],
        help="Human table spec: mode:path or mode:reader_id:path. Example: human_alone:R1:human_r1.xlsx",
    )
    parser.add_argument(
        "--human-truth-source",
        type=str,
        default=None,
        help="Optional prediction CSV used to attach y_true to human tables. Defaults to model prediction data.",
    )

    return parser.parse_args()


def _default_prefix(args) -> str:
    snap_view = normalize_view_spec(args.snap_view)
    single_view = normalize_view_spec(args.single_view)
    return f"{args.grade_type}_snap{snap_view}_single{single_view}"


def main():
    args = parse_args()

    prefix = args.output_prefix or _default_prefix(args)

    df = load_exported_predictions(
        args.input_root,
        args.grade_type,
        args.datasets
    )

    if args.extra_input:
        extra = [pd.read_csv(p) for p in args.extra_input if os.path.exists(p)]
        if extra:
            df = pd.concat([df] + extra, ignore_index=True)

    df = ensure_columns(df)

    # Build analysis-ready y_pred for single-view model families.
    # SnapRegressor and SingleViewTimeFusionRegressor export y_pred_D/C/N/P only.
    # This step selects one or multiple views and writes their mean into y_pred.
    df = apply_view_selection(
        df,
        snap_view=args.snap_view,
        single_view=args.single_view,
        rename_models=True,
    )

    if args.human_table:
        if args.human_truth_source:
            truth_df = pd.read_csv(args.human_truth_source)
            truth_df = ensure_columns(truth_df)
            truth_df = apply_view_selection(
                truth_df,
                snap_view=args.snap_view,
                single_view=args.single_view,
                rename_models=True,
            )
        else:
            truth_df = df

        human_df = load_human_tables(args.human_table, truth_df)
        df = ensure_columns(pd.concat([df, human_df], ignore_index=True))

        if "human_ai" not in args.tasks:
            args.tasks.append("human_ai")

    out_root = os.path.join(args.output_dir, args.grade_type)
    os.makedirs(out_root, exist_ok=True)

    df.to_csv(
        os.path.join(out_root, f"{prefix}_combined_analysis_source.csv"),
        index=False
    )

    for task in args.tasks:
        for dataset_group in args.dataset_groups:
            print("=" * 80)
            print(
                f"[*] task={task} | dataset_group={dataset_group} | "
                f"grade={args.grade_type} | snap_view={normalize_view_spec(args.snap_view)} | "
                f"single_view={normalize_view_spec(args.single_view)}"
            )
            print("=" * 80)

            run_task(
                df=df,
                task=task,
                dataset_group=dataset_group,
                output_dir=out_root,
                prefix=prefix,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                cluster_col=args.cluster_col,
                run_gee=(not args.skip_gee),
                run_mixedlm=(not args.skip_mixedlm),
            )

    print("[*] All analyses finished.")


if __name__ == "__main__":
    main()
