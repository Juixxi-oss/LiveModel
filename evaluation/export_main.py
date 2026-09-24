import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from dataset import getTimePointList, getTimePointsList
from evaluation_util import setSeed, getTransforms, load_patient_split, filter_by_fold, aggregate_final
from export_predictions import export_snap, export_viewfusion, export_single_time, export_multi_time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--dataset_name', type=str, required=True, choices=['internal', 'external'])
    parser.add_argument('--grade_type', type=str, required=True)

    parser.add_argument('--models', nargs='+', default=[
        'snap',
        'viewfusion',
    ])

    parser.add_argument('--tabular_path', type=str, default='data/label/label.xlsx')
    parser.add_argument('--photo_path', type=str, default='data/photo/*.jpg')

    parser.add_argument(
        '--internal_lst',
        nargs='+',
        default=['S02', 'S03', 'S05', 'S06', 'S08', 'S10', 'S11']
    )
    parser.add_argument(
        '--external_lst',
        nargs='*',
        default=['S01', 'S09']
    )

    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_path', type=str, default=None)

    parser.add_argument('--model_family', type=str, default='efficientnet')
    parser.add_argument('--backbone_type', type=str, default='b0')
    parser.add_argument('--features_size', type=int, default=256)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_first', action='store_true', default=True)

    parser.add_argument('--resize_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)

    parser.add_argument('--batch_size_single', type=int, default=64)
    parser.add_argument('--batch_size_series', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--output_root', type=str, default='file/long_predictions')

    parser.add_argument('--snap_root', type=str, default='file/SnapRegressor')
    parser.add_argument('--view_root', type=str, default='file/ViewFusionRegressor')
    parser.add_argument('--single_root', type=str, default='file/SingleViewTimeFusionRegressor')
    parser.add_argument('--multi_root', type=str, default='file/MultiViewTimeFusionRegressor')

    parser.add_argument('--num_1x_repeats', type=int, default=1)

    return parser.parse_args()



def main():
    args = parse_args()
    setSeed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    patient_to_fold, split_path = load_patient_split(args)
    print(f'[*] Using split file: {split_path}')
    print(f'[*] Using device: {device}')

    transform = getTransforms(args.resize_size, args.crop_size)

    if args.dataset_name == 'internal':
        center_lst = args.internal_lst
    else:
        center_lst = args.external_lst

    single_all = getTimePointList(
        args.tabular_path,
        args.photo_path,
        args.grade_type,
        center_lst=center_lst
    )

    series_all = getTimePointsList(
        args.tabular_path,
        args.photo_path,
        args.grade_type,
        center_lst=center_lst
    )

    all_rows = []

    for fold_idx in range(args.num_folds):
        if args.dataset_name == 'internal':
            single_selectors = filter_by_fold(single_all, patient_to_fold, fold_idx)
            series_selectors = filter_by_fold(series_all, patient_to_fold, fold_idx)
        else:
            single_selectors = single_all
            series_selectors = series_all

        print('=' * 80)
        print(f'[*] Fold {fold_idx + 1}/{args.num_folds}')
        print(f'    single samples : {len(single_selectors)}')
        print(f'    series samples : {len(series_selectors)}')
        print('=' * 80)

        if 'snap' in args.models:
            all_rows.extend(export_snap(args, single_selectors, fold_idx, device, transform))

        if 'viewfusion' in args.models:
            all_rows.extend(export_viewfusion(args, single_selectors, fold_idx, device, transform))

        if 'single_evaluator' in args.models:
            all_rows.extend(export_single_time(args, series_selectors, fold_idx, device, transform, 'evaluator'))

        if 'single_predictor' in args.models:
            all_rows.extend(export_single_time(args, series_selectors, fold_idx, device, transform, 'predictor'))

        if 'multi_evaluator' in args.models:
            all_rows.extend(export_multi_time(args, series_selectors, fold_idx, device, transform, 'evaluator'))

        if 'multi_predictor' in args.models:
            all_rows.extend(export_multi_time(args, series_selectors, fold_idx, device, transform, 'predictor'))

    df_fold = pd.DataFrame(all_rows)

    out_dir = os.path.join(args.output_root, args.grade_type, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    fold_path = os.path.join(out_dir, f'fold_long_predictions_{args.grade_type}_{args.dataset_name}.csv')
    final_path = os.path.join(out_dir, f'final_long_predictions_{args.grade_type}_{args.dataset_name}.csv')

    df_fold.to_csv(fold_path, index=False)

    df_final = aggregate_final(df_fold)
    df_final.to_csv(final_path, index=False)

    print('[*] Export finished.')
    print(f'    fold-level : {fold_path}')
    print(f'    final-level: {final_path}')


if __name__ == '__main__':
    main()




