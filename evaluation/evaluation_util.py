import os
import json
import random
import numpy as np
import torch
from torchvision import transforms


def setSeed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _getTransforms(
        mean,
        std,
        resize_size=224,
        crop_size=224,
):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def getTransforms(
        resize_size=224,
        crop_size=224
):
    """
        D -> 0, C -> 1, N -> 2, P -> 3
    """
    mean_D = [0.3920, 0.3308, 0.2738]
    std_D = [0.3140, 0.2747, 0.2546]

    mean_C = [0.2646, 0.2010, 0.1370]
    std_C = [0.2935, 0.2254, 0.1877]

    mean_N = [0.3823, 0.3360, 0.2885]
    std_N = [0.3014, 0.2734, 0.2575]

    mean_P = [0.3537, 0.3030, 0.2510]
    std_P = [0.3044, 0.2691, 0.2507]


    transform_D = _getTransforms(mean_D, std_D, resize_size, crop_size)
    transform_C = _getTransforms(mean_C, std_C, resize_size, crop_size)
    transform_N = _getTransforms(mean_N, std_N, resize_size, crop_size)
    transform_P = _getTransforms(mean_P, std_P, resize_size, crop_size)

    return [transform_D, transform_C, transform_N, transform_P]


def get_split_path(args):
    if args.split_path is not None:
        return args.split_path

    center_tag = '_'.join(args.internal_lst)
    return os.path.join(
        'file',
        'data_split',
        f'patient_split_{center_tag}_grade{args.grade_type}_{args.num_folds}fold_seed{args.seed}.json'
    )


def load_patient_split(args):
    path = get_split_path(args)
    with open(path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    return {str(k): int(v) for k, v in split.items()}, path


def filter_by_fold(selectors, patient_to_fold, fold_idx):
    return [
        s for s in selectors
        if str(s.patient_id) in patient_to_fold
        and int(patient_to_fold[str(s.patient_id)]) == fold_idx
    ]


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def model_name(args):
    return f'{args.model_family}_{args.backbone_type}'


def snap_path(args, view, fold):
    name = model_name(args)
    return os.path.join(
        args.snap_root,
        name,
        'model',
        f'SnapRegressor_{name}_{args.grade_type}_photo{view}_{fold}.pt'
    )


def viewfusion_path(args, fold):
    name = model_name(args)
    return os.path.join(
        args.view_root,
        name,
        'model',
        f'ViewFusionRegressor_{name}_{args.grade_type}_{fold}.pt'
    )


def single_time_path(args, task_type, view, fold):
    name = model_name(args)
    return os.path.join(
        args.single_root,
        name,
        'model',
        f'SingleViewTimeFusionRegressor_{task_type}_{name}_{args.grade_type}_photo{view}_{fold}.pt'
    )


def multi_time_path(args, task_type, fold):
    name = model_name(args)
    return os.path.join(
        args.multi_root,
        name,
        'model',
        f'MultiViewTimeFusionRegressor_{task_type}_{name}_{args.grade_type}_{fold}.pt'
    )


def make_evaluator_mask(batch_size, seq_len, device):
    return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)


def make_predictor_mask(batch_size, valid_indices, device):
    mask = torch.ones(batch_size, 8, dtype=torch.bool, device=device)
    mask[:, valid_indices] = False
    return mask


def mask_pattern_from_valid(valid_indices, length):
    return ''.join(['1' if i in valid_indices else '0' for i in range(length)])


def visit_list_from_indices(indices):
    return ','.join([f'V{i + 1}' for i in indices])


def aggregate_final(df):
    extra_mean_cols = [
        c for c in df.columns
        if (
            c.startswith('view_weight_')
            or c.startswith('time_weight_')
            or c.startswith('y_pred_')
        )
    ]

    group_cols = [
        'dataset_name',
        'eval_mode',
        'model_name',
        'grade_type',
        'SeriesID',
        'center',
        'patient_id',
        'eye_id',
        'visit',
        'target_visit',
        'padding_length',
        'padding_tag',
        'available_visits',
        'mask_pattern',
        'repeat_idx',
    ]

    group_cols = [c for c in group_cols if c in df.columns]

    agg_dict = {
        'y_true': 'first',
        'fold': 'nunique',
        'y_pred': ['mean', 'std'],
    }

    for c in extra_mean_cols:
        agg_dict[c] = 'mean'

    out = df.groupby(group_cols, dropna=False).agg(agg_dict)
    out.columns = [
        '_'.join([x for x in col if x])
        for col in out.columns.to_flat_index()
    ]
    out = out.reset_index()

    out = out.rename(columns={
        'y_true_first': 'y_true',
        'fold_nunique': 'n_folds',
        'y_pred_mean': 'y_pred',
        'y_pred_std': 'y_pred_std',
    })

    if 'y_pred' in out.columns:
        out['abs_error'] = (out['y_pred'] - out['y_true']).abs()
        out['squared_error'] = (out['y_pred'] - out['y_true']) ** 2

    return out
