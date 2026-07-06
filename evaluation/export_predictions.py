import torch
import numpy as np

from model.SnapRegressor import SnapRegressor
from model.ViewFusionRegressor import ViewFusionRegressor
from model.SingleViewTimeFusionRegressor import SingleViewTimeFusionRegressor
from model.MultiViewTimeFusionRegressor import MultiViewTimeFusionRegressor

from evaluation_dataset import base_meta_from_series_id
from evaluation_dataset import SinglePointExportDataset, SeriesExportDataset
from evaluation_dataset import collate_single, collate_series, make_loader, make_record

from evaluation_util import snap_path, viewfusion_path, single_time_path, multi_time_path
from evaluation_util import load_checkpoint, make_evaluator_mask, make_predictor_mask
from evaluation_util import visit_list_from_indices, mask_pattern_from_valid

VIEW_ORDER = ['D', 'C', 'N', 'P']
VIEW_TO_IDX = {'D': 0, 'C': 1, 'N': 2, 'P': 3}


@torch.no_grad()
def export_snap(args, selectors, fold_idx, device, transform):
    dataset = SinglePointExportDataset(selectors, transform)
    loader = make_loader(dataset, args.batch_size_single, args.num_workers, collate_single)

    models = {}
    for view in VIEW_ORDER:
        m = SnapRegressor(args.model_family, args.backbone_type, args.features_size)
        models[view] = load_checkpoint(m, snap_path(args, view, fold_idx + 1), device)

    rows = []
    for batch in loader:
        photos = batch['photos'].to(device)
        grade = batch['grade'].cpu().numpy()
        preds = []
        for view in VIEW_ORDER:
            x = photos[:, VIEW_TO_IDX[view]]
            pred = models[view](x).squeeze(1).detach().cpu().numpy()
            preds.append(pred)
        for i, series_id in enumerate(batch['series_id']):
            _, _, _, visit = base_meta_from_series_id(series_id)
            rows.append(make_record(
                dataset_name=args.dataset_name,
                eval_mode='single_time_all_visits',
                model_name='SnapRegressor',
                grade_type=args.grade_type,
                fold=fold_idx + 1,
                series_id=series_id,
                target_visit=visit,
                y_true=grade[i],
                y_pred=None,
                view_predictions=[preds[j][i] for j in range(len(VIEW_ORDER))],
            ))

    return rows


@torch.no_grad()
def export_viewfusion(args, selectors, fold_idx, device, transform):
    dataset = SinglePointExportDataset(selectors, transform)
    loader = make_loader(dataset, args.batch_size_single, args.num_workers, collate_single)

    model = ViewFusionRegressor(
        args.grade_type,
        args.model_family,
        args.backbone_type,
        args.features_size
    )
    model = load_checkpoint(model, viewfusion_path(args, fold_idx + 1), device)

    rows = []
    for batch in loader:
        photos = batch['photos'].to(device)
        grade = batch['grade'].cpu().numpy()
        pred, weights = model(photos)
        pred = pred.squeeze(1).detach().cpu().numpy()
        weights = weights.detach().cpu().numpy()
        for i, series_id in enumerate(batch['series_id']):
            _, _, _, visit = base_meta_from_series_id(series_id)
            rows.append(make_record(
                dataset_name=args.dataset_name,
                eval_mode='single_time_all_visits',
                model_name='ViewFusionRegressor',
                grade_type=args.grade_type,
                fold=fold_idx + 1,
                series_id=series_id,
                target_visit=visit,
                y_true=grade[i],
                y_pred=pred[i],
                view_weights=weights[i],
            ))
    return rows


@torch.no_grad()
def export_single_time(args, selectors, fold_idx, device, transform, task_type):
    dataset = SeriesExportDataset(selectors, transform)
    loader = make_loader(dataset, args.batch_size_series, args.num_workers, collate_series)

    models = {}
    for view in VIEW_ORDER:
        m = SingleViewTimeFusionRegressor(
            task_type,
            args.grade_type,
            args.model_family,
            args.backbone_type,
            args.features_size,
            args.num_heads,
            args.hidden_size,
            args.batch_first,
            args.num_layers
        )
        models[view] = load_checkpoint(m, single_time_path(args, task_type, view, fold_idx + 1), device)

    rows = []
    for batch in loader:
        photos_series = batch['photos_series'].to(device)
        grade_series = batch['grade_series'].cpu().numpy()
        observed = batch['observed'].cpu().numpy()
        target_series_ids = batch['target_series_ids']
        batch_size = photos_series.size(0)

        if task_type == 'evaluator':
            for padding_length in range(9):
                target_idx = 8 - padding_length
                seq_len = target_idx + 1
                mask = make_evaluator_mask(batch_size, seq_len, device)
                view_preds = []
                view_weights = []

                for view in VIEW_ORDER:
                    x = photos_series[:, :seq_len, VIEW_TO_IDX[view]]
                    pred, weights = models[view](x, mask, True)
                    view_preds.append(pred.squeeze(1).detach().cpu().numpy())
                    view_weights.append(weights.detach().cpu().numpy())

                for i in range(batch_size):
                    if not observed[i, target_idx]:
                        continue
                    rows.append(make_record(
                        dataset_name=args.dataset_name,
                        eval_mode='evaluator_padding',
                        model_name='SingleViewTimeFusionRegressor',
                        grade_type=args.grade_type,
                        fold=fold_idx + 1,
                        series_id=target_series_ids[i][target_idx],
                        target_visit=f'V{target_idx + 1}',
                        y_true=grade_series[i, target_idx],
                        y_pred=None,
                        view_predictions=[view_preds[j][i] for j in range(len(VIEW_ORDER))],
                        padding_length=padding_length,
                        padding_tag=None,
                        available_visits=visit_list_from_indices(list(range(seq_len))),
                        mask_pattern='1' * seq_len,
                    ))
        else:
            specs = []
            for padding_length in range(8):
                input_end = 8 - padding_length
                valid_indices = list(range(input_end))
                specs.append(('predictor_padding_length', padding_length, None, valid_indices, 0))

            tag_specs = [
                # ('all', list(range(8))),
                ('half', [0, 4]),
                # ('1', [0]),
            ]

            for tag, valid_indices in tag_specs:
                specs.append(('predictor_padding_tag', None, tag, valid_indices, 0))
            for repeat_idx in range(args.num_1x_repeats):
                for j in range(1, 8):
                    specs.append(('predictor_padding_tag', None, '1x', [0, j], repeat_idx))
            for eval_mode, padding_length, padding_tag, valid_indices, repeat_idx in specs:
                mask = make_predictor_mask(batch_size, valid_indices, device)
                view_preds = []
                view_weights = []
                for view in VIEW_ORDER:
                    x = photos_series[:, :, VIEW_TO_IDX[view]]
                    pred, weights = models[view](x, mask, True)
                    view_preds.append(pred.squeeze(1).detach().cpu().numpy())
                    view_weights.append(weights.detach().cpu().numpy())

                for i in range(batch_size):
                    if not observed[i, 8]:
                        continue

                    rows.append(make_record(
                        dataset_name=args.dataset_name,
                        eval_mode=eval_mode,
                        model_name='SingleViewTimeFusionRegressor',
                        grade_type=args.grade_type,
                        fold=fold_idx + 1,
                        series_id=target_series_ids[i][8],
                        target_visit='V9',
                        y_true=grade_series[i, 8],
                        y_pred=None,
                        view_predictions=[view_preds[j][i] for j in range(len(VIEW_ORDER))],
                        padding_length=padding_length,
                        padding_tag=padding_tag,
                        available_visits=visit_list_from_indices(valid_indices),
                        mask_pattern=mask_pattern_from_valid(valid_indices, 8),
                        repeat_idx=repeat_idx,
                    ))
    return rows


@torch.no_grad()
def export_multi_time(args, selectors, fold_idx, device, transform, task_type):
    dataset = SeriesExportDataset(selectors, transform)
    loader = make_loader(dataset, args.batch_size_series, args.num_workers, collate_series)

    model = MultiViewTimeFusionRegressor(
        task_type,
        args.grade_type,
        args.model_family,
        args.backbone_type,
        args.features_size,
        args.num_heads,
        args.hidden_size,
        args.batch_first,
        args.num_layers
    )
    model = load_checkpoint(model, multi_time_path(args, task_type, fold_idx + 1), device)

    rows = []
    for batch in loader:
        photos_series = batch['photos_series'].to(device)
        grade_series = batch['grade_series'].cpu().numpy()
        observed = batch['observed'].cpu().numpy()
        target_series_ids = batch['target_series_ids']
        batch_size = photos_series.size(0)

        if task_type == 'evaluator':
            for padding_length in range(9):
                target_idx = 8 - padding_length
                seq_len = target_idx + 1

                mask = make_evaluator_mask(batch_size, seq_len, device)
                pred, weights = model(photos_series[:, :seq_len], mask, True)

                pred = pred.squeeze(1).detach().cpu().numpy()
                weights = weights.detach().cpu().numpy()
                for i in range(batch_size):
                    if not observed[i, target_idx]:
                        continue

                    rows.append(make_record(
                        dataset_name=args.dataset_name,
                        eval_mode='evaluator_padding',
                        model_name='MultiViewTimeFusionRegressor',
                        grade_type=args.grade_type,
                        fold=fold_idx + 1,
                        series_id=target_series_ids[i][target_idx],
                        target_visit=f'V{target_idx + 1}',
                        y_true=grade_series[i, target_idx],
                        y_pred=pred[i],
                        padding_length=padding_length,
                        available_visits=visit_list_from_indices(list(range(seq_len))),
                        mask_pattern='1' * seq_len,
                        time_weights=weights[i],
                    ))
        else:
            specs = []
            for padding_length in range(8):
                input_end = 8 - padding_length
                valid_indices = list(range(input_end))
                specs.append(('predictor_padding_length', padding_length, None, valid_indices, 0))

            tag_specs = [
                # ('all', list(range(8))),
                ('half', [0, 4]),
                # ('1', [0]),
            ]

            for tag, valid_indices in tag_specs:
                specs.append(('predictor_padding_tag', None, tag, valid_indices, 0))

            for repeat_idx in range(args.num_1x_repeats):
                for j in range(1, 8):
                    specs.append(('predictor_padding_tag', None, '1x', [0, j], repeat_idx))

            for eval_mode, padding_length, padding_tag, valid_indices, repeat_idx in specs:
                mask = make_predictor_mask(batch_size, valid_indices, device)
                pred, weights = model(photos_series, mask, True)

                pred = pred.squeeze(1).detach().cpu().numpy()
                weights = weights.detach().cpu().numpy()

                for i in range(batch_size):
                    if not observed[i, 8]:
                        continue

                    rows.append(make_record(
                        dataset_name=args.dataset_name,
                        eval_mode=eval_mode,
                        model_name='MultiViewTimeFusionRegressor',
                        grade_type=args.grade_type,
                        fold=fold_idx + 1,
                        series_id=target_series_ids[i][8],
                        target_visit='V9',
                        y_true=grade_series[i, 8],
                        y_pred=pred[i],
                        padding_length=padding_length,
                        padding_tag=padding_tag,
                        available_visits=visit_list_from_indices(valid_indices),
                        mask_pattern=mask_pattern_from_valid(valid_indices, 8),
                        repeat_idx=repeat_idx,
                        time_weights=weights[i],
                    ))
    return rows
