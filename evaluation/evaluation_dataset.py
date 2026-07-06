import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


VIEW_ORDER = ['D', 'C', 'N', 'P']
VIEW_TO_IDX = {'D': 0, 'C': 1, 'N': 2, 'P': 3}


def series_id_from_timepoint(tp):
    name = os.path.basename(tp.photo_paths.photo_pathD)
    return name.replace('_D.jpg', '')


def base_meta_from_series_id(series_id):
    center = series_id[:3]
    patient_id = series_id[:6]
    eye_id = series_id[:9]
    visit = f'V{series_id[-1]}'
    return center, patient_id, eye_id, visit


def replace_none(items) -> list:
    """
    Replaces 'None' values in a list using nearest neighbor imputation,
    prioritizing Last Observation Carried Forward (LOCF).
    """
    items = list(items)
    while None in items:
        for idx_1 in range(len(items)):
            if items[idx_1] is None:
                left = next(
                    (items[idx_2] for idx_2 in range(idx_1 - 1, -1, -1) if items[idx_2] is not None), None
                )
                right = next(
                    (items[idx_2] for idx_2 in range(idx_1 + 1, len(items)) if items[idx_2] is not None), None
                )
                if left is not None:
                    items[idx_1] = left
                elif right is not None:
                    items[idx_1] = right
    return items


class SinglePointExportDataset(Dataset):
    def __init__(self, selectors, transform):
        self.selectors = list(selectors)
        self.transform = transform

    def __len__(self):
        return len(self.selectors)

    def _load_view(self, path, view_idx):
        return self.transform[view_idx](Image.open(path).convert('RGB'))

    def __getitem__(self, idx):
        tp = self.selectors[idx]

        photos = torch.stack([
            self._load_view(tp.photo_paths.photo_pathD, 0),
            self._load_view(tp.photo_paths.photo_pathC, 1),
            self._load_view(tp.photo_paths.photo_pathN, 2),
            self._load_view(tp.photo_paths.photo_pathP, 3),
        ])

        series_id = series_id_from_timepoint(tp)

        return {
            'photos': photos,
            'grade': torch.tensor(float(tp.grade), dtype=torch.float32),
            'series_id': series_id,
        }


class SeriesExportDataset(Dataset):
    def __init__(self, selectors, transform):
        self.selectors = list(selectors)
        self.transform = transform

    def __len__(self):
        return len(self.selectors)

    def _load_point(self, tp):
        photos = torch.stack([
            self.transform[0](Image.open(tp.photo_paths.photo_pathD).convert('RGB')),
            self.transform[1](Image.open(tp.photo_paths.photo_pathC).convert('RGB')),
            self.transform[2](Image.open(tp.photo_paths.photo_pathN).convert('RGB')),
            self.transform[3](Image.open(tp.photo_paths.photo_pathP).convert('RGB')),
        ])
        grade = torch.tensor(float(tp.grade), dtype=torch.float32)
        return photos, grade

    def __getitem__(self, idx):
        obj = self.selectors[idx]
        raw_points = [getattr(obj, f'point_{i}') for i in range(1, 10)]
        observed = [p is not None for p in raw_points]
        filled_points = replace_none(raw_points)

        photos_series = []
        grade_series = []

        for p in filled_points:
            photos, grade = self._load_point(p)
            photos_series.append(photos)
            grade_series.append(grade)

        photos_series = torch.stack(photos_series)
        grade_series = torch.stack(grade_series)

        first_real = next(p for p in raw_points if p is not None)
        eye_id = series_id_from_timepoint(first_real)[:9]

        target_series_ids = []
        for i in range(9):
            if raw_points[i] is not None:
                target_series_ids.append(series_id_from_timepoint(raw_points[i]))
            else:
                target_series_ids.append(f'{eye_id}{i + 1}')

        return {
            'photos_series': photos_series,
            'grade_series': grade_series,
            'observed': torch.tensor(observed, dtype=torch.bool),
            'target_series_ids': target_series_ids,
        }


def collate_single(batch):
    return {
        'photos': torch.stack([b['photos'] for b in batch]),
        'grade': torch.stack([b['grade'] for b in batch]),
        'series_id': [b['series_id'] for b in batch],
    }


def collate_series(batch):
    return {
        'photos_series': torch.stack([b['photos_series'] for b in batch]),
        'grade_series': torch.stack([b['grade_series'] for b in batch]),
        'observed': torch.stack([b['observed'] for b in batch]),
        'target_series_ids': [b['target_series_ids'] for b in batch],
    }


def make_loader(dataset, batch_size, num_workers, collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        collate_fn=collate_fn,
    )


def make_record(
        dataset_name,
        eval_mode,
        model_name,
        grade_type,
        fold,
        series_id,
        target_visit,
        y_true,
        y_pred=None,
        padding_length=None,
        padding_tag=None,
        available_visits=None,
        mask_pattern=None,
        repeat_idx=None,
        view_weights=None,
        time_weights=None,
        view_predictions=None
):
    center, patient_id, eye_id, visit = base_meta_from_series_id(series_id)

    row = {
        'dataset_name': dataset_name,
        'eval_mode': eval_mode,
        'model_name': model_name,
        'grade_type': grade_type,
        'fold': fold,
        'SeriesID': series_id,
        'center': center,
        'patient_id': patient_id,
        'eye_id': eye_id,
        'visit': visit,
        'target_visit': target_visit,
        'padding_length': padding_length,
        'padding_tag': padding_tag,
        'available_visits': available_visits,
        'mask_pattern': mask_pattern,
        'repeat_idx': repeat_idx,
        'y_true': float(y_true),
        'y_pred': np.nan if y_pred is None else float(y_pred),
    }

    if y_pred is not None:
        row['abs_error'] = abs(row['y_pred'] - row['y_true'])
        row['squared_error'] = (row['y_pred'] - row['y_true']) ** 2
    else:
        row['abs_error'] = np.nan
        row['squared_error'] = np.nan

    if view_predictions is not None:
        for v, p in zip(VIEW_ORDER, view_predictions):
            row[f'y_pred_{v}'] = float(p)

    if view_weights is not None:
        for v, w in zip(VIEW_ORDER, view_weights):
            row[f'view_weight_{v}'] = float(w)

    if time_weights is not None:
        for i in range(9):
            row[f'time_weight_V{i + 1}'] = float(time_weights[i]) if i < len(time_weights) else np.nan

    return row
