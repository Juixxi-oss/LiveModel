import os
import json
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedGroupKFold

from dataset import getTimePointList, getTimePointsList
from dataset import TimePointDataset, TimePointsDataset, getDataloader
from .util import setSeed


class BaseTrainer(ABC):
    """
    Base single-GPU trainer with StratifiedGroupKFold.

    This class keeps patient-level isolation and reproducible fold splits.
    """

    def __init__(self, is_series_task: bool, split_path: str = None, **kwargs) -> None:
        setSeed(kwargs.get('seed', 42))

        self.model = kwargs['model']
        self.device = next(self.model.parameters()).device

        self.load_model_path = kwargs.get('load_model_path', None)

        self.save_model_path = kwargs['save_model_path']
        self.lr = kwargs['lr']

        self.tabular_path = kwargs['tabular_path']
        self.photo_path = kwargs['photo_path']
        self.grade_type = kwargs['grade_type']

        self.num_folds = kwargs['num_folds']
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']
        self.seed = kwargs['seed']
        self.drop_last = kwargs['drop_last']

        self.transform_1 = kwargs['transform_1']
        self.transform_2 = kwargs['transform_2']
        self.internal_lst = kwargs['internal_lst']
        self.external_lst = kwargs.get('external_lst', [])

        self.start_fold = kwargs.get('start_fold', 0)
        self.current_epoch = 0

        self.is_series_task = is_series_task
        self.split_path = split_path

        self.internal_selector = None
        self.external_selector = []
        self.patient_to_fold = {}

        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)

        self.init_selector()

    def _build_or_load_patient_split(self) -> None:
        """
        Build or load the fixed patient-level fold split.
        """
        if self.split_path is None:
            split_root = os.path.join('file', 'data_split')
            center_tag = '_'.join(self.internal_lst)

            self.split_path = os.path.join(
                split_root,
                f'patient_split_{center_tag}_grade{self.grade_type}_{self.num_folds}fold_seed{self.seed}.json'
            )

        if os.path.exists(self.split_path):
            print(f'[*] Found existing patient split file: {self.split_path}')
            print('[*] Loading it to ensure full reproducibility.')

            with open(self.split_path, 'r', encoding='utf-8') as f:
                self.patient_to_fold = json.load(f)

            self.patient_to_fold = {
                str(pid): int(fold)
                for pid, fold in self.patient_to_fold.items()
            }
            return

        print(f'[*] No split file found. Creating a new split file: {self.split_path}')

        tabular_data = pd.read_excel(self.tabular_path)

        internal_mask = tabular_data['SeriesID'].astype(str).str[:3].isin(self.internal_lst)
        valid_data = tabular_data[internal_mask].dropna(subset=['SeriesID', self.grade_type])

        series_ids = valid_data['SeriesID'].astype(str)

        groups = series_ids.str[:6].values

        y_raw = valid_data[self.grade_type].values

        if np.issubdtype(y_raw.dtype, np.number) and len(np.unique(y_raw)) > 10:
            try:
                y = pd.qcut(
                    y_raw,
                    q=min(5, len(np.unique(y_raw))),
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                y = pd.cut(
                    y_raw,
                    bins=min(5, len(np.unique(y_raw))),
                    labels=False
                )

            y = np.asarray(y, dtype=np.int64)

        else:
            y = np.asarray(y_raw)

            if np.issubdtype(y.dtype, np.number):
                y = y.astype(np.int64)
            else:
                y = y.astype(str)

        print(f'[*] Stratification labels: {len(np.unique(y))} bins/classes')
        print(f'[*] Stratification dtype: {y.dtype}')

        X = np.zeros(len(y))

        sgkf = StratifiedGroupKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed
        )

        self.patient_to_fold = {}

        for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
            val_groups = groups[val_idx]

            for pid in val_groups:
                self.patient_to_fold[str(pid)] = int(fold_idx)

        os.makedirs(os.path.dirname(self.split_path), exist_ok=True)

        with open(self.split_path, 'w', encoding='utf-8') as f:
            json.dump(self.patient_to_fold, f, indent=4, ensure_ascii=False)

        print('[*] Patient split file saved successfully.')

    def init_selector(self) -> None:
        get_list_fn = getTimePointsList if self.is_series_task else getTimePointList

        internal_selector = get_list_fn(
            self.tabular_path,
            self.photo_path,
            self.grade_type,
            center_lst=self.internal_lst
        )
        self.internal_selector = np.array(internal_selector, dtype=object)

        self.external_selector = []

        for center in self.external_lst:
            selector = get_list_fn(
                self.tabular_path,
                self.photo_path,
                self.grade_type,
                center_lst=(center,)
            )
            self.external_selector.append(np.array(selector, dtype=object))

        self._build_or_load_patient_split()

    def prepare_dataloader_1(self, fold_idx: int) -> tuple:
        """
        Prepare train and validation dataloaders for the given fold.
        """
        train_patients = {
            str(pid)
            for pid, f in self.patient_to_fold.items()
            if int(f) != fold_idx
        }

        val_patients = {
            str(pid)
            for pid, f in self.patient_to_fold.items()
            if int(f) == fold_idx
        }

        train_selector = np.array(
            [
                item
                for item in self.internal_selector
                if str(item.patient_id) in train_patients
            ],
            dtype=object
        )

        val_selector = np.array(
            [
                item
                for item in self.internal_selector
                if str(item.patient_id) in val_patients
            ],
            dtype=object
        )

        DatasetClass = TimePointsDataset if self.is_series_task else TimePointDataset

        train_dataset = DatasetClass(train_selector, self.transform_1)
        val_dataset = DatasetClass(val_selector, self.transform_2)

        train_loader = getDataloader(
            train_dataset,
            self.batch_size,
            self.num_workers,
            self.seed,
            self.drop_last,
            True
        )

        val_loader = getDataloader(
            val_dataset,
            self.batch_size,
            self.num_workers,
            self.seed,
            False,
            False
        )

        print(f'[*] Fold {fold_idx + 1}: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}')

        return train_loader, val_loader

    def prepare_dataloader_2(self) -> list:
        """
        Prepare external dataloaders.
        """
        DatasetClass = TimePointsDataset if self.is_series_task else TimePointDataset

        loader_lst = []

        for selector_lst in self.external_selector:
            dataset = DatasetClass(selector_lst, self.transform_2)

            loader = getDataloader(
                dataset,
                self.batch_size,
                self.num_workers,
                self.seed,
                False,
                False
            )

            loader_lst.append(loader)

        return loader_lst

    def save_model(
            self,
            model_weights: dict,
            path: str,
            extra_data: dict = None
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'model_state': model_weights,
            'device': str(self.device)
        }

        if extra_data:
            save_dict.update(extra_data)

        torch.save(save_dict, path)

    def load_model(self, path: str) -> None:
        if path is None:
            print('[*] load_model_path is None. Skipping model loading.')
            return

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )

        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        print(f'[*] Model loaded from: {path}')

    @abstractmethod
    def init_optimizer(self):
        pass

    @abstractmethod
    def train_evaluate(self):
        pass