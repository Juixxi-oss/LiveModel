import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader

from .BaseTrainer import BaseTrainer
from .util import setSeed, update_dropout
from .schedulers import build_snap_scheduler


class SnapRegressorTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.photo_type = kwargs.pop('photo_type')
        self.num_iters = kwargs.pop('num_iters', 0)
        # Opt-in extension: the default retains the historical Snap schedule.
        self.scheduler_config = {
            'scheduler_type': kwargs.pop('scheduler_type', 'none'),
            'warmup_epochs': kwargs.pop('warmup_epochs', 5),
            'start_factor': kwargs.pop('start_factor', 0.1),
            'T_0': kwargs.pop('T_0', 15),
            'T_mult': kwargs.pop('T_mult', 2),
            'eta_min': kwargs.pop('eta_min', 1e-7),
        }
        split_path = kwargs.pop('split_path', None)

        super().__init__(is_series_task=False, split_path=split_path, **kwargs)

        assert self.photo_type in ['D', 'C', 'N', 'P'], \
            f"photo_type must be one of ['D', 'C', 'N', 'P'], got {self.photo_type}"

        print('[*] SnapRegressorTrainer initialized.')
        print(f'    photo_type      : {self.photo_type}')
        print(f'    device          : {self.device}')
        print(f'    grade_type      : {self.grade_type}')
        print(f'    num_folds       : {self.num_folds}')
        print(f'    num_epochs      : {self.num_epochs}')
        print(f'    batch_size      : {self.batch_size}')
        print(f'    lr              : {self.lr}')
        print(f'    save_model_path : {self.save_model_path}')

    def init_optimizer(self) -> None:
        params_to_update = [
            param for param in self.model.parameters()
            if param.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.lr,
            weight_decay=1e-3
        )
        # Recreated with the optimizer at the start of every fold.
        self.scheduler = build_snap_scheduler(
            self.optimizer, **self.scheduler_config
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in params_to_update)

        print('[*] Optimizer initialized: AdamW')
        print(f'    trainable params: {trainable_params:,}')
        print(f'    total params    : {total_params:,}')
        print(f'    scheduler       : {self.scheduler_config}')

    def _select_photo(self, photos: torch.Tensor) -> torch.Tensor:
        """
        Select one view from photos.

        Input:
            photos: [B, 4, C, H, W]

        Output:
            selected photo: [B, C, H, W]

        View order:
            D -> 0
            C -> 1
            N -> 2
            P -> 3
        """
        dict_photos = {
            'D': photos[:, 0],
            'C': photos[:, 1],
            'N': photos[:, 2],
            'P': photos[:, 3]
        }

        return dict_photos[self.photo_type]

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> tuple:
        self.model.eval()

        total_loss1 = 0.0
        total_loss2 = 0.0
        total_count = 0

        for photos, grade in dataloader:
            photos = self._select_photo(photos).to(self.device)
            grade = grade.unsqueeze(1).to(self.device)

            prediction = self.model(photos)

            loss1 = F.smooth_l1_loss(
                prediction,
                grade,
                beta=0.5,
                reduction='sum'
            )

            loss2 = F.mse_loss(
                prediction,
                grade,
                reduction='sum'
            )

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_count += grade.shape[0]

        return (
            total_loss1 / max(total_count, 1),
            total_loss2 / max(total_count, 1)
        )

    def _eval(self, dataloader_lst: list[DataLoader]) -> tuple:
        loss_lst1 = []
        loss_lst2 = []

        for dataloader in dataloader_lst:
            smooth_l1, mse = self._eval_epoch(dataloader)
            loss_lst1.append(smooth_l1)
            loss_lst2.append(mse)

        return loss_lst1, loss_lst2

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()

        total_loss = 0.0
        total_count = 0

        for photos, grade in train_loader:
            photos = self._select_photo(photos).to(self.device)
            grade = grade.unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            pred_grade = self.model(photos)

            loss = F.smooth_l1_loss(
                pred_grade,
                grade,
                beta=0.5,
                reduction='mean'
            )

            loss.backward()
            self.optimizer.step()

            batch_size = grade.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / max(total_count, 1)

    def _train(self, fold_idx: int, dataloader_lst: list[DataLoader]) -> tuple:
        train_loader, val_loader = dataloader_lst

        best_loss = float('inf')
        best_model_weights = deepcopy(self.model.state_dict())

        save_model_path = f'{self.save_model_path}{fold_idx + 1}.pt'

        print('-' * 80)
        print(f'[*] Start training SnapRegressor | Fold {fold_idx + 1}/{self.num_folds}')
        print(f'    photo_type    : {self.photo_type}')
        print(f'    train batches : {len(train_loader)}')
        print(f'    val batches   : {len(val_loader)}')
        print(f'    save path     : {save_model_path}')
        print('-' * 80)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_lr = self.optimizer.param_groups[0]['lr']

            update_dropout(
                self.model,
                self.current_epoch,
                self.num_epochs
            )

            self._train_epoch(train_loader)
            loss_1, loss_2 = self._eval(dataloader_lst)

            train_smooth_l1 = loss_1[0]
            val_smooth_l1 = loss_1[1]

            train_mse = loss_2[0]
            val_mse = loss_2[1]

            if val_smooth_l1 < best_loss:
                best_loss = val_smooth_l1
                best_model_weights = deepcopy(self.model.state_dict())

                print(
                    f'    [Update] Fold {fold_idx + 1} | '
                    f'Epoch {epoch + 1:03d}/{self.num_epochs} | '
                    f'Best Val SmoothL1: {best_loss:.6f} | '
                    f'Val MSE: {val_mse:.6f}'
                )

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.num_epochs - 1:
                print(
                    f'    Fold {fold_idx + 1} | '
                    f'Epoch {epoch + 1:03d}/{self.num_epochs} | '
                    f'LR: {epoch_lr:.8g} | '
                    f'Train Loss: {train_smooth_l1:.6f} | '
                    f'Val Loss: {val_smooth_l1:.6f} | '
                    f'Train MSE: {train_mse:.6f} | '
                    f'Val MSE: {val_mse:.6f}'
                )

            if self.scheduler is not None:
                self.scheduler.step()

        used_time = time.time() - start_time

        self.save_model(
            best_model_weights,
            save_model_path,
            extra_data={
                'fold_idx': fold_idx,
                'photo_type': self.photo_type,
                'best_val_loss': best_loss,
                'grade_type': self.grade_type,
                'scheduler_config': dict(self.scheduler_config)
            }
        )

        print(f'[*] Fold {fold_idx + 1} finished.')
        print(f'    best val loss : {best_loss:.6f}')
        print(f'    model saved   : {save_model_path}')
        print(f'    elapsed time  : {used_time / 60:.2f} min')

        return best_loss

    def _evaluate(self, fold_idx: int, dataloader_lst: list[DataLoader]) -> tuple:
        model_path = f'{self.save_model_path}{fold_idx + 1}.pt'

        print(f'[*] Loading best model for Fold {fold_idx + 1}: {model_path}')
        self.load_model(model_path)

        loss_1, loss_2 = self._eval(dataloader_lst)

        print(
            f'[*] Fold {fold_idx + 1} final evaluation | '
            f'Train SmoothL1: {loss_1[0]:.6f} | '
            f'Val SmoothL1: {loss_1[1]:.6f} | '
            f'Train MSE: {loss_2[0]:.6f} | '
            f'Val MSE: {loss_2[1]:.6f}'
        )

        return loss_1[0], loss_1[1]

    def train_evaluate(self) -> None:
        setSeed(self.seed)

        print('=' * 80)
        print('[*] Start SnapRegressor training/evaluation')
        print(f'    seed       : {self.seed}')
        print(f'    folds      : {self.num_folds}')
        print(f'    photo_type : {self.photo_type}')
        print(f'    grade_type : {self.grade_type}')
        print('=' * 80)

        for fold_idx in range(self.num_folds):
            if fold_idx < self.start_fold:
                print(f'[*] Skip Fold {fold_idx + 1}, start_fold={self.start_fold}')
                continue

            print('=' * 80)
            print(f'[*] Preparing Fold {fold_idx + 1}/{self.num_folds}')
            print('=' * 80)

            self.load_model(self.load_model_path)
            self.init_optimizer()

            dataloader_lst = list(self.prepare_dataloader_1(fold_idx))

            self._train(
                fold_idx,
                dataloader_lst
            )

            self._evaluate(
                fold_idx,
                dataloader_lst
            )

        print('[*] SnapRegressor training/evaluation finished.')
