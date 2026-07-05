from copy import deepcopy
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .BaseTrainer import BaseTrainer
from .util import setSeed, get_suplambda, generate_attention_2


class SingleViewTimeFusionRegressorTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.photo_type = kwargs.pop('photo_type')

        self.start_factor = kwargs.pop('start_factor')
        self.end_factor = kwargs.pop('end_factor')
        self.total_iters = kwargs.pop('total_iters')
        self.T_0 = kwargs.pop('T_0')
        self.T_mult = kwargs.pop('T_mult')
        self.eta_min = kwargs.pop('eta_min')

        self.warmup_epochs = kwargs.pop('warmup_epochs')
        self.stable_epochs = kwargs.pop('stable_epochs')
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.max_lambda = kwargs.pop('max_lambda')
        self.min_lambda = kwargs.pop('min_lambda')

        self.num_iters = kwargs.pop('num_iters', 0)
        split_path = kwargs.pop('split_path', None)

        super().__init__(is_series_task=True, split_path=split_path, **kwargs)

        assert self.photo_type in ['D', 'C', 'N', 'P'], \
            f"photo_type must be one of ['D', 'C', 'N', 'P'], got {self.photo_type}"

        self.task_type = self.model.task_type
        self.attention_lamda = 0.0

        self.model_family = self.model.model_family
        self.backbone_type = self.model.backbone_type
        self.features_size = self.model.features_size

        print('[*] SingleViewTimeFusionRegressorTrainer initialized.')
        print(f'    task_type       : {self.task_type}')
        print(f'    photo_type      : {self.photo_type}')
        print(f'    device          : {self.device}')
        print(f'    model_family    : {self.model_family}')
        print(f'    backbone_type   : {self.backbone_type}')
        print(f'    features_size   : {self.features_size}')
        print(f'    grade_type      : {self.grade_type}')
        print(f'    num_folds       : {self.num_folds}')
        print(f'    num_epochs      : {self.num_epochs}')
        print(f'    batch_size      : {self.batch_size}')
        print(f'    lr              : {self.lr}')
        print(f'    max_lambda      : {self.max_lambda}')
        print(f'    min_lambda      : {self.min_lambda}')
        print(f'    save_model_path : {self.save_model_path}')

    def init_optimizer(self) -> None:
        params_to_update = [
            {'params': self.model.extractor.parameters(), 'lr': 0.2 * self.lr},
            {'params': self.model.time_transformer.parameters(), 'lr': self.lr},
            {'params': self.model.regressor.parameters(), 'lr': self.lr},
        ]

        self.optimizer = torch.optim.AdamW(
            params_to_update,
            weight_decay=1e-3
        )

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            self.start_factor,
            self.end_factor,
            self.total_iters
        )

        print('[*] Optimizer initialized: AdamW')
        print(f'    extractor lr        : {0.2 * self.lr}')
        print(f'    time_transformer lr : {self.lr}')
        print(f'    regressor lr        : {self.lr}')
        print('    scheduler           : LinearLR warmup')

    def _select_photo(self, photos: torch.Tensor) -> torch.Tensor:
        """
        Select one view from a time-series sample.

        Input:
            photos: [B, T, 4, C, H, W]

        Output:
            selected photos: [B, T, C, H, W]

        View order:
            D -> 0
            C -> 1
            N -> 2
            P -> 3
        """
        dict_photos = {
            'D': photos[:, :, 0],
            'C': photos[:, :, 1],
            'N': photos[:, :, 2],
            'P': photos[:, :, 3],
        }

        return dict_photos[self.photo_type]

    def _make_mask_and_grade(self, grade, batch_size, time_steps):
        """
        Create random time mask and target grade.

        evaluator:
            Use all T time points. The target is the last visible time point.

        predictor:
            The model internally uses photos_series[:, :-1].
            The target is the last real time point.
        """
        if self.task_type == 'evaluator':
            seq_len = time_steps
        elif self.task_type == 'predictor':
            seq_len = time_steps - 1
        else:
            raise ValueError(f'Unknown task_type: {self.task_type}')

        src_key_padding_mask = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.float32,
            device=self.device
        )

        for i in range(batch_size):
            valid_indices = torch.arange(1, seq_len, device=self.device)

            if len(valid_indices) == 0:
                continue

            if torch.rand(1).item() < 0.75:
                high = min(5, seq_len)
                num_pad = torch.randint(1, high, (1,)).item() if high > 1 else 1
            else:
                num_pad = torch.randint(5, seq_len, (1,)).item() if seq_len > 5 else 1

            num_pad = min(num_pad, len(valid_indices))

            pad_indices = valid_indices[
                torch.randperm(len(valid_indices), device=self.device)[:num_pad]
            ]

            src_key_padding_mask[i, pad_indices] = 1

        src_key_padding_mask = src_key_padding_mask.bool()

        if self.task_type == 'evaluator':
            indices = torch.arange(
                seq_len,
                device=self.device
            ).expand_as(src_key_padding_mask)

            valid_indices = torch.where(
                ~src_key_padding_mask,
                indices,
                torch.tensor(-1, device=self.device)
            )

            last_valid_indices = valid_indices.max(dim=1)[0].unsqueeze(1)
            target_grade = grade.gather(1, last_valid_indices)

        else:
            target_grade = grade[:, -1].unsqueeze(1)

        return src_key_padding_mask, target_grade

    @torch.no_grad()
    def _eval_epoch(self, epoch: int, dataloader: DataLoader) -> tuple:
        self.model.eval()
        setSeed(epoch)

        total_loss1 = 0.0
        total_loss2 = 0.0
        total_count = 0
        weights_lst = []

        for photos, grade in dataloader:
            photos = photos.to(self.device)
            grade = grade.to(self.device)

            photos = self._select_photo(photos)

            batch_size = photos.shape[0]
            time_steps = photos.shape[1]

            mask, target_grade = self._make_mask_and_grade(
                grade,
                batch_size,
                time_steps
            )

            prediction, weights = self.model(photos, mask)

            loss1 = F.smooth_l1_loss(
                prediction,
                target_grade,
                beta=0.5,
                reduction='sum'
            )

            loss2 = F.mse_loss(
                prediction,
                target_grade,
                reduction='sum'
            )

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_count += target_grade.shape[0]

            weights_lst.append(weights.sum(dim=0).detach().cpu())

        if len(weights_lst) > 0:
            w = torch.stack(weights_lst).sum(dim=0)
            w = w / (w.sum() + 1e-8)
        else:
            w = torch.zeros(1)

        return (
            total_loss1 / max(total_count, 1),
            total_loss2 / max(total_count, 1),
            w
        )

    def _eval(self, epoch: int, dataloader_lst: list[DataLoader]) -> tuple:
        loss_lst1 = []
        loss_lst2 = []
        weights_lst = []

        for dataloader in dataloader_lst:
            loss1, loss2, weights = self._eval_epoch(epoch, dataloader)
            loss_lst1.append(loss1)
            loss_lst2.append(loss2)
            weights_lst.append(weights)

        return loss_lst1, loss_lst2, weights_lst

    def _train_epoch(self, epoch: int, train_loader: DataLoader) -> tuple:
        self.model.train()
        setSeed(epoch)

        total_loss = 0.0
        total_main_loss = 0.0
        total_attention_loss = 0.0
        total_count = 0

        for photos, grade in train_loader:
            photos = photos.to(self.device)
            grade = grade.to(self.device)

            photos = self._select_photo(photos)

            batch_size = photos.shape[0]
            time_steps = photos.shape[1]

            mask, target_grade = self._make_mask_and_grade(
                grade,
                batch_size,
                time_steps
            )

            self.optimizer.zero_grad()

            pred_grade, weights = self.model(
                photos,
                mask,
                True
            )

            main_loss = F.smooth_l1_loss(
                pred_grade,
                target_grade,
                beta=0.5,
                reduction='mean'
            )

            attention_label = generate_attention_2(
                mask.size(0),
                mask.size(1),
                mask,
                weights.device
            )

            attention_loss = F.kl_div(
                (weights + 1e-8).log(),
                attention_label,
                reduction='batchmean'
            )

            loss = main_loss + self.attention_lamda * attention_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_main_loss += main_loss.item() * batch_size
            total_attention_loss += attention_loss.item() * batch_size
            total_count += batch_size

        return (
            total_loss / max(total_count, 1),
            total_main_loss / max(total_count, 1),
            total_attention_loss / max(total_count, 1)
        )

    def _train(self, fold_idx: int, dataloader_lst: list[DataLoader]) -> float:
        train_loader, val_loader = dataloader_lst

        best_loss = float('inf')
        best_weights = deepcopy(self.model.state_dict())
        update_tag = 0

        save_path = f'{self.save_model_path}{fold_idx + 1}.pt'

        print('-' * 80)
        print(
            f'[*] Start training SingleViewTimeFusionRegressor | '
            f'Fold {fold_idx + 1}/{self.num_folds}'
        )
        print(f'    task_type     : {self.task_type}')
        print(f'    photo_type    : {self.photo_type}')
        print(f'    train batches : {len(train_loader)}')
        print(f'    val batches   : {len(val_loader)}')
        print(f'    save path     : {save_path}')
        print('-' * 80)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            self.attention_lamda = get_suplambda(
                self.current_epoch,
                self.warmup_epochs,
                self.stable_epochs,
                self.decay_epochs,
                self.max_lambda,
                self.min_lambda
            )

            train_loss, train_main_loss, train_attention_loss = self._train_epoch(
                epoch,
                train_loader
            )

            loss_1, loss_2, weights = self._eval(
                epoch,
                dataloader_lst
            )

            train_smooth_l1 = loss_1[0]
            val_smooth_l1 = loss_1[1]
            train_mse = loss_2[0]
            val_mse = loss_2[1]

            if val_smooth_l1 < best_loss:
                best_loss = val_smooth_l1
                best_weights = deepcopy(self.model.state_dict())
                update_tag = 0

                print(
                    f'    [Update] Fold {fold_idx + 1} | '
                    f'Epoch {epoch + 1:03d}/{self.num_epochs} | '
                    f'Best Val SmoothL1: {best_loss:.6f} | '
                    f'Val MSE: {val_mse:.6f}'
                )
            else:
                if epoch >= self.total_iters:
                    update_tag += 1

            if epoch < self.total_iters:
                self.scheduler.step()

            elif epoch == self.total_iters:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.T_0,
                    T_mult=self.T_mult,
                    eta_min=self.eta_min
                )
                self.scheduler.step()
                print('[*] Scheduler switched to CosineAnnealingWarmRestarts.')

            elif epoch > self.total_iters and update_tag >= 2:
                self.scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.num_epochs - 1:
                print(
                    f'    Fold {fold_idx + 1} | '
                    f'Epoch {epoch + 1:03d}/{self.num_epochs} | '
                    f'Lambda: {self.attention_lamda:.6f} | '
                    f'Train Loss: {train_smooth_l1:.6f} | '
                    f'Val Loss: {val_smooth_l1:.6f} | '
                    f'Train MSE: {train_mse:.6f} | '
                    f'Val MSE: {val_mse:.6f} | '
                    f'Main Loss: {train_main_loss:.6f} | '
                    f'Attn Loss: {train_attention_loss:.6f}'
                )

                print(f'        Train Time Weights: {weights[0].tolist()}')
                print(f'        Val Time Weights  : {weights[1].tolist()}')

        used_time = time.time() - start_time

        self.save_model(
            best_weights,
            save_path,
            extra_data={
                'fold_idx': fold_idx,
                'best_val_loss': best_loss,
                'task_type': self.task_type,
                'photo_type': self.photo_type,
                'grade_type': self.grade_type
            }
        )

        print(f'[*] Fold {fold_idx + 1} finished.')
        print(f'    best val loss : {best_loss:.6f}')
        print(f'    model saved   : {save_path}')
        print(f'    elapsed time  : {used_time / 60:.2f} min')

        return best_loss

    def _evaluate(self, fold_idx: int, dataloader_lst: list[DataLoader]) -> tuple:
        model_path = f'{self.save_model_path}{fold_idx + 1}.pt'

        print(f'[*] Loading best model for Fold {fold_idx + 1}: {model_path}')
        self.load_model(model_path)

        loss_1, loss_2, weights = self._eval(42, dataloader_lst)

        print(
            f'[*] Fold {fold_idx + 1} final evaluation | '
            f'Train SmoothL1: {loss_1[0]:.6f} | '
            f'Val SmoothL1: {loss_1[1]:.6f} | '
            f'Train MSE: {loss_2[0]:.6f} | '
            f'Val MSE: {loss_2[1]:.6f}'
        )

        print(f'    Final Train Time Weights: {weights[0].tolist()}')
        print(f'    Final Val Time Weights  : {weights[1].tolist()}')

        return loss_1[0], loss_1[1]

    def load_train_model(self, path: str, fold_idx: int) -> None:
        """
        Load full SingleViewTimeFusion checkpoint if provided.
        Otherwise, load the pretrained SnapRegressor extractor for the selected photo_type.
        """
        if path is not None:
            print(f'[*] Loading SingleViewTimeFusion initial model from: {path}')
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(checkpoint['model_state'])
            print('[*] Full SingleViewTimeFusion model loaded.')
            return

        fold_no = fold_idx + 1

        _name = f'{self.model_family}_{self.backbone_type}'
        _prefix = f'file/SnapRegressor/{_name}/model/SnapRegressor_{_name}'

        snap_model_path = (
            f'{_prefix}_{self.grade_type}_photo{self.photo_type}_{fold_no}.pt'
        )

        print('[*] load_model_path is None.')
        print('[*] Loading pretrained SnapRegressor extractor:')
        print(f'    Snap model: {snap_model_path}')

        try:
            state = torch.load(
                snap_model_path,
                map_location=self.device,
                weights_only=True
            )['model_state']

            extractor_state = {
                k[len('extractor.'):]: v
                for k, v in state.items()
                if k.startswith('extractor.')
            }

            self.model.extractor.load_state_dict(
                extractor_state,
                strict=True
            )

            print('[*] SnapRegressor extractor loaded successfully.')

        except FileNotFoundError:
            print('[!] SnapRegressor checkpoint not found. Training from current initialization.')

        except Exception as e:
            print('[!] Failed to load SnapRegressor extractor. Training from current initialization.')
            print(f'    Error: {e}')

    def train_evaluate(self) -> None:
        setSeed(self.seed)

        print('=' * 80)
        print('[*] Start SingleViewTimeFusionRegressor training/evaluation')
        print(f'    seed       : {self.seed}')
        print(f'    folds      : {self.num_folds}')
        print(f'    task_type  : {self.task_type}')
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

            self.load_train_model(self.load_model_path, fold_idx)
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

        print('[*] SingleViewTimeFusionRegressor training/evaluation finished.')