from copy import deepcopy
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .BaseTrainer import BaseTrainer
from .util import setSeed, get_suplambda, generate_attention_1


class ViewFusionRegressorTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        # Scheduler and attention lambda parameters
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

        super().__init__(is_series_task=False, split_path=split_path, **kwargs)

        self.attention_lamda = 0.0

        self.model_family = self.model.model_family
        self.backbone_type = self.model.backbone_type
        self.features_size = self.model.features_size

        print('[*] ViewFusionRegressorTrainer initialized.')
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
            {'params': self.model.extractor_1.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.model.extractor_2.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.model.extractor_3.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.model.extractor_4.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.model.cs_fusion.parameters(), 'lr': self.lr},
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
        print(f'    extractor lr   : {0.1 * self.lr}')
        print(f'    fusion/head lr : {self.lr}')
        print('    scheduler      : LinearLR warmup')

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> tuple:
        self.model.eval()

        total_loss1 = 0.0
        total_loss2 = 0.0
        total_count = 0
        weights_lst = []

        for photos, grade in dataloader:
            photos = photos.to(self.device)
            grade = grade.unsqueeze(1).to(self.device)

            prediction, weights = self.model(photos)

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

            weights_lst.append(weights.detach().cpu())

        if len(weights_lst) > 0:
            w = torch.cat(weights_lst, dim=0).sum(dim=0)
            w = w / (w.sum() + 1e-8)
        else:
            w = torch.zeros(4)

        return (
            total_loss1 / max(total_count, 1),
            total_loss2 / max(total_count, 1),
            w
        )

    def _eval(self, dataloader_lst: list[DataLoader]) -> tuple:
        loss_lst1 = []
        loss_lst2 = []
        weights_lst = []

        for dataloader in dataloader_lst:
            loss1, loss2, weights = self._eval_epoch(dataloader)
            loss_lst1.append(loss1)
            loss_lst2.append(loss2)
            weights_lst.append(weights)

        return loss_lst1, loss_lst2, weights_lst

    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        self.model.train()

        total_loss = 0.0
        total_main_loss = 0.0
        total_attention_loss = 0.0
        total_count = 0

        for photos, grade in train_loader:
            photos = photos.to(self.device)
            grade = grade.unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            pred_grade, weights = self.model(photos)

            main_loss = F.smooth_l1_loss(
                pred_grade,
                grade,
                beta=0.5,
                reduction='mean'
            )

            attention_label = generate_attention_1(
                self.grade_type,
                weights.size(0),
                self.device
            )

            attention_loss = F.kl_div(
                (weights + 1e-8).log(),
                attention_label,
                reduction='batchmean'
            )

            loss = main_loss + self.attention_lamda * attention_loss

            loss.backward()
            self.optimizer.step()

            batch_size = grade.shape[0]

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
        best_model_weights = deepcopy(self.model.state_dict())
        update_tag = 0

        save_model_path = f'{self.save_model_path}{fold_idx + 1}.pt'

        print('-' * 80)
        print(f'[*] Start training ViewFusionRegressor | Fold {fold_idx + 1}/{self.num_folds}')
        print(f'    train batches : {len(train_loader)}')
        print(f'    val batches   : {len(val_loader)}')
        print(f'    save path     : {save_model_path}')
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
                train_loader
            )

            loss_1, loss_2, weights = self._eval(dataloader_lst)

            train_smooth_l1 = loss_1[0]
            val_smooth_l1 = loss_1[1]
            train_mse = loss_2[0]
            val_mse = loss_2[1]

            if val_smooth_l1 < best_loss:
                best_loss = val_smooth_l1
                best_model_weights = deepcopy(self.model.state_dict())
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
                train_w = weights[0].tolist()
                val_w = weights[1].tolist()

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

                print(
                    f'        Train View Weights: '
                    f'D={train_w[0]:.4f}, C={train_w[1]:.4f}, '
                    f'N={train_w[2]:.4f}, P={train_w[3]:.4f}'
                )

                print(
                    f'        Val View Weights  : '
                    f'D={val_w[0]:.4f}, C={val_w[1]:.4f}, '
                    f'N={val_w[2]:.4f}, P={val_w[3]:.4f}'
                )

        used_time = time.time() - start_time

        self.save_model(
            best_model_weights,
            save_model_path,
            extra_data={
                'fold_idx': fold_idx,
                'best_val_loss': best_loss,
                'grade_type': self.grade_type
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

        loss_1, loss_2, weights = self._eval(dataloader_lst)

        train_w = weights[0].tolist()
        val_w = weights[1].tolist()

        print(
            f'[*] Fold {fold_idx + 1} final evaluation | '
            f'Train SmoothL1: {loss_1[0]:.6f} | '
            f'Val SmoothL1: {loss_1[1]:.6f} | '
            f'Train MSE: {loss_2[0]:.6f} | '
            f'Val MSE: {loss_2[1]:.6f}'
        )

        print(
            f'    Final Train View Weights: '
            f'D={train_w[0]:.4f}, C={train_w[1]:.4f}, '
            f'N={train_w[2]:.4f}, P={train_w[3]:.4f}'
        )

        print(
            f'    Final Val View Weights  : '
            f'D={val_w[0]:.4f}, C={val_w[1]:.4f}, '
            f'N={val_w[2]:.4f}, P={val_w[3]:.4f}'
        )

        return loss_1[0], loss_1[1]

    def load_train_model(self, path: str, fold_idx: int) -> None:
        """
        Load full ViewFusion checkpoint if provided.
        Otherwise, load four pretrained SnapRegressor extractors.
        """
        if path is not None:
            print(f'[*] Loading ViewFusion initial model from: {path}')
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(checkpoint['model_state'])
            print('[*] Full ViewFusion model loaded.')
            # return

        def _strip(d: dict, p: str) -> dict:
            return {
                k[len(p):]: v
                for k, v in d.items()
                if k.startswith(p)
            }

        fold_no = fold_idx + 1

        _name = f'{self.model_family}_{self.backbone_type}'
        _prefix = f'file/SnapRegressor/{_name}/model/SnapRegressor_{_name}'

        model_D = f'{_prefix}_{self.grade_type}_photoD_{fold_no}.pt'
        model_C = f'{_prefix}_{self.grade_type}_photoC_{fold_no}.pt'
        model_N = f'{_prefix}_{self.grade_type}_photoN_{fold_no}.pt'
        model_P = f'{_prefix}_{self.grade_type}_photoP_{fold_no}.pt'

        # print('[*] load_model_path is None.')
        print('[*] Loading pretrained single-view extractors:')
        print(f'    D: {model_D}')
        print(f'    C: {model_C}')
        print(f'    N: {model_N}')
        print(f'    P: {model_P}')

        ext_D = _strip(
            torch.load(model_D, map_location=self.device, weights_only=True)['model_state'],
            'extractor.'
        )
        ext_C = _strip(
            torch.load(model_C, map_location=self.device, weights_only=True)['model_state'],
            'extractor.'
        )
        ext_N = _strip(
            torch.load(model_N, map_location=self.device, weights_only=True)['model_state'],
            'extractor.'
        )
        ext_P = _strip(
            torch.load(model_P, map_location=self.device, weights_only=True)['model_state'],
            'extractor.'
        )

        self.model.extractor_1.load_state_dict(ext_D, strict=True)
        self.model.extractor_2.load_state_dict(ext_C, strict=True)
        self.model.extractor_3.load_state_dict(ext_N, strict=True)
        self.model.extractor_4.load_state_dict(ext_P, strict=True)

        print('[*] Pretrained single-view extractors loaded successfully.')

    def train_evaluate(self) -> None:
        setSeed(self.seed)

        print('=' * 80)
        print('[*] Start ViewFusionRegressor training/evaluation')
        print(f'    seed       : {self.seed}')
        print(f'    folds      : {self.num_folds}')
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

        print('[*] ViewFusionRegressor training/evaluation finished.')