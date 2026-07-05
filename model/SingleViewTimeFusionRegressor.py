import torch
import torch.nn as nn
from torch import Tensor

from .util import FeaturesExtractor, AdjustableDropout, TimeSeriesTransformer


class SingleViewTimeFusionRegressor(nn.Module):
    """
    Model for processing longitudinal (time-series) data from single-view images.
    Can be configured as an 'evaluator' (assessing current state) or a 'predictor' (forecasting future state).
    """

    def __init__(
            self,
            task_type: str,
            grade_type: str,
            model_family: str,
            backbone_type: str,
            features_size: int = 256,
            num_heads: int = 4,
            hidden_size: int = 512,
            batch_first: bool = True,
            num_layers: int = 2,
    ) -> None:
        super().__init__()
        if task_type not in {"evaluator", "predictor"}:
            raise ValueError("task_type must be 'evaluator' or 'predictor'")

        self.task_type = task_type
        self.grade_type = grade_type
        self.model_family = model_family
        self.backbone_type = backbone_type
        self.features_size = features_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.extractor = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.time_transformer = TimeSeriesTransformer(
            self.features_size,
            self.num_heads,
            self.hidden_size,
            self.batch_first,
            self.num_layers,
        )
        self.adaptive_dropout = AdjustableDropout(initial_p=0.1)
        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.features_size, out_features=self.features_size // 16),
            nn.LeakyReLU(0.01),
            self.adaptive_dropout,
            nn.Linear(in_features=self.features_size // 16, out_features=1),
        )

    def forward(
            self,
            photos_series: Tensor,
            src_key_padding_mask: Tensor | None = None,
            is_causal: bool = True,
    ) -> tuple:
        """
        Args:
            photos_series (Tensor): Time-series image tensor [Batch, TimeSteps, Channels, Height, Width].
            src_key_padding_mask (Tensor, optional): Boolean mask for padded/missing time steps.
            is_causal (bool): Whether to apply causal masking in the transformer.
        Returns:
            tuple: Predicted grade and attention weights across time steps.
        """
        # If forecasting, drop the final observation to predict it
        if self.task_type == "predictor":
            photos_series = photos_series[:, :-1]

        batch_size, time_steps, C, H, W = photos_series.shape
        features_lst = []

        # Extract features for each time step individually
        for t in range(time_steps):
            feature_t = self.extractor(photos_series[:, t])
            features_lst.append(feature_t)

        features = torch.stack(features_lst, dim=1)  # [B, T, F]

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool().to(features.device)

        last_outputs, attn_weights = self.time_transformer(features, src_key_padding_mask, is_causal)
        grade = self.regressor(last_outputs)

        return grade, attn_weights