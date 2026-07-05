from .util import *


class MultiViewTimeFusionRegressor(nn.Module):
    """
    Comprehensive model for evaluating a grade from longitudinal (multi-time) and
    multi-view (4 views) images (Task 3).
    Combines spatial cross-attention fusion (for views) with a temporal transformer (for time steps).
    Supports 'evaluator' and 'predictor' task modes.
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
            num_layers: int = 2
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

        # Independent feature extractors for the 4 views
        self.extractor_1 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_2 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_3 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_4 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)

        self.cs_fusion = CrossAttentionFusion(self.features_size)
        self.time_transformer = TimeSeriesTransformer(
            self.features_size,
            self.num_heads,
            self.hidden_size,
            self.batch_first,
            self.num_layers
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
            is_causal: bool = True
    ) -> tuple:
        """
        Args:
            photos_series (Tensor): Time-series multi-view tensor [Batch, TimeSteps, 4, Channels, Height, Width].
            src_key_padding_mask (Tensor, optional): Boolean mask for time steps to ignore.
            is_causal (bool): Autoregressive indicator for temporal sequence.
        Returns:
            tuple: Predicted grade [Batch, 1] and temporal attention weights.
        """
        if self.task_type == "predictor":
            photos_series = photos_series[:, :-1]

        batch_size, time_steps, num_photos, C, H, W = photos_series.shape
        fused_features_lst = []

        # Process multi-view spatial fusion for each time step
        for t in range(time_steps):
            feature_1 = self.extractor_1(photos_series[:, t, 0])
            feature_2 = self.extractor_2(photos_series[:, t, 1])
            feature_3 = self.extractor_3(photos_series[:, t, 2])
            feature_4 = self.extractor_4(photos_series[:, t, 3])

            stack_features = torch.stack([feature_1, feature_2, feature_3, feature_4], dim=1)
            fused_features, weights = self.cs_fusion(stack_features)
            fused_features_lst.append(fused_features)

        fused_features = torch.stack(fused_features_lst, dim=1)  # [B, T, F]

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool().to(fused_features.device)

        last_outputs, attn_weights = self.time_transformer(fused_features, src_key_padding_mask, is_causal)
        grade = self.regressor(last_outputs)

        return grade, attn_weights