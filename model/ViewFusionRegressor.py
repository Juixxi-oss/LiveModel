import torch
import torch.nn as nn
from torch import Tensor
from .util import FeaturesExtractor, CrossAttentionFusion, AdjustableDropout


class ViewFusionRegressor(nn.Module):
    """
    Model for predicting a grade from a single-time, multi-view (4 views) image set (Task 2).
    Utilizes four separate feature extractors and fuses them via Cross-Attention.

    Args:
        grade_type (str): The specific grading metric to evaluate (guides training strategy).
        model_family (str): The CNN architecture family.
        backbone_type (str): Specific version of the architecture.
        features_size (int): Dimension of the extracted feature vector.
    """

    def __init__(
            self,
            grade_type: str,
            model_family: str,
            backbone_type: str,
            features_size: int = 256,
    ) -> None:
        super().__init__()
        self.grade_type = grade_type
        self.model_family = model_family
        self.backbone_type = backbone_type
        self.features_size = features_size

        # Independent feature extractors for each of the 4 views
        self.extractor_1 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_2 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_3 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.extractor_4 = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)

        self.cs_fusion = CrossAttentionFusion(self.features_size)
        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.features_size, out_features=self.features_size // 16),
            nn.LeakyReLU(0.01),
            AdjustableDropout(initial_p=0.1),
            nn.Linear(in_features=self.features_size // 16, out_features=1),
        )

    def forward(self, photos: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            photos (Tensor): Multi-view image tensor of shape [Batch, 4, Channels, Height, Width].
        Returns:
            tuple: Predicted grade [Batch, 1] and attention weights [Batch, 4].
        """
        feature_1 = self.extractor_1(photos[:, 0])
        feature_2 = self.extractor_2(photos[:, 1])
        feature_3 = self.extractor_3(photos[:, 2])
        feature_4 = self.extractor_4(photos[:, 3])

        stack_features = torch.stack([feature_1, feature_2, feature_3, feature_4], dim=1)
        fused_features, weights = self.cs_fusion(stack_features)

        grade = self.regressor(fused_features)
        return grade, weights