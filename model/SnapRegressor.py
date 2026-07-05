import torch
import torch.nn as nn
from torch import Tensor
from .util import FeaturesExtractor, AdjustableDropout


class SnapRegressor(nn.Module):
    """
    Model for predicting a grade from a single-time, single-view image (Task 1).

    Args:
        model_family (str): The CNN architecture family (e.g., 'ResNet', 'EfficientNet').
        backbone_type (str): Specific version of the architecture.
        features_size (int): Dimension of the extracted feature vector.
    """

    def __init__(
            self,
            model_family: str,
            backbone_type: str,
            features_size: int = 256
    ) -> None:
        super().__init__()
        self.model_family = model_family
        self.backbone_type = backbone_type
        self.features_size = features_size
        self.extractor = FeaturesExtractor(self.model_family, self.backbone_type, self.features_size)
        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.features_size, out_features=self.features_size // 16),
            nn.LeakyReLU(0.01),
            AdjustableDropout(initial_p=0.1),
            nn.Linear(in_features=self.features_size // 16, out_features=1),
        )

    def forward(self, photo: Tensor) -> Tensor:
        """
        Args:
            photo (Tensor): Image tensor of shape [Batch, Channels, Height, Width].
        Returns:
            grade (Tensor): Predicted grade of shape [Batch, 1].
        """
        features = self.extractor(photo)
        grade = self.regressor(features)
        return grade