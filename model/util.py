import math
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from .transformers import *


def getBackbone(model_family: str, backbone_type: str) -> nn.Module:
    """
    Instantiates the backbone for the feature extractor.
    Supported families: 'EfficientNet' (e.g., 'b0', 'b1') and 'ResNet' (e.g., '18', '50').
    Removes the final classifier layer to output raw features.
    """
    model_family = model_family.lower()
    if model_family.lower() == 'efficientnet':
        model_name = f'efficientnet_{backbone_type.lower()}'
        model_weight = getattr(models, f'EfficientNet_{backbone_type.upper()}_Weights')
        backbone = getattr(models, model_name)(weights=model_weight.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()  # Remove the classification head
    else:  # ResNet
        model_name = f'resnet{backbone_type}'
        model_weight = getattr(models, f'ResNet{backbone_type.lower()}_Weights')
        backbone = getattr(models, model_name)(weights=model_weight.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # Remove the classification head
    return backbone


class AdjustableDropout(nn.Module):
    """
    A custom Dropout layer that allows dynamic adjustment of the dropout probability.
    """

    def __init__(self, initial_p: float = 0.1) -> None:
        super().__init__()
        self.p = initial_p
        self.dropout = nn.Dropout(p=self.p)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

    def set_dropout(self, new_p: float) -> None:
        self.p = new_p
        self.dropout.p = new_p


class FeaturesExtractor(nn.Module):
    """
    Extracts features from an input image using a specified CNN backbone,
    followed by a custom projection head to yield a fixed-size feature vector.
    """

    def __init__(
            self,
            model_family: str,
            backbone_type: str,
            out_features: int = 256
    ) -> None:
        super(FeaturesExtractor, self).__init__()
        self.model_family = model_family
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.backbone = getBackbone(model_family=model_family, backbone_type=self.backbone_type)

        # Determine the output feature dimension of the backbone dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            self.in_features = self.backbone(dummy_input).shape[1]

        self.adaptive_dropout = AdjustableDropout(initial_p=0.1)
        self.extractor = nn.Sequential(
            self.backbone,
            nn.Linear(in_features=self.in_features, out_features=self.in_features // 2),
            nn.BatchNorm1d(self.in_features // 2),
            nn.LeakyReLU(0.05),
            nn.Linear(in_features=self.in_features // 2, out_features=self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.LeakyReLU(0.05),
            self.adaptive_dropout,
        )

    def forward(self, photo: Tensor) -> Tensor:
        features = self.extractor(photo)
        return features


class CrossAttentionFusion(nn.Module):
    """
    Fuses multiple feature vectors using a cross-attention mechanism.
    Learns a query bias to aggregate information across different views.
    """

    def __init__(self, feature_size: int = 256) -> None:
        super().__init__()
        self.query_bias = nn.Parameter(torch.randn(1, 1, feature_size) * 0.1)
        self.key_proj = nn.Linear(feature_size, feature_size, bias=False)
        self.value_proj = nn.Linear(feature_size, feature_size, bias=False)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        queries = features.mean(dim=1, keepdim=True) + self.query_bias
        keys = self.key_proj(features)
        values = self.value_proj(features)
        attn_scores = torch.matmul(queries, keys.transpose(1, 2))
        attn_scores = attn_scores / (features.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute the weighted sum of values (fused representation)
        fused = torch.matmul(attn_weights, values)
        return fused.squeeze(1), attn_weights.squeeze(1)


class TimeEncoder(torch.nn.Module):
    """
    Encodes temporal positional information into the feature vectors
    to retain sequence order for longitudinal data.
    """

    def __init__(self, features_size: int = 256) -> None:
        super().__init__()
        self.features_size = features_size
        self.encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=int(math.sqrt(self.features_size))),
            nn.Tanh(),
            nn.Linear(in_features=int(math.sqrt(self.features_size)), out_features=self.features_size)
        )

    def forward(self, features, times):
        times = times.unsqueeze(-1).float().to(features.device)
        time_embeddings = self.encoder(times)
        return features + time_embeddings


class TimeSeriesTransformer(nn.Module):
    """
    Processes time-series feature sequences using a Transformer Encoder
    to capture longitudinal dependencies.
    """

    def __init__(
            self,
            features_size: int,
            num_heads=2,
            hidden_size=512,
            batch_first=True,
            num_layers=2
    ) -> None:
        super().__init__()
        self.features_size = features_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.time_encoder = TimeEncoder(features_size)
        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.features_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            batch_first=self.batch_first
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(
            self,
            features,
            src_key_padding_mask=None,
            is_causal=False
    ) -> tuple:
        shape = features.shape
        # Inject time embeddings
        features = self.time_encoder(features, torch.arange(1, shape[1] + 1).unsqueeze(0).repeat(shape[0], 1))

        encoded_output, attn_maps = self.transformer_encoder(
            features,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )

        # Extract the representation from the last valid time step based on the padding mask
        if src_key_padding_mask is not None:
            batch_size, seq_len = src_key_padding_mask.shape
            indices = torch.arange(seq_len, device=src_key_padding_mask.device).unsqueeze(0).expand(batch_size, -1)
            valid_indices = torch.where(~src_key_padding_mask, indices, torch.tensor(-1, device=indices.device))
            last_valid_indices = valid_indices.max(dim=1).values  # [batch_size]
            last_outputs = encoded_output[torch.arange(batch_size), last_valid_indices, :]
            attn_maps = attn_maps[torch.arange(batch_size), last_valid_indices, :]
        else:
            last_outputs, attn_maps = encoded_output[:, -1], attn_maps[:, -1]

        return last_outputs, attn_maps