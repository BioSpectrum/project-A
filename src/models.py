# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.models as tv_models
import timm
import logging
from typing import Dict, Any, Tuple

from config import MODEL_NAME, PRETRAINED, NUM_CHANNELS

logger = logging.getLogger(__name__)

class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer for Domain Adversarial Neural Networks.
    During forward pass: acts as identity
    During backward pass: reverses gradient and scales by alpha
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def gradient_reversal(x, alpha=1.0):
    """Functional interface for gradient reversal."""
    return GradientReversalLayer.apply(x, alpha)

class FeatureExtractor(nn.Module):
    """
    Feature extractor for 6-channel RxRx1 fluorescence images.
    Learns batch-invariant representations when used with adversarial training.
    """

    def __init__(self, backbone_name: str = MODEL_NAME, pretrained: bool = PRETRAINED):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            self.backbone = tv_models.resnet50(pretrained=pretrained)

            # Modify first conv layer for 6-channel input (fluorescence channels)
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize new conv layer weights intelligently
            if pretrained:
                with torch.no_grad():
                    # Average RGB weights and replicate for 6 channels
                    rgb_weights = original_conv1.weight  # [64, 3, 7, 7]
                    new_weights = rgb_weights.repeat(1, 2, 1, 1)  # [64, 6, 7, 7]
                    self.backbone.conv1.weight = nn.Parameter(new_weights)

            # Remove classification head
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "resnet18":
            self.backbone = tv_models.resnet18(pretrained=pretrained)

            # Modify for 6-channel input
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            if pretrained:
                with torch.no_grad():
                    rgb_weights = original_conv1.weight
                    new_weights = rgb_weights.repeat(1, 2, 1, 1)
                    self.backbone.conv1.weight = nn.Parameter(new_weights)

            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            # Try loading from timm for other architectures
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    in_chans=NUM_CHANNELS,  # timm supports in_chans parameter
                    num_classes=0  # Remove head, return features
                )
                self.feature_dim = self.backbone.num_features
                logger.info(f"Loaded {backbone_name} from timm with {NUM_CHANNELS} input channels")
            except Exception as e:
                logger.error(f"Failed to load {backbone_name}: {e}")
                raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        """Extract features from 6-channel input."""
        return self.backbone(x)

class PerturbationClassifier(nn.Module):
    """
    Classifier for genetic perturbation prediction (biological signal).
    This represents the "main task" we want to preserve.
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)

class BatchDiscriminator(nn.Module):
    """
    Domain discriminator that predicts experimental plate (batch).
    Acts as adversary to create plate-invariant features.
    """

    def __init__(self, feature_dim: int, num_plates: int, dropout: float = 0.5):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_plates)
        )

    def forward(self, features, alpha=1.0):
        """Apply gradient reversal before classification."""
        reversed_features = gradient_reversal(features, alpha)
        return self.discriminator(reversed_features)

class DANNModel(nn.Module):
    """
    Domain Adversarial Neural Network for batch effect correction in RxRx1.

    Architecture components:
    1. Feature Extractor: Maps 6-channel images to batch-invariant features
    2. Perturbation Classifier: Predicts genetic perturbation (siRNA) classes
    3. Batch Discriminator: Predicts experimental plate with gradient reversal

    The adversarial training forces the feature extractor to learn representations
    that are useful for perturbation classification but uninformative about batch.
    """

    def __init__(self,
                 num_perturbations: int,
                 num_plates: int,
                 backbone_name: str = MODEL_NAME,
                 pretrained: bool = PRETRAINED):
        super().__init__()

        self.feature_extractor = FeatureExtractor(backbone_name, pretrained)
        feature_dim = self.feature_extractor.feature_dim

        self.perturbation_classifier = PerturbationClassifier(feature_dim, num_perturbations)
        self.batch_discriminator = BatchDiscriminator(feature_dim, num_plates)

        self.num_perturbations = num_perturbations
        self.num_plates = num_plates

        logger.info(f"DANN Model initialized:")
        logger.info(f"  Backbone: {backbone_name}")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Genetic perturbations: {num_perturbations}")
        logger.info(f"  Experimental plates: {num_plates}")

    def forward(self, x, alpha=1.0):
        """
        Forward pass through DANN.

        Args:
            x: 6-channel fluorescence images [batch_size, 6, H, W]
            alpha: Gradient reversal strength

        Returns:
            Dict with perturbation predictions, batch predictions, and features
        """
        # Extract batch-invariant features
        features = self.feature_extractor(x)

        # Predict genetic perturbations (main biological task)
        perturbation_predictions = self.perturbation_classifier(features)

        # Predict experimental plate (adversarial task)
        batch_predictions = self.batch_discriminator(features, alpha)

        return {
            'features': features,
            'perturbation_predictions': perturbation_predictions,
            'batch_predictions': batch_predictions
        }

# Update baseline model to handle 6 channels
def get_baseline_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create baseline model for 6-channel RxRx1 images without adversarial training.
    Updated to handle fluorescence microscopy data.
    """
    logger.info(f"Loading baseline model: {model_name} for {num_classes} classes (6-channel input)")

    # Use the FeatureExtractor + simple classifier
    feature_extractor = FeatureExtractor(model_name, pretrained)
    classifier = PerturbationClassifier(feature_extractor.feature_dim, num_classes)

    class BaselineModel(nn.Module):
        def __init__(self, feature_extractor, classifier):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classifier = classifier

        def forward(self, x):
            features = self.feature_extractor(x)
            predictions = self.classifier(features)
            return predictions

    return BaselineModel(feature_extractor, classifier)

def get_model(model_type: str, num_classes: int, num_domains: int = None, **kwargs):
    """
    Factory function for creating models.

    Args:
        model_type: 'dann' for adversarial or 'baseline' for standard
        num_classes: Number of genetic perturbation classes
        num_domains: Number of experimental plates (for DANN)
    """
    if model_type.lower() == 'dann':
        if num_domains is None:
            raise ValueError("num_domains (plates) required for DANN model")
        return DANNModel(num_classes, num_domains, **kwargs)

    elif model_type.lower() == 'baseline':
        return get_baseline_model(MODEL_NAME, num_classes, **kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
