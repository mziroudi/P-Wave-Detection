"""1D CNN for classification and P-arrival regression on seismic windows."""

from __future__ import annotations

import torch
import torch.nn as nn


def _feature_stack() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(3, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
    )


class SeismicCNN1D(nn.Module):
    """
    Compact 1D CNN treating a 3-channel seismogram like a multi-channel audio clip.

    Input:  (batch, 3, 1000)  — E/N/Z over 10 seconds @ 100 Hz
    Output: (batch, 2)        — logits for [noise, earthquake]
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = _feature_stack()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SeismicCNN1DRegressor(nn.Module):
    """
    Predict P-wave sample index within a 10 s window.

    Output is in sample units (0 .. window_samples-1). Convert to milliseconds
    with: t_ms = pred_sample * (1000 / sample_rate).
    """

    def __init__(self, window_samples: int = 1000, dropout: float = 0.3) -> None:
        super().__init__()
        self.window_samples = window_samples
        self.features = _feature_stack()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frac = torch.sigmoid(self.head(self.features(x)).squeeze(-1))
        return frac * float(self.window_samples - 1)

    def load_pretrained_features(self, classifier_state: dict) -> None:
        """Initialize feature extractor from a trained SeismicCNN1D checkpoint."""
        feat_state = {
            k[len("features.") :]: v
            for k, v in classifier_state.items()
            if k.startswith("features.")
        }
        self.features.load_state_dict(feat_state)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_classifier_checkpoint(path, device: torch.device) -> SeismicCNN1D:
    model = SeismicCNN1D().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
