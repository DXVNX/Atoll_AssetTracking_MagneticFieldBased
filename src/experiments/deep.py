"""Small PyTorch models and fold-local training helpers."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GRUClassifier(nn.Module):
    def __init__(self, input_channels: int, n_classes: int, hidden_size: int = 64):
        super().__init__()
        self.recurrent = nn.GRU(input_channels, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(values.transpose(1, 2))
        return self.classifier(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(self, input_channels: int, n_classes: int, hidden_size: int = 64):
        super().__init__()
        self.recurrent = nn.LSTM(input_channels, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(values.transpose(1, 2))
        return self.classifier(hidden[-1])


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int
    best_macro_f1: float


def _evaluate(model: nn.Module, values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(values).float()).argmax(dim=1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
    }


def train_classifier(
    model: nn.Module,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    learning_rate: float = 1e-3,
    max_epochs: int = 60,
    patience: int = 8,
    class_weights: np.ndarray | None = None,
    batch_size: int = 128,
) -> TrainingResult:
    set_torch_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_values).float(), torch.from_numpy(train_labels).long()), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight_tensor = torch.from_numpy(class_weights).float() if class_weights is not None else None
    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_values, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_values), batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _evaluate(model, validation_values, validation_labels)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        if metrics["macro_f1"] > best_score + 1e-8:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(model, history, best_epoch, best_score)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))