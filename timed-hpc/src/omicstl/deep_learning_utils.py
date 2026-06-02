"""Utilities for deep learning models."""

from enum import Enum, auto

import torch
import torch.nn.functional as tf
from torch import nn


class PredictionMode(Enum):
    """An enum for specifying a models prediction mode."""

    CLASSIFICATION = auto()
    REGRESSION = auto()


def add_last_layer(out: torch.Tensor, prediction_mode: PredictionMode) -> torch.Tensor:
    """Pass a torch tensor through the last layer to an internal deep learning model based on its prediction mode.

    Args:
        out: A torch tensor containing entries for the second-to-last layer of a neural network.
        prediction_mode: Prediction mode of the model

    Returns:
        torch.Tensor: The input layer passed through the final layer of the network

    """
    match prediction_mode:
        case PredictionMode.CLASSIFICATION:
            pass  # return raw logits; loss fns apply log_softmax internally, prediction uses softmax explicitly
        case PredictionMode.REGRESSION:
            pass
    return out


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute MSE of torch model.
    
    Args:
        predictions: a tensor containing regression prediction values
        targets: a tensor containing expected predicted values
        
    Returns:
        the mean squared error between the predictions and the targets
    """
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        raise ValueError("predictions contains NaN or infinity, which is not allowed")

    return ((predictions.detach().cpu().numpy() - targets.detach().cpu().numpy()) ** 2).mean()


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy of torch model.
    
    Args:
        predictions: a two-dimensional tensor containing class prediction probabilities, e.g. torch.tensor([[0.1, 0.7, 0.2], ...])
        targets: a one-dimensional tensor containing the target predicted class, e.g. torch.tensor([1, ...])

    Returns:
        the accuracy metric of the predictions when compared to the targets
    """
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        raise ValueError("predictions contains NaN or infinity, which is not allowed")

    return (predictions.argmax(dim=1).numpy() == targets.numpy()).mean()


def freeze_layers(model: nn.Module, layers: list[str] | None = None) -> None:
    """Freeze all parameters in the model."""
    for name, param in model.named_parameters():
        if layers is not None and not any(layer_name in name for layer_name in layers):
            continue
        param.requires_grad = False


def freeze_until_layer(model: nn.Module, target_layer: str | None = None) -> None:
    """Freeze all layers until a specific layer."""
    reached_target = False
    for name, param in model.named_parameters():
        if target_layer and target_layer in name:
            reached_target = True
        if not reached_target:
            param.requires_grad = False


def mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_bw: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy between source and target latent distributions.

    Uses an RBF kernel to measure the distance between the distributions of
    source and target representations. Used for explicit latent space alignment
    in heterogeneous transfer learning where the KL regularisation alone
    (VAE) or batch-normalisation alone (MLP) may not be sufficient.

    Args:
        source: Tensor of shape (n_source, d) — source latent representations.
        target: Tensor of shape (n_target, d) — target latent representations.
        kernel_bw: RBF kernel bandwidth. Defaults to 1.0.

    Returns:
        Scalar MMD loss (non-negative).

    """
    def rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(x, y) ** 2
        return torch.exp(-dist_sq / (2.0 * kernel_bw ** 2))

    n_s = source.shape[0]
    n_t = target.shape[0]
    k_ss = rbf(source, source)
    k_tt = rbf(target, target)
    k_st = rbf(source, target)
    return (
        k_ss.sum() / (n_s * n_s)
        + k_tt.sum() / (n_t * n_t)
        - 2.0 * k_st.sum() / (n_s * n_t)
    ).clamp(min=0.0)
