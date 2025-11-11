"""Loss helpers for EV controller training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


LossType = Literal["l2", "smooth_l1"]


@dataclass(slots=True)
class RegressionLossConfig:
    type: LossType = "l2"
    beta: float = 1.0  # smooth L1 transition point


def _per_sample_reduction(loss: Tensor) -> Tensor:
    if loss.ndim == 1:
        return loss
    return loss.view(loss.shape[0], -1).mean(dim=1)


def regression_loss(
    pred: Tensor,
    target: Tensor,
    cfg: RegressionLossConfig,
    weight: Tensor | None = None,
) -> Tensor:
    if cfg.type == "l2":
        raw = nn.functional.mse_loss(pred, target, reduction="none")
    elif cfg.type == "smooth_l1":
        raw = nn.functional.smooth_l1_loss(pred, target, beta=cfg.beta, reduction="none")
    else:
        raise ValueError(f"Unsupported loss type: {cfg.type}")

    per_sample = _per_sample_reduction(raw)
    if weight is not None:
        w = weight.view(-1).to(per_sample.dtype)
        weighted = per_sample * w
        denom = torch.clamp(w.sum(), min=1e-12)
        return weighted.sum() / denom
    return per_sample.mean()


def cyclic_consistency_loss(
    forward_model,
    inverse_model,
    history_states: Tensor,
    history_actions: Tensor,
    desired_future_states: Tensor,
    cfg: RegressionLossConfig,
    predicted_actions: Tensor | None = None,
    weight: Tensor | None = None,
) -> Tensor:
    """Enforce approximate inverse relationship between models."""

    if predicted_actions is None:
        predicted_actions = inverse_model(
            history_states,
            history_actions,
            desired_future_states,
        )
    reconstructed_states = forward_model(
        history_states,
        history_actions,
        predicted_actions,
    )
    return regression_loss(reconstructed_states, desired_future_states, cfg, weight=weight)


__all__ = ["RegressionLossConfig", "regression_loss", "cyclic_consistency_loss"]


