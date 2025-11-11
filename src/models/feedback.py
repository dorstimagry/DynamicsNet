"""Feedback residual policy to correct feedforward actuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .common import (
    SegmentEmbedding,
    SequenceProjector,
    TransformerConfig,
    apply_masks,
    build_backbone,
)


@dataclass(slots=True)
class FeedbackModelConfig:
    state_dim: int
    action_dim: int
    transformer: TransformerConfig = TransformerConfig()
    proj_dropout: float = 0.1
    residual_scale: float = 1.0


class FeedbackResidualModel(nn.Module):
    """Predict additive actuation to compensate residual tracking error."""

    def __init__(self, cfg: FeedbackModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        history_input_dim = cfg.state_dim + cfg.action_dim * 2
        future_input_dim = cfg.state_dim + cfg.action_dim

        self.history_proj = SequenceProjector(
            history_input_dim,
            cfg.transformer.d_model,
            dropout=cfg.proj_dropout,
        )
        self.future_proj = SequenceProjector(
            future_input_dim,
            cfg.transformer.d_model,
            dropout=cfg.proj_dropout,
        )
        self.segment_embed = SegmentEmbedding(num_segments=2, d_model=cfg.transformer.d_model)
        self.positional_encoding, self.backbone = build_backbone(cfg.transformer)

        self.output_head = nn.Sequential(
            nn.Linear(cfg.transformer.d_model, cfg.transformer.d_model),
            nn.ReLU(),
            nn.Linear(cfg.transformer.d_model, cfg.action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        desired_future_states: Tensor,
        feedforward_future_actions: Tensor,
        history_residual_actions: Optional[Tensor] = None,
        history_mask: Optional[Tensor] = None,
        future_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return residual actuation horizon to add to feedforward commands."""

        batch, history_len, _ = history_states.shape
        horizon = desired_future_states.size(1)

        if history_residual_actions is None:
            history_residual_actions = torch.zeros_like(history_actions)

        history_features = torch.cat(
            [history_states, history_actions, history_residual_actions], dim=-1
        )
        history_tokens = self.history_proj(history_features)
        history_tokens = history_tokens + self.segment_embed(
            torch.zeros(batch, history_len, dtype=torch.long, device=history_tokens.device)
        )

        future_features = torch.cat(
            [desired_future_states, feedforward_future_actions], dim=-1
        )
        future_tokens = self.future_proj(future_features)
        future_tokens = future_tokens + self.segment_embed(
            torch.ones(batch, horizon, dtype=torch.long, device=future_tokens.device)
        )

        tokens = torch.cat([history_tokens, future_tokens], dim=1)
        tokens = self.positional_encoding(tokens)
        key_padding_mask = apply_masks(history_mask, future_mask)

        encoded = self.backbone(tokens, key_padding_mask=key_padding_mask)
        future_encoded = encoded[:, history_len:, :]
        residual = self.output_head(future_encoded) * self.cfg.residual_scale
        return residual


__all__ = ["FeedbackResidualModel", "FeedbackModelConfig"]


