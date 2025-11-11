"""Transformer-based forward dynamics model."""

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
class ForwardModelConfig:
    state_dim: int
    action_dim: int
    transformer: TransformerConfig = TransformerConfig()
    proj_dropout: float = 0.1


class ForwardDynamicsModel(nn.Module):
    """Predict future states given history and a planned actuation horizon."""

    def __init__(self, cfg: ForwardModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.history_proj = SequenceProjector(
            cfg.state_dim + cfg.action_dim,
            cfg.transformer.d_model,
            dropout=cfg.proj_dropout,
        )
        self.future_proj = SequenceProjector(
            cfg.action_dim,
            cfg.transformer.d_model,
            dropout=cfg.proj_dropout,
        )
        self.segment_embed = SegmentEmbedding(num_segments=2, d_model=cfg.transformer.d_model)
        self.positional_encoding, self.backbone = build_backbone(cfg.transformer)

        self.output_head = nn.Sequential(
            nn.Linear(cfg.transformer.d_model, cfg.transformer.d_model),
            nn.ReLU(),
            nn.Linear(cfg.transformer.d_model, cfg.state_dim),
        )

    def forward(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        future_actions: Tensor,
        history_mask: Optional[Tensor] = None,
        future_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            history_states: ``(batch, history_len, state_dim)`` tensor.
            history_actions: ``(batch, history_len, action_dim)`` tensor.
            future_actions: ``(batch, horizon, action_dim)`` tensor describing
                the actuation sequence to be simulated.
            history_mask: optional boolean mask with shape ``(batch, history_len)``
                where ``True`` entries are padding to ignore.
            future_mask: optional boolean mask with shape ``(batch, horizon)``.

        Returns:
            ``(batch, horizon, state_dim)`` tensor of predicted future states.
        """

        batch, history_len, _ = history_states.shape
        horizon = future_actions.size(1)

        history_tokens = torch.cat([history_states, history_actions], dim=-1)
        history_tokens = self.history_proj(history_tokens)
        history_tokens = history_tokens + self.segment_embed(
            torch.zeros(batch, history_len, dtype=torch.long, device=history_tokens.device)
        )

        future_tokens = self.future_proj(future_actions)
        future_tokens = future_tokens + self.segment_embed(
            torch.ones(batch, horizon, dtype=torch.long, device=future_tokens.device)
        )

        tokens = torch.cat([history_tokens, future_tokens], dim=1)
        tokens = self.positional_encoding(tokens)

        key_padding_mask = apply_masks(history_mask, future_mask)

        encoded = self.backbone(tokens, key_padding_mask=key_padding_mask)
        future_encoded = encoded[:, history_len:, :]
        predictions = self.output_head(future_encoded)
        return predictions


__all__ = ["ForwardModelConfig", "ForwardDynamicsModel"]


