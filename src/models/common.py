"""Shared neural building blocks for EV sequence models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from *Attention Is All You Need*.

    The encoding is added to token embeddings and does not require learning any
    additional parameters.  Inputs are expected in ``(batch, seq, d_model)``
    format.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds positional encoding table")
        return x + self.pe[:, : x.size(1)]


class SequenceProjector(nn.Module):
    """Linear projection with layer norm + dropout for per-step features."""

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBackbone(nn.Module):
    """Wrapper around ``nn.TransformerEncoder`` using batch-first input."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )

    def forward(
        self,
        tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the encoder.

        Args:
            tokens: ``(batch, seq, d_model)`` tensor of token embeddings.
            key_padding_mask: optional boolean mask with shape ``(batch, seq)``
                where ``True`` entries indicate positions that should be ignored.
            attn_mask: standard attention mask broadcastable to ``(seq, seq)``.
        """

        return self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )


@dataclass(slots=True)
class TransformerConfig:
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1024


def build_backbone(cfg: TransformerConfig) -> tuple[nn.Module, nn.Module]:
    """Create positional encoding and transformer backbone modules."""

    pos_encoding = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
    backbone = TransformerBackbone(
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )
    return pos_encoding, backbone


def merge_padding_masks(*masks: Optional[Tensor]) -> Optional[Tensor]:
    """Combine multiple key padding masks respecting ``None`` entries."""

    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    base = masks[0].clone()
    for mask in masks[1:]:
        base = torch.cat([base, mask], dim=1)
    return base


def segment_embedding(segment_ids: Tensor, num_segments: int, d_model: int) -> Tensor:
    """Return learnable embeddings for segment identifiers."""

    embedding = nn.Embedding(num_segments, d_model).to(segment_ids.device)
    # The embedding should be created once; modules should cache the layer.
    raise RuntimeError("segment_embedding should be wrapped by a module")


class SegmentEmbedding(nn.Module):
    """Learnable embedding lookup for token type/segment ids."""

    def __init__(self, num_segments: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_segments, d_model)

    def forward(self, segment_ids: Tensor) -> Tensor:
        return self.embedding(segment_ids)


def apply_masks(
    history_mask: Optional[Tensor],
    future_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """Concatenate optional masks for history/future sequences."""

    masks = []
    if history_mask is not None:
        masks.append(history_mask)
    if future_mask is not None:
        masks.append(future_mask)
    if not masks:
        return None
    return torch.cat(masks, dim=1)


__all__ = [
    "SinusoidalPositionalEncoding",
    "SequenceProjector",
    "TransformerBackbone",
    "TransformerConfig",
    "build_backbone",
    "apply_masks",
    "SegmentEmbedding",
]


