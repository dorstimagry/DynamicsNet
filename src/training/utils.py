"""Utility functions for training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}


@dataclass
class AverageMeter:
    value: float = 0.0
    count: int = 0

    def update(self, new_value: float, n: int = 1) -> None:
        self.value += new_value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.value / max(self.count, 1)


def to_tensor(x, device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


__all__ = ["move_batch_to_device", "AverageMeter", "to_tensor"]


