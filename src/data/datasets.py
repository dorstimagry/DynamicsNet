"""Dataset utilities for training transformer-based controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SequenceWindowConfig:
    """Configuration for extracting rolling windows from trip segments."""

    history: int = 50
    horizon: int = 50
    stride: int = 5
    max_throttle: float = 100.0
    max_brake: float = 100.0
    allow_overlap_actuation: bool = False


STATIONARY_SPEED_EPS: float = 0.15  # m/s
STATIONARY_ACCEL_EPS: float = 0.1  # m/s^2


class EVSequenceDataset(Dataset):
    """PyTorch dataset that serves history/horizon windows from saved trips."""

    def __init__(
        self,
        data_path: Path,
        window: SequenceWindowConfig | None = None,
        state_features: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.window = window or SequenceWindowConfig()
        self.state_features = list(state_features) if state_features else ["speed", "grade"]

        raw = torch.load(data_path)
        self._metadata = raw.get("metadata", {})
        self._segments: List[dict[str, np.ndarray]] = []
        for key, value in raw.items():
            if key == "metadata":
                continue
            self._segments.append({name: np.asarray(arr) for name, arr in value.items()})

        self._sample_time = self._infer_sample_time(self._metadata)
        self._accelerations: List[np.ndarray] = []
        for seg in self._segments:
            speed = seg["speed"]
            if speed.size < 3:
                accel = np.gradient(speed, self._sample_time, edge_order=1)
            else:
                accel = np.gradient(speed, self._sample_time, edge_order=2)
            self._accelerations.append(accel)

        self._index: List[tuple[int, int]]
        self._stationary_flags: np.ndarray
        self._index, self._stationary_flags = self._build_index()

        self.stationary_fraction: float = float(self._stationary_flags.mean()) if len(self._stationary_flags) else 0.0
        percentage = max(self.stationary_fraction * 100.0, 1e-6)
        self.stationary_weight: float = 1.0 / percentage

    # ------------------------------------------------------------------
    def _infer_sample_time(self, metadata: dict) -> float:
        for key in ("sample_time", "dt", "delta_t", "time_step"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
        return 0.1

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seg_idx, anchor = self._index[idx]
        seg = self._segments[seg_idx]
        w = self.window

        history_slice = slice(anchor - w.history, anchor)
        future_slice = slice(anchor, anchor + w.horizon)

        throttle = seg["throttle"]
        brake = seg["brake"]
        angle = seg.get("angle")

        def _states(slice_) -> np.ndarray:
            feats: List[np.ndarray] = []
            for name in self.state_features:
                if name == "speed":
                    feats.append(seg["speed"][slice_])
                elif name == "grade":
                    if angle is None:
                        feats.append(np.zeros_like(seg["speed"][slice_]))
                    else:
                        feats.append(np.sin(angle[slice_]))
                elif name == "angle":
                    feats.append(angle[slice_] if angle is not None else np.zeros_like(seg["speed"][slice_]))
                else:
                    raise KeyError(f"Unsupported state feature: {name}")
            return np.stack(feats, axis=-1)

        throttle_hist = throttle[history_slice]
        brake_hist = brake[history_slice]
        throttle_future = throttle[future_slice]
        brake_future = brake[future_slice]

        history_states = torch.from_numpy(_states(history_slice)).float()
        history_actions = torch.from_numpy(
            np.stack(
                [
                    throttle_hist / self.window.max_throttle,
                    brake_hist / self.window.max_brake,
                ],
                axis=-1,
            )
        ).float()

        future_states = torch.from_numpy(_states(future_slice)).float()
        future_actions = torch.from_numpy(
            np.stack(
                [
                    throttle_future / self.window.max_throttle,
                    brake_future / self.window.max_brake,
                ],
                axis=-1,
            )
        ).float()

        # Composite actuation signal: positive -> throttle, negative -> brake
        actuation = self._compute_actuation(throttle_future, brake_future)

        loss_weight = self.stationary_weight if self._stationary_flags[idx] else 1.0

        return {
            "history_states": history_states,
            "history_actions": history_actions,
            "future_states": future_states,
            "future_actions": future_actions,
            "future_actuation": torch.from_numpy(actuation).float(),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "is_stationary": torch.tensor(bool(self._stationary_flags[idx])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> Tuple[List[tuple[int, int]], np.ndarray]:
        index: List[tuple[int, int]] = []
        stationary: List[bool] = []
        w = self.window
        for seg_idx, seg in enumerate(self._segments):
            length = len(seg["speed"])
            min_length = w.history + w.horizon
            if length < min_length:
                continue
            for anchor in range(w.history, length - w.horizon + 1, w.stride):
                if self._window_valid(seg, anchor):
                    index.append((seg_idx, anchor))
                    stationary.append(self._window_stationary(seg_idx, anchor))
        return index, np.asarray(stationary, dtype=bool)

    def _window_valid(self, seg: dict[str, np.ndarray], anchor: int) -> bool:
        w = self.window
        history_slice = slice(anchor - w.history, anchor)
        future_slice = slice(anchor, anchor + w.horizon)

        throttle_hist = seg["throttle"][history_slice]
        brake_hist = seg["brake"][history_slice]
        throttle_future = seg["throttle"][future_slice]
        brake_future = seg["brake"][future_slice]

        arrays: Iterable[np.ndarray] = (
            throttle_hist,
            brake_hist,
            throttle_future,
            brake_future,
            seg["speed"][history_slice],
            seg["speed"][future_slice],
        )

        for arr in arrays:
            if not np.all(np.isfinite(arr)):
                return False

        if np.any(throttle_hist < 0) or np.any(throttle_future < 0):
            return False
        if np.any(brake_hist < 0) or np.any(brake_future < 0):
            return False
        if np.any(throttle_hist > w.max_throttle) or np.any(throttle_future > w.max_throttle):
            return False
        if np.any(brake_hist > w.max_brake) or np.any(brake_future > w.max_brake):
            return False

        if not w.allow_overlap_actuation:
            if np.any((throttle_future > 1e-3) & (brake_future > 1e-3)):
                return False
        return True

    def _window_stationary(self, seg_idx: int, anchor: int) -> bool:
        w = self.window
        future_slice = slice(anchor, anchor + w.horizon)
        speed_future = self._segments[seg_idx]["speed"][future_slice]
        accel_future = self._accelerations[seg_idx][future_slice]
        return bool(
            np.all(np.abs(speed_future) <= STATIONARY_SPEED_EPS)
            and np.all(np.abs(accel_future) <= STATIONARY_ACCEL_EPS)
        )

    def _compute_actuation(self, throttle: np.ndarray, brake: np.ndarray) -> np.ndarray:
        throttle_component = throttle / self.window.max_throttle
        brake_component = brake / self.window.max_brake
        actuation = np.where(throttle_component > 0, throttle_component, 0.0)
        actuation -= np.where(brake_component > 0, brake_component, 0.0)
        return actuation.astype(np.float32)


__all__ = [
    "EVSequenceDataset",
    "SequenceWindowConfig",
    "STATIONARY_SPEED_EPS",
    "STATIONARY_ACCEL_EPS",
]


