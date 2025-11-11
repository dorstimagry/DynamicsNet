#!/usr/bin/env python3
"""Interactive simulation of EV longitudinal controllers."""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.data.datasets import SequenceWindowConfig
from src.models import FeedbackResidualModel, ForwardDynamicsModel, InverseActuationModel
from src.models.common import TransformerConfig
from src.models.feedback import FeedbackModelConfig
from src.models.forward import ForwardModelConfig
from src.models.inverse import InverseModelConfig


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def load_models(stage1: Path, stage2: Path, state_dim: int, action_dim: int, device: torch.device):
    cfg = TransformerConfig()
    forward = ForwardDynamicsModel(ForwardModelConfig(state_dim, action_dim, cfg))
    inverse = InverseActuationModel(InverseModelConfig(state_dim, action_dim, cfg))
    feedback = FeedbackResidualModel(FeedbackModelConfig(state_dim, action_dim, cfg))

    stage1_state = torch.load(stage1, map_location="cpu")
    forward.load_state_dict(stage1_state["forward_model"])
    inverse.load_state_dict(stage1_state["inverse_model"])

    stage2_state = torch.load(stage2, map_location="cpu")
    feedback.load_state_dict(stage2_state["feedback_model"])

    forward.to(device).eval()
    inverse.to(device).eval()
    feedback.to(device).eval()
    return forward, inverse, feedback


# ---------------------------------------------------------------------------
# Feature construction & perturbations
# ---------------------------------------------------------------------------

def build_state_matrix(speed: np.ndarray, grade: np.ndarray, features: List[str]) -> np.ndarray:
    feats = []
    for name in features:
        if name == "speed":
            feats.append(speed)
        elif name == "grade":
            feats.append(np.sin(grade))
        elif name == "angle":
            feats.append(grade)
        else:
            raise KeyError(f"Unsupported state feature: {name}")
    return np.stack(feats, axis=-1).astype(np.float32)


def clamp_actions(throttle: Tensor, brake: Tensor) -> Tensor:
    throttle = torch.clamp(throttle, 0.0, 1.0)
    brake = torch.clamp(brake, 0.0, 1.0)
    both = (throttle > 0) & (brake > 0)
    throttle = torch.where(both & (throttle >= brake), throttle, torch.where(both, torch.zeros_like(throttle), throttle))
    brake = torch.where(both & (brake > throttle), brake, torch.where(both, torch.zeros_like(brake), brake))
    return torch.stack([throttle, brake], dim=-1)


def clamp_actions_numpy(actions: np.ndarray) -> np.ndarray:
    throttle = np.clip(actions[..., 0], 0.0, 1.0)
    brake = np.clip(actions[..., 1], 0.0, 1.0)
    both = (throttle > 0.0) & (brake > 0.0)
    throttle = np.where(both & (throttle >= brake), throttle, np.where(both, 0.0, throttle))
    brake = np.where(both & (brake > throttle), brake, np.where(both, 0.0, brake))
    return np.stack([throttle, brake], axis=-1).astype(np.float32)


def load_history_seed_from_dataset(
    dataset_path: Path,
    history: int,
    segment_idx: Optional[int] = None,
    anchor: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = torch.load(dataset_path)
    segments = [value for key, value in raw.items() if key != "metadata"]
    if not segments:
        raise RuntimeError(f"No segments found in dataset {dataset_path}")

    if segment_idx is None:
        lengths = [len(seg["speed"]) for seg in segments]
        segment_idx = int(np.argmax(lengths))

    if segment_idx < 0 or segment_idx >= len(segments):
        raise IndexError(f"Segment index {segment_idx} out of range (0..{len(segments)-1})")

    segment = segments[segment_idx]
    length = len(segment["speed"])
    if anchor is None:
        anchor = history

    if anchor < history or anchor > length:
        raise ValueError(
            f"Anchor {anchor} invalid for segment length {length} with history {history}"
        )

    sl = slice(anchor - history, anchor)
    history_speed = np.asarray(segment["speed"][sl], dtype=np.float32)
    throttle_hist = np.asarray(segment["throttle"][sl], dtype=np.float32)
    brake_hist = np.asarray(segment["brake"][sl], dtype=np.float32)
    angle = segment.get("angle")
    if angle is not None:
        history_grade = np.asarray(angle[sl], dtype=np.float32)
    else:
        history_grade = np.zeros_like(history_speed)

    history_actions = np.stack([throttle_hist, brake_hist], axis=-1)
    return history_speed, history_actions, history_grade


@dataclass
class PerturbationConfig:
    speed_noise_std: float = 0.0
    speed_delay: int = 0
    grade_noise_std: float = 0.0
    actuation_noise_std: float = 0.0

    def _apply_delay(self, values: np.ndarray, delay: int) -> np.ndarray:
        if delay <= 0:
            return values
        delayed = values.astype(np.float32, copy=True)
        delayed[delay:] = values[:-delay]
        delayed[:delay] = values[0]
        return delayed

    def apply_history(self, speed: np.ndarray, grade: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        meas_speed = self._apply_delay(speed, self.speed_delay)
        meas_grade = grade.astype(np.float32, copy=True)
        if self.speed_noise_std > 0:
            meas_speed = meas_speed + np.random.normal(0.0, self.speed_noise_std, size=meas_speed.shape).astype(np.float32)
        if self.grade_noise_std > 0:
            meas_grade = meas_grade + np.random.normal(0.0, self.grade_noise_std, size=meas_grade.shape).astype(np.float32)
        return meas_speed.astype(np.float32), meas_grade.astype(np.float32)

    def apply_future(self, speed: np.ndarray, grade: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        meas_speed = self._apply_delay(speed, self.speed_delay)
        meas_grade = grade.astype(np.float32, copy=True)
        if self.speed_noise_std > 0:
            meas_speed = meas_speed + np.random.normal(0.0, self.speed_noise_std, size=meas_speed.shape).astype(np.float32)
        if self.grade_noise_std > 0:
            meas_grade = meas_grade + np.random.normal(0.0, self.grade_noise_std, size=meas_grade.shape).astype(np.float32)
        return meas_speed.astype(np.float32), meas_grade.astype(np.float32)

    def apply_actuation(self, actions_norm: np.ndarray) -> np.ndarray:
        if self.actuation_noise_std <= 0:
            return clamp_actions_numpy(actions_norm.astype(np.float32))
        noisy = actions_norm + np.random.normal(0.0, self.actuation_noise_std, size=actions_norm.shape).astype(np.float32)
        return clamp_actions_numpy(noisy)


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def simulate_forward(forward: ForwardDynamicsModel,
                     history_speed: np.ndarray,
                     history_actions: np.ndarray,
                     future_actions: np.ndarray,
                     grade_profile: np.ndarray,
                     window: SequenceWindowConfig,
                     features: List[str],
                     device: torch.device) -> np.ndarray:
    history_states = build_state_matrix(history_speed, grade_profile[:window.history], features)
    history_actions_norm = history_actions / np.array([window.max_throttle, window.max_brake], dtype=np.float32)
    future_actions_norm = future_actions / np.array([window.max_throttle, window.max_brake], dtype=np.float32)

    states_tensor = torch.from_numpy(history_states).unsqueeze(0).to(device)
    actions_tensor = torch.from_numpy(history_actions_norm).unsqueeze(0).to(device)
    future_tensor = torch.from_numpy(future_actions_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = forward(states_tensor, actions_tensor, future_tensor)
    return preds.squeeze(0).cpu().numpy()


def simulate_inverse(inverse: InverseActuationModel,
                     history_speed: np.ndarray,
                     history_actions: np.ndarray,
                     target_speed: np.ndarray,
                     grade_profile: np.ndarray,
                     window: SequenceWindowConfig,
                     features: List[str],
                     device: torch.device,
                     perturb: Optional[PerturbationConfig] = None) -> np.ndarray:
    grade_history = grade_profile[:window.history]
    history_speed_meas, grade_history_meas = (
        (history_speed, grade_history)
        if perturb is None
        else perturb.apply_history(history_speed, grade_history)
    )
    desired_grade = grade_profile[window.history : window.history + window.horizon]
    desired_speed_meas, desired_grade_meas = (
        (target_speed, desired_grade)
        if perturb is None
        else perturb.apply_future(target_speed, desired_grade)
    )

    history_states = build_state_matrix(history_speed_meas, grade_history_meas, features)
    desired_states = build_state_matrix(desired_speed_meas, desired_grade_meas, features)

    history_actions_norm = history_actions / np.array([window.max_throttle, window.max_brake], dtype=np.float32)

    history_states_tensor = torch.from_numpy(history_states).unsqueeze(0).to(device)
    history_actions_tensor = torch.from_numpy(history_actions_norm).unsqueeze(0).to(device)
    desired_future_tensor = torch.from_numpy(desired_states).unsqueeze(0).to(device)

    with torch.no_grad():
        feedforward = inverse(history_states_tensor, history_actions_tensor, desired_future_tensor)
    feedforward_actions = clamp_actions(feedforward[..., 0], feedforward[..., 1]).squeeze(0).cpu().numpy()
    feedforward_actions = feedforward_actions * np.array([window.max_throttle, window.max_brake], dtype=np.float32)
    return feedforward_actions


def simulate_closed_loop(forward: ForwardDynamicsModel,
                         inverse: InverseActuationModel,
                         feedback: Optional[FeedbackResidualModel],
                         history_speed: np.ndarray,
                         history_actions: np.ndarray,
                         target_speed: np.ndarray,
                         grade_profile: np.ndarray,
                         window: SequenceWindowConfig,
                         features: List[str],
                         device: torch.device,
                         perturb: Optional[PerturbationConfig] = None,
                         warmup_seconds: float = 0.0,
                         time_step: float = 0.1,
                         warmup_speed_schedule: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    history_len = window.history
    horizon = window.horizon

    scale = np.array([window.max_throttle, window.max_brake], dtype=np.float32)

    if warmup_seconds > 0 or warmup_speed_schedule is not None:
        speed_history_true = np.zeros(history_len, dtype=np.float32)
        grade_history_true = np.zeros(history_len, dtype=np.float32)
        actions_history_norm = np.zeros((history_len, 2), dtype=np.float32)
    else:
        speed_history_true = history_speed.astype(np.float32).copy()
        grade_history_true = grade_profile[:history_len].astype(np.float32).copy()
        actions_history_norm = (history_actions / scale).astype(np.float32)

    simulated_speed: List[float] = []
    applied_actions: List[np.ndarray] = []
    feedforward_actions_rec: List[np.ndarray] = []
    feedback_actions_rec: List[np.ndarray] = []
    warmup_speed_trace: List[float] = []
    warmup_target_speed_trace: List[float] = []
    warmup_grade_trace: List[float] = []
    warmup_actions_trace: List[np.ndarray] = []

    total_steps = target_speed.shape[0] - history_len - horizon + 1
    if total_steps <= 0:
        raise ValueError("Target speed profile must extend beyond the history window")

    if warmup_speed_schedule is not None and warmup_speed_schedule.size > 0:
        warmup_schedule = np.asarray(warmup_speed_schedule, dtype=np.float32)
        desired_init_speed = float(warmup_schedule[-1])
    else:
        warmup_schedule = None
        desired_init_speed = float(target_speed[0]) if target_speed.size else 0.0
    if grade_profile.size > history_len:
        desired_init_grade = float(grade_profile[history_len])
    elif grade_profile.size:
        desired_init_grade = float(grade_profile[-1])
    else:
        desired_init_grade = 0.0

    def controller_step(desired_speed_vec: np.ndarray,
                        desired_grade_vec: np.ndarray,
                        record: bool,
                        apply_noise: bool,
                        log_warmup: bool = False) -> bool:
        nonlocal speed_history_true, grade_history_true, actions_history_norm

        if desired_speed_vec.shape[0] != horizon or desired_grade_vec.shape[0] != horizon:
            return False

        desired_speed_vec = desired_speed_vec.astype(np.float32, copy=False)
        desired_grade_vec = desired_grade_vec.astype(np.float32, copy=False)

        if apply_noise and perturb is not None:
            history_speed_meas, grade_history_meas = perturb.apply_history(speed_history_true, grade_history_true)
            desired_speed_meas, desired_grade_meas = perturb.apply_future(desired_speed_vec, desired_grade_vec)
        else:
            history_speed_meas, grade_history_meas = speed_history_true, grade_history_true
            desired_speed_meas, desired_grade_meas = desired_speed_vec, desired_grade_vec

        history_states_meas = build_state_matrix(history_speed_meas, grade_history_meas, features)
        desired_states_meas = build_state_matrix(desired_speed_meas, desired_grade_meas, features)
        history_states_true = build_state_matrix(speed_history_true, grade_history_true, features)

        history_states_meas_tensor = torch.from_numpy(history_states_meas).unsqueeze(0).to(device)
        history_actions_tensor = torch.from_numpy(actions_history_norm).unsqueeze(0).to(device)
        desired_future_tensor = torch.from_numpy(desired_states_meas).unsqueeze(0).to(device)

        with torch.no_grad():
            feedforward = inverse(history_states_meas_tensor, history_actions_tensor, desired_future_tensor)
            feedforward_actions = clamp_actions(feedforward[..., 0], feedforward[..., 1])

            if feedback is not None:
                history_residual = torch.zeros_like(history_actions_tensor)
                residual = feedback(
                    history_states_meas_tensor,
                    history_actions_tensor,
                    desired_future_tensor,
                    feedforward,
                    history_residual_actions=history_residual,
                )
                combined = feedforward + residual
            else:
                combined = feedforward

            combined = clamp_actions(combined[..., 0], combined[..., 1])

        feedforward_norm = feedforward_actions.squeeze(0).cpu().numpy()
        combined_norm = combined.squeeze(0).cpu().numpy()
        executed_actions_norm = combined_norm

        history_states_true_tensor = torch.from_numpy(history_states_true).unsqueeze(0).to(device)
        executed_tensor = torch.from_numpy(executed_actions_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            future_states_pred = forward(history_states_true_tensor, history_actions_tensor, executed_tensor)

        future_states_np = future_states_pred.squeeze(0).cpu().numpy()
        next_speed = future_states_np[0, 0]
        next_grade = desired_grade_vec[0]

        speed_history_true = np.concatenate([speed_history_true[1:], np.array([next_speed], dtype=np.float32)])
        grade_history_true = np.concatenate([grade_history_true[1:], np.array([next_grade], dtype=np.float32)])
        actions_history_norm = np.concatenate([actions_history_norm[1:], executed_actions_norm[:1]], axis=0)

        if record:
            applied_actions.append(executed_actions_norm[0] * scale)
            simulated_speed.append(float(next_speed))
            feedforward_actions_rec.append(feedforward_norm[0] * scale)
            feedback_actions_rec.append((combined_norm[0] - feedforward_norm[0]) * scale)
        elif log_warmup:
            warmup_speed_trace.append(float(next_speed))
            warmup_target_speed_trace.append(float(desired_speed_vec[0]))
            warmup_grade_trace.append(float(desired_grade_vec[0]))
            warmup_actions_trace.append(executed_actions_norm[0] * scale)

        return True

    if warmup_schedule is not None:
        ramp_steps = warmup_schedule.size
    elif warmup_seconds > 0:
        ramp_steps = max(1, int(np.ceil(warmup_seconds / max(time_step, 1e-6))))
    else:
        ramp_steps = 0

    if ramp_steps > 0:
        settle_tolerance = 0.1
        max_settle_iters = max(2000, ramp_steps * 100)

        if warmup_schedule is None:
            tau = max(1, int(max(1, ramp_steps) * 0.1))

        for i in range(ramp_steps):
            if warmup_schedule is not None:
                speed_target = float(
                    np.clip(
                        warmup_schedule[i],
                        0.0,
                        desired_init_speed if desired_init_speed > 0 else float("inf"),
                    )
                )
                grade_target = (
                    desired_init_grade
                    if desired_init_speed == 0
                    else desired_init_grade
                    * min(speed_target / max(desired_init_speed, 1e-6), 1.0)
                )
            else:
                alpha = 1.0 - math.exp(-(i + 1) / tau)
                alpha = min(alpha, 1.0)
                speed_target = desired_init_speed * alpha
                grade_target = desired_init_grade * alpha
            target_speed_vec = np.full(horizon, speed_target, dtype=np.float32)
            target_grade_vec = np.full(horizon, grade_target, dtype=np.float32)
            controller_step(target_speed_vec, target_grade_vec, record=False, apply_noise=False, log_warmup=True)

        settle_iter = 0
        converged = False
        while settle_iter < max_settle_iters:
            current_error = abs(float(speed_history_true[-1]) - desired_init_speed)
            if current_error <= settle_tolerance:
                converged = True
                break
            target_speed_vec = np.full(horizon, desired_init_speed, dtype=np.float32)
            target_grade_vec = np.full(horizon, desired_init_grade, dtype=np.float32)
            controller_step(target_speed_vec, target_grade_vec, record=False, apply_noise=False, log_warmup=True)
            settle_iter += 1
        if not converged:
            warnings.warn(
                f"Warmup did not converge within {settle_tolerance:.3f} m/s "
                f"after {max_settle_iters} iterations (last error={current_error:.3f}).",
                RuntimeWarning,
            )

        if target_speed.shape[0] >= horizon:
            target_speed[:horizon] = desired_init_speed
        if grade_profile.shape[0] >= history_len:
            grade_profile[:history_len] = grade_history_true

    for step in range(total_steps):
        desired_speed_slice = target_speed[step : step + horizon]
        desired_grade_slice = grade_profile[history_len + step : history_len + step + horizon]
        if not controller_step(desired_speed_slice, desired_grade_slice, record=True, apply_noise=True):
            break

    warmup_actions_array = np.asarray(warmup_actions_trace, dtype=np.float32) if warmup_actions_trace else np.zeros((0, 2), dtype=np.float32)
    warmup_data = {
        "speed": np.asarray(warmup_speed_trace, dtype=np.float32),
        "target_speed": np.asarray(warmup_target_speed_trace, dtype=np.float32),
        "grade": np.asarray(warmup_grade_trace, dtype=np.float32),
        "actions": warmup_actions_array,
    }

    return {
        "simulated_speed": np.asarray(simulated_speed, dtype=np.float32),
        "applied_actions": np.asarray(applied_actions, dtype=np.float32),
        "feedforward_actions": np.asarray(feedforward_actions_rec, dtype=np.float32),
        "feedback_actions": np.asarray(feedback_actions_rec, dtype=np.float32),
        "warmup": warmup_data,
        "final_history_speed": speed_history_true.copy(),
        "final_history_grade": grade_history_true.copy(),
        "final_history_actions": (actions_history_norm * scale).copy(),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_profile(file_path: Optional[Path], length: int, default_fn: Callable[[int], np.ndarray]) -> np.ndarray:
    if file_path:
        arr = np.load(file_path)
        if arr.shape[0] < length:
            raise ValueError(f"Profile {file_path} shorter than required length {length}")
        return arr.astype(np.float32)
    return default_fn(length).astype(np.float32)


def default_speed_profile(length: int) -> np.ndarray:
    t = np.arange(length)
    return 15.0 + 2.0 * np.sin(t / 40.0)


def default_grade_profile(length: int) -> np.ndarray:
    t = np.arange(length)
    return 0.02 * np.sin(t / 60.0)


def default_actuation_profile(length: int) -> np.ndarray:
    throttle = np.clip(40 + 20 * np.sin(np.linspace(0, 2 * np.pi, length)), 0, 100)
    brake = np.zeros_like(throttle)
    return np.stack([throttle, brake], axis=-1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation toolkit for EV longitudinal controllers")
    parser.add_argument("--stage1", type=Path, required=True, help="Stage 1 checkpoint path")
    parser.add_argument("--stage2", type=Path, required=True, help="Stage 2 checkpoint path")
    parser.add_argument("--mode", choices=["forward", "inverse", "closed"], default="closed")
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--state-features", nargs="*", default=["speed", "grade"], help="State feature list")
    parser.add_argument("--speed-noise", type=float, default=0.0)
    parser.add_argument("--speed-delay", type=int, default=0)
    parser.add_argument("--grade-noise", type=float, default=0.0)
    parser.add_argument("--actuation-noise", type=float, default=0.0)
    parser.add_argument("--target-speed", type=Path)
    parser.add_argument("--grade-profile", type=Path)
    parser.add_argument("--actuation-profile", type=Path)
    parser.add_argument("--length", type=int, default=300, help="Total simulation length for closed loop")
    parser.add_argument("--warmup-seconds", type=float, default=15.0, help="Duration of the warmup ramp in seconds")
    parser.add_argument("--time-step", type=float, default=0.1, help="Controller integration timestep in seconds")
    parser.add_argument("--history-dataset", type=Path, help="Optional dataset (.pt) to seed initial history")
    parser.add_argument("--history-segment", type=int, help="Segment index within the dataset for history seeding")
    parser.add_argument("--history-anchor", type=int, help="Anchor index (>= history) for history seeding")
    parser.add_argument("--match-seed-target", action="store_true", help="Replace first history steps of target speed with the seeded history speeds")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = SequenceWindowConfig(history=args.history, horizon=args.horizon)
    state_dim = len(args.state_features)
    action_dim = 2

    forward_model, inverse_model, feedback_model = load_models(args.stage1, args.stage2, state_dim, action_dim, device)

    perturb = PerturbationConfig(
        speed_noise_std=args.speed_noise,
        speed_delay=args.speed_delay,
        grade_noise_std=args.grade_noise,
        actuation_noise_std=args.actuation_noise,
    )

    if args.mode == "forward":
        profile_length = window.history + window.horizon
        history_speed = load_profile(args.target_speed, window.history, default_speed_profile)
        grade_profile = load_profile(args.grade_profile, profile_length, default_grade_profile)
        history_actions = np.zeros((window.history, 2), dtype=np.float32)
        action_profile = load_profile(args.actuation_profile, window.horizon, default_actuation_profile)

        if args.history_dataset:
            seed_speed, seed_actions, seed_grade = load_history_seed_from_dataset(
                args.history_dataset,
                window.history,
                segment_idx=args.history_segment,
                anchor=args.history_anchor,
            )
            history_speed = seed_speed
            history_actions = seed_actions.astype(np.float32)
            grade_profile[:window.history] = seed_grade
            print(
                "Loaded history seed from dataset",
                f"segment={args.history_segment if args.history_segment is not None else 'longest'}",
                f"anchor={args.history_anchor if args.history_anchor is not None else window.history}",
            )

        preds = simulate_forward(
            forward_model,
            history_speed[:window.history],
            history_actions,
            action_profile[:window.horizon],
            grade_profile[:profile_length],
            window,
            args.state_features,
            device,
        )
        print("Predicted speed trajectory:", preds[:, 0])
        return 0

    if args.mode == "inverse":
        profile_length = window.history + window.horizon
        speed_profile = load_profile(args.target_speed, profile_length, default_speed_profile)
        grade_profile = load_profile(args.grade_profile, profile_length, default_grade_profile)
        history_actions = np.zeros((window.history, 2), dtype=np.float32)

        if args.history_dataset:
            seed_speed, seed_actions, seed_grade = load_history_seed_from_dataset(
                args.history_dataset,
                window.history,
                segment_idx=args.history_segment,
                anchor=args.history_anchor,
            )
            speed_profile = speed_profile.copy()
            speed_profile[:window.history] = seed_speed
            history_actions = seed_actions.astype(np.float32)
            grade_profile[:window.history] = seed_grade
            print(
                "Loaded history seed from dataset",
                f"segment={args.history_segment if args.history_segment is not None else 'longest'}",
                f"anchor={args.history_anchor if args.history_anchor is not None else window.history}",
            )

        actions = simulate_inverse(
            inverse_model,
            speed_profile[:window.history],
            history_actions,
            speed_profile[window.history : window.history + window.horizon],
            grade_profile[:profile_length],
            window,
            args.state_features,
            device,
            perturb,
        )
        print("Feedforward action horizon:", actions)
        return 0

    # Closed-loop simulation
    total_length = max(args.length, window.history + window.horizon + 1)
    speed_profile = load_profile(args.target_speed, total_length, default_speed_profile)
    grade_profile = load_profile(args.grade_profile, total_length, default_grade_profile)
    history_speed = np.zeros((window.history,), dtype=np.float32)
    history_actions = np.zeros((window.history, 2), dtype=np.float32)
    warmup_seconds = args.warmup_seconds

    if args.history_dataset:
        seed_speed, seed_actions, seed_grade = load_history_seed_from_dataset(
            args.history_dataset,
            window.history,
            segment_idx=args.history_segment,
            anchor=args.history_anchor,
        )
        history_speed = seed_speed
        history_actions = seed_actions.astype(np.float32)
        if grade_profile.shape[0] < window.history:
            raise ValueError("Grade profile shorter than history window")
        grade_profile[:window.history] = seed_grade
        if args.match_seed_target:
            speed_profile[:window.history] = seed_speed
        warmup_seconds = 0.0
        print(
            "Loaded history seed from dataset",
            f"segment={args.history_segment if args.history_segment is not None else 'longest'}",
            f"anchor={args.history_anchor if args.history_anchor is not None else window.history}",
        )

    results = simulate_closed_loop(
        forward_model,
        inverse_model,
        feedback_model,
        history_speed,
        history_actions,
        speed_profile,
        grade_profile,
        window,
        args.state_features,
        device,
        perturb,
        warmup_seconds=warmup_seconds,
        time_step=args.time_step,
    )

    print("Closed-loop simulated speed (first 10 samples):", results["simulated_speed"][:10])
    print("Closed-loop applied actions (first 10 samples):", results["applied_actions"][:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
