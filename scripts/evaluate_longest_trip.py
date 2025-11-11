#!/usr/bin/env python3
"""Visualise controller behaviour on the longest validation trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split
from tqdm.auto import tqdm

from src.data.datasets import EVSequenceDataset
from src.models import FeedbackResidualModel, ForwardDynamicsModel, InverseActuationModel
from src.models.common import TransformerConfig
from src.models.feedback import FeedbackModelConfig
from src.models.forward import ForwardModelConfig
from src.models.inverse import InverseModelConfig


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned controllers on a validation trip")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to processed dataset (.pt)")
    parser.add_argument("--stage1", type=Path, required=True, help="Stage 1 checkpoint with forward/inverse weights")
    parser.add_argument("--stage2", type=Path, required=True, help="Stage 2 checkpoint with feedback weights")
    parser.add_argument("--val-share", type=float, default=0.1, help="Validation share used during training")
    parser.add_argument("--split-seed", type=int, default=123, help="Random seed used for the stage 2 train/val split")
    parser.add_argument("--output", type=Path, default=Path("plots/validation_longest_trip.png"), help="Where to save the figure")
    parser.add_argument("--results-out", type=Path, help="Optional path to save raw evaluation results (.npz)")
    parser.add_argument("--stats-out", type=Path, help="Optional path to save detailed statistics (.json)")
    return parser.parse_args(argv)


def load_models(dataset: EVSequenceDataset, stage1_path: Path, stage2_path: Path, device: torch.device) -> Tuple[ForwardDynamicsModel, InverseActuationModel, FeedbackResidualModel]:
    sample = dataset[0]
    state_dim = sample["history_states"].shape[-1]
    action_dim = sample["history_actions"].shape[-1]

    transformer_cfg = TransformerConfig()

    forward_model = ForwardDynamicsModel(ForwardModelConfig(state_dim, action_dim, transformer_cfg))
    inverse_model = InverseActuationModel(InverseModelConfig(state_dim, action_dim, transformer_cfg))
    feedback_model = FeedbackResidualModel(FeedbackModelConfig(state_dim, action_dim, transformer_cfg))

    stage1_state = torch.load(stage1_path, map_location="cpu")
    forward_model.load_state_dict(stage1_state["forward_model"])
    inverse_model.load_state_dict(stage1_state["inverse_model"])

    stage2_state = torch.load(stage2_path, map_location="cpu")
    feedback_model.load_state_dict(stage2_state["feedback_model"])

    forward_model.to(device).eval()
    inverse_model.to(device).eval()
    feedback_model.to(device).eval()

    return forward_model, inverse_model, feedback_model


def build_segment_mapping(raw: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    return [key for key in raw.keys() if key != "metadata"]


def pick_longest_validation_segment(dataset: EVSequenceDataset, val_share: float, seed: int, segment_keys: List[str]) -> Tuple[int, Dict[str, np.ndarray]]:
    if val_share <= 0:
        raise ValueError("Validation share must be > 0 to pick a validation segment")

    total = len(dataset)
    val_len = max(1, int(total * val_share))
    train_len = total - val_len
    if train_len <= 0:
        raise ValueError("Validation split leaves no training samples; adjust val_share")

    torch.manual_seed(seed)
    _, val_subset = random_split(dataset, [train_len, val_len])
    val_indices = val_subset.indices

    candidate_segments: Dict[int, int] = {}
    for idx in val_indices:
        seg_idx, _ = dataset._index[idx]
        candidate_segments.setdefault(seg_idx, 0)
        candidate_segments[seg_idx] += 1

    if not candidate_segments:
        raise RuntimeError("Validation split produced no candidate segments")

    def segment_length(seg_idx: int) -> int:
        return len(dataset._segments[seg_idx]["speed"])

    longest_seg_idx = max(candidate_segments.keys(), key=segment_length)
    segment = dataset._segments[longest_seg_idx]
    segment_name = segment_keys[longest_seg_idx] if longest_seg_idx < len(segment_keys) else f"segment_{longest_seg_idx}"
    print(f"Selected validation segment: {segment_name} (length={segment_length(longest_seg_idx)} samples)")
    return longest_seg_idx, segment


def build_state_matrix(segment: Dict[str, np.ndarray], features: List[str]) -> np.ndarray:
    states: List[np.ndarray] = []
    for name in features:
        if name == "speed":
            states.append(segment["speed"])
        elif name == "grade":
            angle = segment.get("angle")
            if angle is None:
                states.append(np.zeros_like(segment["speed"]))
            else:
                states.append(np.sin(angle))
        elif name == "angle":
            angle = segment.get("angle")
            states.append(angle if angle is not None else np.zeros_like(segment["speed"]))
        else:
            raise KeyError(f"Unsupported state feature: {name}")
    return np.stack(states, axis=-1).astype(np.float32)


def clamp_actions(throttle: torch.Tensor, brake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    throttle = torch.clamp(throttle, 0.0, 1.0)
    brake = torch.clamp(brake, 0.0, 1.0)
    both_active = (throttle > 0) & (brake > 0)
    throttle = torch.where(both_active & (throttle >= brake), throttle, torch.where(both_active, torch.zeros_like(throttle), throttle))
    brake = torch.where(both_active & (brake > throttle), brake, torch.where(both_active, torch.zeros_like(brake), brake))
    return throttle, brake


def run_inference(
    segment: Dict[str, np.ndarray],
    dataset: EVSequenceDataset,
    forward_model: ForwardDynamicsModel,
    inverse_model: InverseActuationModel,
    feedback_model: FeedbackResidualModel,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    window = dataset.window
    history = window.history
    horizon = window.horizon

    speed = segment["speed"].astype(np.float32)
    throttle = segment["throttle"].astype(np.float32)
    brake = segment["brake"].astype(np.float32)
    time_axis = segment["time"]

    states = build_state_matrix(segment, dataset.state_features)
    throttle_norm = throttle / window.max_throttle
    brake_norm = brake / window.max_brake
    actions = np.stack([throttle_norm, brake_norm], axis=-1).astype(np.float32)

    num_windows = len(speed) - history - horizon + 1
    if num_windows <= 0:
        raise ValueError("Segment too short for configured history/horizon")

    times: List[float] = []
    target_speed: List[float] = []
    predicted_speed: List[float] = []
    gt_throttle: List[float] = []
    gt_brake: List[float] = []
    pred_throttle: List[float] = []
    pred_brake: List[float] = []
    feedforward_throttle_vals: List[float] = []
    feedforward_brake_vals: List[float] = []
    feedback_throttle_vals: List[float] = []
    feedback_brake_vals: List[float] = []
    final_throttle_seq_list: List[np.ndarray] = []
    final_brake_seq_list: List[np.ndarray] = []
    feedforward_throttle_seq_list: List[np.ndarray] = []
    feedforward_brake_seq_list: List[np.ndarray] = []
    feedback_throttle_seq_list: List[np.ndarray] = []
    feedback_brake_seq_list: List[np.ndarray] = []
    gt_throttle_seq_list: List[np.ndarray] = []
    gt_brake_seq_list: List[np.ndarray] = []
    target_speed_seq_list: List[np.ndarray] = []
    predicted_speed_seq_list: List[np.ndarray] = []
    window_time_offsets: List[np.ndarray] = []

    history_residual = torch.zeros(1, history, actions.shape[-1], device=device)

    with torch.no_grad():
        iterator = tqdm(
            range(history, history + num_windows),
            total=num_windows,
            desc="Simulating horizon windows",
            leave=False,
        )
        for anchor in iterator:
            hist_states = torch.from_numpy(states[anchor - history : anchor]).unsqueeze(0).to(device)
            hist_actions = torch.from_numpy(actions[anchor - history : anchor]).unsqueeze(0).to(device)
            desired_future = torch.from_numpy(states[anchor : anchor + horizon]).unsqueeze(0).to(device)

            feedforward_future = inverse_model(hist_states, hist_actions, desired_future)

            residual_future = feedback_model(
                hist_states,
                hist_actions,
                desired_future,
                feedforward_future,
                history_residual_actions=history_residual,
            )

            feedforward_throttle_norm, feedforward_brake_norm = clamp_actions(
                feedforward_future[..., 0], feedforward_future[..., 1]
            )

            combined_future = feedforward_future + residual_future
            throttle_pred, brake_pred = clamp_actions(combined_future[..., 0], combined_future[..., 1])
            combined = torch.stack([throttle_pred, brake_pred], dim=-1)

            forward_pred = forward_model(hist_states, hist_actions, combined)

            time_offsets = time_axis[anchor : anchor + horizon] - time_axis[anchor]
            window_time_offsets.append(time_offsets.astype(np.float32))

            final_throttle_seq_list.append(
                throttle_pred[0].detach().cpu().numpy() * window.max_throttle
            )
            final_brake_seq_list.append(
                brake_pred[0].detach().cpu().numpy() * window.max_brake
            )
            feedforward_throttle_seq_list.append(
                feedforward_throttle_norm[0].detach().cpu().numpy() * window.max_throttle
            )
            feedforward_brake_seq_list.append(
                feedforward_brake_norm[0].detach().cpu().numpy() * window.max_brake
            )
            feedback_throttle_seq_list.append(
                residual_future[0, :, 0].detach().cpu().numpy() * window.max_throttle
            )
            feedback_brake_seq_list.append(
                residual_future[0, :, 1].detach().cpu().numpy() * window.max_brake
            )
            gt_throttle_seq_list.append(throttle[anchor : anchor + horizon].copy())
            gt_brake_seq_list.append(brake[anchor : anchor + horizon].copy())
            target_speed_seq_list.append(desired_future[0, :, 0].detach().cpu().numpy())
            predicted_speed_seq_list.append(forward_pred[0, :, 0].detach().cpu().numpy())

            predicted_speed.append(forward_pred[0, 0, 0].item())
            target_speed.append(desired_future[0, 0, 0].item())
            times.append(float(time_axis[anchor]))

            pred_throttle.append(throttle_pred[0, 0].item() * window.max_throttle)
            pred_brake.append(brake_pred[0, 0].item() * window.max_brake)
            gt_throttle.append(float(throttle[anchor]))
            gt_brake.append(float(brake[anchor]))

            feedforward_throttle_vals.append(feedforward_throttle_norm[0, 0].item() * window.max_throttle)
            feedforward_brake_vals.append(feedforward_brake_norm[0, 0].item() * window.max_brake)
            feedback_throttle_vals.append(residual_future[0, 0, 0].item() * window.max_throttle)
            feedback_brake_vals.append(residual_future[0, 0, 1].item() * window.max_brake)

    return {
        "time": np.asarray(times),
        "target_speed": np.asarray(target_speed),
        "predicted_speed": np.asarray(predicted_speed),
        "gt_throttle": np.asarray(gt_throttle),
        "gt_brake": np.asarray(gt_brake),
        "pred_throttle": np.asarray(pred_throttle),
        "pred_brake": np.asarray(pred_brake),
        "feedforward_throttle": np.asarray(feedforward_throttle_vals),
        "feedforward_brake": np.asarray(feedforward_brake_vals),
        "feedback_throttle": np.asarray(feedback_throttle_vals),
        "feedback_brake": np.asarray(feedback_brake_vals),
        "window_time_offsets": np.stack(window_time_offsets, axis=0),
        "target_speed_horizons": np.stack(target_speed_seq_list, axis=0),
        "predicted_speed_horizons": np.stack(predicted_speed_seq_list, axis=0),
        "gt_throttle_horizons": np.stack(gt_throttle_seq_list, axis=0),
        "gt_brake_horizons": np.stack(gt_brake_seq_list, axis=0),
        "final_throttle_horizons": np.stack(final_throttle_seq_list, axis=0),
        "final_brake_horizons": np.stack(final_brake_seq_list, axis=0),
        "feedforward_throttle_horizons": np.stack(feedforward_throttle_seq_list, axis=0),
        "feedforward_brake_horizons": np.stack(feedforward_brake_seq_list, axis=0),
        "feedback_throttle_horizons": np.stack(feedback_throttle_seq_list, axis=0),
        "feedback_brake_horizons": np.stack(feedback_brake_seq_list, axis=0),
    }


def make_plot(results: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(results["time"], results["target_speed"], label="GT speed", color="tab:blue")
    axes[0].plot(results["time"], results["predicted_speed"], label="Predicted speed", color="tab:orange")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Vehicle speed tracking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["time"], results["gt_throttle"], label="GT throttle", color="tab:green")
    axes[1].plot(results["time"], results["pred_throttle"], label="Pred throttle", linestyle="--", color="tab:orange")
    axes[1].set_ylabel("Throttle (%)")
    axes[1].set_title("Throttle comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results["time"], results["gt_brake"], label="GT brake", color="tab:red")
    axes[2].plot(results["time"], results["pred_brake"], label="Pred brake", linestyle="--", color="tab:purple")
    axes[2].set_ylabel("Brake (%)")
    axes[2].set_title("Brake comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(results["time"], results["feedforward_throttle"], label="Feedforward throttle", color="tab:olive")
    axes[3].plot(results["time"], results["feedback_throttle"], label="Feedback throttle", linestyle="--", color="tab:brown")
    axes[3].set_ylabel("Throttle (%)")
    axes[3].set_title("Feedforward vs Feedback Throttle")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(results["time"], results["feedforward_brake"], label="Feedforward brake", color="tab:pink")
    axes[4].plot(results["time"], results["feedback_brake"], label="Feedback brake", linestyle="--", color="tab:gray")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Brake (%)")
    axes[4].set_title("Feedforward vs Feedback Brake")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualisation to {output_path}")


def make_zoom_plots(results: Dict[str, np.ndarray], output_path: Path, num_examples: int = 4) -> None:
    time_offsets = results["window_time_offsets"]
    target_speed = results["target_speed_horizons"]
    predicted_speed = results["predicted_speed_horizons"]
    gt_throttle = results["gt_throttle_horizons"]
    gt_brake = results["gt_brake_horizons"]
    final_throttle = results["final_throttle_horizons"]
    final_brake = results["final_brake_horizons"]
    feedforward_throttle = results["feedforward_throttle_horizons"]
    feedforward_brake = results["feedforward_brake_horizons"]
    feedback_throttle = results["feedback_throttle_horizons"]
    feedback_brake = results["feedback_brake_horizons"]

    num_windows = time_offsets.shape[0]
    if num_windows == 0:
        return
    num_examples = min(num_examples, num_windows)
    indices = np.linspace(0, num_windows - 1, num=num_examples, dtype=int)

    fig, axes = plt.subplots(3, num_examples, figsize=(4 * num_examples, 9), sharex="col")
    if num_examples == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, idx in enumerate(indices):
        t = time_offsets[idx]

        ax_speed = axes[0, col]
        ax_speed.plot(t, target_speed[idx], label="GT speed", color="tab:blue")
        ax_speed.plot(t, predicted_speed[idx], label="Pred speed", color="tab:orange")
        ax_speed.set_ylabel("Speed (m/s)")
        ax_speed.set_title(f"Window {idx}")
        ax_speed.grid(True, alpha=0.3)
        if col == 0:
            ax_speed.legend()

        ax_throttle = axes[1, col]
        ax_throttle.plot(t, gt_throttle[idx], label="GT throttle", color="tab:green")
        ax_throttle.plot(t, feedforward_throttle[idx], label="Feedforward", linestyle="--", color="tab:olive")
        ax_throttle.plot(t, final_throttle[idx], label="Final", linestyle="-.", color="tab:orange")
        ax_throttle.plot(t, feedback_throttle[idx], label="Feedback", linestyle=":", color="tab:brown")
        ax_throttle.set_ylabel("Throttle (%)")
        ax_throttle.grid(True, alpha=0.3)
        if col == 0:
            ax_throttle.legend()

        ax_brake = axes[2, col]
        ax_brake.plot(t, gt_brake[idx], label="GT brake", color="tab:red")
        ax_brake.plot(t, feedforward_brake[idx], label="Feedforward", linestyle="--", color="tab:pink")
        ax_brake.plot(t, final_brake[idx], label="Final", linestyle="-.", color="tab:purple")
        ax_brake.plot(t, feedback_brake[idx], label="Feedback", linestyle=":", color="tab:gray")
        ax_brake.set_ylabel("Brake (%)")
        ax_brake.set_xlabel("Time offset (s)")
        ax_brake.grid(True, alpha=0.3)
        if col == 0:
            ax_brake.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved zoomed horizon examples to {output_path}")


def make_error_plots(results: Dict[str, np.ndarray], output_path: Path) -> None:
    speed_error_matrix = results["predicted_speed_horizons"] - results["target_speed_horizons"]
    throttle_error_matrix = results["final_throttle_horizons"] - results["gt_throttle_horizons"]
    brake_error_matrix = results["final_brake_horizons"] - results["gt_brake_horizons"]

    speed_errors = speed_error_matrix.ravel()
    throttle_errors = throttle_error_matrix.ravel()
    brake_errors = brake_error_matrix.ravel()

    throttle_gt = results["gt_throttle_horizons"].ravel()
    throttle_pred = results["final_throttle_horizons"].ravel()
    brake_gt = results["gt_brake_horizons"].ravel()
    brake_pred = results["final_brake_horizons"].ravel()

    horizon_length = results["target_speed_horizons"].shape[1]
    horizon_indices = np.arange(horizon_length)
    speed_mae_per_h = np.mean(np.abs(speed_error_matrix), axis=0)
    throttle_mae_per_h = np.mean(np.abs(throttle_error_matrix), axis=0)
    brake_mae_per_h = np.mean(np.abs(brake_error_matrix), axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(speed_errors, bins=50, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title("Speed Error Distribution")
    axes[0, 0].set_xlabel("Error (m/s)")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(throttle_errors, bins=50, color="tab:orange", alpha=0.8)
    axes[0, 1].set_title("Throttle Error Distribution")
    axes[0, 1].set_xlabel("Error (%)")
    axes[0, 1].set_ylabel("Count")

    axes[0, 2].hist(brake_errors, bins=50, color="tab:red", alpha=0.8)
    axes[0, 2].set_title("Brake Error Distribution")
    axes[0, 2].set_xlabel("Error (%)")
    axes[0, 2].set_ylabel("Count")

    axes[1, 0].scatter(throttle_gt, throttle_pred, s=5, alpha=0.3, color="tab:orange")
    axes[1, 0].set_title("Throttle: GT vs Prediction")
    axes[1, 0].set_xlabel("GT throttle (%)")
    axes[1, 0].set_ylabel("Pred throttle (%)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(brake_gt, brake_pred, s=5, alpha=0.3, color="tab:red")
    axes[1, 1].set_title("Brake: GT vs Prediction")
    axes[1, 1].set_xlabel("GT brake (%)")
    axes[1, 1].set_ylabel("Pred brake (%)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(horizon_indices, speed_mae_per_h, label="Speed", color="tab:blue")
    axes[1, 2].plot(horizon_indices, throttle_mae_per_h, label="Throttle", color="tab:orange")
    axes[1, 2].plot(horizon_indices, brake_mae_per_h, label="Brake", color="tab:red")
    axes[1, 2].set_title("MAE vs Horizon Step")
    axes[1, 2].set_xlabel("Horizon step")
    axes[1, 2].set_ylabel("MAE")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved error diagnostics to {output_path}")


def compute_statistics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    median_ae = float(np.median(np.abs(error)))
    p95_ae = float(np.percentile(np.abs(error), 95))
    max_abs_error = float(np.max(np.abs(error)))
    min_error = float(np.min(error))
    max_error = float(np.max(error))
    bias = float(np.mean(error))
    mean_gt = float(np.mean(target))
    mean_pred = float(np.mean(prediction))
    std_gt = float(np.std(target))
    std_pred = float(np.std(prediction))
    denom = float(np.sum((target - mean_gt) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denom) if denom > 1e-8 else float("nan")
    if std_gt > 1e-8 and std_pred > 1e-8:
        corr = float(np.corrcoef(target, prediction)[0, 1])
    else:
        corr = float("nan")

    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "median_abs_error": median_ae,
        "p95_abs_error": p95_ae,
        "max_abs_error": max_abs_error,
        "min_error": min_error,
        "max_error": max_error,
        "bias": bias,
        "mean_ground_truth": mean_gt,
        "mean_prediction": mean_pred,
        "std_ground_truth": std_gt,
        "std_prediction": std_pred,
        "r2": r2,
        "correlation": corr,
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    dataset = EVSequenceDataset(args.dataset)
    raw = torch.load(args.dataset)
    segment_keys = build_segment_mapping(raw)

    stage1_path = args.stage1
    stage2_path = args.stage2
    if not stage1_path.exists():
        raise FileNotFoundError(stage1_path)
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_model, inverse_model, feedback_model = load_models(dataset, stage1_path, stage2_path, device)

    _, segment = pick_longest_validation_segment(dataset, args.val_share, args.split_seed, segment_keys)
    results = run_inference(segment, dataset, forward_model, inverse_model, feedback_model, device)

    stats = {
        "speed": compute_statistics(results["target_speed"], results["predicted_speed"]),
        "throttle": compute_statistics(results["gt_throttle"], results["pred_throttle"]),
        "brake": compute_statistics(results["gt_brake"], results["pred_brake"]),
    }

    print("Detailed statistics:")
    for key, value in stats.items():
        print(f"  {key.capitalize()} metrics:")
        for metric, scalar in value.items():
            print(f"    {metric}: {scalar:.6f}")

    make_plot(results, args.output)
    zoom_output = args.output.with_name(f"{args.output.stem}_zoom{args.output.suffix}")
    diagnostics_output = args.output.with_name(f"{args.output.stem}_diagnostics{args.output.suffix}")
    make_zoom_plots(results, zoom_output)
    make_error_plots(results, diagnostics_output)

    results_path = args.results_out or args.output.with_suffix(".npz")
    stats_path = args.stats_out or args.output.with_suffix(".json")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(results_path, **results)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Saved raw results to {results_path}")
    print(f"Saved statistics to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


