#!/usr/bin/env python3
"""Evaluate trained models across the entire validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.datasets import EVSequenceDataset
from scripts.evaluate_longest_trip import (
    clamp_actions,
    compute_statistics,
    load_models,
    make_error_plots,
    make_plot,
    make_zoom_plots,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on full validation data")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to processed dataset .pt file")
    parser.add_argument("--stage1", type=Path, required=True, help="Stage 1 checkpoint path")
    parser.add_argument("--stage2", type=Path, required=True, help="Stage 2 checkpoint path")
    parser.add_argument("--val-share", type=float, default=0.1, help="Validation fraction used in training")
    parser.add_argument("--split-seed", type=int, default=123, help="Random split seed")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/validation_full"))
    parser.add_argument("--num-zoom-examples", type=int, default=8, help="Zoomed horizon examples to plot")
    return parser.parse_args(argv)


def split_validation(dataset: EVSequenceDataset, val_share: float, seed: int) -> List[int]:
    val_len = int(len(dataset) * val_share)
    train_len = len(dataset) - val_len
    if val_len <= 0:
        raise ValueError("Validation split empty; adjust val_share")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_len, val_len],
        generator=generator,
    )
    return list(val_subset.indices)


def aggregate_results(dataset: EVSequenceDataset, indices: List[int], stage1: Path, stage2: Path) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_model, inverse_model, feedback_model = load_models(dataset, stage1, stage2, device)

    window = dataset.window
    dt = float(dataset._metadata.get("dt", 1.0))
    history_len = window.history
    horizon = window.horizon

    aggregates: Dict[str, List[np.ndarray]] = {
        "time": [],
        "target_speed": [],
        "predicted_speed": [],
        "gt_throttle": [],
        "gt_brake": [],
        "pred_throttle": [],
        "pred_brake": [],
        "feedforward_throttle": [],
        "feedforward_brake": [],
        "feedback_throttle": [],
        "feedback_brake": [],
        "window_time_offsets": [],
        "target_speed_horizons": [],
        "predicted_speed_horizons": [],
        "gt_throttle_horizons": [],
        "gt_brake_horizons": [],
        "final_throttle_horizons": [],
        "final_brake_horizons": [],
        "feedforward_throttle_horizons": [],
        "feedforward_brake_horizons": [],
        "feedback_throttle_horizons": [],
        "feedback_brake_horizons": [],
    }

    history_residual_template = torch.zeros(1, history_len, 2, device=device)

    for global_idx, idx in enumerate(tqdm(indices, desc="Evaluating validation windows")):
        sample = dataset[idx]
        history_states = sample["history_states"].unsqueeze(0).to(device)
        history_actions = sample["history_actions"].unsqueeze(0).to(device)
        desired_future_states = sample["future_states"].unsqueeze(0).to(device)

        feedforward_future = inverse_model(history_states, history_actions, desired_future_states)
        residual_future = feedback_model(
            history_states,
            history_actions,
            desired_future_states,
            feedforward_future,
            history_residual_actions=history_residual_template,
        )

        feedforward_throttle_norm, feedforward_brake_norm = clamp_actions(
            feedforward_future[..., 0], feedforward_future[..., 1]
        )

        combined_future = feedforward_future + residual_future
        throttle_pred_norm, brake_pred_norm = clamp_actions(
            combined_future[..., 0], combined_future[..., 1]
        )
        combined_actions = torch.stack([throttle_pred_norm, brake_pred_norm], dim=-1)

        forward_pred = forward_model(history_states, history_actions, combined_actions)

        # Convert tensors to cpu numpy
        forward_pred_np = forward_pred.squeeze(0).detach().cpu().numpy()
        desired_future_np = desired_future_states.squeeze(0).detach().cpu().numpy()

        throttle_pred_np = throttle_pred_norm.squeeze(0).detach().cpu().numpy() * window.max_throttle
        brake_pred_np = brake_pred_norm.squeeze(0).detach().cpu().numpy() * window.max_brake
        feedforward_throttle_np = (
            feedforward_throttle_norm.squeeze(0).detach().cpu().numpy() * window.max_throttle
        )
        feedforward_brake_np = (
            feedforward_brake_norm.squeeze(0).detach().cpu().numpy() * window.max_brake
        )
        feedback_throttle_np = (
            residual_future.squeeze(0)[:, 0].detach().cpu().numpy() * window.max_throttle
        )
        feedback_brake_np = (
            residual_future.squeeze(0)[:, 1].detach().cpu().numpy() * window.max_brake
        )

        gt_throttle_np = sample["future_actions"][:, 0].cpu().numpy() * window.max_throttle
        gt_brake_np = sample["future_actions"][:, 1].cpu().numpy() * window.max_brake

        aggregates["time"].append(np.array([global_idx * dt], dtype=np.float32))
        aggregates["target_speed"].append(np.array([desired_future_np[0, 0]], dtype=np.float32))
        aggregates["predicted_speed"].append(np.array([forward_pred_np[0, 0]], dtype=np.float32))
        aggregates["gt_throttle"].append(np.array([gt_throttle_np[0]], dtype=np.float32))
        aggregates["gt_brake"].append(np.array([gt_brake_np[0]], dtype=np.float32))
        aggregates["pred_throttle"].append(np.array([throttle_pred_np[0]], dtype=np.float32))
        aggregates["pred_brake"].append(np.array([brake_pred_np[0]], dtype=np.float32))
        aggregates["feedforward_throttle"].append(np.array([feedforward_throttle_np[0]], dtype=np.float32))
        aggregates["feedforward_brake"].append(np.array([feedforward_brake_np[0]], dtype=np.float32))
        aggregates["feedback_throttle"].append(np.array([feedback_throttle_np[0]], dtype=np.float32))
        aggregates["feedback_brake"].append(np.array([feedback_brake_np[0]], dtype=np.float32))

        time_offsets = np.arange(horizon, dtype=np.float32) * dt
        aggregates["window_time_offsets"].append(time_offsets)
        aggregates["target_speed_horizons"].append(desired_future_np[:, 0])
        aggregates["predicted_speed_horizons"].append(forward_pred_np[:, 0])
        aggregates["gt_throttle_horizons"].append(gt_throttle_np)
        aggregates["gt_brake_horizons"].append(gt_brake_np)
        aggregates["final_throttle_horizons"].append(throttle_pred_np)
        aggregates["final_brake_horizons"].append(brake_pred_np)
        aggregates["feedforward_throttle_horizons"].append(feedforward_throttle_np)
        aggregates["feedforward_brake_horizons"].append(feedforward_brake_np)
        aggregates["feedback_throttle_horizons"].append(feedback_throttle_np)
        aggregates["feedback_brake_horizons"].append(feedback_brake_np)

    # Concatenate lists into arrays
    results: Dict[str, np.ndarray] = {}
    for key, values in aggregates.items():
        if not values:
            raise RuntimeError("No validation samples collected")
        results[key] = np.stack(values, axis=0)

    # time array is currently shape (num_samples, 1); flatten to 1D for plotting convenience
    results["time"] = results["time"].reshape(-1)
    results["target_speed"] = results["target_speed"].reshape(-1)
    results["predicted_speed"] = results["predicted_speed"].reshape(-1)
    results["gt_throttle"] = results["gt_throttle"].reshape(-1)
    results["pred_throttle"] = results["pred_throttle"].reshape(-1)
    results["gt_brake"] = results["gt_brake"].reshape(-1)
    results["pred_brake"] = results["pred_brake"].reshape(-1)
    results["feedforward_throttle"] = results["feedforward_throttle"].reshape(-1)
    results["feedforward_brake"] = results["feedforward_brake"].reshape(-1)
    results["feedback_throttle"] = results["feedback_throttle"].reshape(-1)
    results["feedback_brake"] = results["feedback_brake"].reshape(-1)

    return results


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    dataset = EVSequenceDataset(args.dataset)
    val_indices = split_validation(dataset, args.val_share, args.split_seed)
    results = aggregate_results(dataset, val_indices, args.stage1, args.stage2)

    stats = {
        "speed": compute_statistics(results["target_speed"], results["predicted_speed"]),
        "throttle": compute_statistics(results["gt_throttle"], results["pred_throttle"]),
        "brake": compute_statistics(results["gt_brake"], results["pred_brake"]),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_png = output_dir / "validation_full_overview.png"
    zoom_png = output_dir / "validation_full_zoom.png"
    diagnostics_png = output_dir / "validation_full_diagnostics.png"
    results_npz = output_dir / "validation_full_results.npz"
    stats_json = output_dir / "validation_full_stats.json"

    make_plot(results, summary_png)
    make_zoom_plots(results, zoom_png, num_examples=args.num_zoom_examples)
    make_error_plots(results, diagnostics_png)

    np.savez_compressed(results_npz, **results)
    stats_json.write_text(json.dumps(stats, indent=2))

    print(f"Saved overview to {summary_png}")
    print(f"Saved zoom plot to {zoom_png}")
    print(f"Saved diagnostics to {diagnostics_png}")
    print(f"Saved aggregated results to {results_npz}")
    print(f"Saved statistics to {stats_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


