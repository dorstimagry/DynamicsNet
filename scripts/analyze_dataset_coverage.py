"""Dataset coverage analysis for vehicle state-action space.

This script loads a processed EV dataset (as produced by `scripts/parse_trips.py`)
and computes descriptive statistics and coverage diagnostics for the joint space
spanned by vehicle speed, longitudinal acceleration, throttle, and brake
commands. It produces textual summaries as well as pairwise heatmaps that help
identify sparsely sampled regions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import STATIONARY_ACCEL_EPS, STATIONARY_SPEED_EPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset coverage in action/state space.")
    parser.add_argument("dataset", type=Path, help="Path to processed dataset (.pt)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coverage_reports"),
        help="Directory for plots and JSON summaries (default: ./coverage_reports)",
    )
    parser.add_argument(
        "--sample-time",
        type=float,
        default=None,
        help="Sampling interval in seconds (falls back to dataset metadata or 0.1).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of bins per axis for histograms (default: 80)",
    )
    parser.add_argument(
        "--percentile-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.5, 99.5),
        help="Percentile range to bound histograms (default: 0.5 99.5)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and produce textual/JSON summaries only.",
    )
    return parser.parse_args()


def load_segments(dataset_path: Path) -> tuple[list[dict[str, np.ndarray]], dict]:
    raw = torch.load(dataset_path, map_location="cpu")
    metadata = raw.get("metadata", {})
    segments: list[dict[str, np.ndarray]] = []
    for key, value in raw.items():
        if key == "metadata":
            continue
        segments.append({name: np.asarray(arr) for name, arr in value.items()})
    if not segments:
        raise RuntimeError(f"No trip segments found in dataset {dataset_path}")
    return segments, metadata


def estimate_sample_time(meta: dict, cli_value: Optional[float]) -> float:
    if cli_value is not None:
        return cli_value
    for key in ("sample_time", "dt", "delta_t", "time_step"):
        if key in meta:
            value = meta[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.1


def concatenate(arrays: Iterable[np.ndarray]) -> np.ndarray:
    filtered = [np.asarray(a).astype(np.float64, copy=False).ravel() for a in arrays if a.size]
    if not filtered:
        raise RuntimeError("Cannot concatenate empty collection")
    return np.concatenate(filtered, axis=0)


def compute_acceleration(speed: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("Sample time must be positive to compute acceleration")
    return np.gradient(speed, dt)


def summary_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Encountered dimension with no finite samples")
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    perc_values = np.percentile(finite, percentiles)
    stats = {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }
    stats.update({f"p{p}": float(v) for p, v in zip(percentiles, perc_values, strict=True)})
    return stats


def histogram_2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    prange: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = prange
    x_bounds = np.percentile(x, [lo, hi])
    y_bounds = np.percentile(y, [lo, hi])
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[x_bounds, y_bounds])
    return hist, x_edges, y_edges


def histogram_4d(
    data: np.ndarray,
    bins: int,
    prange: tuple[float, float],
    *,
    weights: np.ndarray | None = None,
    bounds: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    lo, hi = prange
    if bounds is None:
        bounds = [np.percentile(data[:, i], [lo, hi]) for i in range(data.shape[1])]
    hist, edges = np.histogramdd(data, bins=bins, range=bounds, weights=weights)
    return hist, edges


def histogram_1d(values: np.ndarray, bins: int, prange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = prange
    bounds = np.percentile(values, [lo, hi])
    hist, edges = np.histogram(values, bins=bins, range=tuple(bounds))
    return hist, edges


def bin_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) * 0.5


def describe_hist(hist: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray, top_k: int = 5) -> list[dict[str, float]]:
    flat = hist.ravel()
    if flat.sum() == 0:
        return []
    indices = np.argsort(flat)[::-1][:top_k]
    cx = bin_centers(edges_x)
    cy = bin_centers(edges_y)
    descriptions: list[dict[str, float]] = []
    for idx in indices:
        ix, iy = np.unravel_index(idx, hist.shape)
        descriptions.append(
            {
                "x_center": float(cx[ix]),
                "y_center": float(cy[iy]),
                "count": float(flat[idx]),
            }
        )
    return descriptions


def plot_heatmap_comparison(
    raw_hist: np.ndarray,
    weighted_hist: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for ax, hist, label in zip(
        axes,
        (raw_hist, weighted_hist),
        ("Raw counts", "Weighted counts"),
    ):
        mesh = ax.pcolormesh(x_edges, y_edges, hist.T, cmap="viridis", shading="auto")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} – {label}")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Samples per bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_marginal_panel(
    marginals: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
) -> None:
    order = ["speed", "acceleration", "throttle", "brake"]
    titles = {
        "speed": "Speed distribution",
        "acceleration": "Acceleration distribution",
        "throttle": "Throttle distribution",
        "brake": "Brake distribution",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes.flat, order):
        raw_hist, edges = marginals[name]["raw"]
        weighted_hist, _ = marginals[name]["weighted"]
        centers = bin_centers(edges)
        widths = np.diff(edges)
        ax.bar(
            centers,
            raw_hist,
            width=widths,
            align="center",
            edgecolor="black",
            alpha=0.8,
            label="Raw count",
        )
        ax.bar(
            centers,
            weighted_hist,
            width=widths,
            align="center",
            edgecolor="none",
            alpha=0.6,
            label="Weighted count",
        )
        ax.set_title(titles[name])
        ax.set_xlabel(name.capitalize().replace("_", " "))
        ax.set_ylabel("Samples per bin")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def histogram_summary(hist: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray) -> dict:
    total = hist.sum()
    occupancy = float(np.count_nonzero(hist) / hist.size)
    top_bins = describe_hist(hist, edges_x, edges_y)
    return {
        "total_samples": float(total),
        "occupancy_fraction": occupancy,
        "top_bins": top_bins,
    }


def main() -> None:
    args = parse_args()
    segments, metadata = load_segments(args.dataset)
    dt = estimate_sample_time(metadata, args.sample_time)

    speed_samples = concatenate(seg["speed"] for seg in segments)
    throttle_samples = concatenate(seg["throttle"] for seg in segments)
    brake_samples = concatenate(seg["brake"] for seg in segments)

    acceleration_samples = concatenate(compute_acceleration(seg["speed"], dt) for seg in segments)

    stationary_mask = (np.abs(speed_samples) <= STATIONARY_SPEED_EPS) & (
        np.abs(acceleration_samples) <= STATIONARY_ACCEL_EPS
    )
    stationary_fraction = float(np.mean(stationary_mask))
    stationary_percentage = stationary_fraction * 100.0
    stationary_weight = 1.0 / max(stationary_percentage, 1e-6) if stationary_fraction > 0 else 1.0
    sample_weights = np.where(stationary_mask, stationary_weight, 1.0)

    dimensions = {
        "speed": speed_samples,
        "acceleration": acceleration_samples,
        "throttle": throttle_samples,
        "brake": brake_samples,
    }

    stats = {name: summary_stats(values) for name, values in dimensions.items()}

    marginal_histograms: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, values in dimensions.items():
        hist, edges = histogram_1d(values, args.bins, tuple(args.percentile_range))
        weighted_hist, _ = np.histogram(
            values,
            bins=args.bins,
            range=(edges[0], edges[-1]),
            weights=sample_weights,
        )
        marginal_histograms[name] = {
            "raw": (hist, edges),
            "weighted": (weighted_hist, edges),
        }

    pairs = [
        ("speed", "acceleration"),
        ("speed", "throttle"),
        ("speed", "brake"),
        ("acceleration", "throttle"),
        ("acceleration", "brake"),
        ("throttle", "brake"),
    ]

    coverage_summaries: list[dict] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for x_name, y_name in pairs:
        x = dimensions[x_name]
        y = dimensions[y_name]
        prange = tuple(args.percentile_range)
        x_bounds = np.percentile(x, prange)
        y_bounds = np.percentile(y, prange)
        bounds = [x_bounds, y_bounds]
        hist_raw, x_edges, y_edges = np.histogram2d(x, y, bins=args.bins, range=bounds)
        hist_weighted, _, _ = np.histogram2d(
            x,
            y,
            bins=args.bins,
            range=bounds,
            weights=sample_weights,
        )

        coverage_summaries.append(
            {
                "name": f"{x_name}_vs_{y_name}",
                "raw": histogram_summary(hist_raw, x_edges, y_edges),
                "weighted": histogram_summary(hist_weighted, x_edges, y_edges),
            }
        )

        if not args.no_plots:
            plot_path = output_dir / f"heatmap_{x_name}_vs_{y_name}.png"
            plot_heatmap_comparison(
                hist_raw,
                hist_weighted,
                x_edges,
                y_edges,
                x_name.replace("_", " "),
                y_name.replace("_", " "),
                f"{x_name} vs {y_name}",
                plot_path,
            )

    if not args.no_plots:
        marginal_path = output_dir / "marginal_distributions.png"
        plot_marginal_panel(marginal_histograms, marginal_path)

    joint_features = ("speed", "acceleration", "throttle", "brake")
    joint_matrix = np.column_stack([dimensions[name] for name in joint_features])
    prange = tuple(args.percentile_range)
    bounds4d = [np.percentile(joint_matrix[:, i], prange) for i in range(joint_matrix.shape[1])]
    bins4d = max(8, args.bins // 4)
    hist4d_raw, edges4d = histogram_4d(
        joint_matrix,
        bins=bins4d,
        prange=prange,
        bounds=bounds4d,
    )
    hist4d_weighted, _ = histogram_4d(
        joint_matrix,
        bins=bins4d,
        prange=prange,
        weights=sample_weights,
        bounds=bounds4d,
    )
    occupancy4d_raw = float(np.count_nonzero(hist4d_raw) / hist4d_raw.size)
    occupancy4d_weighted = float(np.count_nonzero(hist4d_weighted) / hist4d_weighted.size)

    report = {
        "dataset": str(args.dataset.resolve()),
        "sample_time": dt,
        "metadata": metadata,
        "summary_stats": stats,
        "marginal_histograms": {
            name: {
                "raw": {
                    "edges": hist_info["raw"][1].astype(float).tolist(),
                    "counts": hist_info["raw"][0].astype(int).tolist(),
                },
                "weighted": {
                    "edges": hist_info["weighted"][1].astype(float).tolist(),
                    "counts": hist_info["weighted"][0].astype(float).tolist(),
                },
            }
            for name, hist_info in marginal_histograms.items()
        },
        "pairwise_coverage": coverage_summaries,
        "joint_occupancy_fraction": occupancy4d_raw,
        "joint_occupancy_fraction_weighted": occupancy4d_weighted,
        "joint_total_weight_raw": float(hist4d_raw.sum()),
        "joint_total_weight_weighted": float(hist4d_weighted.sum()),
        "bins": args.bins,
        "percentile_range": list(args.percentile_range),
        "stationary_fraction": stationary_fraction,
        "stationary_percentage": stationary_percentage,
        "stationary_weight": stationary_weight,
        "stationary_thresholds": {
            "speed_abs_max": STATIONARY_SPEED_EPS,
            "accel_abs_max": STATIONARY_ACCEL_EPS,
        },
    }

    json_path = output_dir / "coverage_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Dataset Coverage Summary ===")
    print(f"Dataset: {report['dataset']}")
    print(f"Sample time (s): {report['sample_time']:.6f}")
    print("-- Marginal statistics --")
    for name, values in stats.items():
        print(f"{name.capitalize()}: count={values['count']} mean={values['mean']:.3f} std={values['std']:.3f} min={values['min']:.3f} max={values['max']:.3f}")
    print("-- Stationary window weighting --")
    print(f"Speed≈0 & accel≈0 fraction: {stationary_fraction:.6f} ({stationary_percentage:.3f}%)")
    if not args.no_plots:
        print(f"Weighted marginal histograms saved to: {output_dir / 'marginal_distributions.png'}")
    else:
        print("Weighted marginal histograms skipped (--no-plots)")
    print(f"Loss weight applied to stationary windows: {stationary_weight:.6f}")
    print("-- Pairwise occupancy (raw vs weighted) --")
    for entry in coverage_summaries:
        raw = entry["raw"]
        weighted = entry["weighted"]
        print(
            f"{entry['name']} [raw]: occupancy={raw['occupancy_fraction']:.4f} "
            f"total={raw['total_samples']:.0f}"
        )
        print(
            f"{entry['name']} [weighted]: occupancy={weighted['occupancy_fraction']:.4f} "
            f"total={weighted['total_samples']:.3f}"
        )
    print(
        f"Four-dimensional occupancy fraction (raw): {occupancy4d_raw:.6f} "
        f"(total={hist4d_raw.sum():.0f})"
    )
    print(
        f"Four-dimensional occupancy fraction (weighted): {occupancy4d_weighted:.6f} "
        f"(total={hist4d_weighted.sum():.3f})"
    )
    print(f"Detailed report written to: {json_path}")


if __name__ == "__main__":
    main()

