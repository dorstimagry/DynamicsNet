#!/usr/bin/env python3
"""Train feedback residual controller with frozen forward/inverse models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.data.datasets import EVSequenceDataset, SequenceWindowConfig
from src.models import FeedbackResidualModel, ForwardDynamicsModel, InverseActuationModel
from src.models.common import TransformerConfig
from src.models.feedback import FeedbackModelConfig
from src.models.forward import ForwardModelConfig
from src.models.inverse import InverseModelConfig
from src.training import FeedbackTrainer, FeedbackTrainingConfig, RegressionLossConfig


def build_dataloaders(dataset: EVSequenceDataset, batch_size: int, val_share: float) -> Tuple[DataLoader, DataLoader | None]:
    if val_share <= 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None

    val_len = int(len(dataset) * val_share)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def load_stage1_models(state_path: Path, state_dim: int, action_dim: int, transformer_cfg: TransformerConfig):
    forward_model = ForwardDynamicsModel(ForwardModelConfig(state_dim, action_dim, transformer_cfg))
    inverse_model = InverseActuationModel(InverseModelConfig(state_dim, action_dim, transformer_cfg))

    checkpoint = torch.load(state_path, map_location="cpu")
    forward_model.load_state_dict(checkpoint["forward_model"])
    inverse_model.load_state_dict(checkpoint["inverse_model"])
    return forward_model, inverse_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train feedback controller")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/stage2"))
    parser.add_argument("--loss", choices=["l2", "smooth_l1"], default="l2")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--history", type=int, default=50, help="History window length")
    parser.add_argument("--horizon", type=int, default=200, help="Prediction horizon length")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume training from")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    window_cfg = SequenceWindowConfig(history=args.history, horizon=args.horizon)
    dataset = EVSequenceDataset(args.dataset, window=window_cfg)
    sample = dataset[0]
    state_dim = sample["history_states"].shape[-1]
    action_dim = sample["history_actions"].shape[-1]

    stationary_pct = dataset.stationary_fraction * 100.0
    print(
        f"[Stage2] Stationary windows: {stationary_pct:.3f}% "
        f"-> loss weight {dataset.stationary_weight:.6f}"
    )

    transformer_cfg = TransformerConfig()
    forward_model, inverse_model = load_stage1_models(args.stage1_checkpoint, state_dim, action_dim, transformer_cfg)
    feedback_model = FeedbackResidualModel(FeedbackModelConfig(state_dim, action_dim, transformer_cfg))

    train_loader, val_loader = build_dataloaders(dataset, args.batch_size, args.val_share)

    loss_cfg = RegressionLossConfig(type=args.loss)
    trainer_cfg = FeedbackTrainingConfig(
        device=args.device,
        max_epochs=args.epochs,
        lr=args.lr,
        residual_loss=loss_cfg,
        tracking_loss=loss_cfg,
        checkpoint_dir=args.out_dir,
    )

    trainer = FeedbackTrainer(feedback_model, forward_model, inverse_model, trainer_cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    history = trainer.fit(train_loader, val_loader)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_final(args.out_dir / "stage2.pth")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


