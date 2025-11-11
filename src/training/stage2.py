"""Stage 2 training: feedback residual policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .losses import RegressionLossConfig, regression_loss
from .utils import AverageMeter, move_batch_to_device


@dataclass(slots=True)
class FeedbackTrainingConfig:
    device: Optional[str] = None
    max_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True
    log_interval: int = 50
    residual_weight: float = 1.0
    tracking_weight: float = 0.5
    residual_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    tracking_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    checkpoint_dir: Optional[Path] = None
    best_checkpoint_name: str = "stage2_best.pth"
    latest_checkpoint_name: str = "stage2_latest.pth"
    loss_plot_name: str = "stage2_loss.png"


class FeedbackTrainer:
    """Train feedback model with forward/inverse networks frozen."""

    def __init__(
        self,
        feedback_model: nn.Module,
        forward_model: nn.Module,
        inverse_model: nn.Module,
        cfg: FeedbackTrainingConfig,
    ) -> None:
        self.feedback_model = feedback_model
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.cfg = cfg

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.feedback_model.to(self.device)
        self.forward_model.to(self.device)
        self.inverse_model.to(self.device)

        self.history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.start_epoch: int = 0
        self.forward_model.eval()
        self.inverse_model.eval()
        for param in self.forward_model.parameters():
            param.requires_grad = False
        for param in self.inverse_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.feedback_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = GradScaler(enabled=cfg.use_amp)

        self.checkpoint_dir: Optional[Path] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.latest_checkpoint_path: Optional[Path] = None
        self.loss_plot_path: Optional[Path] = None
        self.best_val_loss: float = float("inf")
        if cfg.checkpoint_dir is not None:
            self.checkpoint_dir = Path(cfg.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint_path = self.checkpoint_dir / cfg.best_checkpoint_name
            self.latest_checkpoint_path = self.checkpoint_dir / cfg.latest_checkpoint_name
            self.loss_plot_path = self.checkpoint_dir / cfg.loss_plot_name

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None, epochs: Optional[int] = None) -> Dict[str, list[float]]:
        history = self.history
        target_total_epochs = epochs if epochs is not None else self.cfg.max_epochs
        self.cfg.max_epochs = target_total_epochs

        if target_total_epochs <= self.start_epoch:
            tqdm.write(
                f"[Stage2] No epochs to run (start_epoch={self.start_epoch}, target={target_total_epochs})."
            )
            return history

        for epoch in range(self.start_epoch + 1, target_total_epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, epoch=epoch)
            history["train_loss"].append(train_metrics["loss"])

            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(val_loader, train=False, epoch=epoch)
                history["val_loss"].append(val_metrics["loss"])

                if self.best_checkpoint_path is not None and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_best_checkpoint(epoch, self.best_val_loss)
            else:
                val_metrics = None

            ckpt_val = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            self.start_epoch = epoch
            self._save_latest_checkpoint(epoch, ckpt_val)
            self._save_loss_plot()

        return history

    # ------------------------------------------------------------------
    def _run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        self.feedback_model.train(mode=train)

        loss_meter = AverageMeter()
        residual_meter = AverageMeter()
        tracking_meter = AverageMeter()

        if not train:
            torch.set_grad_enabled(False)

        iterator = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            disable=not train,
            desc=f"Epoch {epoch}{' [val]' if not train else ''}",
        )

        for step, batch in iterator:
            batch = move_batch_to_device(batch, self.device)
            weights = batch.get("loss_weight")

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                history_states = batch["history_states"]
                history_actions = batch["history_actions"]
                future_states = batch["future_states"]
                future_actions = batch["future_actions"]

                with torch.no_grad():
                    feedforward_future = self.inverse_model(
                        history_states,
                        history_actions,
                        future_states,
                    )

                history_residual = torch.zeros_like(history_actions)

                residual_pred = self.feedback_model(
                    history_states,
                    history_actions,
                    future_states,
                    feedforward_future,
                    history_residual_actions=history_residual,
                )

                residual_target = future_actions - feedforward_future
                residual_loss = regression_loss(
                    residual_pred,
                    residual_target,
                    cfg.residual_loss,
                    weight=weights,
                )

                tracking_loss = torch.tensor(0.0, device=self.device)
                if cfg.tracking_weight > 0:
                    combined_future = feedforward_future + residual_pred
                    tracking_predictions = self.forward_model(
                        history_states,
                        history_actions,
                        combined_future,
                    )
                    tracking_loss = regression_loss(
                        tracking_predictions,
                        future_states,
                        cfg.tracking_loss,
                        weight=weights,
                    )

                total_loss = cfg.residual_weight * residual_loss + cfg.tracking_weight * tracking_loss

            if train:
                self.scaler.scale(total_loss).backward()
                if cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.feedback_model.parameters(), cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_meter.update(total_loss.item(), batch["history_states"].size(0))
            residual_meter.update(residual_loss.item(), batch["history_states"].size(0))
            tracking_meter.update(tracking_loss.item(), batch["history_states"].size(0))

            postfix = {
                ("loss" if train else "val_loss"): f"{loss_meter.average:.4f}",
                ("res" if train else "val_res"): f"{residual_meter.average:.4f}",
                ("trk" if train else "val_trk"): f"{tracking_meter.average:.4f}",
            }
            iterator.set_postfix(postfix)

        if not train:
            torch.set_grad_enabled(True)

        return {"loss": loss_meter.average}

    # ------------------------------------------------------------------
    def _build_checkpoint(self, epoch: int, val_loss: float) -> Dict[str, object]:
        return {
            "epoch": epoch,
            "start_epoch": self.start_epoch,
            "val_loss": float(val_loss),
            "best_val_loss": float(self.best_val_loss),
            "feedback_model": self.feedback_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.cfg),
            "history": {
                "train_loss": list(self.history.get("train_loss", [])),
                "val_loss": list(self.history.get("val_loss", [])),
            },
        }

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.best_checkpoint_path is None:
            return
        torch.save(self._build_checkpoint(epoch, val_loss), self.best_checkpoint_path)
        tqdm.write(
            f"[Stage2] Saved best checkpoint to {self.best_checkpoint_path} (val_loss={val_loss:.4f})"
        )

    def _save_latest_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.latest_checkpoint_path is None:
            return
        torch.save(self._build_checkpoint(epoch, val_loss), self.latest_checkpoint_path)

    def _save_loss_plot(self) -> None:
        if self.loss_plot_path is None:
            return
        train_losses = self.history.get("train_loss", [])
        val_losses = self.history.get("val_loss", [])
        if not train_losses and not val_losses:
            return

        epochs_train = list(range(1, len(train_losses) + 1))
        epochs_val = list(range(1, len(val_losses) + 1))

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        axes[0].plot(epochs_train, train_losses, color="tab:blue", label="Train Loss")
        axes[0].set_ylabel("Train Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        if val_losses:
            axes[1].plot(epochs_val, val_losses, color="tab:orange", label="Val Loss")
            axes[1].legend(loc="upper right")
        else:
            axes[1].text(0.5, 0.5, "No validation data", ha="center", va="center", fontsize=10)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val Loss")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Stage 2 Losses")
        fig.tight_layout()
        fig.savefig(self.loss_plot_path, bbox_inches="tight")
        plt.close(fig)

    def load_checkpoint(
        self,
        path: Path,
        load_optimizer: bool = True,
        load_scaler: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.feedback_model.load_state_dict(checkpoint["feedback_model"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        history = checkpoint.get("history", {})
        self.history["train_loss"] = list(history.get("train_loss", self.history["train_loss"]))
        self.history["val_loss"] = list(history.get("val_loss", self.history["val_loss"]))

        self.start_epoch = int(checkpoint.get("epoch", checkpoint.get("start_epoch", 0)))
        best_val = checkpoint.get("best_val_loss")
        if best_val is not None:
            self.best_val_loss = float(best_val)

        if self.loss_plot_path is not None:
            self._save_loss_plot()

        tqdm.write(f"[Stage2] Resumed from {path} at epoch {self.start_epoch}.")

    def save_final(self, path: Path, val_loss: Optional[float] = None) -> None:
        if val_loss is None:
            if self.history["val_loss"]:
                val_loss = self.history["val_loss"][-1]
            elif self.history["train_loss"]:
                val_loss = self.history["train_loss"][-1]
            else:
                val_loss = 0.0
        torch.save(self._build_checkpoint(self.start_epoch, float(val_loss)), path)


__all__ = ["FeedbackTrainer", "FeedbackTrainingConfig"]


