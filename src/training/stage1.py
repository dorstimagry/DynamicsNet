"""Stage 1 training: joint optimisation of forward & inverse models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .losses import (
    RegressionLossConfig,
    cyclic_consistency_loss,
    regression_loss,
)
from .utils import AverageMeter, move_batch_to_device


@dataclass(slots=True)
class ForwardInverseTrainingConfig:
    device: Optional[str] = None
    max_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True
    log_interval: int = 0
    forward_weight: float = 1.0
    inverse_weight: float = 1.0
    consistency_weight: float = 0.1
    decouple_models: bool = False
    forward_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    inverse_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    consistency_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    checkpoint_dir: Optional[Path] = None
    best_checkpoint_name: str = "stage1_best.pth"
    latest_checkpoint_name: str = "stage1_latest.pth"
    loss_plot_name: str = "stage1_loss.png"


class ForwardInverseTrainer:
    """Jointly train forward and inverse models with consistency regularisation."""

    def __init__(
        self,
        forward_model: nn.Module,
        inverse_model: nn.Module,
        cfg: ForwardInverseTrainingConfig,
    ) -> None:
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.cfg = cfg

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.forward_model.to(self.device)
        self.inverse_model.to(self.device)

        self.history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.start_epoch: int = 0
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

        if cfg.decouple_models:
            self.optimizer_fwd = torch.optim.AdamW(
                self.forward_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            self.optimizer_inv = torch.optim.AdamW(
                self.inverse_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            params = list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
            self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = GradScaler(enabled=cfg.use_amp)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
    ) -> Dict[str, list[float]]:
        history = self.history
        target_total_epochs = epochs if epochs is not None else self.cfg.max_epochs
        self.cfg.max_epochs = target_total_epochs

        if target_total_epochs <= self.start_epoch:
            tqdm.write(
                f"[Stage1] No epochs to run (start_epoch={self.start_epoch}, target={target_total_epochs})."
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
        self.forward_model.train(mode=train)
        self.inverse_model.train(mode=train)

        loss_meter = AverageMeter()
        forward_meter = AverageMeter()
        inverse_meter = AverageMeter()
        consistency_meter = AverageMeter()

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

            if train and not cfg.decouple_models:
                self.optimizer.zero_grad(set_to_none=True)
            elif train and cfg.decouple_models:
                self.optimizer_fwd.zero_grad(set_to_none=True)
                self.optimizer_inv.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                history_states = batch["history_states"]
                history_actions = batch["history_actions"]
                future_states = batch["future_states"]
                future_actions = batch["future_actions"]

                forward_pred = self.forward_model(
                    history_states,
                    history_actions,
                    future_actions,
                )
                forward_loss = regression_loss(
                    forward_pred,
                    future_states,
                    cfg.forward_loss,
                    weight=weights,
                )

                inverse_pred = self.inverse_model(
                    history_states,
                    history_actions,
                    future_states,
                )
                inverse_loss = regression_loss(
                    inverse_pred,
                    future_actions,
                    cfg.inverse_loss,
                    weight=weights,
                )

                consistency = torch.tensor(0.0, device=self.device)
                if not cfg.decouple_models and cfg.consistency_weight > 0:
                    consistency = cyclic_consistency_loss(
                        self.forward_model,
                        self.inverse_model,
                        history_states,
                        history_actions,
                        future_states,
                        cfg.consistency_loss,
                        predicted_actions=inverse_pred,
                        weight=weights,
                    )

                total_loss = (
                    cfg.forward_weight * forward_loss
                    + cfg.inverse_weight * inverse_loss
                    + (cfg.consistency_weight if not cfg.decouple_models else 0.0) * consistency
                )

            if train:
                self.scaler.scale(total_loss).backward()
                if cfg.grad_clip is not None:
                    if cfg.decouple_models:
                        self.scaler.unscale_(self.optimizer_fwd)
                        self.scaler.unscale_(self.optimizer_inv)
                        torch.nn.utils.clip_grad_norm_(
                            self.forward_model.parameters(), cfg.grad_clip
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self.inverse_model.parameters(), cfg.grad_clip
                        )
                    else:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
                            cfg.grad_clip,
                        )
                if cfg.decouple_models:
                    self.scaler.step(self.optimizer_fwd)
                    self.scaler.step(self.optimizer_inv)
                else:
                    self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_meter.update(total_loss.item(), batch["history_states"].size(0))
            forward_meter.update(forward_loss.item(), batch["history_states"].size(0))
            inverse_meter.update(inverse_loss.item(), batch["history_states"].size(0))
            consistency_meter.update(float(consistency.item()), batch["history_states"].size(0))

            if train:
                iterator.set_postfix(
                    {
                        "loss": f"{loss_meter.average:.4f}",
                        "fwd": f"{forward_meter.average:.4f}",
                        "inv": f"{inverse_meter.average:.4f}",
                        "cyc": f"{consistency_meter.average:.4f}",
                    }
                )
            else:
                iterator.set_postfix(
                    {
                        "val_loss": f"{loss_meter.average:.4f}",
                        "val_fwd": f"{forward_meter.average:.4f}",
                        "val_inv": f"{inverse_meter.average:.4f}",
                        "val_cyc": f"{consistency_meter.average:.4f}",
                    }
                )

        if not train:
            torch.set_grad_enabled(True)

        return {"loss": loss_meter.average}

    # ------------------------------------------------------------------
    def _build_checkpoint(self, epoch: int, val_loss: float) -> Dict[str, object]:
        state: Dict[str, object] = {
            "epoch": epoch,
            "start_epoch": self.start_epoch,
            "val_loss": float(val_loss),
            "best_val_loss": float(self.best_val_loss),
            "forward_model": self.forward_model.state_dict(),
            "inverse_model": self.inverse_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.cfg),
            "history": {
                "train_loss": list(self.history.get("train_loss", [])),
                "val_loss": list(self.history.get("val_loss", [])),
            },
        }
        if self.cfg.decouple_models:
            state["optimizer_fwd"] = self.optimizer_fwd.state_dict()
            state["optimizer_inv"] = self.optimizer_inv.state_dict()
        else:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.best_checkpoint_path is None:
            return
        torch.save(self._build_checkpoint(epoch, val_loss), self.best_checkpoint_path)
        tqdm.write(
            f"[Stage1] Saved best checkpoint to {self.best_checkpoint_path} (val_loss={val_loss:.4f})"
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
        else:
            axes[1].text(0.5, 0.5, "No validation data", ha="center", va="center", fontsize=10)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val Loss")
        axes[1].grid(True, alpha=0.3)
        if val_losses:
            axes[1].legend(loc="upper right")

        fig.suptitle("Stage 1 Losses")
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
        self.forward_model.load_state_dict(checkpoint["forward_model"])
        self.inverse_model.load_state_dict(checkpoint["inverse_model"])

        if self.cfg.decouple_models:
            if load_optimizer and "optimizer_fwd" in checkpoint and "optimizer_inv" in checkpoint:
                self.optimizer_fwd.load_state_dict(checkpoint["optimizer_fwd"])
                self.optimizer_inv.load_state_dict(checkpoint["optimizer_inv"])
        else:
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

        tqdm.write(f"[Stage1] Resumed from {path} at epoch {self.start_epoch}.")

    def save_final(self, path: Path, val_loss: Optional[float] = None) -> None:
        if val_loss is None:
            if self.history["val_loss"]:
                val_loss = self.history["val_loss"][-1]
            elif self.history["train_loss"]:
                val_loss = self.history["train_loss"][-1]
            else:
                val_loss = 0.0
        torch.save(self._build_checkpoint(self.start_epoch, float(val_loss)), path)


__all__ = ["ForwardInverseTrainer", "ForwardInverseTrainingConfig"]


