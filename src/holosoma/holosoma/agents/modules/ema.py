"""Exponential Moving Average (EMA) for model weights.

EMA maintains a smoothed copy of model parameters updated at each training step:
    ema_param = decay * ema_param + (1 - decay) * current_param

Ported from the official FPO++ implementation (fpo-control).
"""

from __future__ import annotations

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintains exponential moving average of model parameters.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to track.
    decay : float
        The EMA decay rate (e.g., 0.95, 0.99, 0.999).
    device : torch.device | None
        Device to store EMA parameters on.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.95,
        device: torch.device | None = None,
    ):
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device

        self.shadow_params: dict[str, torch.Tensor] = {}
        self.model_params: dict[str, nn.Parameter] = {}
        self.backup_params: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.model_params[name] = param
                self.shadow_params[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA parameters. Call after each optimizer step."""
        for name, param in self.model_params.items():
            if param.requires_grad:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def reset_to_current(self) -> None:
        """Reset EMA shadow parameters to current model parameters.

        Called at warmup step to initialize EMA with trained weights
        instead of initial weights.
        """
        for name, param in self.model_params.items():
            if param.requires_grad:
                self.shadow_params[name].copy_(param.data.to(self.device))

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow_params = state_dict["shadow_params"]

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters to the model (for evaluation / export)."""
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def store(self, model: nn.Module) -> None:
        """Save current model parameters before temporarily applying EMA weights."""
        self.backup_params = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                self.backup_params[name] = param.data.clone()

    def restore(self, model: nn.Module) -> None:
        """Restore model parameters after using EMA weights."""
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def get_ema_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict with EMA parameters suitable for saving."""
        return {name: param.clone() for name, param in self.shadow_params.items()}
