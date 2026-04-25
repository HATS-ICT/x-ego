from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.optim import AdamW, Optimizer


_MUON_EXCLUDED_NAME_PARTS = (
    "bias",
    "bn",
    "classifier",
    "embed",
    "embedding",
    "embeddings",
    "head",
    "layernorm",
    "logit",
    "norm",
    "projector",
    "token",
)


def _cfg_select(cfg, key: str, default=None):
    return OmegaConf.select(cfg, key, default=default)


def _as_tuple(value, default):
    if value is None:
        return default
    return tuple(value)


def _is_muon_hidden_weight(name: str, param) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False

    normalized_name = name.lower()
    if any(part in normalized_name for part in _MUON_EXCLUDED_NAME_PARTS):
        return False

    # For ConvNets, keep the first input convolution on AdamW as recommended.
    if normalized_name.endswith("conv1.weight"):
        return False

    return True


def _trainable_params(model: nn.Module) -> Iterable:
    return (param for param in model.parameters() if param.requires_grad)


def build_optimizer(model: nn.Module, opt_config) -> Optimizer:
    optimizer_name = str(_cfg_select(opt_config, "optimizer", "adamw")).lower()

    if optimizer_name == "adamw":
        return AdamW(
            _trainable_params(model),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            fused=opt_config.fused_optimizer,
        )

    if optimizer_name == "muon":
        return _build_muon_optimizer(model, opt_config)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_muon_optimizer(model: nn.Module, opt_config) -> Optimizer:
    try:
        from muon import MuonWithAuxAdam, adam_update, muon_update
    except ImportError as exc:
        raise ImportError(
            "optimization.optimizer=muon requires the `muon` package. "
            "Install it in the environment that runs training."
        ) from exc

    hidden_weights = []
    aux_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_hidden_weight(name, param):
            hidden_weights.append(param)
        else:
            aux_params.append(param)

    if not hidden_weights:
        return AdamW(
            aux_params,
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            fused=opt_config.fused_optimizer,
        )

    param_groups = [
        {
            "params": hidden_weights,
            "use_muon": True,
            "lr": _cfg_select(opt_config, "muon.lr", 0.02),
            "weight_decay": _cfg_select(opt_config, "muon.weight_decay", opt_config.weight_decay),
        }
    ]

    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "use_muon": False,
                "lr": opt_config.lr,
                "betas": _as_tuple(_cfg_select(opt_config, "betas", None), (0.9, 0.95)),
                "weight_decay": opt_config.weight_decay,
            }
        )

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        class SingleProcessMuonWithAuxAdam(MuonWithAuxAdam):
            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    if group["use_muon"]:
                        for param in group["params"]:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)

                            state = self.state[param]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(param)

                            update = muon_update(
                                param.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                            )
                            param.mul_(1 - group["lr"] * group["weight_decay"])
                            param.add_(update.reshape(param.shape), alpha=-group["lr"])
                    else:
                        for param in group["params"]:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)

                            state = self.state[param]
                            if len(state) == 0:
                                state["exp_avg"] = torch.zeros_like(param)
                                state["exp_avg_sq"] = torch.zeros_like(param)
                                state["step"] = 0

                            state["step"] += 1
                            update = adam_update(
                                param.grad,
                                state["exp_avg"],
                                state["exp_avg_sq"],
                                state["step"],
                                group["betas"],
                                group["eps"],
                            )
                            param.mul_(1 - group["lr"] * group["weight_decay"])
                            param.add_(update, alpha=-group["lr"])

                return loss

        return SingleProcessMuonWithAuxAdam(param_groups)

    return MuonWithAuxAdam(param_groups)
