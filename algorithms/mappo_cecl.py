from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.algorithms.common import Algorithm
from benchmarl.models.common import ModelConfig


def _reshape_to_b_a_d(emb: torch.Tensor, n_agents: int) -> torch.Tensor:
    if emb.ndim < 2:
        raise ValueError(f"Expected embedding with agent dim, got shape={tuple(emb.shape)}")
    if emb.shape[-2] != n_agents:
        raise ValueError(
            f"Expected agent dim at -2 with size n_agents={n_agents}, got shape={tuple(emb.shape)}"
        )
    return emb.reshape(-1, n_agents, emb.shape[-1])


def _create_label_matrix(batch_size: int, num_agents: int, device: torch.device) -> torch.Tensor:
    labels = torch.zeros(batch_size * num_agents, batch_size * num_agents, device=device)
    for b in range(batch_size):
        start_idx = b * num_agents
        end_idx = (b + 1) * num_agents
        labels[start_idx:end_idx, start_idx:end_idx] = 1.0
    return labels


class CeclLossModule(torch.nn.Module):
    def __init__(self, n_agents: int):
        super().__init__()
        self.n_agents = n_agents

    def forward(
        self,
        embeddings: torch.Tensor,
        start_times: Optional[torch.Tensor] = None,
        end_times: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = _reshape_to_b_a_d(embeddings, self.n_agents)
        batch_size, n_agents, emb_dim = embeddings.shape

        normalized = F.normalize(embeddings, dim=-1, eps=1e-6)
        flat = normalized.reshape(batch_size * n_agents, emb_dim)
        logits = torch.matmul(flat, flat.t())

        if logit_scale is not None:
            logits = logits * logit_scale
        if logit_bias is not None:
            logits = logits + logit_bias

        labels = _create_label_matrix(batch_size, n_agents, device=embeddings.device)

        mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
        labels = labels.masked_fill(mask, 0.0)

        m1_diag1 = 2 * labels - 1
        loglik = F.logsigmoid(m1_diag1 * logits)
        loglik = loglik.masked_fill(mask, 0.0)

        valid = ~mask
        if start_times is not None and end_times is not None:
            start_flat = start_times.reshape(-1)
            end_flat = end_times.reshape(-1)
            non_overlap = (end_flat[:, None] < start_flat[None, :]) | (
                start_flat[:, None] > end_flat[None, :]
            )
            valid = valid & ((labels == 1) | non_overlap)

        valid_counts = valid.float().sum(dim=-1)
        loglik = loglik.masked_fill(~valid, 0.0)
        nll = -(loglik.sum(dim=-1) / valid_counts.clamp_min(1.0))
        return nll.mean()


class CeclWrappedLoss(LossModule):
    def __init__(
        self,
        ppo_loss: ClipPPOLoss,
        cecl_loss: CeclLossModule,
        group: str,
        emb_key: str,
        start_time_key: Optional[str],
        end_time_key: Optional[str],
        logit_scale_key: Optional[str],
        logit_bias_key: Optional[str],
    ):
        super().__init__()
        self.ppo_loss = ppo_loss
        self.cecl_loss = cecl_loss
        self.group = group
        self.emb_key = emb_key
        self.start_time_key = start_time_key
        self.end_time_key = end_time_key
        self.logit_scale_key = logit_scale_key
        self.logit_bias_key = logit_bias_key

    def forward(self, batch: TensorDictBase) -> TensorDictBase:
        loss_td = self.ppo_loss(batch)

        if (self.group, self.emb_key) not in batch.keys(True, True):
            _ = self.ppo_loss.actor(batch)

        embeddings = batch.get((self.group, self.emb_key))
        start_times = (
            batch.get((self.group, self.start_time_key))
            if self.start_time_key is not None
            else None
        )
        end_times = (
            batch.get((self.group, self.end_time_key))
            if self.end_time_key is not None
            else None
        )
        logit_scale = (
            batch.get((self.group, self.logit_scale_key))
            if self.logit_scale_key is not None
            else None
        )
        logit_bias = (
            batch.get((self.group, self.logit_bias_key))
            if self.logit_bias_key is not None
            else None
        )

        if logit_scale is not None:
            logit_scale = logit_scale.reshape(-1)[0]
        if logit_bias is not None:
            logit_bias = logit_bias.reshape(-1)[0]

        loss_cecl = self.cecl_loss(
            embeddings,
            start_times=start_times,
            end_times=end_times,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        loss_td.set("loss_cecl", loss_cecl)
        return loss_td


class MappoCecl(Mappo):
    def __init__(
        self,
        lambda_cecl: float,
        cecl_emb_key: str = "cecl_emb",
        cecl_use_logit_scale: bool = False,
        cecl_use_logit_bias: bool = False,
        cecl_logit_scale_key: str = "logit_scale",
        cecl_logit_bias_key: str = "logit_bias",
        cecl_use_time_mask: bool = True,
        cecl_start_time_key: str = "obs_start_time",
        cecl_end_time_key: str = "obs_end_time",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_cecl = lambda_cecl
        self.cecl_emb_key = cecl_emb_key
        self.cecl_use_logit_scale = cecl_use_logit_scale
        self.cecl_use_logit_bias = cecl_use_logit_bias
        self.cecl_logit_scale_key = cecl_logit_scale_key
        self.cecl_logit_bias_key = cecl_logit_bias_key
        self.cecl_use_time_mask = cecl_use_time_mask
        self.cecl_start_time_key = cecl_start_time_key
        self.cecl_end_time_key = cecl_end_time_key

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = CompositeSpec(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        cecl_dim = getattr(model_config, "projector_out_dim", None)
        if cecl_dim is None:
            raise ValueError(
                "CECL requires actor config to set projector_out_dim for embedding spec."
            )

        actor_output_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {"logits": UnboundedContinuousTensorSpec(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=IndependentNormal
                if not self.use_tanh_normal
                else TanhNormal,
                distribution_kwargs={
                    "min": self.action_spec[(group, "action")].space.low,
                    "max": self.action_spec[(group, "action")].space.high,
                }
                if self.use_tanh_normal
                else {},
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )

        cecl_loss = CeclLossModule(n_agents=len(self.group_map[group]))
        wrapped = CeclWrappedLoss(
            ppo_loss=loss_module,
            cecl_loss=cecl_loss,
            group=group,
            emb_key=self.cecl_emb_key,
            start_time_key=self.cecl_start_time_key if self.cecl_use_time_mask else None,
            end_time_key=self.cecl_end_time_key if self.cecl_use_time_mask else None,
            logit_scale_key=self.cecl_logit_scale_key if self.cecl_use_logit_scale else None,
            logit_bias_key=self.cecl_logit_bias_key if self.cecl_use_logit_bias else None,
        )
        return wrapped, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        if isinstance(loss, CeclWrappedLoss):
            return super()._get_parameters(group, loss.ppo_loss)
        return super()._get_parameters(group, loss)

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals = super().process_loss_vals(group, loss_vals)
        if "loss_cecl" in loss_vals.keys():
            loss_vals.set(
                "loss_objective",
                loss_vals["loss_objective"] + self.lambda_cecl * loss_vals["loss_cecl"],
            )
        return loss_vals


@dataclass
class MappoCeclConfig(MappoConfig):
    """Configuration dataclass for :class:`~algorithms.mappo_cecl.MappoCecl`."""

    lambda_cecl: float = 0.0
    cecl_emb_key: str = "cecl_emb"
    cecl_use_logit_scale: bool = False
    cecl_use_logit_bias: bool = False
    cecl_logit_scale_key: str = "logit_scale"
    cecl_logit_bias_key: str = "logit_bias"
    cecl_use_time_mask: bool = True
    cecl_start_time_key: str = "obs_start_time"
    cecl_end_time_key: str = "obs_end_time"
    minibatch_advantage: bool = False

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MappoCecl

