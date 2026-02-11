"""
MASAC + CECL (Cross-Ego Contrastive Learning) algorithm.

Mirrors the structure of ``mappo_cecl.py`` but wraps the SAC / DiscreteSAC
loss instead of ClipPPO.  The CECL contrastive term is added to the actor
loss (``loss_actor``) so that the actor encoder learns representations that
are aligned across agents.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import DiscreteSACLoss, LossModule, SACLoss, ValueEstimators

from benchmarl.algorithms.masac import Masac, MasacConfig
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

# ---------------------------------------------------------------------------
# Re-use the CECL helpers that are algorithm-agnostic
# ---------------------------------------------------------------------------
from algorithms.mappo_cecl import (
    _reshape_to_b_a_d,
    _create_label_matrix,
    CeclLossModule,
)


# =====================================================================
# Wrapped loss: SAC (or DiscreteSAC) + CECL
# =====================================================================

class CeclWrappedSacLoss(LossModule):
    """Wraps a ``SACLoss`` or ``DiscreteSACLoss`` and appends a CECL term."""

    def __init__(
        self,
        sac_loss: Union[SACLoss, DiscreteSACLoss],
        cecl_loss: CeclLossModule,
        group: str,
        emb_key: str,
        start_time_key: Optional[str],
        end_time_key: Optional[str],
        logit_scale_key: Optional[str],
        logit_bias_key: Optional[str],
    ):
        super().__init__()
        self.sac_loss = sac_loss
        self.cecl_loss = cecl_loss
        self.group = group
        self.emb_key = emb_key
        self.start_time_key = start_time_key
        self.end_time_key = end_time_key
        self.logit_scale_key = logit_scale_key
        self.logit_bias_key = logit_bias_key

    # ------------------------------------------------------------------
    # Delegate value_estimator to the inner SAC loss so BenchMARL's
    # generic loop can call  loss.value_estimator(...)  on this wrapper.
    # ------------------------------------------------------------------
    @property
    def value_estimator(self):
        return self.sac_loss.value_estimator

    @value_estimator.setter
    def value_estimator(self, value):
        self.sac_loss.value_estimator = value

    def __getattr__(self, name: str):
        """
        Proxy attribute look-ups to the inner SAC loss so that BenchMARL
        code expecting a plain SAC loss (e.g.  loss.actor_network_params,
        loss.qvalue_network_params, loss.log_alpha) still works.
        """
        sac_loss = self.__dict__.get("sac_loss", None)
        if sac_loss is not None and hasattr(sac_loss, name):
            return getattr(sac_loss, name)

        if name in (
            "actor_network_params",
            "target_actor_network_params",
            "qvalue_network_params",
            "target_qvalue_network_params",
        ):
            return None

        return super().__getattr__(name)

    # ------------------------------------------------------------------
    # Forward: run SAC loss, then compute and append CECL
    # ------------------------------------------------------------------
    def forward(self, batch: TensorDictBase) -> TensorDictBase:
        loss_td = self.sac_loss(batch)

        # If the actor hasn't already written its embedding (e.g. the SAC
        # loss internally re-samples actions), run a forward pass to get it.
        if (self.group, self.emb_key) not in batch.keys(True, True):
            _ = self.sac_loss.actor_network(batch)

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


# =====================================================================
# MasacCecl algorithm
# =====================================================================

class MasacCecl(Masac):
    """Multi-Agent SAC with Cross-Ego Contrastive Learning."""

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

    # ------------------------------------------------------------------
    # Override _get_policy_for_loss to add the CECL embedding spec
    # (identical to MappoCecl._get_policy_for_loss but uses Composite /
    # Unbounded from torchrl.data which is what MASAC expects).
    # ------------------------------------------------------------------
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

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        cecl_dim = getattr(model_config, "projector_out_dim", None)
        if cecl_dim is None:
            raise ValueError(
                "CECL requires the actor model config to set "
                "projector_out_dim for the embedding spec."
            )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
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
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
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
                    distribution_kwargs={"neg_inf": -18.0},
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        return policy

    # ------------------------------------------------------------------
    # Override _get_loss to wrap the SAC loss with CECL
    # ------------------------------------------------------------------
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Get the plain SAC loss (+ delay flag) from the parent
        sac_loss_module, has_delay = super()._get_loss(group, policy_for_loss, continuous)

        # Wrap it with CECL
        cecl_loss = CeclLossModule(n_agents=len(self.group_map[group]))
        wrapped = CeclWrappedSacLoss(
            sac_loss=sac_loss_module,
            cecl_loss=cecl_loss,
            group=group,
            emb_key=self.cecl_emb_key,
            start_time_key=self.cecl_start_time_key if self.cecl_use_time_mask else None,
            end_time_key=self.cecl_end_time_key if self.cecl_use_time_mask else None,
            logit_scale_key=self.cecl_logit_scale_key if self.cecl_use_logit_scale else None,
            logit_bias_key=self.cecl_logit_bias_key if self.cecl_use_logit_bias else None,
        )

        # Propagate the value estimator so BenchMARL sees it on the wrapper.
        wrapped.value_estimator = sac_loss_module.value_estimator

        return wrapped, has_delay

    # ------------------------------------------------------------------
    # Override get_loss_and_updater: pass the *inner* SAC loss to the
    # target-net updater so it can find the ``target_*_params`` children.
    # The wrapper itself only exposes ``sac_loss`` and ``cecl_loss`` as
    # named children, which SoftUpdate / HardUpdate cannot discover.
    # ------------------------------------------------------------------
    def get_loss_and_updater(self, group: str):
        if group not in self._losses_and_updaters:
            from torchrl.data import Categorical, OneHot
            from torchrl.objectives.utils import HardUpdate, SoftUpdate

            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(action_space, (Categorical, OneHot))
            loss, use_target = self._get_loss(
                group=group,
                policy_for_loss=self.get_policy_for_loss(group),
                continuous=continuous,
            )
            if use_target:
                # Use the inner SAC loss so SoftUpdate sees target_*_params
                updater_loss = (
                    loss.sac_loss if isinstance(loss, CeclWrappedSacLoss) else loss
                )
                if self.experiment_config.soft_target_update:
                    target_net_updater = SoftUpdate(
                        updater_loss, tau=self.experiment_config.polyak_tau
                    )
                else:
                    target_net_updater = HardUpdate(
                        updater_loss,
                        value_network_update_interval=self.experiment_config.hard_target_update_frequency,
                    )
            else:
                target_net_updater = None
            self._losses_and_updaters[group] = (loss, target_net_updater)
        return self._losses_and_updaters[group]

    # ------------------------------------------------------------------
    # Override _get_parameters: delegate to base Masac using the inner
    # SAC loss (so the optimizer groups stay the same).
    # ------------------------------------------------------------------
    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        if isinstance(loss, CeclWrappedSacLoss):
            return super()._get_parameters(group, loss.sac_loss)
        return super()._get_parameters(group, loss)

    # ------------------------------------------------------------------
    # process_loss_vals: fold λ * CECL into the actor loss
    # ------------------------------------------------------------------
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals = super().process_loss_vals(group, loss_vals)

        if "loss_cecl" in loss_vals.keys():
            loss_vals.set(
                "loss_actor",
                loss_vals["loss_actor"] + self.lambda_cecl * loss_vals["loss_cecl"],
            )
        return loss_vals


# =====================================================================
# Config dataclass
# =====================================================================

@dataclass
class MasacCeclConfig(MasacConfig):
    """Configuration dataclass for :class:`~algorithms.masac_cecl.MasacCecl`."""

    lambda_cecl: float = 0.0
    cecl_emb_key: str = "cecl_emb"
    cecl_use_logit_scale: bool = False
    cecl_use_logit_bias: bool = False
    cecl_logit_scale_key: str = "logit_scale"
    cecl_logit_bias_key: str = "logit_bias"
    cecl_use_time_mask: bool = True
    cecl_start_time_key: str = "obs_start_time"
    cecl_end_time_key: str = "obs_end_time"

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MasacCecl

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
