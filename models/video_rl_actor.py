from __future__ import annotations

from dataclasses import dataclass, MISSING
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig
from models.video_encoder import VideoEncoder


class VideoPolicyModel(Model):
    """
    Policy (actor) model for BenchMARL.

    Args:
        video_encoder_cfg (dict): config for VideoEncoder
        video_key (str): key for video observations
        aux_keys (list[str], optional): extra feature keys to concatenate
        head_hidden_sizes (Sequence[int]): MLP hidden sizes
        activation_class (Type[nn.Module]): activation class for the head
        dropout (float): dropout probability for the head
        return_temporal_features (bool): mean-pool temporal encoder outputs if True
    """

    def __init__(self, **kwargs):
        # Pull config needed by _perform_checks before base __init__ runs.
        video_encoder_cfg: Dict[str, Any] = kwargs.pop("video_encoder_cfg")
        if isinstance(video_encoder_cfg, dict):
            video_encoder_cfg = SimpleNamespace(**video_encoder_cfg)
        video_key: str = kwargs.pop("video_key")
        aux_keys: Optional[Sequence[str]] = kwargs.pop("aux_keys")
        head_hidden_sizes: Sequence[int] = kwargs.pop("head_hidden_sizes")
        activation_class: Type[nn.Module] = kwargs.pop("activation_class")
        dropout: float = kwargs.pop("dropout", 0.0)
        return_temporal_features: bool = kwargs.pop("return_temporal_features", False)
        projector_hidden_sizes: Sequence[int] = kwargs.pop("projector_hidden_sizes", ())
        projector_out_dim: Optional[int] = kwargs.pop("projector_out_dim", None)
        projector_key: str = kwargs.pop("projector_key", "proj_embedding")
        use_logit_scale: bool = kwargs.pop("use_logit_scale", False)
        use_logit_bias: bool = kwargs.pop("use_logit_bias", False)
        logit_scale_init: float = kwargs.pop("logit_scale_init", 1.0)
        logit_scale_key: str = kwargs.pop("logit_scale_key", "logit_scale")
        logit_bias_key: str = kwargs.pop("logit_bias_key", "logit_bias")

        # Stash attributes needed by _perform_checks before super().__init__.
        self.video_key = video_key
        self.aux_keys: List[str] = list(aux_keys) if aux_keys is not None else []
        self.return_temporal_features = return_temporal_features
        self.activation_class = activation_class
        self.projector_key = projector_key
        self.logit_scale_key = logit_scale_key
        self.logit_bias_key = logit_bias_key

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.projector_key = self._resolve_group_key(self.projector_key)
        self.logit_scale_key = self._resolve_group_key(self.logit_scale_key)
        self.logit_bias_key = self._resolve_group_key(self.logit_bias_key)

        # ---- Store config ----

        # ---- Build frozen video encoder (generic factory) ----
        # VideoEncoder selects the concrete backbone based on video_encoder_cfg["model_type"].
        self.encoder = VideoEncoder(video_encoder_cfg).to(self.device)

        # Freeze + eval mode for stability (and to avoid dropout etc. inside encoder)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Embedding dimension
        self.embed_dim = int(self.encoder.embed_dim)

        # ---- Determine output features expected by BenchMARL ----
        # BenchMARL sets output specs based on algorithm + env action space.
        self.output_features = int(self.output_leaf_spec.shape[-1])

        # ---- Determine aux feature size from input specs (if any) ----
        # NOTE: input_spec entries are per-key specs. For aux keys, we expect
        # the last dimension to be the feature dimension.
        aux_dim = 0
        for k in self.aux_keys:
            if k not in self.input_spec.keys(True, True):
                raise KeyError(
                    f"Aux key '{k}' not found in input_spec keys: {list(self.input_spec.keys(True, True))}"
                )
            aux_dim += int(self.input_spec[k].shape[-1])

        # ---- Build projection head (optional) ----
        proj_layers: List[nn.Module] = []
        proj_prev = self.embed_dim
        for hs in projector_hidden_sizes:
            proj_layers.append(nn.Linear(proj_prev, int(hs)))
            proj_layers.append(self.activation_class())
            proj_prev = int(hs)
        if projector_out_dim is not None:
            proj_layers.append(nn.Linear(proj_prev, int(projector_out_dim)))
            proj_prev = int(projector_out_dim)
        self.projector = nn.Sequential(*proj_layers).to(self.device) if proj_layers else None
        self.projector_out_dim = proj_prev

        # Optional learnable contrastive scalars
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init))) if use_logit_scale else None
        self.logit_bias = nn.Parameter(torch.zeros(())) if use_logit_bias else None

        # ---- Build trainable policy head ----
        # Head input = projected embedding + aux features
        head_in = self.projector_out_dim + aux_dim

        layers: List[nn.Module] = []
        prev = head_in
        for hs in head_hidden_sizes:
            layers.append(nn.Linear(prev, int(hs)))
            layers.append(self.activation_class())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(hs)

        # Final layer to policy outputs (logits or distribution params)
        layers.append(nn.Linear(prev, self.output_features))

        self.head = nn.Sequential(*layers).to(self.device)

    def _perform_checks(self):
        super()._perform_checks()

        # Actor policies in BenchMARL typically receive per-agent data,
        # so this should usually be True. We still support False for completeness.
        # This model is ALWAYS decentralized: it never aggregates across agents.
        if self.centralised:
            raise ValueError(
                "VideoPolicyModel is intended to be a decentralized actor. "
                "centralised=True would imply pooling across agents, which we do not do here."
            )

        # Ensure the video key exists (allow common nested-key layouts).
        input_keys = list(self.input_spec.keys(True, True))
        if self.video_key not in input_keys:
            candidates = []
            if isinstance(self.video_key, str):
                candidates.append(("observation", self.video_key))
                if getattr(self, "agent_group", None) is not None:
                    candidates.append((self.agent_group, "observation", self.video_key))
                    candidates.append((self.agent_group, self.video_key))
            for key in candidates:
                if key in input_keys:
                    self.video_key = key
                    break
            else:
                raise KeyError(
                    f"Video key '{self.video_key}' not found in input_spec keys: {input_keys}"
                )

    def _resolve_group_key(self, key: Any) -> Any:
        if isinstance(key, str) and getattr(self, "agent_group", None) is not None:
            return (self.agent_group, key)
        return key

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of videos -> embeddings.

        Expected shapes (common cases):
          - [B, T, C, H, W]  (no agent dim)
          - [B, n_agents, T, C, H, W] (agent dim present)
        We will flatten agent dimension into batch before calling the encoder.
        """
        # Ensure float (many envs output uint8 pixels)
        if video.dtype != torch.float32:
            video = video.float() / 255.0

        # Move to device
        video = video.to(self.device)

        # Resize frames if the encoder expects a fixed image size (e.g., 224x224).
        target_hw = self._get_target_hw()
        if target_hw is not None and tuple(video.shape[-2:]) != target_hw:
            B = int(video.numel() // (video.shape[-4] * video.shape[-3] * video.shape[-2] * video.shape[-1]))
            flat = video.reshape(B * video.shape[-4], video.shape[-3], video.shape[-2], video.shape[-1])
            flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
            video = flat.reshape(B, video.shape[-4], video.shape[-3], *target_hw)

        if self.input_has_agent_dim:
            # video: [*, n_agents, T, C, H, W]
            # Flatten: [(* * n_agents), T, C, H, W]
            n_agents = self.n_agents
            flat = video.reshape(-1, n_agents, *video.shape[-4:])  # [Bflat, n_agents, T, C, H, W]
            flat = flat.reshape(-1, *video.shape[-4:])            # [Bflat*n_agents, T, C, H, W]
        else:
            # video: [*, T, C, H, W]
            flat = video.reshape(-1, *video.shape[-4:])           # [Bflat, T, C, H, W]

        # Frozen encoder forward (no grad)
        with torch.no_grad():
            emb = self.encoder(flat, return_temporal_features=self.return_temporal_features)

        # If you ever set return_temporal_features=True, emb might be [B, Time, D].
        # For a PPO policy head we want a single vector; simplest is mean pool over time.
        if emb.dim() == 3:
            emb = emb.mean(dim=1)  # [B, D]

        # Unflatten back to [*, n_agents, D] if needed
        if self.input_has_agent_dim:
            emb = emb.reshape(-1, self.n_agents, self.embed_dim)  # [Bflat, n_agents, D]
        else:
            # [Bflat, D]
            pass

        return emb

    def _get_target_hw(self) -> Optional[tuple[int, int]]:
        encoder = getattr(self.encoder, "video_encoder", None)
        vision_model = getattr(encoder, "vision_model", None)
        config = getattr(vision_model, "config", None)
        size = getattr(config, "image_size", None)
        if isinstance(size, int):
            return (size, size)
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        return None

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        BenchMARL calls this and expects us to write tensordict[self.out_key] with:
          - shape including agent dim if self.output_has_agent_dim is True
          - shape without agent dim otherwise
        For a decentralized shared-parameter actor:
          input_has_agent_dim=True, centralised=False, share_params=True
          => output_has_agent_dim should be True (per-agent outputs)
        """
        # 1) Encode video -> per-agent embeddings
        video = tensordict.get(self.video_key)
        z = self._encode_video(video)  # [B, n_agents, D] if input_has_agent_dim else [B, D]

        # 1b) Project encoder embeddings -> h (trainable)
        if self.projector is not None:
            if self.input_has_agent_dim:
                z_flat = z.reshape(-1, z.shape[-1])
                h_flat = self.projector(z_flat)
                h = h_flat.reshape(-1, self.n_agents, h_flat.shape[-1])
            else:
                h = self.projector(z)
        else:
            h = z

        tensordict.set(self.projector_key, h)
        if self.logit_scale is not None:
            if isinstance(self.logit_scale_key, tuple):
                batch_shape = tensordict.get(self.logit_scale_key[0]).batch_size
            else:
                batch_shape = tensordict.batch_size
            tensordict.set(
                self.logit_scale_key,
                self.logit_scale.expand(*batch_shape),
            )
        if self.logit_bias is not None:
            if isinstance(self.logit_bias_key, tuple):
                batch_shape = tensordict.get(self.logit_bias_key[0]).batch_size
            else:
                batch_shape = tensordict.batch_size
            tensordict.set(
                self.logit_bias_key,
                self.logit_bias.expand(*batch_shape),
            )

        # 2) Gather aux features (optional) and concatenate
        if self.aux_keys:
            aux_list = [tensordict.get(k).to(self.device) for k in self.aux_keys]
            aux = torch.cat(aux_list, dim=-1)  # should align with agent dim presence

            # If input has agent dim, aux should be [B, n_agents, aux_dim]
            # else [B, aux_dim]
            x = torch.cat([h, aux], dim=-1)
        else:
            x = h

        # 3) Run trainable head
        if self.input_has_agent_dim:
            # Apply head per agent by flattening agent into batch:
            # x: [B, n_agents, Din] -> [B*n_agents, Din]
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = self.head(x_flat)  # [B*n_agents, Dout]
            y = y_flat.reshape(-1, self.n_agents, self.output_features)  # [B, n_agents, Dout]
        else:
            y = self.head(x)  # [B, Dout]

        # 4) Conform to BenchMARL expectation about agent dim in the output
        # For a normal decentralized policy, output_has_agent_dim should be True and we keep [B, n_agents, Dout].
        # If output_has_agent_dim is False (rare for policies), remove agent dim safely.
        if self.input_has_agent_dim and (not self.output_has_agent_dim):
            # Centralised + share_params would produce broadcasted output; not expected for policies,
            # but we keep compatibility.
            y = y[..., 0, :]

        tensordict.set(self.out_key, y)
        return tensordict


@dataclass
class VideoPolicyModelConfig(ModelConfig):
    """Dataclass config for a :class:`~models.video_rl_actor.VideoPolicyModel`."""

    video_encoder_cfg: Dict[str, Any] = MISSING
    video_key: str = MISSING

    head_hidden_sizes: Sequence[int] = MISSING
    activation_class: Type[nn.Module] = MISSING
    aux_keys: Optional[Sequence[str]] = None
    dropout: float = 0.0

    return_temporal_features: bool = False
    projector_hidden_sizes: Sequence[int] = ()
    projector_out_dim: Optional[int] = None
    projector_key: str = "proj_embedding"
    use_logit_scale: bool = False
    use_logit_bias: bool = False
    logit_scale_init: float = 1.0
    logit_scale_key: str = "logit_scale"
    logit_bias_key: str = "logit_bias"

    @staticmethod
    def associated_class():
        return VideoPolicyModel

    @property
    def is_rnn(self) -> bool:
        return False
