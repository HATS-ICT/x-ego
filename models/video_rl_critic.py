#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

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


class VideoCentralCriticModel(Model):
    """
    Centralized critic for MAPPO in BenchMARL.

    Args:
        video_encoder_cfg (dict): config for VideoEncoder
        video_key (str): key for per-agent video observations
        aux_keys (list[str], optional): extra feature keys to concatenate
        fusion (str): "concat" or "mean"
        head_hidden_sizes (Sequence[int]): MLP hidden sizes
        activation_class (Type[nn.Module]): activation class for the head
        dropout (float): dropout probability for the head
        return_temporal_features (bool): mean-pool temporal encoder outputs if True
        layer_norm (bool): apply LayerNorm before the head
    """

    def __init__(self, **kwargs):
        # Pull config needed by _perform_checks before base __init__ runs.
        video_encoder_cfg: Dict[str, Any] = kwargs.pop("video_encoder_cfg")
        if isinstance(video_encoder_cfg, dict):
            video_encoder_cfg = SimpleNamespace(**video_encoder_cfg)
        video_key: str = kwargs.pop("video_key")
        aux_keys: Optional[Sequence[str]] = kwargs.pop("aux_keys")
        fusion: str = kwargs.pop("fusion")
        head_hidden_sizes: Sequence[int] = kwargs.pop("head_hidden_sizes")
        activation_class: Type[nn.Module] = kwargs.pop("activation_class")
        dropout: float = kwargs.pop("dropout", 0.0)
        return_temporal_features: bool = kwargs.pop("return_temporal_features", False)
        layer_norm: bool = kwargs.pop("layer_norm", False)

        # Stash attributes needed by _perform_checks before super().__init__.
        self.video_key = video_key
        self.aux_keys: List[str] = list(aux_keys) if aux_keys is not None else []
        self.fusion = fusion
        self.return_temporal_features = return_temporal_features
        self.activation_class = activation_class
        self.layer_norm = layer_norm

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

        # ---- Config ----

        # ---- Frozen encoder (critic-side copy, generic factory) ----
        # VideoEncoder selects the concrete backbone based on video_encoder_cfg["model_type"].
        self.encoder = VideoEncoder(video_encoder_cfg).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.embed_dim = int(self.encoder.embed_dim)

        # ---- Value output dim ----
        # BenchMARL sets this from output spec (for critics, usually 1).
        self.value_dim = int(self.output_leaf_spec.shape[-1])

        # ---- Aux feature dim (optional extra low-dim inputs per agent) ----
        aux_dim = 0
        for k in self.aux_keys:
            if k not in self.input_spec.keys(True, True):
                raise KeyError(
                    f"Aux key '{k}' not found in input_spec keys: {list(self.input_spec.keys(True, True))}"
                )
            aux_dim += int(self.input_spec[k].shape[-1])

        # ---- Build the shared value head ----
        # Input to value head will be: [z_i ; g ; aux_i]
        # - z_i: D
        # - g: either N*D (concat) or D (mean)
        # - aux_i: aux_dim
        if fusion == "concat":
            global_dim = self.n_agents * self.embed_dim
        elif fusion == "mean":
            global_dim = self.embed_dim
        else:
            raise ValueError("fusion must be one of {'concat', 'mean'}")

        head_in = self.embed_dim + global_dim + aux_dim

        layers: List[nn.Module] = []
        prev = head_in

        if self.layer_norm:
            layers.append(nn.LayerNorm(prev))

        for hs in head_hidden_sizes:
            layers.append(nn.Linear(prev, int(hs)))
            layers.append(self.activation_class())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(hs)

        layers.append(nn.Linear(prev, self.value_dim))
        self.value_head = nn.Sequential(*layers).to(self.device)

    def _perform_checks(self):
        super()._perform_checks()

        # This critic is intended to be centralized over local per-agent inputs.
        # That means: input_has_agent_dim should be True and centralised should be True.
        # BenchMARL will set these flags depending on algorithm usage.
        if not self.input_has_agent_dim:
            raise ValueError(
                "VideoCentralCriticModel expects per-agent inputs (input_has_agent_dim=True). "
                "If you want a global-state critic, use a different model that reads a global 'state' key."
            )
        if not self.centralised:
            raise ValueError(
                "VideoCentralCriticModel is a centralized critic (centralised=True). "
                "If you want a decentralized critic, use a different model."
            )
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

    def _encode_videos_per_agent(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode per-agent videos into embeddings z.

        Expected input (typical critic case):
          video: [B, n_agents, T, C, H, W] (plus possibly extra leading batch dims)

        Returns:
          z: [Bflat, n_agents, D]
        """
        # Normalize pixels if needed
        if video.dtype != torch.float32:
            video = video.float() / 255.0
        video = video.to(self.device)

        # Resize frames if the encoder expects a fixed image size (e.g., 224x224).
        target_hw = self._get_target_hw()
        if target_hw is not None and tuple(video.shape[-2:]) != target_hw:
            flat = video.reshape(-1, video.shape[-3], video.shape[-2], video.shape[-1])
            flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
            video = flat.reshape(*video.shape[:-2], *target_hw)

        # Flatten leading dims except agent/time/channel/height/width.
        # Accept either:
        #  - [..., n_agents, T, C, H, W]  (agent at -6)
        #  - [..., n_agents, T, C, H, W]  (agent at -5 when there is one fewer leading dim)
        n_agents = self.n_agents
        if video.shape[-6] == n_agents:
            leading = video.shape[:-6]
            tail = video.shape[-5:]  # (T, C, H, W)
        elif video.shape[-5] == n_agents:
            leading = video.shape[:-5]
            tail = video.shape[-4:]  # (T, C, H, W)
        else:
            raise ValueError(
                f"Expected agent dim at -6 or -5 with size n_agents={n_agents}, got video.shape={tuple(video.shape)}"
            )

        # Collapse leading batch dims into one for encoding.
        Bflat = int(torch.tensor(leading).prod().item()) if len(leading) > 0 else 1

        # Reshape to [Bflat, n_agents, T, C, H, W]
        v = video.reshape(Bflat, n_agents, *tail)

        # Flatten agents into batch for a single encoder call:
        # [Bflat*n_agents, T, C, H, W]
        v_flat = v.reshape(Bflat * n_agents, *tail)

        with torch.no_grad():
            emb = self.encoder(v_flat, return_temporal_features=self.return_temporal_features)

        # If encoder returns temporal features [B, Time, D], pool time -> [B, D]
        if emb.dim() == 3:
            emb = emb.mean(dim=1)

        # Unflatten back to [Bflat, n_agents, D]
        z = emb.reshape(Bflat, n_agents, self.embed_dim)
        return z

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

    def _fuse_global(self, z: torch.Tensor) -> torch.Tensor:
        """
        Build global context g from per-agent embeddings z.

        z: [Bflat, n_agents, D]
        returns g:
          - concat: [Bflat, n_agents*D]
          - mean:   [Bflat, D]
        """
        if self.fusion == "concat":
            return z.reshape(z.shape[0], -1)
        # mean pooling
        return z.mean(dim=-2)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Outputs values into tensordict[self.out_key].

        For centralized critics with local inputs, BenchMARL typically expects
        an output with agent dimension (per-agent values). This is exactly what we produce.

        Output shape:
          values: [Bflat, n_agents, value_dim]
        then reshaped back to match the leading batch dims in the tensordict.
        """
        video = tensordict.get(self.video_key)
        z = self._encode_videos_per_agent(video)  # [Bflat, n_agents, D]
        g = self._fuse_global(z)                  # [Bflat, G]

        # Aux features per agent (optional): expected shape [..., n_agents, aux_dim]
        aux = None
        if self.aux_keys:
            aux_list = [tensordict.get(k).to(self.device) for k in self.aux_keys]
            aux_cat = torch.cat(aux_list, dim=-1)

            # Collapse leading dims to align with Bflat
            # expected aux_cat: [..., n_agents, aux_dim]
            n_agents = self.n_agents
            if aux_cat.shape[-2] != n_agents:
                raise ValueError(
                    f"Expected aux agent dim at -2 with size n_agents={n_agents}, got aux.shape={tuple(aux_cat.shape)}"
                )
            leading = aux_cat.shape[:-2]
            Bflat = int(torch.tensor(leading).prod().item()) if len(leading) > 0 else 1
            aux = aux_cat.reshape(Bflat, n_agents, aux_cat.shape[-1])  # [Bflat, n_agents, aux_dim]

        # Build per-agent head inputs: [z_i ; g ; aux_i]
        # g is global, so broadcast it to all agents:
        if g.dim() == 2:
            g_b = g.unsqueeze(-2).expand(z.shape[0], self.n_agents, g.shape[-1])  # [Bflat, n_agents, G]
        else:
            raise RuntimeError("Global context g must be 2D [Bflat, G].")

        if aux is not None:
            x = torch.cat([z, g_b, aux], dim=-1)  # [Bflat, n_agents, D+G+aux]
        else:
            x = torch.cat([z, g_b], dim=-1)       # [Bflat, n_agents, D+G]

        # Apply shared value head per agent by flattening agent dim into batch
        x_flat = x.reshape(-1, x.shape[-1])                 # [Bflat*n_agents, Din]
        v_flat = self.value_head(x_flat)                    # [Bflat*n_agents, value_dim]
        values = v_flat.reshape(z.shape[0], self.n_agents, self.value_dim)  # [Bflat, n_agents, value_dim]

        # Reshape to match tensordict.batch_size.
        # In TorchRL / BenchMARL, the agent dimension is often part of the tensordict batch.
        td_batch = tuple(int(d) for d in tensordict.batch_size)
        n_agents = int(self.n_agents)
        value_dim = int(self.value_dim)

        if len(td_batch) > 0 and td_batch[-1] == n_agents:
            # Agent dim is already a batch dim: output should be (*td_batch, value_dim)
            batch_wo_agent = td_batch[:-1]
            out = values.reshape(*batch_wo_agent, n_agents, value_dim)
        elif len(td_batch) > 0:
            # Agent dim is NOT in batch: output is (*td_batch, n_agents, value_dim)
            out = values.reshape(*td_batch, n_agents, value_dim)
        else:
            out = values

        # Handle output_has_agent_dim compatibility:
        # Normally for a centralized critic with local inputs, output_has_agent_dim is True.
        if self.input_has_agent_dim and (not self.output_has_agent_dim):
            # Some BenchMARL/TensorDict pipelines still keep the agent dimension in the tensordict batch
            # even when output_has_agent_dim=False. Collapsing here would cause a batch mismatch on set().
            # Only collapse if the tensordict batch does NOT include the agent dimension.
            if len(td_batch) == 0 or td_batch[-1] != n_agents:
                # Spec truly expects no agent dim: collapse agent dimension (e.g., mean).
                out = out.mean(dim=-2)

        tensordict.set(self.out_key, out)
        return tensordict


@dataclass
class VideoCentralCriticModelConfig(ModelConfig):
    """Dataclass config for a :class:`~models.video_rl_critic.VideoCentralCriticModel`."""

    video_encoder_cfg: Dict[str, Any] = MISSING
    video_key: str = MISSING

    fusion: str = MISSING

    head_hidden_sizes: Sequence[int] = MISSING
    activation_class: Type[nn.Module] = MISSING
    aux_keys: Optional[Sequence[str]] = None
    dropout: float = 0.0
    layer_norm: bool = False

    return_temporal_features: bool = False

    @staticmethod
    def associated_class():
        return VideoCentralCriticModel

    @property
    def is_rnn(self) -> bool:
        return False
