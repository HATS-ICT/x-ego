"""
Entry-point for running a MASAC + CECL BenchMARL experiment.

Mirrors ``benchmarl_cecl_experiment.py`` (MAPPO variant) but uses the
off-policy MASAC algorithm.  Key differences from the MAPPO CECL experiment:

* No GAE / advantage precomputation (SAC does not use advantages).
* The custom ``_optimizer_loop`` still retains the computation graph across
  the multiple backward passes (actor, qvalue, alpha, with CECL folded into
  the actor loss) to avoid "trying to backward through the graph a second
  time" errors when the actor and critic share an encoder.
* The MASAC discrete-critic device-mismatch patch from
  ``benchmarl_experiment.py`` is applied automatically.
"""
from pathlib import Path
from copy import deepcopy
import argparse

import torch
from tensordict import TensorDictBase

import benchmarl.algorithms.masac as masac_module
from benchmarl.experiment import Experiment, ExperimentConfig
from scripts.doom_wrappers.DoomPettingzoo import DoomPettingZooTask

from algorithms.masac_cecl import MasacCeclConfig, CeclWrappedSacLoss
from models.video_rl_actor import VideoPolicyModelConfig
from models.video_encoder import MODEL_TYPE_TO_PRETRAINED
from models.video_rl_critic import VideoCentralCriticModelConfig


# ------------------------------------------------------------------
# Device-mismatch patch (same as benchmarl_experiment.py)
# ------------------------------------------------------------------
def _patch_masac_spec_device_mismatch() -> None:
    """Patch MASAC discrete critic spec to keep nested specs on the algorithm device."""
    if getattr(masac_module.Masac, "_xego_spec_device_patch", False):
        return

    def _patched_get_discrete_value_module_coupled(self, group: str):
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        critic_output_spec = masac_module.Composite(
            {
                "action_value": masac_module.Unbounded(
                    shape=(n_actions * n_agents,),
                    device=self.device,
                )
            },
            device=self.device,
        )

        if self.state_spec is not None:
            critic_input_spec = self.state_spec
            input_has_agent_dim = False
        else:
            critic_input_spec = masac_module.Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )
            input_has_agent_dim = True

        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=True,
            device=self.device,
            action_spec=self.action_spec,
        )

        expand_module = masac_module.TensorDictModule(
            lambda value: value.reshape(*value.shape[:-1], n_agents, n_actions),
            in_keys=["action_value"],
            out_keys=[(group, "action_value")],
        )
        return masac_module.TensorDictSequential(value_module, expand_module)

    masac_module.Masac.get_discrete_value_module_coupled = (
        _patched_get_discrete_value_module_coupled
    )
    masac_module.Masac._xego_spec_device_patch = True


# ------------------------------------------------------------------
# Custom Experiment with retained-graph backward for shared encoders
# ------------------------------------------------------------------
class MasacCeclExperiment(Experiment):
    """
    Custom Experiment variant for MASAC+CECL.

    The only customisation over the stock ``Experiment`` is the
    ``_optimizer_loop``: when multiple loss terms (actor, qvalue, alpha)
    share computation graphs (e.g. because actor and critic share a video
    encoder), we must retain the graph until the last backward call.
    """

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        subdata = self.replay_buffers[group].sample().to(self.config.train_device)

        # Forward loss (SAC + CECL)
        loss_vals = self.losses[group](subdata)

        # Detached copy for logging
        training_td = loss_vals.detach()

        # Let the algorithm post-process (folds λ*CECL into loss_actor)
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        # Collect all trainable losses and backward with retained graph
        trainable_items = [
            (name, value)
            for name, value in loss_vals.items()
            if name in self.optimizers[group].keys()
        ]

        for idx, (loss_name, loss_value) in enumerate(trainable_items):
            optimizer = self.optimizers[group][loss_name]
            retain = idx < len(trainable_items) - 1
            loss_value.backward(retain_graph=retain)

            grad_norm = self._grad_clip(optimizer)
            training_td.set(
                f"grad_norm_{loss_name}",
                torch.tensor(grad_norm, device=self.config.train_device),
            )

            optimizer.step()
            optimizer.zero_grad()

        # Priority update + target-net update
        self.replay_buffers[group].update_tensordict_priority(subdata)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()

        callback_loss = self._on_train_step(subdata, group)
        if callback_loss is not None:
            training_td.update(callback_loss)

        return training_td


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    _patch_masac_spec_device_mismatch()

    parser = argparse.ArgumentParser(description="Run BenchMARL MASAC+CECL experiment.")
    parser.add_argument(
        "--restore-file",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--restore-map-location",
        type=str,
        default=None,
        help='torch.load map_location, e.g. \'{"cuda:0":"cpu"}\'',
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Use the CECL-specific task config (different base_port for parallelism)
    task_cfg = repo_root / "configs" / "benchmarl" / "task" / "doom_pettingzoo_cecl.yaml"
    actor_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_actor_contrastive.yaml"
    critic_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_critic.yaml"
    exp_cfg = repo_root / "configs" / "benchmarl" / "experiment" / "small_gpu.yaml"
    algo_cfg_path = repo_root / "configs" / "benchmarl" / "algorithm" / "masac_cecl.yaml"

    # Base model configs
    base_actor_cfg = VideoPolicyModelConfig.get_from_yaml(str(actor_cfg))
    base_critic_cfg = VideoCentralCriticModelConfig.get_from_yaml(str(critic_cfg))

    # Build per-backbone actor / critic configs
    # video_model_names = list(MODEL_TYPE_TO_PRETRAINED.keys())
    video_model_names = ["cnn"]
    actor_cfgs = []
    critic_cfgs = []
    for model_name in video_model_names:
        cfg = deepcopy(base_actor_cfg)
        encoder_cfg = dict(cfg.video_encoder_cfg)
        encoder_cfg["model_type"] = model_name
        cfg.video_encoder_cfg = encoder_cfg
        actor_cfgs.append(cfg)

        critic_cfg_i = deepcopy(base_critic_cfg)
        critic_encoder_cfg = dict(critic_cfg_i.video_encoder_cfg)
        critic_encoder_cfg["model_type"] = model_name
        critic_cfg_i.video_encoder_cfg = critic_encoder_cfg
        critic_cfgs.append(critic_cfg_i)

    task = DoomPettingZooTask.DOOM_PZ.get_from_yaml(str(task_cfg))
    algo_cfg = MasacCeclConfig.get_from_yaml(str(algo_cfg_path))
    exp_config = ExperimentConfig.get_from_yaml(str(exp_cfg))
    if args.restore_file:
        exp_config.restore_file = args.restore_file
    if args.restore_map_location:
        exp_config.restore_map_location = args.restore_map_location

    for i, (actor_cfg_i, critic_cfg_i) in enumerate(zip(actor_cfgs, critic_cfgs)):
        print(
            f"\nRunning MASAC+CECL experiment {i+1}/{len(actor_cfgs)} "
            f"(actor={actor_cfg_i.video_encoder_cfg['model_type']}, "
            f"critic={critic_cfg_i.video_encoder_cfg['model_type']}).\n"
        )
        experiment = MasacCeclExperiment(
            task=task,
            algorithm_config=algo_cfg,
            model_config=actor_cfg_i,
            critic_model_config=critic_cfg_i,
            config=exp_config,
            seed=0,
        )
        experiment.run()


if __name__ == "__main__":
    main()
