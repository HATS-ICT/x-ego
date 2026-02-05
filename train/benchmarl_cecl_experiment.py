from pathlib import Path
from copy import deepcopy

from benchmarl.experiment import Experiment, ExperimentConfig
from scripts.doom_wrappers.DoomPettingzoo import DoomPettingZooTask

from algorithms.mappo_cecl import MappoCeclConfig, CeclWrappedLoss
from models.video_rl_actor import VideoPolicyModelConfig
from models.video_encoder import MODEL_TYPE_TO_PRETRAINED
from models.video_rl_critic import VideoCentralCriticModelConfig

import torch
from tensordict import TensorDictBase


class MappoCeclExperiment(Experiment):
   """
   Custom Experiment variant for MAPPO+CECL that:
   - precomputes advantages / value targets with GAE under no_grad on each
     optimizer minibatch, and
   - safely handles multiple loss terms (actor + critic) that may share
     computation graphs by retaining the graph until the last backward.
   """

   def _optimizer_loop(self, group: str) -> TensorDictBase:
      # Sample batch and move to train device
      subdata = self.replay_buffers[group].sample().to(self.config.train_device)

      # ------------------------------------------------------------------
      # 1) Ensure advantages / value targets are present WITHOUT grad
      # ------------------------------------------------------------------
      # TorchRL's PPO loss will (by default) compute GAE internally if the
      # advantage key is missing. That GAE implementation uses in-place
      # slice updates and will be part of the autograd graph if called
      # under grad, which is exactly what triggers the anomaly warning on
      # `CopySlices` and contributes to the double-backward error.
      #
      # To avoid this, we proactively run the value_estimator here under
      # `torch.no_grad()` so that:
      #   - advantages / value_targets are already written into `subdata`
      #   - the PPO loss sees them and *skips* its own value_estimator call
      #
      # This mirrors what `Mappo.process_batch` does, but we do it on the
      # actual minibatch sampled from the replay buffer.
      loss_module = self.losses[group]
      base_loss = loss_module.ppo_loss if isinstance(loss_module, CeclWrappedLoss) else loss_module
      with torch.no_grad():
         try:
            base_loss.value_estimator(
               subdata,
               params=base_loss.critic_network_params,
               target_params=base_loss.target_critic_network_params,
            )
         except AttributeError:
            # If for some reason the loss does not expose functional params,
            # just call the estimator without them.
            base_loss.value_estimator(subdata)

      # ------------------------------------------------------------------
      # 2) Forward loss (PPO + CECL)
      # ------------------------------------------------------------------
      loss_vals = self.losses[group](subdata)

      # Keep a detached copy for logging
      training_td = loss_vals.detach()

      # Let the algorithm post-process the losses (e.g. MAPPO folding entropy
      # and adding CECL into 'loss_objective').
      loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

      # Collect all trainable losses for this group so we can handle multiple
      # backwards safely on a shared graph by retaining it until the last loss.
      trainable_items = [
         (name, value)
         for name, value in loss_vals.items()
         if name in self.optimizers[group].keys()
      ]

      for idx, (loss_name, loss_value) in enumerate(trainable_items):
         optimizer = self.optimizers[group][loss_name]

         # Retain the graph for all but the last loss to avoid the
         # "Trying to backward through the graph a second time" error
         # when multiple losses share computation.
         retain = idx < len(trainable_items) - 1
         loss_value.backward(retain_graph=retain)

         grad_norm = self._grad_clip(optimizer)
         training_td.set(
            f"grad_norm_{loss_name}",
            torch.tensor(grad_norm, device=self.config.train_device),
         )

         optimizer.step()
         optimizer.zero_grad()

      # Priority update + target-net update as usual
      self.replay_buffers[group].update_tensordict_priority(subdata)
      if self.target_updaters[group] is not None:
         self.target_updaters[group].step()

      callback_loss = self._on_train_step(subdata, group)
      if callback_loss is not None:
         training_td.update(callback_loss)

      return training_td


def main() -> None:
   repo_root = Path(__file__).resolve().parents[1]
   # Use a task config with a different base_port so CECL experiments can
   # run in parallel with other DoomPettingZoo experiments without
   # competing for the same VizDoom network ports.
   task_cfg = repo_root / "configs" / "benchmarl" / "task" / "doom_pettingzoo_cecl.yaml"
   actor_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_actor_contrastive.yaml"
   critic_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_critic.yaml"
   exp_cfg = repo_root / "configs" / "benchmarl" / "experiment" / "small_gpu.yaml"
   algo_cfg = repo_root / "configs" / "benchmarl" / "algorithm" / "mappo_cecl.yaml"

   # Base configs
   base_actor_cfg = VideoPolicyModelConfig.get_from_yaml(str(actor_cfg))
   base_critic_cfg = VideoCentralCriticModelConfig.get_from_yaml(str(critic_cfg))

   # Take all available video backbone names and build independent actor / critic configs
   video_model_names = list(MODEL_TYPE_TO_PRETRAINED.keys())
   actor_cfgs = []
   critic_cfgs = []
   for model_name in video_model_names:
      cfg = deepcopy(base_actor_cfg)
      # Ensure we have an independent encoder cfg per actor
      encoder_cfg = dict(cfg.video_encoder_cfg)
      encoder_cfg["model_type"] = model_name
      cfg.video_encoder_cfg = encoder_cfg
      actor_cfgs.append(cfg)

      critic_cfg_i = deepcopy(base_critic_cfg)
      critic_encoder_cfg = dict(critic_cfg_i.video_encoder_cfg)
      critic_encoder_cfg["model_type"] = model_name
      critic_cfg_i.video_encoder_cfg = critic_encoder_cfg
      critic_cfgs.append(critic_cfg_i)

   # Mirror Benchmark.get_experiments: loop over experiments explicitly,
   # but with paired actor / critic configs (no cross-product between them).
   task = DoomPettingZooTask.DOOM_PZ.get_from_yaml(str(task_cfg))
   algo_cfg = MappoCeclConfig.get_from_yaml(str(algo_cfg))
   exp_config = ExperimentConfig.get_from_yaml(str(exp_cfg))
   # exp_config = ExperimentConfig.get_from_yaml()
   for i, (actor_cfg_i, critic_cfg_i) in enumerate(zip(actor_cfgs, critic_cfgs)):
      print(
         f"\nRunning experiments for pair {i+1}/{len(actor_cfgs)} "
         f"(actor={actor_cfg_i.video_encoder_cfg['model_type']}, "
         f"critic={critic_cfg_i.video_encoder_cfg['model_type']}).\n"
      )
      # Here we only have one algorithm and one task; if you add more later,
      # you can nest loops over algorithms / tasks / seeds like Benchmark does.
      experiment = MappoCeclExperiment(
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
