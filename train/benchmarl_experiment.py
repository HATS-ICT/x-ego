from pathlib import Path
from copy import deepcopy
import argparse

from benchmarl.algorithms import MappoConfig
from scripts.doom_wrappers.DoomPettingzoo import DoomPettingZooTask
# from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from models.video_rl_actor import VideoPolicyModelConfig
from models.video_encoder import MODEL_TYPE_TO_PRETRAINED
from models.video_rl_critic import VideoCentralCriticModelConfig
# from benchmarl.models.mlp import MlpConfig


def main() -> None:
   parser = argparse.ArgumentParser(description="Run BenchMARL experiment.")
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
   task_cfg = repo_root / "configs" / "benchmarl" / "task" / "doom_pettingzoo.yaml"
   actor_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_actor.yaml"
   critic_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_critic.yaml"
   exp_cfg = repo_root / "configs" / "benchmarl" / "experiment" / "small_gpu.yaml"

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
   algo_cfg = MappoConfig.get_from_yaml()
   exp_config = ExperimentConfig.get_from_yaml(str(exp_cfg))
   if args.restore_file:
      exp_config.restore_file = args.restore_file
   if args.restore_map_location:
      exp_config.restore_map_location = args.restore_map_location

   for i, (actor_cfg_i, critic_cfg_i) in enumerate(zip(actor_cfgs, critic_cfgs)):
      print(
         f"\nRunning experiments for pair {i+1}/{len(actor_cfgs)} "
         f"(actor={actor_cfg_i.video_encoder_cfg['model_type']}, "
         f"critic={critic_cfg_i.video_encoder_cfg['model_type']}).\n"
      )

      # Here we only have one algorithm and one task; if you add more later,
      # you can nest loops over algorithms / tasks / seeds like Benchmark does.
      experiment = Experiment(
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