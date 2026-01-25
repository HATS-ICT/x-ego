from pathlib import Path

from benchmarl.algorithms import MappoConfig
from scripts.doom_wrappers.DoomPettingzoo import DoomPettingZooTask
# from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from models.video_rl_actor import VideoPolicyModelConfig
from models.video_rl_critic import VideoCentralCriticModelConfig
# from benchmarl.models.mlp import MlpConfig

def main() -> None:
   repo_root = Path(__file__).resolve().parents[1]
   task_cfg = repo_root / "configs" / "benchmarl" / "task" / "doom_pettingzoo.yaml"
   actor_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_actor.yaml"
   critic_cfg = repo_root / "configs" / "benchmarl" / "model" / "video_critic.yaml"
   exp_cfg = repo_root / "configs" / "benchmarl" / "experiment" / "fast_gpu.yaml"
   experiment = Experiment(
      task=DoomPettingZooTask.DOOM_PZ.get_from_yaml(str(task_cfg)),
      algorithm_config=MappoConfig.get_from_yaml(),
      model_config=VideoPolicyModelConfig.get_from_yaml(str(actor_cfg)),
      critic_model_config=VideoCentralCriticModelConfig.get_from_yaml(str(critic_cfg)),
      seed=0,
      config=ExperimentConfig.get_from_yaml(str(exp_cfg)),
   )
   experiment.run()


if __name__ == "__main__":
   main()