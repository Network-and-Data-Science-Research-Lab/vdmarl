from vdmarl.algorithms import QmixConfig
from vdmarl.environments import VmasTask
from vdmarl.experiment import Experiment, ExperimentConfig
from vdmarl.models.mlp import MlpConfig


experiment_config = ExperimentConfig.get_from_yaml()
task = VmasTask.BALANCE.get_from_yaml()
algorithm_config = QmixConfig.get_from_yaml()
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

experiment_config.max_n_frames = 12_000
experiment_config.loggers = []



experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)