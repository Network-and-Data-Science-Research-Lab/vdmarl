from vdmarl.algorithms import QmixConfig, VdnConfig, QplexConfig, QtranConfig, QattenConfig, AvdnetConfig, TransmixConfig, WqmixConfig
from vdmarl.benchmark import Benchmark
from vdmarl.environments import VmasTask
from vdmarl.experiment import ExperimentConfig
from vdmarl.models.mlp import MlpConfig
import warnings
import os

os.environ["AUTO_UNWRAP_TRANSFORMED_ENV"] = "False"
warnings.filterwarnings("ignore", category=UserWarning, message=".*wasn't part of the annotations.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Action shape.*does not match expected shape.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The default behavior of TransformedEnv will change.*")

if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = 120_000

    experiment_config.render = True
    experiment_config.save_folder = "/home/jlcg/projects/vdmarl/scripts/runs"

    benchmark = Benchmark(
        algorithm_configs=[
            VdnConfig.get_from_yaml(),
            QmixConfig.get_from_yaml(),
            QplexConfig.get_from_yaml(),
            QtranConfig.get_from_yaml(),
            QattenConfig.get_from_yaml(),
            AvdnetConfig.get_from_yaml(),
            TransmixConfig.get_from_yaml(),
            WqmixConfig.get_from_yaml(),
        ],
        tasks=[
            VmasTask.BALANCE.get_from_yaml(),
        ],
        seeds={0},
        experiment_config=experiment_config,
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
    )

    benchmark.run_sequential()

