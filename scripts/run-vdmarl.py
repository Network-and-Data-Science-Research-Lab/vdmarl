from vdmarl.algorithms import QmixConfig, VdnConfig, QplexConfig, QtranConfig, QattenConfig, AvdnetConfig, TransmixConfig, WqmixConfig, QmixGnnConfig
from vdmarl.benchmark import Benchmark
from vdmarl.environments import Smacv2Task
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
    experiment_config.create_json = True
    
    # Standard SMAC/SMACv2 evaluation settings from literature (e.g. PyMARL, EPyMARL)
    # Most papers run for 2M to 10M frames. Using 2M as a standard baseline.
    experiment_config.max_n_frames = 2_000_000
    # Evaluate every 10,000 frames
    experiment_config.evaluation_interval = 10_000
    # Common to evaluate over 32 test episodes
    experiment_config.evaluation_episodes = 32
    
    # Render is usually disabled for high-throughput benchmarks
    experiment_config.render = False
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
            QmixGnnConfig.get_from_yaml(),
        ],
        tasks=[
            Smacv2Task.PROTOSS_5_VS_5.get_from_yaml(),
            Smacv2Task.TERRAN_5_VS_5.get_from_yaml(),
            Smacv2Task.ZERG_5_VS_5.get_from_yaml(),
        ],
        # Standardised Performance Evaluation Protocol for Cooperative MARL (marl-eval)
        # recommends N=10 independent training runs/seeds for robust confidence intervals
        seeds={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        experiment_config=experiment_config,
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
    )

    benchmark.run_sequential()
