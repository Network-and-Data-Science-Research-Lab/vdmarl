import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vdmarl.hydra_config import load_experiment_from_hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    """Runs an experiment loading its config from hydra.

    This function is decorated as ``@hydra.main`` and is called by running

    .. code-block:: console

       python vdmarl/run.py algorithm=qmix task=vmas/balance


    Args:
        cfg (DictConfig): the hydra config dictionary

    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
