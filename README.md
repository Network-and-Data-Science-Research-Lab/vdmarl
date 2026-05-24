# VDMARL
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

VDMARL is a value-decomposition-focused MARL experimentation library based on BenchMARL.

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| **AVDNet** | Attention Value-Decomposition Network |
| **QAtten** | Q-Attention |
| **QMIX** | Monotonic Value Function Factorisation |
| **QMIX-GNN** | QMIX with Graph Neural Networks |
| **QPLEX** | Duplex Dueling Multi-Agent Q-Learning |
| **QTRAN** | Learning to Factorize with Transformation for Cooperative MARL |
| **TransMix** | Transformer-based Value Decomposition |
| **VDN** | Value-Decomposition Networks |
| **WQMIX** | Weighted QMIX |

## Supported Environments

| Environment | Description |
|-------------|-------------|
| **VMAS** | Vectorized Multi-Agent Simulator |
| **SMACv2** | StarCraft Multi-Agent Challenge v2 |
| **PettingZoo** | PettingZoo Environments |
| **MeltingPot** | DeepMind Melting Pot |
| **MAgent** | MAgent Many-Agent Reinforcement Learning |

## Usage

### Command Line (Hydra)
You can run an experiment directly from the command line using Hydra configurations:
```bash
python src/vdmarl/run.py algorithm=qmix task=vmas/balance
```

### Python Script
You can also run experiments directly using the Python API. For example, using the `Benchmark` module to run multiple algorithms on multiple tasks sequentially:
```python
from vdmarl.algorithms import QmixConfig, VdnConfig
from vdmarl.benchmark import Benchmark
from vdmarl.environments import VmasTask
from vdmarl.experiment import ExperimentConfig
from vdmarl.models.mlp import MlpConfig

# Configure experiment
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.max_n_frames = 12_000
experiment_config.loggers = ["csv"]

# Setup and run benchmark
benchmark = Benchmark(
    algorithm_configs=[
        QmixConfig.get_from_yaml(),
        VdnConfig.get_from_yaml(),
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
```