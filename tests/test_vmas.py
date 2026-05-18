#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
from vdmarl.algorithms import (
    algorithm_config_registry,
    AvdnetConfig,
    QattenConfig,
    QmixConfig,
    QmixGnnConfig,
    QplexConfig,
    QtranConfig,
    TransmixConfig,
    VdnConfig,
    WqmixConfig,
)
from vdmarl.algorithms.common import AlgorithmConfig
from vdmarl.environments import Task, VmasTask
from vdmarl.experiment import Experiment
from vdmarl.models import MlpConfig
from torch import nn
from utils import _has_torch_geometric, _has_vmas
from utils_experiment import ExperimentUtils


VALUE_DECOMPOSITION_ALGOS = [
    AvdnetConfig,
    QattenConfig,
    QmixConfig,
    QplexConfig,
    QtranConfig,
    TransmixConfig,
    VdnConfig,
    WqmixConfig,
] + ([QmixGnnConfig] if _has_torch_geometric else [])


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestVmas:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize("task", [VmasTask.BALANCE])
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        prefer_continuous,
        experiment_config,
        mlp_sequence_config,
    ):
        # To not run the same test twice
        if algo_config is QmixGnnConfig and not _has_torch_geometric:
            pytest.skip("QMIX-GNN requires torch_geometric")
        if (prefer_continuous and not algo_config.supports_continuous_actions()) or (
            not prefer_continuous and not algo_config.supports_discrete_actions()
        ):
            pytest.skip()

        task = task.get_from_yaml()
        experiment_config.prefer_continuous_actions = prefer_continuous
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", list(VmasTask))
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        task = task.get_from_yaml()
        if not task.supports_discrete_actions():
            pytest.skip("Value-decomposition algorithms require discrete actions")
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_collect_with_grad(
        self,
        experiment_config,
        mlp_sequence_config,
        algo_config: AlgorithmConfig = QmixConfig,
        task: Task = VmasTask.BALANCE,
    ):
        task = task.get_from_yaml()
        experiment_config.collect_with_grad = True
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_gnn_sequence_config,
            critic_model_config=mlp_gnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_gru(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        gru_mlp_sequence_config,
        share_params: bool = False,
    ):
        algo_config = algo_config.get_from_yaml()
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=gru_mlp_sequence_config,
            critic_model_config=gru_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        lstm_mlp_sequence_config,
        share_params: bool = False,
    ):
        algo_config = algo_config.get_from_yaml()
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=lstm_mlp_sequence_config,
            critic_model_config=lstm_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [VmasTask.BALANCE])
    def test_reloading_trainer(
        self,
        algo_config,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        if isinstance(algo_config, QmixGnnConfig) and not _has_torch_geometric:
            pytest.skip("QMIX-GNN requires torch_geometric")

        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=mlp_sequence_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        mlp_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        critic_model_config = MlpConfig(
            num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear
        )
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
