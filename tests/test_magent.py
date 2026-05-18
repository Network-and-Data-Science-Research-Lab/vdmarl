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
from vdmarl.environments import MAgentTask, Task
from vdmarl.experiment import Experiment

from utils import _has_magent2, _has_torch_geometric
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


@pytest.mark.skipif(not _has_magent2, reason="magent2 not found")
class TestMagent:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
        mlp_sequence_config,
    ):

        # To not run unsupported algo-task pairs
        if algo_config is QmixGnnConfig and not _has_torch_geometric:
            pytest.skip("QMIX-GNN requires torch_geometric")
        if not algo_config.supports_discrete_actions():
            pytest.skip()

        task = task.get_from_yaml()
        model_config = (
            mlp_sequence_config if algo_config is QmixGnnConfig else cnn_sequence_config
        )
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_gnn_sequence_config,
        mlp_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        model_config = (
            mlp_gnn_sequence_config
            if algo_config is QmixGnnConfig
            else cnn_gnn_sequence_config
        )
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=model_config,
            critic_model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_lstm_sequence_config,
        lstm_mlp_sequence_config,
    ):
        is_qmix_gnn = algo_config is QmixGnnConfig
        algo_config = algo_config.get_from_yaml()
        experiment_config.share_policy_params = False
        task = task.get_from_yaml()
        model_config = (
            lstm_mlp_sequence_config if is_qmix_gnn else cnn_lstm_sequence_config
        )
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=model_config,
            critic_model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
        mlp_sequence_config,
    ):
        # To not run unsupported algo-task pairs
        if not algo_config.supports_discrete_actions():
            pytest.skip()
        is_qmix_gnn = algo_config is QmixGnnConfig
        algo_config = algo_config.get_from_yaml()
        model_config = mlp_sequence_config if is_qmix_gnn else cnn_sequence_config

        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=model_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize("algo_config", VALUE_DECOMPOSITION_ALGOS)
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        cnn_sequence_config,
        mlp_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        model_config = (
            mlp_sequence_config if algo_config is QmixGnnConfig else cnn_sequence_config
        )
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
