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
    QmixGnnConfig,
    QplexConfig,
    QtranConfig,
    TransmixConfig,
    WqmixConfig,
)
from vdmarl.algorithms.common import AlgorithmConfig
from vdmarl.hydra_config import load_algorithm_config_from_hydra
from hydra import compose, initialize


@pytest.mark.parametrize("algo_name", algorithm_config_registry.keys())
def test_loading_algorithms(algo_name):
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algo_name}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert algo_config == algorithm_config_registry[algo_name].get_from_yaml()


@pytest.mark.parametrize("variant", ["base", "alt"])
def test_loading_qtran_variants(variant):
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=qtran",
                f"algorithm.variant={variant}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, QtranConfig)
        assert algo_config.variant == variant


def test_loading_qplex_defaults():
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=qplex",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, QplexConfig)
        assert algo_config.num_attention_heads == 4
        assert algo_config.stop_local_advantage_gradient


def test_loading_avdnet_defaults():
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=avdnet",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, AvdnetConfig)
        assert algo_config.num_attention_heads == 4
        assert algo_config.use_previous_action


def test_loading_transmix_defaults():
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=transmix",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, TransmixConfig)
        assert algo_config.num_transformer_layers == 2
        assert algo_config.num_attention_heads == 4
        assert algo_config.use_previous_action


def test_loading_transmix_without_previous_action():
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=transmix",
                "algorithm.use_previous_action=false",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, TransmixConfig)
        assert not algo_config.use_previous_action


def test_loading_qmix_gnn_defaults():
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=qmix_gnn",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, QmixGnnConfig)
        assert algo_config.graph_topology == "full"
        assert algo_config.num_attention_heads == 4


@pytest.mark.parametrize("variant", ["base", "weighted"])
def test_loading_qatten_variants(variant):
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=qatten",
                f"algorithm.variant={variant}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, QattenConfig)
        assert algo_config.variant == variant
        assert algo_config.num_attention_heads == 4


@pytest.mark.parametrize("variant", ["ow", "cw"])
def test_loading_wqmix_variants(variant):
    with initialize(version_base=None, config_path="../vdmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=wqmix",
                f"algorithm.variant={variant}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert isinstance(algo_config, WqmixConfig)
        assert algo_config.variant == variant
