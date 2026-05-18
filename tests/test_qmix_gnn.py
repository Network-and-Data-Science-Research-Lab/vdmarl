#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Categorical, OneHot
from torchrl.modules import QMixer, QValueModule
from torchrl.objectives import QMixerLoss, ValueEstimators
from torchrl.objectives.utils import SoftUpdate

pytest.importorskip("torch_geometric")

from vdmarl.algorithms.qmix_gnn import InformationInfusionModule


class FixedQValues(nn.Module):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.q_values = nn.Parameter(torch.randn(n_agents, n_actions))

    def forward(self, observation):
        return self.q_values.expand(*observation.shape[:-2], *self.q_values.shape)


def _make_infusion(
    *,
    n_agents=3,
    obs_features=5,
    projection_dim=7,
    gnn_hidden_dim=11,
    graph_topology="full",
    position_key_index=None,
    knn_k=1,
    edge_radius=None,
):
    shapes = [torch.Size([n_agents, obs_features])]
    if position_key_index is not None:
        shapes.append(torch.Size([n_agents, 2]))
    return InformationInfusionModule(
        observation_shapes=shapes,
        n_agents=n_agents,
        projection_dim=projection_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        num_attention_heads=2,
        gnn_num_layers=1,
        gnn_dropout=0.0,
        graph_topology=graph_topology,
        position_key_index=position_key_index,
        knn_k=knn_k,
        edge_radius=edge_radius,
        self_loops=False,
        device="cpu",
    )


def _make_loss(
    *,
    action_spec,
    n_agents=3,
    n_actions=4,
    obs_features=5,
    delay_value=True,
):
    projection_dim = 7
    gnn_hidden_dim = 11
    policy = _make_policy(
        action_spec=action_spec,
        n_agents=n_agents,
        n_actions=n_actions,
        obs_features=obs_features,
        projection_dim=projection_dim,
        gnn_hidden_dim=gnn_hidden_dim,
    )
    mixer = TensorDictModule(
        QMixer(
            state_shape=(gnn_hidden_dim,),
            mixing_embed_dim=8,
            n_agents=n_agents,
            device="cpu",
        ),
        in_keys=[
            ("agents", "chosen_action_value"),
            ("agents", "qmix_gnn_team_info"),
        ],
        out_keys=["chosen_action_value"],
    )
    loss = QMixerLoss(
        policy,
        mixer,
        delay_value=delay_value,
        loss_function="l2",
        action_space=action_spec,
    )
    loss.set_keys(
        reward="reward",
        action=("agents", "action"),
        done="done",
        terminated="terminated",
        action_value=("agents", "action_value"),
        local_value=("agents", "chosen_action_value"),
        global_value="chosen_action_value",
        priority="td_error",
    )
    loss.make_value_estimator(ValueEstimators.TD0, gamma=0.99)
    return loss


def _make_policy(
    *,
    action_spec,
    n_agents,
    n_actions,
    obs_features,
    projection_dim,
    gnn_hidden_dim,
):
    infusion = TensorDictModule(
        _make_infusion(
            n_agents=n_agents,
            obs_features=obs_features,
            projection_dim=projection_dim,
            gnn_hidden_dim=gnn_hidden_dim,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[
            ("agents", "qmix_gnn_agent_input"),
            ("agents", "qmix_gnn_team_info"),
        ],
    )
    q_network = TensorDictModule(
        FixedQValues(n_agents, n_actions),
        in_keys=[("agents", "qmix_gnn_agent_input")],
        out_keys=[("agents", "action_value")],
    )
    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        action_mask_key=("agents", "action_mask"),
        out_keys=[
            ("agents", "action"),
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=action_spec,
        action_space=None,
    )
    return TensorDictSequential(infusion, q_network, value_module)


def _make_batch(
    *,
    action_spec,
    batch_size=(2, 3),
    n_agents=3,
    n_actions=4,
    obs_features=5,
    action_mask=False,
):
    obs = torch.randn(*batch_size, n_agents, obs_features)
    next_obs = torch.randn(*batch_size, n_agents, obs_features)
    action_index = torch.randint(n_actions, (*batch_size, n_agents))
    if isinstance(action_spec, OneHot):
        action = torch.nn.functional.one_hot(action_index, n_actions).to(torch.float)
    else:
        action = action_index

    data = {
        ("agents", "observation"): obs,
        ("agents", "action"): action,
        ("next", "agents", "observation"): next_obs,
        ("next", "reward"): torch.randn(*batch_size, 1),
        ("next", "done"): torch.zeros(*batch_size, 1, dtype=torch.bool),
        ("next", "terminated"): torch.zeros(*batch_size, 1, dtype=torch.bool),
    }
    if action_mask:
        mask = torch.ones(*batch_size, n_agents, n_actions, dtype=torch.bool)
        mask[..., 3] = False
        data[("agents", "action_mask")] = mask
        data[("next", "agents", "action_mask")] = mask
    return TensorDict(data, batch_size=batch_size)


def test_qmix_gnn_information_infusion_shapes():
    batch_size = (2, 3)
    n_agents = 3
    projection_dim = 7
    gnn_hidden_dim = 11
    infusion = _make_infusion(
        n_agents=n_agents,
        projection_dim=projection_dim,
        gnn_hidden_dim=gnn_hidden_dim,
    )
    observation = torch.randn(*batch_size, n_agents, 5)

    agent_input, team_info = infusion(observation)

    assert agent_input.shape == torch.Size(
        [*batch_size, n_agents, projection_dim + gnn_hidden_dim]
    )
    assert team_info.shape == torch.Size([*batch_size, gnn_hidden_dim])
    assert torch.isfinite(agent_input).all()
    assert torch.isfinite(team_info).all()


def test_qmix_gnn_knn_graph_construction():
    infusion = _make_infusion(graph_topology="knn", position_key_index=1, knn_k=1)
    position = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]])

    edge_index = infusion._knn_edge_index(position)

    assert edge_index.shape == torch.Size([2, 3])
    assert torch.equal(edge_index[:, 0], torch.tensor([1, 0]))


def test_qmix_gnn_radius_graph_construction():
    infusion = _make_infusion(
        graph_topology="radius",
        position_key_index=1,
        edge_radius=1.5,
    )
    position = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]])

    edge_index = infusion._radius_edge_index(position)

    assert edge_index.shape == torch.Size([2, 2])
    assert {tuple(edge.tolist()) for edge in edge_index.T} == {(0, 1), (1, 0)}


@pytest.mark.parametrize(
    "action_spec",
    [
        Categorical(n=4, shape=(3,)),
        OneHot(n=4, shape=(3, 4)),
    ],
)
def test_qmix_gnn_loss_outputs_are_finite(action_spec):
    loss = _make_loss(action_spec=action_spec)
    batch = _make_batch(action_spec=action_spec, action_mask=True)

    loss_vals = loss(batch)

    for key, value in loss_vals.items():
        if key.startswith("loss"):
            assert value.shape == torch.Size([])
        assert torch.isfinite(value).all()


def test_qmix_gnn_action_mask_reaches_greedy_selection():
    action_spec = Categorical(n=4, shape=(3,))
    policy = _make_policy(
        action_spec=action_spec,
        n_agents=3,
        n_actions=4,
        obs_features=5,
        projection_dim=7,
        gnn_hidden_dim=11,
    )
    batch = _make_batch(action_spec=action_spec, batch_size=(2,), action_mask=True)

    policy_out = policy(batch.clone())

    assert (policy_out.get(("agents", "action")) != 3).all()


def test_qmix_gnn_loss_target_updater_compatibility():
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)), delay_value=True)

    updater = SoftUpdate(loss, tau=0.5)
    updater.step()
