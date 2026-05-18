#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Categorical, OneHot
from torchrl.objectives.utils import SoftUpdate

from vdmarl.algorithms.wqmix import QStarNetwork, WQMIXLoss, WQMIXMixer


class FixedQValues(nn.Module):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.q_values = nn.Parameter(torch.randn(n_agents, n_actions))

    def forward(self, observation):
        return self.q_values.expand(*observation.shape[:-2], *self.q_values.shape)


def _make_loss(
    *,
    action_spec,
    n_agents=3,
    n_actions=4,
    obs_features=5,
    variant="ow",
    delay_value=True,
):
    context_features = n_agents * obs_features
    policy_network = TensorDictModule(
        FixedQValues(n_agents, n_actions),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action_value")],
    )
    qtot_mixer = WQMIXMixer(
        context_features=context_features,
        mixing_embed_dim=8,
        n_agents=n_agents,
        device="cpu",
    )
    qstar_network = QStarNetwork(
        context_features=context_features,
        n_agents=n_agents,
        n_actions=n_actions,
        hidden_cells=[8],
        device="cpu",
    )
    return WQMIXLoss(
        group="agents",
        policy_network=policy_network,
        qtot_mixer=qtot_mixer,
        qstar_network=qstar_network,
        context_keys=[("agents", "observation")],
        context_shapes=[torch.Size([n_agents, obs_features])],
        action_spec=action_spec,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=0.99,
        delay_value=delay_value,
        loss_function="l2",
        variant=variant,
        alpha=0.5,
    )


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
        mask[..., 2] = False
        data[("agents", "action_mask")] = mask
        data[("next", "agents", "action_mask")] = mask
    return TensorDict(data, batch_size=batch_size)


@pytest.mark.parametrize(
    "action_spec",
    [
        Categorical(n=4, shape=(3,)),
        OneHot(n=4, shape=(3, 4)),
    ],
)
@pytest.mark.parametrize("variant", ["ow", "cw"])
def test_wqmix_loss_outputs_are_finite(action_spec, variant):
    loss = _make_loss(action_spec=action_spec, variant=variant)
    batch = _make_batch(action_spec=action_spec, action_mask=True)

    loss_vals = loss(batch)

    assert set(loss_vals.keys()) == {
        "loss",
        "loss_qtot",
        "loss_qstar",
        "weight_mean",
        "td_error",
    }
    for value in loss_vals.values():
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)
    assert ("agents", "td_error") in batch.keys(True, True)


def test_wqmix_loss_respects_action_mask_for_greedy_actions():
    n_agents = 3
    n_actions = 4
    loss = _make_loss(action_spec=Categorical(n=n_actions, shape=(n_agents,)))
    action_values = torch.arange(n_actions, dtype=torch.float).expand(
        2, n_agents, n_actions
    )
    mask = torch.ones(2, n_agents, n_actions, dtype=torch.bool)
    mask[..., 3] = False

    action_index, _ = loss._greedy_actions(action_values, mask)

    assert torch.equal(action_index, torch.full((2, n_agents), 2))


def test_wqmix_ow_weights():
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)), variant="ow")

    weights = loss._weights(
        q_tot=torch.tensor([[0.0], [2.0]]),
        target=torch.tensor([[1.0], [1.0]]),
        q_star=torch.zeros(2, 1),
        action_index=torch.zeros(2, 3, dtype=torch.long),
        current_greedy_index=torch.zeros(2, 3, dtype=torch.long),
        current_greedy_q_star=torch.zeros(2, 1),
    )

    assert torch.equal(weights, torch.tensor([[1.0], [0.5]]))


def test_wqmix_cw_weights():
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)), variant="cw")

    weights = loss._weights(
        q_tot=torch.zeros(3, 1),
        target=torch.tensor([[0.0], [2.0], [0.0]]),
        q_star=torch.zeros(3, 1),
        action_index=torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
        current_greedy_index=torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        current_greedy_q_star=torch.tensor([[1.0], [1.0], [1.0]]),
    )

    assert torch.equal(weights, torch.tensor([[1.0], [1.0], [0.5]]))


def test_wqmix_qstar_shapes_for_sampled_and_greedy_actions():
    batch_size = (2, 3)
    n_agents = 3
    n_actions = 4
    context_features = 7
    qstar = QStarNetwork(
        context_features=context_features,
        n_agents=n_agents,
        n_actions=n_actions,
        hidden_cells=[8],
        device="cpu",
    )
    context = torch.randn(*batch_size, context_features)
    sampled_action = torch.randint(n_actions, (*batch_size, n_agents))
    greedy_action = torch.randint(n_actions, (*batch_size, n_agents))

    sampled_qstar = qstar(context, sampled_action)
    greedy_qstar = qstar(context, greedy_action)

    assert sampled_qstar.shape == torch.Size([*batch_size, 1])
    assert greedy_qstar.shape == torch.Size([*batch_size, 1])
    assert torch.isfinite(sampled_qstar).all()
    assert torch.isfinite(greedy_qstar).all()


def test_wqmix_loss_target_updater_compatibility():
    loss = _make_loss(
        action_spec=Categorical(n=4, shape=(3,)),
        delay_value=True,
    )

    updater = SoftUpdate(loss, tau=0.5)
    updater.step()

    assert hasattr(loss, "target_policy_network_params")
    assert hasattr(loss, "target_qtot_mixer_params")
    assert hasattr(loss, "target_qstar_network_params")
