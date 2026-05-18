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

from vdmarl.algorithms.qtran import QTranLoss


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
    variant="base",
    delay_value=True,
):
    context_features = n_agents * obs_features
    policy_network = TensorDictModule(
        FixedQValues(n_agents, n_actions),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action_value")],
    )
    joint_network = TensorDictModule(
        nn.Linear(context_features + n_agents * n_actions, 1),
        in_keys=["_qtran_joint_input"],
        out_keys=["_qtran_joint_value"],
    )
    value_network = TensorDictModule(
        nn.Linear(context_features, 1),
        in_keys=["_qtran_context"],
        out_keys=["_qtran_state_value"],
    )
    return QTranLoss(
        group="agents",
        policy_network=policy_network,
        joint_network=joint_network,
        value_network=value_network,
        context_keys=[("agents", "observation")],
        context_shapes=[torch.Size([n_agents, obs_features])],
        action_spec=action_spec,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=0.99,
        delay_value=delay_value,
        loss_function="l2",
        lambda_opt=1.0,
        lambda_nopt=1.0,
        variant=variant,
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
@pytest.mark.parametrize("variant", ["base", "alt"])
def test_qtran_loss_outputs_are_finite(action_spec, variant):
    loss = _make_loss(action_spec=action_spec, variant=variant)
    batch = _make_batch(action_spec=action_spec)

    loss_vals = loss(batch)

    assert set(loss_vals.keys()) == {
        "loss",
        "loss_td",
        "loss_opt",
        "loss_nopt",
        "td_error",
    }
    for value in loss_vals.values():
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)
    assert ("agents", "td_error") in batch.keys(True, True)


def test_qtran_loss_respects_action_mask_for_greedy_actions():
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


def test_qtran_loss_target_updater_compatibility():
    loss = _make_loss(
        action_spec=Categorical(n=4, shape=(3,)),
        delay_value=True,
    )

    updater = SoftUpdate(loss, tau=0.5)
    updater.step()

    assert hasattr(loss, "target_policy_network_params")
    assert hasattr(loss, "target_joint_network_params")
    assert hasattr(loss, "target_value_network_params")
