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

from vdmarl.algorithms.qatten import QAttenLoss, QAttenMixer


class FixedQValues(nn.Module):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.q_values = nn.Parameter(torch.randn(n_agents, n_actions))

    def forward(self, observation):
        return self.q_values.expand(*observation.shape[:-2], *self.q_values.shape)


def _make_mixer(
    *,
    context_features,
    agent_feature_dim,
    n_agents=3,
    variant="base",
    include_local_q_in_keys=False,
):
    return QAttenMixer(
        context_features=context_features,
        agent_feature_dim=agent_feature_dim,
        n_agents=n_agents,
        num_attention_heads=2,
        attention_embed_dim=5,
        query_embedding_num_cells=[8],
        key_embedding_num_cells=[8],
        head_weight_num_cells=[8],
        constant_value_num_cells=[8],
        variant=variant,
        include_local_q_in_keys=include_local_q_in_keys,
        device="cpu",
    )


def _make_loss(
    *,
    action_spec,
    n_agents=3,
    n_actions=4,
    obs_features=5,
    variant="base",
    include_local_q_in_keys=False,
    delay_value=True,
):
    context_features = n_agents * obs_features
    policy_network = TensorDictModule(
        FixedQValues(n_agents, n_actions),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action_value")],
    )
    mixer = _make_mixer(
        context_features=context_features,
        agent_feature_dim=obs_features,
        n_agents=n_agents,
        variant=variant,
        include_local_q_in_keys=include_local_q_in_keys,
    )
    return QAttenLoss(
        group="agents",
        policy_network=policy_network,
        mixer=mixer,
        context_keys=[("agents", "observation")],
        context_shapes=[torch.Size([n_agents, obs_features])],
        agent_feature_keys=[("agents", "observation")],
        agent_feature_shapes=[torch.Size([n_agents, obs_features])],
        action_spec=action_spec,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=0.99,
        delay_value=delay_value,
        loss_function="l2",
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
        mask[..., 3] = False
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
@pytest.mark.parametrize("variant", ["base", "weighted"])
def test_qatten_loss_outputs_are_finite(action_spec, variant):
    loss = _make_loss(action_spec=action_spec, variant=variant)
    batch = _make_batch(action_spec=action_spec, action_mask=True)

    loss_vals = loss(batch)

    assert set(loss_vals.keys()) == {"loss", "loss_td", "td_error"}
    for value in loss_vals.values():
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)
    assert ("agents", "td_error") in batch.keys(True, True)


def test_qatten_loss_respects_action_mask_for_greedy_actions():
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


def test_qatten_attention_weights_normalize_over_agents():
    batch_size = (2, 3)
    n_agents = 3
    context_features = 7
    agent_feature_dim = 5
    mixer = _make_mixer(
        context_features=context_features,
        agent_feature_dim=agent_feature_dim,
        n_agents=n_agents,
    )
    context = torch.randn(*batch_size, context_features)
    agent_features = torch.randn(*batch_size, n_agents, agent_feature_dim)
    local_values = torch.randn(*batch_size, n_agents)

    attention = mixer.attention_weights(context, agent_features, local_values)

    assert attention.shape == torch.Size([*batch_size, 2, n_agents])
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones(*batch_size, 2),
        atol=1e-6,
    )


def test_qatten_weighted_head_weights_are_positive():
    mixer = _make_mixer(
        context_features=7,
        agent_feature_dim=5,
        variant="weighted",
    )
    context = torch.randn(2, 3, 7)

    weights = mixer.head_weights(context)

    assert weights.shape == torch.Size([2, 3, 2])
    assert (weights > 0).all()


def test_qatten_mixer_supports_local_q_keys():
    batch_size = (2, 3)
    n_agents = 3
    context_features = 7
    agent_feature_dim = 5
    mixer = _make_mixer(
        context_features=context_features,
        agent_feature_dim=agent_feature_dim,
        n_agents=n_agents,
        include_local_q_in_keys=True,
    )
    context = torch.randn(*batch_size, context_features)
    agent_features = torch.randn(*batch_size, n_agents, agent_feature_dim)
    local_values = torch.randn(*batch_size, n_agents)

    q_tot = mixer(context, agent_features, local_values)

    assert q_tot.shape == torch.Size([*batch_size, 1])
    assert torch.isfinite(q_tot).all()


def test_qatten_loss_target_updater_compatibility():
    loss = _make_loss(
        action_spec=Categorical(n=4, shape=(3,)),
        delay_value=True,
    )

    updater = SoftUpdate(loss, tau=0.5)
    updater.step()

    assert hasattr(loss, "target_policy_network_params")
    assert hasattr(loss, "target_mixer_params")


def test_qatten_context_and_agent_features_fallback_shapes():
    batch = _make_batch(action_spec=Categorical(n=4, shape=(3,)))
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)))

    context = loss._context(batch)
    agent_features = loss._agent_features(batch)

    assert context.shape == torch.Size([2, 3, 15])
    assert agent_features.shape == torch.Size([2, 3, 3, 5])
