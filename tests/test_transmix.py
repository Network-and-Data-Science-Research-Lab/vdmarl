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

from vdmarl.algorithms.transmix import (
    TRANSMIX_PREVIOUS_ACTION_KEY,
    _FastformerLayer,
    _TransMixPreviousActionTransform,
    TransMixLoss,
    TransMixMixer,
)


class FixedQValues(nn.Module):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.q_values = nn.Parameter(torch.randn(n_agents, n_actions))

    def forward(self, observation, previous_action=None):
        return self.q_values.expand(*observation.shape[:-2], *self.q_values.shape)


def _make_mixer(
    *,
    context_features,
    agent_feature_dim,
    n_agents=3,
    num_transformer_layers=2,
    use_skip_connection=True,
):
    return TransMixMixer(
        context_features=context_features,
        agent_feature_dim=agent_feature_dim,
        n_agents=n_agents,
        num_transformer_layers=num_transformer_layers,
        num_attention_heads=2,
        embedding_dim=12,
        feedforward_dim=16,
        agent_context_num_cells=[8],
        state_embedding_num_cells=[8],
        skip_bottleneck_dim=8,
        dropout=0.0,
        use_skip_connection=use_skip_connection,
        device="cpu",
    )


def _make_loss(
    *,
    action_spec,
    n_agents=3,
    n_actions=4,
    obs_features=5,
    use_previous_action=True,
    delay_value=True,
    num_transformer_layers=2,
    use_skip_connection=True,
):
    context_features = n_agents * obs_features
    previous_action_dim = n_actions if use_previous_action else 0
    policy_in_keys = [("agents", "observation")]
    agent_feature_keys = [("agents", "observation")]
    agent_feature_shapes = [torch.Size([n_agents, obs_features])]
    if use_previous_action:
        policy_in_keys.append(("agents", TRANSMIX_PREVIOUS_ACTION_KEY))
        agent_feature_keys.append(("agents", TRANSMIX_PREVIOUS_ACTION_KEY))
        agent_feature_shapes.append(torch.Size([n_agents, n_actions]))

    policy_network = TensorDictModule(
        FixedQValues(n_agents, n_actions),
        in_keys=policy_in_keys,
        out_keys=[("agents", "action_value")],
    )
    mixer = _make_mixer(
        context_features=context_features,
        agent_feature_dim=obs_features + previous_action_dim,
        n_agents=n_agents,
        num_transformer_layers=num_transformer_layers,
        use_skip_connection=use_skip_connection,
    )
    return TransMixLoss(
        group="agents",
        policy_network=policy_network,
        mixer=mixer,
        context_keys=[("agents", "observation")],
        context_shapes=[torch.Size([n_agents, obs_features])],
        agent_feature_keys=agent_feature_keys,
        agent_feature_shapes=agent_feature_shapes,
        action_spec=action_spec,
        n_agents=n_agents,
        n_actions=n_actions,
        gamma=0.99,
        delay_value=delay_value,
        loss_function="l2",
        previous_action_key=(
            ("agents", TRANSMIX_PREVIOUS_ACTION_KEY) if use_previous_action else None
        ),
    )


def _make_previous_action(action_index, n_actions):
    return torch.nn.functional.one_hot(action_index, n_actions).to(torch.float)


def _make_batch(
    *,
    action_spec,
    batch_size=(2, 3),
    n_agents=3,
    n_actions=4,
    obs_features=5,
    action_mask=False,
    use_previous_action=True,
):
    obs = torch.randn(*batch_size, n_agents, obs_features)
    next_obs = torch.randn(*batch_size, n_agents, obs_features)
    action_index = torch.randint(n_actions, (*batch_size, n_agents))
    next_previous_action_index = torch.randint(n_actions, (*batch_size, n_agents))
    if isinstance(action_spec, OneHot):
        action = _make_previous_action(action_index, n_actions)
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
    if use_previous_action:
        data[("agents", TRANSMIX_PREVIOUS_ACTION_KEY)] = _make_previous_action(
            action_index, n_actions
        )
        data[("next", "agents", TRANSMIX_PREVIOUS_ACTION_KEY)] = (
            _make_previous_action(next_previous_action_index, n_actions)
        )
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
def test_transmix_loss_outputs_are_finite(action_spec):
    loss = _make_loss(action_spec=action_spec)
    batch = _make_batch(action_spec=action_spec, action_mask=True)

    loss_vals = loss(batch)

    assert set(loss_vals.keys()) == {"loss", "loss_td", "td_error"}
    for value in loss_vals.values():
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)
    assert ("agents", "td_error") in batch.keys(True, True)


def test_transmix_previous_action_transform_reset_and_step():
    n_agents = 3
    n_actions = 4
    action_spec = Categorical(n=n_actions, shape=(n_agents,))
    transform = _TransMixPreviousActionTransform(
        group="agents",
        n_agents=n_agents,
        n_actions=n_actions,
        action_spec=action_spec,
    )
    reset_td = TensorDict({}, batch_size=(2,))

    reset_td = transform._reset(None, reset_td)

    assert torch.equal(
        reset_td.get(("agents", TRANSMIX_PREVIOUS_ACTION_KEY)),
        torch.zeros(2, n_agents, n_actions),
    )

    action = torch.tensor([[0, 1, 2], [2, 3, 0]])
    current_td = TensorDict({("agents", "action"): action}, batch_size=(2,))
    next_td = transform._step(current_td, TensorDict({}, batch_size=(2,)))

    assert torch.equal(
        next_td.get(("agents", TRANSMIX_PREVIOUS_ACTION_KEY)),
        _make_previous_action(action, n_actions),
    )

    one_hot_transform = _TransMixPreviousActionTransform(
        group="agents",
        n_agents=n_agents,
        n_actions=n_actions,
        action_spec=OneHot(n=n_actions, shape=(n_agents, n_actions)),
    )
    one_hot_action = _make_previous_action(action, n_actions)
    current_td = TensorDict({("agents", "action"): one_hot_action}, batch_size=(2,))
    next_td = one_hot_transform._step(current_td, TensorDict({}, batch_size=(2,)))

    assert torch.equal(
        next_td.get(("agents", TRANSMIX_PREVIOUS_ACTION_KEY)),
        one_hot_action,
    )


def test_transmix_loss_respects_action_mask_for_greedy_actions():
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


@pytest.mark.parametrize("num_transformer_layers", [1, 2, 6])
def test_transmix_mixer_shapes_for_transformer_depths(num_transformer_layers):
    batch_size = (2, 3)
    n_agents = 3
    context_features = 7
    agent_feature_dim = 5
    mixer = _make_mixer(
        context_features=context_features,
        agent_feature_dim=agent_feature_dim,
        n_agents=n_agents,
        num_transformer_layers=num_transformer_layers,
    )
    context = torch.randn(*batch_size, context_features)
    agent_features = torch.randn(*batch_size, n_agents, agent_feature_dim)
    local_values = torch.randn(*batch_size, n_agents)

    tokens = mixer.token_embeddings(context, agent_features, local_values)
    q_tot = mixer(context, agent_features, local_values)

    assert tokens.shape == torch.Size([*batch_size, 1 + 2 * n_agents, 12])
    assert q_tot.shape == torch.Size([*batch_size, 1])
    assert torch.isfinite(q_tot).all()


def test_transmix_fastformer_attention_shape_and_finite_values():
    layer = _FastformerLayer(
        embedding_dim=12,
        num_attention_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        device="cpu",
    )
    tokens = torch.randn(2, 7, 12)

    attention_output = layer.fast_attention(tokens)
    output = layer(tokens)

    assert attention_output.shape == tokens.shape
    assert output.shape == tokens.shape
    assert torch.isfinite(attention_output).all()
    assert torch.isfinite(output).all()


def test_transmix_mixer_without_skip_connection():
    mixer = _make_mixer(
        context_features=7,
        agent_feature_dim=5,
        use_skip_connection=False,
    )
    context = torch.randn(2, 7)
    agent_features = torch.randn(2, 3, 5)
    local_values = torch.randn(2, 3)

    q_tot = mixer(context, agent_features, local_values)

    assert q_tot.shape == torch.Size([2, 1])
    assert torch.isfinite(q_tot).all()


def test_transmix_loss_target_updater_compatibility():
    loss = _make_loss(
        action_spec=Categorical(n=4, shape=(3,)),
        delay_value=True,
    )

    updater = SoftUpdate(loss, tau=0.5)
    updater.step()

    assert hasattr(loss, "target_policy_network_params")
    assert hasattr(loss, "target_mixer_params")


def test_transmix_context_and_agent_features_fallback_shapes():
    batch = _make_batch(action_spec=Categorical(n=4, shape=(3,)))
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)))

    context = loss._context(batch)
    agent_features = loss._agent_features(batch)

    assert context.shape == torch.Size([2, 3, 15])
    assert agent_features.shape == torch.Size([2, 3, 3, 9])


def test_transmix_missing_previous_action_has_clear_error():
    loss = _make_loss(action_spec=Categorical(n=4, shape=(3,)))
    batch = _make_batch(
        action_spec=Categorical(n=4, shape=(3,)),
        use_previous_action=False,
    )

    with pytest.raises(KeyError, match="previous-action features"):
        loss(batch)


def test_transmix_loss_without_previous_action_outputs_are_finite():
    loss = _make_loss(
        action_spec=Categorical(n=4, shape=(3,)),
        use_previous_action=False,
    )
    batch = _make_batch(
        action_spec=Categorical(n=4, shape=(3,)),
        use_previous_action=False,
    )

    loss_vals = loss(batch)

    assert torch.isfinite(loss_vals["loss"])
