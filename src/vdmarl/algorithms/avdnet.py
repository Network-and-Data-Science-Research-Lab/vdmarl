from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, MISSING
from math import prod, sqrt
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Composite, OneHot, Unbounded
from torchrl.envs import Compose, EnvBase, Transform, TransformedEnv
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import LossModule

from vdmarl.algorithms.common import Algorithm, AlgorithmConfig
from vdmarl.models.common import ModelConfig


AVDNET_PREVIOUS_ACTION_KEY = "avdnet_previous_action"


def _make_mlp(
    in_features: int,
    hidden_cells: Sequence[int],
    out_features: int,
    device,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_features = in_features
    for hidden_features in hidden_cells:
        layers.append(nn.Linear(last_features, hidden_features, device=device))
        layers.append(nn.ReLU())
        last_features = hidden_features
    layers.append(nn.Linear(last_features, out_features, device=device))
    return nn.Sequential(*layers)


def _nested_key(prefix, key):
    key = key if isinstance(key, tuple) else (key,)
    return (*prefix, *key)


class _AVDNetPreviousActionTransform(Transform):
    """Adds a one-hot previous-action feature for each agent in a group."""

    def __init__(
        self,
        group: str,
        n_agents: int,
        n_actions: int,
        action_spec,
        key: str = AVDNET_PREVIOUS_ACTION_KEY,
    ):
        super().__init__(
            in_keys=[],
            out_keys=[(group, key)],
        )
        self.group = group
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_spec = action_spec
        self.key = key

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset.set((self.group, self.key), self._zeros(tensordict_reset))
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        action = tensordict.get((self.group, "action"))
        previous_action = self._one_hot(action).to(dtype=torch.float32)
        if next_tensordict.device is not None:
            previous_action = previous_action.to(next_tensordict.device)
        next_tensordict.set((self.group, self.key), previous_action)
        return next_tensordict

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clone()
        group_shape = observation_spec[self.group].shape
        observation_spec.set(
            (self.group, self.key),
            Unbounded(
                shape=(*group_shape, self.n_actions),
                device=observation_spec.device,
            ),
        )
        return observation_spec

    def _zeros(self, tensordict: TensorDictBase) -> torch.Tensor:
        kwargs = {"dtype": torch.float32}
        if tensordict.device is not None:
            kwargs["device"] = tensordict.device
        return torch.zeros(
            *tensordict.batch_size,
            self.n_agents,
            self.n_actions,
            **kwargs,
        )

    def _one_hot(self, action: torch.Tensor) -> torch.Tensor:
        if isinstance(self.action_spec, OneHot) or (
            action.shape[-1] == self.n_actions and action.dtype.is_floating_point
        ):
            return action.to(torch.float32)
        action_index = action.to(torch.long)
        while action_index.ndim > 0 and action_index.shape[-1] == 1:
            action_index = action_index.squeeze(-1)
        return F.one_hot(action_index, self.n_actions).to(torch.float32)


class AVDNetMixer(nn.Module):
    """Attention value-decomposition mixer for AVD-Net."""

    def __init__(
        self,
        context_features: int,
        agent_feature_dim: int,
        n_agents: int,
        num_attention_heads: int,
        attention_embed_dim: int,
        agent_embedding_num_cells: Sequence[int],
        attention_weight_num_cells: Sequence[int],
        head_weight_num_cells: Sequence[int],
        state_bias_num_cells: Sequence[int],
        positive_eps: float,
        device,
    ):
        super().__init__()
        if num_attention_heads <= 0:
            raise ValueError("AVD-Net num_attention_heads must be greater than 0")
        if attention_embed_dim <= 0:
            raise ValueError("AVD-Net attention_embed_dim must be greater than 0")
        if positive_eps <= 0:
            raise ValueError("AVD-Net positive_eps must be greater than 0")

        self.context_features = context_features
        self.agent_feature_dim = agent_feature_dim
        self.n_agents = n_agents
        self.num_attention_heads = num_attention_heads
        self.attention_embed_dim = attention_embed_dim
        self.positive_eps = positive_eps

        embedding_out_features = num_attention_heads * attention_embed_dim
        self.agent_encoder = _make_mlp(
            in_features=agent_feature_dim,
            hidden_cells=agent_embedding_num_cells,
            out_features=embedding_out_features,
            device=device,
        )
        self.attention_weight_net = _make_mlp(
            in_features=attention_embed_dim,
            hidden_cells=attention_weight_num_cells,
            out_features=1,
            device=device,
        )
        self.head_weight_net = _make_mlp(
            in_features=context_features,
            hidden_cells=head_weight_num_cells,
            out_features=num_attention_heads,
            device=device,
        )
        self.state_bias_net = _make_mlp(
            in_features=context_features,
            hidden_cells=state_bias_num_cells,
            out_features=1,
            device=device,
        )

    def forward(
        self,
        context: torch.Tensor,
        agent_features: torch.Tensor,
        local_values: torch.Tensor,
    ) -> torch.Tensor:
        context = context.to(local_values.dtype)
        attention = self.attention_weights(agent_features, dtype=local_values.dtype)
        head_values = (attention * local_values.unsqueeze(-2)).sum(dim=-1)
        weighted_head_values = self.head_weights(context) * head_values
        return weighted_head_values.sum(dim=-1, keepdim=True) + self.state_bias_net(
            context
        )

    def attention_weights(
        self,
        agent_features: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if dtype is not None:
            agent_features = agent_features.to(dtype)

        embeddings = self.agent_encoder(agent_features).reshape(
            *agent_features.shape[:-2],
            self.n_agents,
            self.num_attention_heads,
            self.attention_embed_dim,
        )
        embeddings = embeddings.movedim(-3, -2)

        logits = torch.matmul(embeddings, embeddings.transpose(-1, -2)) / sqrt(
            self.attention_embed_dim
        )
        coordination = logits.softmax(dim=-1)
        attended_embeddings = torch.matmul(coordination, embeddings)

        weight_logits = self.attention_weight_net(attended_embeddings).squeeze(-1)
        return weight_logits.softmax(dim=-1)

    def head_weights(self, context: torch.Tensor) -> torch.Tensor:
        return self.head_weight_net(context).abs() + self.positive_eps


class AVDNetLoss(LossModule):
    """TD loss for AVD-Net with target-policy greedy bootstrap."""

    def __init__(
        self,
        group: str,
        policy_network: TensorDictModule,
        mixer: AVDNetMixer,
        context_keys: List[Tuple],
        context_shapes: List[torch.Size],
        agent_feature_keys: List[Tuple],
        agent_feature_shapes: List[torch.Size],
        action_spec,
        n_agents: int,
        n_actions: int,
        gamma: float,
        delay_value: bool,
        loss_function: str,
        previous_action_key: Tuple | None,
    ):
        super().__init__()
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "AVD-Net loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

        self.group = group
        self.context_keys = context_keys
        self.context_shapes = context_shapes
        self.agent_feature_keys = agent_feature_keys
        self.agent_feature_shapes = agent_feature_shapes
        self.action_spec = action_spec
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.delay_value = delay_value
        self.loss_function = loss_function
        self.previous_action_key = previous_action_key

        self.convert_to_functional(
            policy_network,
            "policy_network",
            create_target_params=delay_value,
        )
        self.convert_to_functional(
            mixer,
            "mixer",
            create_target_params=delay_value,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._validate_previous_action(tensordict)

        action = tensordict.get((self.group, "action"))
        policy_td = self._run_policy(tensordict.clone(), target=False)
        action_values = self._canonical_action_values(
            policy_td.get((self.group, "action_value"))
        )

        action_index = self._action_index(action)
        local_values = self._chosen_action_values(action_values, action_index)
        q_tot = self._mix(
            context=self._context(tensordict),
            agent_features=self._agent_features(tensordict),
            local_values=local_values,
            target=False,
        )

        with torch.no_grad():
            next_td = tensordict.get("next").clone()
            
            # Online network for action selection (Double Q-Learning)
            online_next_policy_td = self._run_policy(next_td, target=False)
            online_next_action_values = self._canonical_action_values(
                online_next_policy_td.get((self.group, "action_value"))
            )
            next_action_mask = self._canonical_mask(
                self._get_optional(next_td, (self.group, "action_mask"))
            )
            next_action_index, _ = self._greedy_actions(
                online_next_action_values, next_action_mask
            )

            # Target network for value evaluation
            next_policy_td = self._run_policy(next_td, target=True)
            next_action_values = self._canonical_action_values(
                next_policy_td.get((self.group, "action_value"))
            )
            next_local_values = self._chosen_action_values(
                next_action_values, next_action_index
            )
            next_q_tot = self._mix(
                context=self._context(tensordict, next=True),
                agent_features=self._agent_features(tensordict, next=True),
                local_values=next_local_values,
                target=True,
            )
            reward = self._match_value_shape(
                tensordict.get(("next", "reward")), next_q_tot
            )
            terminated = self._match_value_shape(
                tensordict.get(("next", "terminated")), next_q_tot
            )
            target_q_tot = reward + self.gamma * (
                1 - terminated.to(next_q_tot.dtype)
            ) * next_q_tot

        loss_td = self._distance(q_tot, target_q_tot).mean()
        td_error = (q_tot - target_q_tot).detach().abs().squeeze(-1)
        tensordict.set((self.group, "td_error"), td_error.unsqueeze(-1).expand(*td_error.shape, self.n_agents))

        return TensorDict(
            {
                "loss": loss_td,
                "loss_td": loss_td,
                "td_error": td_error.mean(),
            },
            batch_size=[],
        )

    def _module_params_context(self, module_name: str, target: bool):
        params_name = f"{'target_' if target else ''}{module_name}_params"
        params = getattr(self, params_name, None)
        if params is None and target:
            params = getattr(self, f"{module_name}_params", None)
        module = getattr(self, module_name)
        return params.to_module(module) if params is not None else nullcontext()

    def _run_policy(self, tensordict: TensorDictBase, target: bool) -> TensorDictBase:
        with self._module_params_context("policy_network", target=target):
            return self.policy_network(tensordict)

    def _mix(
        self,
        context: torch.Tensor,
        agent_features: torch.Tensor,
        local_values: torch.Tensor,
        target: bool,
    ) -> torch.Tensor:
        with self._module_params_context("mixer", target=target):
            return self.mixer(context, agent_features, local_values)

    def _context(self, tensordict: TensorDictBase, next: bool = False) -> torch.Tensor:
        prefix = ("next",) if next else ()
        values = []
        for key, shape in zip(self.context_keys, self.context_shapes):
            value = tensordict.get(_nested_key(prefix, key))
            event_ndim = len(shape)
            if event_ndim == 0:
                values.append(value.unsqueeze(-1))
            else:
                values.append(value.reshape(*value.shape[:-event_ndim], -1))
        return torch.cat(values, dim=-1)

    def _agent_features(
        self, tensordict: TensorDictBase, next: bool = False
    ) -> torch.Tensor:
        prefix = ("next",) if next else ()
        values = []
        for key, shape in zip(self.agent_feature_keys, self.agent_feature_shapes):
            value = tensordict.get(_nested_key(prefix, key))
            event_ndim = len(shape)
            values.append(value.reshape(*value.shape[:-event_ndim], self.n_agents, -1))
        return torch.cat(values, dim=-1)

    def _validate_previous_action(self, tensordict: TensorDictBase) -> None:
        if self.previous_action_key is None:
            return
        missing_keys = []
        for key in (
            self.previous_action_key,
            _nested_key(("next",), self.previous_action_key),
        ):
            try:
                tensordict.get(key)
            except KeyError:
                missing_keys.append(key)
        if missing_keys:
            raise KeyError(
                "AVD-Net requires previous-action features in sampled batches. "
                f"Missing keys: {missing_keys}. Ensure algorithm.process_env_fun "
                "wraps the environment, or disable algorithm.use_previous_action."
            )

    def _canonical_action_values(self, action_values: torch.Tensor) -> torch.Tensor:
        if action_values.shape[-1] != self.n_actions:
            raise ValueError(
                f"Expected action-values last dimension {self.n_actions}, got {action_values.shape}"
            )
        if action_values.shape[-2] == self.n_agents:
            return action_values
        if action_values.ndim >= 3 and action_values.shape[-3] == self.n_agents:
            extra_shape = action_values.shape[-2:-1]
            if prod(extra_shape) == 1:
                return action_values.reshape(
                    *action_values.shape[:-3], self.n_agents, self.n_actions
                )
        raise ValueError(
            "AVD-Net expects one discrete action dimension per agent, got "
            f"action-values with shape {action_values.shape}"
        )

    def _canonical_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape[-1] != self.n_actions:
            raise ValueError(
                f"Expected action-mask last dimension {self.n_actions}, got {mask.shape}"
            )
        if mask.shape[-2] == self.n_agents:
            return mask.to(torch.bool)
        if mask.ndim >= 3 and mask.shape[-3] == self.n_agents:
            extra_shape = mask.shape[-2:-1]
            if prod(extra_shape) == 1:
                return mask.reshape(*mask.shape[:-3], self.n_agents, self.n_actions).to(
                    torch.bool
                )
        raise ValueError(
            "AVD-Net expects one action-mask vector per agent, got "
            f"mask with shape {mask.shape}"
        )

    def _action_index(self, action: torch.Tensor) -> torch.Tensor:
        if isinstance(self.action_spec, OneHot) or (
            action.shape[-1] == self.n_actions and action.dtype.is_floating_point
        ):
            return action.argmax(dim=-1)
        index = action.to(torch.long)
        while index.ndim > 0 and index.shape[-1] == 1:
            index = index.squeeze(-1)
        if index.shape[-1] != self.n_agents:
            raise ValueError(
                f"Expected action index shape ending in {self.n_agents}, got {index.shape}"
            )
        return index

    def _chosen_action_values(
        self, action_values: torch.Tensor, action_index: torch.Tensor
    ) -> torch.Tensor:
        return torch.gather(action_values, -1, action_index.unsqueeze(-1)).squeeze(-1)

    def _greedy_actions(
        self, action_values: torch.Tensor, mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            action_values = action_values.masked_fill(
                ~mask, torch.finfo(action_values.dtype).min
            )
        action_index = action_values.argmax(dim=-1)
        values = self._chosen_action_values(action_values, action_index)
        return action_index, values

    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_function == "l1":
            return (pred - target).abs()
        if self.loss_function == "l2":
            return (pred - target).pow(2)
        return F.smooth_l1_loss(pred, target, reduction="none")

    @staticmethod
    def _match_value_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape == reference.shape:
            return value
        if value.shape == reference.shape[:-1]:
            return value.unsqueeze(-1)
        if value.shape[-1:] == (1,):
            return value.expand_as(reference)
        return value.reshape_as(reference)

    @staticmethod
    def _get_optional(tensordict: TensorDictBase, key):
        try:
            return tensordict.get(key)
        except KeyError:
            return None


class Avdnet(Algorithm):
    """AVD-Net attention value-decomposition algorithm."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_embed_dim: int,
        agent_embedding_num_cells: Sequence[int],
        attention_weight_num_cells: Sequence[int],
        head_weight_num_cells: Sequence[int],
        state_bias_num_cells: Sequence[int],
        use_previous_action: bool,
        positive_eps: float,
        delay_value: bool,
        loss_function: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_attention_heads <= 0:
            raise ValueError("AVD-Net num_attention_heads must be greater than 0")
        if attention_embed_dim <= 0:
            raise ValueError("AVD-Net attention_embed_dim must be greater than 0")
        if positive_eps <= 0:
            raise ValueError("AVD-Net positive_eps must be greater than 0")
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "AVD-Net loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_embed_dim = attention_embed_dim
        self.agent_embedding_num_cells = agent_embedding_num_cells
        self.attention_weight_num_cells = attention_weight_num_cells
        self.head_weight_num_cells = head_weight_num_cells
        self.state_bias_num_cells = state_bias_num_cells
        self.use_previous_action = use_previous_action
        self.positive_eps = positive_eps
        self.delay_value = delay_value
        self.loss_function = loss_function

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError(
                "AVD-Net is not compatible with continuous actions."
            )
        context_keys, context_shapes, context_features = self._get_context_specs(group)
        agent_feature_keys, agent_feature_shapes, agent_feature_dim = (
            self._get_agent_feature_specs(group)
        )
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        mixer = AVDNetMixer(
            context_features=context_features,
            agent_feature_dim=agent_feature_dim,
            n_agents=n_agents,
            num_attention_heads=self.num_attention_heads,
            attention_embed_dim=self.attention_embed_dim,
            agent_embedding_num_cells=self.agent_embedding_num_cells,
            attention_weight_num_cells=self.attention_weight_num_cells,
            head_weight_num_cells=self.head_weight_num_cells,
            state_bias_num_cells=self.state_bias_num_cells,
            positive_eps=self.positive_eps,
            device=self.device,
        )
        loss_module = AVDNetLoss(
            group=group,
            policy_network=policy_for_loss,
            mixer=mixer,
            context_keys=context_keys,
            context_shapes=context_shapes,
            agent_feature_keys=agent_feature_keys,
            agent_feature_shapes=agent_feature_shapes,
            action_spec=self.action_spec[group, "action"],
            n_agents=n_agents,
            n_actions=n_actions,
            gamma=self.experiment_config.gamma,
            delay_value=self.delay_value,
            loss_function=self.loss_function,
            previous_action_key=(
                (group, AVDNET_PREVIOUS_ACTION_KEY)
                if self.use_previous_action
                else None
            ),
        )
        return loss_module, self.delay_value

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {"loss": loss.parameters()}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n
        action_shape = self.action_spec[group, "action"].shape
        logits_shape = (
            [*action_shape]
            if isinstance(self.action_spec[group, "action"], OneHot)
            else [*action_shape, n_actions]
        )

        actor_group_spec = self._actor_group_input_spec(group).to(self.device)
        actor_input_spec = Composite({group: actor_group_spec})
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"action_value": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        action_mask_key = (
            (group, "action_mask") if self.action_mask_spec is not None else None
        )

        value_module = QValueModule(
            action_value_key=(group, "action_value"),
            action_mask_key=action_mask_key,
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value"),
            ],
            spec=self.action_spec[group, "action"],
            action_space=None,
        )
        return TensorDictSequential(actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        action_mask_key = (
            (group, "action_mask") if self.action_mask_spec is not None else None
        )

        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
            device=self.device,
        )
        return TensorDictSequential(*policy_for_loss, greedy)

    def process_env_fun(
        self,
        env_fun: Callable[[], EnvBase],
    ) -> Callable[[], EnvBase]:
        if not self.use_previous_action:
            return env_fun

        def wrapped_env_fun():
            env = env_fun()
            transforms = [
                _AVDNetPreviousActionTransform(
                    group=group,
                    n_agents=len(agents),
                    n_actions=self.action_spec[group, "action"].space.n,
                    action_spec=self.action_spec[group, "action"],
                )
                for group, agents in self.group_map.items()
            ]
            return TransformedEnv(env, Compose(*transforms))

        return wrapped_env_fun

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))

        done_key = ("next", "done")
        terminated_key = ("next", "terminated")
        reward_key = ("next", "reward")

        if done_key not in keys:
            batch.set(done_key, batch.get(("next", group, "done")).any(-2))
        if terminated_key not in keys:
            batch.set(terminated_key, batch.get(("next", group, "terminated")).any(-2))
        if reward_key not in keys:
            batch.set(reward_key, batch.get(("next", group, "reward")).mean(-2))
        return batch

    def _actor_group_input_spec(self, group: str) -> Composite:
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n
        group_spec = self.observation_spec[group].clone()
        if self.use_previous_action:
            group_spec.set(
                AVDNET_PREVIOUS_ACTION_KEY,
                Unbounded(shape=(*group_spec.shape, n_actions), device=self.device),
            )
        return group_spec

    def _get_context_specs(self, group: str) -> Tuple[List[Tuple], List[torch.Size], int]:
        if self.state_spec is not None:
            keys = [list(self.state_spec.keys(True, True))[0]]
            specs = [self.state_spec[keys[0]]]
            context_keys = [keys[0] if isinstance(keys[0], tuple) else (keys[0],)]
        else:
            keys = list(self.observation_spec[group].keys(True, True))
            specs = [self.observation_spec[group][key] for key in keys]
            context_keys = [_nested_key((group,), key) for key in keys]
        context_shapes = [spec.shape for spec in specs]
        context_features = sum(prod(shape) for shape in context_shapes)
        return context_keys, context_shapes, context_features

    def _get_agent_feature_specs(
        self, group: str
    ) -> Tuple[List[Tuple], List[torch.Size], int]:
        n_agents = len(self.group_map[group])
        keys = list(self.observation_spec[group].keys(True, True))
        specs = [self.observation_spec[group][key] for key in keys]
        agent_feature_keys = [_nested_key((group,), key) for key in keys]
        agent_feature_shapes = [spec.shape for spec in specs]
        if self.use_previous_action:
            n_actions = self.action_spec[group, "action"].space.n
            agent_feature_keys.append((group, AVDNET_PREVIOUS_ACTION_KEY))
            agent_feature_shapes.append(torch.Size([n_agents, n_actions]))

        for key, shape in zip(agent_feature_keys, agent_feature_shapes):
            if len(shape) == 0 or shape[0] != n_agents:
                raise ValueError(
                    "AVD-Net agent features require observation leaves with an "
                    f"agent dimension, got key {key} with shape {shape}"
                )
        agent_feature_dim = sum(prod(shape[1:]) for shape in agent_feature_shapes)
        return agent_feature_keys, agent_feature_shapes, agent_feature_dim


@dataclass
class AvdnetConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~vdmarl.algorithms.Avdnet`."""

    num_attention_heads: int = MISSING
    attention_embed_dim: int = MISSING
    agent_embedding_num_cells: Sequence[int] = MISSING
    attention_weight_num_cells: Sequence[int] = MISSING
    head_weight_num_cells: Sequence[int] = MISSING
    state_bias_num_cells: Sequence[int] = MISSING
    use_previous_action: bool = MISSING
    positive_eps: float = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING

    def __post_init__(self):
        if self.num_attention_heads <= 0:
            raise ValueError("AVD-Net num_attention_heads must be greater than 0")
        if self.attention_embed_dim <= 0:
            raise ValueError("AVD-Net attention_embed_dim must be greater than 0")
        self.positive_eps = float(self.positive_eps)
        if self.positive_eps <= 0:
            raise ValueError("AVD-Net positive_eps must be greater than 0")
        if self.loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "AVD-Net loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Avdnet

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
