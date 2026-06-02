from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, MISSING
from math import prod, sqrt
from typing import Dict, Iterable, List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Composite, OneHot, Unbounded
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import LossModule

from vdmarl.algorithms.common import Algorithm, AlgorithmConfig
from vdmarl.models.common import ModelConfig


QATTEN_VARIANTS = {"base", "weighted"}
_POSITIVE_EPS = 1e-10


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


class QAttenMixer(nn.Module):
    """Attention-based monotonic mixer for QAtten."""

    def __init__(
        self,
        context_features: int,
        agent_feature_dim: int,
        n_agents: int,
        num_attention_heads: int,
        attention_embed_dim: int,
        query_embedding_num_cells: Sequence[int],
        key_embedding_num_cells: Sequence[int],
        head_weight_num_cells: Sequence[int],
        constant_value_num_cells: Sequence[int],
        variant: str,
        include_local_q_in_keys: bool,
        device,
    ):
        super().__init__()
        if variant not in QATTEN_VARIANTS:
            raise ValueError(
                f"QAtten variant must be one of {QATTEN_VARIANTS}, got {variant}"
            )
        if num_attention_heads <= 0:
            raise ValueError("QAtten num_attention_heads must be greater than 0")
        if attention_embed_dim <= 0:
            raise ValueError("QAtten attention_embed_dim must be greater than 0")

        self.context_features = context_features
        self.agent_feature_dim = agent_feature_dim
        self.n_agents = n_agents
        self.num_attention_heads = num_attention_heads
        self.attention_embed_dim = attention_embed_dim
        self.variant = variant
        self.include_local_q_in_keys = include_local_q_in_keys

        query_out_features = num_attention_heads * attention_embed_dim
        key_in_features = agent_feature_dim + int(include_local_q_in_keys)
        key_out_features = num_attention_heads * attention_embed_dim

        self.query_net = _make_mlp(
            in_features=context_features,
            hidden_cells=query_embedding_num_cells,
            out_features=query_out_features,
            device=device,
        )
        self.key_net = _make_mlp(
            in_features=key_in_features,
            hidden_cells=key_embedding_num_cells,
            out_features=key_out_features,
            device=device,
        )
        self.head_weight_net = _make_mlp(
            in_features=context_features,
            hidden_cells=head_weight_num_cells,
            out_features=num_attention_heads,
            device=device,
        )
        self.constant_net = _make_mlp(
            in_features=context_features,
            hidden_cells=constant_value_num_cells,
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
        attention = self.attention_weights(context, agent_features, local_values)
        head_q_values = (attention * local_values.unsqueeze(-2)).sum(dim=-1)
        if self.variant == "weighted":
            head_q_values = self.head_weights(context) * head_q_values
        return head_q_values.sum(dim=-1, keepdim=True) + self.constant_net(context)

    def attention_weights(
        self,
        context: torch.Tensor,
        agent_features: torch.Tensor,
        local_values: torch.Tensor,
    ) -> torch.Tensor:
        context = context.to(local_values.dtype)
        agent_features = agent_features.to(local_values.dtype)
        if self.include_local_q_in_keys:
            agent_features = torch.cat(
                [agent_features, local_values.unsqueeze(-1)], dim=-1
            )

        query = self.query_net(context).reshape(
            *context.shape[:-1], self.num_attention_heads, self.attention_embed_dim
        )
        key = self.key_net(agent_features).reshape(
            *agent_features.shape[:-2],
            self.n_agents,
            self.num_attention_heads,
            self.attention_embed_dim,
        )
        key = key.movedim(-3, -2)

        logits = (query.unsqueeze(-2) * key).sum(dim=-1) / sqrt(
            self.attention_embed_dim
        )
        return logits.softmax(dim=-1)

    def head_weights(self, context: torch.Tensor) -> torch.Tensor:
        return self.head_weight_net(context).abs() + _POSITIVE_EPS


class QAttenLoss(LossModule):
    """TD loss for QAtten with target-policy greedy bootstrap."""

    def __init__(
        self,
        group: str,
        policy_network: TensorDictModule,
        mixer: QAttenMixer,
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
    ):
        super().__init__()
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QAtten loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
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
            "QAtten expects one discrete action dimension per agent, got "
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
            "QAtten expects one action-mask vector per agent, got "
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


class Qatten(Algorithm):
    """QAtten attention-based value decomposition."""

    def __init__(
        self,
        variant: str,
        num_attention_heads: int,
        query_embedding_num_cells: Sequence[int],
        key_embedding_num_cells: Sequence[int],
        head_weight_num_cells: Sequence[int],
        constant_value_num_cells: Sequence[int],
        attention_embed_dim: int,
        include_local_q_in_keys: bool,
        delay_value: bool,
        loss_function: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if variant not in QATTEN_VARIANTS:
            raise ValueError(
                f"QAtten variant must be one of {QATTEN_VARIANTS}, got {variant}"
            )
        if num_attention_heads <= 0:
            raise ValueError("QAtten num_attention_heads must be greater than 0")
        if attention_embed_dim <= 0:
            raise ValueError("QAtten attention_embed_dim must be greater than 0")
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QAtten loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

        self.variant = variant
        self.num_attention_heads = num_attention_heads
        self.query_embedding_num_cells = query_embedding_num_cells
        self.key_embedding_num_cells = key_embedding_num_cells
        self.head_weight_num_cells = head_weight_num_cells
        self.constant_value_num_cells = constant_value_num_cells
        self.attention_embed_dim = attention_embed_dim
        self.include_local_q_in_keys = include_local_q_in_keys
        self.delay_value = delay_value
        self.loss_function = loss_function

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("QAtten is not compatible with continuous actions.")
        context_keys, context_shapes, context_features = self._get_context_specs(group)
        agent_feature_keys, agent_feature_shapes, agent_feature_dim = (
            self._get_agent_feature_specs(group)
        )
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        mixer = QAttenMixer(
            context_features=context_features,
            agent_feature_dim=agent_feature_dim,
            n_agents=n_agents,
            num_attention_heads=self.num_attention_heads,
            attention_embed_dim=self.attention_embed_dim,
            query_embedding_num_cells=self.query_embedding_num_cells,
            key_embedding_num_cells=self.key_embedding_num_cells,
            head_weight_num_cells=self.head_weight_num_cells,
            constant_value_num_cells=self.constant_value_num_cells,
            variant=self.variant,
            include_local_q_in_keys=self.include_local_q_in_keys,
            device=self.device,
        )
        loss_module = QAttenLoss(
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

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
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
        for key, shape in zip(keys, agent_feature_shapes):
            if len(shape) == 0 or shape[0] != n_agents:
                raise ValueError(
                    "QAtten agent features require observation leaves with an "
                    f"agent dimension, got key {key} with shape {shape}"
                )
        agent_feature_dim = sum(prod(shape[1:]) for shape in agent_feature_shapes)
        return agent_feature_keys, agent_feature_shapes, agent_feature_dim


@dataclass
class QattenConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~vdmarl.algorithms.Qatten`."""

    variant: str = MISSING
    num_attention_heads: int = MISSING
    query_embedding_num_cells: Sequence[int] = MISSING
    key_embedding_num_cells: Sequence[int] = MISSING
    head_weight_num_cells: Sequence[int] = MISSING
    constant_value_num_cells: Sequence[int] = MISSING
    attention_embed_dim: int = MISSING
    include_local_q_in_keys: bool = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING

    def __post_init__(self):
        if self.variant not in QATTEN_VARIANTS:
            raise ValueError(
                f"QAtten variant must be one of {QATTEN_VARIANTS}, got {self.variant}"
            )
        if self.num_attention_heads <= 0:
            raise ValueError("QAtten num_attention_heads must be greater than 0")
        if self.attention_embed_dim <= 0:
            raise ValueError("QAtten attention_embed_dim must be greater than 0")
        if self.loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QAtten loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qatten

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
