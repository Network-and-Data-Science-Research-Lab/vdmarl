from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, MISSING
from math import prod
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


class QPLEXMixer(nn.Module):
    """QPLEX duplex dueling mixer.

    The mixer receives decentralized action-values and a centralized context. It
    preserves the IGM constraint by using positive state/action-conditioned
    weights on the transformed local advantages.
    """

    def __init__(
        self,
        context_features: int,
        n_agents: int,
        n_actions: int,
        num_attention_heads: int,
        transformation_mlp_num_cells: Sequence[int],
        attention_mlp_num_cells: Sequence[int],
        positive_eps: float,
        stop_local_advantage_gradient: bool,
        device,
    ):
        super().__init__()
        if num_attention_heads <= 0:
            raise ValueError("QPLEX num_attention_heads must be greater than 0")
        if positive_eps <= 0:
            raise ValueError("QPLEX positive_eps must be greater than 0")

        self.context_features = context_features
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.num_attention_heads = num_attention_heads
        self.positive_eps = positive_eps
        self.stop_local_advantage_gradient = stop_local_advantage_gradient

        self.transformation = _make_mlp(
            in_features=context_features,
            hidden_cells=transformation_mlp_num_cells,
            out_features=2 * n_agents,
            device=device,
        )
        attention_features = context_features + n_agents * n_actions
        attention_out_features = num_attention_heads * n_agents
        self.lambda_net = _make_mlp(
            in_features=context_features,
            hidden_cells=attention_mlp_num_cells,
            out_features=attention_out_features,
            device=device,
        )
        self.phi_net = _make_mlp(
            in_features=attention_features,
            hidden_cells=attention_mlp_num_cells,
            out_features=attention_out_features,
            device=device,
        )
        self.key_net = _make_mlp(
            in_features=attention_features,
            hidden_cells=attention_mlp_num_cells,
            out_features=attention_out_features,
            device=device,
        )

    def forward(
        self,
        context: torch.Tensor,
        action_values: torch.Tensor,
        action_index: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chosen_action_values = torch.gather(
            action_values, dim=-1, index=action_index.unsqueeze(-1)
        ).squeeze(-1)

        if action_mask is not None:
            action_values_for_max = action_values.masked_fill(
                ~action_mask, torch.finfo(action_values.dtype).min
            )
        else:
            action_values_for_max = action_values
        max_action_values = action_values_for_max.max(dim=-1).values

        raw_transformation = self.transformation(context).reshape(
            *context.shape[:-1], self.n_agents, 2
        )
        transformation_weight = raw_transformation[..., 0].abs() + self.positive_eps
        transformation_bias = raw_transformation[..., 1]

        transformed_q = (
            transformation_weight * chosen_action_values + transformation_bias
        )
        local_advantage = chosen_action_values - max_action_values
        if self.stop_local_advantage_gradient:
            local_advantage = local_advantage.detach()
        transformed_advantage = transformation_weight * local_advantage

        lambda_weight = self._lambda_weight(context, action_index)
        q_tot = transformed_q.sum(dim=-1, keepdim=True) + (
            (lambda_weight - 1.0) * transformed_advantage
        ).sum(dim=-1, keepdim=True)
        return q_tot

    def _lambda_weight(
        self, context: torch.Tensor, action_index: torch.Tensor
    ) -> torch.Tensor:
        joint_action = F.one_hot(action_index, self.n_actions).to(
            context.dtype
        ).reshape(*action_index.shape[:-1], self.n_agents * self.n_actions)
        attention_input = torch.cat([context, joint_action], dim=-1)

        lambda_values = self.lambda_net(context).reshape(
            *context.shape[:-1], self.num_attention_heads, self.n_agents
        )
        phi_values = self.phi_net(attention_input).reshape(
            *context.shape[:-1], self.num_attention_heads, self.n_agents
        )
        key_values = self.key_net(attention_input).reshape(
            *context.shape[:-1], self.num_attention_heads, self.n_agents
        )

        attention = (
            torch.sigmoid(lambda_values)
            * torch.sigmoid(phi_values)
            * (key_values.abs() + self.positive_eps)
        )
        return attention.sum(dim=-2)


class QPLEXLoss(LossModule):
    """TD loss for QPLEX with target-policy greedy bootstrap."""

    def __init__(
        self,
        group: str,
        policy_network: TensorDictModule,
        mixer: QPLEXMixer,
        context_keys: List[Tuple],
        context_shapes: List[torch.Size],
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
                "QPLEX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )

        self.group = group
        self.context_keys = context_keys
        self.context_shapes = context_shapes
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

        context = self._context(tensordict)
        action_index = self._action_index(action)
        action_mask = self._canonical_mask(
            self._get_optional(tensordict, (self.group, "action_mask"))
        )
        q_tot = self._mix(
            context=context,
            action_values=action_values,
            action_index=action_index,
            action_mask=action_mask,
            target=False,
        )

        with torch.no_grad():
            next_td = tensordict.get("next").clone()
            next_policy_td = self._run_policy(next_td, target=True)
            next_action_values = self._canonical_action_values(
                next_policy_td.get((self.group, "action_value"))
            )
            next_action_mask = self._canonical_mask(
                self._get_optional(next_td, (self.group, "action_mask"))
            )
            next_action_index, _ = self._greedy_actions(
                next_action_values, next_action_mask
            )
            next_context = self._context(tensordict, next=True)
            next_q_tot = self._mix(
                context=next_context,
                action_values=next_action_values,
                action_index=next_action_index,
                action_mask=next_action_mask,
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
        action_values: torch.Tensor,
        action_index: torch.Tensor,
        action_mask: torch.Tensor | None,
        target: bool,
    ) -> torch.Tensor:
        with self._module_params_context("mixer", target=target):
            return self.mixer(context, action_values, action_index, action_mask)

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
            "QPLEX expects one discrete action dimension per agent, got "
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
            "QPLEX expects one action-mask vector per agent, got "
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

    def _greedy_actions(
        self, action_values: torch.Tensor, mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            action_values = action_values.masked_fill(
                ~mask, torch.finfo(action_values.dtype).min
            )
        action_index = action_values.argmax(dim=-1)
        values = torch.gather(action_values, -1, action_index.unsqueeze(-1)).squeeze(-1)
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


class Qplex(Algorithm):
    """QPLEX duplex dueling value decomposition."""

    def __init__(
        self,
        num_attention_heads: int,
        transformation_mlp_num_cells: Sequence[int],
        attention_mlp_num_cells: Sequence[int],
        delay_value: bool,
        loss_function: str,
        positive_eps: float,
        stop_local_advantage_gradient: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_attention_heads <= 0:
            raise ValueError("QPLEX num_attention_heads must be greater than 0")
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QPLEX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if positive_eps <= 0:
            raise ValueError("QPLEX positive_eps must be greater than 0")

        self.num_attention_heads = num_attention_heads
        self.transformation_mlp_num_cells = transformation_mlp_num_cells
        self.attention_mlp_num_cells = attention_mlp_num_cells
        self.delay_value = delay_value
        self.loss_function = loss_function
        self.positive_eps = positive_eps
        self.stop_local_advantage_gradient = stop_local_advantage_gradient

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("QPLEX is not compatible with continuous actions.")
        context_keys, context_shapes, context_features = self._get_context_specs(group)
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        mixer = QPLEXMixer(
            context_features=context_features,
            n_agents=n_agents,
            n_actions=n_actions,
            num_attention_heads=self.num_attention_heads,
            transformation_mlp_num_cells=self.transformation_mlp_num_cells,
            attention_mlp_num_cells=self.attention_mlp_num_cells,
            positive_eps=self.positive_eps,
            stop_local_advantage_gradient=self.stop_local_advantage_gradient,
            device=self.device,
        )
        loss_module = QPLEXLoss(
            group=group,
            policy_network=policy_for_loss,
            mixer=mixer,
            context_keys=context_keys,
            context_shapes=context_shapes,
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
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

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
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

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


@dataclass
class QplexConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~vdmarl.algorithms.Qplex`."""

    num_attention_heads: int = MISSING
    transformation_mlp_num_cells: Sequence[int] = MISSING
    attention_mlp_num_cells: Sequence[int] = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING
    positive_eps: float = MISSING
    stop_local_advantage_gradient: bool = MISSING

    def __post_init__(self):
        if self.num_attention_heads <= 0:
            raise ValueError("QPLEX num_attention_heads must be greater than 0")
        if self.loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QPLEX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        self.positive_eps = float(self.positive_eps)
        if self.positive_eps <= 0:
            raise ValueError("QPLEX positive_eps must be greater than 0")

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qplex

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
