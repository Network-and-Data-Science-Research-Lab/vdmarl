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
from torchrl.modules import EGreedyModule, QMixer, QValueModule
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


class WQMIXMixer(nn.Module):
    """Flattened-context wrapper around the TorchRL QMIX monotonic mixer."""

    def __init__(
        self,
        context_features: int,
        mixing_embed_dim: int,
        n_agents: int,
        device,
    ):
        super().__init__()
        self.mixer = QMixer(
            state_shape=(context_features,),
            mixing_embed_dim=mixing_embed_dim,
            n_agents=n_agents,
            device=device,
        )

    def forward(self, context: torch.Tensor, local_values: torch.Tensor) -> torch.Tensor:
        if local_values.shape[-1] != 1:
            local_values_unsqueezed = local_values.unsqueeze(-1)
        else:
            local_values_unsqueezed = local_values
        value = self.mixer(local_values_unsqueezed, context)
        if value.shape == local_values_unsqueezed.shape[:-1]:
            value = value.unsqueeze(-1)
        return value


class QStarNetwork(nn.Module):
    """Unrestricted centralized joint-action value network for W-QMIX."""

    def __init__(
        self,
        context_features: int,
        n_agents: int,
        n_actions: int,
        hidden_cells: Sequence[int],
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.net = _make_mlp(
            in_features=context_features + n_agents * n_actions,
            hidden_cells=hidden_cells,
            out_features=1,
            device=device,
        )

    def forward(self, context: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
        joint_action = F.one_hot(action_index, self.n_actions).to(
            context.dtype
        ).reshape(*action_index.shape[:-1], self.n_agents * self.n_actions)
        return self.net(torch.cat([context, joint_action], dim=-1))


class WQMIXLoss(LossModule):
    """Weighted QMIX loss with OW-QMIX and CW-QMIX weighting variants."""

    def __init__(
        self,
        group: str,
        policy_network: TensorDictModule,
        qtot_mixer: WQMIXMixer,
        qstar_network: QStarNetwork,
        context_keys: List[Tuple],
        context_shapes: List[torch.Size],
        action_spec,
        n_agents: int,
        n_actions: int,
        gamma: float,
        delay_value: bool,
        loss_function: str,
        variant: str,
        alpha: float,
    ):
        super().__init__()
        if variant not in ("ow", "cw"):
            raise ValueError(f"W-QMIX variant must be 'ow' or 'cw', got {variant}")
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "W-QMIX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if not 0 < alpha <= 1:
            raise ValueError("W-QMIX alpha must be in the interval (0, 1]")

        self.group = group
        self.context_keys = context_keys
        self.context_shapes = context_shapes
        self.action_spec = action_spec
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.delay_value = delay_value
        self.loss_function = loss_function
        self.variant = variant
        self.alpha = alpha

        self.convert_to_functional(
            policy_network,
            "policy_network",
            create_target_params=delay_value,
        )
        self.convert_to_functional(
            qtot_mixer,
            "qtot_mixer",
            create_target_params=delay_value,
        )
        self.convert_to_functional(
            qstar_network,
            "qstar_network",
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
        local_chosen_values = self._chosen_action_values(action_values, action_index)
        q_tot = self._qtot(context, local_chosen_values, target=False)
        q_star = self._qstar(context, action_index, target=False)

        current_mask = self._canonical_mask(
            self._get_optional(tensordict, (self.group, "action_mask"))
        )
        current_greedy_index, _ = self._greedy_actions(
            action_values, current_mask
        )
        with torch.no_grad():
            current_greedy_q_star = self._qstar(
                context, current_greedy_index, target=False
            )

        with torch.no_grad():
            next_td = tensordict.get("next").clone()
            next_policy_td = self._run_policy(next_td, target=True)
            next_action_values = self._canonical_action_values(
                next_policy_td.get((self.group, "action_value"))
            )
            next_mask = self._canonical_mask(
                self._get_optional(next_td, (self.group, "action_mask"))
            )
            next_greedy_index, _ = self._greedy_actions(next_action_values, next_mask)
            next_context = self._context(tensordict, next=True)
            next_q_star = self._qstar(next_context, next_greedy_index, target=True)
            reward = self._match_value_shape(
                tensordict.get(("next", "reward")), next_q_star
            )
            terminated = self._match_value_shape(
                tensordict.get(("next", "terminated")), next_q_star
            )
            target = reward + self.gamma * (
                1 - terminated.to(next_q_star.dtype)
            ) * next_q_star

        weights = self._weights(
            q_tot=q_tot,
            target=target,
            q_star=q_star,
            action_index=action_index,
            current_greedy_index=current_greedy_index,
            current_greedy_q_star=current_greedy_q_star,
        )
        loss_qtot = (weights * self._distance(q_tot, target)).mean()
        loss_qstar = self._distance(q_star, target).mean()
        loss = loss_qtot + loss_qstar

        td_error = (q_tot - target).detach().abs().squeeze(-1)
        tensordict.set((self.group, "td_error"), td_error.unsqueeze(-1).expand(*td_error.shape, self.n_agents))

        return TensorDict(
            {
                "loss": loss,
                "loss_qtot": loss_qtot,
                "loss_qstar": loss_qstar,
                "weight_mean": weights.detach().mean(),
                "td_error": td_error.mean(),
            },
            batch_size=[],
        )

    def _weights(
        self,
        q_tot: torch.Tensor,
        target: torch.Tensor,
        q_star: torch.Tensor,
        action_index: torch.Tensor,
        current_greedy_index: torch.Tensor,
        current_greedy_q_star: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.variant == "ow":
                full_weight = q_tot < target
            else:
                is_current_greedy = (action_index == current_greedy_index).all(
                    dim=-1, keepdim=True
                )
                target_above_greedy = target > current_greedy_q_star
                full_weight = is_current_greedy | target_above_greedy
            return torch.where(
                full_weight,
                torch.ones_like(q_star),
                torch.full_like(q_star, self.alpha),
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

    def _qtot(
        self, context: torch.Tensor, local_values: torch.Tensor, target: bool
    ) -> torch.Tensor:
        with self._module_params_context("qtot_mixer", target=target):
            return self.qtot_mixer(context, local_values)

    def _qstar(
        self, context: torch.Tensor, action_index: torch.Tensor, target: bool
    ) -> torch.Tensor:
        with self._module_params_context("qstar_network", target=target):
            return self.qstar_network(context, action_index)

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
            "W-QMIX expects one discrete action dimension per agent, got "
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
            "W-QMIX expects one action-mask vector per agent, got "
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


class Wqmix(Algorithm):
    """Weighted QMIX with OW-QMIX and CW-QMIX weighting variants."""

    def __init__(
        self,
        variant: str,
        mixing_embed_dim: int,
        qstar_mlp_num_cells: Sequence[int],
        alpha: float,
        delay_value: bool,
        loss_function: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if variant not in ("ow", "cw"):
            raise ValueError(f"W-QMIX variant must be 'ow' or 'cw', got {variant}")
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "W-QMIX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if not 0 < alpha <= 1:
            raise ValueError("W-QMIX alpha must be in the interval (0, 1]")

        self.variant = variant
        self.mixing_embed_dim = mixing_embed_dim
        self.qstar_mlp_num_cells = qstar_mlp_num_cells
        self.alpha = alpha
        self.delay_value = delay_value
        self.loss_function = loss_function

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("W-QMIX is not compatible with continuous actions.")
        context_keys, context_shapes, context_features = self._get_context_specs(group)
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        qtot_mixer = WQMIXMixer(
            context_features=context_features,
            mixing_embed_dim=self.mixing_embed_dim,
            n_agents=n_agents,
            device=self.device,
        )
        qstar_network = QStarNetwork(
            context_features=context_features,
            n_agents=n_agents,
            n_actions=n_actions,
            hidden_cells=self.qstar_mlp_num_cells,
            device=self.device,
        )
        loss_module = WQMIXLoss(
            group=group,
            policy_network=policy_for_loss,
            qtot_mixer=qtot_mixer,
            qstar_network=qstar_network,
            context_keys=context_keys,
            context_shapes=context_shapes,
            action_spec=self.action_spec[group, "action"],
            n_agents=n_agents,
            n_actions=n_actions,
            gamma=self.experiment_config.gamma,
            delay_value=self.delay_value,
            loss_function=self.loss_function,
            variant=self.variant,
            alpha=self.alpha,
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
class WqmixConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~vdmarl.algorithms.Wqmix`."""

    variant: str = MISSING
    mixing_embed_dim: int = MISSING
    qstar_mlp_num_cells: Sequence[int] = MISSING
    alpha: float = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING

    def __post_init__(self):
        if self.variant not in ("ow", "cw"):
            raise ValueError(
                f"W-QMIX variant must be either 'ow' or 'cw', got {self.variant}"
            )
        if self.loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "W-QMIX loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if not 0 < self.alpha <= 1:
            raise ValueError("W-QMIX alpha must be in the interval (0, 1]")

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Wqmix

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
