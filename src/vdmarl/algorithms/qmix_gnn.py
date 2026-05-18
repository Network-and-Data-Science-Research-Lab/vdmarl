from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from math import prod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Composite, OneHot, Unbounded
from torchrl.modules import EGreedyModule, QMixer, QValueModule
from torchrl.objectives import LossModule, QMixerLoss, ValueEstimators

from vdmarl.algorithms.common import Algorithm, AlgorithmConfig
from vdmarl.models.common import ModelConfig
from torch_geometric.nn import GATv2Conv


_QMIX_GNN_AGENT_INPUT = "qmix_gnn_agent_input"
_QMIX_GNN_TEAM_INFO = "qmix_gnn_team_info"
_GRAPH_TOPOLOGIES = {"full", "knn", "radius"}


def _nested_key(prefix, key):
    key = key if isinstance(key, tuple) else (key,)
    return (*prefix, *key)


class InformationInfusionModule(nn.Module):
    """Graph information infusion module used by QMIX-GNN actors."""

    def __init__(
        self,
        observation_shapes: Sequence[torch.Size],
        n_agents: int,
        projection_dim: int,
        gnn_hidden_dim: int,
        num_attention_heads: int,
        gnn_num_layers: int,
        gnn_dropout: float,
        graph_topology: str,
        position_key_index: Optional[int],
        knn_k: int,
        edge_radius: Optional[float],
        self_loops: bool,
        device,
    ):
        super().__init__()
        if graph_topology not in _GRAPH_TOPOLOGIES:
            raise ValueError(
                f"QMIX-GNN graph_topology must be one of {_GRAPH_TOPOLOGIES}, "
                f"got {graph_topology}"
            )
        if graph_topology in ("knn", "radius") and position_key_index is None:
            raise ValueError(
                "QMIX-GNN graph_topology='knn' or 'radius' requires position_key"
            )
        if graph_topology == "radius" and edge_radius is None:
            raise ValueError("QMIX-GNN graph_topology='radius' requires edge_radius")
        if graph_topology == "knn" and knn_k <= 0:
            raise ValueError("QMIX-GNN knn_k must be positive")
        if gnn_num_layers <= 0:
            raise ValueError("QMIX-GNN gnn_num_layers must be positive")

        self.observation_shapes = list(observation_shapes)
        self.n_agents = n_agents
        self.projection_dim = projection_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.graph_topology = graph_topology
        self.position_key_index = position_key_index
        self.knn_k = knn_k
        self.edge_radius = edge_radius
        self.self_loops = self_loops
        self.gnn_dropout = gnn_dropout

        input_features = sum(self._features_from_shape(shape) for shape in observation_shapes)
        self.projection = nn.Linear(input_features, projection_dim, device=device)

        gnns: List[nn.Module] = []
        in_channels = projection_dim
        for _ in range(gnn_num_layers):
            gnns.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=gnn_hidden_dim,
                    heads=num_attention_heads,
                    concat=False,
                    dropout=gnn_dropout,
                    add_self_loops=False,
                ).to(device)
            )
            in_channels = gnn_hidden_dim
        self.gnns = nn.ModuleList(gnns)

    def forward(self, *observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(observations) != len(self.observation_shapes):
            raise ValueError(
                f"QMIX-GNN expected {len(self.observation_shapes)} observation tensors, "
                f"got {len(observations)}"
            )

        flattened = [
            self._reshape_agent_value(value, shape)
            for value, shape in zip(observations, self.observation_shapes)
        ]
        local_observation = torch.cat(flattened, dim=-1).to(
            self.projection.weight.dtype
        )
        batch_shape = local_observation.shape[:-2]
        batch_size = prod(batch_shape) if len(batch_shape) else 1

        projected = self.projection(local_observation)
        x = projected.reshape(batch_size * self.n_agents, self.projection_dim)

        if self.graph_topology == "full":
            edge_index = self._full_edge_index(batch_size, x.device)
        else:
            position = self._reshape_agent_value(
                observations[self.position_key_index],
                self.observation_shapes[self.position_key_index],
            ).reshape(batch_size, self.n_agents, -1)
            position = position.to(projected.dtype)
            if self.graph_topology == "knn":
                edge_index = self._knn_edge_index(position)
            else:
                edge_index = self._radius_edge_index(position)

        for i, gnn in enumerate(self.gnns):
            x = gnn(x, edge_index)
            if i != len(self.gnns) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.gnn_dropout, training=self.training)

        fused = x.reshape(*batch_shape, self.n_agents, self.gnn_hidden_dim)
        team_info = fused.mean(dim=-2)
        expanded_team_info = team_info.unsqueeze(-2).expand(
            *batch_shape, self.n_agents, self.gnn_hidden_dim
        )
        agent_input = torch.cat([projected, expanded_team_info], dim=-1)
        return agent_input, team_info

    def _reshape_agent_value(
        self, value: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        event_ndim = len(shape)
        if event_ndim == 0:
            raise ValueError("QMIX-GNN observation leaves must include an agent dimension")
        return value.reshape(*value.shape[:-event_ndim], self.n_agents, -1)

    def _full_edge_index(self, batch_size: int, device) -> torch.Tensor:
        agents = torch.arange(self.n_agents, device=device)
        source = agents.repeat_interleave(self.n_agents)
        target = agents.repeat(self.n_agents)
        if not self.self_loops:
            mask = source != target
            source = source[mask]
            target = target[mask]
        edge_index = torch.stack([source, target])
        return self._repeat_edge_index(edge_index, batch_size)

    def _knn_edge_index(self, position: torch.Tensor) -> torch.Tensor:
        batch_size = position.shape[0]
        k = min(self.knn_k, self.n_agents if self.self_loops else self.n_agents - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=position.device)

        distance = torch.cdist(position, position)
        if not self.self_loops:
            index = torch.arange(self.n_agents, device=position.device)
            distance[:, index, index] = float("inf")

        source = distance.topk(k, largest=False, dim=-1).indices
        target = torch.arange(self.n_agents, device=position.device).view(1, -1, 1)
        target = target.expand_as(source)
        offset = (
            torch.arange(batch_size, device=position.device).view(-1, 1, 1)
            * self.n_agents
        )
        return torch.stack(
            [(source + offset).reshape(-1), (target + offset).reshape(-1)]
        )

    def _radius_edge_index(self, position: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(position, position)
        mask = distance <= self.edge_radius
        if not self.self_loops:
            index = torch.arange(self.n_agents, device=position.device)
            mask[:, index, index] = False

        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=position.device)
        batch, target, source = nonzero.unbind(-1)
        offset = batch * self.n_agents
        return torch.stack([source + offset, target + offset])

    def _repeat_edge_index(
        self, edge_index: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        n_edges = edge_index.shape[1]
        if n_edges == 0:
            return edge_index
        offsets = (
            torch.arange(batch_size, device=edge_index.device).repeat_interleave(n_edges)
            * self.n_agents
        )
        return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)

    @staticmethod
    def _features_from_shape(shape: torch.Size) -> int:
        if len(shape) <= 1:
            return 1
        return prod(shape[1:])


class QmixGnn(Algorithm):
    """QMIX-GNN with graph information infusion before the local Q networks."""

    def __init__(
        self,
        mixing_embed_dim: int,
        delay_value: bool,
        loss_function: str,
        projection_dim: int,
        gnn_hidden_dim: int,
        num_attention_heads: int,
        gnn_num_layers: int,
        gnn_dropout: float,
        graph_topology: str,
        position_key: Optional[str],
        knn_k: int,
        edge_radius: Optional[float],
        self_loops: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QMIX-GNN loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if graph_topology not in _GRAPH_TOPOLOGIES:
            raise ValueError(
                f"QMIX-GNN graph_topology must be one of {_GRAPH_TOPOLOGIES}, "
                f"got {graph_topology}"
            )

        self.mixing_embed_dim = mixing_embed_dim
        self.delay_value = delay_value
        self.loss_function = loss_function
        self.projection_dim = projection_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.gnn_num_layers = gnn_num_layers
        self.gnn_dropout = gnn_dropout
        self.graph_topology = graph_topology
        self.position_key = position_key
        self.knn_k = knn_k
        self.edge_radius = edge_radius
        self.self_loops = self_loops

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("QMIX-GNN is not compatible with continuous actions.")

        loss_module = QMixerLoss(
            policy_for_loss,
            self.get_mixer(group),
            delay_value=self.delay_value,
            loss_function=self.loss_function,
            action_space=self.action_spec[group, "action"],
        )
        loss_module.set_keys(
            reward="reward",
            action=(group, "action"),
            done="done",
            terminated="terminated",
            action_value=(group, "action_value"),
            local_value=(group, "chosen_action_value"),
            global_value="chosen_action_value",
            priority="td_error",
        )
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {"loss": loss.parameters()}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            raise NotImplementedError("QMIX-GNN is not compatible with continuous actions.")

        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n
        action_shape = self.action_spec[group, "action"].shape
        logits_shape = (
            [*action_shape]
            if isinstance(self.action_spec[group, "action"], OneHot)
            else [*action_shape, n_actions]
        )

        observation_keys, observation_shapes, position_key_index = (
            self._get_observation_specs(group)
        )
        information_infusion = TensorDictModule(
            module=InformationInfusionModule(
                observation_shapes=observation_shapes,
                n_agents=n_agents,
                projection_dim=self.projection_dim,
                gnn_hidden_dim=self.gnn_hidden_dim,
                num_attention_heads=self.num_attention_heads,
                gnn_num_layers=self.gnn_num_layers,
                gnn_dropout=self.gnn_dropout,
                graph_topology=self.graph_topology,
                position_key_index=position_key_index,
                knn_k=self.knn_k,
                edge_radius=self.edge_radius,
                self_loops=self.self_loops,
                device=self.device,
            ),
            in_keys=observation_keys,
            out_keys=[
                (group, _QMIX_GNN_AGENT_INPUT),
                (group, _QMIX_GNN_TEAM_INFO),
            ],
        )

        agent_input_dim = self.projection_dim + self.gnn_hidden_dim
        actor_input_spec = Composite(
            {
                group: Composite(
                    {
                        _QMIX_GNN_AGENT_INPUT: Unbounded(
                            shape=(n_agents, agent_input_dim),
                            device=self.device,
                        )
                    },
                    shape=(n_agents,),
                )
            }
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

        return TensorDictSequential(information_infusion, actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            raise NotImplementedError("QMIX-GNN is not compatible with continuous actions.")

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

    def get_mixer(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        if self.state_spec is not None:
            global_state_key = list(self.state_spec.keys(True, True))[0]
            state_shape = self.state_spec[global_state_key].shape
            in_keys = [(group, "chosen_action_value"), global_state_key]
        else:
            state_shape = (self.gnn_hidden_dim,)
            in_keys = [
                (group, "chosen_action_value"),
                (group, _QMIX_GNN_TEAM_INFO),
            ]

        mixer = TensorDictModule(
            module=QMixer(
                state_shape=state_shape,
                mixing_embed_dim=self.mixing_embed_dim,
                n_agents=n_agents,
                device=self.device,
            ),
            in_keys=in_keys,
            out_keys=["chosen_action_value"],
        )
        return mixer

    def _get_observation_specs(
        self, group: str
    ) -> Tuple[List[Tuple], List[torch.Size], Optional[int]]:
        n_agents = len(self.group_map[group])
        observation_leaf_keys = list(self.observation_spec[group].keys(True, True))
        observation_keys = [_nested_key((group,), key) for key in observation_leaf_keys]
        observation_shapes = [
            self.observation_spec[group][key].shape for key in observation_leaf_keys
        ]

        position_key_index = None
        for i, (key, shape) in enumerate(zip(observation_leaf_keys, observation_shapes)):
            if len(shape) == 0 or shape[0] != n_agents:
                raise ValueError(
                    "QMIX-GNN observations must have an agent dimension as their "
                    f"first event dimension, got key {key} with shape {shape}"
                )
            key_tuple = key if isinstance(key, tuple) else (key,)
            if self.position_key is not None and key_tuple[-1] == self.position_key:
                position_key_index = i

        return observation_keys, observation_shapes, position_key_index


@dataclass
class QmixGnnConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~vdmarl.algorithms.QmixGnn`."""

    mixing_embed_dim: int = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING
    projection_dim: int = MISSING
    gnn_hidden_dim: int = MISSING
    num_attention_heads: int = MISSING
    gnn_num_layers: int = MISSING
    gnn_dropout: float = MISSING
    graph_topology: str = MISSING
    position_key: Optional[str] = MISSING
    knn_k: int = MISSING
    edge_radius: Optional[float] = MISSING
    self_loops: bool = MISSING

    def __post_init__(self):
        if self.loss_function not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                "QMIX-GNN loss_function must be one of 'l1', 'l2' or 'smooth_l1'"
            )
        if self.graph_topology not in _GRAPH_TOPOLOGIES:
            raise ValueError(
                f"QMIX-GNN graph_topology must be one of {_GRAPH_TOPOLOGIES}, "
                f"got {self.graph_topology}"
            )
        if self.projection_dim <= 0:
            raise ValueError("QMIX-GNN projection_dim must be positive")
        if self.gnn_hidden_dim <= 0:
            raise ValueError("QMIX-GNN gnn_hidden_dim must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("QMIX-GNN num_attention_heads must be positive")
        if self.gnn_num_layers <= 0:
            raise ValueError("QMIX-GNN gnn_num_layers must be positive")
        if not 0 <= self.gnn_dropout < 1:
            raise ValueError("QMIX-GNN gnn_dropout must be in the interval [0, 1)")
        if self.graph_topology == "knn" and self.knn_k <= 0:
            raise ValueError("QMIX-GNN knn_k must be positive")
        if self.graph_topology == "radius" and self.edge_radius is None:
            raise ValueError("QMIX-GNN graph_topology='radius' requires edge_radius")

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        if path is None:
            path = (
                Path(__file__).parent.parent
                / "conf"
                / "algorithm"
                / "qmix_gnn.yaml"
            )
        return super().get_from_yaml(path=str(path))

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return QmixGnn

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
