from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_targets: int = MISSING
    lidar_range: float = MISSING
    covering_range: float = MISSING
    agents_per_target: int = MISSING
    targets_respawn: bool = MISSING
    shared_reward: bool = MISSING
