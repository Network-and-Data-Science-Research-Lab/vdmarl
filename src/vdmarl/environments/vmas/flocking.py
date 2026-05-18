from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_obstacles: int = MISSING
    collision_reward: float = MISSING
