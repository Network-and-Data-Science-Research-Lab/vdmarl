from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_food: int = MISSING
    share_reward: bool = MISSING
    food_radius: float = MISSING
    penalise_by_time: bool = MISSING
