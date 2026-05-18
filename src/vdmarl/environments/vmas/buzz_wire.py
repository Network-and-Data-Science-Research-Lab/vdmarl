from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    random_start_angle: bool = MISSING
    collision_reward: float = MISSING
