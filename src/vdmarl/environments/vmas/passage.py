from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_passages: int = MISSING
    shared_reward: bool = MISSING
