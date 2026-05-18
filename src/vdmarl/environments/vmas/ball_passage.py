from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_passages: int = MISSING
    fixed_passage: bool = MISSING
    random_start_angle: bool = MISSING
