from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    energy_coeff: float = MISSING
    start_same_point: bool = MISSING
