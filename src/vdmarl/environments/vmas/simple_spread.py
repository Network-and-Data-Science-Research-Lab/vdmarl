from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    obs_agents: bool = MISSING
    n_agents: int = MISSING
