from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    mirror_passage: bool = MISSING
    observe_rel_pos: bool = MISSING
    done_on_completion: bool = MISSING
    final_reward: float = MISSING
