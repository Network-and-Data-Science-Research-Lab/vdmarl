from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    map_size: int = MISSING
    minimap_mode: bool = MISSING
    tag_penalty: float = MISSING
    max_cycles: int = MISSING
    extra_features: bool = MISSING
