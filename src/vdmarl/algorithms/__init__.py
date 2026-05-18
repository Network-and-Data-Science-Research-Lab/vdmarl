from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .avdnet import Avdnet, AvdnetConfig
from .qatten import Qatten, QattenConfig
from .qmix import Qmix, QmixConfig
from .qmix_gnn import QmixGnn, QmixGnnConfig
from .qplex import Qplex, QplexConfig
from .qtran import Qtran, QtranConfig
from .transmix import Transmix, TransmixConfig
from .vdn import Vdn, VdnConfig
from .wqmix import Wqmix, WqmixConfig

classes = [
    "Avdnet",
    "AvdnetConfig",
    "Qatten",
    "QattenConfig",
    "Qmix",
    "QmixConfig",
    "QmixGnn",
    "QmixGnnConfig",
    "Qplex",
    "QplexConfig",
    "Qtran",
    "QtranConfig",
    "Transmix",
    "TransmixConfig",
    "Vdn",
    "VdnConfig",
    "Wqmix",
    "WqmixConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "avdnet": AvdnetConfig,
    "qatten": QattenConfig,
    "qmix": QmixConfig,
    "qmix_gnn": QmixGnnConfig,
    "qplex": QplexConfig,
    "qtran": QtranConfig,
    "transmix": TransmixConfig,
    "vdn": VdnConfig,
    "wqmix": WqmixConfig,
}
