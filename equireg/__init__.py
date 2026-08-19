from __future__ import annotations

from .equireg import EquiReg
from .penalty import FORMS, REDUCTIONS, equi, equi_plus
from .groups import REGISTERED_GROUPS, TransformationGroup, get_group
from .phase import equi_phase
from .schedules import SCHEDULES, Schedule, get_schedule

__all__ = [
    "EquiReg",
    "equi_phase",
    "FORMS",
    "REDUCTIONS",
    "equi",
    "equi_plus",
    "REGISTERED_GROUPS",
    "TransformationGroup",
    "get_group",
    "SCHEDULES",
    "Schedule",
    "get_schedule",
]


def __getattr__(name: str):
    # load_mpe_from_config lives behind the optional 'mpe' extra; expose it
    # lazily so importing equireg never requires omegaconf/torchvision.
    if name == "load_mpe_from_config":
        from .mpe.models import load_mpe_from_config

        return load_mpe_from_config
    raise AttributeError(f"module 'equireg' has no attribute {name!r}")
