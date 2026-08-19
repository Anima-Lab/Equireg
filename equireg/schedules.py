from __future__ import annotations

import inspect
from typing import Dict, Set

DEFAULT_CUTOFF_STEPS = 100


class Schedule:
    """Maps a step index to an effective equi_alpha.

    Returning 0.0 means the penalty is skipped at this step.
    """

    #: True for schedules whose alpha() is meaningless without a total_steps.
    #: EquiReg consults this so that selecting a cutoff schedule without a
    #: step count is an error rather than a silent degradation to `constant`.
    requires_total_steps: bool = False

    def __init__(self, equi_alpha: float, equi_freq: float = 1.0, **kwargs) -> None:
        self.equi_alpha = equi_alpha
        self.equi_freq = equi_freq

    def alpha(self, step: int, total_steps: int | None = None) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(equi_alpha={self.equi_alpha}, equi_freq={self.equi_freq})"


class Constant(Schedule):
    """Fires every int(1/equi_freq) steps. The corrected general cadence.

    Truncating (int) rather than rounding is deliberate: the spec, the README
    and this docstring all say ``int(1/equi_freq)``, and the two disagree at
    e.g. equi_freq=0.6 (int -> 1, round -> 2). The documented form wins.
    """

    def _period(self) -> int:
        if self.equi_freq <= 0:
            return 1
        return max(1, int(1.0 / self.equi_freq))

    def alpha(self, step: int, total_steps: int | None = None) -> float:
        return self.equi_alpha if step % self._period() == 0 else 0.0


class Cutoff(Constant):
    """Constant, but zero for the final ``cutoff_steps`` steps."""

    requires_total_steps = True

    def __init__(self, equi_alpha: float, equi_freq: float = 1.0,
                 cutoff_steps: int = DEFAULT_CUTOFF_STEPS, **kwargs) -> None:
        super().__init__(equi_alpha, equi_freq)
        self.cutoff_steps = cutoff_steps

    def alpha(self, step: int, total_steps: int | None = None) -> float:
        if total_steps is not None and step > total_steps - self.cutoff_steps:
            return 0.0
        return super().alpha(step, total_steps)


class LegacyPSLD(Schedule):
    """PSLD's default: the penalty fires on every iteration.

    $SRC/src/psld/ldm/models/diffusion/psld_equi.py:450 applies a cutoff only
    when equi_schedule == 'cutoff'; otherwise alpha is constant throughout.
    """

    def alpha(self, step: int, total_steps: int | None = None) -> float:
        return self.equi_alpha


class LegacyPSLDCutoff(Schedule):
    """PSLD's equi_schedule='cutoff': alpha -> 0 once
    ``iteration > total_steps - 100``."""

    requires_total_steps = True

    def __init__(self, equi_alpha: float, equi_freq: float = 1.0,
                 cutoff_steps: int = DEFAULT_CUTOFF_STEPS, **kwargs) -> None:
        super().__init__(equi_alpha, equi_freq)
        self.cutoff_steps = cutoff_steps

    def alpha(self, step: int, total_steps: int | None = None) -> float:
        if total_steps is not None and step > total_steps - self.cutoff_steps:
            return 0.0
        return self.equi_alpha


SCHEDULES: Dict[str, type] = {
    "constant": Constant,
    "cutoff": Cutoff,
    "legacy_psld": LegacyPSLD,
    "legacy_psld_cutoff": LegacyPSLDCutoff,
}


def schedule_kwarg_names(name: str) -> Set[str]:
    """The extra keyword arguments a named schedule legitimately accepts.

    Every Schedule subclass declares ``**kwargs`` so that a shared construction
    call works for all of them, which means a typo would otherwise be absorbed
    in silence. This reads the concrete named parameters off the class's own
    __init__ instead, so callers can reject anything else.
    """
    key = name.lower()
    if key not in SCHEDULES:
        raise ValueError(f"Unknown equi_schedule '{name}'. Available: {list(SCHEDULES)}")
    params = inspect.signature(SCHEDULES[key].__init__).parameters
    return {
        p.name
        for p in params.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    } - {"self", "equi_alpha", "equi_freq"}


def get_schedule(name: str, equi_alpha: float, equi_freq: float = 1.0, **kwargs) -> Schedule:
    key = name.lower()
    if key not in SCHEDULES:
        raise ValueError(f"Unknown equi_schedule '{name}'. Available: {list(SCHEDULES)}")
    allowed = schedule_kwarg_names(key)
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keyword argument(s) {unknown} for equi_schedule '{name}'. "
            f"This schedule accepts: {sorted(allowed) or 'no extra keywords'}. "
            "Unknown keys are rejected rather than ignored: a silently absorbed "
            "typo would leave the setting you meant at its default."
        )
    return SCHEDULES[key](equi_alpha=equi_alpha, equi_freq=equi_freq, **kwargs)
