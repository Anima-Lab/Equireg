from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional

import torch

from .penalty import FORMS
from .groups import TransformationGroup, get_group
from .schedules import Schedule, get_schedule, schedule_kwarg_names

MPEFunction = Callable[[torch.Tensor], torch.Tensor]

#: Sentinel accepted in the `f:` field of an EquiReg config, meaning "the live
#: solver's own decoder, bound onto reg.f by the user once the solver model
#: exists". Used by PSLD, whose original penalised the decoder applied to a
#: latent; a decoder cannot be loaded from an mpe_config file because it only
#: exists on the running solver model.
SOLVER_DECODER = "solver_decoder"


class EquiReg:
    """The EquiReg penalty, ready to add to a solver's data-consistency loss.

    Composes three pieces: f (an MPE function), a TransformationGroup supplying
    T_g, and a Schedule controlling equi_alpha over steps.

        reg = EquiReg(f=encoder, group=get_group("hflip"), equi_alpha=0.01)
        norm = torch.linalg.norm(difference) + reg(x_0_hat, step=idx)

    Calling returns a 0-d tensor already scaled by the effective equi_alpha, and
    a zero tensor when the schedule skips this step, so callers never branch.
    """

    def __init__(
        self,
        f: Optional[MPEFunction],
        group: TransformationGroup | str,
        form: str = "equi",
        equi_alpha: float = 0.01,
        equi_freq: float = 0.1,
        equi_schedule: str = "constant",
        reduction: str = "norm",
        total_steps: int | None = None,
        seed: int | None = None,
        **schedule_kwargs: Any,
    ) -> None:
        if form not in FORMS:
            raise ValueError(f"Unknown form '{form}'. Available: {list(FORMS)}")
        # Reject unknown keys rather than absorbing them into the schedule's
        # **kwargs. EquiReg(..., equi_alpah=0.5) used to be accepted with alpha
        # left at its default — a silent no-op, the exact failure mode this
        # library exists to surface.
        allowed = schedule_kwarg_names(equi_schedule)
        unknown = sorted(set(schedule_kwargs) - allowed)
        if unknown:
            known = [
                "f", "group", "form", "equi_alpha", "equi_freq", "equi_schedule",
                "reduction", "total_steps", "seed",
            ]
            raise ValueError(
                f"Unknown EquiReg argument(s) {unknown}. Valid arguments: {known}"
                + (f", plus {sorted(allowed)} for equi_schedule='{equi_schedule}'."
                   if allowed else f". equi_schedule='{equi_schedule}' takes no extra keywords.")
            )
        # f may be None only when the solver's decoder is bound later (see
        # SOLVER_DECODER); calling the penalty before that happens raises.
        self.f = f
        self.group = get_group(group) if isinstance(group, str) else group
        self.form = form
        self._form_fn = FORMS[form]
        self.equi_alpha = equi_alpha
        self.equi_freq = equi_freq
        self.reduction = reduction
        self.total_steps = total_steps
        self.schedule: Schedule = get_schedule(
            equi_schedule, equi_alpha=equi_alpha, equi_freq=equi_freq, **schedule_kwargs
        )
        self._rng = random.Random(seed) if seed is not None else random

    def alpha_at(self, step: int, total_steps: int | None = None) -> float:
        resolved = total_steps if total_steps is not None else self.total_steps
        if resolved is None and getattr(self.schedule, "requires_total_steps", False):
            # Cutoff.alpha and LegacyPSLDCutoff.alpha return early when
            # total_steps is None, so a cutoff schedule with no step count
            # would degrade silently to `constant` — exactly the class of
            # no-op this library exists to prevent.
            raise ValueError(
                f"equi_schedule={type(self.schedule).__name__} needs total_steps, "
                "but neither EquiReg(total_steps=...) nor this call supplied one. "
                "Without it the cutoff window can never be computed and the "
                "schedule would silently behave like 'constant'. Pass the number "
                "of solver steps in one sampling run."
            )
        return self.schedule.alpha(step, resolved)

    def __call__(
        self,
        x: torch.Tensor,
        step: int = 0,
        total_steps: int | None = None,
        idx: int | None = None,
    ) -> torch.Tensor:
        alpha = self.alpha_at(step, total_steps)
        if alpha == 0.0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        if self.f is None:
            raise RuntimeError(
                "EquiReg.f is None: this EquiReg was built with f='solver_decoder' "
                "(or f=None) and the solver's decoder has not been bound yet. Set "
                "reg.f = model.differentiable_decode_first_stage once the solver "
                "model exists, or construct EquiReg with an explicit f."
            )
        element = idx if idx is not None else self.group.sample_index(self._rng)
        penalty = self._form_fn(self.f, self.group, x, element, self.reduction)
        return alpha * penalty

    def __repr__(self) -> str:
        return (
            f"EquiReg(form={self.form!r}, group={self.group.name!r}, "
            f"equi_alpha={self.equi_alpha}, equi_freq={self.equi_freq}, "
            f"reduction={self.reduction!r}, schedule={self.schedule!r})"
        )

    @classmethod
    def from_config(cls, path: str, device: torch.device | None = None, **overrides: Any) -> "EquiReg":
        """Build from a YAML file.

        The config says where `f` comes from, in exactly one of two ways.

        1. `mpe_config` — a path to an MPE config as consumed by
           load_mpe_from_config, resolved relative to this file. Used by
           SITCOM, which penalises an MPE encoder applied to a pixel-space
           x_0_hat:

               mpe_config: ../../configs/mpe/ffhq_ldm_encoder.yaml
               group: hflip
               form: equi
               equi_alpha: 0.01
               equi_freq: 0.1
               equi_schedule: constant
               reduction: norm

        2. `f: solver_decoder` — no `mpe_config`. Used by PSLD, whose
           original computed the penalty with
           `model.differentiable_decode_first_stage` applied to a latent. That
           decoder exists only on the live solver model, so it cannot be
           loaded from a file; bind it yourself once the model exists
           (`reg.f = model.differentiable_decode_first_stage`). Until then
           `reg.f` is None and calling the penalty raises.
        """
        from pathlib import Path

        import yaml

        from .mpe.models import load_mpe_from_config

        config_file = Path(path).expanduser().resolve()
        with open(config_file, "r", encoding="utf-8") as handle:
            spec: Dict[str, Any] = yaml.safe_load(handle) or {}
        spec.update(overrides)

        mpe_config = spec.pop("mpe_config", None)
        f_spec = spec.pop("f", None)

        if mpe_config is not None and f_spec is not None:
            raise ValueError(
                "EquiReg config sets both 'mpe_config' and 'f'; they are two "
                "mutually exclusive ways of saying where f comes from."
            )
        if mpe_config is None:
            if f_spec is None:
                raise ValueError(
                    "EquiReg config requires either an 'mpe_config' field or "
                    f"'f: {SOLVER_DECODER}'."
                )
            if callable(f_spec):
                return cls(f=f_spec, **spec)
            if f_spec != SOLVER_DECODER:
                raise ValueError(
                    f"Unsupported 'f' value {f_spec!r} in an EquiReg config. The "
                    f"only supported string is '{SOLVER_DECODER}', meaning the live "
                    "solver's own differentiable_decode_first_stage, which you bind "
                    "onto reg.f yourself once the solver model exists."
                )
            return cls(f=None, **spec)

        mpe_path = Path(mpe_config).expanduser()
        if not mpe_path.is_absolute():
            mpe_path = (config_file.parent / mpe_path).resolve()
        f, _ = load_mpe_from_config(str(mpe_path), device=device)
        return cls(f=f, **spec)
