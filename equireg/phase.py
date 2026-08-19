from __future__ import annotations

import torch

from .equireg import EquiReg

DEFAULT_NUM_STEPS_EQUI = 30


def equi_phase(
    reg: EquiReg,
    variable: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_steps_equi: int = DEFAULT_NUM_STEPS_EQUI,
    total_steps: int | None = None,
) -> None:
    """Minimize the EquiReg penalty alone over ``variable`` — a second,
    sequential optimization phase run after a solver's own data-consistency
    loop has converged.

    ``variable`` must require grad and be the tensor ``optimizer`` owns. The
    step index the schedule sees is this phase's own iteration counter, so
    ``total_steps`` defaults to the phase length; pass it explicitly only to
    compute a cutoff window against a different horizon.

    Steps where the schedule returns an alpha of zero are skipped entirely
    (no backward, no optimizer step) rather than stepping on a zero loss.
    """
    steps = total_steps if total_steps is not None else num_steps_equi
    for step in range(num_steps_equi):
        optimizer.zero_grad()
        penalty = reg(variable, step=step, total_steps=steps)
        if float(penalty.detach()) == 0.0:
            continue
        penalty.backward(retain_graph=True)
        optimizer.step()
