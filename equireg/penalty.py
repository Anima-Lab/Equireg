from __future__ import annotations

from typing import Callable, Dict

import torch

from .groups import TransformationGroup

MPEFunction = Callable[[torch.Tensor], torch.Tensor]


def _norm(d: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(d)


def _sqnorm(d: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(d) ** 2


def _mse(d: torch.Tensor) -> torch.Tensor:
    # equivalent to torch.nn.MSELoss()(a, b) for d = a - b
    return torch.mean(d**2)


REDUCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "norm": _norm,
    "sqnorm": _sqnorm,
    "mse": _mse,
}


def _reduce(d: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction not in REDUCTIONS:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Available: {list(REDUCTIONS)}"
        )
    return REDUCTIONS[reduction](d)


def equi(
    f: MPEFunction,
    group: TransformationGroup,
    x: torch.Tensor,
    idx: int,
    reduction: str = "norm",
) -> torch.Tensor:
    """‖ρ(g)f(x) − f(g·x)‖ — the equivariance error of f at x."""
    lhs = group.apply_to_feature(f(x), idx)
    rhs = f(group.apply_to_input(x, idx))
    return _reduce(lhs - rhs, reduction)


def equi_plus(
    f: MPEFunction,
    group: TransformationGroup,
    x: torch.Tensor,
    idx: int,
    reduction: str = "norm",
) -> torch.Tensor:
    """‖x − g⁻¹f(g·x)‖.

    Requires f to map back into x's own space. For PSLD, f is the decoder
    followed by the encoder, so the comparison happens in latent space.
    """
    y = f(group.apply_to_input(x, idx))
    y = group.apply_inverse_to_input(y, idx)
    return _reduce(x - y, reduction)


FORMS: Dict[str, Callable[..., torch.Tensor]] = {
    "equi": equi,
    "equi_plus": equi_plus,
}
