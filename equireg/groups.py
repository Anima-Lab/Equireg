from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence

import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _apply_rot90(k: int) -> TensorTransform:
    def transform(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=k, dims=(-2, -1))

    return transform


def _apply_flip(dim: int) -> TensorTransform:
    def transform(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(dim,))

    return transform


def _apply_identity(x: torch.Tensor) -> torch.Tensor:
    return x


class TransformationGroup:
    """Container for the equivariant operations T_g.

    ``input_ops`` act on x, ``feature_ops`` act on f(x) — these are the two
    sides of the equivariance relation and may differ when f changes spatial
    layout. ``inverse_indices[i]`` gives the index of the inverse of element i.
    ``sample_from`` lists the elements the regularizer draws from at run time.
    """

    def __init__(
        self,
        name: str,
        input_ops: List[TensorTransform],
        feature_ops: List[TensorTransform] | None = None,
        inverse_indices: Sequence[int] | None = None,
        sample_from: Sequence[int] | None = None,
    ) -> None:
        self.name = name
        self.input_ops = input_ops
        self.feature_ops = feature_ops if feature_ops is not None else input_ops
        self.inverse_indices = (
            list(inverse_indices) if inverse_indices is not None else list(range(len(input_ops)))
        )
        self.sample_from = (
            list(sample_from) if sample_from is not None else list(range(1, len(input_ops)))
        )

    def apply_to_input(self, tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return self.input_ops[idx](tensor)

    def apply_to_feature(self, tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return self.feature_ops[idx](tensor)

    def inverse_index(self, idx: int) -> int:
        return self.inverse_indices[idx]

    def apply_inverse_to_input(self, tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return self.input_ops[self.inverse_index(idx)](tensor)

    def sample_index(self, rng: random.Random | None = None, full: bool = False) -> int:
        """Draw a group element index.

        ``full=False`` (default) draws from ``sample_from``, reproducing the
        elements the original code actually used. ``full=True`` draws uniformly
        from every non-identity element.
        """
        chooser = rng if rng is not None else random
        pool = list(range(1, self.order)) if full else self.sample_from
        return chooser.choice(pool)

    @property
    def order(self) -> int:
        return len(self.input_ops)


def _build_groups() -> Dict[str, TransformationGroup]:
    groups: Dict[str, TransformationGroup] = {}

    rot_ops: List[TensorTransform] = [
        _apply_identity,
        _apply_rot90(1),
        _apply_rot90(2),
        _apply_rot90(3),
    ]
    # inverse of rot90^k is rot90^(4-k)
    # sample_from = (1, 3): the source drew only between a clockwise and a
    # counter-clockwise quarter turn, never identity or 180 degrees.
    groups["rot90"] = TransformationGroup(
        "rot90", rot_ops, inverse_indices=(0, 3, 2, 1), sample_from=(1, 3)
    )

    flip_ops: List[TensorTransform] = [_apply_identity, _apply_flip(-1)]
    groups["hflip"] = TransformationGroup(
        "hflip", flip_ops, inverse_indices=(0, 1), sample_from=(1,)
    )

    rot180_ops: List[TensorTransform] = [_apply_identity, _apply_rot90(2)]
    groups["rot180"] = TransformationGroup(
        "rot180", rot180_ops, inverse_indices=(0, 1), sample_from=(1,)
    )

    return groups


REGISTERED_GROUPS = _build_groups()


def get_group(name: str) -> TransformationGroup:
    key = name.lower()
    if key not in REGISTERED_GROUPS:
        raise ValueError(
            f"Unknown transformation group '{name}'. Available: {list(REGISTERED_GROUPS)}"
        )
    return REGISTERED_GROUPS[key]
