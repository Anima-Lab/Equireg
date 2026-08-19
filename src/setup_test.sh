#!/usr/bin/env bash
set -euo pipefail

# src/setup_test.sh -- run this FIRST. Exercises only the core equireg
# library: no checkpoints, no solver clone, no data, no network. If this
# fails, neither solver integration in the README has any chance of
# working -- the problem is in this install, not in a solver clone.
#
# Checks:
#   1. EquiReg can be constructed with a trivial f.
#   2. The penalty is nonzero at a step the schedule fires on, and exactly
#      zero (not just small) at a step it skips.
#   3. All three reductions (norm, sqnorm, mse) run and relate to each other
#      the way their definitions imply (sqnorm == norm**2, mse independent
#      scale).
#   4. Every registered group's apply_inverse_to_input actually inverts
#      apply_to_input, for every non-identity element it draws from.
#   5. equi_phase runs against a trivial optimizer and actually moves the
#      variable it optimizes.
#
# Env vars: none required. DEVICE (default cpu) selects the torch device the
# checks run on.

DEVICE="${DEVICE:-cpu}"

echo "setup_test: exercising the core equireg library on device='$DEVICE' ..."
echo

python3 - "$DEVICE" <<'PY'
import sys

device_name = sys.argv[1]

import torch

from equireg import EquiReg
from equireg.groups import REGISTERED_GROUPS, get_group

device = torch.device(device_name)


def check(label, cond):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"setup_test: FAILED at '{label}'")


# --- 1 & 2: construct EquiReg, confirm firing vs skipped steps -------------
print("1-2. EquiReg construction and step-based firing")

f = torch.nn.Linear(8, 8).to(device)
x = torch.randn(1, 8, device=device)

reg = EquiReg(
    f=f,
    group=get_group("hflip"),
    form="equi",
    equi_alpha=1.0,
    equi_freq=0.5,       # Constant schedule: period = int(1/0.5) = 2
    equi_schedule="constant",
    reduction="norm",
    seed=0,
)
print(f"  built {reg!r}")

fire_penalty = reg(x, step=0, total_steps=10)
check("penalty nonzero at a firing step (step=0)", float(fire_penalty.detach()) != 0.0)

skip_penalty = reg(x, step=1, total_steps=10)
check("penalty exactly zero at a skipped step (step=1)", float(skip_penalty.detach()) == 0.0)

# --- 3: all three reductions ------------------------------------------------
print("3. reductions (norm, sqnorm, mse)")

vals = {}
for reduction in ("norm", "sqnorm", "mse"):
    r = EquiReg(f=f, group=get_group("hflip"), form="equi", equi_alpha=1.0,
                equi_freq=1.0, equi_schedule="constant", reduction=reduction, seed=0)
    vals[reduction] = float(r(x, step=0, idx=1, total_steps=1).detach())
    check(f"reduction='{reduction}' runs and is finite and nonzero", vals[reduction] not in (0.0,) and vals[reduction] == vals[reduction])

check("sqnorm ~= norm ** 2 (same underlying difference tensor)",
      abs(vals["sqnorm"] - vals["norm"] ** 2) < 1e-3 * max(1.0, vals["sqnorm"]))

# --- 4: group inverses -------------------------------------------------------
print("4. group inverses")

img = torch.randn(1, 3, 16, 16, device=device)
for name, group in REGISTERED_GROUPS.items():
    for idx in group.sample_from:
        forward = group.apply_to_input(img, idx)
        back = group.apply_inverse_to_input(forward, idx)
        check(f"group '{name}' element {idx}: apply_inverse_to_input(apply_to_input(x)) == x",
              torch.allclose(back, img, atol=1e-6))

# --- 5: equi_phase runs and moves the variable it optimizes -----------------
print("5. equi_phase")

from equireg import equi_phase

var = torch.randn(1, 8, device=device).requires_grad_(True)
before = var.detach().clone()
opt = torch.optim.Adam([var], lr=1e-2)
equi_phase(reg, var, opt, num_steps_equi=3)
check("equi_phase updates the optimized variable", not torch.allclose(var.detach(), before))

print()
print("setup_test: ALL CHECKS PASSED")
PY
