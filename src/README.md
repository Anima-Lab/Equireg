## Examples
### PSLD

```bash
git clone https://github.com/LituRout/PSLD && cd PSLD
```

Set up the regularizer:
```python
from equireg import EquiReg

reg = EquiReg.from_config("src/psld/equireg.yaml", total_steps=50)  # match your DDIM step count
reg.f = model.differentiable_decode_first_stage                     
```

Then, in `ldm/models/diffusion/psld.py`, inside `p_sample_ddim`:

```python
error = error + reg(pred_z_0, step=iteration)
```


### SITCOM

```bash
git clone https://github.com/sjames40/SITCOM && cd SITCOM
```

Set up the regularizer:
```python
from equireg import EquiReg, equi_phase
reg = EquiReg.from_config("src/sitcom/equireg.yaml")
```

Then, at the end of `optimize_input` (in `SITCOM_with_noise.py`):

```python
if reg.equi_alpha > 0:
    pred_original_sample = pred_original_sample.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pred_original_sample], lr=learning_rate)
    equi_phase(reg, pred_original_sample, optimizer)
return input_tensor, pred_original_sample
```

### Regularizers/ Args

- `equi`: `‖ρ(g)f(x) − f(g·x)‖` — the equivariance error of `f` at `x` under
  group element `g`.
- `equi_plus`: `‖x − g⁻¹f(g·x)‖` — requires `f` to map back into `x`'s own
  space (e.g. a decoder-then-encoder composition where `x` is a latent).

| Arg | Description |
|-----|-------------|
| `equi_alpha` | The penalty's weight; multiplies the raw equivariance error before it is added to the solver's loss/norm. |
| `equi_freq` | How often the penalty fires; interpreted by the `Schedule`. `constant`/`cutoff` fire every `int(1/equi_freq)` steps. |
| `equi_schedule` | Which `Schedule` maps a step index to an effective `equi_alpha`: `constant`, `cutoff`, or a `legacy_*` schedule (see below). |
| `group` | The `TransformationGroup` supplying `T_g`: `hflip`, `rot90`, `rot180`. |
| `form` | `equi` or `equi_plus`, selecting which equivariance relation is penalized. |
| `reduction` | How the equivariance error tensor is collapsed to a scalar: `norm`, `sqnorm`, or `mse`. |
