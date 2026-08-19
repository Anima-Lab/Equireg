# EquiReg: Equivariance Regularized Diffusion for Inverse Problems

Diffusion models represent the state-of-the-art for solving inverse problems such as image restoration tasks. Diffusion-based inverse solvers incorporate a likelihood term to guide prior sampling, generating data consistent with the posterior distribution. However, due to the intractability of the likelihood, most methods rely on isotropic Gaussian approximations, which can push estimates off the data manifold and produce inconsistent, poor reconstructions. We propose Equivariance Regularized (EquiReg) diffusion, a general plug-in framework that improves posterior sampling by penalizing trajectories that deviate from the data manifold. EquiReg formalizes manifold-preferential equivariant functions that exhibit low equivariance error for on-manifold samples and high error for off-manifold ones, thereby guiding sampling toward symmetry-preserving regions of the solution space. We highlight that such functions naturally emerge when training non-equivariant models with augmentation or on data with symmetries. EquiReg's largest gains are under reduced sampling and measurement consistency steps, where many methods suffer severe quality degradation. By regularizing trajectories toward the manifold, EquiReg implicitly accelerates convergence and enables high-quality reconstructions. EquiReg consistently improves performance in linear and nonlinear image restoration tasks and solving partial differential equations. 


## Integrating EquiReg with Inverse Solvers

To integrate EquiReg, clone the solver you want to use from its own repository and add the EquiReg term where that solver assembles its loss. 

## Install

Requires Python ≥ 3.10.

```bash
pip install -e .
```

Extras:

```bash
pip install -e ".[mpe]"   # training MPE functions, and the LDM-encoder MPE function
pip install -e ".[clip]"  # CLIP-backed MPE functions
```

`src/setup_test.sh` runs a self-contained check of the installed library —
no checkpoints, no solver clone, no data. Run it before anything else.

## Adding EquiReg to a solver

Place the following where your solver assembles the loss it differentiates.
```python
from equireg import EquiReg
from equireg.groups import get_group

reg = EquiReg(
    f=encoder,               # an MPE function: tensor in, tensor out
    group=get_group("hflip"),
    equi_alpha=0.01,
)

# Anywhere a solver assembles its data-consistency norm before differentiating:
norm = torch.linalg.norm(difference) + reg(x_0_hat, step=idx)
```

## Training MPE functions

```bash
python -m equireg.mpe.train \
  --dataset ffhq256 --data_path data/ffhq256 \
  --image_size 256 --group rot90 \
  --objective contrastive \
  --epochs 50 --batch_size 256 \
  --learning_rate 5e-4 --weight_decay 1e-4 \
  --projection_dim 256 --projection_hidden_dim 512 \
  --output_dir models/MPE --save_name ffhq256_contrastive.pt
```

See [equireg/mpe/train.py](equireg/mpe/train.py) for the full set of
objectives (`classification`, `rotation`, `equivariance`, `contrastive`) and
flags.

## Citation

If you build upon EquiReg, please consider citing:

```bibtex
@misc{tolooshams2025equiregequivarianceregularizeddiffusion,
      title={EquiReg: Equivariance Regularized Diffusion for Inverse Problems}, 
      author={Bahareh Tolooshams and Aditi Chandrashekar and Rayhan Zirvi and Abbas Mammadov and Jiachen Yao and Chuwei Wang and Anima Anandkumar},
      year={2025},
      eprint={2505.22973},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.22973}, 
}
```
