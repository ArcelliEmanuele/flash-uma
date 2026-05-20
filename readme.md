***

# FlashUMA

Fast inference for [UMA](https://huggingface.co/facebook/UMA) and eSCN models. Drop-in backbone replacement that executes custom Triton kernels and avoids materializing intermediate tensors to HBM — up to **2× faster** and **2.5× less peak VRAM** versus the FAIRChem GPU backend, with numerically equivalent outputs.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-2.1%2B-00A6D6)](https://github.com/openai/triton)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


![Benchmark](images/benchmark.png)

*`uma-s-1p2` · TF32 enabled · `torch.compile(dynamic=True)`*

## Installation

Requires a CUDA GPU, PyTorch ≥ 2.0, Triton ≥ 2.1, and `fairchem-core`.

```bash
pip install fairchem-core triton>=2.1.0
git clone https://github.com/JabekAceus/FlashUMA.git
cd FlashUMA
```

## Quick Start & Examples

`ZeroAllocBackbone` wraps an existing FAIRChem backbone and requires no changes to the surrounding ASE workflow. 

**Important Integration Rules:**
1. **MOLE (Mixture of Experts) Merging:** You must call `prepare_for_inference` on your exact system *before* wrapping the backbone. This ensures the extracted weights are perfectly optimized for your system's elemental composition.
2. **Device Casting:** The model must be explicitly cast to CUDA before and after `prepare_for_inference`.
3. **Isolated Molecules:** When using `external_graph_gen=True`, isolated molecules must be placed inside a dummy cell (e.g. `atoms.set_cell([20, 20, 20])`) to prevent `pymatgen` from crashing.

---

### 1. Relax an adsorbate on a catalytic surface
```python
import torch
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Setup system
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
add_adsorbate(slab, molecule("CO"), 2.0, "bridge")
slab.info["charge"] = 0
slab.info["spin"] = 0

# Trigger MOLE merge for this specific composition
setup_data = calc.a2g(slab).to("cuda")
model = model.to("cuda")
model.prepare_for_inference(setup_data, settings)
model = model.to("cuda")

# Swap to FlashUMA backbone
model.backbone.regress_config.forces = True
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)
# model = torch.compile(model, dynamic=True) # Optional: Maximize performance

slab.calc = calc
opt = LBFGS(slab)
opt.run(0.05, 100)
```

### 2. Relax an inorganic crystal (Cell Relaxation)
*Requires enabling stress on the backbone for the `FrechetCellFilter`.*

```python
import torch
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.info["charge"] = 0
atoms.info["spin"] = 0

setup_data = calc.a2g(atoms).to("cuda")
model = model.to("cuda")
model.prepare_for_inference(setup_data, settings)
model = model.to("cuda")

model.backbone.regress_config.forces = True
model.backbone.regress_config.stress = True
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)

atoms.calc = calc
opt = FIRE(FrechetCellFilter(atoms))
opt.run(fmax=0.05, steps=50)
```

### 3. Run Molecular Dynamics (MD)
*Caching the graph topology in FlashUMA makes MD incredibly fast.*

```python
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("H2O")
atoms.info["charge"] = 0
atoms.info["spin"] = 1 
atoms.set_cell([20.0, 20.0, 20.0])
atoms.center()

setup_data = calc.a2g(atoms).to("cuda")
model = model.to("cuda")
model.prepare_for_inference(setup_data, settings)
model = model.to("cuda")

model.backbone.regress_config.forces = True
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)

atoms.calc = calc
dyn = Langevin(atoms, timestep=0.1 * units.fs, temperature_K=400, friction=0.001 / units.fs)
dyn.run(steps=100)
```

### 4. Calculate a spin gap
*We must load a fresh model for each state because their MOLE mixing coefficients (optimized during `prepare_for_inference`) differ based on spin.*

```python
import torch
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)

def get_energy(system_name, spin):
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
    model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
    calc = FAIRChemCalculator(predictor, task_name="omol")
    
    atoms = molecule(system_name)
    atoms.info.update({"spin": spin, "charge": 0})
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.center()

    model = model.to("cuda")
    model.prepare_for_inference(calc.a2g(atoms).to("cuda"), settings)
    model = model.to("cuda")
    
    model.backbone.regress_config.forces = True 
    model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)
    atoms.calc = calc
    return atoms.get_potential_energy()

e_singlet = get_energy("CH2_s1A1d", spin=1)
e_triplet = get_energy("CH2_s3B1d", spin=3)

print(f"Singlet: {e_singlet:.4f} eV | Triplet: {e_triplet:.4f} eV")
print(f"Spin Gap: {e_triplet - e_singlet:.4f} eV")
```

## Numerical precision

Errors relative to the FAIRChem GPU backend (600-atom system, FP32 weights):

|                | TF32 off              |                       | TF32 on               |                       |
|----------------|----------------------:|-----------------------:|----------------------:|-----------------------:|
|                | MaxAbs                | MaxRel                | MaxAbs                | MaxRel                |
| Energy         | 4.9 × 10⁻³           | 8.4 × 10⁻⁷            | 1.4 × 10⁻²           | 2.4 × 10⁻⁶            |
| Forces         | 7.3 × 10⁻⁴           | 6.9 × 10⁻³            | 2.9 × 10⁻²           | —                     |
| Stress         | 1.2 × 10⁻²           | 4.6 × 10⁻⁵            | 8.2 × 10⁻²           | 4.5 × 10⁻⁴            |

*Forces MaxRel with TF32 is dominated by near-zero denominator components and is not a useful metric; absolute error remains within typical MD tolerances.*

## How it works

The memory reduction comes from never writing intermediate tensors — Wigner-D matrices, radial embeddings, geometric frames — to HBM. All intermediate values are computed on-the-fly inside fused Triton kernels and kept in SRAM registers. The key kernels are:

- **Geometry + Wigner precomputation**: analytical axis-angle rotations and 34 Wigner-D polynomials computed from raw edge coordinates without staging to global memory.
- **RotateScatter**: M→L rotation fused with the scatter-add reduction, using FP64 accumulation in registers to suppress rounding error during large reductions.
- **LayerNorm + SiLU** and **split gating**: fused pointwise passes that avoid separate activation materializations.

For molecular dynamics, CSR topology (row pointers and column indices) is cached across steps so graph structure is not re-derived when atom positions change but connectivity does not. Morton reordering at the start of each forward pass improves memory locality for the scatter operations.

## License

MIT. Pre-trained UMA weights retain their original license as defined by the FAIRChem repository.
