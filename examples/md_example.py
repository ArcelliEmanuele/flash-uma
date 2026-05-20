import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

seed = np.random.randint(0, np.iinfo(np.int32).max, dtype=int)
settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)

print("Loading UMA-S-1p2...")
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda", seed=seed)
model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
calc = FAIRChemCalculator(predictor, task_name="omol")

print("Setting up Water Molecule...")
atoms = molecule("H2O")
atoms.info["charge"] = 0
atoms.info["spin"] = 1  # Singlet state for H2O

atoms.set_cell([20.0, 20.0, 20.0])
atoms.center()

print("Merging MoE weights...")
setup_data = calc.a2g(atoms).to("cuda")
model = model.to('cuda')
model.prepare_for_inference(setup_data, settings)
model = model.to('cuda')

print("Swapping to FlashUMA ZeroAllocBackbone...")
model.backbone.regress_config.forces = True
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)

atoms.calc = calc

print("Configuring Langevin Dynamics...")
dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)

trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)

def print_dyn():
    print(f"Step: {dyn.get_number_of_steps():>3d} | Energy: {atoms.get_potential_energy():.4f} eV")
dyn.attach(print_dyn, interval=10)

print("Running MD for 100 steps...")
dyn.run(steps=100)
print("MD Complete! Trajectory saved to 'my_md.traj'")