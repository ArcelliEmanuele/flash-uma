import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

print("Loading UMA-S-1p2...")
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model = predictor.model.module if hasattr(predictor.model, "module") else predictor.model
calc = FAIRChemCalculator(predictor, task_name="omat")

print("Setting up Bulk Iron...")
atoms = bulk("Fe")
atoms.info["charge"] = 0
atoms.info["spin"] = 0

print("Merging MoE weights...")
setup_data = calc.a2g(atoms).to("cuda")
model = model.to('cuda')
model.prepare_for_inference(setup_data, settings)
model = model.to('cuda')

print("Swapping to FlashUMA ZeroAllocBackbone...")
model.backbone.regress_config.forces = True
model.backbone.regress_config.stress = True # CRITICAL: Required by FrechetCellFilter
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)

atoms.calc = calc

print("Starting FIRE cell relaxation...")
opt = FIRE(FrechetCellFilter(atoms))
opt.run(fmax=0.05, steps=50)