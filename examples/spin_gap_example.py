import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from wrapper import ZeroAllocBackbone

settings = InferenceSettings(
    execution_mode="umas_fast_gpu", merge_mole=True,
    external_graph_gen=True, use_quaternion_wigner=True, activation_checkpointing=False
)

# ==========================================
# 1. Singlet Calculation
# ==========================================
print("Loading UMA-S-1p2 for Singlet...")
pred_singlet = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model_s = pred_singlet.model.module if hasattr(pred_singlet.model, "module") else pred_singlet.model
calc_s = FAIRChemCalculator(pred_singlet, task_name="omol")

print("Setting up Singlet CH2...")
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})

# --- FIX: Add a dummy vacuum box so pymatgen doesn't crash! ---
singlet.set_cell([20.0, 20.0, 20.0])
singlet.center()
# --------------------------------------------------------------

# Trigger MOLE merge for Singlet (generates spin=1 weights)
print("Merging MoE weights for Singlet (spin=1)...")
model_s = model_s.to('cuda')
model_s.prepare_for_inference(calc_s.a2g(singlet).to("cuda"), settings)
model_s = model_s.to('cuda')

# Swap backbone
print("Swapping to FlashUMA ZeroAllocBackbone...")
model_s.backbone.regress_config.forces = True 
model_s.backbone = ZeroAllocBackbone(model_s.backbone, dtype=torch.float32)

singlet.calc = calc_s
e_singlet = singlet.get_potential_energy()
print(f"--> Singlet Energy: {e_singlet:.4f} eV\n")


# ==========================================
# 2. Triplet Calculation
# ==========================================
print("Loading fresh UMA-S-1p2 for Triplet...")
# MUST load a fresh predictor because the previous one had its MOLE layers 
# permanently merged and optimized specifically for the singlet state!
pred_triplet = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model_t = pred_triplet.model.module if hasattr(pred_triplet.model, "module") else pred_triplet.model
calc_t = FAIRChemCalculator(pred_triplet, task_name="omol")

print("Setting up Triplet CH2...")
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})

# --- FIX: Add a dummy vacuum box so pymatgen doesn't crash! ---
triplet.set_cell([20.0, 20.0, 20.0])
triplet.center()
# --------------------------------------------------------------

# Trigger MOLE merge for Triplet (generates spin=3 weights)
print("Merging MoE weights for Triplet (spin=3)...")
model_t = model_t.to('cuda')
model_t.prepare_for_inference(calc_t.a2g(triplet).to("cuda"), settings)
model_t = model_t.to('cuda')
# Swap backbone
print("Swapping to FlashUMA ZeroAllocBackbone...")
model_t.backbone.regress_config.forces = True
model_t.backbone = ZeroAllocBackbone(model_t.backbone, dtype=torch.float32)

triplet.calc = calc_t
e_triplet = triplet.get_potential_energy()
print(f"--> Triplet Energy: {e_triplet:.4f} eV\n")


# ==========================================
# 3. Spin Gap Result
# ==========================================
spin_gap = e_triplet - e_singlet
print("==========================================")
print(f"Spin Gap (Triplet - Singlet): {spin_gap:.4f} eV")
print("==========================================")