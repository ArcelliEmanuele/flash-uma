import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

from wrapper import ZeroAllocBackbone

# 1. Setup Inference Settings exactly as your Triton backend expects
settings = InferenceSettings(
    execution_mode="umas_fast_gpu",
    merge_mole=True,
    external_graph_gen=True, 
    use_quaternion_wigner=True,
    activation_checkpointing=False
)

# 2. Load official predictor
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings=settings, device="cuda")
model = predictor.model
if hasattr(model, "module"): 
    model = model.module

# 3. Create the calculator
calc = FAIRChemCalculator(predictor, task_name="oc20")

# 4. Create the system FIRST
print("Setting up the system...")
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

# Explicitly set charge and spin
slab.info["charge"] = 0
slab.info["spin"] = 0

# 5. Trigger MOLE merge using the ACTUAL system
# This guarantees the MoE weights are perfectly mixed for Cu + C + O
print("Merging MoE weights based on system composition...")
setup_data = calc.a2g(slab).to("cuda")
model = model.to('cuda')
model.prepare_for_inference(setup_data, settings)
model = model.to('cuda')
# 6. Swap the official backbone for your FlashUMA Triton backbone
print("Swapping to FlashUMA ZeroAllocBackbone...")
model.backbone.regress_config.forces = True
model.backbone = ZeroAllocBackbone(model.backbone, dtype=torch.float32)

# (Optional)
# model = torch.compile(model, dynamic=True)

# 7. Run your ASE Simulation!
print("Starting ASE optimization with FlashUMA backbone...")
slab.calc = calc

opt = LBFGS(slab)
opt.run(0.05, 100)