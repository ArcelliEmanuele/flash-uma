import torch
import torch.nn as nn
import torch.nn.functional as F

from model import eSCN_Inference_Model

def validate_atoms_data(self, atoms, task_name):
        """Sets default charge and spin so ASE doesn't crash if they are missing."""
        if "charge" not in atoms.info: 
            atoms.info["charge"] = 0
        if "spin" not in atoms.info: 
            atoms.info["spin"] = 1 if task_name == "omol" else 0

def on_predict_check(self, data):
    """Bypass the official consistency checking."""
    pass

def prepare_for_inference(self, data, settings):
    """Return self since your backbone doesn't need dynamic graph alterations."""
    return self

@torch.no_grad()
def transfer_weights(off_model, cust_model):
    """Copies parameters from official eSCNMDBackbone to Custom eSCN."""
    C = cust_model.C
    
    # 1. Global Embeddings
    cust_model.W_sphere.copy_(off_model.sphere_embedding.weight)
    cust_model.W_src.copy_(off_model.source_embedding.weight)
    cust_model.W_tgt.copy_(off_model.target_embedding.weight)
    
    # 2. Initialization Radial (Phase 1)
    rad_init = off_model.edge_degree_embedding.rad_func.net
    cust_model.W_rad1_1.copy_(rad_init[0].weight); cust_model.b_rad1_1.copy_(rad_init[0].bias)
    cust_model.gamma_rad1_1.copy_(rad_init[1].weight); cust_model.beta_rad1_1.copy_(rad_init[1].bias)
    
    cust_model.W_rad1_2.copy_(rad_init[3].weight); cust_model.b_rad1_2.copy_(rad_init[3].bias)
    cust_model.gamma_rad1_2.copy_(rad_init[4].weight); cust_model.beta_rad1_2.copy_(rad_init[4].bias)
    
    cust_model.W_rad1_3.copy_(rad_init[6].weight); cust_model.b_rad1_3.copy_(rad_init[6].bias)
    
    # 3. Layer Loop
    for off_blk, cust_blk in zip(off_model.blocks, cust_model.layers):
        so2_1 = off_blk.edge_wise.so2_conv_1
        so2_2 = off_blk.edge_wise.so2_conv_2
        
        # Radial MLP
        rad = so2_1.rad_func.net
        cust_blk.W_rad1_1.copy_(rad[0].weight); cust_blk.b_rad1_1.copy_(rad[0].bias)
        cust_blk.gamma_rad1_1.copy_(rad[1].weight); cust_blk.beta_rad1_1.copy_(rad[1].bias)
        
        cust_blk.W_rad1_2.copy_(rad[3].weight); cust_blk.b_rad1_2.copy_(rad[3].bias)
        cust_blk.gamma_rad1_2.copy_(rad[4].weight); cust_blk.beta_rad1_2.copy_(rad[4].bias)
        
        cust_blk.W_rad1_3.copy_(rad[6].weight); cust_blk.b_rad1_3.copy_(rad[6].bias)
        
        # SO(2) Conv 1
        cust_blk.W1_m0.copy_(so2_1.fc_m0.weight); cust_blk.b1_m0.copy_(so2_1.fc_m0.bias)
        
        w_m1 = so2_1.so2_m_conv[0].fc.weight
        cust_blk.W1_m1_R.copy_(w_m1[:2*C, :]); cust_blk.W1_m1_I.copy_(w_m1[2*C:, :])
        w_m2 = so2_1.so2_m_conv[1].fc.weight
        cust_blk.W1_m2_R.copy_(w_m2[:1*C, :]); cust_blk.W1_m2_I.copy_(w_m2[1*C:, :])
        
        # SO(2) Conv 2
        cust_blk.W2_m0.copy_(so2_2.fc_m0.weight); cust_blk.b2_m0.copy_(so2_2.fc_m0.bias)
        
        w_m1 = so2_2.so2_m_conv[0].fc.weight
        cust_blk.W2_m1_R.copy_(w_m1[:2*C, :]); cust_blk.W2_m1_I.copy_(w_m1[2*C:, :])
        w_m2 = so2_2.so2_m_conv[1].fc.weight
        cust_blk.W2_m2_R.copy_(w_m2[:1*C, :]); cust_blk.W2_m2_I.copy_(w_m2[1*C:, :])

        # Attach the Pointwise Nodes directly to our Custom block!
        cust_blk.norm_1 = off_blk.norm_1
        cust_blk.norm_2 = off_blk.norm_2
        cust_blk.atom_wise = off_blk.atom_wise
    
    # 4. Final Norm
    cust_model.final_norm = off_model.norm

    # 5. Extract Channel Balancing Boundaries
    cust_model.charge_start = off_model.charge_channel_start
    cust_model.charge_end = off_model.charge_channel_end
    cust_model.spin_start = off_model.spin_channel_start
    cust_model.spin_end = off_model.spin_channel_end


class ZeroAllocBackbone(nn.Module):
    """
    Drop-in replacement for the FairChem eSCNMDBackbone.
    Intercepts the data object and feeds it efficiently into the custom kernels.

    Inference only. ``prepare_for_inference`` repacks and deletes the original
    parameters and sets ``requires_grad=False``, so the wrapped model cannot be
    fine-tuned or trained after construction. The backward pass in ``ops/`` is
    used internally for autograd-based force/stress (dE/dpos, dE/dcell) but is
    not validated for parameter gradients.

    Only lmax == 2 is supported: every Triton kernel is hardcoded for the
    9-channel (lmax=2) spherical harmonic layout.
    """
    def __init__(self, off_bb, dtype=torch.float32):
        super().__init__()

        # Kernels assume lmax == 2 (9 spherical channels); a backbone with a
        # different lmax would silently produce wrong numbers, so reject it
        # here rather than at some opaque point inside a kernel.
        lmax = getattr(off_bb, "lmax", None)
        if lmax is not None and lmax != 2:
            raise ValueError(
                f"ZeroAllocBackbone only supports lmax=2 (Triton kernels are "
                f"hardcoded for the 9-channel layout); got lmax={lmax}."
            )
        mmax = getattr(off_bb, "mmax", None)
        if mmax is not None and mmax != 2:
            raise ValueError(
                f"ZeroAllocBackbone only supports mmax=2; got mmax={mmax}."
            )

        device = next(off_bb.parameters()).device
        
        B = off_bb.num_distance_basis
        cutoff = off_bb.cutoff
        offset_g = torch.linspace(0.0, cutoff, B)
        coeff_g = -0.5 / (2.0 * (offset_g[1] - offset_g[0])) ** 2
        
        self.ce = eSCN_Inference_Model(
            num_layers=off_bb.num_layers,
            sphere_channels=off_bb.sphere_channels,
            edge_channels=off_bb.edge_channels,
            num_distance_basis=B,
            cutoff=cutoff,
            coeff_g=coeff_g.item()
        ).to(device)
        
        transfer_weights(off_bb, self.ce)
        self.ce.prepare_for_inference(dtype=dtype)
        
        self.charge_embedding = off_bb.charge_embedding
        self.spin_embedding = off_bb.spin_embedding
        self.dataset_embedding = getattr(off_bb, "dataset_embedding", None)
        self.mix_csd = off_bb.mix_csd
        self.use_dataset_embedding = getattr(off_bb, "use_dataset_embedding", False)
        
        # Inherit regress config (Stress enabled)
        self.regress_config = off_bb.regress_config

    def validate_atoms_data(self, atoms, task_name):
        """Sets default charge and spin so ASE doesn't crash if they are missing."""
        if "charge" not in atoms.info: 
            atoms.info["charge"] = 0
        if "spin" not in atoms.info: 
            atoms.info["spin"] = 1 if task_name == "omol" else 0

    def on_predict_check(self, data):
        """Bypass the official MOLE consistency checking."""
        pass

    def prepare_for_inference(self, data, settings):
        """Return self since your backbone doesn't need dynamic graph alterations."""
        return self

    def csd_embedding(self, charge, spin, dataset=None):
        chg_emb = self.charge_embedding(charge)
        spin_emb = self.spin_embedding(spin)
        if self.use_dataset_embedding and dataset is not None:
            dataset_emb = self.dataset_embedding(dataset)
            return F.silu(self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1)))
        return F.silu(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def forward(self, data):
        # 0. Satisfy FAIRChem heads by injecting expected attributes 
        data.atomic_numbers = data.atomic_numbers.long()
        data.atomic_numbers_full = data.atomic_numbers
        data.batch_full = data.batch

        # --- NEW: Enable gradients for autograd forces/stress ---
        if not self.regress_config.direct_forces:
            if getattr(self.regress_config, 'forces', False) or getattr(self.regress_config, 'stress', False):
                data.pos.requires_grad_(True)
            if getattr(self.regress_config, 'stress', False):
                # Only enable cell grad if the cell exists and is a float tensor
                if data.cell is not None and data.cell.is_floating_point():
                    data.cell.requires_grad_(True)
        # --------------------------------------------------------

        # 1. System Embeddings
        csd_mixed_emb = self.csd_embedding(
            charge=data.charge, spin=data.spin, dataset=getattr(data, 'dataset', None)
        )
        
        # 2. Extract shifts directly from Jacobian-tracked cells
        if len(data.natoms) == 1:
            shifts = data.cell_offsets.to(data.cell.dtype) @ data.cell.squeeze(0)
        else:
            cell_per_edge = data.cell.repeat_interleave(data.nedges, dim=0)
            shifts = torch.einsum("ij,ijk->ik", data.cell_offsets.to(cell_per_edge.dtype), cell_per_edge)
        
        # 3. Feed to Custom PyTorch Graph
        x_node = self.ce(
            pos=data.pos, 
            Z=data.atomic_numbers, 
            batch=data.batch, 
            edge_index=data.edge_index, 
            shifts=shifts, 
            csd_emb=csd_mixed_emb, 
            charge=data.charge, 
            spin=data.spin, 
            natoms=data.natoms
        )
        
        return {"node_embedding": x_node, "batch": data.batch}