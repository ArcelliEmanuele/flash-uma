import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ops.ln_silu import layer_norm_silu
from ops.gating import fused_gating_split
from ops.gather import gather_wigner_split
from ops.scatter import rotate_scatter_split, init_node_scatter
from ops.geom import precompute_geometry
from fairchem.core.models.uma.escn_md import balance_channels_batched

# ==============================================================================
# 1. GRAPH PREPROCESSING (MORTON & CSR CACHING)
# ==============================================================================

_CSR_CACHE_CAPACITY = 8
_csr_cache: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

def _build_csr_uncached(idx_j: torch.Tensor, N: int):
    sort_order = torch.argsort(idx_j, stable=True)
    counts = torch.bincount(idx_j, minlength=N).to(torch.int32)
    row_ptr = torch.zeros(N + 1, dtype=torch.int32, device=idx_j.device)
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    col_idx = sort_order.to(torch.int32)
    return row_ptr, col_idx

def build_csr_cached(idx_j: torch.Tensor, N: int):
    # MD steps usually keep connectivity fixed while positions change, so the
    # CSR topology is reused. The key must invalidate on ANY change to idx_j:
    #   - data_ptr(): catches a freshly allocated / replaced tensor
    #   - _version: bumped by in-place mutation (same storage, edited contents),
    #     which data_ptr alone would miss
    #   - numel()/N/device: guard against shape or device mismatches
    # Without _version, an in-place edge update would silently reuse stale CSR.
    key = (idx_j.data_ptr(), idx_j._version, idx_j.numel(), N, idx_j.device.index if idx_j.device.type == "cuda" else -1)
    hit = _csr_cache.get(key)
    if hit is not None:
        _csr_cache.move_to_end(key)
        return hit
    row_ptr, col_idx = _build_csr_uncached(idx_j, N)
    _csr_cache[key] = (row_ptr, col_idx)
    if len(_csr_cache) > _CSR_CACHE_CAPACITY: _csr_cache.popitem(last=False)
    return row_ptr, col_idx

def clear_csr_cache(): _csr_cache.clear()

def expand_bits_torch(v: torch.Tensor) -> torch.Tensor:
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    v = (v | (v <<  1)) & 0x12492492
    return v

def morton3d_torch(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x_b = expand_bits_torch(x); y_b = expand_bits_torch(y) << 1; z_b = expand_bits_torch(z) << 2
    return x_b | y_b | z_b

def reorder_graph_morton(pos, Z, batch, idx_i, idx_j):
    pos_min = pos.min(dim=0, keepdim=True)[0]
    pos_max = pos.max(dim=0, keepdim=True)[0]
    scale = 1023.0 / torch.clamp(pos_max - pos_min, min=1e-5)
    pos_int = ((pos - pos_min) * scale).to(torch.int32)
    morton_codes = morton3d_torch(pos_int[:, 0], pos_int[:, 1], pos_int[:, 2])
    
    batch_shift = batch.to(torch.int64) << 32 
    sort_keys = morton_codes.to(torch.int64) | batch_shift
    perm = torch.argsort(sort_keys)
    
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=pos.device)
    
    return pos[perm], Z[perm], batch[perm], inv_perm[idx_i], inv_perm[idx_j], perm, inv_perm


# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================

class Packed_eSCN_Layer(nn.Module):
    def __init__(self, C: int, B: int):
        super().__init__()
        self.C = C
        self.W_rad1_1 = nn.Parameter(torch.randn(C, B + 2 * C)); self.b_rad1_1 = nn.Parameter(torch.zeros(C))
        self.gamma_rad1_1 = nn.Parameter(torch.ones(C));         self.beta_rad1_1 = nn.Parameter(torch.zeros(C))
        self.W_rad1_2 = nn.Parameter(torch.randn(C, C));         self.b_rad1_2 = nn.Parameter(torch.zeros(C))
        self.gamma_rad1_2 = nn.Parameter(torch.ones(C));         self.beta_rad1_2 = nn.Parameter(torch.zeros(C))
        self.W_rad1_3 = nn.Parameter(torch.randn(12 * C, C));    self.b_rad1_3 = nn.Parameter(torch.zeros(12 * C))

        self.W1_m0 = nn.Parameter(torch.randn(5 * C, 6 * C));    self.b1_m0 = nn.Parameter(torch.zeros(5 * C))
        self.W1_m1_R = nn.Parameter(torch.randn(2 * C, 4 * C));  self.W1_m1_I = nn.Parameter(torch.randn(2 * C, 4 * C))
        self.W1_m2_R = nn.Parameter(torch.randn(1 * C, 2 * C));  self.W1_m2_I = nn.Parameter(torch.randn(1 * C, 2 * C))

        self.W2_m0 = nn.Parameter(torch.randn(3 * C, 3 * C));    self.b2_m0 = nn.Parameter(torch.zeros(3 * C))
        self.W2_m1_R = nn.Parameter(torch.randn(2 * C, 2 * C));  self.W2_m1_I = nn.Parameter(torch.randn(2 * C, 2 * C))
        self.W2_m2_R = nn.Parameter(torch.randn(1 * C, 1 * C));  self.W2_m2_I = nn.Parameter(torch.randn(1 * C, 1 * C))

        self.norm_1 = None; self.norm_2 = None; self.atom_wise = None

    def prepare_for_inference(self, dtype=torch.float32):
        C, device = self.C, self.W1_m0.device
        if self.norm_1:
            for p in self.norm_1.parameters(): p.requires_grad = False
            for p in self.norm_2.parameters(): p.requires_grad = False
            for p in self.atom_wise.parameters(): p.requires_grad = False

        self.register_buffer('W_conv1_m0', self.W1_m0.data.clone().to(dtype))
        self.register_buffer('b_conv1_m0', self.b1_m0.data.clone().to(dtype))

        # Each m>0 SO(2) channel is a complex-valued linear map W = R + iI acting
        # on a complex feature x = a + ib, giving (Ra - Ib) + i(Ia + Rb). The
        # kernels operate on real tensors only, so we stack [real; imag] and
        # pack the equivalent real block matrix [[R, -I], [I, R]] once here.
        # Matmul against the stacked vector then reproduces complex multiply
        # exactly, keeping the SO(2) rotation equivariance with no complex ops
        # in the hot path.
        W_conv1_m1 = torch.zeros((4 * C, 8 * C), device=device)
        W_conv1_m1[0:2*C, 0:4*C] = self.W1_m1_R.data;   W_conv1_m1[0:2*C, 4*C:8*C] = -self.W1_m1_I.data
        W_conv1_m1[2*C:4*C, 0:4*C] = self.W1_m1_I.data; W_conv1_m1[2*C:4*C, 4*C:8*C] = self.W1_m1_R.data
        self.register_buffer('W_conv1_m1', W_conv1_m1.to(dtype))
        
        W_conv1_m2 = torch.zeros((2 * C, 4 * C), device=device)
        W_conv1_m2[0:1*C, 0:2*C] = self.W1_m2_R.data;   W_conv1_m2[0:1*C, 2*C:4*C] = -self.W1_m2_I.data
        W_conv1_m2[1*C:2*C, 0:2*C] = self.W1_m2_I.data; W_conv1_m2[1*C:2*C, 2*C:4*C] = self.W1_m2_R.data
        self.register_buffer('W_conv1_m2', W_conv1_m2.to(dtype))

        self.register_buffer('W_conv2_m0', self.W2_m0.data.clone().to(dtype))
        self.register_buffer('b_conv2_m0', self.b2_m0.data.clone().to(dtype))
        W_conv2_m1 = torch.zeros((4 * C, 4 * C), device=device)
        W_conv2_m1[0:2*C, 0:2*C] = self.W2_m1_R.data;   W_conv2_m1[0:2*C, 2*C:4*C] = -self.W2_m1_I.data
        W_conv2_m1[2*C:4*C, 0:2*C] = self.W2_m1_I.data; W_conv2_m1[2*C:4*C, 2*C:4*C] = self.W2_m1_R.data
        self.register_buffer('W_conv2_m1', W_conv2_m1.to(dtype))

        W_conv2_m2 = torch.zeros((2 * C, 2 * C), device=device)
        W_conv2_m2[0:1*C, 0:1*C] = self.W2_m2_R.data;   W_conv2_m2[0:1*C, 1*C:2*C] = -self.W2_m2_I.data
        W_conv2_m2[1*C:2*C, 0:1*C] = self.W2_m2_I.data; W_conv2_m2[1*C:2*C, 1*C:2*C] = self.W2_m2_R.data
        self.register_buffer('W_conv2_m2', W_conv2_m2.to(dtype))

        del self.W1_m0, self.b1_m0, self.W1_m1_R, self.W1_m1_I, self.W1_m2_R, self.W1_m2_I
        del self.W2_m0, self.b2_m0, self.W2_m1_R, self.W2_m1_I, self.W2_m2_R, self.W2_m2_I

    def forward(self, x_in, x_edge_raw, wigner, env, idx_i, idx_j, row_ptr, col_idx, csd_emb, batch):
        x_res_block = x_in
        x_norm = self.norm_1(x_in)

        h1 = layer_norm_silu(F.linear(x_edge_raw, self.W_rad1_1, self.b_rad1_1), self.gamma_rad1_1, self.beta_rad1_1)
        h2 = layer_norm_silu(F.linear(h1, self.W_rad1_2, self.b_rad1_2), self.gamma_rad1_2, self.beta_rad1_2)
        rad_out = F.linear(h2, self.W_rad1_3, self.b_rad1_3)

        X_m0, X_m1, X_m2 = gather_wigner_split(x_norm, idx_i, idx_j, wigner, rad_out, csd_emb, batch)
        Y_m0, Y_m1, Y_m2 = F.linear(X_m0, self.W_conv1_m0, self.b_conv1_m0), F.linear(X_m1, self.W_conv1_m1), F.linear(X_m2, self.W_conv1_m2)
        Z_m0, Z_m1, Z_m2 = fused_gating_split(Y_m0, Y_m1, Y_m2, self.C)
        Z_out_m0, Z_out_m1, Z_out_m2 = F.linear(Z_m0, self.W_conv2_m0, self.b_conv2_m0), F.linear(Z_m1, self.W_conv2_m1), F.linear(Z_m2, self.W_conv2_m2)

        x_mid = rotate_scatter_split(Z_out_m0, Z_out_m1, Z_out_m2, wigner, env, x_res_block, idx_j, row_ptr, col_idx, x_in.shape[0])
        return self.atom_wise(self.norm_2(x_mid)) + x_mid


class eSCN_Inference_Model(nn.Module):
    def __init__(self, num_layers=4, sphere_channels=128, edge_channels=128, num_distance_basis=32, cutoff=5.0, coeff_g=-0.5):
        super().__init__()
        self.C, self.cutoff, self.coeff_g, self.B = sphere_channels, cutoff, coeff_g, num_distance_basis
        
        self.W_sphere = nn.Parameter(torch.randn(100, self.C))  
        self.W_src = nn.Parameter(torch.randn(100, self.C)); self.W_tgt = nn.Parameter(torch.randn(100, self.C))
        self.offset_g = nn.Parameter(torch.linspace(0, cutoff, self.B))

        self.W_rad1_1 = nn.Parameter(torch.randn(self.C, self.B + 2 * self.C)); self.b_rad1_1 = nn.Parameter(torch.zeros(self.C))
        self.gamma_rad1_1 = nn.Parameter(torch.ones(self.C));                   self.beta_rad1_1 = nn.Parameter(torch.zeros(self.C))
        self.W_rad1_2 = nn.Parameter(torch.randn(self.C, self.C));              self.b_rad1_2 = nn.Parameter(torch.zeros(self.C))
        self.gamma_rad1_2 = nn.Parameter(torch.ones(self.C));                   self.beta_rad1_2 = nn.Parameter(torch.zeros(self.C))
        self.W_rad1_3 = nn.Parameter(torch.randn(3 * self.C, self.C));          self.b_rad1_3 = nn.Parameter(torch.zeros(3 * self.C))

        self.layers = nn.ModuleList([Packed_eSCN_Layer(self.C, self.B) for _ in range(num_layers)])
        self.final_norm = None 
        self.charge_start = 0; self.charge_end = 0; self.spin_start = 0; self.spin_end = 0

    def prepare_for_inference(self, dtype=torch.float32):
        self.eval()
        for p in self.parameters(): p.requires_grad = False
        if self.final_norm:
            for p in self.final_norm.parameters(): p.requires_grad = False
        for layer in self.layers: layer.prepare_for_inference(dtype)
        self.to(dtype)

    def forward(self, pos, Z, batch, edge_index, shifts, csd_emb, charge=None, spin=None, natoms=None):
        # 1. Morton Reorder
        pos_s, Z_s, batch_s, idx_i_s, idx_j_s, perm, inv_perm = reorder_graph_morton(
            pos, Z, batch, edge_index[0], edge_index[1]
        )
        N = pos_s.shape[0]

        # 2. CSR Topology Cache
        row_ptr, col_idx = build_csr_cached(idx_j_s, N)

        # 3. Analytic Geometry Kernel
        x_edge_raw, wigner, env = precompute_geometry(
            pos_s, Z_s, idx_i_s, idx_j_s, shifts,
            self.W_src, self.W_tgt, self.offset_g, self.coeff_g, self.cutoff
        )

        h1 = layer_norm_silu(F.linear(x_edge_raw, self.W_rad1_1, self.b_rad1_1), self.gamma_rad1_1, self.beta_rad1_1)
        h2 = layer_norm_silu(F.linear(h1, self.W_rad1_2, self.b_rad1_2), self.gamma_rad1_2, self.beta_rad1_2)
        rad_out_p1 = F.linear(h2, self.W_rad1_3, self.b_rad1_3) 

        x_node_s = init_node_scatter(rad_out_p1, wigner, env, Z_s, batch_s, self.W_sphere, csd_emb, idx_j_s, row_ptr, col_idx, N)

        with torch.autocast('cuda', dtype=pos_s.dtype):
            for layer in self.layers:
                x_node_s = layer(x_node_s, x_edge_raw, wigner, env, idx_i_s, idx_j_s, row_ptr, col_idx, csd_emb, batch_s)
                
                if self.charge_end > self.charge_start:
                    x_node_s = balance_channels_batched(emb=x_node_s, target=charge, natoms=natoms, batch=batch_s, start_idx=self.charge_start, end_idx=self.charge_end, target_offset=0.0)
                if self.spin_end > self.spin_start:
                    x_node_s = balance_channels_batched(emb=x_node_s, target=spin, natoms=natoms, batch=batch_s, start_idx=self.spin_start, end_idx=self.spin_end, target_offset=1.0)

            x_node_s = self.final_norm(x_node_s)

        # 4. Inverse Permute to original Atomic layout
        return x_node_s[inv_perm]