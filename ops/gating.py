"""
Fused Pointwise Gating Kernels (Split Memory Variant).

Reads the split outputs from Conv1 (Y_m0, Y_m1, Y_m2), applies self-gating to M=0 
and cross-gating to M=1/M=2. Writes to dense output tensors (Z_m0, Z_m1, Z_m2).
Uses a 1D Grid over edges with an internal loop over channels to guarantee zero
register spilling.
"""

import torch
import triton
import triton.language as tl

from ops.constants import (
    Y_M0_MULT, Y_M1_MULT, Y_M2_MULT,
    Z_M0_MULT, Z_M1_MULT, Z_M2_MULT,
    TRUNC_Y_M0_MULT, TRUNC_Z_M0_MULT
)

def _generate_configs():
    configs = []
    for e in [16, 32, 64, 128]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                for ns in [1, 2]:
                    configs.append(triton.Config({'BLOCK_E': e, 'BLOCK_C': c}, num_warps=w, num_stages=ns))
    return configs

# =========================================================================
# FORWARD KERNEL: Split Gating
# =========================================================================
@triton.autotune(configs=_generate_configs(), key=['C'])
@triton.jit
def _gating_split_fwd_kernel(
    Y_m0_ptr, Y_m1_ptr, Y_m2_ptr,
    Z_m0_ptr, Z_m1_ptr, Z_m2_ptr,
    E, C,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 1D Grid over Edges
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    
    # Loop over Channels to reuse registers
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        # Calculate base offsets for each block
        y_base_m0 = e_offs[:, None] * Y_M0_MULT * C + c_offs[None, :]
        y_base_m1 = e_offs[:, None] * Y_M1_MULT * C + c_offs[None, :]
        y_base_m2 = e_offs[:, None] * Y_M2_MULT * C + c_offs[None, :]

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        # ----------------------------------------------------
        # 1. Load Inputs from 3 Split Pointers
        # ----------------------------------------------------
        # M=0
        x0_e0 = tl.load(Y_m0_ptr + y_base_m0 + 0*C, mask=mask, other=0.0)
        x0_e1 = tl.load(Y_m0_ptr + y_base_m0 + 1*C, mask=mask, other=0.0)
        y2    = tl.load(Y_m0_ptr + y_base_m0 + 2*C, mask=mask, other=0.0)
        y3    = tl.load(Y_m0_ptr + y_base_m0 + 3*C, mask=mask, other=0.0)
        y4    = tl.load(Y_m0_ptr + y_base_m0 + 4*C, mask=mask, other=0.0)
        
        # M=1
        y1r_0 = tl.load(Y_m1_ptr + y_base_m1 + 0*C, mask=mask, other=0.0)
        y1r_1 = tl.load(Y_m1_ptr + y_base_m1 + 1*C, mask=mask, other=0.0)
        y1i_0 = tl.load(Y_m1_ptr + y_base_m1 + 2*C, mask=mask, other=0.0)
        y1i_1 = tl.load(Y_m1_ptr + y_base_m1 + 3*C, mask=mask, other=0.0)
        
        # M=2
        y2r   = tl.load(Y_m2_ptr + y_base_m2 + 0*C, mask=mask, other=0.0)
        y2i   = tl.load(Y_m2_ptr + y_base_m2 + 1*C, mask=mask, other=0.0)

        # ----------------------------------------------------
        # 2. Compute Activation Gates
        # ----------------------------------------------------
        g0 = tl.sigmoid(x0_e0)
        g1 = tl.sigmoid(x0_e1)
        sy2 = tl.sigmoid(y2)

        # ----------------------------------------------------
        # 3. Write Gated Output to 3 Split Pointers
        # ----------------------------------------------------
        # M=0 (3C Output)
        tl.store(Z_m0_ptr + z_base_m0 + 0*C, y2 * sy2, mask=mask)
        tl.store(Z_m0_ptr + z_base_m0 + 1*C, y3 * g0, mask=mask)
        tl.store(Z_m0_ptr + z_base_m0 + 2*C, y4 * g1, mask=mask)
        
        # M=1 (4C Output)
        tl.store(Z_m1_ptr + z_base_m1 + 0*C, y1r_0 * g0, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 1*C, y1r_1 * g1, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 2*C, y1i_0 * g0, mask=mask)
        tl.store(Z_m1_ptr + z_base_m1 + 3*C, y1i_1 * g1, mask=mask)
        
        # M=2 (2C Output)
        tl.store(Z_m2_ptr + z_base_m2 + 0*C, y2r * g1, mask=mask)
        tl.store(Z_m2_ptr + z_base_m2 + 1*C, y2i * g1, mask=mask)


# =========================================================================
# BACKWARD KERNEL: Split Gating Adjoints
# =========================================================================
@triton.autotune(configs=_generate_configs(), key=['C'])
@triton.jit
def _gating_split_bwd_kernel(
    g_z_m0_ptr, g_z_m1_ptr, g_z_m2_ptr,
    Y_m0_ptr, Y_m1_ptr, Y_m2_ptr,
    g_y_m0_ptr, g_y_m1_ptr, g_y_m2_ptr,
    E, C,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        mask = e_mask[:, None] & c_mask[None, :]

        y_base_m0 = e_offs[:, None] * Y_M0_MULT * C + c_offs[None, :]
        y_base_m1 = e_offs[:, None] * Y_M1_MULT * C + c_offs[None, :]
        y_base_m2 = e_offs[:, None] * Y_M2_MULT * C + c_offs[None, :]

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        # 1. Recompute forward values (faster than saving them to HBM)
        x0_e0 = tl.load(Y_m0_ptr + y_base_m0 + 0*C, mask=mask, other=0.0)
        x0_e1 = tl.load(Y_m0_ptr + y_base_m0 + 1*C, mask=mask, other=0.0)
        y2    = tl.load(Y_m0_ptr + y_base_m0 + 2*C, mask=mask, other=0.0)
        y3    = tl.load(Y_m0_ptr + y_base_m0 + 3*C, mask=mask, other=0.0)
        y4    = tl.load(Y_m0_ptr + y_base_m0 + 4*C, mask=mask, other=0.0)
        
        y1r_0 = tl.load(Y_m1_ptr + y_base_m1 + 0*C, mask=mask, other=0.0)
        y1r_1 = tl.load(Y_m1_ptr + y_base_m1 + 1*C, mask=mask, other=0.0)
        y1i_0 = tl.load(Y_m1_ptr + y_base_m1 + 2*C, mask=mask, other=0.0)
        y1i_1 = tl.load(Y_m1_ptr + y_base_m1 + 3*C, mask=mask, other=0.0)
        
        y2r   = tl.load(Y_m2_ptr + y_base_m2 + 0*C, mask=mask, other=0.0)
        y2i   = tl.load(Y_m2_ptr + y_base_m2 + 1*C, mask=mask, other=0.0)

        g0 = tl.sigmoid(x0_e0); dg0 = g0 * (1.0 - g0)
        g1 = tl.sigmoid(x0_e1); dg1 = g1 * (1.0 - g1)
        sy2 = tl.sigmoid(y2); dsy2 = sy2 * (1.0 - sy2)

        # 2. Load incoming gradients from Conv2
        gy2 = tl.load(g_z_m0_ptr + z_base_m0 + 0*C, mask=mask, other=0.0)
        gy3 = tl.load(g_z_m0_ptr + z_base_m0 + 1*C, mask=mask, other=0.0)
        gy4 = tl.load(g_z_m0_ptr + z_base_m0 + 2*C, mask=mask, other=0.0)
        
        gy1r_0 = tl.load(g_z_m1_ptr + z_base_m1 + 0*C, mask=mask, other=0.0)
        gy1r_1 = tl.load(g_z_m1_ptr + z_base_m1 + 1*C, mask=mask, other=0.0)
        gy1i_0 = tl.load(g_z_m1_ptr + z_base_m1 + 2*C, mask=mask, other=0.0)
        gy1i_1 = tl.load(g_z_m1_ptr + z_base_m1 + 3*C, mask=mask, other=0.0)
        
        gy2r = tl.load(g_z_m2_ptr + z_base_m2 + 0*C, mask=mask, other=0.0)
        gy2i = tl.load(g_z_m2_ptr + z_base_m2 + 1*C, mask=mask, other=0.0)

        # 3. Chain Rule: Derivative of Output w.r.t Y_in
        g_x0_e0 = dg0 * (gy3 * y3 + gy1r_0 * y1r_0 + gy1i_0 * y1i_0)
        g_x0_e1 = dg1 * (gy4 * y4 + gy1r_1 * y1r_1 + gy1i_1 * y1i_1 + gy2r * y2r + gy2i * y2i)
        g_y2 = gy2 * (sy2 + y2 * dsy2)

        # 4. Store Gradients to 3 Split Pointers
        tl.store(g_y_m0_ptr + y_base_m0 + 0*C, g_x0_e0, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 1*C, g_x0_e1, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 2*C, g_y2, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 3*C, gy3 * g0, mask=mask)
        tl.store(g_y_m0_ptr + y_base_m0 + 4*C, gy4 * g1, mask=mask)
        
        tl.store(g_y_m1_ptr + y_base_m1 + 0*C, gy1r_0 * g0, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 1*C, gy1r_1 * g1, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 2*C, gy1i_0 * g0, mask=mask)
        tl.store(g_y_m1_ptr + y_base_m1 + 3*C, gy1i_1 * g1, mask=mask)
        
        tl.store(g_y_m2_ptr + y_base_m2 + 0*C, gy2r * g1, mask=mask)
        tl.store(g_y_m2_ptr + y_base_m2 + 1*C, gy2i * g1, mask=mask)


class FusedGatingSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y_m0, Y_m1, Y_m2, C):
        E = Y_m0.shape[0]
        device = Y_m0.device
        dtype = Y_m0.dtype
        
        # Allocate the 3 output tensors perfectly sized for Conv2
        Z_m0 = torch.empty((E, Z_M0_MULT.value * C), device=device, dtype=dtype)
        Z_m1 = torch.empty((E, Z_M1_MULT.value * C), device=device, dtype=dtype)
        Z_m2 = torch.empty((E, Z_M2_MULT.value * C), device=device, dtype=dtype)
        
        # 1D Grid Launch
        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']), )
        _gating_split_fwd_kernel[grid](
            Y_m0, Y_m1, Y_m2,
            Z_m0, Z_m1, Z_m2,
            E, C
        )
        
        ctx.save_for_backward(Y_m0, Y_m1, Y_m2)
        ctx.C = C
        return Z_m0, Z_m1, Z_m2

    @staticmethod
    def backward(ctx, g_z_m0, g_z_m1, g_z_m2):
        Y_m0, Y_m1, Y_m2 = ctx.saved_tensors
        E, C = Y_m0.shape[0], ctx.C
        
        g_z_m0 = g_z_m0.contiguous()
        g_z_m1 = g_z_m1.contiguous()
        g_z_m2 = g_z_m2.contiguous()

        g_y_m0 = torch.empty_like(Y_m0, memory_format=torch.contiguous_format)
        g_y_m1 = torch.empty_like(Y_m1, memory_format=torch.contiguous_format)
        g_y_m2 = torch.empty_like(Y_m2, memory_format=torch.contiguous_format)

        # 1D Grid Launch
        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']), )
        _gating_split_bwd_kernel[grid](
            g_z_m0, g_z_m1, g_z_m2,
            Y_m0, Y_m1, Y_m2,
            g_y_m0, g_y_m1, g_y_m2,
            E, C
        )
        
        return g_y_m0, g_y_m1, g_y_m2, None

fused_gating_split = FusedGatingSplit.apply