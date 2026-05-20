"""
Packed Rotate-Out & Scatter Kernel (Split Memory Variant).

Reads the three contiguous block-diagonal outputs from Conv2 (Z_m0, Z_m1, Z_m2), 
applies the M->L Inverse Wigner rotations and distance envelope, and lock-free 
scatters into the target nodes using the CSR graph topology.

Features exact FP64 accumulation to prevent reduction variance, eliminating MD drift.
"""

import torch
import triton
import triton.language as tl

from ops.constants import (
    W11, W12, W13, W21, W22, W23, W31, W32, W33,
    W44, W45, W46, W47, W48, W54, W55, W56, W57, W58,
    W64, W65, W66, W67, W68, W74, W75, W76, W77, W78,
    W84, W85, W86, W87, W88,
    Z_M0_MULT, Z_M1_MULT, Z_M2_MULT
)

def _generate_fwd_configs():
    configs = []
    for nb in [16, 32, 64]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                for s in [1, 2]:
                    configs.append(triton.Config({'BLOCK_NB': nb, 'BLOCK_C': c}, num_warps=w, num_stages=s))
    return configs

def _generate_bwd_configs():
    configs = []
    for e in [16, 32, 64]:
        for c in [16, 32, 64, 128]:
            for w in [4, 8]:
                for s in [1, 2]:
                    configs.append(triton.Config({'BLOCK_E': e, 'BLOCK_C': c}, num_warps=w, num_stages=s))
    return configs


# =========================================================================
# PHASE 2 FORWARD: Lock-Free FP64 Scatter
# =========================================================================
@triton.autotune(configs=_generate_fwd_configs(), key=['C'])
@triton.jit
def _scatter_split_fwd_kernel(
    row_ptr, col_idx, 
    Z_m0_ptr, Z_m1_ptr, Z_m2_ptr, 
    wig_ptr, env_ptr, x_res_ptr, x_out_ptr,
    N, C, 
    stride_xn, stride_xl, stride_xc, 
    stride_wig_e,
    BLOCK_C: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    atom_j = tl.program_id(0)
    c_block = tl.program_id(1)
    if atom_j >= N:
        return

    nb_start = tl.load(row_ptr + atom_j).to(tl.int32)
    nb_end = tl.load(row_ptr + atom_j + 1).to(tl.int32)

    c_offs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # Load residual and IMMEDIATELY upcast to float64
    acc_L0   = tl.load(x_res_ptr + atom_j * stride_xn + 0 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L1_1 = tl.load(x_res_ptr + atom_j * stride_xn + 1 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L1_2 = tl.load(x_res_ptr + atom_j * stride_xn + 2 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L1_3 = tl.load(x_res_ptr + atom_j * stride_xn + 3 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L2_4 = tl.load(x_res_ptr + atom_j * stride_xn + 4 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L2_5 = tl.load(x_res_ptr + atom_j * stride_xn + 5 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L2_6 = tl.load(x_res_ptr + atom_j * stride_xn + 6 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L2_7 = tl.load(x_res_ptr + atom_j * stride_xn + 7 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)
    acc_L2_8 = tl.load(x_res_ptr + atom_j * stride_xn + 8 * stride_xl + c_offs * stride_xc, mask=c_mask, other=0.0).to(tl.float64)

    nb_cur = nb_start
    while nb_cur < nb_end:
        nb_offs = nb_cur + tl.arange(0, BLOCK_NB)
        nb_mask = nb_offs < nb_end
        e_pos = tl.load(col_idx + nb_offs, mask=nb_mask, other=0).to(tl.int32)
        
        full_mask = nb_mask[:, None] & c_mask[None, :]

        env = tl.load(env_ptr + e_pos, mask=nb_mask, other=0.0)[:, None]
        wb = e_pos * stride_wig_e

        # Load Wigner Scalars
        w11 = tl.load(wig_ptr + wb + W11, mask=nb_mask, other=0.0)[:, None]; w12 = tl.load(wig_ptr + wb + W12, mask=nb_mask, other=0.0)[:, None]; w13 = tl.load(wig_ptr + wb + W13, mask=nb_mask, other=0.0)[:, None]
        w21 = tl.load(wig_ptr + wb + W21, mask=nb_mask, other=0.0)[:, None]; w22 = tl.load(wig_ptr + wb + W22, mask=nb_mask, other=0.0)[:, None]; w23 = tl.load(wig_ptr + wb + W23, mask=nb_mask, other=0.0)[:, None]
        w31 = tl.load(wig_ptr + wb + W31, mask=nb_mask, other=0.0)[:, None]; w32 = tl.load(wig_ptr + wb + W32, mask=nb_mask, other=0.0)[:, None]; w33 = tl.load(wig_ptr + wb + W33, mask=nb_mask, other=0.0)[:, None]
        w44 = tl.load(wig_ptr + wb + W44, mask=nb_mask, other=0.0)[:, None]; w45 = tl.load(wig_ptr + wb + W45, mask=nb_mask, other=0.0)[:, None]; w46 = tl.load(wig_ptr + wb + W46, mask=nb_mask, other=0.0)[:, None]; w47 = tl.load(wig_ptr + wb + W47, mask=nb_mask, other=0.0)[:, None]; w48 = tl.load(wig_ptr + wb + W48, mask=nb_mask, other=0.0)[:, None]
        w54 = tl.load(wig_ptr + wb + W54, mask=nb_mask, other=0.0)[:, None]; w55 = tl.load(wig_ptr + wb + W55, mask=nb_mask, other=0.0)[:, None]; w56 = tl.load(wig_ptr + wb + W56, mask=nb_mask, other=0.0)[:, None]; w57 = tl.load(wig_ptr + wb + W57, mask=nb_mask, other=0.0)[:, None]; w58 = tl.load(wig_ptr + wb + W58, mask=nb_mask, other=0.0)[:, None]
        w64 = tl.load(wig_ptr + wb + W64, mask=nb_mask, other=0.0)[:, None]; w65 = tl.load(wig_ptr + wb + W65, mask=nb_mask, other=0.0)[:, None]; w66 = tl.load(wig_ptr + wb + W66, mask=nb_mask, other=0.0)[:, None]; w67 = tl.load(wig_ptr + wb + W67, mask=nb_mask, other=0.0)[:, None]; w68 = tl.load(wig_ptr + wb + W68, mask=nb_mask, other=0.0)[:, None]
        w74 = tl.load(wig_ptr + wb + W74, mask=nb_mask, other=0.0)[:, None]; w75 = tl.load(wig_ptr + wb + W75, mask=nb_mask, other=0.0)[:, None]; w76 = tl.load(wig_ptr + wb + W76, mask=nb_mask, other=0.0)[:, None]; w77 = tl.load(wig_ptr + wb + W77, mask=nb_mask, other=0.0)[:, None]; w78 = tl.load(wig_ptr + wb + W78, mask=nb_mask, other=0.0)[:, None]
        w84 = tl.load(wig_ptr + wb + W84, mask=nb_mask, other=0.0)[:, None]; w85 = tl.load(wig_ptr + wb + W85, mask=nb_mask, other=0.0)[:, None]; w86 = tl.load(wig_ptr + wb + W86, mask=nb_mask, other=0.0)[:, None]; w87 = tl.load(wig_ptr + wb + W87, mask=nb_mask, other=0.0)[:, None]; w88 = tl.load(wig_ptr + wb + W88, mask=nb_mask, other=0.0)[:, None]

        # Base Offsets
        z_base_m0 = e_pos[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_pos[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_pos[:, None] * Z_M2_MULT * C + c_offs[None, :]

        # Load Z components
        Z0_0 = tl.load(Z_m0_ptr + z_base_m0 + 0*C, mask=full_mask, other=0.0)
        Z0_1 = tl.load(Z_m0_ptr + z_base_m0 + 1*C, mask=full_mask, other=0.0)
        Z0_2 = tl.load(Z_m0_ptr + z_base_m0 + 2*C, mask=full_mask, other=0.0)
        
        Z1R_0 = tl.load(Z_m1_ptr + z_base_m1 + 0*C, mask=full_mask, other=0.0)
        Z1R_1 = tl.load(Z_m1_ptr + z_base_m1 + 1*C, mask=full_mask, other=0.0)
        Z1I_0 = tl.load(Z_m1_ptr + z_base_m1 + 2*C, mask=full_mask, other=0.0)
        Z1I_1 = tl.load(Z_m1_ptr + z_base_m1 + 3*C, mask=full_mask, other=0.0)
        
        Z2R_0 = tl.load(Z_m2_ptr + z_base_m2 + 0*C, mask=full_mask, other=0.0)
        Z2I_0 = tl.load(Z_m2_ptr + z_base_m2 + 1*C, mask=full_mask, other=0.0)

        # M -> L Inverse Rotation (Multiply in FP32, Accumulate in FP64)
        acc_L0   += tl.sum(Z0_0 * env, axis=0).to(tl.float64)
        acc_L1_1 += tl.sum((w21*Z0_1 + w11*Z1I_0 + w31*Z1R_0) * env, axis=0).to(tl.float64)
        acc_L1_2 += tl.sum((w22*Z0_1 + w12*Z1I_0 + w32*Z1R_0) * env, axis=0).to(tl.float64)
        acc_L1_3 += tl.sum((w23*Z0_1 + w13*Z1I_0 + w33*Z1R_0) * env, axis=0).to(tl.float64)
        
        acc_L2_4 += tl.sum((w64*Z0_2 + w54*Z1I_1 + w74*Z1R_1 + w44*Z2I_0 + w84*Z2R_0) * env, axis=0).to(tl.float64)
        acc_L2_5 += tl.sum((w65*Z0_2 + w55*Z1I_1 + w75*Z1R_1 + w45*Z2I_0 + w85*Z2R_0) * env, axis=0).to(tl.float64)
        acc_L2_6 += tl.sum((w66*Z0_2 + w56*Z1I_1 + w76*Z1R_1 + w46*Z2I_0 + w86*Z2R_0) * env, axis=0).to(tl.float64)
        acc_L2_7 += tl.sum((w67*Z0_2 + w57*Z1I_1 + w77*Z1R_1 + w47*Z2I_0 + w87*Z2R_0) * env, axis=0).to(tl.float64)
        acc_L2_8 += tl.sum((w68*Z0_2 + w58*Z1I_1 + w78*Z1R_1 + w48*Z2I_0 + w88*Z2R_0) * env, axis=0).to(tl.float64)

        nb_cur += BLOCK_NB

    # Final Store: Cast EXACT float64 results back to float32
    tl.store(x_out_ptr + atom_j * stride_xn + 0 * stride_xl + c_offs * stride_xc, acc_L0.to(tl.float32),   mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 1 * stride_xl + c_offs * stride_xc, acc_L1_1.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 2 * stride_xl + c_offs * stride_xc, acc_L1_2.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 3 * stride_xl + c_offs * stride_xc, acc_L1_3.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 4 * stride_xl + c_offs * stride_xc, acc_L2_4.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 5 * stride_xl + c_offs * stride_xc, acc_L2_5.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 6 * stride_xl + c_offs * stride_xc, acc_L2_6.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 7 * stride_xl + c_offs * stride_xc, acc_L2_7.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 8 * stride_xl + c_offs * stride_xc, acc_L2_8.to(tl.float32), mask=c_mask)


# =========================================================================
# PHASE 2 BACKWARD: Split L1 and L2
# =========================================================================
@triton.autotune(configs=_generate_bwd_configs(), key=['C'])
@triton.jit
def _scatter_split_bwd_l1_kernel(
    g_out_ptr, idx_j_ptr, 
    Z_m0_ptr, Z_m1_ptr, Z_m2_ptr, 
    wig_ptr, env_ptr,
    g_z_m0_ptr, g_z_m1_ptr, g_z_m2_ptr, 
    g_wig_ptr, g_env_l1_ptr,
    E, C, stride_xn, stride_xl, stride_xc, stride_wig_e,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env_e = env[:, None]
    wb = e_offs * stride_wig_e

    w11 = tl.load(wig_ptr + wb + W11, mask=e_mask, other=0.0)[:, None]; w12 = tl.load(wig_ptr + wb + W12, mask=e_mask, other=0.0)[:, None]; w13 = tl.load(wig_ptr + wb + W13, mask=e_mask, other=0.0)[:, None]
    w21 = tl.load(wig_ptr + wb + W21, mask=e_mask, other=0.0)[:, None]; w22 = tl.load(wig_ptr + wb + W22, mask=e_mask, other=0.0)[:, None]; w23 = tl.load(wig_ptr + wb + W23, mask=e_mask, other=0.0)[:, None]
    w31 = tl.load(wig_ptr + wb + W31, mask=e_mask, other=0.0)[:, None]; w32 = tl.load(wig_ptr + wb + W32, mask=e_mask, other=0.0)[:, None]; w33 = tl.load(wig_ptr + wb + W33, mask=e_mask, other=0.0)[:, None]

    gw11_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw12_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw13_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw21_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw22_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw23_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw31_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw32_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw33_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]

        g0 = tl.load(g_out_ptr + j[:, None] * stride_xn + 0 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g1 = tl.load(g_out_ptr + j[:, None] * stride_xn + 1 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g2 = tl.load(g_out_ptr + j[:, None] * stride_xn + 2 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g3 = tl.load(g_out_ptr + j[:, None] * stride_xn + 3 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]

        Z0_0  = tl.load(Z_m0_ptr + z_base_m0 + 0*C, mask=full_mask, other=0.0)
        Z0_1  = tl.load(Z_m0_ptr + z_base_m0 + 1*C, mask=full_mask, other=0.0)
        Z1R_0 = tl.load(Z_m1_ptr + z_base_m1 + 0*C, mask=full_mask, other=0.0)
        Z1I_0 = tl.load(Z_m1_ptr + z_base_m1 + 2*C, mask=full_mask, other=0.0)

        aL1_1 = g1 * env_e; aL1_2 = g2 * env_e; aL1_3 = g3 * env_e
        
        # Store L1 gradients to M0 and M1 blocks
        tl.store(g_z_m0_ptr + z_base_m0 + 0*C, g0 * env_e, mask=full_mask)
        tl.store(g_z_m0_ptr + z_base_m0 + 1*C, aL1_1 * w21 + aL1_2 * w22 + aL1_3 * w23, mask=full_mask)
        tl.store(g_z_m1_ptr + z_base_m1 + 0*C, aL1_1 * w31 + aL1_2 * w32 + aL1_3 * w33, mask=full_mask)
        tl.store(g_z_m1_ptr + z_base_m1 + 2*C, aL1_1 * w11 + aL1_2 * w12 + aL1_3 * w13, mask=full_mask)

        gw11_acc += tl.sum(g1 * Z1I_0, axis=1); gw12_acc += tl.sum(g2 * Z1I_0, axis=1); gw13_acc += tl.sum(g3 * Z1I_0, axis=1)
        gw21_acc += tl.sum(g1 * Z0_1,  axis=1); gw22_acc += tl.sum(g2 * Z0_1,  axis=1); gw23_acc += tl.sum(g3 * Z0_1,  axis=1)
        gw31_acc += tl.sum(g1 * Z1R_0, axis=1); gw32_acc += tl.sum(g2 * Z1R_0, axis=1); gw33_acc += tl.sum(g3 * Z1R_0, axis=1)

        g_env_acc += tl.sum(g0 * Z0_0, axis=1) + tl.sum(g1 * (w21*Z0_1 + w11*Z1I_0 + w31*Z1R_0) + g2 * (w22*Z0_1 + w12*Z1I_0 + w32*Z1R_0) + g3 * (w23*Z0_1 + w13*Z1I_0 + w33*Z1R_0), axis=1)

    tl.store(g_wig_ptr + wb + W11, gw11_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W12, gw12_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W13, gw13_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W21, gw21_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W22, gw22_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W23, gw23_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W31, gw31_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W32, gw32_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W33, gw33_acc * env, mask=e_mask)
    tl.store(g_env_l1_ptr + e_offs, g_env_acc, mask=e_mask)


@triton.autotune(configs=_generate_bwd_configs(), key=['C'])
@triton.jit
def _scatter_split_bwd_l2_kernel(
    g_out_ptr, idx_j_ptr, 
    Z_m0_ptr, Z_m1_ptr, Z_m2_ptr, 
    wig_ptr, env_ptr,
    g_z_m0_ptr, g_z_m1_ptr, g_z_m2_ptr, 
    g_wig_ptr, g_env_l2_ptr,
    E, C, stride_xn, stride_xl, stride_xc, stride_wig_e,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env_e = env[:, None]
    wb = e_offs * stride_wig_e

    w44 = tl.load(wig_ptr + wb + W44, mask=e_mask, other=0.0)[:, None]; w45 = tl.load(wig_ptr + wb + W45, mask=e_mask, other=0.0)[:, None]; w46 = tl.load(wig_ptr + wb + W46, mask=e_mask, other=0.0)[:, None]; w47 = tl.load(wig_ptr + wb + W47, mask=e_mask, other=0.0)[:, None]; w48 = tl.load(wig_ptr + wb + W48, mask=e_mask, other=0.0)[:, None]
    w54 = tl.load(wig_ptr + wb + W54, mask=e_mask, other=0.0)[:, None]; w55 = tl.load(wig_ptr + wb + W55, mask=e_mask, other=0.0)[:, None]; w56 = tl.load(wig_ptr + wb + W56, mask=e_mask, other=0.0)[:, None]; w57 = tl.load(wig_ptr + wb + W57, mask=e_mask, other=0.0)[:, None]; w58 = tl.load(wig_ptr + wb + W58, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64, mask=e_mask, other=0.0)[:, None]; w65 = tl.load(wig_ptr + wb + W65, mask=e_mask, other=0.0)[:, None]; w66 = tl.load(wig_ptr + wb + W66, mask=e_mask, other=0.0)[:, None]; w67 = tl.load(wig_ptr + wb + W67, mask=e_mask, other=0.0)[:, None]; w68 = tl.load(wig_ptr + wb + W68, mask=e_mask, other=0.0)[:, None]
    w74 = tl.load(wig_ptr + wb + W74, mask=e_mask, other=0.0)[:, None]; w75 = tl.load(wig_ptr + wb + W75, mask=e_mask, other=0.0)[:, None]; w76 = tl.load(wig_ptr + wb + W76, mask=e_mask, other=0.0)[:, None]; w77 = tl.load(wig_ptr + wb + W77, mask=e_mask, other=0.0)[:, None]; w78 = tl.load(wig_ptr + wb + W78, mask=e_mask, other=0.0)[:, None]
    w84 = tl.load(wig_ptr + wb + W84, mask=e_mask, other=0.0)[:, None]; w85 = tl.load(wig_ptr + wb + W85, mask=e_mask, other=0.0)[:, None]; w86 = tl.load(wig_ptr + wb + W86, mask=e_mask, other=0.0)[:, None]; w87 = tl.load(wig_ptr + wb + W87, mask=e_mask, other=0.0)[:, None]; w88 = tl.load(wig_ptr + wb + W88, mask=e_mask, other=0.0)[:, None]

    gw44_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw45_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw46_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw47_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw48_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw54_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw55_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw56_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw57_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw58_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw64_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw65_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw66_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw67_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw68_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw74_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw75_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw76_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw77_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw78_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw84_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw85_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw86_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw87_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw88_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]

        g4 = tl.load(g_out_ptr + j[:, None] * stride_xn + 4 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g5 = tl.load(g_out_ptr + j[:, None] * stride_xn + 5 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g6 = tl.load(g_out_ptr + j[:, None] * stride_xn + 6 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g7 = tl.load(g_out_ptr + j[:, None] * stride_xn + 7 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)
        g8 = tl.load(g_out_ptr + j[:, None] * stride_xn + 8 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0)

        z_base_m0 = e_offs[:, None] * Z_M0_MULT * C + c_offs[None, :]
        z_base_m1 = e_offs[:, None] * Z_M1_MULT * C + c_offs[None, :]
        z_base_m2 = e_offs[:, None] * Z_M2_MULT * C + c_offs[None, :]

        Z0_2  = tl.load(Z_m0_ptr + z_base_m0 + 2*C, mask=full_mask, other=0.0)
        Z1R_1 = tl.load(Z_m1_ptr + z_base_m1 + 1*C, mask=full_mask, other=0.0)
        Z1I_1 = tl.load(Z_m1_ptr + z_base_m1 + 3*C, mask=full_mask, other=0.0)
        Z2R_0 = tl.load(Z_m2_ptr + z_base_m2 + 0*C, mask=full_mask, other=0.0)
        Z2I_0 = tl.load(Z_m2_ptr + z_base_m2 + 1*C, mask=full_mask, other=0.0)

        aL2_4 = g4 * env_e; aL2_5 = g5 * env_e; aL2_6 = g6 * env_e; aL2_7 = g7 * env_e; aL2_8 = g8 * env_e
        
        tl.store(g_z_m0_ptr + z_base_m0 + 2*C, aL2_4*w64 + aL2_5*w65 + aL2_6*w66 + aL2_7*w67 + aL2_8*w68, mask=full_mask)
        tl.store(g_z_m1_ptr + z_base_m1 + 1*C, aL2_4*w74 + aL2_5*w75 + aL2_6*w76 + aL2_7*w77 + aL2_8*w78, mask=full_mask)
        tl.store(g_z_m1_ptr + z_base_m1 + 3*C, aL2_4*w54 + aL2_5*w55 + aL2_6*w56 + aL2_7*w57 + aL2_8*w58, mask=full_mask)
        tl.store(g_z_m2_ptr + z_base_m2 + 0*C, aL2_4*w84 + aL2_5*w85 + aL2_6*w86 + aL2_7*w87 + aL2_8*w88, mask=full_mask)
        tl.store(g_z_m2_ptr + z_base_m2 + 1*C, aL2_4*w44 + aL2_5*w45 + aL2_6*w46 + aL2_7*w47 + aL2_8*w48, mask=full_mask)

        gw44_acc += tl.sum(g4 * Z2I_0, axis=1); gw45_acc += tl.sum(g5 * Z2I_0, axis=1); gw46_acc += tl.sum(g6 * Z2I_0, axis=1); gw47_acc += tl.sum(g7 * Z2I_0, axis=1); gw48_acc += tl.sum(g8 * Z2I_0, axis=1)
        gw54_acc += tl.sum(g4 * Z1I_1, axis=1); gw55_acc += tl.sum(g5 * Z1I_1, axis=1); gw56_acc += tl.sum(g6 * Z1I_1, axis=1); gw57_acc += tl.sum(g7 * Z1I_1, axis=1); gw58_acc += tl.sum(g8 * Z1I_1, axis=1)
        gw64_acc += tl.sum(g4 * Z0_2,  axis=1); gw65_acc += tl.sum(g5 * Z0_2,  axis=1); gw66_acc += tl.sum(g6 * Z0_2,  axis=1); gw67_acc += tl.sum(g7 * Z0_2,  axis=1); gw68_acc += tl.sum(g8 * Z0_2,  axis=1)
        gw74_acc += tl.sum(g4 * Z1R_1, axis=1); gw75_acc += tl.sum(g5 * Z1R_1, axis=1); gw76_acc += tl.sum(g6 * Z1R_1, axis=1); gw77_acc += tl.sum(g7 * Z1R_1, axis=1); gw78_acc += tl.sum(g8 * Z1R_1, axis=1)
        gw84_acc += tl.sum(g4 * Z2R_0, axis=1); gw85_acc += tl.sum(g5 * Z2R_0, axis=1); gw86_acc += tl.sum(g6 * Z2R_0, axis=1); gw87_acc += tl.sum(g7 * Z2R_0, axis=1); gw88_acc += tl.sum(g8 * Z2R_0, axis=1)

        g_env_acc += tl.sum(g4 * (w64*Z0_2 + w54*Z1I_1 + w74*Z1R_1 + w44*Z2I_0 + w84*Z2R_0) +
                            g5 * (w65*Z0_2 + w55*Z1I_1 + w75*Z1R_1 + w45*Z2I_0 + w85*Z2R_0) +
                            g6 * (w66*Z0_2 + w56*Z1I_1 + w76*Z1R_1 + w46*Z2I_0 + w86*Z2R_0) +
                            g7 * (w67*Z0_2 + w57*Z1I_1 + w77*Z1R_1 + w47*Z2I_0 + w87*Z2R_0) +
                            g8 * (w68*Z0_2 + w58*Z1I_1 + w78*Z1R_1 + w48*Z2I_0 + w88*Z2R_0), axis=1)

    tl.store(g_wig_ptr + wb + W44, gw44_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W45, gw45_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W46, gw46_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W47, gw47_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W48, gw48_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W54, gw54_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W55, gw55_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W56, gw56_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W57, gw57_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W58, gw58_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W64, gw64_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W65, gw65_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W66, gw66_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W67, gw67_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W68, gw68_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W74, gw74_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W75, gw75_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W76, gw76_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W77, gw77_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W78, gw78_acc * env, mask=e_mask)
    tl.store(g_wig_ptr + wb + W84, gw84_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W85, gw85_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W86, gw86_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W87, gw87_acc * env, mask=e_mask); tl.store(g_wig_ptr + wb + W88, gw88_acc * env, mask=e_mask)
    tl.store(g_env_l2_ptr + e_offs, g_env_acc, mask=e_mask)


class RotateScatterSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z_m0, Z_m1, Z_m2, wigner, env, x_res, idx_j, row_ptr, col_idx, N):
        E = Z_m0.shape[0]
        C = Z_m0.shape[1] // Z_M0_MULT.value
        device = Z_m0.device
        dtype = Z_m0.dtype
        x_out = torch.empty(N, 9, C, device=device, dtype=dtype)

        grid = lambda META: (N, triton.cdiv(C, META['BLOCK_C']))
        _scatter_split_fwd_kernel[grid](
            row_ptr, col_idx,
            Z_m0, Z_m1, Z_m2, wigner, env, x_res, x_out,
            N, C,
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
            wigner.stride(0)
        )

        ctx.save_for_backward(Z_m0, Z_m1, Z_m2, wigner, env, idx_j)
        ctx.C = C
        return x_out

    @staticmethod
    def backward(ctx, g_out):
        Z_m0, Z_m1, Z_m2, wigner, env, idx_j = ctx.saved_tensors
        E = Z_m0.shape[0]; C = ctx.C

        g_out = g_out.contiguous()
        g_Z_m0 = torch.empty_like(Z_m0, memory_format=torch.contiguous_format)
        g_Z_m1 = torch.empty_like(Z_m1, memory_format=torch.contiguous_format)
        g_Z_m2 = torch.empty_like(Z_m2, memory_format=torch.contiguous_format)
        
        g_wigner = torch.empty_like(wigner, memory_format=torch.contiguous_format)
        g_env_l1 = torch.empty_like(env, memory_format=torch.contiguous_format)
        g_env_l2 = torch.empty_like(env, memory_format=torch.contiguous_format)

        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']),)
        
        _scatter_split_bwd_l1_kernel[grid](
            g_out, idx_j, 
            Z_m0, Z_m1, Z_m2, wigner, env,
            g_Z_m0, g_Z_m1, g_Z_m2, 
            g_wigner, g_env_l1,
            E, C,
            g_out.stride(0), g_out.stride(1), g_out.stride(2), wigner.stride(0)
        )
        
        _scatter_split_bwd_l2_kernel[grid](
            g_out, idx_j, 
            Z_m0, Z_m1, Z_m2, wigner, env,
            g_Z_m0, g_Z_m1, g_Z_m2, 
            g_wigner, g_env_l2,
            E, C,
            g_out.stride(0), g_out.stride(1), g_out.stride(2), wigner.stride(0)
        )

        g_env = g_env_l1 + g_env_l2
        return g_Z_m0, g_Z_m1, g_Z_m2, g_wigner, g_env, g_out, None, None, None, None

rotate_scatter_split = RotateScatterSplit.apply


# =========================================================================
# PHASE 1 FORWARD/BACKWARD: Init Node Scatter (with FP64 Accumulation)
# =========================================================================
@triton.autotune(configs=_generate_fwd_configs(), key=[ 'C'])
@triton.jit
def _init_scatter_fwd_kernel(
    row_ptr, col_idx, rad_out_ptr, wig_ptr, env_ptr,
    Z_ptr, batch_ptr, W_sphere_ptr, csd_emb_ptr, x_out_ptr,
    N, C, stride_rad_e, stride_rad_c, stride_wm_e, stride_ws_z, stride_ws_c,
    stride_ce_b, stride_ce_c, stride_xn, stride_xl, stride_xc,
    BLOCK_C: tl.constexpr, BLOCK_NB: tl.constexpr,
):
    atom_j = tl.program_id(0)
    c_block = tl.program_id(1)
    if atom_j >= N: return

    nb_start = tl.load(row_ptr + atom_j).to(tl.int32)
    nb_end = tl.load(row_ptr + atom_j + 1).to(tl.int32)

    c_offs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    z_val = tl.load(Z_ptr + atom_j)
    batch_val = tl.load(batch_ptr + atom_j)

    w_sph = tl.load(W_sphere_ptr + z_val * stride_ws_z + c_offs * stride_ws_c, mask=c_mask, other=0.0)
    c_emb = tl.load(csd_emb_ptr + batch_val * stride_ce_b + c_offs * stride_ce_c, mask=c_mask, other=0.0)

    # Initialize FP64 accumulators
    acc_l0 = (w_sph + c_emb).to(tl.float64)
    acc_l1_1 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l1_2 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l1_3 = tl.zeros([BLOCK_C], dtype=tl.float64)
    acc_l2_4 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l2_5 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l2_6 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l2_7 = tl.zeros([BLOCK_C], dtype=tl.float64); acc_l2_8 = tl.zeros([BLOCK_C], dtype=tl.float64)

    nb_cur = nb_start
    while nb_cur < nb_end:
        nb_offs = nb_cur + tl.arange(0, BLOCK_NB)
        nb_mask = nb_offs < nb_end
        e_pos = tl.load(col_idx + nb_offs, mask=nb_mask, other=0).to(tl.int32)

        env = tl.load(env_ptr + e_pos, mask=nb_mask, other=0.0)[:, None]
        wb = e_pos * stride_wm_e

        w21 = tl.load(wig_ptr + wb + W21, mask=nb_mask, other=0.0)[:, None]; w22 = tl.load(wig_ptr + wb + W22, mask=nb_mask, other=0.0)[:, None]; w23 = tl.load(wig_ptr + wb + W23, mask=nb_mask, other=0.0)[:, None]
        w64 = tl.load(wig_ptr + wb + W64, mask=nb_mask, other=0.0)[:, None]; w65 = tl.load(wig_ptr + wb + W65, mask=nb_mask, other=0.0)[:, None]; w66 = tl.load(wig_ptr + wb + W66, mask=nb_mask, other=0.0)[:, None]; w67 = tl.load(wig_ptr + wb + W67, mask=nb_mask, other=0.0)[:, None]; w68 = tl.load(wig_ptr + wb + W68, mask=nb_mask, other=0.0)[:, None]

        full_mask = nb_mask[:, None] & c_mask[None, :]
        rb = e_pos[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c
        r0 = tl.load(rad_out_ptr + rb + 0*C, mask=full_mask, other=0.0)
        r1 = tl.load(rad_out_ptr + rb + 1*C, mask=full_mask, other=0.0)
        r2 = tl.load(rad_out_ptr + rb + 2*C, mask=full_mask, other=0.0)

        # FP64 Accumulation Loop
        acc_l0   += tl.sum((r0 * env) / 5.0, axis=0).to(tl.float64)
        acc_l1_1 += tl.sum((r1 * w21 * env) / 5.0, axis=0).to(tl.float64); acc_l1_2 += tl.sum((r1 * w22 * env) / 5.0, axis=0).to(tl.float64); acc_l1_3 += tl.sum((r1 * w23 * env) / 5.0, axis=0).to(tl.float64)
        acc_l2_4 += tl.sum((r2 * w64 * env) / 5.0, axis=0).to(tl.float64); acc_l2_5 += tl.sum((r2 * w65 * env) / 5.0, axis=0).to(tl.float64); acc_l2_6 += tl.sum((r2 * w66 * env) / 5.0, axis=0).to(tl.float64); acc_l2_7 += tl.sum((r2 * w67 * env) / 5.0, axis=0).to(tl.float64); acc_l2_8 += tl.sum((r2 * w68 * env) / 5.0, axis=0).to(tl.float64)
        nb_cur += BLOCK_NB

    # Cast back to FP32 only when storing
    tl.store(x_out_ptr + atom_j * stride_xn + 0 * stride_xl + c_offs * stride_xc, acc_l0.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 1 * stride_xl + c_offs * stride_xc, acc_l1_1.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 2 * stride_xl + c_offs * stride_xc, acc_l1_2.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 3 * stride_xl + c_offs * stride_xc, acc_l1_3.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 4 * stride_xl + c_offs * stride_xc, acc_l2_4.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 5 * stride_xl + c_offs * stride_xc, acc_l2_5.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 6 * stride_xl + c_offs * stride_xc, acc_l2_6.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 7 * stride_xl + c_offs * stride_xc, acc_l2_7.to(tl.float32), mask=c_mask)
    tl.store(x_out_ptr + atom_j * stride_xn + 8 * stride_xl + c_offs * stride_xc, acc_l2_8.to(tl.float32), mask=c_mask)


@triton.autotune(configs=_generate_bwd_configs(), key=['C'])
@triton.jit
def _init_scatter_bwd_kernel(
    g_out_ptr, idx_j_ptr, rad_out_ptr, wig_ptr, env_ptr,
    g_rad_ptr, g_wig_ptr, g_env_ptr, E, C,
    stride_xn, stride_xl, stride_xc, stride_rad_e, stride_rad_c, stride_wm_e,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int64)
    env_1d = tl.load(env_ptr + e_offs, mask=e_mask, other=0.0)
    env = env_1d[:, None]
    wb = e_offs * stride_wm_e

    w21 = tl.load(wig_ptr + wb + W21, mask=e_mask, other=0.0)[:, None]; w22 = tl.load(wig_ptr + wb + W22, mask=e_mask, other=0.0)[:, None]; w23 = tl.load(wig_ptr + wb + W23, mask=e_mask, other=0.0)[:, None]
    w64 = tl.load(wig_ptr + wb + W64, mask=e_mask, other=0.0)[:, None]; w65 = tl.load(wig_ptr + wb + W65, mask=e_mask, other=0.0)[:, None]; w66 = tl.load(wig_ptr + wb + W66, mask=e_mask, other=0.0)[:, None]; w67 = tl.load(wig_ptr + wb + W67, mask=e_mask, other=0.0)[:, None]; w68 = tl.load(wig_ptr + wb + W68, mask=e_mask, other=0.0)[:, None]

    gw21_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw22_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw23_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    gw64_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw65_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw66_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw67_acc = tl.zeros([BLOCK_E], dtype=tl.float32); gw68_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    g_env_acc = tl.zeros([BLOCK_E], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        full_mask = e_mask[:, None] & c_mask[None, :]
        rb = e_offs[:, None] * stride_rad_e + c_offs[None, :] * stride_rad_c

        g0 = tl.load(g_out_ptr + j[:, None] * stride_xn + 0 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g1 = tl.load(g_out_ptr + j[:, None] * stride_xn + 1 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g2 = tl.load(g_out_ptr + j[:, None] * stride_xn + 2 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g3 = tl.load(g_out_ptr + j[:, None] * stride_xn + 3 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g4 = tl.load(g_out_ptr + j[:, None] * stride_xn + 4 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g5 = tl.load(g_out_ptr + j[:, None] * stride_xn + 5 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g6 = tl.load(g_out_ptr + j[:, None] * stride_xn + 6 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g7 = tl.load(g_out_ptr + j[:, None] * stride_xn + 7 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0
        g8 = tl.load(g_out_ptr + j[:, None] * stride_xn + 8 * stride_xl + c_offs[None, :] * stride_xc, mask=full_mask, other=0.0) / 5.0

        r1 = tl.load(rad_out_ptr + rb + 1*C, mask=full_mask, other=0.0)
        r2 = tl.load(rad_out_ptr + rb + 2*C, mask=full_mask, other=0.0)

        tl.store(g_rad_ptr + rb + 0*C, g0 * env, mask=full_mask)
        tl.store(g_rad_ptr + rb + 1*C, (g1*w21 + g2*w22 + g3*w23) * env, mask=full_mask)
        tl.store(g_rad_ptr + rb + 2*C, (g4*w64 + g5*w65 + g6*w66 + g7*w67 + g8*w68) * env, mask=full_mask)

        gw21_acc += tl.sum(g1 * r1, axis=1); gw22_acc += tl.sum(g2 * r1, axis=1); gw23_acc += tl.sum(g3 * r1, axis=1)
        gw64_acc += tl.sum(g4 * r2, axis=1); gw65_acc += tl.sum(g5 * r2, axis=1); gw66_acc += tl.sum(g6 * r2, axis=1); gw67_acc += tl.sum(g7 * r2, axis=1); gw68_acc += tl.sum(g8 * r2, axis=1)

        r0 = tl.load(rad_out_ptr + rb + 0*C, mask=full_mask, other=0.0)
        g_env_acc += tl.sum(g0*r0 + (g1*w21 + g2*w22 + g3*w23)*r1 + (g4*w64 + g5*w65 + g6*w66 + g7*w67 + g8*w68)*r2, axis=1)

    tl.store(g_wig_ptr + wb + W21, gw21_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W22, gw22_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W23, gw23_acc * env_1d, mask=e_mask)
    tl.store(g_wig_ptr + wb + W64, gw64_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W65, gw65_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W66, gw66_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W67, gw67_acc * env_1d, mask=e_mask); tl.store(g_wig_ptr + wb + W68, gw68_acc * env_1d, mask=e_mask)
    tl.store(g_env_ptr + e_offs, g_env_acc, mask=e_mask)


class InitNodeScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rad_out, wigner, env, Z, batch, W_sphere, csd_emb, idx_j, row_ptr, col_idx, N):
        E = rad_out.shape[0]; C = rad_out.shape[1] // 3
        x_out = torch.empty(N, 9, C, device=rad_out.device, dtype=rad_out.dtype)

        grid = lambda META: (N, triton.cdiv(C, META['BLOCK_C']))
        _init_scatter_fwd_kernel[grid](
            row_ptr, col_idx, rad_out, wigner, env,
            Z, batch, W_sphere, csd_emb, x_out,
            N, C, rad_out.stride(0), 1, wigner.stride(0),
            W_sphere.stride(0), W_sphere.stride(1),
            csd_emb.stride(0), csd_emb.stride(1),
            x_out.stride(0), x_out.stride(1), x_out.stride(2)
        )
        ctx.save_for_backward(rad_out, wigner, env, idx_j, Z, batch)
        ctx.num_z = W_sphere.shape[0]; ctx.num_batches = csd_emb.shape[0]; ctx.C = C
        return x_out

    @staticmethod
    def backward(ctx, g_out):
        rad_out, wigner, env, idx_j, Z, batch = ctx.saved_tensors
        E = rad_out.shape[0]; C = ctx.C
        g_out = g_out.contiguous()

        g_rad_out = torch.empty_like(rad_out)
        g_wigner = torch.zeros_like(wigner)
        g_env = torch.empty_like(env)

        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']),)
        _init_scatter_bwd_kernel[grid](
            g_out, idx_j, rad_out, wigner, env, g_rad_out, g_wigner, g_env, E, C,
            g_out.stride(0), g_out.stride(1), g_out.stride(2), g_rad_out.stride(0), 1, wigner.stride(0)
        )
        
        g_out_l0 = g_out[:, 0, :].contiguous()
        g_W_sphere = torch.zeros((ctx.num_z, C), device=g_out.device, dtype=g_out.dtype)
        g_W_sphere.index_add_(0, Z, g_out_l0)
        g_csd_emb = torch.zeros((ctx.num_batches, C), device=g_out.device, dtype=g_out.dtype)
        g_csd_emb.index_add_(0, batch, g_out_l0)

        return g_rad_out, g_wigner, g_env, None, None, g_W_sphere, g_csd_emb, None, None, None, None

init_node_scatter = InitNodeScatter.apply