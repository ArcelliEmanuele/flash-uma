"""
Analytical Geometry & Precomputation.
Calculates Distances, Gaussians, Embeddings, Quaternions, and 34 Wigner Scalars.
Executes exactly ONCE per forward pass.
"""

import torch
import triton
import triton.language as tl
import itertools

from ops.constants import (
    W11, W12, W13, W21, W22, W23, W31, W32, W33,
    W44, W45, W46, W47, W48, W54, W55, W56, W57, W58,
    W64, W65, W66, W67, W68, W74, W75, W76, W77, W78,
    W84, W85, W86, W87, W88,
    S3, S3X2, S3X3, S3X4
)

def _generate_configs():
    configs = []
    for e in [16, 32, 64]:
        for w in [4, 8]:
            for s in [1, 2]:
                configs.append(triton.Config({'BLOCK_E': e}, num_warps=w, num_stages=s))
    return configs

def _np2(x):
    p = 1
    while p < x: p <<= 1
    return p

# =========================================================================
# FORWARD KERNEL
# =========================================================================
@triton.autotune(configs=_generate_configs(), key=['B', 'C'])
@triton.jit
def _geom_fwd_kernel(
    pos_ptr, Z_ptr, idx_i_ptr, idx_j_ptr, shifts_ptr,
    W_src_ptr, W_tgt_ptr, offset_g_ptr,
    x_edge_ptr, wig_ptr, env_ptr,
    E, B, C,
    cutoff, coeff_g,
    stride_pos_n, stride_shift_e, stride_shift_c,
    stride_Wsrc_z, stride_Wsrc_c,
    stride_Wtgt_z, stride_Wtgt_c,
    stride_xe_e, stride_xe_c,
    stride_wig_e, stride_wig_k,
    BLOCK_E: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    idx_i = tl.load(idx_i_ptr + e_offs, mask=e_mask, other=0).to(tl.int32)
    idx_j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int32)
    z_i = tl.load(Z_ptr + idx_i, mask=e_mask, other=0).to(tl.int32)
    z_j = tl.load(Z_ptr + idx_j, mask=e_mask, other=0).to(tl.int32)

    pi_x = tl.load(pos_ptr + idx_i * stride_pos_n + 0, mask=e_mask, other=0.0)
    pi_y = tl.load(pos_ptr + idx_i * stride_pos_n + 1, mask=e_mask, other=0.0)
    pi_z = tl.load(pos_ptr + idx_i * stride_pos_n + 2, mask=e_mask, other=0.0)
    pj_x = tl.load(pos_ptr + idx_j * stride_pos_n + 0, mask=e_mask, other=0.0)
    pj_y = tl.load(pos_ptr + idx_j * stride_pos_n + 1, mask=e_mask, other=0.0)
    pj_z = tl.load(pos_ptr + idx_j * stride_pos_n + 2, mask=e_mask, other=0.0)
    
    # Load PBC shifts and apply to edge vector
    shift_x = tl.load(shifts_ptr + e_offs * stride_shift_e + 0 * stride_shift_c, mask=e_mask, other=0.0)
    shift_y = tl.load(shifts_ptr + e_offs * stride_shift_e + 1 * stride_shift_c, mask=e_mask, other=0.0)
    shift_z = tl.load(shifts_ptr + e_offs * stride_shift_e + 2 * stride_shift_c, mask=e_mask, other=0.0)
    
    dx = pi_x - pj_x + shift_x
    dy = pi_y - pj_y + shift_y
    dz = pi_z - pj_z + shift_z
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)
    dist_safe = tl.maximum(dist, 1e-7)

    d_sc = dist / cutoff
    d_sc_2 = d_sc * d_sc
    d_sc_5 = d_sc_2 * d_sc_2 * d_sc
    env = 1.0 + d_sc_5 * (-21.0 + d_sc * (35.0 - 15.0 * d_sc))
    env = tl.where(d_sc < 1.0, env, 0.0)
    tl.store(env_ptr + e_offs, env, mask=e_mask)

    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < B
    offset_g_vals = tl.load(offset_g_ptr + b_offs, mask=b_mask, other=0.0)
    
    full_mask_b = e_mask[:, None] & b_mask[None, :]
    diff = dist[:, None] - offset_g_vals[None, :]
    gauss = tl.exp(coeff_g * diff * diff)
    
    xe_base = e_offs[:, None] * stride_xe_e
    tl.store(x_edge_ptr + xe_base + b_offs[None, :] * stride_xe_c, gauss, mask=full_mask_b)

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    full_mask_c = e_mask[:, None] & c_mask[None, :]
    
    src_emb = tl.load(W_src_ptr + z_i[:, None] * stride_Wsrc_z + c_offs[None, :] * stride_Wsrc_c, mask=full_mask_c, other=0.0)
    tgt_emb = tl.load(W_tgt_ptr + z_j[:, None] * stride_Wtgt_z + c_offs[None, :] * stride_Wtgt_c, mask=full_mask_c, other=0.0)
    
    tl.store(x_edge_ptr + xe_base + (B + c_offs)[None, :] * stride_xe_c, src_emb, mask=full_mask_c)
    tl.store(x_edge_ptr + xe_base + (B + C + c_offs)[None, :] * stride_xe_c, tgt_emb, mask=full_mask_c)

    eps = 1e-7
    ex = dx / dist_safe; ey = dy / dist_safe; ez = dz / dist_safe
    n1 = tl.sqrt((1.0 + ey) * (1.0 + ey) + ez * ez + ex * ex + eps)
    q1w = (1.0 + ey) / n1; q1x = -ez / n1; q1z = ex / n1
    n2 = tl.sqrt(ez * ez + (1.0 - ey) * (1.0 - ey) + ex * ex + eps)
    q2w = -ez / n2; q2x = (1.0 - ey) / n2; q2y = ex / n2

    t = tl.minimum(tl.maximum((ey + 0.9) / 1.8, 0.0), 1.0)
    num = 2.0 * t - 1.0
    den = tl.maximum(t * (1.0 - t), eps)
    s = tl.sigmoid(num / den)
    s = tl.where(t < eps, 0.0, s)
    s = tl.where(t > 1.0 - eps, 1.0, s)

    dot_q = q1w * q2w + q1x * q2x 
    flip = dot_q < 0.0
    q1w = tl.where(flip, -q1w, q1w)
    q1x = tl.where(flip, -q1x, q1x)
    q1z = tl.where(flip, -q1z, q1z)

    qw_e = (1.0 - s) * q2w + s * q1w
    qx_e = (1.0 - s) * q2x + s * q1x
    qy_e = (1.0 - s) * q2y         
    qz_e = s * q1z                  
    ne = tl.sqrt(qw_e * qw_e + qx_e * qx_e + qy_e * qy_e + qz_e * qz_e + eps)
    qw = qw_e / ne; qx = qx_e / ne; qy = qy_e / ne; qz = qz_e / ne

    qxx = qx * qx; qxy = qx * qy; qxz = qx * qz
    qyy = qy * qy; qyz = qy * qz; qzz = qz * qz
    qwx = qw * qx; qwy = qw * qy; qwz = qw * qz; qww = qw * qw

    w11 = 1.0 - 2.0 * (qyy + qzz); w12 = 2.0 * (qxy - qwz); w13 = 2.0 * (qxz + qwy)
    w21 = 2.0 * (qxy + qwz);       w22 = 1.0 - 2.0 * (qxx + qzz); w23 = 2.0 * (qyz - qwx)
    w31 = 2.0 * (qxz - qwy);       w32 = 2.0 * (qyz + qwx);       w33 = 1.0 - 2.0 * (qxx + qyy)

    m_w4 = qww * qww
    m_w3x = qww * qwx; m_w3y = qww * qwy; m_w3z = qww * qwz
    m_w2x2 = qww * qxx; m_w2xy = qww * qxy; m_w2xz = qww * qxz
    m_w2y2 = qww * qyy; m_w2yz = qww * qyz; m_w2z2 = qww * qzz
    m_wx3 = qwx * qxx; m_wx2y = qwx * qxy; m_wx2z = qwx * qxz
    m_wxy2 = qwx * qyy; m_wxyz = qwx * qyz; m_wxz2 = qwx * qzz
    m_wy3 = qwy * qyy; m_wy2z = qwy * qyz; m_wyz2 = qwy * qzz; m_wz3 = qwz * qzz
    m_x4 = qxx * qxx; m_x3y = qxx * qxy; m_x3z = qxx * qxz
    m_x2y2 = qxx * qyy; m_x2yz = qxx * qyz; m_x2z2 = qxx * qzz
    m_xy3 = qxy * qyy; m_xy2z = qxy * qyz; m_xyz2 = qxy * qzz; m_xz3 = qxz * qzz
    m_y4 = qyy * qyy; m_y3z = qyy * qyz; m_y2z2 = qyy * qzz
    m_yz3 = qyz * qzz; m_z4 = qzz * qzz

    w44 = m_w4 - 6.0*m_w2y2 - m_x4 + 6.0*m_x2z2 + m_y4 - m_z4
    w45 = 2.0*m_w3x + 6.0*m_w2yz + 2.0*m_wx3 - 6.0*m_wxy2 - 6.0*m_wxz2 + 6.0*m_x2yz - 2.0*m_y3z - 2.0*m_yz3
    w46 = -S3X2*m_w2xz + S3X2*m_wx2y - S3X2*m_wyz2 + S3X2*m_xy2z
    w47 = -2.0*m_w3z + 6.0*m_w2xy + 6.0*m_wx2z + 6.0*m_wy2z - 2.0*m_wz3 - 2.0*m_x3y - 2.0*m_xy3 + 6.0*m_xyz2
    w48 = 4.0*m_w3y - 4.0*m_wy3 - 4.0*m_x3z + 4.0*m_xz3
    w54 = -2.0*m_w3x + 6.0*m_w2yz - 2.0*m_wx3 + 6.0*m_wxy2 + 6.0*m_wxz2 + 6.0*m_x2yz - 2.0*m_y3z - 2.0*m_yz3
    w55 = m_w4 - 6.0*m_w2z2 - m_x4 + 6.0*m_x2y2 - m_y4 + m_z4
    w56 = -S3*m_w3z + S3*m_w2xy + S3*m_wx2z - S3*m_wy2z + S3*m_wz3 - S3*m_x3y + S3*m_xy3 - S3*m_xyz2
    w57 = 2.0*m_w3y + 6.0*m_w2xz - 6.0*m_wx2y + 2.0*m_wy3 - 6.0*m_wyz2 - 2.0*m_x3z + 6.0*m_xy2z - 2.0*m_xz3
    w58 = -2.0*m_w3z - 6.0*m_w2xy - 6.0*m_wx2z + 6.0*m_wy2z + 2.0*m_wz3 - 2.0*m_x3y + 2.0*m_xy3 + 6.0*m_xyz2
    w64 = -S3X2*m_w2xz - S3X2*m_wx2y + S3X2*m_wyz2 + S3X2*m_xy2z
    w65 = S3*m_w3z + S3*m_w2xy - S3*m_wx2z + S3*m_wy2z - S3*m_wz3 - S3*m_x3y + S3*m_xy3 - S3*m_xyz2
    w66 = (m_w4 - 4.0*m_w2x2 + 2.0*m_w2y2 - 4.0*m_w2z2 + m_x4 - 4.0*m_x2y2 + 2.0*m_x2z2 + m_y4 - 4.0*m_y2z2 + m_z4)
    w67 = -S3*m_w3x + S3*m_w2yz + S3*m_wx3 - S3*m_wxy2 + S3*m_wxz2 - S3*m_x2yz + S3*m_y3z - S3*m_yz3
    w68 = S3*m_w2x2 - S3*m_w2z2 - S3X4*m_wxyz - S3*m_x2y2 + S3*m_y2z2
    w74 = 2.0*m_w3z + 6.0*m_w2xy - 6.0*m_wx2z - 6.0*m_wy2z + 2.0*m_wz3 - 2.0*m_x3y - 2.0*m_xy3 + 6.0*m_xyz2
    w75 = -2.0*m_w3y + 6.0*m_w2xz + 6.0*m_wx2y - 2.0*m_wy3 + 6.0*m_wyz2 - 2.0*m_x3z + 6.0*m_xy2z - 2.0*m_xz3
    w76 = S3*m_w3x + S3*m_w2yz - S3*m_wx3 + S3*m_wxy2 - S3*m_wxz2 - S3*m_x2yz + S3*m_y3z - S3*m_yz3
    w77 = m_w4 - 6.0*m_w2x2 + m_x4 - m_y4 + 6.0*m_y2z2 - m_z4
    w78 = -2.0*m_w3x + 6.0*m_w2yz + 2.0*m_wx3 + 6.0*m_wxy2 - 6.0*m_wxz2 - 6.0*m_x2yz - 2.0*m_y3z + 2.0*m_yz3
    w84 = -4.0*m_w3y + 4.0*m_wy3 - 4.0*m_x3z + 4.0*m_xz3
    w85 = 2.0*m_w3z - 6.0*m_w2xy + 6.0*m_wx2z - 6.0*m_wy2z - 2.0*m_wz3 - 2.0*m_x3y + 2.0*m_xy3 + 6.0*m_xyz2
    w86 = S3*m_w2x2 - S3*m_w2z2 + S3X4*m_wxyz - S3*m_x2y2 + S3*m_y2z2
    w87 = 2.0*m_w3x + 6.0*m_w2yz - 2.0*m_wx3 - 6.0*m_wxy2 + 6.0*m_wxz2 - 6.0*m_x2yz - 2.0*m_y3z + 2.0*m_yz3
    w88 = m_w4 - 6.0*m_w2y2 + m_x4 - 6.0*m_x2z2 + m_y4 + m_z4

    wb = e_offs * stride_wig_e
    tl.store(wig_ptr + wb + W11 * stride_wig_k, w11, mask=e_mask)
    tl.store(wig_ptr + wb + W12 * stride_wig_k, w12, mask=e_mask)
    tl.store(wig_ptr + wb + W13 * stride_wig_k, w13, mask=e_mask)
    tl.store(wig_ptr + wb + W21 * stride_wig_k, w21, mask=e_mask)
    tl.store(wig_ptr + wb + W22 * stride_wig_k, w22, mask=e_mask)
    tl.store(wig_ptr + wb + W23 * stride_wig_k, w23, mask=e_mask)
    tl.store(wig_ptr + wb + W31 * stride_wig_k, w31, mask=e_mask)
    tl.store(wig_ptr + wb + W32 * stride_wig_k, w32, mask=e_mask)
    tl.store(wig_ptr + wb + W33 * stride_wig_k, w33, mask=e_mask)
    tl.store(wig_ptr + wb + W44 * stride_wig_k, w44, mask=e_mask)
    tl.store(wig_ptr + wb + W45 * stride_wig_k, w45, mask=e_mask)
    tl.store(wig_ptr + wb + W46 * stride_wig_k, w46, mask=e_mask)
    tl.store(wig_ptr + wb + W47 * stride_wig_k, w47, mask=e_mask)
    tl.store(wig_ptr + wb + W48 * stride_wig_k, w48, mask=e_mask)
    tl.store(wig_ptr + wb + W54 * stride_wig_k, w54, mask=e_mask)
    tl.store(wig_ptr + wb + W55 * stride_wig_k, w55, mask=e_mask)
    tl.store(wig_ptr + wb + W56 * stride_wig_k, w56, mask=e_mask)
    tl.store(wig_ptr + wb + W57 * stride_wig_k, w57, mask=e_mask)
    tl.store(wig_ptr + wb + W58 * stride_wig_k, w58, mask=e_mask)
    tl.store(wig_ptr + wb + W64 * stride_wig_k, w64, mask=e_mask)
    tl.store(wig_ptr + wb + W65 * stride_wig_k, w65, mask=e_mask)
    tl.store(wig_ptr + wb + W66 * stride_wig_k, w66, mask=e_mask)
    tl.store(wig_ptr + wb + W67 * stride_wig_k, w67, mask=e_mask)
    tl.store(wig_ptr + wb + W68 * stride_wig_k, w68, mask=e_mask)
    tl.store(wig_ptr + wb + W74 * stride_wig_k, w74, mask=e_mask)
    tl.store(wig_ptr + wb + W75 * stride_wig_k, w75, mask=e_mask)
    tl.store(wig_ptr + wb + W76 * stride_wig_k, w76, mask=e_mask)
    tl.store(wig_ptr + wb + W77 * stride_wig_k, w77, mask=e_mask)
    tl.store(wig_ptr + wb + W78 * stride_wig_k, w78, mask=e_mask)
    tl.store(wig_ptr + wb + W84 * stride_wig_k, w84, mask=e_mask)
    tl.store(wig_ptr + wb + W85 * stride_wig_k, w85, mask=e_mask)
    tl.store(wig_ptr + wb + W86 * stride_wig_k, w86, mask=e_mask)
    tl.store(wig_ptr + wb + W87 * stride_wig_k, w87, mask=e_mask)
    tl.store(wig_ptr + wb + W88 * stride_wig_k, w88, mask=e_mask)


# =========================================================================
# BACKWARD KERNEL (Now includes Virial/Stress logic via g_shifts)
# =========================================================================
@triton.autotune(configs=_generate_configs(), key=['B', 'C'])
@triton.jit
def _geom_bwd_kernel(
    pos_ptr, idx_i_ptr, idx_j_ptr, shifts_ptr, offset_g_ptr,
    g_x_edge_ptr, g_wig_ptr, g_env_ptr,
    g_pos_ptr, g_shifts_ptr,
    E, B, C,
    cutoff, coeff_g,
    stride_pos_n, stride_shift_e, stride_shift_c,
    stride_xe_e, stride_xe_c,
    stride_wig_e, stride_wig_k,
    BLOCK_E: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    idx_i = tl.load(idx_i_ptr + e_offs, mask=e_mask, other=0).to(tl.int32)
    idx_j = tl.load(idx_j_ptr + e_offs, mask=e_mask, other=0).to(tl.int32)

    pi_x = tl.load(pos_ptr + idx_i * stride_pos_n + 0, mask=e_mask, other=0.0)
    pi_y = tl.load(pos_ptr + idx_i * stride_pos_n + 1, mask=e_mask, other=0.0)
    pi_z = tl.load(pos_ptr + idx_i * stride_pos_n + 2, mask=e_mask, other=0.0)
    pj_x = tl.load(pos_ptr + idx_j * stride_pos_n + 0, mask=e_mask, other=0.0)
    pj_y = tl.load(pos_ptr + idx_j * stride_pos_n + 1, mask=e_mask, other=0.0)
    pj_z = tl.load(pos_ptr + idx_j * stride_pos_n + 2, mask=e_mask, other=0.0)
    
    shift_x = tl.load(shifts_ptr + e_offs * stride_shift_e + 0 * stride_shift_c, mask=e_mask, other=0.0)
    shift_y = tl.load(shifts_ptr + e_offs * stride_shift_e + 1 * stride_shift_c, mask=e_mask, other=0.0)
    shift_z = tl.load(shifts_ptr + e_offs * stride_shift_e + 2 * stride_shift_c, mask=e_mask, other=0.0)
    
    dx = pi_x - pj_x + shift_x
    dy = pi_y - pj_y + shift_y
    dz = pi_z - pj_z + shift_z
    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)
    dist_safe = tl.maximum(dist, 1e-7)
    d_sc = dist / cutoff

    eps = 1e-7
    ex = dx / dist_safe; ey = dy / dist_safe; ez = dz / dist_safe
    n1 = tl.sqrt((1.0 + ey) * (1.0 + ey) + ez * ez + ex * ex + eps)
    q1w_n = (1.0 + ey) / n1; q1x_n = -ez / n1; q1z_n = ex / n1
    n2 = tl.sqrt(ez * ez + (1.0 - ey) * (1.0 - ey) + ex * ex + eps)
    q2w = -ez / n2; q2x = (1.0 - ey) / n2; q2y = ex / n2

    t_env = tl.minimum(tl.maximum((ey + 0.9) / 1.8, 0.0), 1.0)
    num = 2.0 * t_env - 1.0; den = tl.maximum(t_env * (1.0 - t_env), eps)
    s_val = tl.sigmoid(num / den)
    s_val = tl.where(t_env < eps, 0.0, s_val)
    s_val = tl.where(t_env > 1.0 - eps, 1.0, s_val)

    dot_q = q1w_n * q2w + q1x_n * q2x 
    flip = dot_q < 0.0
    q1w_f = tl.where(flip, -q1w_n, q1w_n)
    q1x_f = tl.where(flip, -q1x_n, q1x_n)
    q1y_f = 0.0
    q1z_f = tl.where(flip, -q1z_n, q1z_n)

    qw_u = (1.0 - s_val) * q2w + s_val * q1w_f
    qx_u = (1.0 - s_val) * q2x + s_val * q1x_f
    qy_u = (1.0 - s_val) * q2y + s_val * q1y_f
    qz_u = s_val * q1z_f 
    ne = tl.sqrt(qw_u * qw_u + qx_u * qx_u + qy_u * qy_u + qz_u * qz_u + eps)
    qw = qw_u / ne; qx = qx_u / ne; qy = qy_u / ne; qz = qz_u / ne

    qxx = qx * qx; qxy = qx * qy; qxz = qx * qz
    qyy = qy * qy; qyz = qy * qz; qzz = qz * qz
    qwx = qw * qx; qwy = qw * qy; qwz = qw * qz; qww = qw * qw

    wb = e_offs * stride_wig_e
    g_w11 = tl.load(g_wig_ptr + wb + W11 * stride_wig_k, mask=e_mask, other=0.0)
    g_w12 = tl.load(g_wig_ptr + wb + W12 * stride_wig_k, mask=e_mask, other=0.0)
    g_w13 = tl.load(g_wig_ptr + wb + W13 * stride_wig_k, mask=e_mask, other=0.0)
    g_w21 = tl.load(g_wig_ptr + wb + W21 * stride_wig_k, mask=e_mask, other=0.0)
    g_w22 = tl.load(g_wig_ptr + wb + W22 * stride_wig_k, mask=e_mask, other=0.0)
    g_w23 = tl.load(g_wig_ptr + wb + W23 * stride_wig_k, mask=e_mask, other=0.0)
    g_w31 = tl.load(g_wig_ptr + wb + W31 * stride_wig_k, mask=e_mask, other=0.0)
    g_w32 = tl.load(g_wig_ptr + wb + W32 * stride_wig_k, mask=e_mask, other=0.0)
    g_w33 = tl.load(g_wig_ptr + wb + W33 * stride_wig_k, mask=e_mask, other=0.0)
    g_w44 = tl.load(g_wig_ptr + wb + W44 * stride_wig_k, mask=e_mask, other=0.0); g_w45 = tl.load(g_wig_ptr + wb + W45 * stride_wig_k, mask=e_mask, other=0.0)
    g_w46 = tl.load(g_wig_ptr + wb + W46 * stride_wig_k, mask=e_mask, other=0.0); g_w47 = tl.load(g_wig_ptr + wb + W47 * stride_wig_k, mask=e_mask, other=0.0)
    g_w48 = tl.load(g_wig_ptr + wb + W48 * stride_wig_k, mask=e_mask, other=0.0); g_w54 = tl.load(g_wig_ptr + wb + W54 * stride_wig_k, mask=e_mask, other=0.0)
    g_w55 = tl.load(g_wig_ptr + wb + W55 * stride_wig_k, mask=e_mask, other=0.0); g_w56 = tl.load(g_wig_ptr + wb + W56 * stride_wig_k, mask=e_mask, other=0.0)
    g_w57 = tl.load(g_wig_ptr + wb + W57 * stride_wig_k, mask=e_mask, other=0.0); g_w58 = tl.load(g_wig_ptr + wb + W58 * stride_wig_k, mask=e_mask, other=0.0)
    g_w64 = tl.load(g_wig_ptr + wb + W64 * stride_wig_k, mask=e_mask, other=0.0); g_w65 = tl.load(g_wig_ptr + wb + W65 * stride_wig_k, mask=e_mask, other=0.0)
    g_w66 = tl.load(g_wig_ptr + wb + W66 * stride_wig_k, mask=e_mask, other=0.0); g_w67 = tl.load(g_wig_ptr + wb + W67 * stride_wig_k, mask=e_mask, other=0.0)
    g_w68 = tl.load(g_wig_ptr + wb + W68 * stride_wig_k, mask=e_mask, other=0.0); g_w74 = tl.load(g_wig_ptr + wb + W74 * stride_wig_k, mask=e_mask, other=0.0)
    g_w75 = tl.load(g_wig_ptr + wb + W75 * stride_wig_k, mask=e_mask, other=0.0); g_w76 = tl.load(g_wig_ptr + wb + W76 * stride_wig_k, mask=e_mask, other=0.0)
    g_w77 = tl.load(g_wig_ptr + wb + W77 * stride_wig_k, mask=e_mask, other=0.0); g_w78 = tl.load(g_wig_ptr + wb + W78 * stride_wig_k, mask=e_mask, other=0.0)
    g_w84 = tl.load(g_wig_ptr + wb + W84 * stride_wig_k, mask=e_mask, other=0.0); g_w85 = tl.load(g_wig_ptr + wb + W85 * stride_wig_k, mask=e_mask, other=0.0)
    g_w86 = tl.load(g_wig_ptr + wb + W86 * stride_wig_k, mask=e_mask, other=0.0); g_w87 = tl.load(g_wig_ptr + wb + W87 * stride_wig_k, mask=e_mask, other=0.0)
    g_w88 = tl.load(g_wig_ptr + wb + W88 * stride_wig_k, mask=e_mask, other=0.0)

    g_qw_l1 = 2.0*qx*(g_w32 - g_w23) + 2.0*qy*(g_w13 - g_w31) + 2.0*qz*(g_w21 - g_w12)
    g_qx_l1 = 2.0*qw*(g_w32 - g_w23) + 2.0*qy*(g_w12 + g_w21) + 2.0*qz*(g_w13 + g_w31) - 4.0*qx*(g_w22 + g_w33)
    g_qy_l1 = 2.0*qw*(g_w13 - g_w31) + 2.0*qx*(g_w12 + g_w21) + 2.0*qz*(g_w23 + g_w32) - 4.0*qy*(g_w11 + g_w33)
    g_qz_l1 = 2.0*qw*(g_w21 - g_w12) + 2.0*qx*(g_w13 + g_w31) + 2.0*qy*(g_w23 + g_w32) - 4.0*qz*(g_w11 + g_w22)

    g_qw_l2 = (
        qww*qw*(4.0*g_w44 + 4.0*g_w55 + 4.0*g_w66 + 4.0*g_w77 + 4.0*g_w88)
        + qww*qx*(6.0*g_w45 - 6.0*g_w54 - S3X3*g_w67 + S3X3*g_w76 - 6.0*g_w78 + 6.0*g_w87)
        + qww*qy*(12.0*g_w48 + 6.0*g_w57 - 6.0*g_w75 - 12.0*g_w84)
        + qww*qz*(-6.0*g_w47 - S3X3*g_w56 - 6.0*g_w58 + S3X3*g_w65 + 6.0*g_w74 + 6.0*g_w85)
        + qwx*qy*(12.0*g_w47 + S3X2*g_w56 - 12.0*g_w58 + S3X2*g_w65 + 12.0*g_w74 - 12.0*g_w85)
        + qwx*qz*(-S3X4*g_w46 + 12.0*g_w57 - S3X4*g_w64 + 12.0*g_w75)
        + qwy*qz*(12.0*g_w45 + 12.0*g_w54 + S3X2*g_w67 + S3X2*g_w76 + 12.0*g_w78 + 12.0*g_w87)
        + qxx*qw*(-8.0*g_w66 + S3X2*g_w68 - 12.0*g_w77 + S3X2*g_w86)
        + qxx*qx*(2.0*g_w45 - 2.0*g_w54 + S3*g_w67 - S3*g_w76 + 2.0*g_w78 - 2.0*g_w87)
        + qxx*qy*(S3X2*g_w46 - 6.0*g_w57 - S3X2*g_w64 + 6.0*g_w75)
        + qxx*qz*(6.0*g_w47 + S3*g_w56 - 6.0*g_w58 - S3*g_w65 - 6.0*g_w74 + 6.0*g_w85)
        + qxy*qz*(-S3X4*g_w68 + S3X4*g_w86)
        + qyy*qw*(-12.0*g_w44 + 4.0*g_w66 - 12.0*g_w88)
        + qyy*qx*(-6.0*g_w45 + 6.0*g_w54 - S3*g_w67 + S3*g_w76 + 6.0*g_w78 - 6.0*g_w87)
        + qyy*qy*(-4.0*g_w48 + 2.0*g_w57 - 2.0*g_w75 + 4.0*g_w84)
        + qyy*qz*(6.0*g_w47 - S3*g_w56 + 6.0*g_w58 + S3*g_w65 - 6.0*g_w74 - 6.0*g_w85)
        + qzz*qw*(-12.0*g_w55 - 8.0*g_w66 - S3X2*g_w68 - S3X2*g_w86)
        + qzz*qx*(-6.0*g_w45 + 6.0*g_w54 + S3*g_w67 - S3*g_w76 - 6.0*g_w78 + 6.0*g_w87)
        + qzz*qy*(-S3X2*g_w46 - 6.0*g_w57 + S3X2*g_w64 + 6.0*g_w75)
        + qzz*qz*(-2.0*g_w47 + S3*g_w56 + 2.0*g_w58 - S3*g_w65 + 2.0*g_w74 - 2.0*g_w85)
    )

    g_qx_l2 = (
        qww*qw*(2.0*g_w45 - 2.0*g_w54 - S3*g_w67 + S3*g_w76 - 2.0*g_w78 + 2.0*g_w87)
        + qww*qx*(-8.0*g_w66 + S3X2*g_w68 - 12.0*g_w77 + S3X2*g_w86)
        + qww*qy*(6.0*g_w47 + S3*g_w56 - 6.0*g_w58 + S3*g_w65 + 6.0*g_w74 - 6.0*g_w85)
        + qww*qz*(-S3X2*g_w46 + 6.0*g_w57 - S3X2*g_w64 + 6.0*g_w75)
        + qwx*qy*(S3X4*g_w46 - 12.0*g_w57 - S3X4*g_w64 + 12.0*g_w75)
        + qwx*qz*(12.0*g_w47 + S3X2*g_w56 - 12.0*g_w58 - S3X2*g_w65 - 12.0*g_w74 + 12.0*g_w85)
        + qwy*qz*(-S3X4*g_w68 + S3X4*g_w86)
        + qxx*qw*(6.0*g_w45 - 6.0*g_w54 + S3X3*g_w67 - S3X3*g_w76 + 6.0*g_w78 - 6.0*g_w87)
        + qxx*qx*(-4.0*g_w44 - 4.0*g_w55 + 4.0*g_w66 + 4.0*g_w77 + 4.0*g_w88)
        + qxx*qy*(-6.0*g_w47 - S3X3*g_w56 - 6.0*g_w58 - S3X3*g_w65 - 6.0*g_w74 - 6.0*g_w85)
        + qxx*qz*(-12.0*g_w48 - 6.0*g_w57 - 6.0*g_w75 - 12.0*g_w84)
        + qxy*qz*(12.0*g_w45 + 12.0*g_w54 - S3X2*g_w67 - S3X2*g_w76 - 12.0*g_w78 - 12.0*g_w87)
        + qyy*qw*(-6.0*g_w45 + 6.0*g_w54 - S3*g_w67 + S3*g_w76 + 6.0*g_w78 - 6.0*g_w87)
        + qyy*qx*(12.0*g_w55 - 8.0*g_w66 - S3X2*g_w68 - S3X2*g_w86)
        + qyy*qy*(-2.0*g_w47 + S3*g_w56 + 2.0*g_w58 + S3*g_w65 - 2.0*g_w74 + 2.0*g_w85)
        + qyy*qz*(S3X2*g_w46 + 6.0*g_w57 + S3X2*g_w64 + 6.0*g_w75)
        + qzz*qw*(-6.0*g_w45 + 6.0*g_w54 + S3*g_w67 - S3*g_w76 - 6.0*g_w78 + 6.0*g_w87)
        + qzz*qx*(12.0*g_w44 + 4.0*g_w66 - 12.0*g_w88)
        + qzz*qy*(6.0*g_w47 - S3*g_w56 + 6.0*g_w58 - S3*g_w65 + 6.0*g_w74 + 6.0*g_w85)
        + qzz*qz*(4.0*g_w48 - 2.0*g_w57 - 2.0*g_w75 + 4.0*g_w84)
    )

    g_qy_l2 = (
        qww*qw*(4.0*g_w48 + 2.0*g_w57 - 2.0*g_w75 - 4.0*g_w84)
        + qww*qx*(6.0*g_w47 + S3*g_w56 - 6.0*g_w58 + S3*g_w65 + 6.0*g_w74 - 6.0*g_w85)
        + qww*qy*(-12.0*g_w44 + 4.0*g_w66 - 12.0*g_w88)
        + qww*qz*(6.0*g_w45 + 6.0*g_w54 + S3*g_w67 + S3*g_w76 + 6.0*g_w78 + 6.0*g_w87)
        + qwx*qy*(-12.0*g_w45 + 12.0*g_w54 - S3X2*g_w67 + S3X2*g_w76 + 12.0*g_w78 - 12.0*g_w87)
        + qwx*qz*(-S3X4*g_w68 + S3X4*g_w86)
        + qwy*qz*(12.0*g_w47 - S3X2*g_w56 + 12.0*g_w58 + S3X2*g_w65 - 12.0*g_w74 - 12.0*g_w85)
        + qxx*qw*(S3X2*g_w46 - 6.0*g_w57 - S3X2*g_w64 + 6.0*g_w75)
        + qxx*qx*(-2.0*g_w47 - S3*g_w56 - 2.0*g_w58 - S3*g_w65 - 2.0*g_w74 - 2.0*g_w85)
        + qxx*qy*(12.0*g_w55 - 8.0*g_w66 - S3X2*g_w68 - S3X2*g_w86)
        + qxx*qz*(6.0*g_w45 + 6.0*g_w54 - S3*g_w67 - S3*g_w76 - 6.0*g_w78 - 6.0*g_w87)
        + qxy*qz*(S3X4*g_w46 + 12.0*g_w57 + S3X4*g_w64 + 12.0*g_w75)
        + qyy*qw*(-12.0*g_w48 + 6.0*g_w57 - 6.0*g_w75 + 12.0*g_w84)
        + qyy*qx*(-6.0*g_w47 + S3X3*g_w56 + 6.0*g_w58 + S3X3*g_w65 - 6.0*g_w74 + 6.0*g_w85)
        + qyy*qy*(4.0*g_w44 - 4.0*g_w55 + 4.0*g_w66 - 4.0*g_w77 + 4.0*g_w88)
        + qyy*qz*(-6.0*g_w45 - 6.0*g_w54 + S3X3*g_w67 + S3X3*g_w76 - 6.0*g_w78 - 6.0*g_w87)
        + qzz*qw*(-S3X2*g_w46 - 6.0*g_w57 + S3X2*g_w64 + 6.0*g_w75)
        + qzz*qx*(6.0*g_w47 - S3*g_w56 + 6.0*g_w58 - S3*g_w65 + 6.0*g_w74 + 6.0*g_w85)
        + qzz*qy*(-8.0*g_w66 + S3X2*g_w68 + 12.0*g_w77 + S3X2*g_w86)
        + qzz*qz*(-2.0*g_w45 - 2.0*g_w54 - S3*g_w67 - S3*g_w76 + 2.0*g_w78 + 2.0*g_w87)
    )

    g_qz_l2 = (
        qww*qw*(-2.0*g_w47 - S3*g_w56 - 2.0*g_w58 + S3*g_w65 + 2.0*g_w74 + 2.0*g_w85)
        + qww*qx*(-S3X2*g_w46 + 6.0*g_w57 - S3X2*g_w64 + 6.0*g_w75)
        + qww*qy*(6.0*g_w45 + 6.0*g_w54 + S3*g_w67 + S3*g_w76 + 6.0*g_w78 + 6.0*g_w87)
        + qww*qz*(-12.0*g_w55 - 8.0*g_w66 - S3X2*g_w68 - S3X2*g_w86)
        + qwx*qy*(-S3X4*g_w68 + S3X4*g_w86)
        + qwx*qz*(-12.0*g_w45 + 12.0*g_w54 + S3X2*g_w67 - S3X2*g_w76 - 12.0*g_w78 + 12.0*g_w87)
        + qwy*qz*(-S3X4*g_w46 - 12.0*g_w57 + S3X4*g_w64 + 12.0*g_w75)
        + qxx*qw*(6.0*g_w47 + S3*g_w56 - 6.0*g_w58 - S3*g_w65 - 6.0*g_w74 + 6.0*g_w85)
        + qxx*qx*(-4.0*g_w48 - 2.0*g_w57 - 2.0*g_w75 - 4.0*g_w84)
        + qxx*qy*(6.0*g_w45 + 6.0*g_w54 - S3*g_w67 - S3*g_w76 - 6.0*g_w78 - 6.0*g_w87)
        + qxx*qz*(12.0*g_w44 + 4.0*g_w66 - 12.0*g_w88)
        + qxy*qz*(12.0*g_w47 - S3X2*g_w56 + 12.0*g_w58 - S3X2*g_w65 + 12.0*g_w74 + 12.0*g_w85)
        + qyy*qw*(6.0*g_w47 - S3*g_w56 + 6.0*g_w58 + S3*g_w65 - 6.0*g_w74 - 6.0*g_w85)
        + qyy*qx*(S3X2*g_w46 + 6.0*g_w57 + S3X2*g_w64 + 6.0*g_w75)
        + qyy*qy*(-2.0*g_w45 - 2.0*g_w54 + S3*g_w67 + S3*g_w76 - 2.0*g_w78 - 2.0*g_w87)
        + qyy*qz*(-8.0*g_w66 + S3X2*g_w68 + 12.0*g_w77 + S3X2*g_w86)
        + qzz*qw*(-6.0*g_w47 + S3X3*g_w56 + 6.0*g_w58 - S3X3*g_w65 + 6.0*g_w74 - 6.0*g_w85)
        + qzz*qx*(12.0*g_w48 - 6.0*g_w57 - 6.0*g_w75 + 12.0*g_w84)
        + qzz*qy*(-6.0*g_w45 - 6.0*g_w54 - S3X3*g_w67 - S3X3*g_w76 + 6.0*g_w78 + 6.0*g_w87)
        + qzz*qz*(-4.0*g_w44 + 4.0*g_w55 + 4.0*g_w66 - 4.0*g_w77 + 4.0*g_w88)
    )

    g_qw_e = g_qw_l1 + g_qw_l2
    g_qx_e = g_qx_l1 + g_qx_l2
    g_qy_e = g_qy_l1 + g_qy_l2
    g_qz_e = g_qz_l1 + g_qz_l2

    proj = g_qw_e*qw + g_qx_e*qx + g_qy_e*qy + g_qz_e*qz
    g_qw_u = (g_qw_e - qw * proj) / ne
    g_qx_u = (g_qx_e - qx * proj) / ne
    g_qy_u = (g_qy_e - qy * proj) / ne
    g_qz_u = (g_qz_e - qz * proj) / ne

    g_s = (g_qw_u * (q1w_f - q2w) + g_qx_u * (q1x_f - q2x) + g_qy_u * (q1y_f - q2y) + g_qz_u * (q1z_f - 0.0))

    g_q1w_norm = tl.where(flip, -(g_qw_u * s_val), g_qw_u * s_val)
    g_q1x_norm = tl.where(flip, -(g_qx_u * s_val), g_qx_u * s_val)
    g_q1z_norm = tl.where(flip, -(g_qz_u * s_val), g_qz_u * s_val)
    g_q2w = g_qw_u * (1.0 - s_val)
    g_q2x = g_qx_u * (1.0 - s_val)
    g_q2y = g_qy_u * (1.0 - s_val)

    ds_darg = s_val * (1.0 - s_val)
    dt_den = tl.where(t_env * (1.0 - t_env) <= eps, 0.0, 1.0 - 2.0 * t_env)
    
    d_arg_dt = (2.0 * den - num * dt_den) / (den * den)
    dt_dey = 1.0 / 1.8
    
    valid = (t_env > 1e-7) & (t_env < 1.0 - 1e-7)
    g_ey_from_s = tl.where(valid, g_s * ds_darg * d_arg_dt * dt_dey, 0.0)

    proj1 = g_q1w_norm * q1w_f + g_q1x_norm * q1x_f + g_q1z_norm * q1z_f 
    g_ex_c1 = (g_q1z_norm - q1z_f * proj1) / n1
    g_ey_c1 = (g_q1w_norm - q1w_f * proj1) / n1
    g_ez_c1 = -(g_q1x_norm - q1x_f * proj1) / n1

    proj2 = g_q2w * q2w + g_q2x * q2x + g_q2y * q2y
    g_ex_c2 = (g_q2y - q2y * proj2) / n2
    g_ey_c2 = -(g_q2x - q2x * proj2) / n2
    g_ez_c2 = -(g_q2w - q2w * proj2) / n2

    g_ex = g_ex_c1 + g_ex_c2
    g_ey = g_ey_c1 + g_ey_c2 + g_ey_from_s
    g_ez = g_ez_c1 + g_ez_c2

    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < B
    offset_g_vals = tl.load(offset_g_ptr + b_offs, mask=b_mask, other=0.0)
    
    full_mask_b = e_mask[:, None] & b_mask[None, :]
    diff = dist[:, None] - offset_g_vals[None, :]
    gauss = tl.exp(coeff_g * diff * diff)
    gauss = tl.where(full_mask_b, gauss, 0.0)
    
    g_xb = tl.load(g_x_edge_ptr + e_offs[:, None] * stride_xe_e + b_offs[None, :] * stride_xe_c, mask=full_mask_b, other=0.0)
    dg_ddist = 2.0 * coeff_g * diff * gauss
    g_dist_gauss = tl.sum(g_xb * dg_ddist, axis=1)

    g_env = tl.load(g_env_ptr + e_offs, mask=e_mask, other=0.0)
    d4 = d_sc * d_sc * d_sc * d_sc
    denv_dd = d4 * (-105.0 + d_sc * (210.0 - 105.0 * d_sc))
    g_dist_env = tl.where(d_sc < 1.0, g_env * denv_dd / cutoff, 0.0)

    g_dist_total = g_dist_env + g_dist_gauss

    proj_e = g_ex * ex + g_ey * ey + g_ez * ez
    g_ex_final = (g_ex - ex * proj_e) / dist_safe + g_dist_total * ex
    g_ey_final = (g_ey - ey * proj_e) / dist_safe + g_dist_total * ey
    g_ez_final = (g_ez - ez * proj_e) / dist_safe + g_dist_total * ez

    g_ex_final = tl.where(e_mask, g_ex_final, 0.0)
    g_ey_final = tl.where(e_mask, g_ey_final, 0.0)
    g_ez_final = tl.where(e_mask, g_ez_final, 0.0)

    # 1. SCATTER TO ATOM POSITIONS
    tl.atomic_add(g_pos_ptr + idx_i * stride_pos_n + 0, g_ex_final, mask=e_mask)
    tl.atomic_add(g_pos_ptr + idx_i * stride_pos_n + 1, g_ey_final, mask=e_mask)
    tl.atomic_add(g_pos_ptr + idx_i * stride_pos_n + 2, g_ez_final, mask=e_mask)
    tl.atomic_add(g_pos_ptr + idx_j * stride_pos_n + 0, -g_ex_final, mask=e_mask)
    tl.atomic_add(g_pos_ptr + idx_j * stride_pos_n + 1, -g_ey_final, mask=e_mask)
    tl.atomic_add(g_pos_ptr + idx_j * stride_pos_n + 2, -g_ez_final, mask=e_mask)

    # 2. STORE TO SHIFTS (Natively handles the PyTorch Cell Virial math!)
    tl.store(g_shifts_ptr + e_offs * stride_shift_e + 0 * stride_shift_c, g_ex_final, mask=e_mask)
    tl.store(g_shifts_ptr + e_offs * stride_shift_e + 1 * stride_shift_c, g_ey_final, mask=e_mask)
    tl.store(g_shifts_ptr + e_offs * stride_shift_e + 2 * stride_shift_c, g_ez_final, mask=e_mask)


class PrecomputeGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, Z, idx_i, idx_j, shifts, W_src, W_tgt, offset_g, coeff_g, cutoff):
        E = idx_i.shape[0]
        B = offset_g.shape[0]
        C = W_src.shape[1]
        device = pos.device
        dtype = pos.dtype

        x_edge_raw = torch.empty(E, B + 2 * C, device=device, dtype=dtype)
        wigner = torch.empty(E, 34, device=device, dtype=dtype)
        env = torch.empty(E, device=device, dtype=dtype)

        BLOCK_B = _np2(B)
        BLOCK_C = _np2(C)
        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']),)
        
        _geom_fwd_kernel[grid](
            pos, Z, idx_i, idx_j, shifts, W_src, W_tgt, offset_g,
            x_edge_raw, wigner, env,
            E, B, C,
            float(cutoff), float(coeff_g),
            pos.stride(0), shifts.stride(0), shifts.stride(1),
            W_src.stride(0), W_src.stride(1),
            W_tgt.stride(0), W_tgt.stride(1),
            x_edge_raw.stride(0), x_edge_raw.stride(1),
            wigner.stride(0), wigner.stride(1),
            BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C
        )

        ctx.save_for_backward(pos, idx_i, idx_j, shifts, offset_g)
        ctx.cutoff = float(cutoff)
        ctx.coeff_g = float(coeff_g)
        ctx.B = B
        ctx.C = C
        return x_edge_raw, wigner, env

    @staticmethod
    def backward(ctx, g_x_edge_raw, g_wigner, g_env):
        pos, idx_i, idx_j, shifts, offset_g = ctx.saved_tensors
        E = idx_i.shape[0]
        B = ctx.B
        C = ctx.C

        g_x_edge_raw = g_x_edge_raw.contiguous()
        g_wigner = g_wigner.contiguous()
        g_env = g_env.contiguous()

        BLOCK_B = _np2(B)
        BLOCK_C = _np2(C)
        grid = lambda META: (triton.cdiv(E, META['BLOCK_E']),)

        # Autotune warmup trick: this kernel accumulates g_pos with atomic_add.
        # Triton autotuning launches every candidate config to time it, so on
        # the first call each trial would atomic_add into the real gradient
        # buffer, multiply-counting g_pos by the number of configs tried. Prime
        # the autotune cache once against throwaway buffers; the real launch
        # below then hits the cache and runs exactly once.
        if not _geom_bwd_kernel.cache:
            _g_pos_warmup = torch.zeros_like(pos)
            _g_shifts_warmup = torch.zeros_like(shifts)
            _geom_bwd_kernel[grid](
                pos, idx_i, idx_j, shifts, offset_g,
                g_x_edge_raw, g_wigner, g_env,
                _g_pos_warmup, _g_shifts_warmup,
                E, B, C,
                ctx.cutoff, ctx.coeff_g,
                pos.stride(0), shifts.stride(0), shifts.stride(1),
                g_x_edge_raw.stride(0), g_x_edge_raw.stride(1),
                g_wigner.stride(0), g_wigner.stride(1),
                BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C
            )
            torch.cuda.synchronize()
            del _g_pos_warmup
            del _g_shifts_warmup

        # Actual execution
        g_pos = torch.zeros_like(pos)
        g_shifts = torch.empty_like(shifts)
        
        _geom_bwd_kernel[grid](
            pos, idx_i, idx_j, shifts, offset_g,
            g_x_edge_raw, g_wigner, g_env,
            g_pos, g_shifts,
            E, B, C,
            ctx.cutoff, ctx.coeff_g,
            pos.stride(0), shifts.stride(0), shifts.stride(1),
            g_x_edge_raw.stride(0), g_x_edge_raw.stride(1),
            g_wigner.stride(0), g_wigner.stride(1),
            BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C
        )

        # 10 Outputs: g_pos, g_Z, g_idx_i, g_idx_j, g_shifts, g_Wsrc, g_Wtgt, g_offset, g_coeff, g_cutoff
        return g_pos, None, None, None, g_shifts, None, None, None, None, None

def precompute_geometry(pos, Z, idx_i, idx_j, shifts, W_src, W_tgt, offset_g, coeff_g, cutoff):
    return PrecomputeGeometry.apply(pos, Z, idx_i, idx_j, shifts, W_src, W_tgt, offset_g, coeff_g, cutoff)