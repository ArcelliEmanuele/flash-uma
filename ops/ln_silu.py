"""

Fused LayerNorm + SiLU Kernel.
Replaces the pattern:
    h = F.layer_norm(x, (D,), gamma, beta, eps)
    h = F.silu(h)

Inference Optimization:
Because all weights (gamma, beta) have `requires_grad = False`, the backward
pass has been stripped of the complex partial-reduction atomics used to 
compute `g_gamma` and `g_beta`. It strictly routes the gradient back to `x`.
"""

import torch
import triton
import triton.language as tl

def _ln_configs():
    # Single-row-per-program kernel. We sweep over warps and stages.
    configs = []
    for w in [2, 4, 8]:
        for s in [1, 2, 3, 4]:
            configs.append(triton.Config({}, num_warps=w, num_stages=s))
    return configs

# =========================================================================
# FORWARD KERNEL: LayerNorm + SiLU
# =========================================================================
@triton.autotune(configs=_ln_configs(), key=["D"])
@triton.jit
def _ln_silu_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr,
    out_ptr, mean_ptr, rstd_ptr,
    M, D, eps,
    stride_xm, stride_xd,
    stride_om, stride_od,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Load row
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)

    # Compute Mean and Variance
    sum_x = tl.sum(tl.where(d_mask, x, 0.0), axis=0)
    mean = sum_x / D
    xc = tl.where(d_mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # Save Mean and Rstd for the backward pass
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Apply Normalization
    xhat = xc * rstd
    gamma = tl.load(gamma_ptr + d_offs, mask=d_mask, other=0.0)
    beta = tl.load(beta_ptr + d_offs, mask=d_mask, other=0.0)
    
    # Linear Affine Transform
    y = xhat * gamma + beta
    
    # SiLU Activation (y * sigmoid(y))
    sig = tl.sigmoid(y)
    out = y * sig
    
    # Store Output
    tl.store(out_ptr + row * stride_om + d_offs * stride_od, out, mask=d_mask)

# =========================================================================
# BACKWARD KERNEL: Optimized for Inference (No d_gamma / d_beta)
# =========================================================================
@triton.autotune(configs=_ln_configs(), key=["D"])
@triton.jit
def _ln_silu_bwd_dx_kernel(
    x_ptr, gamma_ptr, beta_ptr,
    mean_ptr, rstd_ptr, g_out_ptr,
    g_x_ptr,
    M, D,
    stride_xm, stride_xd,
    stride_gom, stride_god,
    stride_gxm, stride_gxd,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # 1. Recompute forward values
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=d_mask, other=0.0)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    gamma = tl.load(gamma_ptr + d_offs, mask=d_mask, other=0.0)
    beta = tl.load(beta_ptr + d_offs, mask=d_mask, other=0.0)

    xhat = (x - mean) * rstd
    y = xhat * gamma + beta
    sig = tl.sigmoid(y)
    
    # 2. SiLU Derivative: d(silu(y))/dy = sig + y * sig * (1 - sig)
    dsilu_dy = sig + y * sig * (1.0 - sig)

    # 3. Chain Rule: g_out -> g_y -> g_xhat
    g_out = tl.load(g_out_ptr + row * stride_gom + d_offs * stride_god, mask=d_mask, other=0.0)
    g_y = g_out * dsilu_dy           
    g_xhat = g_y * gamma             

    # 4. Standard LayerNorm backward formula for g_x
    # g_x = rstd * (g_xhat - (mean(g_xhat) + xhat * mean(g_xhat * xhat)))
    sum_gxhat = tl.sum(tl.where(d_mask, g_xhat, 0.0), axis=0)
    sum_gxhat_xhat = tl.sum(tl.where(d_mask, g_xhat * xhat, 0.0), axis=0)

    g_x = rstd * (g_xhat - (sum_gxhat + xhat * sum_gxhat_xhat) / D)
    
    # 5. Store gradient w.r.t inputs
    tl.store(g_x_ptr + row * stride_gxm + d_offs * stride_gxd, g_x, mask=d_mask)


class _LayerNormSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        M, D = x.shape
        x_c = x.contiguous()
        gamma_c = gamma.contiguous()
        beta_c = beta.contiguous()

        out = torch.empty_like(x_c)
        mean = torch.empty(M, device=x.device, dtype=torch.float32)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        # BLOCK_D must cover D in one shot. 
        # For this network, D is at most 12*C (e.g. 1536).
        BLOCK_D = triton.next_power_of_2(D)
        
        grid = (M,)
        _ln_silu_fwd_kernel[grid](
            x_c, gamma_c, beta_c,
            out, mean, rstd,
            M, D, float(eps),
            x_c.stride(0), x_c.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_c, gamma_c, beta_c, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, g_out):
        x, gamma, beta, mean, rstd = ctx.saved_tensors
        M, D = x.shape
        BLOCK_D = ctx.BLOCK_D

        g_out_c = g_out.contiguous()
        g_x = torch.empty_like(x)

        grid = (M,)
        _ln_silu_bwd_dx_kernel[grid](
            x, gamma, beta, mean, rstd, g_out_c,
            g_x,
            M, D,
            x.stride(0), x.stride(1),
            g_out_c.stride(0), g_out_c.stride(1),
            g_x.stride(0), g_x.stride(1),
            BLOCK_D=BLOCK_D,
        )

        # We return None for gamma, beta, and eps because they do not require gradients
        return g_x, None, None, None


def layer_norm_silu(x, gamma, beta, eps=1e-7):
    """
    One-pass fused LayerNorm + SiLU optimized for Inference.
    Expects `x` of shape (E, D).
    """
    return _LayerNormSiLUFn.apply(x, gamma, beta, eps)