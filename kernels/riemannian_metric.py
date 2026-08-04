"""
Triton forward+backward kernels for RiemannianMetric.

MATH:
    forward:
        raw    = x @ weight_h                          # (n_params,)
        packed = tanh(raw)                              # (n_params,)
        L      = lower_triangular(packed)                # (head_dim, head_dim)
                with L[i,i] = softplus(packed_at_diag) + epsilon
        u      = x @ L                                   # (head_dim,)
        out    = u @ L^T                                  # (head_dim,)

    backward:
        grad_u      = grad_out @ L
        grad_L      = outer(grad_out, u) + outer(x, grad_u)
        grad_packed[k] = grad_L[row_k, col_k] * (sigmoid(packed[k]) if diagonal else 1)
        grad_raw    = grad_packed * (1 - packed**2)          # tanh'
        grad_x      = grad_u @ L^T + grad_raw @ weight_h^T
        grad_weight_h += outer(x, grad_raw)                   # summed over all tokens sharing this head

DESIGN:
    `raw = x @ weight_h` is a genuine dense matmul shared across every
    token of a given head (weight_h doesn't vary per token) and, because
    n_params ~ head_dim^2/2, it's also the overwhelming majority of the FLOPs
    here (~94% at head_dim=64) -- so it's computed as a real batched `tl.dot`
    (tensor cores) across many tokens at once, in `_riemannian_project_kernel`
    and its backward-side mirror in `_riemannian_weight_and_gradxw_kernel`.
    
    The rest of the computation (building L from packed, applying L to x, and their
    gradients) is genuinely per-token -- each token gets its own L -- so it
    can't be expressed as a shared-matrix matmul and stays as an elementwise,
    one-program-per-token computation (`_riemannian_apply_fwd_kernel`,
    `_riemannian_grad_L_and_raw_kernel`). That step is only ~6% of the FLOPs, so
    leaving it unbatched costs little as long as the dominant matmul step above
    is fast.

DATA FLOW:
    four kernels, connected via global-memory scratch buffers rather
    than one large fused kernel -- deliberately, so each piece can be verified
    independently against the reference and diagnosed in isolation if something
    is wrong on a given GPU/Triton build):

    forward:
        1. _riemannian_project_kernel:      (x, weight) -> packed_scratch
        2. _riemannian_apply_fwd_kernel:    (x, packed_scratch) -> out

    backward:
        recomputes packed rather than saving it from forward -- only x
        is saved via ctx.save_for_backward; weight is a parameter, always
        available
        
        1. _riemannian_project_kernel:        (x, weight) -> packed_scratch   [recompute]
        3. _riemannian_grad_L_and_raw_kernel: (x, grad_out, packed_scratch) -> grad_x_L, grad_raw_scratch
        4. _riemannian_weight_and_gradxw_kernel: (x, weight, grad_raw_scratch) -> grad_weight, grad_x_w
        
        grad_x = grad_x_L + grad_x_w   (plain elementwise add, done in PyTorch)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    # tl.exp/tl.log require fp32/fp64 in this Triton build (they don't accept
    # fp16 -- a real hardware/libdevice constraint, not a numerical choice),
    # so upcast for the computation and cast back to x's own dtype after.
    x32 = x.to(tl.float32)
    result = tl.log(1.0 + tl.exp(x32))
    return result.to(x.dtype)


@triton.jit
def _tanh(x):
    x32 = x.to(tl.float32)
    abs_x = tl.abs(x32)
    e = tl.exp(-2.0 * abs_x)
    t = (1.0 - e) / (1.0 + e)
    result = tl.where(x32 >= 0, t, -t)
    return result.to(x.dtype)


@triton.jit
def _sigmoid(x):
    x32 = x.to(tl.float32)
    pos = 1.0 / (1.0 + tl.exp(-x32))
    e = tl.exp(x32)
    neg = e / (1.0 + e)
    result = tl.where(x32 >= 0, pos, neg)
    return result.to(x.dtype)


@triton.jit
def _riemannian_project_kernel(
    x_ptr, weight_ptr, packed_scratch_ptr,
    seq_len, heads, head_dim, n_params,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_P_TILE: tl.constexpr,
):
    nh = tl.program_id(0)          # flat (batch, head) index: n * heads + h
    s_tile = tl.program_id(1)
    head_id = nh % heads

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D)

    w_base = weight_ptr + head_id * head_dim * n_params
    token_off = nh * seq_len + s_off  # token_id for each row in this block

    n_tiles = tl.cdiv(n_params, BLOCK_P_TILE)
    for tile in range(n_tiles):
        p_off = tile * BLOCK_P_TILE + tl.arange(0, BLOCK_P_TILE)
        p_mask = p_off < n_params
        w_tile = tl.load(
            w_base + d_off[:, None] * n_params + p_off[None, :],
            mask=d_mask[:, None] & p_mask[None, :], other=0.0,
        )  # (BLOCK_D, BLOCK_P_TILE)

        raw_tile = tl.dot(x_block, w_tile, allow_tf32=False)  # (BLOCK_S, BLOCK_P_TILE), fp32 accumulate
        packed_tile = _tanh(raw_tile).to(x_block.dtype)
        packed_tile = tl.where(p_mask[None, :], packed_tile, 0.0)
        tl.store(
            packed_scratch_ptr + token_off[:, None] * n_params + p_off[None, :],
            packed_tile,
            mask=s_mask[:, None] & p_mask[None, :],
        )


@triton.jit
def _riemannian_apply_fwd_kernel(
    x_ptr, out_ptr, packed_scratch_ptr,
    idx_map_ptr, lower_mask_ptr, diag_mask_ptr,
    head_dim, n_params, epsilon,
    BLOCK_D: tl.constexpr,
):
    # Per-token: build L from the already-computed packed_scratch (written
    # by _riemannian_project_kernel)
    token_id = tl.program_id(0)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    x = tl.load(x_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0)

    row = tl.arange(0, BLOCK_D)[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    dd_mask = (row < head_dim) & (col < head_dim)

    idx_map = tl.load(idx_map_ptr + row * head_dim + col, mask=dd_mask, other=0)
    lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0
    is_diag = tl.load(diag_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0

    gathered = tl.load(packed_scratch_ptr + token_id * n_params + idx_map, mask=lower, other=0.0)
    diag_val = (_softplus(gathered) + epsilon).to(gathered.dtype)
    L = tl.where(is_diag, diag_val, gathered)
    L = tl.where(lower, L, 0.0)

    u = tl.sum(x[:, None] * L, axis=0)          # u[j] = sum_i x[i]*L[i,j]
    out = tl.sum(u[None, :] * L, axis=1)        # out[j] = sum_i u[i]*L[j,i]

    tl.store(out_ptr + token_id * head_dim + d_off, out, mask=d_mask)


@triton.jit
def _riemannian_grad_L_and_raw_kernel(
    x_ptr, grad_out_ptr,
    grad_x_L_ptr, grad_raw_scratch_ptr,
    packed_scratch_ptr, grad_L_scratch_ptr,
    idx_map_ptr, lower_mask_ptr, diag_mask_ptr, tril_rows_ptr, tril_cols_ptr,
    head_dim, n_params, epsilon,
    BLOCK_D: tl.constexpr, BLOCK_P_TILE: tl.constexpr,
):
    # Per-token (same reasoning as _riemannian_apply_fwd_kernel): rebuild L,
    # compute grad_u/grad_L, then grad_packed/grad_raw tiled along n_params.
    # Writes grad_x's L-term directly (the weight-term, grad_raw @ weight^T,
    # is computed separately in the batched _riemannian_weight_and_gradxw_kernel
    # and added in plain PyTorch afterward) and grad_raw to scratch, for that
    # next kernel to consume as a batched matmul operand.
    token_id = tl.program_id(0)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim

    x = tl.load(x_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0)

    row = tl.arange(0, BLOCK_D)[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    dd_mask = (row < head_dim) & (col < head_dim)
    idx_map = tl.load(idx_map_ptr + row * head_dim + col, mask=dd_mask, other=0)
    lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0
    is_diag = tl.load(diag_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0
    gathered = tl.load(packed_scratch_ptr + token_id * n_params + idx_map, mask=lower, other=0.0)
    diag_val = (_softplus(gathered) + epsilon).to(gathered.dtype)
    L = tl.where(is_diag, diag_val, gathered)
    L = tl.where(lower, L, 0.0)
    u = tl.sum(x[:, None] * L, axis=0)

    # grad_u[i] = sum_j grad_out[j] * L[j,i]
    grad_u = tl.sum(grad_out[:, None] * L, axis=0)
    # grad_L[a,b] = grad_out[a]*u[b] + x[a]*grad_u[b]
    grad_L = grad_out[:, None] * u[None, :] + x[:, None] * grad_u[None, :]
    grad_L = tl.where(dd_mask, grad_L, 0.0)
    tl.store(grad_L_scratch_ptr + token_id * head_dim * head_dim + row * head_dim + col, grad_L, mask=dd_mask)

    # grad_x's L-term: sum_j grad_u[j]*L[i,j]. The weight-term is added later.
    grad_x_L = tl.sum(grad_u[None, :] * L, axis=1)
    tl.store(grad_x_L_ptr + token_id * head_dim + d_off, grad_x_L, mask=d_mask)

    n_tiles = tl.cdiv(n_params, BLOCK_P_TILE)
    for tile in range(n_tiles):
        p_off = tile * BLOCK_P_TILE + tl.arange(0, BLOCK_P_TILE)
        p_mask = p_off < n_params
        packed_tile = tl.load(packed_scratch_ptr + token_id * n_params + p_off, mask=p_mask, other=0.0)

        # grad_packed[k] = grad_L[tril_rows[k], tril_cols[k]] * (sigmoid(packed[k]) if diagonal else 1)
        tr = tl.load(tril_rows_ptr + p_off, mask=p_mask, other=0)
        tc = tl.load(tril_cols_ptr + p_off, mask=p_mask, other=0)
        grad_L_at_k = tl.load(grad_L_scratch_ptr + token_id * head_dim * head_dim + tr * head_dim + tc, mask=p_mask, other=0.0)
        is_diag_k = tr == tc
        grad_packed = tl.where(is_diag_k, grad_L_at_k * _sigmoid(packed_tile), grad_L_at_k)
        grad_packed = tl.where(p_mask, grad_packed, 0.0)

        grad_raw = grad_packed * (1.0 - packed_tile * packed_tile)  # tanh'
        tl.store(grad_raw_scratch_ptr + token_id * n_params + p_off, grad_raw, mask=p_mask)


@triton.jit
def _riemannian_weight_and_gradxw_kernel(
    x_ptr, weight_ptr, grad_raw_scratch_ptr,
    grad_x_w_ptr, grad_weight_ptr,
    seq_len, heads, head_dim, n_params,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_P_TILE: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # Batched matmul mirror of _riemannian_project_kernel, on the backward
    # side: grad_weight_h[d,p] = sum_s x[s,d]*grad_raw[s,p]  (x_block^T @ grad_raw_block)
    #       grad_x_w[s,d]      = sum_p grad_raw[s,p]*weight[d,p]  (grad_raw_block @ weight_h^T)
    # Both are genuine GEMMs over the token dimension within one (head,
    # seq-tile) block, computed via tl.dot instead of the token-at-a-time
    # atomic_add this file used before the batched rewrite.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    group_id = s_tile % NUM_GROUPS

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D)

    token_off = nh * seq_len + s_off
    w_base = weight_ptr + head_id * head_dim * n_params
    gw_base = grad_weight_ptr + (group_id * heads + head_id) * head_dim * n_params

    grad_x_w_acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    n_tiles = tl.cdiv(n_params, BLOCK_P_TILE)
    for tile in range(n_tiles):
        p_off = tile * BLOCK_P_TILE + tl.arange(0, BLOCK_P_TILE)
        p_mask = p_off < n_params

        grad_raw_tile = tl.load(
            grad_raw_scratch_ptr + token_off[:, None] * n_params + p_off[None, :],
            mask=s_mask[:, None] & p_mask[None, :], other=0.0,
        )  # (BLOCK_S, BLOCK_P_TILE)
        w_tile = tl.load(
            w_base + d_off[:, None] * n_params + p_off[None, :],
            mask=d_mask[:, None] & p_mask[None, :], other=0.0,
        )  # (BLOCK_D, BLOCK_P_TILE)

        grad_x_w_acc += tl.dot(grad_raw_tile, tl.trans(w_tile), allow_tf32=False)
        
        gw_tile = tl.dot(tl.trans(x_block), grad_raw_tile, allow_tf32=False)  # (BLOCK_D, BLOCK_P_TILE), fp32
        tl.atomic_add(
            gw_base + d_off[:, None] * n_params + p_off[None, :], gw_tile,
            mask=d_mask[:, None] & p_mask[None, :],
        )

    grad_x_w = grad_x_w_acc.to(x_block.dtype)
    tl.store(
        grad_x_w_ptr + token_off[:, None] * head_dim + d_off[None, :], grad_x_w,
        mask=s_mask[:, None] & d_mask[None, :],
    )


def _structural_buffers(head_dim, device):
    tril_rows, tril_cols = torch.tril_indices(head_dim, head_dim, device=device)
    idx_map = torch.zeros(head_dim, head_dim, dtype=torch.int64, device=device)
    idx_map[tril_rows, tril_cols] = torch.arange(tril_rows.shape[0], device=device)
    lower_mask = torch.zeros(head_dim, head_dim, dtype=torch.int32, device=device)
    lower_mask[tril_rows, tril_cols] = 1
    diag_mask = torch.zeros(head_dim, head_dim, dtype=torch.int32, device=device)
    diag_idx = torch.arange(head_dim, device=device)
    diag_mask[diag_idx, diag_idx] = 1
    return tril_rows.to(torch.int32), tril_cols.to(torch.int32), idx_map.to(torch.int32), lower_mask, diag_mask


class RiemannianMetricKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon):
        N, heads, S, head_dim = x.shape
        n_params = head_dim * (head_dim + 1) // 2
        n_tokens = N * heads * S
        n_nh = N * heads

        x_c = x.contiguous()
        weight_c = weight.contiguous()
        tril_rows, tril_cols, idx_map, lower_mask, diag_mask = _structural_buffers(head_dim, x.device)

        # Floor at 16: tl.dot (used by the two batched-matmul kernels) needs
        # every operand dimension >= 16 to be a valid tensor-core shape.
        #
        # Of _riemannian_weight_and_gradxw_kernel's five live tiles, only
        # x_block/w_tile/grad_raw_tile scale with the input dtype's element
        # size -- gw_tile and the grad_x_w accumulator are always fp32
        # (tl.dot's own accumulator type), regardless of input dtype. That's
        # why only BLOCK_P_TILE (which those two fp32-fixed tiles also scale
        # with) is scaled down for fp32; BLOCK_S only touches the
        # dtype-scaled tiles, so it doesn't need to shrink for fp32 the way
        # BLOCK_P_TILE does. Measured on an RTX 4090: BLOCK_S=32,
        # BLOCK_P_TILE=128 needed 114944 bytes at fp32 (4-byte elements)
        # against a 101376-byte limit; BLOCK_P_TILE=64 at fp32 leaves enough
        # margin to keep BLOCK_S at the same 32 fp16 uses. This file has hit
        # shared-memory OutOfResources three times already with more
        # aggressive sizing -- these caps are deliberately conservative, not
        # pushed to the exact limit.
        scale = max(1, x.element_size() // 2)
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        BLOCK_S = max(16, min(32, triton.next_power_of_2(S)))
        BLOCK_P_TILE = max(16, min(128 // scale, triton.next_power_of_2(n_params)))

        # num_warps=8 (vs. Triton's default 4): pure occupancy/parallelism
        # tuning for these two matmul-bound kernels, no extra shared memory
        # cost -- unlike growing the tile sizes above, this can't reintroduce
        # an OutOfResources error.
        packed_scratch = torch.empty(n_tokens, n_params, dtype=x.dtype, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid_proj](
            x_c.view(n_tokens, head_dim), weight_c, packed_scratch,
            S, heads, head_dim, n_params,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_P_TILE=BLOCK_P_TILE,
            num_warps=8,
        )

        out = torch.empty_like(x_c)
        _riemannian_apply_fwd_kernel[(n_tokens,)](
            x_c.view(n_tokens, head_dim), out.view(n_tokens, head_dim), packed_scratch,
            idx_map, lower_mask, diag_mask,
            head_dim, n_params, epsilon,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_c, weight_c, tril_rows, tril_cols, idx_map, lower_mask, diag_mask)
        ctx.epsilon = epsilon
        ctx.shapes = (N, heads, S, head_dim, n_params, n_tokens, n_nh)
        ctx.blocks = (BLOCK_D, BLOCK_S, BLOCK_P_TILE)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, tril_rows, tril_cols, idx_map, lower_mask, diag_mask = ctx.saved_tensors
        N, heads, S, head_dim, n_params, n_tokens, n_nh = ctx.shapes
        BLOCK_D, BLOCK_S, BLOCK_P_TILE = ctx.blocks
        grad_out_c = grad_out.contiguous()

        # Recompute packed (not saved from forward) -- see module docstring.
        packed_scratch = torch.empty(n_tokens, n_params, dtype=x.dtype, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid_proj](
            x.view(n_tokens, head_dim), weight, packed_scratch,
            S, heads, head_dim, n_params,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_P_TILE=BLOCK_P_TILE,
            num_warps=8,
        )

        grad_L_scratch = torch.empty(n_tokens, head_dim, head_dim, dtype=x.dtype, device=x.device)
        grad_raw_scratch = torch.empty(n_tokens, n_params, dtype=x.dtype, device=x.device)
        grad_x_L = torch.empty_like(x)
        _riemannian_grad_L_and_raw_kernel[(n_tokens,)](
            x.view(n_tokens, head_dim), grad_out_c.view(n_tokens, head_dim),
            grad_x_L.view(n_tokens, head_dim), grad_raw_scratch,
            packed_scratch, grad_L_scratch,
            idx_map, lower_mask, diag_mask, tril_rows, tril_cols,
            head_dim, n_params, ctx.epsilon,
            BLOCK_D=BLOCK_D, BLOCK_P_TILE=BLOCK_P_TILE,
        )

        num_blocks = n_nh * triton.cdiv(S, BLOCK_S)
        NUM_GROUPS = min(32, num_blocks)
        grad_weight_partial = torch.zeros(NUM_GROUPS, heads, head_dim, n_params, dtype=torch.float32, device=x.device)
        grad_x_w = torch.empty_like(x)
        grid_gw = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_weight_and_gradxw_kernel[grid_gw](
            x.view(n_tokens, head_dim), weight, grad_raw_scratch,
            grad_x_w.view(n_tokens, head_dim), grad_weight_partial,
            S, heads, head_dim, n_params,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_P_TILE=BLOCK_P_TILE, NUM_GROUPS=NUM_GROUPS,
            num_warps=8,
        )

        grad_x = grad_x_L + grad_x_w
        grad_weight_f32 = grad_weight_partial.sum(dim=0)
        return grad_x, grad_weight_f32.to(weight.dtype), None
