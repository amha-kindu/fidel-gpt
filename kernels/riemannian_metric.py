"""
Triton forward+backward kernels for RiemannianMetric.

MATH:
    forward (per head h, per token):
        raw0      = x_h @ weight_h                         # (n_params,)
        raw       = raw0 / temperature                     # temperature = sqrt(head_dim)
        packed[k] = log(1 + softplus(raw[k])) - C           # if row == col (L_raw's diagonal); C = log(1+softplus(0))
        packed[k] = asinh(raw[k])                          # otherwise (L_raw's off-diagonal)
        L_raw     = lower_triangular(packed)               # (head_dim, head_dim)
        L         = identity + L_raw                        # (head_dim, head_dim)
        u         = x_h @ L                                 # (head_dim,)
        out       = u @ L^T                                  # (head_dim,)

    backward (per head h, per token):
        grad_u          = grad_out @ L
        grad_L          = outer(grad_out, u) + outer(x_h, grad_u)
        grad_L_raw      = grad_L                             # dL/dL_raw = 1
        grad_packed[k]  = grad_L_raw[row_k, col_k]
        grad_raw[k]     = grad_packed[k] * [exp(-(packed[k]+C)) - exp(1 - exp(packed[k]+C) - (packed[k]+C))]   # if diagonal, d(packed)/d(raw)
        grad_raw[k]     = grad_packed[k] / cosh(raw[k])      # otherwise (off-diagonal, d(asinh)/d(raw))
        grad_raw0[k]    = grad_raw[k] / temperature
        grad_x          = grad_u @ L^T + grad_raw0 @ weight_h^T   # x enters via BOTH the L-application path (u = x @ L) AND the projection path (raw0 = x @ weight)
        grad_weight_h  += outer(x_h, grad_raw0)              # summed over all tokens sharing this head

    d(packed)/d(raw) for the diagonal is derived by substituting softplus(raw)=e^packed-1
    (inverting packed=log(1+softplus(raw)), the UNSHIFTED value, i.e. packed[k]+C) into
    softplus's own derivative, sigmoid(raw)/(1+softplus(raw)), and using
    sigmoid(raw)=1-e^(-softplus(raw)) (the same identity plain softplus's backward
    already relied on) -- giving a closed form purely in terms of the already-computed
    (unshifted) packed value, verified against autograd directly. Subtracting a
    constant doesn't change a derivative, so re-adding C before evaluating this
    closed form is exactly equivalent to differentiating the shifted packing directly.

DESIGN:
    `raw0 = x_h @ weight_h` is a genuine dense matmul shared across every token of a
    given head (weight_h doesn't vary per token) and, because n_params ~ head_dim^2/2,
    it's also the overwhelming majority of the FLOPs here (~94% at head_dim=64) -- so it's
    computed as a real batched `tl.dot` (tensor cores) across many tokens at once, in
    `_riemannian_project_kernel` and its backward-side mirror in
    `_riemannian_weight_and_gradxw_kernel`.

    The rest of the computation (building L from packed, applying L to x, and their
    gradients) is genuinely per-token -- each token gets its own L -- so it can't be
    expressed as a shared-matrix matmul and stays as an elementwise, one-program-per-token
    computation (`_riemannian_apply_fwd_kernel`, `_riemannian_grad_L_and_raw_kernel`).
    That step is only ~6% of the FLOPs, so leaving it unbatched costs little as long as
    the dominant matmul step above is fast.

DATA FLOW:
    four kernels, connected via global-memory scratch buffers

    forward:
        1. _riemannian_project_kernel:      (x, weight) -> packed_scratch
        2. _riemannian_apply_fwd_kernel:    (x, packed_scratch) -> out

    backward:
        recomputes packed rather than saving it from forward -- only x
        is saved via ctx.save_for_backward; weight is a parameter, always available

        1. _riemannian_project_kernel:        (x, weight) -> packed_scratch   [recompute]
        3. _riemannian_grad_L_and_raw_kernel: (x, grad_out, packed_scratch) -> grad_x_L, grad_raw_scratch
        4. _riemannian_weight_and_gradxw_kernel: (x, weight, grad_raw_scratch) -> grad_weight, grad_x_w

        grad_x = grad_x_L + grad_x_w   (plain elementwise add, done in PyTorch --
        x contributes through both the L-application path and the projection path, so
        both terms are real)
"""

import math
import torch
import triton
import triton.language as tl

_DIAG_PACKED_BASELINE = tl.constexpr(math.log1p(math.log(2.0)))


@triton.jit
def _softplus(x):
    # tl.exp/tl.log require fp32/fp64 in this Triton build (they don't accept
    # fp16 -- a real hardware/libdevice constraint, not a numerical choice),
    # so upcast for the computation and cast back to x's own dtype after.
    x32 = x.to(tl.float32)
    result = tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x32)))
    return result.to(x.dtype)


@triton.jit
def _log_softplus(x):
    # Diagonal packing nonlinearity: log(1 + softplus(x)). softplus(x) grows
    # linearly (~x) for large x; composing an outer log1p compresses that to
    # logarithmic growth -- PRACTICALLY self-bounding (packed=50 needs x~5e21)
    # without an explicit clamp, and smooth everywhere (no dead zone the way a
    # hard clamp has). Still strictly positive: softplus(x)>0 always, so
    # 1+softplus(x)>1, so log(...)>0 always -- same guarantee plain softplus
    # gave. See model.py's RiemannianMetric.forward and this module's docstring
    # for the full reasoning. _softplus(x) is already fp32-stable and can't
    # overflow until x is itself near fp32's own ~3.4e38 ceiling, so 1+softplus(x)
    # and its log stay safe across the entire practically-reachable range.
    x32 = x.to(tl.float32)
    result = tl.log(1.0 + _softplus(x32))
    return result.to(x.dtype)


@triton.jit
def _asinh(x):
    x32 = x.to(tl.float32)
    ax = tl.abs(x32)
    magnitude = tl.log(ax + tl.sqrt(ax * ax + 1.0))
    result = tl.where(x32 < 0, -magnitude, magnitude)
    return result.to(x.dtype)


@triton.jit
def _riemannian_project_kernel(
    x_ptr, weight_ptr, packed_scratch_ptr, diag_param_mask_ptr,
    seq_len, heads, head_dim, n_params, temperature,
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

        raw0_tile = tl.dot(x_block, w_tile, allow_tf32=False)  # (BLOCK_S, BLOCK_P_TILE), fp32 accumulate
        raw_tile = raw0_tile / temperature

        is_diag_tile = tl.load(diag_param_mask_ptr + p_off, mask=p_mask, other=0) != 0  # (BLOCK_P_TILE,)

        # packed_scratch is stored in fp32 unconditionally (see caller), NOT
        # x_block.dtype -- belt-and-suspenders alongside _log_softplus's own
        # practical self-boundedness (see that function's docstring and this
        # module's docstring for the full reasoning). The diagonal branch is
        # shifted by _DIAG_PACKED_BASELINE so it's exactly 0 at raw=0 (identity
        # at init) -- see module docstring.
        packed_tile = tl.where(
            is_diag_tile[None, :],
            _log_softplus(raw_tile) - _DIAG_PACKED_BASELINE,
            _asinh(raw_tile),
        )
        packed_tile = tl.where(p_mask[None, :], packed_tile, 0.0)
        tl.store(
            packed_scratch_ptr + token_off[:, None] * n_params + p_off[None, :],
            packed_tile,
            mask=s_mask[:, None] & p_mask[None, :],
        )


@triton.jit
def _riemannian_apply_fwd_kernel(
    x_ptr, out_ptr, packed_scratch_ptr,
    idx_map_ptr, lower_mask_ptr,
    head_dim, n_params,
    BLOCK_D: tl.constexpr,
):
    # Per-token: build L_raw from the already-computed packed_scratch (written
    # by _riemannian_project_kernel from x), then L = identity + L_raw,
    # then apply L to x itself (self-referential).
    #
    # Everything from here through `out` runs in fp32, not just at the fp16/bf16
    token_id = tl.program_id(0)
    out_dtype = x_ptr.dtype.element_ty

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    x = tl.load(x_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0).to(tl.float32)

    row = tl.arange(0, BLOCK_D)[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    dd_mask = (row < head_dim) & (col < head_dim)

    idx_map = tl.load(idx_map_ptr + row * head_dim + col, mask=dd_mask, other=0)
    lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0

    gathered = tl.load(packed_scratch_ptr + token_id * n_params + idx_map, mask=lower, other=0.0).to(tl.float32)
    L_raw = tl.where(lower, gathered, 0.0)
    L = tl.where(row == col, 1.0, 0.0) + L_raw

    u = tl.sum(x[:, None] * L, axis=0)          # u[j] = sum_i x[i]*L[i,j]
    out = tl.sum(u[None, :] * L, axis=1)        # out[j] = sum_i u[i]*L[j,i]

    tl.store(out_ptr + token_id * head_dim + d_off, out.to(out_dtype), mask=d_mask)


@triton.jit
def _riemannian_grad_L_and_raw_kernel(
    x_ptr, grad_out_ptr,
    grad_x_L_ptr, grad_raw_scratch_ptr,
    packed_scratch_ptr, grad_L_scratch_ptr,
    idx_map_ptr, lower_mask_ptr, tril_rows_ptr, tril_cols_ptr,
    head_dim, n_params, temperature,
    BLOCK_D: tl.constexpr, BLOCK_P_TILE: tl.constexpr,
):
    # Per-token (same reasoning as _riemannian_apply_fwd_kernel): rebuild L,
    # compute grad_u/grad_L, then grad_packed/grad_raw0 tiled along n_params.
    # Writes grad_x's L-application-path term directly (the projection-path
    # term, grad_raw0 @ weight^T, is computed separately in the batched
    # _riemannian_weight_and_gradxw_kernel and added in plain PyTorch
    # afterward) and grad_raw0 to scratch, for that next kernel to consume as
    # a batched matmul operand.
    token_id = tl.program_id(0)
    out_dtype = x_ptr.dtype.element_ty

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim

    # Everything from here through grad_L runs in fp32, not just the final
    # store -- L/u/grad_u/grad_L are each built from several chained fp16
    # multiply-adds, and precision lost early compounds by the time it reaches
    # grad_x_L/grad_L_raw.
    x = tl.load(x_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + token_id * head_dim + d_off, mask=d_mask, other=0.0).to(tl.float32)

    row = tl.arange(0, BLOCK_D)[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    dd_mask = (row < head_dim) & (col < head_dim)
    idx_map = tl.load(idx_map_ptr + row * head_dim + col, mask=dd_mask, other=0)
    lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0
    gathered = tl.load(packed_scratch_ptr + token_id * n_params + idx_map, mask=lower, other=0.0).to(tl.float32)
    L_raw = tl.where(lower, gathered, 0.0)
    L = tl.where(row == col, 1.0, 0.0) + L_raw
    u = tl.sum(x[:, None] * L, axis=0)

    # grad_u[i] = sum_j grad_out[j] * L[j,i]
    grad_u = tl.sum(grad_out[:, None] * L, axis=0)
    # grad_L[a,b] = grad_out[a]*u[b] + x[a]*grad_u[b] -- from L's role in u/out,
    # independent of how L itself is parameterized from L_raw.
    grad_L = grad_out[:, None] * u[None, :] + x[:, None] * grad_u[None, :]
    grad_L = tl.where(dd_mask, grad_L, 0.0)

    # grad_x's L-application-path term: sum_j grad_u[j]*L[i,j]. The
    # projection-path term is added later. Cast back down to the storage dtype
    # here, at the end, rather than carrying fp16 through the computation above.
    grad_x_L = tl.sum(grad_u[None, :] * L, axis=1)
    tl.store(grad_x_L_ptr + token_id * head_dim + d_off, grad_x_L.to(out_dtype), mask=d_mask)

    # dL/dL_raw = 1 (L = identity + L_raw, no metric_scale gate anymore --
    # weight's gradient is the true, undamped gradient from step 0; see
    # module docstring)
    grad_L_raw = grad_L.to(out_dtype)
    tl.store(grad_L_scratch_ptr + token_id * head_dim * head_dim + row * head_dim + col, grad_L_raw, mask=dd_mask)

    # Barrier: the tile loop below gathers grad_packed back out of grad_L_scratch via
    # dynamically-computed (tr,tc) indices -- a different access pattern than the dense
    # (row,col) grid the store above just wrote with. Without an explicit barrier,
    # nothing guarantees that store is actually visible before this program's own later
    # load reads the same (just-allocated, uninitialized-until-written) global memory back
    tl.debug_barrier()

    n_tiles = tl.cdiv(n_params, BLOCK_P_TILE)
    for tile in range(n_tiles):
        p_off = tile * BLOCK_P_TILE + tl.arange(0, BLOCK_P_TILE)
        p_mask = p_off < n_params
        packed_tile = tl.load(packed_scratch_ptr + token_id * n_params + p_off, mask=p_mask, other=0.0)

        tr = tl.load(tril_rows_ptr + p_off, mask=p_mask, other=0)
        tc = tl.load(tril_cols_ptr + p_off, mask=p_mask, other=0)

        grad_packed = tl.load(grad_L_scratch_ptr + token_id * head_dim * head_dim + tr * head_dim + tc, mask=p_mask, other=0.0)

        is_diag_k = tr == tc
        packed_tile32 = packed_tile.to(tl.float32)

        # packed = log(1+softplus(raw)) - C => d(packed)/d(raw) = sigmoid(raw)/(1+softplus(raw))
        # -- the constant shift C doesn't affect the derivative. Substituting
        # softplus(raw)=e^(packed+C)-1 (inverting the UNSHIFTED packing) and
        # sigmoid(raw)=1-e^(-softplus(raw)) (the same identity plain softplus's backward
        # already relied on) gives a closed form purely in terms of the already-computed
        # (shifted) packed value plus the same constant C -- verified against autograd
        # directly (see model.py's RiemannianMetric.forward and this module's docstring
        # for the full derivation).
        packed_diag_unshifted = packed_tile32 + _DIAG_PACKED_BASELINE
        diag_grad = (
            tl.exp(-packed_diag_unshifted) - tl.exp(1.0 - tl.exp(packed_diag_unshifted) - packed_diag_unshifted)
        ).to(packed_tile.dtype)

        # packed = asinh(raw)  =>  d(packed)/d(raw) = 1/sqrt(1+raw^2) = 1/cosh(asinh(raw))
        # = sech(packed) -- same trick as the diagonal (via cosh^2 - sinh^2 = 1,
        # since sinh(packed) = raw here), expressed purely via the already-computed
        # packed value.
        offdiag_grad = (2.0 / (tl.exp(packed_tile32) + tl.exp(-packed_tile32))).to(packed_tile.dtype)

        d_packed_d_raw = tl.where(is_diag_k, diag_grad, offdiag_grad)

        grad_raw = grad_packed * d_packed_d_raw
        grad_raw0 = grad_raw / temperature

        # Explicit cast, not implicit: packed_tile (hence diag_grad/offdiag_grad/
        # grad_raw0) is fp32 now that packed_scratch is stored unconditionally in
        # fp32 (see _riemannian_project_kernel), but grad_raw_scratch is still
        # allocated in the activation's own (possibly narrower) dtype -- don't
        # rely on tl.store's implicit-cast behavior across a real dtype mismatch.
        tl.store(grad_raw_scratch_ptr + token_id * n_params + p_off, grad_raw0.to(out_dtype), mask=p_mask)


@triton.jit
def _riemannian_weight_and_gradxw_kernel(
    x_ptr, weight_ptr, grad_raw_scratch_ptr,
    grad_x_w_ptr, grad_weight_ptr,
    seq_len, heads, head_dim, n_params,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_P_TILE: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # Batched matmul mirror of _riemannian_project_kernel, on the backward
    # side: grad_weight_h[d,p] = sum_s x[s,d]*grad_raw0[s,p]  (x_block^T @ grad_raw0_block)
    #       grad_x_w[s,d]      = sum_p grad_raw0[s,p]*weight_h[d,p]  (grad_raw0_block @ weight_h^T)
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
    diag_param_mask = (tril_rows == tril_cols).to(torch.int32)
    return tril_rows.to(torch.int32), tril_cols.to(torch.int32), idx_map.to(torch.int32), lower_mask, diag_param_mask


class RiemannianMetricKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, temperature):
        N, heads, S, head_dim = x.shape
        n_params = head_dim * (head_dim + 1) // 2
        n_tokens = N * heads * S
        n_nh = N * heads

        x_c = x.contiguous()
        weight_c = weight.contiguous().to(x_c.dtype)
        tril_rows, tril_cols, idx_map, lower_mask, diag_param_mask = _structural_buffers(head_dim, x.device)

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

        # fp32 unconditionally, regardless of x's dtype -- see _riemannian_project_kernel's
        # comment: this scratch buffer is never exposed outside this file, so there's no
        # reason to risk narrowing it.
        packed_scratch = torch.empty(n_tokens, n_params, dtype=torch.float32, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid_proj](
            x_c.view(n_tokens, head_dim), weight_c, packed_scratch, diag_param_mask,
            S, heads, head_dim, n_params, temperature,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_P_TILE=BLOCK_P_TILE,
            num_warps=8,
        )

        out = torch.empty_like(x_c)
        _riemannian_apply_fwd_kernel[(n_tokens,)](
            x_c.view(n_tokens, head_dim), out.view(n_tokens, head_dim), packed_scratch,
            idx_map, lower_mask,
            head_dim, n_params,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_c, weight_c, tril_rows, tril_cols, idx_map, lower_mask, diag_param_mask)
        ctx.temperature = temperature
        ctx.shapes = (N, heads, S, head_dim, n_params, n_tokens, n_nh)
        ctx.blocks = (BLOCK_D, BLOCK_S, BLOCK_P_TILE)
        ctx.orig_weight_dtype = weight.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, tril_rows, tril_cols, idx_map, lower_mask, diag_param_mask = ctx.saved_tensors
        N, heads, S, head_dim, n_params, n_tokens, n_nh = ctx.shapes
        BLOCK_D, BLOCK_S, BLOCK_P_TILE = ctx.blocks
        grad_out_c = grad_out.contiguous()

        # Recompute packed (not saved from forward) -- see module docstring. fp32
        # unconditionally, same reasoning as the forward allocation above.
        packed_scratch = torch.empty(n_tokens, n_params, dtype=torch.float32, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid_proj](
            x.view(n_tokens, head_dim), weight, packed_scratch, diag_param_mask,
            S, heads, head_dim, n_params, ctx.temperature,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_P_TILE=BLOCK_P_TILE,
            num_warps=8,
        )

        num_blocks = n_nh * triton.cdiv(S, BLOCK_S)
        NUM_GROUPS = min(32, num_blocks)

        grad_L_scratch = torch.empty(n_tokens, head_dim, head_dim, dtype=x.dtype, device=x.device)
        grad_raw_scratch = torch.empty(n_tokens, n_params, dtype=x.dtype, device=x.device)
        grad_x_L = torch.empty_like(x)
        _riemannian_grad_L_and_raw_kernel[(n_tokens,)](
            x.view(n_tokens, head_dim), grad_out_c.view(n_tokens, head_dim),
            grad_x_L.view(n_tokens, head_dim), grad_raw_scratch,
            packed_scratch, grad_L_scratch,
            idx_map, lower_mask, tril_rows, tril_cols,
            head_dim, n_params, ctx.temperature,
            BLOCK_D=BLOCK_D, BLOCK_P_TILE=BLOCK_P_TILE,
        )

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
        return (
            grad_x,
            grad_weight_f32.to(ctx.orig_weight_dtype),
            None,
        )
