"""
Triton forward+backward kernels for RiemannianMetric.

MATH:
    forward (per head h, per token, x_h = the key itself -- self-referential:
    x both builds L and gets transformed by it):

        DIAGONAL (head_dim values -- guarantees non-singularity):
            raw_diag[i]    = x_h[i] * weight_diag_h[i]          (element-wise,
                             NOT a dot product)
            packed_diag[i] = log1p(softplus(raw_diag[i])) - C    C = log(1+softplus(0))
            L_diag[i]      = 1 + packed_diag[i]                  always > 1-C ~= 0.4734

        OFF-DIAGONAL (heads*head_dim*rank values -- see model.py's
        RiemannianMetric docstring for the full "why gated low-rank instead
        of a dense per-triangular-position unpack" reasoning):
            raw_gate_k = x_h . weight_W_hk                       k = 1..rank,
                         a genuine dot product over all head_dim components
            gate_k     = asinh(raw_gate_k / scale)               scale = sqrt(head_dim)
            L_offdiag  = (1/sqrt(rank)) * sum_k gate_k * (weight_U_hk (x) weight_V_hk)
                         restricted to strictly-lower-triangular positions

        L   = identity + diag(L_diag - 1) + L_offdiag
        u   = x_h @ L
        out = u @ L^T   (= x_h @ (L @ L^T), the actual SPD metric applied to x_h)

    backward (per head h, per token):
        grad_u      = grad_out @ L
        grad_L      = outer(grad_out, u) + outer(x_h, grad_u)     -- same as before,
                      depends only on L's role in u/out, not on how L was built
        grad_L_diag[i]    = grad_L[i, i]
        grad_L_offdiag    = grad_L restricted to strictly-lower-triangular positions
                            (dL/dL_raw = 1 in both pieces, same as the diagonal-only
                            design this replaced -- no metric_scale-style gate)

        diagonal branch (IDENTICAL closed form to the design this replaced --
        see _riemannian_grad_per_token_kernel for the derivation, unchanged):
            grad_raw_diag[i]    = grad_L_diag[i] * [exp(-(packed_diag[i]+C)) - exp(1 - exp(packed_diag[i]+C) - (packed_diag[i]+C))]
            grad_x            += grad_raw_diag[i] * weight_diag_h[i]
            grad_weight_diag_h[i] += grad_raw_diag[i] * x_h[i]

        off-diagonal branch (new):
            grad_gate_k   = (1/sqrt(rank)) * sum_ij grad_L_offdiag[i,j] * weight_U_hk[i] * weight_V_hk[j]
            grad_weight_U_hk[i] += (1/sqrt(rank)) * gate_k * sum_j grad_L_offdiag[i,j] * weight_V_hk[j]
            grad_weight_V_hk[j] += (1/sqrt(rank)) * gate_k * sum_i grad_L_offdiag[i,j] * weight_U_hk[i]
            grad_raw_gate_k = grad_gate_k / (scale * cosh(gate_k))     -- same "d(asinh)/d(raw) = sech"
                              trick as the design this replaced, now applied to gate_k
            grad_x         += grad_raw_gate_k * weight_W_hk[i]  (summed over k, i)
            grad_weight_W_hk[i] += grad_raw_gate_k * x_h[i]     (summed over tokens sharing this head)

DESIGN:
    Two structurally different pieces, computed by four kernels:

        1. _riemannian_project_kernel: BATCHED across tokens (one real
           tl.dot GEMM per (head, seq-tile) block: x_block @ weight_W_h,
           mirroring the old design's dominant-matmul step) computes
           raw_gate -> gate for every token, plus (cheap, element-wise,
           no matmul needed) raw_diag -> L_diag. Both stored to scratch.

        2. _riemannian_apply_fwd_kernel: BATCHED over (head, seq-tile)
           programs, same grid shape as #1 -- but each token in the tile
           still gets its OWN L (this can't be batched away the way the
           projection step can: L_offdiag must respect a strictly-lower-
           triangular mask that a fully-batched two-GEMM formulation
           (`x@L_offdiag = (gate*(x@U))@V^T`) cannot express, since that
           formulation is only algebraically valid for the FULL, unrestricted
           matrix of matrix-rank at most `rank` -- the mask couples each
           (i,j) individually in a way that doesn't factor through the
           rank-1-sum structure).
           What IS batched here is weight_U/weight_V *loading*: both load
           ONCE per (head, seq-tile) program and get reused by every token
           in the tile, instead of being reloaded from global memory once
           per token (the diagnosed cause of an earlier version's ~2x
           slowdown vs. the naive eager reference). Each token's own
           x/gate/L_diag are pulled out of this tile's batched tensors via
           a one-hot masked-sum -- a SINGLE such extraction per token, not
           nested inside a further per-rank-component loop, so its cost is
           O(BLOCK_S) per token rather than the O(BLOCK_R) blowup an
           earlier, abandoned attempt hit by extracting inside a
           per-rank-component loop instead.
           When rank fits in one BLOCK_R tile (the common/default case),
           x@L_offdiag and (x@L_offdiag)@L_offdiag^T are each computed via
           _cumsum_apply_suffix/_cumsum_apply_prefix -- a cumulative-sum-
           based reformulation (see those functions' shared comment) that
           NEVER materializes the (D,D) L_offdiag matrix a tl.dot-based
           construction would need, dropping this kernel's per-token cost
           from O(head_dim^2 * rank) to O(head_dim * rank). When rank spans
           multiple BLOCK_R tiles (rare -- only past ~BLOCK_R distinct rank
           components, see NOTE on tiling below), the kernel falls back to
           the older tl.dot-based per-tile construction instead, since the
           cumsum trick's additive-across-r-tiles decomposition wasn't
           re-derived for that path.

        3. _riemannian_grad_per_token_kernel: same batching restructure as
           #2, for the same reasons -- weight_U/weight_V/weight_diag_h load
           once per tile, each token still gets its own gradient contribution
           computed individually (grad_L itself depends only on L's role in
           u/out, not on how L was built, so that derivation is unchanged).
           The common (single-BLOCK_R-tile) path mirrors #2: grad_gate/
           grad_U/grad_V/grad_x are all derived from the same suffix/
           exclusive-prefix cumsum intermediates #2 uses, rather than
           building the (D,D) grad_L_offdiag matrix and contracting it via
           two more tl.dot calls -- same O(head_dim^2*rank) ->
           O(head_dim*rank) reduction as #2, see _cumsum_apply_suffix/
           _cumsum_apply_prefix's comment for the derivation. The rare
           multi-BLOCK_R-tile path keeps the old tl.dot-based construction.
           grad_weight_diag/grad_U/grad_V accumulate over the whole tile's
           tokens in registers before a single atomic_add each at the end of
           the program, rather than one atomic_add per token -- an
           incidental drop in atomic contention on top of the weight-reload
           fix. Unlike the previous (dense) design, this needs NO
           tl.debug_barrier: the barrier bug there came from storing to a
           scratch buffer via one access pattern and gathering it back via a
           different one WITHIN the same kernel program. Here, every
           reduction (grad_gate, grad_weight_U/V contributions) operates
           directly on tiles already held in registers from this same
           program -- nothing round-trips through global memory and back
           within one launch.

        4. _riemannian_weight_w_and_gradxw_kernel: BATCHED (mirrors #1 and
           the design this replaced's own weight-and-gradxw kernel exactly,
           just with rank in place of n_params): grad_weight_W and the
           off-diagonal path's contribution to grad_x, both genuine GEMMs
           over the token dimension.

    grad_weight_diag, grad_weight_U, grad_weight_V are genuine reductions
    (summed over every token sharing a head) computed inside the per-token
    kernel where the needed intermediates are already available. Accumulated
    via grouped tl.atomic_add into small (NUM_GROUPS, ...) buffers -- same
    pattern the design this replaced used for its own grad_weight, for the
    same reason: spreading contention across up to 32 buckets instead of
    one address per head/head_dim/rank component.

    NOTE on rank: model.py defaults to rank=head_dim (always full
    rank, no wasted capacity -- see RiemannianMetric's docstring) but this is
    now a decoupled, independently-settable value (ModelConfig.metric_rank
    / --metric-rank), not hardwired to head_dim.

    Earlier revision of this NOTE claimed that at rank=head_dim this kernel
    does the "same asymptotic cost as the dense design it replaced" and that
    the batched restructure in #2/#3 buys no FLOP reduction. That was true of
    the tl.dot-based L_offdiag construction this kernel used to have --
    building the full (D,D) L_offdiag matrix via tl.dot costs O(D^2*rank)
    per token even though every caller immediately contracts it back down to
    a (D,) vector, so at rank=head_dim it was actually paying O(D^3), a
    full D-factor MORE than the dense design's own O(D^2) per-token
    matrix-vector cost -- not "the same", worse. The cumsum-based
    reformulation in #2/#3's single-BLOCK_R-tile path (see
    _cumsum_apply_suffix/_cumsum_apply_prefix) fixes this: it computes
    x@L_offdiag directly in O(D*rank), without ever forming the (D,D)
    matrix. At rank=head_dim that's O(D^2) -- now genuinely on par with the
    dense design, as the low-rank construction was always meant to be one
    the FLOP count only shrinks BELOW dense once rank is set below
    head_dim. The multi-BLOCK_R-tile path (rank > BLOCK_R, rare) has not
    been re-derived for this and still pays the older O(D^2*rank) cost.

    BLOCK sizing below follows the same conservative floors the design this
    replaced arrived at (empirically, on real hardware) for its own tl.dot
    usage, but has NOT been re-validated against actual shared-memory limits
    for this new kernel's tile shapes -- there was no GPU available to do
    that validation while writing this. Treat the num_warps/BLOCK_* choices
    here as a reasonable starting point, not tuned values, until confirmed
    against kernels/verify_riemannian_metric.py on real hardware.
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
    # log(1 + softplus(x)) -- PRACTICALLY self-bounding (packed=50 needs
    # x~5e21) without an explicit clamp, and smooth everywhere. See
    # model.py's RiemannianMetric docstring for the full reasoning.
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


# The two routines below replace the old "materialize L_offdiag (D,D) via
# tl.dot, then apply it to a vector" pattern with a cumsum-based one that
# never materializes the (D,D) matrix at all. This is NOT just a constant-
# factor tuning change: it's an asymptotic reduction from O(head_dim^2 *
# rank) to O(head_dim * rank) per token/per application, because building
# L_offdiag via tl.dot(gated_U, V^T) pays for the FULL (D,D) rank-<=rank
# matrix even though every caller immediately contracts it back down to a
# (D,) vector -- work the old design paid for and then threw away.
#
# The trick: L_offdiag[i,j] = sum_r gate[r]*U[i,r]*V[j,r] for i>j (else 0)
# is a SUM OF RANK-1 TERMS masked to strictly-lower-triangular. Whether a
# given (i,j) pair survives the mask depends only on the ORDERING of i vs j
# -- exactly the structure a prefix/suffix cumulative sum along the shared
# "D" axis captures for free, without ever touching the other (D,D)
# combinations the mask would zero out anyway:
#
#   vec @ L  (used for both u=x@L and, in backward, grad_u=grad_out@L):
#     result[j] = vec[j]*L_diag[j] + sum_{i>j} vec[i]*L_offdiag[i,j]
#               = vec[j]*L_diag[j] + sum_r gate[r]*combine_w[j,r]*suffix[j,r]
#     where suffix[j,r] = sum_{i>j} vec[i]*contract_w[i,r]
#                       = total[r] - inclusive_cumsum_i(vec*contract_w)[j,r]
#     (contract_w=U, combine_w=V for this direction -- see call sites)
#
#   vec @ L^T (used for both out=u@L^T and, in backward, grad_x_L=grad_u@L^T):
#     result[j] = vec[j]*L_diag[j] + sum_{i<j} vec[i]*L_offdiag[j,i]
#               = vec[j]*L_diag[j] + sum_r gate[r]*combine_w[j,r]*excl[j,r]
#     where excl[j,r] = sum_{i<j} vec[i]*contract_w[i,r]
#                     = inclusive_cumsum_i(vec*contract_w)[j,r] - (vec*contract_w)[j,r]
#     (contract_w=V, combine_w=U for this direction -- see call sites)
#
# `gate` here must already have rank_scale folded in (gate*rank_scale, i.e.
# the same "effective per-mode coefficient" the old code got by multiplying
# gated_U by mode_scale) -- callers that need the gradient w.r.t. the
# UNSCALED gate (for the asinh backward chain) apply rank_scale separately,
# same convention the old design used.
#
# Both `total`/`suffix` and `excl` are returned alongside `result` because
# backward's grad_gate/grad_U/grad_V reuse exactly these intermediates
# rather than recomputing them -- see _riemannian_grad_per_token_kernel.
#
# Correctness at head_dim < BLOCK_D (padding): U/V/vec are all loaded with
# other=0.0 for row indices >= head_dim upstream of these routines, so
# `contract_w` is zero at every padded row regardless of `vec`. Padded rows
# sit at the END of the index range (index >= head_dim), and a cumulative
# sum's value at any VALID index j < head_dim depends only on entries at
# indices <= j -- all valid -- so padding can never leak into a valid
# result, whether or not `vec` itself happens to be nonzero there.
@triton.jit
def _cumsum_apply_suffix(vec, contract_w, combine_w, gate, L_diag):
    contracted = vec[:, None] * contract_w                       # (D, R)
    total = tl.sum(contracted, axis=0)                           # (R,)
    incl = tl.cumsum(contracted, axis=0)                         # (D, R)
    suffix = total[None, :] - incl                                # (D, R)
    offdiag = tl.sum(gate[None, :] * combine_w * suffix, axis=1)  # (D,)
    result = vec * L_diag + offdiag
    return result, suffix


@triton.jit
def _cumsum_apply_prefix(vec, contract_w, combine_w, gate, L_diag):
    contracted = vec[:, None] * contract_w                       # (D, R)
    incl = tl.cumsum(contracted, axis=0)                         # (D, R)
    excl = incl - contracted                                      # (D, R)
    offdiag = tl.sum(gate[None, :] * combine_w * excl, axis=1)   # (D,)
    result = vec * L_diag + offdiag
    return result, excl


@triton.jit
def _riemannian_project_kernel(
    x_ptr, weight_W_ptr, weight_diag_ptr, gate_scratch_ptr, L_diag_scratch_ptr,
    seq_len, heads, head_dim, rank, scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_R: tl.constexpr,
):
    # 3rd grid axis tiles the rank dimension: BLOCK_R is a fixed-size tile,
    # NOT a bound on rank -- rank can be arbitrarily larger than
    # BLOCK_R now (that's the whole point of this axis; previously this
    # kernel silently required rank <= BLOCK_R, see module docstring's
    # NOTE on tiling). Each r_tile program computes and stores only ITS OWN
    # column-slice of gate -- no cross-tile accumulation needed here, since
    # each rank component's gate value depends only on that component's own weight_W
    # column, not on any other rank component (this is genuinely embarrassingly
    # parallel across r_tile, unlike the off-diagonal apply/grad kernels
    # below, which need a real reduction over the rank dimension).
    nh = tl.program_id(0)          # flat (batch, head) index: n * heads + h
    s_tile = tl.program_id(1)
    r_tile = tl.program_id(2)
    head_id = nh % heads

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_off < rank

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D)

    token_off = nh * seq_len + s_off  # token_id for each row in this block

    # --- gate: genuine batched GEMM (x_block @ weight_W_h), the dominant
    # FLOPs of this kernel, same tensor-core-via-tl.dot treatment the design
    # this replaced gave its own analogous projection step ---
    w_base = weight_W_ptr + head_id * head_dim * rank
    w_tile = tl.load(
        w_base + d_off[:, None] * rank + r_off[None, :],
        mask=d_mask[:, None] & r_mask[None, :], other=0.0,
    )  # (BLOCK_D, BLOCK_R) -- this r_tile's column-slice of weight_W_h
    raw_gate_tile = tl.dot(x_block, w_tile, allow_tf32=False) / scale  # (BLOCK_S, BLOCK_R), fp32 accumulate
    gate_tile = _asinh(raw_gate_tile)
    gate_tile = tl.where(r_mask[None, :], gate_tile, 0.0)
    tl.store(
        gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
        gate_tile,
        mask=s_mask[:, None] & r_mask[None, :],
    )

    # --- diagonal: element-wise, no matmul needed (only head_dim independent
    # values, unlike the gate's genuine head_dim-wide dot product). Doesn't
    # depend on r_tile at all -- computed/stored only once, by the r_tile==0
    # program, to avoid every r_tile program redundantly repeating it (they'd
    # all compute the identical value, but there's no reason to pay for that
    # BLOCK_S*BLOCK_D work NUM_R_TILES times over). This is a scalar
    # (program-uniform) compile-/run-time comparison, not a per-lane one, so
    # it's ordinary divergence-free control flow.
    if r_tile == 0:
        diag_base = weight_diag_ptr + head_id * head_dim
        weight_diag_tile = tl.load(diag_base + d_off, mask=d_mask, other=0.0).to(tl.float32)  # (BLOCK_D,)
        raw_diag_tile = x_block.to(tl.float32) * weight_diag_tile[None, :]  # (BLOCK_S, BLOCK_D)
        L_diag_tile = 1.0 + _log_softplus(raw_diag_tile) - _DIAG_PACKED_BASELINE
        L_diag_tile = tl.where(d_mask[None, :], L_diag_tile, 0.0)
        tl.store(
            L_diag_scratch_ptr + token_off[:, None] * head_dim + d_off[None, :],
            L_diag_tile,
            mask=s_mask[:, None] & d_mask[None, :],
        )


@triton.jit
def _riemannian_apply_fwd_kernel(
    x_ptr, out_ptr, gate_scratch_ptr, L_diag_scratch_ptr, weight_U_ptr, weight_V_ptr, lower_mask_ptr,
    seq_len, heads, head_dim, rank, rank_scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_R: tl.constexpr, NUM_R_TILES: tl.constexpr,
):
    # Batched over a (head, seq-tile) program -- same grid shape as
    # _riemannian_project_kernel -- so weight_U/weight_V load ONCE per
    # (head, seq-tile, rank-tile) instead of being reloaded from global
    # memory once per token the way the first version of this kernel did
    # (the diagnosed cause of that version's 0.48x "speedup").
    #
    # Each token in the tile still gets its OWN masked L_offdiag built
    # individually via tl.dot against the shared U/V, rather than computing
    # x@L_offdiag for the whole tile via two batched GEMMs
    # ((gate*(x@U))@V^T). That two-GEMM form is only algebraically valid for
    # the FULL, unrestricted matrix of matrix-rank at most `rank` -- it cannot respect the
    # strictly-lower-triangular mask L_offdiag requires, because the mask
    # couples each (i,j) individually in a way that doesn't factor through
    # the rank-1-sum structure. Dropping the mask would silently break the
    # det(L)=prod(diag) non-singularity guarantee the triangular structure
    # exists for, so it was rejected even though it would have been the
    # fastest option. See the surrounding conversation.
    #
    # Per-token values (x, gate, L_diag) are pulled out of this tile's
    # batched tensors via a one-hot masked-sum -- a SINGLE such extraction
    # per token, not nested inside a further per-rank-component loop -- so its cost is
    # O(BLOCK_S) per token, not O(BLOCK_R) like the masked-sum extraction
    # that made the previous (unrolled-loop) attempt regress further.
    #
    # NUM_R_TILES = cdiv(rank, BLOCK_R): rank can now exceed one
    # BLOCK_R tile (previously it silently couldn't -- see module docstring's
    # NOTE on tiling). NUM_R_TILES==1 (the common case, and the only case
    # this kernel supported before) takes the exact same code path as
    # before, byte-for-byte -- the tiling logic below is fully skipped, at
    # zero cost, whenever it isn't needed. NUM_R_TILES>1 is a genuinely
    # different, more expensive code path: see the "else" branch's own
    # comment for why.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    out_dtype = x_ptr.dtype.element_ty

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    token_off = nh * seq_len + s_off

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)  # (BLOCK_S, BLOCK_D)
    L_diag_block = tl.load(
        L_diag_scratch_ptr + token_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D), fp32

    u_base = weight_U_ptr + head_id * head_dim * rank
    v_base = weight_V_ptr + head_id * head_dim * rank

    if NUM_R_TILES == 1:
        # Fast path: per-token L_offdiag is no longer materialized as a
        # (D,D) matrix via tl.dot at all -- see _cumsum_apply_suffix/
        # _cumsum_apply_prefix's shared comment for the O(D^2*R) -> O(D*R)
        # derivation this replaces. lower_mask_ptr is unused here as a
        # result (the causal structure is now implicit in the cumsum
        # direction, not an explicit mask) -- it's still loaded/used by the
        # tiled `else` branch below, which keeps the old tl.dot-based
        # construction.
        r_off = tl.arange(0, BLOCK_R)
        r_mask = r_off < rank
        gate_block = tl.load(
            gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
            mask=s_mask[:, None] & r_mask[None, :], other=0.0,
        )  # (BLOCK_S, BLOCK_R), fp32
        U = tl.load(
            u_base + d_off[:, None] * rank + r_off[None, :],
            mask=d_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)  # (BLOCK_D, BLOCK_R) -- loaded ONCE, reused by every token below
        V = tl.load(
            v_base + d_off[:, None] * rank + r_off[None, :],
            mask=d_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)

        out_block = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
        for s in range(BLOCK_S):
            one_hot_s = tl.arange(0, BLOCK_S) == s
            x_s = tl.sum(tl.where(one_hot_s[:, None], x_block, 0.0), axis=0)            # (BLOCK_D,)
            gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_block, 0.0), axis=0)      # (BLOCK_R,)
            L_diag_s = tl.sum(tl.where(one_hot_s[:, None], L_diag_block, 0.0), axis=0)  # (BLOCK_D,)

            gate_s_scaled = gate_s * rank_scale
            # u = x @ L: contract against U (suffix-cumsum), combine with V.
            u_s, _ = _cumsum_apply_suffix(x_s, U, V, gate_s_scaled, L_diag_s)
            # out = u @ L^T: contract against V (exclusive-prefix-cumsum), combine with U.
            out_s, _ = _cumsum_apply_prefix(u_s, V, U, gate_s_scaled, L_diag_s)

            out_block += out_s[None, :] * one_hot_s[:, None]
    else:
        row = tl.arange(0, BLOCK_D)[:, None]
        col = tl.arange(0, BLOCK_D)[None, :]
        dd_mask = (row < head_dim) & (col < head_dim)
        lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0

        # Tiled path: rank spans multiple BLOCK_R tiles, so U/V/gate
        # can't be loaded in one shot anymore. Two passes over r_tile, each
        # with s as the inner (unrolled) loop:
        #   Pass 1 builds u_block = x@L incrementally, one r_tile's worth of
        #   L_offdiag at a time. Valid because u=x@L is LINEAR in L, and
        #   masking distributes over the tile-wise sum
        #   (mask*(sum_tiles L_tile) == sum_tiles(mask*L_tile)) -- so each
        #   tile's own masked contribution can be added in independently,
        #   without ever materializing the full (untiled) L_offdiag for a
        #   token. rank_scale is likewise just a constant multiplied into
        #   every tile before accumulating, so it distributes the same way.
        #   Pass 2 builds out_block = u@L^T the same way, now that pass 1
        #   has made u_block fully known -- out is linear in L too, once u
        #   is fixed.
        # Each pass reloads weight_U/weight_V/gate once per r_tile (still
        # shared across all BLOCK_S tokens within that r_tile, same sharing
        # the fast path gets) -- so tiling costs O(NUM_R_TILES) reloads per
        # pass (2*NUM_R_TILES total), NOT O(BLOCK_S*NUM_R_TILES); it does
        # NOT reload per token. This only runs when rank actually
        # exceeds one BLOCK_R tile -- the NUM_R_TILES==1 branch above is
        # completely unaffected and stays exactly as fast as before.
        u_block = x_block * L_diag_block  # diagonal contribution to u=x@L (tiling-independent)
        for r_tile in range(NUM_R_TILES):
            r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
            r_mask = r_off < rank
            gate_tile = tl.load(
                gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
                mask=s_mask[:, None] & r_mask[None, :], other=0.0,
            )
            U_tile = tl.load(
                u_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)
            V_tile = tl.load(
                v_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)

            for s in range(BLOCK_S):
                one_hot_s = tl.arange(0, BLOCK_S) == s
                x_s = tl.sum(tl.where(one_hot_s[:, None], x_block, 0.0), axis=0)
                gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_tile, 0.0), axis=0)

                gated_U = U_tile * gate_s[None, :]
                L_offdiag_tile_s = tl.dot(gated_U, tl.trans(V_tile), allow_tf32=False) * rank_scale
                L_offdiag_tile_s = tl.where(lower, L_offdiag_tile_s, 0.0)

                u_contrib_s = tl.sum(x_s[:, None] * L_offdiag_tile_s, axis=0)  # (BLOCK_D,)
                u_block += u_contrib_s[None, :] * one_hot_s[:, None]

        out_block = u_block * L_diag_block  # diagonal contribution to out=u@L^T
        for r_tile in range(NUM_R_TILES):
            r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
            r_mask = r_off < rank
            gate_tile = tl.load(
                gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
                mask=s_mask[:, None] & r_mask[None, :], other=0.0,
            )
            U_tile = tl.load(
                u_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)
            V_tile = tl.load(
                v_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)

            for s in range(BLOCK_S):
                one_hot_s = tl.arange(0, BLOCK_S) == s
                u_s = tl.sum(tl.where(one_hot_s[:, None], u_block, 0.0), axis=0)
                gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_tile, 0.0), axis=0)

                gated_U = U_tile * gate_s[None, :]
                L_offdiag_tile_s = tl.dot(gated_U, tl.trans(V_tile), allow_tf32=False) * rank_scale
                L_offdiag_tile_s = tl.where(lower, L_offdiag_tile_s, 0.0)

                out_contrib_s = tl.sum(u_s[None, :] * L_offdiag_tile_s, axis=1)  # (BLOCK_D,)
                out_block += out_contrib_s[None, :] * one_hot_s[:, None]

    tl.store(
        out_ptr + token_off[:, None] * head_dim + d_off[None, :],
        out_block.to(out_dtype),
        mask=s_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _riemannian_grad_per_token_kernel(
    x_ptr, grad_out_ptr,
    grad_x_ptr, grad_raw_gate_scratch_ptr,
    grad_weight_diag_partial_ptr, grad_U_partial_ptr, grad_V_partial_ptr,
    gate_scratch_ptr, L_diag_scratch_ptr, weight_U_ptr, weight_V_ptr, weight_diag_ptr,
    lower_mask_ptr,
    seq_len, heads, head_dim, rank, scale, rank_scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_R: tl.constexpr, NUM_GROUPS: tl.constexpr,
    NUM_R_TILES: tl.constexpr,
):
    # Batched over a (head, seq-tile) program, same restructuring and for the
    # same reason as the forward kernel above: weight_U/weight_V/weight_diag_h
    # load ONCE per tile instead of once per token, while each token in the
    # tile still gets its own masked L (rebuilt from gate/L_diag, recomputed
    # by _riemannian_project_kernel rather than saved from forward -- same
    # recompute-rather-than-save approach the design this replaced used;
    # measured cheaper than saving here, see backward()'s comment).
    # grad_L itself depends only on L's role in u/out, not on how L is
    # parameterized, so that derivation is unchanged; it's then split into
    # its diagonal and strictly-lower-triangular pieces and backpropped
    # through each piece's own construction, per token.
    #
    # The triangular mask is preserved here for the same reason the forward
    # kernel preserves it -- see that kernel's comment.
    #
    # No tl.debug_barrier needed anywhere here (unlike the design this
    # replaced): every reduction below operates on tiles already held in
    # registers from this same program -- nothing is stored to global memory
    # and gathered back via a different access pattern within this kernel
    # launch, which was specifically what required the barrier before.
    #
    # grad_weight_diag_acc/grad_U_acc/grad_V_acc accumulate over the WHOLE
    # tile's BLOCK_S tokens in registers before a single atomic_add each at
    # the end of the program -- one atomic per tile instead of one per
    # token, an incidental reduction in atomic contention on top of the
    # weight-reload fix.
    #
    # NUM_R_TILES==1 (the common case) takes the exact same code path as
    # before, byte-for-byte. NUM_R_TILES>1 (rank spans multiple BLOCK_R
    # tiles) takes a different, more expensive path -- see its own comment.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    group_id = s_tile % NUM_GROUPS
    out_dtype = x_ptr.dtype.element_ty

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    token_off = nh * seq_len + s_off

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    grad_out_block = tl.load(
        grad_out_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    L_diag_block = tl.load(
        L_diag_scratch_ptr + token_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # fp32

    u_base = weight_U_ptr + head_id * head_dim * rank
    v_base = weight_V_ptr + head_id * head_dim * rank

    weight_diag_h = tl.load(weight_diag_ptr + head_id * head_dim + d_off, mask=d_mask, other=0.0).to(tl.float32)

    grad_x_block = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    if NUM_R_TILES == 1:
        # Fast path: same O(head_dim^2*rank) -> O(head_dim*rank) cumsum-based
        # reformulation as the forward kernel's fast path -- see
        # _cumsum_apply_suffix/_cumsum_apply_prefix's shared comment. Never
        # materializes L_s/grad_L_s/grad_L_offdiag_s as (D,D) matrices, so
        # lower_mask_ptr/row/col/dd_mask are unused here (still used by the
        # tiled `else` branch below, which keeps the old tl.dot-based
        # construction).
        r_off = tl.arange(0, BLOCK_R)
        r_mask = r_off < rank
        gate_block = tl.load(
            gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
            mask=s_mask[:, None] & r_mask[None, :], other=0.0,
        )  # fp32
        U = tl.load(
            u_base + d_off[:, None] * rank + r_off[None, :],
            mask=d_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)  # loaded ONCE, reused by every token below
        V = tl.load(
            v_base + d_off[:, None] * rank + r_off[None, :],
            mask=d_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)

        grad_raw_gate_block = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
        grad_weight_diag_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        grad_U_acc = tl.zeros((BLOCK_D, BLOCK_R), dtype=tl.float32)
        grad_V_acc = tl.zeros((BLOCK_D, BLOCK_R), dtype=tl.float32)

        for s in range(BLOCK_S):
            one_hot_s = tl.arange(0, BLOCK_S) == s
            x_s = tl.sum(tl.where(one_hot_s[:, None], x_block, 0.0), axis=0)
            grad_out_s = tl.sum(tl.where(one_hot_s[:, None], grad_out_block, 0.0), axis=0)
            gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_block, 0.0), axis=0)
            L_diag_s = tl.sum(tl.where(one_hot_s[:, None], L_diag_block, 0.0), axis=0)

            gate_s_scaled = gate_s * rank_scale

            # Recompute u=x@L and grad_u=grad_out@L via the SAME "vec@L"
            # routine (u isn't saved from forward -- see backward()'s
            # comment); both share U/V/gate_s_scaled/L_diag_s, only the
            # vector differs.
            u_s, suffix_x_s = _cumsum_apply_suffix(x_s, U, V, gate_s_scaled, L_diag_s)
            grad_u_s, suffix_gout_s = _cumsum_apply_suffix(grad_out_s, U, V, gate_s_scaled, L_diag_s)

            # grad_x's L-application-path term is exactly "grad_u @ L^T" --
            # the same "vec@L^T" routine forward's own out=u@L^T uses, just
            # applied to grad_u. Also need excl_u (u@L^T's own cumsum
            # intermediate) for grad_U below.
            _, excl_u_s = _cumsum_apply_prefix(u_s, V, U, gate_s_scaled, L_diag_s)
            grad_x_L_s, excl_gu_s = _cumsum_apply_prefix(grad_u_s, V, U, gate_s_scaled, L_diag_s)

            # --- diagonal branch: identical closed form to before. ---
            grad_L_diag_s = grad_out_s * u_s + x_s * grad_u_s
            packed_diag_s = L_diag_s - 1.0
            packed_diag_unshifted_s = packed_diag_s + _DIAG_PACKED_BASELINE
            diag_grad_s = tl.exp(-packed_diag_unshifted_s) - tl.exp(1.0 - tl.exp(packed_diag_unshifted_s) - packed_diag_unshifted_s)
            grad_raw_diag_s = grad_L_diag_s * diag_grad_s

            grad_x_diag_s = grad_raw_diag_s * weight_diag_h
            grad_weight_diag_acc += grad_raw_diag_s * x_s

            # --- off-diagonal branch: grad_gate/grad_U/grad_V, derived from
            # the same suffix/excl cumsum intermediates the applications
            # above already computed (see the module docstring for the
            # derivation) instead of building the (D,D) grad_L_offdiag_s
            # matrix and contracting it via two more tl.dot calls. ---
            inner_V_s = u_s[:, None] * suffix_gout_s + grad_u_s[:, None] * suffix_x_s  # (BLOCK_D, BLOCK_R)
            inner_U_s = grad_out_s[:, None] * excl_u_s + x_s[:, None] * excl_gu_s      # (BLOCK_D, BLOCK_R)

            grad_gate_s = tl.sum(V * inner_V_s, axis=0) * rank_scale   # (BLOCK_R,), w.r.t. UNSCALED gate_s
            grad_U_acc += gate_s_scaled[None, :] * inner_U_s
            grad_V_acc += gate_s_scaled[None, :] * inner_V_s

            # gate = asinh(raw_gate/scale) => d(gate)/d(raw_gate) =
            # 1/(scale*cosh(gate)) -- same sech identity as before.
            grad_raw_gate_s = grad_gate_s * (2.0 / (tl.exp(gate_s) + tl.exp(-gate_s))) / scale

            grad_x_s = grad_x_L_s + grad_x_diag_s
            grad_x_block += grad_x_s[None, :] * one_hot_s[:, None]
            grad_raw_gate_block += grad_raw_gate_s[None, :] * one_hot_s[:, None]

        tl.atomic_add(
            grad_weight_diag_partial_ptr + group_id * heads * head_dim + head_id * head_dim + d_off,
            grad_weight_diag_acc, mask=d_mask,
        )
        tl.atomic_add(
            grad_U_partial_ptr + (group_id * heads + head_id) * head_dim * rank + d_off[:, None] * rank + r_off[None, :],
            grad_U_acc, mask=d_mask[:, None] & r_mask[None, :],
        )
        tl.atomic_add(
            grad_V_partial_ptr + (group_id * heads + head_id) * head_dim * rank + d_off[:, None] * rank + r_off[None, :],
            grad_V_acc, mask=d_mask[:, None] & r_mask[None, :],
        )
        tl.store(
            grad_raw_gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
            grad_raw_gate_block.to(out_dtype), mask=s_mask[:, None] & r_mask[None, :],
        )
    else:
        row = tl.arange(0, BLOCK_D)[:, None]
        col = tl.arange(0, BLOCK_D)[None, :]
        dd_mask = (row < head_dim) & (col < head_dim)
        lower = tl.load(lower_mask_ptr + row * head_dim + col, mask=dd_mask, other=0) != 0

        # Tiled path. Mirrors the forward kernel's two-pass structure:
        #   Pass 1 rebuilds u_block AND grad_u_block together (both are
        #   linear in L, so both accumulate the same way forward's u_block
        #   does -- one r_tile's L_offdiag contribution at a time, applied
        #   to x_block for u and to grad_out_block for grad_u, reusing the
        #   SAME per-tile L_offdiag_tile_s for both since it doesn't depend
        #   on which vector it's being applied to).
        #
        #   Once u_block/grad_u_block are complete, grad_L's DIAGONAL is
        #   just grad_out*u + x*grad_u element-wise (the outer product
        #   grad_L_s=outer(grad_out_s,u_s)+outer(x_s,grad_u_s)'s own
        #   diagonal, grad_L_s[i,i], reduces to this without needing the
        #   full outer product) -- computed once for the whole block, no
        #   r-tiling or per-token loop needed, unlike the fast path (which
        #   builds the full (D,D) outer product per token then extracts the
        #   diagonal -- fine at that granularity, wasteful to replicate
        #   here). The full grad_L_offdiag_s (D,D), however, genuinely is
        #   needed per token in Pass 2 below, since it feeds into per-rank-component
        #   contractions against each tile's own U/V.
        #
        #   Pass 2 uses the now-complete grad_u_block to accumulate
        #   grad_x's L-application term across tiles (same linear-in-L
        #   accumulation as forward's out_block), and, per r_tile
        #   independently (no cross-tile accumulation needed -- each tile
        #   owns disjoint output columns), rebuilds grad_L_offdiag_s from
        #   x_s/grad_out_s/u_s/grad_u_s and contracts it against that
        #   tile's U/V to get grad_gate/grad_U-contribution/
        #   grad_V-contribution/grad_raw_gate for that tile's columns.
        u_block = x_block * L_diag_block
        grad_u_block = grad_out_block * L_diag_block
        for r_tile in range(NUM_R_TILES):
            r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
            r_mask = r_off < rank
            gate_tile = tl.load(
                gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
                mask=s_mask[:, None] & r_mask[None, :], other=0.0,
            )
            U_tile = tl.load(
                u_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)
            V_tile = tl.load(
                v_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)

            for s in range(BLOCK_S):
                one_hot_s = tl.arange(0, BLOCK_S) == s
                x_s = tl.sum(tl.where(one_hot_s[:, None], x_block, 0.0), axis=0)
                grad_out_s = tl.sum(tl.where(one_hot_s[:, None], grad_out_block, 0.0), axis=0)
                gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_tile, 0.0), axis=0)

                gated_U = U_tile * gate_s[None, :]
                L_offdiag_tile_s = tl.dot(gated_U, tl.trans(V_tile), allow_tf32=False) * rank_scale
                L_offdiag_tile_s = tl.where(lower, L_offdiag_tile_s, 0.0)

                u_contrib_s = tl.sum(x_s[:, None] * L_offdiag_tile_s, axis=0)
                grad_u_contrib_s = tl.sum(grad_out_s[:, None] * L_offdiag_tile_s, axis=0)
                u_block += u_contrib_s[None, :] * one_hot_s[:, None]
                grad_u_block += grad_u_contrib_s[None, :] * one_hot_s[:, None]

        # --- diagonal branch: vectorized over the whole (BLOCK_S, BLOCK_D)
        # block at once -- no per-token loop, no r-tiling, identical closed
        # form to the fast path otherwise. ---
        grad_L_diag_block = grad_out_block * u_block + x_block * grad_u_block
        packed_diag_block = L_diag_block - 1.0
        packed_diag_unshifted_block = packed_diag_block + _DIAG_PACKED_BASELINE
        diag_grad_block = tl.exp(-packed_diag_unshifted_block) - tl.exp(1.0 - tl.exp(packed_diag_unshifted_block) - packed_diag_unshifted_block)
        grad_raw_diag_block = grad_L_diag_block * diag_grad_block

        grad_x_block += grad_raw_diag_block * weight_diag_h[None, :]
        grad_weight_diag_acc = tl.sum(grad_raw_diag_block * x_block, axis=0)  # reduce over S -> (BLOCK_D,)
        tl.atomic_add(
            grad_weight_diag_partial_ptr + group_id * heads * head_dim + head_id * head_dim + d_off,
            grad_weight_diag_acc, mask=d_mask,
        )

        # grad_x_L's own DIAGONAL contribution (sum_j grad_u_s[j]*L_s[i,j] at
        # j=i, i.e. grad_u_s[i]*L_diag_s[i]) -- NOT the same thing as the
        # grad_x_diag term just added above (that's the separate chain
        # through L_diag's own dependence on x via the packing function;
        # this one is x's DIRECT appearance in u=x@L, which needs L's FULL
        # structure, diag included, same as the forward kernel's tiled path
        # needing `out_block = u_block * L_diag_block` before its own
        # off-diagonal accumulation).
        grad_x_block += grad_u_block * L_diag_block

        for r_tile in range(NUM_R_TILES):
            r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
            r_mask = r_off < rank
            gate_tile = tl.load(
                gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
                mask=s_mask[:, None] & r_mask[None, :], other=0.0,
            )
            U_tile = tl.load(
                u_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)
            V_tile = tl.load(
                v_base + d_off[:, None] * rank + r_off[None, :],
                mask=d_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)

            grad_raw_gate_tile_block = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
            grad_U_acc_tile = tl.zeros((BLOCK_D, BLOCK_R), dtype=tl.float32)
            grad_V_acc_tile = tl.zeros((BLOCK_D, BLOCK_R), dtype=tl.float32)

            for s in range(BLOCK_S):
                one_hot_s = tl.arange(0, BLOCK_S) == s
                x_s = tl.sum(tl.where(one_hot_s[:, None], x_block, 0.0), axis=0)
                grad_out_s = tl.sum(tl.where(one_hot_s[:, None], grad_out_block, 0.0), axis=0)
                u_s = tl.sum(tl.where(one_hot_s[:, None], u_block, 0.0), axis=0)
                grad_u_s = tl.sum(tl.where(one_hot_s[:, None], grad_u_block, 0.0), axis=0)
                gate_s = tl.sum(tl.where(one_hot_s[:, None], gate_tile, 0.0), axis=0)

                grad_L_offdiag_s = grad_out_s[:, None] * u_s[None, :] + x_s[:, None] * grad_u_s[None, :]
                grad_L_offdiag_s = tl.where(lower, grad_L_offdiag_s, 0.0)

                # grad_x's L-application-path term, this tile's contribution
                # (accumulates across tiles, same linear-in-L reasoning as
                # forward's out_block).
                gated_U = U_tile * gate_s[None, :]
                L_offdiag_tile_s = tl.dot(gated_U, tl.trans(V_tile), allow_tf32=False) * rank_scale
                L_offdiag_tile_s = tl.where(lower, L_offdiag_tile_s, 0.0)
                grad_x_L_contrib_s = tl.sum(grad_u_s[None, :] * L_offdiag_tile_s, axis=1)
                grad_x_block += grad_x_L_contrib_s[None, :] * one_hot_s[:, None]

                # This tile's independent (no cross-tile accumulation needed)
                # per-rank-component outputs.
                tmp_V_tile_s = tl.dot(grad_L_offdiag_s, V_tile, allow_tf32=False)
                tmp_U_tile_s = tl.dot(tl.trans(grad_L_offdiag_s), U_tile, allow_tf32=False)
                grad_gate_tile_s = tl.sum(U_tile * tmp_V_tile_s, axis=0) * rank_scale

                grad_U_acc_tile += gate_s[None, :] * tmp_V_tile_s * rank_scale
                grad_V_acc_tile += gate_s[None, :] * tmp_U_tile_s * rank_scale

                grad_raw_gate_tile_s = grad_gate_tile_s * (2.0 / (tl.exp(gate_s) + tl.exp(-gate_s))) / scale
                grad_raw_gate_tile_block += grad_raw_gate_tile_s[None, :] * one_hot_s[:, None]

            tl.atomic_add(
                grad_U_partial_ptr + (group_id * heads + head_id) * head_dim * rank + d_off[:, None] * rank + r_off[None, :],
                grad_U_acc_tile, mask=d_mask[:, None] & r_mask[None, :],
            )
            tl.atomic_add(
                grad_V_partial_ptr + (group_id * heads + head_id) * head_dim * rank + d_off[:, None] * rank + r_off[None, :],
                grad_V_acc_tile, mask=d_mask[:, None] & r_mask[None, :],
            )
            tl.store(
                grad_raw_gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
                grad_raw_gate_tile_block.to(out_dtype), mask=s_mask[:, None] & r_mask[None, :],
            )

    tl.store(
        grad_x_ptr + token_off[:, None] * head_dim + d_off[None, :],
        grad_x_block.to(out_dtype), mask=s_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _riemannian_weight_w_and_gradxw_kernel(
    x_ptr, weight_W_ptr, grad_raw_gate_scratch_ptr,
    grad_x_w_ptr, grad_weight_W_ptr,
    seq_len, heads, head_dim, rank,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_R: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # Batched matmul mirror of _riemannian_project_kernel, on the backward
    # side -- identical structure to the design this replaced's own
    # weight-and-gradxw kernel, rank standing in for n_params:
    #       grad_weight_W_h[d,k] = sum_s x[s,d]*grad_raw_gate[s,k]  (x_block^T @ grad_raw_gate_block)
    #       grad_x_w[s,d]        = sum_k grad_raw_gate[s,k]*weight_W_h[d,k]  (grad_raw_gate_block @ weight_W_h^T)
    #
    # 3rd grid axis tiles the rank dimension, same as _riemannian_project_kernel
    # -- but unlike that kernel, grad_x_w's k-sum is a genuine REDUCTION over
    # rank components (every rank component contributes to every grad_x_w[s,d]), not an
    # embarrassingly-parallel per-rank-component output. So each r_tile program
    # contributes a PARTIAL sum over just its own BLOCK_R rank components, and these
    # partial sums are combined across r_tile programs via atomic_add into a
    # fp32 accumulator (grad_x_w_ptr, zero-initialized by the caller) --
    # same "accumulate in fp32 via atomic_add, cast down once at the end"
    # convention already used for grad_weight_diag/grad_U/grad_V elsewhere
    # in this file, now applied here too since this is no longer a single-
    # shot store. grad_weight_W_h's per-rank-component columns remain embarrassingly
    # parallel across r_tile (each tile only ever touches its own output
    # columns), so that part is unchanged other than the r_tile offset.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    r_tile = tl.program_id(2)
    head_id = nh % heads
    group_id = s_tile % NUM_GROUPS

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    r_off = r_tile * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_off < rank

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D)

    token_off = nh * seq_len + s_off
    grad_raw_gate_tile = tl.load(
        grad_raw_gate_scratch_ptr + token_off[:, None] * rank + r_off[None, :],
        mask=s_mask[:, None] & r_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_R) -- this r_tile's column-slice

    w_base = weight_W_ptr + head_id * head_dim * rank
    w_tile = tl.load(
        w_base + d_off[:, None] * rank + r_off[None, :],
        mask=d_mask[:, None] & r_mask[None, :], other=0.0,
    )  # (BLOCK_D, BLOCK_R) -- this r_tile's column-slice

    grad_x_w_partial = tl.dot(grad_raw_gate_tile, tl.trans(w_tile), allow_tf32=False)  # (BLOCK_S, BLOCK_D), fp32
    tl.atomic_add(
        grad_x_w_ptr + token_off[:, None] * head_dim + d_off[None, :], grad_x_w_partial,
        mask=s_mask[:, None] & d_mask[None, :],
    )

    gw_tile = tl.dot(tl.trans(x_block), grad_raw_gate_tile, allow_tf32=False)  # (BLOCK_D, BLOCK_R), fp32
    gw_base = grad_weight_W_ptr + (group_id * heads + head_id) * head_dim * rank
    tl.atomic_add(
        gw_base + d_off[:, None] * rank + r_off[None, :], gw_tile,
        mask=d_mask[:, None] & r_mask[None, :],
    )


# head_dim/device are the only things this depends on, and both are static
# for the life of a training run (head_dim never changes call-to-call; device
# is whichever GPU the module lives on) -- yet before this cache existed,
# forward() rebuilt it from scratch (torch.arange x2 + compare + cast, i.e.
# a handful of small kernel launches plus an allocation) on EVERY call, even
# though the *module* already carries an identical mask as
# RiemannianMetric.lower_mask (model.py) that the fused path just never
# reused. Caching here avoids threading an extra tensor through
# autograd.Function.apply (which would need its own None-gradient plumbing
# in backward) while still only paying the arange+compare cost once per
# (head_dim, device) pair instead of once per step.
_LOWER_MASK_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _structural_buffers(head_dim, device):
    key = (head_dim, device)
    lower_mask = _LOWER_MASK_CACHE.get(key)
    if lower_mask is None:
        row = torch.arange(head_dim, device=device).view(head_dim, 1)
        col = torch.arange(head_dim, device=device).view(1, head_dim)
        lower_mask = (row > col).to(torch.int32)
        _LOWER_MASK_CACHE[key] = lower_mask
    return lower_mask


class RiemannianMetricKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_diag, weight_W, weight_U, weight_V, scale):
        N, heads, S, head_dim = x.shape
        rank = weight_W.shape[-1]
        n_tokens = N * heads * S
        n_nh = N * heads

        x_c = x.contiguous()
        weight_diag_c = weight_diag.contiguous().to(x_c.dtype)
        weight_W_c = weight_W.contiguous().to(x_c.dtype)
        weight_U_c = weight_U.contiguous().to(x_c.dtype)
        weight_V_c = weight_V.contiguous().to(x_c.dtype)
        lower_mask = _structural_buffers(head_dim, x.device)

        # Same floor-at-16 reasoning as the design this replaced: tl.dot needs
        # every operand dimension >= 16. See that design's own forward() for
        # the fuller shared-memory-budget derivation this borrows the general
        # shape of (BLOCK_S capped at 32, BLOCK_* scaled down at fp32) --
        # NOT independently re-validated for this kernel's own tile shapes,
        # see module docstring.
        dtype_shrink = max(1, x.element_size() // 2)
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        BLOCK_S = max(16, min(32, triton.next_power_of_2(S)))
        BLOCK_R = max(16, min(128 // dtype_shrink, triton.next_power_of_2(rank)))
        # rank is no longer required to fit in one BLOCK_R tile -- see
        # module docstring's NOTE on tiling. NUM_R_TILES==1 whenever it
        # already did fit (the common case), in which case every kernel
        # below takes its untiled fast path, unchanged from before.
        NUM_R_TILES = triton.cdiv(rank, BLOCK_R)

        rank_scale = rank ** -0.5

        gate_scratch = torch.empty(n_tokens, rank, dtype=torch.float32, device=x.device)
        L_diag_scratch = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S), NUM_R_TILES)
        _riemannian_project_kernel[grid_proj](
            x_c.view(n_tokens, head_dim), weight_W_c, weight_diag_c, gate_scratch, L_diag_scratch,
            S, heads, head_dim, rank, scale,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_R=BLOCK_R,
            num_warps=8,
        )

        # _riemannian_apply_fwd_kernel tiles the rank dimension INTERNALLY
        # (via NUM_R_TILES as a kernel argument, not a grid axis) since it
        # needs a real sequential reduction over the rank dimension per token -- see its
        # own comment. Its grid therefore stays 2D, unlike the two kernels
        # above/below whose per-rank-component work is embarrassingly parallel.
        grid_apply = (n_nh, triton.cdiv(S, BLOCK_S))
        out = torch.empty_like(x_c)
        _riemannian_apply_fwd_kernel[grid_apply](
            x_c.view(n_tokens, head_dim), out.view(n_tokens, head_dim),
            gate_scratch, L_diag_scratch, weight_U_c, weight_V_c, lower_mask,
            S, heads, head_dim, rank, rank_scale,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_R=BLOCK_R, NUM_R_TILES=NUM_R_TILES,
            num_warps=8,
        )

        # gate_scratch/L_diag_scratch are NOT saved here -- measured on real
        # hardware (kernels/verify_riemannian_metric.py section 4), keeping
        # them alive through backward cost ~2x peak memory (62.92x -> 33.10x
        # reduction vs. eager) for a 1.05x -> 1.04x "speedup" (noise, not a
        # real gain): the project-kernel relaunch below is a single genuinely-
        # batched GEMM, not the bottleneck, so avoiding it bought nothing.
        # autograd keeps every ctx.save_for_backward tensor alive across ALL
        # decoder layers until each one's own backward runs (not just its own
        # layer's duration), so the save-vs-recompute tradeoff here is much
        # more expensive than it looks from a single layer's local scratch
        # size -- recomputing is the right call. See backward()'s comment.
        ctx.save_for_backward(x_c, weight_diag_c, weight_W_c, weight_U_c, weight_V_c, lower_mask)
        ctx.scale = scale
        ctx.rank_scale = rank_scale
        ctx.shapes = (N, heads, S, head_dim, rank, n_tokens, n_nh)
        ctx.blocks = (BLOCK_D, BLOCK_S, BLOCK_R, NUM_R_TILES)
        ctx.orig_dtypes = (weight_diag.dtype, weight_W.dtype, weight_U.dtype, weight_V.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_diag, weight_W, weight_U, weight_V, lower_mask = ctx.saved_tensors
        N, heads, S, head_dim, rank, n_tokens, n_nh = ctx.shapes
        BLOCK_D, BLOCK_S, BLOCK_R, NUM_R_TILES = ctx.blocks
        scale = ctx.scale
        rank_scale = ctx.rank_scale
        grad_out_c = grad_out.contiguous()

        # Recompute gate/L_diag rather than save them from forward -- see
        # forward()'s save_for_backward comment for the measured reason
        # (tried saving them; it roughly doubled peak memory for no measured
        # speedup, since this launch is a single genuinely-batched GEMM and
        # was never the bottleneck).
        gate_scratch = torch.empty(n_tokens, rank, dtype=torch.float32, device=x.device)
        L_diag_scratch = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=x.device)
        grid_proj = (n_nh, triton.cdiv(S, BLOCK_S), NUM_R_TILES)
        _riemannian_project_kernel[grid_proj](
            x.view(n_tokens, head_dim), weight_W, weight_diag, gate_scratch, L_diag_scratch,
            S, heads, head_dim, rank, scale,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_R=BLOCK_R,
            num_warps=8,
        )

        num_blocks = n_nh * triton.cdiv(S, BLOCK_S)
        NUM_GROUPS = min(32, num_blocks)

        # _riemannian_grad_per_token_kernel, like _riemannian_apply_fwd_kernel,
        # tiles the rank dimension internally -- 2D grid, NUM_R_TILES passed as an argument.
        grid_apply = (n_nh, triton.cdiv(S, BLOCK_S))
        grad_x = torch.empty_like(x)
        grad_raw_gate_scratch = torch.empty(n_tokens, rank, dtype=x.dtype, device=x.device)
        grad_weight_diag_partial = torch.zeros(NUM_GROUPS, heads, head_dim, dtype=torch.float32, device=x.device)
        grad_U_partial = torch.zeros(NUM_GROUPS, heads, head_dim, rank, dtype=torch.float32, device=x.device)
        grad_V_partial = torch.zeros(NUM_GROUPS, heads, head_dim, rank, dtype=torch.float32, device=x.device)

        _riemannian_grad_per_token_kernel[grid_apply](
            x.view(n_tokens, head_dim), grad_out_c.view(n_tokens, head_dim),
            grad_x.view(n_tokens, head_dim), grad_raw_gate_scratch,
            grad_weight_diag_partial, grad_U_partial, grad_V_partial,
            gate_scratch, L_diag_scratch, weight_U, weight_V, weight_diag,
            lower_mask,
            S, heads, head_dim, rank, scale, rank_scale,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_R=BLOCK_R, NUM_GROUPS=NUM_GROUPS, NUM_R_TILES=NUM_R_TILES,
            num_warps=8,
        )

        # grad_x_w is now accumulated across r_tile programs via atomic_add
        # (see _riemannian_weight_w_and_gradxw_kernel's comment), so it needs
        # a zero-initialized fp32 buffer rather than a plain per-token store
        # -- same "accumulate in fp32, cast down once" convention as
        # grad_weight_diag/grad_U/grad_V.
        grad_weight_W_partial = torch.zeros(NUM_GROUPS, heads, head_dim, rank, dtype=torch.float32, device=x.device)
        # 4D (matching x/grad_x's own shape), NOT a flat (n_tokens, head_dim)
        # buffer -- grad_x_total = grad_x + grad_x_w_f32 below needs both
        # operands at the same rank; only the kernel launch itself needs the
        # flattened .view(n_tokens, head_dim).
        grad_x_w_f32 = torch.zeros_like(x, dtype=torch.float32)
        grid_gw = (n_nh, triton.cdiv(S, BLOCK_S), NUM_R_TILES)
        _riemannian_weight_w_and_gradxw_kernel[grid_gw](
            x.view(n_tokens, head_dim), weight_W, grad_raw_gate_scratch,
            grad_x_w_f32.view(n_tokens, head_dim), grad_weight_W_partial,
            S, heads, head_dim, rank,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_R=BLOCK_R, NUM_GROUPS=NUM_GROUPS,
            num_warps=8,
        )

        grad_x_total = grad_x + grad_x_w_f32.to(x.dtype)
        grad_weight_diag_f32 = grad_weight_diag_partial.sum(dim=0)
        grad_weight_W_f32 = grad_weight_W_partial.sum(dim=0)
        grad_weight_U_f32 = grad_U_partial.sum(dim=0)
        grad_weight_V_f32 = grad_V_partial.sum(dim=0)

        orig_diag_dtype, orig_W_dtype, orig_U_dtype, orig_V_dtype = ctx.orig_dtypes
        return (
            grad_x_total,
            grad_weight_diag_f32.to(orig_diag_dtype),
            grad_weight_W_f32.to(orig_W_dtype),
            grad_weight_U_f32.to(orig_U_dtype),
            grad_weight_V_f32.to(orig_V_dtype),
            None,
        )
