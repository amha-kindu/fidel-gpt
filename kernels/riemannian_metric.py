"""
Triton forward+backward kernels for RiemannianMetric.

MATH
    forward (per head h, per token -- x_h is the key/query itself, self-referential):

        DIAGONAL (head_dim values -- guarantees non-singularity):
            raw_diag[i]  = x_h[i] * weight_diag_h[i]              (element-wise)
            L_diag[i]    = log1p(softplus(raw_diag[i])) + diag_offset
                        diag_offset = 1 - log1p(log(2)), so L_diag > diag_offset
                        always (>= ~0.4734)

        OFF-DIAGONAL:
            raw_gate[m]  = x_h . weight_W_h[:, m]      (one scalar per mode m)
            gate[m]      = silu(raw_gate[m])
                         = raw_gate[m] * sigmoid(raw_gate[m])

            B_hm[i,j]    = sum_r weight_U[h,m,i,r] * weight_V[h,m,j,r]
                        restricted to strictly-lower-triangular (i,j)
            S[i,j]       = sum_m gate[m] * B_hm[i,j]     (gate-weighted mode sum)
            L_offdiag[i,j] = asinh(S[i,j])

    FINAL:
        L   = L_offdiag with its diagonal overwritten by L_diag
        u   = x_h @ L       u[j] = sum_i x_h[i] * L[i,j]
        out = u @ L^T       out[k] = sum_j u[j] * L[k,j]

    backward (per head h, per token; grad_out given):
        grad_u        = grad_out @ L                    (same form as u = x@L)
        grad_x_from_u = grad_u @ L^T                     (same form as out = u@L^T)
        grad_L[p,q]   = grad_out[p]*u[q] + x[p]*grad_u[q]
        grad_L_diag[i]   = grad_L[i,i]
        grad_L_offdiag   = grad_L off the diagonal (upper triangle discarded

        DIAGONAL:
            grad_raw_diag[i] = grad_L_diag[i] * sigmoid(raw_diag[i]) / (1 + softplus(raw_diag[i]))
            grad_x  += grad_raw_diag[i] * weight_diag_h[i]
            grad_weight_diag_h[i] += grad_raw_diag[i] * x_h[i]

        OFF-DIAGONAL:
            grad_S[i,j]   = grad_L_offdiag[i,j] / sqrt(1 + S[i,j]^2)    (d(asinh)/dS)
            grad_gate[m]  = sum_ij grad_S[i,j] * B_hm[i,j]
            grad_B_hm[i,j] += gate[m] * grad_S[i,j]
            grad_raw_gate[m] = grad_gate[m] * silu'(raw_gate[m])
                              silu'(z) = sigmoid(z) * (1 + z*(1 - sigmoid(z)))
            grad_x  += sum_m grad_raw_gate[m] * weight_W_h[i,m]
            grad_weight_W_h[i,m] += grad_raw_gate[m] * x_h[i]

        Once grad_B_hm has been summed over every token sharing a head (see
        "grad_weight_diag and grad_B" below), converting it to grad_weight_U/
        grad_weight_V is plain matrix calculus, done OUTSIDE the per-token
        kernels entirely (see RiemannianMetricKernel.backward()'s own comment
        for why):
            grad_B_hm[i,j], masked to strictly-lower-triangular (i,j) first --
                B_hm is exactly zero outside that region, so its raw
                (unmasked) dot-product value never reached the loss; skipping
                this mask would leak a numerically real but semantically
                meaningless gradient into weight_U/weight_V.
            grad_weight_U_h[m,i,r] = sum_j grad_B_hm[i,j] * weight_V_h[m,j,r]
            grad_weight_V_h[m,j,r] = sum_i grad_B_hm[i,j] * weight_U_h[m,i,r]

DESIGN:
    The old (pre-modes, pre-SwiGLU, pre-off-diagonal-asinh) version of this
    kernel went to considerable lengths (see git history) to avoid ever
    materializing the (D,D) L_offdiag matrix, via a cumsum-based
    reformulation that only works because L was built by a LINEAR
    combination of rank-1 terms, immediately consumed by two more linear
    operations (x@L, u@L^T) -- associativity let the whole chain collapse
    to O(D*rank) without ever forming a (D,D) intermediate.

    That trick is GONE in this design, structurally, not by omission: asinh
    is applied to S AFTER the gate-weighted sum over modes, i.e. to the
    fully-combined off-diagonal matrix. asinh(sum_m ...) != sum_m asinh(...),
    so the nonlinearity cannot be pushed through the sum, and L_offdiag
    (and therefore L, u, out, and their backward counterparts) must be
    materialized as genuine (D,D) per-token tiles. This is NOT a missed
    optimization -- it's the direct consequence of the math this kernel now
    implements (see the conversation that led to this design).

    ROW-CHUNKED, tl.dot-BASED per-token application (kernels #2/#3):
        An earlier version of kernels #2/#3 built each token's (D,D) L tile
        via a Python `for s in range(BLOCK_S)` loop, fully unrolled by
        Triton at compile time -- meaning the token axis was NEVER
        parallelized within a program (only BLOCK_D was), and the mode
        contraction (S = sum_m gate[m]*B_hm) was done per-token via
        elementwise broadcast-multiply-then-reduce on CUDA cores, never via
        tl.dot/tensor cores. Profiling (kernels/profile_riemannian_speed.py)
        showed this made the fused path SLOWER than the eager PyTorch
        reference (0.61x) despite ~75x less peak memory, with
        _riemannian_grad_per_token_kernel alone eating ~88% of total CUDA
        time. A first fix (batching each program's grad_B/grad_weight_diag
        atomic_adds from BLOCK_S*MODES calls down to O(1) per program) got
        to 1.20x, but the underlying per-token loop -- and its total lack of
        tensor-core usage -- was still there.

        This version eliminates that loop entirely. The insight: the ONE
        genuinely GEMM-shaped operation in this whole computation is the
        mode contraction S[s,i,j] = sum_m gate[s,m]*B_hm[m,i,j] -- gate is
        shared per token, B_hm is shared per (head,mode), and modes is a
        real contraction axis. Reshaping B_hm's (i,j) axes into one flat
        axis turns this into an honest 2D GEMM: (BLOCK_S,MODES) @
        (MODES, BLOCK_I*BLOCK_D) -> (BLOCK_S, BLOCK_I*BLOCK_D), computed via
        tl.dot with a runtime-dtype-conditional USE_TF32 constexpr passed in
        by RiemannianMetricKernel.forward()/backward() (see those methods'
        own comments), NOT a hardcoded flag either way. History: an early
        version of this rewrite hardcoded allow_tf32=True everywhere
        (reasoning that these calls' operands are upcast to fp32 regardless
        of model dtype -- unlike this file's other tl.dot calls, which
        operate on native fp16/bf16 model dtype where the flag is a no-op --
        so TF32 tensor cores would give a real speedup here). Measured on
        real hardware, that made every dtype=torch.float32 check in
        verify_riemannian_metric.py fail (max_abs_diff up to ~1.3e-01,
        against a fp64-truth tolerance built for full fp32 precision) while
        every fp16/bf16 check kept passing (their own tolerances are already
        loose enough to absorb TF32's ~1e-3 relative error) -- a clean,
        dtype-exclusive failure pattern. The next version hardcoded
        allow_tf32=False everywhere instead -- correct (2.28x collapsed to
        1.65x, but ALL CHECKS PASSED, including dtype=torch.float32) -- but
        left real speed on the table for the realistic training case: models
        train in fp16/bf16 (or bf16-autocast with fp32 master weights, see
        section 2b's "mismatched dtype" test), where TF32's error is already
        smaller than what that dtype's own rounding contributes downstream,
        so paying full fp32 precision here bought nothing. USE_TF32 (True
        whenever x's dtype isn't torch.float32, False when it is) gets both:
        full precision exactly where the fp64-ground-truth-checked fp32 path
        needs it, tensor-core throughput everywhere training actually runs.

        The row axis `i` (the ROW of L, i.e. the token's own x/grad_out
        index) is walked in small chunks of BLOCK_I rows at a time -- NOT
        per-token, per-ROW-CHUNK, so the trip count is BLOCK_D/BLOCK_I
        (e.g. 4 at head_dim=64) instead of BLOCK_S (e.g. 32), and crucially
        every chunk processes ALL BLOCK_S tokens in the program AT ONCE
        (batched/vectorized), not one token at a time. This bounds the live
        working set per chunk to (BLOCK_M,BLOCK_I,BLOCK_D) / (BLOCK_S,
        BLOCK_I,BLOCK_D) -- small (tens of KB) regardless of BLOCK_S -- while
        still avoiding ever materializing a full (BLOCK_S,BLOCK_D,BLOCK_D)
        tensor, which would blow the shared-memory/register budget (~512KB+
        at this file's own benchmark scale) the same way an early version of
        this kernel's B_hm caching once did (see kernel #2's own comment).
        This is the "GEMM with accumulation" pattern: like flash-attention's
        online softmax accumulates over KV tiles, u=x@L and grad_u=grad_out@L
        are accumulated over row-chunks of L rather than computed from one
        fully-materialized L.

        u[s,j] = sum_i x[s,i]*L[s,i,j] is NOT itself GEMM-shaped (L differs
        per token, so this is a batched/diagonal contraction, not a
        shared-weight matmul) -- it's computed per row-chunk as an
        elementwise broadcast-multiply-then-reduce over the small BLOCK_I
        axis (`tl.sum(x_chunk[:,:,None]*L_chunk, axis=1)`), accumulated (+=)
        across chunks. out[s,k] = sum_j u[s,j]*L[s,k,j] has no such
        accumulation need across its own (k) chunks -- each k-chunk's output
        columns are independent of every other k-chunk's, so it's computed
        and stored directly per chunk, no running accumulator required.
        Backward mirrors this exactly (see kernel #3's own comment) and
        additionally gets grad_gate and grad_B for free as two MORE tl.dot
        GEMMs per chunk (contracting over the row-chunk's flattened
        (BLOCK_I,BLOCK_D) axis and the token axis respectively), replacing
        what used to be per-token elementwise reductions.

        B_hm chunks (and S/L chunks) end up recomputed twice per kernel
        invocation (once in each of the two row-chunk passes) rather than
        cached -- consistent with this file's established "recompute rather
        than save" preference (see kernel #3's own comment on why raw_gate/
        L_diag are recomputed, not saved, across the forward/backward
        boundary) and cheap now that the recomputation itself is a tensor-
        core GEMM rather than a scalar loop.

    Four kernels:
        1. _riemannian_project_kernel: BATCHED across tokens (one real
           tl.dot GEMM per (head, seq-tile) block: x_block @ weight_W_h)
           computes raw_gate for every token (saved UNACTIVATED -- see MATH's
           note on why silu needs raw_gate, not gate, in backward), plus the
           (cheap, element-wise) diagonal L_diag. Both stored to scratch.
           Directly analogous to the design this replaced's own projection
           kernel, with modes standing in for rank and no asinh here (gate's
           own activation now happens in kernel 2/3, right where it's used,
           since only ONE of those kernels needs the unactivated value too).
           Unchanged by the row-chunked rewrite above -- this kernel never had
           a per-token loop to begin with.

        2. _riemannian_apply_fwd_kernel: batched over (head, seq-tile)
           programs, entirely loop-over-ROW-CHUNKS now (see DESIGN above),
           not loop-over-tokens. Two passes, both walking `range(0, BLOCK_D,
           BLOCK_I)`:
               Pass 1 (i-chunks): computes u = x@L, accumulating across
                   chunks (u depends on ALL rows of L).
               Pass 2 (k-chunks): computes out = u@L^T and stores it
                   directly per chunk (out's k-th chunk is independent of
                   every other chunk, no accumulation needed).
           Each chunk reloads its own (MODES,BLOCK_I,BLOCK_D) slice of B_hm
           (small -- tens of KB, not the (MODES,BLOCK_D,BLOCK_D) full tensor
           an earlier version of this kernel held live for its whole
           program) and derives that chunk's S/L via one tl.dot GEMM (see
           DESIGN above) instead of a per-token reduction.

           B itself -- see RiemannianMetricKernel._build_B()'s own comment --
           is built by ONE plain torch.matmul call inside
           RiemannianMetricKernel.forward()/backward(), NOT inside this
           kernel and NOT by model.py. An earlier version of this file DID
           build B_hm inside this kernel (from weight_U_h/weight_V_h loaded
           per program, via a per-mode tl.dot loop) -- correct, but B_hm
           depends only on (head,mode), never on token or seq-tile, so
           recomputing it once per (head,seq-tile) PROGRAM was S/BLOCK_S
           -times redundant work for no benefit; reverted after it measured
           as an 11x wall-clock regression (see _build_B's comment for the
           numbers). This kernel doesn't know or care whether B was built by
           Triton or plain torch -- it only ever sees the finished (heads,
           modes,D,D) tensor, exactly as it did before weight_U/weight_V
           existed as this class's own inputs.

        3. _riemannian_grad_per_token_kernel: same row-chunked structure as
           #2, for the same reason. Recomputes S/L/u from scratch (gate,
           L_diag) rather than saving them from forward -- same "recompute
           rather than save" call the design this replaced made after
           measuring the memory-vs-recompute tradeoff on real hardware (see
           that design's own backward() comment; not re-measured here, but
           there's no structural reason this tradeoff would flip). Two
           passes:
               Pass 1 (i-chunks): computes u AND grad_u together (same index
                   structure -- u=x@L, grad_u=grad_out@L -- so they share the
                   same L_chunk per chunk, halving the recomputation this
                   pass would otherwise need), each accumulated across chunks.
               Pass 2 (p-chunks): recomputes L a second time (this row range
                   is what pass 1 called `i`, renamed `p` here to match
                   MATH's grad_L[p,q] notation) to derive, per chunk:
                     - grad_x (diagonal branch's grad_x_diag + grad_u@L^T's
                       grad_x_L for this chunk's rows) -- stored directly,
                       no cross-chunk accumulation needed (disjoint rows).
                     - grad_weight_diag_h's slice for this chunk's rows --
                       a genuine reduction over tokens (sum over s), done via
                       tl.sum then atomic_add-ed directly (chunks are
                       disjoint in the row axis, so unlike the OLD per-token
                       design, no register accumulator across chunks is
                       needed here to avoid redundant atomics -- there's
                       nothing redundant to avoid: every chunk's atomic_add
                       already targets a different address range).
                     - grad_gate's contribution from this chunk -- a REAL
                       tl.dot GEMM (grad_S_chunk_flat @ B_chunk_flat^T),
                       accumulated across chunks (grad_gate sums over the
                       FULL (i,j) domain, i.e. every chunk contributes).
                     - grad_B_hm's contribution from this chunk -- ANOTHER
                       real tl.dot GEMM (gate_block^T @ grad_S_chunk_flat),
                       which already sums over every token in the program as
                       part of the matmul itself (the token axis IS the
                       contraction axis here) -- so a single atomic_add per
                       chunk (covering all modes at once via one 3D masked
                       atomic_add call, not a `for m in range(MODES)` loop)
                       is both correct and non-redundant, unlike the old
                       per-token design where every token's atomic_add hit
                       the exact same address.
           grad_B/grad_weight_U/grad_weight_V conversion still does NOT
           happen in here -- see kernel #2's own comment and
           RiemannianMetricKernel.backward()'s, for why that conversion runs
           exactly ONCE, outside any @triton.jit kernel, after this one
           finishes.

        4. _riemannian_weight_w_and_gradxw_kernel: BATCHED (mirrors #1):
           grad_weight_W (genuine reduction over tokens -> grouped
           atomic_add, same convention as grad_weight_diag/grad_B below) and
           grad_x's gate-branch contribution (grad_raw_gate @ weight_W_h^T --
           NOT a reduction across kernel programs in this design, since
           modes is never tiled across the grid here, so a direct per-token
           store suffices; this is the one place this kernel's lack of a
           rank/modes tiling axis makes it strictly simpler than the design
           it replaces, which needed atomics here specifically because ITS
           rank axis COULD span multiple grid tiles). Unchanged by the
           row-chunked rewrite above -- this kernel never had a per-token
           loop either.

    grad_weight_diag and grad_B are genuine reductions (summed over every
    token sharing a head), accumulated via grouped tl.atomic_add into small
    (NUM_GROUPS, ...) buffers -- same pattern and same reasoning (spreading
    atomic contention across up to 32 buckets) as the design this replaced
    used for its own grad_weight/grad_U/grad_V. Both are now derived, per
    row-chunk, from operations that already sum over every token in the
    program BEFORE the atomic_add fires (a plain tl.sum reduction for
    grad_weight_diag, a real tl.dot GEMM contracting the token axis for
    grad_B -- see kernel #3's own comment) -- so each row-chunk's atomic_add
    is a single, non-redundant, disjoint-address write, a structural
    improvement over an earlier version of this kernel where EVERY token
    (and, before that, every (token,mode) pair) issued its own atomic_add to
    the exact same address. grad_B is then converted to grad_weight_U/
    grad_weight_V by two small torch.matmul calls in
    RiemannianMetricKernel.backward() -- see that method's own comment, and
    _build_B's, for why this conversion (like B's own construction)
    deliberately happens OUTSIDE any @triton.jit kernel.

    SCOPE / LIMITATIONS (deliberately not handled, unlike the design this
    replaced's rank-tiling):
        - `modes` must fit in one BLOCK_M tile -- there is no multi-tile
          fallback the way the old kernel had for rank > BLOCK_R. BLOCK_M is
          padded to >= 16 (a real tl.dot/tensor-core constraint, same
          reasoning as kernel #1's own BLOCK_M) in ALL FOUR kernels now,
          since kernels #2/#3 tl.dot along the modes axis too as of the
          row-chunked rewrite above; this is sized for small mode counts (a
          handful, think <=8-16), not dozens.
        - head_dim must fit in one BLOCK_D tile (same assumption the design
          this replaced made -- head_dim is typically 32-256, well within a
          single tile). BLOCK_I (the row-chunk size used by kernels #2/#3)
          always evenly divides BLOCK_D by construction (both are powers of
          2 >= 16), so the row-chunk loop never needs a ragged final
          iteration of its own -- only the usual head_dim-vs-BLOCK_D masking
          (`i_mask`/`p_mask`/`k_mask` below) handles head_dim not itself
          being a power of 2.
        - rank never appears inside any @triton.jit kernel in this file: it
          only affects how B_hm is built (weight_U/weight_V's own rank axis),
          which happens via one plain torch.matmul call inside
          RiemannianMetricKernel.forward()/backward() (see _build_B), not
          inside a kernel and not in model.py. This is a deliberate
          performance choice, not an oversight -- see _build_B's own comment
          for the 11x regression measured when B_hm (and, in backward, its
          conversion to grad_weight_U/grad_weight_V) were instead recomputed
          from weight_U/weight_V inside the per-(head,seq-tile)/per-token
          Triton kernels, work that is wasted almost everywhere it ran since
          neither quantity depends on token or seq-tile position. model.py
          never builds B either way (fused or eager) -- for the fused path it
          calls RiemannianMetricKernel.apply(x, weight_diag, weight_W,
          weight_U, weight_V) directly; the only PyTorch autograd.Function
          this file exposes takes weight_U/weight_V as first-class inputs and
          returns grad_weight_U/grad_weight_V as first-class outputs.
"""

import math
import torch
import triton
import triton.language as tl

_DIAG_OFFSET = tl.constexpr(1.0 - math.log1p(math.log(2.0)))


@triton.jit
def _softplus(x):
    # tl.exp/tl.log require fp32/fp64 in this Triton build (a real
    # hardware/libdevice constraint, not a numerical choice, per the design
    # this replaced), so upcast for the computation and cast back after.
    x32 = x.to(tl.float32)
    result = tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x32)))
    return result.to(x.dtype)


@triton.jit
def _log_softplus(x):
    # log(1 + softplus(x)) -- see model.py's RiemannianMetric docstring.
    x32 = x.to(tl.float32)
    result = tl.log(1.0 + _softplus(x32))
    return result.to(x.dtype)


@triton.jit
def _sigmoid(x):
    x32 = x.to(tl.float32)
    result = 1.0 / (1.0 + tl.exp(-x32))
    return result.to(x.dtype)


@triton.jit
def _silu(x):
    x32 = x.to(tl.float32)
    result = x32 * _sigmoid(x32)
    return result.to(x.dtype)


@triton.jit
def _silu_grad(x):
    # d(silu)/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
    x32 = x.to(tl.float32)
    sig = _sigmoid(x32)
    result = sig * (1.0 + x32 * (1.0 - sig))
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
    x_ptr, weight_W_ptr, weight_diag_ptr, raw_gate_scratch_ptr, L_diag_scratch_ptr,
    seq_len, heads, head_dim, modes,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    nh = tl.program_id(0)          # flat (batch, head) index: n * heads + h
    s_tile = tl.program_id(1)
    head_id = nh % heads

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    m_off = tl.arange(0, BLOCK_M)
    m_mask = m_off < modes

    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_D)

    token_off = nh * seq_len + s_off  # token_id for each row in this block

    # --- gate projection: genuine batched GEMM (x_block @ weight_W_h), the
    # dominant FLOPs of this kernel, same tensor-core-via-tl.dot treatment
    # the design this replaced gave its own analogous projection step.
    # Saved UNACTIVATED (raw_gate, not silu(raw_gate)) -- see module docstring. ---
    w_base = weight_W_ptr + head_id * head_dim * modes
    w_tile = tl.load(
        w_base + d_off[:, None] * modes + m_off[None, :],
        mask=d_mask[:, None] & m_mask[None, :], other=0.0,
    )  # (BLOCK_D, BLOCK_M)
    raw_gate_tile = tl.dot(x_block, w_tile, allow_tf32=False)  # (BLOCK_S, BLOCK_M), fp32 accumulate
    raw_gate_tile = tl.where(m_mask[None, :], raw_gate_tile, 0.0)
    tl.store(
        raw_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        raw_gate_tile,
        mask=s_mask[:, None] & m_mask[None, :],
    )

    # --- diagonal: element-wise, no matmul needed. ---
    diag_base = weight_diag_ptr + head_id * head_dim
    weight_diag_tile = tl.load(diag_base + d_off, mask=d_mask, other=0.0).to(tl.float32)  # (BLOCK_D,)
    raw_diag_tile = x_block.to(tl.float32) * weight_diag_tile[None, :]  # (BLOCK_S, BLOCK_D)
    L_diag_tile = _log_softplus(raw_diag_tile) + _DIAG_OFFSET
    L_diag_tile = tl.where(d_mask[None, :], L_diag_tile, 0.0)
    tl.store(
        L_diag_scratch_ptr + token_off[:, None] * head_dim + d_off[None, :],
        L_diag_tile,
        mask=s_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _riemannian_apply_fwd_kernel(
    x_ptr, out_ptr, raw_gate_scratch_ptr, L_diag_scratch_ptr, B_ptr,
    seq_len, heads, head_dim, modes,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    # Batched over a (head, seq-tile) program. NO per-token loop -- see module
    # docstring's DESIGN section for the row-chunked tl.dot rewrite this
    # replaced a `for s in range(BLOCK_S)` Python-unrolled loop with. Two
    # passes, both walking `range(0, BLOCK_D, BLOCK_I)` (row-chunks, not
    # tokens): pass 1 accumulates u=x@L across chunks (u depends on every row
    # of L); pass 2 computes out=u@L^T and stores each chunk directly (out's
    # k-th chunk doesn't depend on any other chunk, so no accumulator needed).
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    out_dtype = x_ptr.dtype.element_ty

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    m_off = tl.arange(0, BLOCK_M)
    m_mask = m_off < modes
    token_off = nh * seq_len + s_off

    # gate is shared by every row-chunk below (it doesn't vary with i/k), so
    # it's computed once here, padded to BLOCK_M>=16 -- a real tl.dot operand
    # dimension now (the mode contraction below is a genuine GEMM), unlike an
    # earlier version of this kernel where BLOCK_M only bounded a masked load
    # and could stay unpadded.
    raw_gate_block = tl.load(
        raw_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        mask=s_mask[:, None] & m_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_M), fp32
    gate_block = _silu(raw_gate_block)

    b_base = B_ptr + head_id * modes * head_dim * head_dim

    # --- pass 1: u = x @ L, accumulated over row-chunks of L's first index ---
    u_block = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for i0 in range(0, BLOCK_D, BLOCK_I):
        i_off = i0 + tl.arange(0, BLOCK_I)
        i_mask = i_off < head_dim

        # This chunk's B_hm slice, (MODES,BLOCK_I,BLOCK_D) -- small (tens of
        # KB), not the (MODES,BLOCK_D,BLOCK_D) full tensor an earlier version
        # of this kernel held live for the whole program.
        B_chunk = tl.load(
            b_base + m_off[:, None, None] * head_dim * head_dim
            + i_off[None, :, None] * head_dim + d_off[None, None, :],
            mask=m_mask[:, None, None] & i_mask[None, :, None] & d_mask[None, None, :], other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_I, BLOCK_D)
        B_chunk_flat = tl.reshape(B_chunk, (BLOCK_M, BLOCK_I * BLOCK_D))

        # S[s,i_local,j] = sum_m gate[s,m]*B[m,i_local,j] -- a genuine GEMM
        # (gate is shared across i_local/j, B is shared across s -- exactly
        # what tl.dot wants), not the per-token elementwise reduction an
        # earlier version of this kernel used. allow_tf32=True is deliberate
        # here -- see module docstring's DESIGN section.
        S_flat = tl.dot(gate_block, B_chunk_flat, allow_tf32=USE_TF32)  # (BLOCK_S, BLOCK_I*BLOCK_D), fp32
        S_chunk = tl.reshape(S_flat, (BLOCK_S, BLOCK_I, BLOCK_D))
        Loff_chunk = _asinh(S_chunk)  # diagonal of this is exactly 0 (see module docstring)

        L_diag_chunk = tl.load(
            L_diag_scratch_ptr + token_off[:, None] * head_dim + i_off[None, :],
            mask=s_mask[:, None] & i_mask[None, :], other=0.0,
        )  # (BLOCK_S, BLOCK_I), fp32
        eye_chunk = i_off[:, None] == d_off[None, :]  # (BLOCK_I, BLOCK_D)
        L_chunk = tl.where(eye_chunk[None, :, :], L_diag_chunk[:, :, None], Loff_chunk)  # (BLOCK_S,BLOCK_I,BLOCK_D)

        x_chunk = tl.load(
            x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + i_off[None, :],
            mask=s_mask[:, None] & i_mask[None, :], other=0.0,
        ).to(tl.float32)  # (BLOCK_S, BLOCK_I)

        # u[s,j] = sum_i x[s,i]*L[s,i,j] -- L differs per token, so this is a
        # batched/diagonal contraction, not GEMM-shaped; a small elementwise
        # broadcast-reduce over just this chunk's BLOCK_I rows, accumulated
        # across chunks (unlike out below, u genuinely needs every row of L).
        u_block += tl.sum(x_chunk[:, :, None] * L_chunk, axis=1)  # (BLOCK_S, BLOCK_D)

    # --- pass 2: out = u @ L^T, each row-chunk of L's first index (k) stored
    # directly -- no accumulation needed since out[:,k_chunk] doesn't depend
    # on any other k-chunk. ---
    for k0 in range(0, BLOCK_D, BLOCK_I):
        k_off = k0 + tl.arange(0, BLOCK_I)
        k_mask = k_off < head_dim

        B_chunk = tl.load(
            b_base + m_off[:, None, None] * head_dim * head_dim
            + k_off[None, :, None] * head_dim + d_off[None, None, :],
            mask=m_mask[:, None, None] & k_mask[None, :, None] & d_mask[None, None, :], other=0.0,
        ).to(tl.float32)
        B_chunk_flat = tl.reshape(B_chunk, (BLOCK_M, BLOCK_I * BLOCK_D))
        S_flat = tl.dot(gate_block, B_chunk_flat, allow_tf32=USE_TF32)
        S_chunk = tl.reshape(S_flat, (BLOCK_S, BLOCK_I, BLOCK_D))
        Loff_chunk = _asinh(S_chunk)

        L_diag_chunk = tl.load(
            L_diag_scratch_ptr + token_off[:, None] * head_dim + k_off[None, :],
            mask=s_mask[:, None] & k_mask[None, :], other=0.0,
        )
        eye_chunk = k_off[:, None] == d_off[None, :]
        L_chunk = tl.where(eye_chunk[None, :, :], L_diag_chunk[:, :, None], Loff_chunk)  # L[s,k_local,j]

        out_chunk = tl.sum(u_block[:, None, :] * L_chunk, axis=2)  # (BLOCK_S, BLOCK_I)
        tl.store(
            out_ptr + token_off[:, None] * head_dim + k_off[None, :],
            out_chunk.to(out_dtype),
            mask=s_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def _riemannian_grad_per_token_kernel(
    x_ptr, grad_out_ptr,
    grad_x_ptr, grad_gate_scratch_ptr,
    grad_weight_diag_partial_ptr, grad_B_partial_ptr,
    raw_gate_scratch_ptr, L_diag_scratch_ptr, B_ptr, weight_diag_ptr,
    seq_len, heads, head_dim, modes,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr,
    NUM_GROUPS: tl.constexpr, USE_TF32: tl.constexpr,
):
    # Same row-chunked structure as _riemannian_apply_fwd_kernel, for the
    # same reasons (see module docstring's DESIGN section) -- no per-token
    # loop. Recomputes S/L/u (and now grad_u) from gate/L_diag rather than
    # saving them from forward, same "recompute rather than save" call the
    # design this replaced made after measuring the memory-vs-recompute
    # tradeoff on real hardware.
    #
    # grad_gate is stored here UNACTIVATED-derivative-applied -- i.e. this
    # kernel stores grad_gate (d(loss)/d(gate)), and
    # _riemannian_weight_w_and_gradxw_kernel is the one that multiplies by
    # silu'(raw_gate) to get grad_raw_gate, since that kernel is where
    # raw_gate already needs to be reloaded anyway for its own GEMMs -- this
    # avoids loading raw_gate a third time in this kernel just for that one
    # multiply. grad_gate_scratch_ptr's name reflects that: it holds
    # grad_gate, not grad_raw_gate, despite the "raw_gate" naming elsewhere.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    group_id = s_tile % NUM_GROUPS
    out_dtype = x_ptr.dtype.element_ty

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    m_off = tl.arange(0, BLOCK_M)
    m_mask = m_off < modes
    token_off = nh * seq_len + s_off

    raw_gate_block = tl.load(
        raw_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        mask=s_mask[:, None] & m_mask[None, :], other=0.0,
    )  # (BLOCK_S, BLOCK_M), fp32
    gate_block = _silu(raw_gate_block)

    b_base = B_ptr + head_id * modes * head_dim * head_dim

    # --- pass 1: u=x@L and grad_u=grad_out@L together (identical index
    # structure -- both contract L's first axis against a (BLOCK_S,BLOCK_D)
    # operand -- so they share the same L_chunk per row-chunk instead of
    # recomputing it twice). Both accumulated across chunks. ---
    u_block = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    grad_u_block = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for i0 in range(0, BLOCK_D, BLOCK_I):
        i_off = i0 + tl.arange(0, BLOCK_I)
        i_mask = i_off < head_dim

        B_chunk = tl.load(
            b_base + m_off[:, None, None] * head_dim * head_dim
            + i_off[None, :, None] * head_dim + d_off[None, None, :],
            mask=m_mask[:, None, None] & i_mask[None, :, None] & d_mask[None, None, :], other=0.0,
        ).to(tl.float32)
        B_chunk_flat = tl.reshape(B_chunk, (BLOCK_M, BLOCK_I * BLOCK_D))
        S_flat = tl.dot(gate_block, B_chunk_flat, allow_tf32=USE_TF32)
        S_chunk = tl.reshape(S_flat, (BLOCK_S, BLOCK_I, BLOCK_D))
        Loff_chunk = _asinh(S_chunk)

        L_diag_chunk = tl.load(
            L_diag_scratch_ptr + token_off[:, None] * head_dim + i_off[None, :],
            mask=s_mask[:, None] & i_mask[None, :], other=0.0,
        )
        eye_chunk = i_off[:, None] == d_off[None, :]
        L_chunk = tl.where(eye_chunk[None, :, :], L_diag_chunk[:, :, None], Loff_chunk)

        x_chunk = tl.load(
            x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + i_off[None, :],
            mask=s_mask[:, None] & i_mask[None, :], other=0.0,
        ).to(tl.float32)
        grad_out_chunk = tl.load(
            grad_out_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + i_off[None, :],
            mask=s_mask[:, None] & i_mask[None, :], other=0.0,
        ).to(tl.float32)

        u_block += tl.sum(x_chunk[:, :, None] * L_chunk, axis=1)
        grad_u_block += tl.sum(grad_out_chunk[:, :, None] * L_chunk, axis=1)

    # --- pass 2: recompute L a second time per row-chunk (p, matching MATH's
    # grad_L[p,q] notation) to derive grad_x, grad_weight_diag, grad_gate,
    # grad_B for that chunk's rows. grad_x/grad_weight_diag are stored/
    # atomic_add-ed directly per chunk (disjoint rows -> no cross-chunk
    # accumulator needed); grad_gate is accumulated across chunks (it sums
    # over the FULL (i,j) domain); grad_B's token-axis reduction happens
    # INSIDE the tl.dot below, so its atomic_add is also direct-per-chunk. ---
    grad_gate_acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)
    for p0 in range(0, BLOCK_D, BLOCK_I):
        p_off = p0 + tl.arange(0, BLOCK_I)
        p_mask = p_off < head_dim

        B_chunk = tl.load(
            b_base + m_off[:, None, None] * head_dim * head_dim
            + p_off[None, :, None] * head_dim + d_off[None, None, :],
            mask=m_mask[:, None, None] & p_mask[None, :, None] & d_mask[None, None, :], other=0.0,
        ).to(tl.float32)
        B_chunk_flat = tl.reshape(B_chunk, (BLOCK_M, BLOCK_I * BLOCK_D))
        S_flat = tl.dot(gate_block, B_chunk_flat, allow_tf32=USE_TF32)
        S_chunk = tl.reshape(S_flat, (BLOCK_S, BLOCK_I, BLOCK_D))
        Loff_chunk = _asinh(S_chunk)

        L_diag_chunk = tl.load(
            L_diag_scratch_ptr + token_off[:, None] * head_dim + p_off[None, :],
            mask=s_mask[:, None] & p_mask[None, :], other=0.0,
        )
        eye_chunk = p_off[:, None] == d_off[None, :]  # (BLOCK_I, BLOCK_D)
        not_eye_chunk = eye_chunk == 0
        L_chunk = tl.where(eye_chunk[None, :, :], L_diag_chunk[:, :, None], Loff_chunk)

        x_chunk = tl.load(
            x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + p_off[None, :],
            mask=s_mask[:, None] & p_mask[None, :], other=0.0,
        ).to(tl.float32)
        grad_out_chunk = tl.load(
            grad_out_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + p_off[None, :],
            mask=s_mask[:, None] & p_mask[None, :], other=0.0,
        ).to(tl.float32)

        # grad_L[p,q] = grad_out[p]*u[q] + x[p]*grad_u[q], this chunk's rows only
        grad_L_chunk = (
            grad_out_chunk[:, :, None] * u_block[:, None, :]
            + x_chunk[:, :, None] * grad_u_block[:, None, :]
        )  # (BLOCK_S, BLOCK_I, BLOCK_D)

        # --- diagonal branch ---
        grad_L_diag_chunk = tl.sum(tl.where(eye_chunk[None, :, :], grad_L_chunk, 0.0), axis=2)  # (BLOCK_S,BLOCK_I)
        weight_diag_chunk = tl.load(
            weight_diag_ptr + head_id * head_dim + p_off, mask=p_mask, other=0.0,
        ).to(tl.float32)  # (BLOCK_I,)
        raw_diag_chunk = x_chunk * weight_diag_chunk[None, :]
        packed_chunk = _softplus(raw_diag_chunk)
        d_Ldiag_d_rawdiag_chunk = _sigmoid(raw_diag_chunk) / (1.0 + packed_chunk)
        grad_raw_diag_chunk = grad_L_diag_chunk * d_Ldiag_d_rawdiag_chunk

        grad_x_diag_chunk = grad_raw_diag_chunk * weight_diag_chunk[None, :]
        # grad_weight_diag_h[p] = sum_s grad_raw_diag[s,p]*x[s,p] -- a genuine
        # reduction over every token in this program, done via tl.sum BEFORE
        # the atomic_add fires (not accumulated token-by-token via repeated
        # atomics the way an earlier version of this kernel did it).
        grad_weight_diag_chunk = tl.sum(grad_raw_diag_chunk * x_chunk, axis=0)  # (BLOCK_I,)
        tl.atomic_add(
            grad_weight_diag_partial_ptr + group_id * heads * head_dim + head_id * head_dim + p_off,
            grad_weight_diag_chunk, mask=p_mask,
        )

        # --- off-diagonal branch ---
        grad_Loff_chunk = tl.where(not_eye_chunk[None, :, :], grad_L_chunk, 0.0)  # upper triangle discarded too (see module docstring)
        grad_S_chunk = grad_Loff_chunk / tl.sqrt(1.0 + S_chunk * S_chunk)  # asinh'(S) = 1/sqrt(1+S^2)
        grad_S_chunk_flat = tl.reshape(grad_S_chunk, (BLOCK_S, BLOCK_I * BLOCK_D))

        # grad_gate[s,m] += sum_{p_local,j} grad_S[s,p_local,j]*B[m,p_local,j]
        # -- a genuine GEMM (grad_S_chunk_flat @ B_chunk_flat^T) contracting
        # this chunk's flattened (p_local,j) axis, replacing the per-token
        # elementwise double-reduction an earlier version of this kernel used.
        grad_gate_acc += tl.dot(grad_S_chunk_flat, tl.trans(B_chunk_flat), allow_tf32=USE_TF32)  # (BLOCK_S, BLOCK_M)

        # grad_B[m,p_local,j] = sum_s gate[s,m]*grad_S[s,p_local,j] -- ALSO a
        # genuine GEMM (gate_block^T @ grad_S_chunk_flat), and this one
        # contracts the TOKEN axis itself, so the reduction over every token
        # sharing this program is done by tl.dot before the atomic_add below
        # ever fires -- one atomic_add per chunk (covering every mode at
        # once via a single 3D masked call, not a `for m in range(MODES)`
        # loop), each targeting a disjoint row-range of grad_B_partial, unlike
        # an earlier version of this kernel where every token's atomic_add
        # hit the exact same address.
        grad_B_chunk_flat = tl.dot(tl.trans(gate_block), grad_S_chunk_flat, allow_tf32=USE_TF32)  # (BLOCK_M,BLOCK_I*BLOCK_D)
        grad_B_chunk = tl.reshape(grad_B_chunk_flat, (BLOCK_M, BLOCK_I, BLOCK_D))
        tl.atomic_add(
            grad_B_partial_ptr + (group_id * heads + head_id) * modes * head_dim * head_dim
            + m_off[:, None, None] * head_dim * head_dim + p_off[None, :, None] * head_dim + d_off[None, None, :],
            grad_B_chunk,
            mask=m_mask[:, None, None] & p_mask[None, :, None] & d_mask[None, None, :],
        )

        # --- grad_x: L-application term (grad_u@L^T, this chunk's rows) + diagonal term ---
        grad_x_L_chunk = tl.sum(grad_u_block[:, None, :] * L_chunk, axis=2)  # (BLOCK_S, BLOCK_I)
        grad_x_chunk = grad_x_L_chunk + grad_x_diag_chunk
        tl.store(
            grad_x_ptr + token_off[:, None] * head_dim + p_off[None, :],
            grad_x_chunk.to(out_dtype), mask=s_mask[:, None] & p_mask[None, :],
        )

    tl.store(
        grad_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        grad_gate_acc.to(out_dtype), mask=s_mask[:, None] & m_mask[None, :],
    )


@triton.jit
def _riemannian_weight_w_and_gradxw_kernel(
    x_ptr, weight_W_ptr, raw_gate_scratch_ptr, grad_gate_scratch_ptr,
    grad_x_w_ptr, grad_weight_W_partial_ptr,
    seq_len, heads, head_dim, modes,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # Batched matmul mirror of _riemannian_project_kernel, on the backward
    # side -- identical structure to the design this replaced's own
    # weight-and-gradxw kernel, modes standing in for rank:
    #     grad_raw_gate[s,m]  = grad_gate[s,m] * silu'(raw_gate[s,m])   (elementwise, computed HERE)
    #     grad_weight_W_h[d,m] = sum_s x[s,d]*grad_raw_gate[s,m]  (x_block^T @ grad_raw_gate_block)
    #     grad_x_w[s,d]        = sum_m grad_raw_gate[s,m]*weight_W_h[d,m]  (grad_raw_gate_block @ weight_W_h^T)
    #
    # grad_weight_W is a genuine reduction over every token sharing a head
    # (every seq-tile program contributes a partial sum), so it's accumulated
    # via grouped atomic_add, same convention as grad_weight_diag/grad_B in
    # _riemannian_grad_per_token_kernel. grad_x_w is NOT a reduction across
    # kernel programs in this design (unlike the design this replaced, whose
    # rank axis could span multiple grid tiles) -- modes is never tiled
    # across the grid here, so each program computes its own token rows'
    # grad_x_w completely and can store it directly, no atomics needed.
    nh = tl.program_id(0)
    s_tile = tl.program_id(1)
    head_id = nh % heads
    group_id = s_tile % NUM_GROUPS

    s_off = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    m_off = tl.arange(0, BLOCK_M)
    m_mask = m_off < modes

    out_dtype = x_ptr.dtype.element_ty
    # Explicit .to(tl.float32) on every load feeding a tl.dot below -- without this,
    # grad_raw_gate_block ends up fp32 (raw_gate_scratch is always stored fp32, so
    # multiplying grad_gate_block by _silu_grad(raw_gate_block) implicitly promotes
    # the product to fp32) while x_block/w_tile stay in the model's native dtype
    # (fp16/bf16), and tl.dot asserts both operands share a dtype -- confirmed on
    # real hardware: "AssertionError: Both operands must be same dtype. Got fp32
    # and fp16". Every other kernel in this file already casts loads to fp32 for
    # this exact reason; this one had missed it.
    x_block = tl.load(
        x_ptr + nh * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :],
        mask=s_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)  # (BLOCK_S, BLOCK_D)

    token_off = nh * seq_len + s_off
    raw_gate_block = tl.load(
        raw_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        mask=s_mask[:, None] & m_mask[None, :], other=0.0,
    ).to(tl.float32)
    grad_gate_block = tl.load(
        grad_gate_scratch_ptr + token_off[:, None] * modes + m_off[None, :],
        mask=s_mask[:, None] & m_mask[None, :], other=0.0,
    ).to(tl.float32)
    grad_raw_gate_block = grad_gate_block * _silu_grad(raw_gate_block)  # (BLOCK_S, BLOCK_M), fp32

    w_base = weight_W_ptr + head_id * head_dim * modes
    w_tile = tl.load(
        w_base + d_off[:, None] * modes + m_off[None, :],
        mask=d_mask[:, None] & m_mask[None, :], other=0.0,
    ).to(tl.float32)  # (BLOCK_D, BLOCK_M)

    grad_x_w = tl.dot(grad_raw_gate_block, tl.trans(w_tile), allow_tf32=False)  # (BLOCK_S, BLOCK_D), fp32
    tl.store(
        grad_x_w_ptr + token_off[:, None] * head_dim + d_off[None, :], grad_x_w.to(out_dtype),
        mask=s_mask[:, None] & d_mask[None, :],
    )

    gw_tile = tl.dot(tl.trans(x_block), grad_raw_gate_block, allow_tf32=False)  # (BLOCK_D, BLOCK_M), fp32
    gw_base = grad_weight_W_partial_ptr + (group_id * heads + head_id) * head_dim * modes
    tl.atomic_add(
        gw_base + d_off[:, None] * modes + m_off[None, :], gw_tile,
        mask=d_mask[:, None] & m_mask[None, :],
    )


def _block_i(block_s, block_d, cap):
    # Row-chunk tile size for _riemannian_apply_fwd_kernel/_riemannian_grad_
    # per_token_kernel's tl.dot-based rewrite (see module docstring's DESIGN
    # section) -- bounds BLOCK_S*BLOCK_I*BLOCK_D at `cap`, a shared-memory
    # budget calibrated against a real triton.runtime.errors.OutOfResources
    # measurement, not guessed. Shared by RiemannianMetricKernel.forward()
    # (cap=4096, the exact proven-safe point for the forward kernel that was
    # actually measured) and .backward() (cap=2048, deliberately halved for
    # the backward kernel's own unmeasured, larger working set) -- see
    # forward()'s own comment for the full calibration story and why the two
    # kernels get different caps instead of sharing one.
    return max(1, min(block_d, cap // (block_s * block_d)))


class RiemannianMetricKernel(torch.autograd.Function):
    # Inputs: x (N,H,S,D), weight_diag (H,D), weight_W (H,D,M), weight_U (H,M,D,R),
    # weight_V (H,M,D,R). B_hm = mask(weight_U @ weight_V^T) is built by ONE plain
    # torch.matmul call inside forward()/backward() below -- NOT inside the
    # per-(head,seq-tile)/per-token Triton kernels (that was tried and reverted
    # for being catastrophically slow, see _build_B()'s own comment and the
    # module docstring's DESIGN section) and NOT in model.py (that would put B
    # back on the autograd-tracked eager path this class exists to avoid).
    # Because this matmul runs inside a torch.autograd.Function's forward()/
    # backward(), it executes with grad mode already disabled by PyTorch itself
    # -- B never becomes a graph node in the CALLER's autograd graph, and never
    # exists outside this file, even though it's built via an ordinary (cheap,
    # cuBLAS-backed) torch op rather than a hand-written Triton kernel.
    @staticmethod
    def _build_B(weight_U_c, weight_V_c, head_dim):
        # heads*modes tiny (D,R)@(R,D) matmuls, batched by torch.matmul's normal
        # broadcasting over the leading (heads,modes) dims -- O(heads*modes*D^2*R)
        # work, negligible next to the O(N*heads*S*D^2) the rest of this file
        # does. Deliberately NOT a Triton kernel: an earlier version of this
        # class instead recomputed B_hm (and, in backward, the grad_weight_U/
        # grad_weight_V conversion) from scratch INSIDE _riemannian_apply_fwd_
        # kernel/_riemannian_grad_per_token_kernel -- correct, but B_hm depends
        # only on (head,mode), never on token or seq-tile, so building it once
        # per (head,seq-tile) PROGRAM was S/BLOCK_S-times redundant, and doing
        # the grad_weight_U/grad_weight_V matmuls once per TOKEN inside the
        # per-token backward loop was BLOCK_S*MODES-times MORE redundant on top
        # of that. Measured on real hardware: an 11x wall-clock regression
        # (6-decoder-layer benchmark in verify_riemannian_metric.py went from
        # ~289ms/iter to ~3193ms/iter) for a memory profile that barely moved.
        # A single small matmul here, run once per forward/backward call, gives
        # back that entire regression.
        row = torch.arange(head_dim, device=weight_U_c.device).view(head_dim, 1)
        col = torch.arange(head_dim, device=weight_U_c.device).view(1, head_dim)
        lower_mask = row > col  # (D, D), strictly-lower-triangular -- same convention as model.py's own lower_mask
        B = torch.matmul(weight_U_c, weight_V_c.transpose(-2, -1))
        return torch.where(lower_mask, B, 0.0), lower_mask

    @staticmethod
    def forward(ctx, x, weight_diag, weight_W, weight_U, weight_V):
        N, heads, S, head_dim = x.shape
        modes = weight_W.shape[-1]
        n_tokens = N * heads * S
        n_nh = N * heads

        x_c = x.contiguous()
        weight_diag_c = weight_diag.contiguous().to(x_c.dtype)
        weight_W_c = weight_W.contiguous().to(x_c.dtype)
        weight_U_c = weight_U.contiguous().to(x_c.dtype)
        weight_V_c = weight_V.contiguous().to(x_c.dtype)
        B_c, _ = RiemannianMetricKernel._build_B(weight_U_c, weight_V_c, head_dim)

        # tl.dot needs every operand dimension >= 16 (tensor-core constraint).
        # BLOCK_M is now shared by all four kernels -- kernels #2/#3 tl.dot
        # along the modes axis too as of the row-chunked rewrite (see module
        # docstring's DESIGN section), so the single BLOCK_M_TOK/BLOCK_M_DOT
        # split an earlier version of this file needed (to avoid inflating a
        # full (MODES,BLOCK_D,BLOCK_D) B_all tile that #2/#3 no longer hold
        # live) is gone -- there's no such full tile left to inflate.
        #
        # BLOCK_I: the row-chunk size kernels #2/#3 walk BLOCK_D in, BLOCK_I
        # at a time, instead of looping per-token. Bounds each chunk's live
        # working set to (BLOCK_M,BLOCK_I,BLOCK_D)/(BLOCK_S,BLOCK_I,BLOCK_D).
        # A flat `min(16, BLOCK_D)` (this file's first attempt) measured, on
        # real hardware, at head_dim=64 (BLOCK_D=64, BLOCK_I=16, BLOCK_S=16),
        # in _riemannian_apply_fwd_kernel (the forward pass -- the simpler of
        # the two row-chunked kernels): `triton.runtime.errors.OutOfResources:
        # out of resource: shared memory, Required: 140288, Hardware limit:
        # 101376` -- while every head_dim=8/16 config (BLOCK_D=16, same
        # BLOCK_I=16 by construction) ran fine. That gives a real calibration
        # point: BLOCK_S*BLOCK_I*BLOCK_D = 16*16*16 = 4096 fits comfortably;
        # 16*16*64 = 16384 (4x that) doesn't.
        #
        # TWO caps, not one: BLOCK_I_FWD uses that proven-safe 4096 point
        # directly (this exact number, measured in this exact kernel, is what
        # fixed the OutOfResources above -- verified by a subsequent full run
        # of verify_riemannian_metric.py, ALL CHECKS PASSED). BLOCK_I_BWD
        # halves it to 2048: _riemannian_grad_per_token_kernel carries
        # noticeably more concurrently-live per-chunk tensors than the
        # forward kernel that was actually measured (grad_L_chunk,
        # grad_S_chunk, two more tl.dot calls each needing their own staging,
        # on top of everything forward already has), and its own shared-
        # memory footprint has never been measured at the boundary the way
        # forward's was -- the runs that got this far always had SOME margin
        # applied to backward already. Halving is a deliberate hedge for that
        # gap, not a re-derivation from its own measurement; if backward ever
        # overflows, THIS is the constant to shrink further (1024, then
        # 512, ...) -- forward's 4096 should stay as-is, it's proven. Both
        # always evenly divide BLOCK_D: BLOCK_S and BLOCK_D are both powers
        # of 2, so cap/(BLOCK_S*BLOCK_D) is a power of 2 too (or, once
        # BLOCK_S*BLOCK_D exceeds the cap at large head_dim/BLOCK_S, floored
        # to the minimum viable BLOCK_I=1 -- correctness holds either way,
        # just with more, smaller row-chunks).
        #
        # USE_TF32: whether the two row-chunked kernels' tl.dot calls (all of
        # which upcast their operands to fp32 regardless of model dtype, see
        # module docstring) are allowed to round those fp32 operands down to
        # TF32 for tensor-core throughput. True except when x itself is
        # fp32 -- that's the one path verify_riemannian_metric.py checks
        # against fp64 ground truth at full-fp32 tolerance, and TF32's
        # ~1e-3 relative error fails it outright (measured on real hardware,
        # see module docstring's DESIGN section); fp16/bf16's own tolerances
        # already absorb an error that size, so there's no correctness reason
        # to pay full fp32 precision's throughput cost on those paths too.
        BLOCK_D = max(16, triton.next_power_of_2(head_dim))
        BLOCK_S = max(16, min(32, triton.next_power_of_2(S)))
        BLOCK_M = max(16, triton.next_power_of_2(modes))
        BLOCK_I_FWD = _block_i(BLOCK_S, BLOCK_D, 4096)
        USE_TF32 = x_c.dtype != torch.float32

        raw_gate_scratch = torch.empty(n_tokens, modes, dtype=torch.float32, device=x.device)
        L_diag_scratch = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=x.device)
        grid = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid](
            x_c.view(n_tokens, head_dim), weight_W_c, weight_diag_c, raw_gate_scratch, L_diag_scratch,
            S, heads, head_dim, modes,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=8,
        )

        out = torch.empty_like(x_c)
        # num_stages=2 pinned explicitly (not left to Triton's own default) on
        # this kernel and _riemannian_grad_per_token_kernel below -- both now
        # contain loops around tl.dot (see module docstring's DESIGN section),
        # and software-pipelining stages multiply the shared-memory cost of
        # each loop-body buffer; pinning this removes one more source of
        # uncontrolled shared-memory growth on top of the BLOCK_I sizing above.
        _riemannian_apply_fwd_kernel[grid](
            x_c.view(n_tokens, head_dim), out.view(n_tokens, head_dim),
            raw_gate_scratch, L_diag_scratch, B_c,
            S, heads, head_dim, modes,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I_FWD, USE_TF32=USE_TF32,
            num_warps=8, num_stages=2,
        )

        # raw_gate_scratch/L_diag_scratch/B_c NOT saved -- all three recomputed
        # in backward instead. raw_gate_scratch/L_diag_scratch: same recompute-
        # not-save call the design this replaced made after measuring the
        # memory-vs-recompute tradeoff on real hardware (see that design's own
        # backward() comment; not re-measured here, but there's no structural
        # reason it would flip for this design). B_c: recomputing costs one
        # tiny matmul (see _build_B), cheaper than saving a second (heads,modes,
        # D,D) tensor across the forward/backward boundary for no real benefit.
        ctx.save_for_backward(x_c, weight_diag_c, weight_W_c, weight_U_c, weight_V_c)
        ctx.shapes = (N, heads, S, head_dim, modes, n_tokens, n_nh)
        # BLOCK_I NOT stored -- forward's and backward's row-chunked kernels use
        # DIFFERENT caps (see _block_i's own comment), so each recomputes its own
        # from BLOCK_D/BLOCK_S below rather than one being derived from a value
        # the other picked.
        ctx.blocks = (BLOCK_D, BLOCK_S, BLOCK_M)
        ctx.orig_dtypes = (weight_diag.dtype, weight_W.dtype, weight_U.dtype, weight_V.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_diag, weight_W, weight_U, weight_V = ctx.saved_tensors
        N, heads, S, head_dim, modes, n_tokens, n_nh = ctx.shapes
        BLOCK_D, BLOCK_S, BLOCK_M = ctx.blocks
        # cap=4096, same as forward's -- CONFIRMED at this cap: originally
        # halved to 2048 here as an unmeasured-margin hedge (_block_i's own
        # comment has the full history), then raised back to match forward's
        # proven point once kernels/tune_riemannian_block_s.py's real BLOCK_S
        # sweep showed BLOCK_S=32 (production's own default) already drove
        # BLOCK_I_BWD to its floor of 1 at the 2048 cap -- meaning no further
        # BLOCK_S retuning could help backward until BLOCK_I got some room
        # back. A subsequent full verify_riemannian_metric.py run at 4096
        # came back ALL CHECKS PASSED (no OutOfResources), and the benchmark
        # ticked up from 2.29x to 2.32x -- both directly attributable to
        # BLOCK_I_BWD going from 1 to 2 at BLOCK_S=32 (4096//(32*64)=2 vs.
        # 2048//(32*64)=1). Not re-verified above 4096 -- if head_dim/BLOCK_S
        # combinations far outside this file's own benchmark scale ever hit
        # OutOfResources here, that's real signal this cap has limits too,
        # not evidence to keep raising it further on reasoning alone.
        BLOCK_I_BWD = _block_i(BLOCK_S, BLOCK_D, 4096)
        USE_TF32 = x.dtype != torch.float32
        grad_out_c = grad_out.contiguous()
        B_c, lower_mask = RiemannianMetricKernel._build_B(weight_U, weight_V, head_dim)

        raw_gate_scratch = torch.empty(n_tokens, modes, dtype=torch.float32, device=x.device)
        L_diag_scratch = torch.empty(n_tokens, head_dim, dtype=torch.float32, device=x.device)
        grid = (n_nh, triton.cdiv(S, BLOCK_S))
        _riemannian_project_kernel[grid](
            x.view(n_tokens, head_dim), weight_W, weight_diag, raw_gate_scratch, L_diag_scratch,
            S, heads, head_dim, modes,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
            num_warps=8,
        )

        num_blocks = n_nh * triton.cdiv(S, BLOCK_S)
        NUM_GROUPS = min(32, num_blocks)

        grad_x = torch.empty_like(x)
        grad_gate_scratch = torch.empty(n_tokens, modes, dtype=x.dtype, device=x.device)
        grad_weight_diag_partial = torch.zeros(NUM_GROUPS, heads, head_dim, dtype=torch.float32, device=x.device)
        grad_B_partial = torch.zeros(NUM_GROUPS, heads, modes, head_dim, head_dim, dtype=torch.float32, device=x.device)

        # num_stages=2 pinned here too -- see the matching comment on
        # _riemannian_apply_fwd_kernel's own launch above.
        _riemannian_grad_per_token_kernel[grid](
            x.view(n_tokens, head_dim), grad_out_c.view(n_tokens, head_dim),
            grad_x.view(n_tokens, head_dim), grad_gate_scratch,
            grad_weight_diag_partial, grad_B_partial,
            raw_gate_scratch, L_diag_scratch, B_c, weight_diag,
            S, heads, head_dim, modes,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I_BWD, USE_TF32=USE_TF32,
            NUM_GROUPS=NUM_GROUPS,
            num_warps=8, num_stages=2,
        )

        grad_weight_W_partial = torch.zeros(NUM_GROUPS, heads, head_dim, modes, dtype=torch.float32, device=x.device)
        grad_x_w = torch.empty_like(x)
        _riemannian_weight_w_and_gradxw_kernel[grid](
            x.view(n_tokens, head_dim), weight_W, raw_gate_scratch, grad_gate_scratch,
            grad_x_w.view(n_tokens, head_dim), grad_weight_W_partial,
            S, heads, head_dim, modes,
            BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M, NUM_GROUPS=NUM_GROUPS,
            num_warps=8,
        )

        grad_x_total = grad_x + grad_x_w.to(x.dtype)
        grad_weight_diag_f32 = grad_weight_diag_partial.sum(dim=0)
        grad_weight_W_f32 = grad_weight_W_partial.sum(dim=0)
        grad_B_f32 = grad_B_partial.sum(dim=0)

        # grad_B_f32 is NOT already zero outside strictly-lower-triangular: the
        # per-token kernel derives it from grad_S, which (per MATH's backward
        # OFF-DIAGONAL note) can be numerically nonzero there even though B_hm
        # itself is exactly zero there. Re-mask before it touches weight_U/
        # weight_V -- this is the one thing the OLD (pre-U/V) design got for
        # free from `torch.where(lower_mask, B, 0)`'s own autograd backward,
        # which no longer runs since B is built with grad disabled now (see
        # _build_B). Skipping this would leak a numerically real but
        # semantically meaningless gradient into weight_U/weight_V's masked-out
        # entries.
        grad_B_masked_f32 = torch.where(lower_mask, grad_B_f32, 0.0)
        # grad_weight_U_h[m,i,r] = sum_j grad_B_hm[i,j]*weight_V_h[m,j,r]
        grad_weight_U_f32 = torch.matmul(grad_B_masked_f32, weight_V.to(torch.float32))
        # grad_weight_V_h[m,j,r] = sum_i grad_B_hm[i,j]*weight_U_h[m,i,r]
        grad_weight_V_f32 = torch.matmul(grad_B_masked_f32.transpose(-2, -1), weight_U.to(torch.float32))

        orig_diag_dtype, orig_W_dtype, orig_U_dtype, orig_V_dtype = ctx.orig_dtypes
        return (
            grad_x_total,
            grad_weight_diag_f32.to(orig_diag_dtype),
            grad_weight_W_f32.to(orig_W_dtype),
            grad_weight_U_f32.to(orig_U_dtype),
            grad_weight_V_f32.to(orig_V_dtype),
        )
