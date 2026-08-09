"""
Run this on a CUDA GPU to validate riemannian_metric.py before trusting it in
real training. It was written and hand-verified for its underlying math
(against torch.autograd.grad, in pure PyTorch) without any GPU access, so the
one thing NOT yet checked is whether the Triton kernel code itself is correct
-- that's what this script checks.

Usage:
    python verify_riemannian_metric.py

Report back: which sections PASSED/FAILED, and the benchmark numbers at the
end (especially peak memory, since that's the actual point of this).
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def _nudge_off_init(ref, tri):
    # weight starts at exactly 0, which makes raw=0 uniformly and L_raw exactly 0
    # (identity at init -- see model.py's RiemannianMetric docstring,
    # _DIAG_PACKED_BASELINE). L_raw carries no real per-token/per-direction
    # structure until weight moves -- nudging it off zero (matched between
    # ref/tri) is required to actually exercise L_raw's nonlinear packing in
    # these checks.
    with torch.no_grad():
        ref.weight.add_(0.05 * torch.randn_like(ref.weight))
        tri.weight.data.copy_(ref.weight.data)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU. Aborting.")
        sys.exit(1)

    try:
        import triton  # noqa: F401
    except ImportError:
        print("triton is not installed. `pip install triton==3.0.0` (or matching your torch build) and retry.")
        sys.exit(1)

    from model import RiemannianMetric

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # =========================================================================
    print("=" * 78)
    print("0. Identity-at-init: L must be (numerically) the identity matrix at a")
    print("   fresh (unnudged) init, i.e. RiemannianMetric(x) ~= x, for both the")
    print("   eager and fused paths. Not bit-exact equality: log1p/softplus computed")
    print("   via PyTorch tensor ops (here) vs. the Python float64 math.log1p/math.log")
    print("   _DIAG_PACKED_BASELINE was derived from (model.py) aren't guaranteed to")
    print("   round to the identical bit pattern, especially at fp16/bf16 -- a tight")
    print("   tolerance is the correct bar here, not torch.equal.")
    print("=" * 78)
    identity_tol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.5e-2}
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=2, heads=16, S=8, head_dim=64)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.manual_seed(0)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device).to(dtype)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device).to(dtype)
            # No _nudge_off_init here -- weight must stay at its real zero init.

            x = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=dtype)
            with torch.no_grad():
                out_ref = ref(x)
                out_tri = tri(x)
            atol = identity_tol[dtype]
            check(
                f"eager output ~= input cfg={cfg} dtype={dtype}",
                torch.allclose(out_ref.float(), x.float(), rtol=0, atol=atol),
                f"max_abs_diff={(out_ref.float()-x.float()).abs().max().item():.2e}",
            )
            check(
                f"fused output ~= input cfg={cfg} dtype={dtype}",
                torch.allclose(out_tri.float(), x.float(), rtol=0, atol=atol),
                f"max_abs_diff={(out_tri.float()-x.float()).abs().max().item():.2e}",
            )

    # =========================================================================
    print()
    print("=" * 78)
    print("1. Forward numerical match (Triton vs. reference), several configs/dtypes")
    print("=" * 78)
    configs = [
        dict(N=2, heads=4, S=8, head_dim=8),
        dict(N=2, heads=4, S=8, head_dim=16),
        dict(N=4, heads=4, S=16, head_dim=64),   # matches your small test config's head_dim
        dict(N=2, heads=16, S=8, head_dim=64),   # matches your real config's head_dim/heads
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    tol = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2), torch.bfloat16: (2e-2, 2e-2)}

    for cfg in configs:
        for dtype in dtypes:
            torch.manual_seed(0)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device)
            _nudge_off_init(ref, tri)

            x = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=dtype)
            ref_d, tri_d = ref.to(dtype), tri.to(dtype)
            with torch.no_grad():
                out_ref = ref_d(x)
                out_tri = tri_d(x)
            diff = (out_ref.float() - out_tri.float()).abs()
            rtol, atol = tol[dtype]
            ok = torch.allclose(out_ref.float(), out_tri.float(), rtol=rtol, atol=atol)
            check(
                f"cfg={cfg} dtype={dtype}",
                ok,
                f"max_abs_diff={diff.max().item():.2e}",
            )

    # =========================================================================
    print()
    print("=" * 78)
    print("1b. Extreme-magnitude stress test, isolated to the OFF-DIAGONAL (asinh)")
    print("    path -- the regime that produced a real silent NaN in training. Diagonal")
    print("    (softplus) weight columns are zeroed out on purpose: softplus is")
    print("    UNBOUNDED for large positive raw (softplus(x)~=x), so M=L@L^T squares")
    print("    that into the metric's diagonal -- blowing up BOTH paths at once (an")
    print("    earlier version of this test did exactly that) saturates fp16/bf16 for")
    print("    a completely different, unrelated reason and masks the thing actually")
    print("    being checked here. weight stays fp32 throughout, matching")
    print("    real autocast training (nn.Parameters are never cast down) -- only the")
    print("    activation x is cast per dtype, via torch.autocast like section 2b.")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=2, heads=16, S=8, head_dim=64)]:
        torch.manual_seed(4)
        ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device)
        tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device)
        _nudge_off_init(ref, tri)
        with torch.no_grad():
            for m in (ref, tri):
                m.weight.data[..., m.diag_mask] = 0.0          # raw stays exactly 0 -> softplus(0), safe
                m.weight.data[..., ~m.diag_mask] *= 20000.0    # only the asinh-fed columns blow up

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (3.0 * torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device)).to(dtype)

            with torch.no_grad():
                raw_check = torch.einsum("nhsd,hdp->nhsp", x.float(), ref.weight.float()) / ref.temperature
            max_raw_offdiag = raw_check[..., ~ref.diag_mask].abs().max().item()
            check(
                f"(sanity) |raw| off-diagonal reaches extreme regime cfg={cfg} dtype={dtype}",
                max_raw_offdiag > 4096,
                f"max|raw|offdiag={max_raw_offdiag:.2e}",
            )

            x_ref = x.clone().requires_grad_(True)
            x_tri = x.clone().requires_grad_(True)
            if dtype == torch.float32:
                out_ref = ref(x_ref)
                out_tri = tri(x_tri)
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out_ref = ref(x_ref)
                    out_tri = tri(x_tri)

            ref_finite = torch.isfinite(out_ref).all().item()
            tri_finite = torch.isfinite(out_tri).all().item()
            check(f"no NaN/Inf in Triton forward        cfg={cfg} dtype={dtype}", tri_finite)
            if dtype == torch.float32:
                check(f"no NaN/Inf in reference forward    cfg={cfg} dtype={dtype}", ref_finite)
            else:
                print(f"  [INFO] reference forward finite cfg={cfg} dtype={dtype}: {ref_finite} (PyTorch's own autocast path, not our kernel -- informational)")

            if ref_finite:
                diff = (out_ref.float() - out_tri.float()).abs()
                out_scale = out_ref.float().abs().max().clamp_min(1.0).item()
                rel_err = diff.max().item() / out_scale
                check(
                    f"forward tracks reference (rel. to dynamic range) cfg={cfg} dtype={dtype}", rel_err < 0.05,
                    f"max_abs_diff={diff.max().item():.2e}  scale={out_scale:.2e}  rel_err={rel_err:.2e}",
                )

            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            grad_x_ref_finite = torch.isfinite(x_ref.grad).all().item()
            grad_w_ref_finite = torch.isfinite(ref.weight.grad).all().item()
            check(f"no NaN/Inf in Triton grad_x         cfg={cfg} dtype={dtype}", torch.isfinite(x_tri.grad).all().item())
            check(f"no NaN/Inf in Triton grad_weight    cfg={cfg} dtype={dtype}", torch.isfinite(tri.weight.grad).all().item())
            if dtype == torch.float32:
                check(f"no NaN/Inf in reference grad_x      cfg={cfg} dtype={dtype}", grad_x_ref_finite)
                check(f"no NaN/Inf in reference grad_weight cfg={cfg} dtype={dtype}", grad_w_ref_finite)
            else:
                print(f"  [INFO] reference grad_x/grad_weight finite cfg={cfg} dtype={dtype}: {grad_x_ref_finite}/{grad_w_ref_finite} (informational)")

            ref.weight.grad = None
            tri.weight.grad = None

    # =========================================================================
    print()
    print("=" * 78)
    print("1c. Extreme-magnitude stress test, isolated to the DIAGONAL (log1p(softplus))")
    print("    path -- plain softplus(x)~=x is UNBOUNDED (unlike asinh's deliberately")
    print("    bounded growth), so nothing stopped raw from drifting large as weight grows")
    print("    over real training steps, and M=L@L^T squares whatever L's diagonal reaches")
    print("    -- large enough to make the module's actual OUTPUT (not just an internal")
    print("    value) exceed fp16's ~65504 ceiling, which no amount of internal fp32")
    print("    precision can rescue once the true mathematical answer is itself too large")
    print("    to represent. The diagonal packing nonlinearity is now log(1+softplus(x)),")
    print("    not plain softplus(x) -- composing the outer log1p compresses softplus's")
    print("    linear growth to logarithmic, making this PRACTICALLY self-bounding (see")
    print("    model.py's RiemannianMetric.forward) without an explicit clamp, so both the")
    print("    output AND grad_x must stay finite AND within fp16 range at every dtype, not")
    print("    just fp32/bf16. off-diagonal (asinh) weight columns are zeroed to isolate this,")
    print("    same reasoning as section 1b in reverse.")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=2, heads=16, S=8, head_dim=64)]:
        torch.manual_seed(5)
        ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device)
        tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device)
        _nudge_off_init(ref, tri)
        with torch.no_grad():
            for m in (ref, tri):
                m.weight.data[..., ~m.diag_mask] = 0.0          # off-diagonal (asinh) stays at 0, isolated out
                m.weight.data[..., m.diag_mask] *= 300000.0     # only the softplus-fed (diagonal) columns blow up

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (3.0 * torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device)).to(dtype)

            with torch.no_grad():
                raw_check = torch.einsum("nhsd,hdp->nhsp", x.float(), ref.weight.float()) / ref.temperature
            max_raw_diag = raw_check[..., ref.diag_mask].abs().max().item()
            check(
                f"(sanity) raw diagonal exceeds fp16's max (65504) cfg={cfg} dtype={dtype}",
                max_raw_diag > 65504,
                f"max_raw_diag={max_raw_diag:.2e}",
            )

            x_ref = x.clone().requires_grad_(True)
            x_tri = x.clone().requires_grad_(True)
            if dtype == torch.float32:
                out_ref = ref(x_ref)
                out_tri = tri(x_tri)
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out_ref = ref(x_ref)
                    out_tri = tri(x_tri)

            # Triton finite AND fp16-representable is the hard requirement at every
            # dtype -- that's what this section exists to check, and it's the path
            # real (fused=True) training actually uses. The reference is plain PyTorch
            # (einsum/softplus/log1p) running through real torch.autocast: raw itself
            # (~1e5-2e5 here, deliberately) gets cast to fp16 by autocast's own einsum
            # policy and overflows BEFORE log1p(softplus(...)) ever sees it -- nothing
            # our fix does can prevent PyTorch's own intermediate from truncating to
            # fp16 on the way there. Our kernel never hits this because tl.dot's
            # accumulator is fp32 regardless of input dtype, so raw never touches fp16
            # internally -- same "our kernel is more robust than the reference's own
            # autocast path, not less" situation as section 1b, so reference is only
            # required finite at fp32 (the real ground truth); informational elsewhere.
            fp16_ceiling = 65504.0
            ref_finite = torch.isfinite(out_ref).all().item()
            tri_finite = torch.isfinite(out_tri).all().item()
            tri_in_range = out_tri.float().abs().max().item() < fp16_ceiling
            check(f"no NaN/Inf in Triton forward         cfg={cfg} dtype={dtype}", tri_finite)
            check(
                f"Triton forward stays fp16-representable    cfg={cfg} dtype={dtype}", tri_in_range,
                f"max_abs={out_tri.float().abs().max().item():.2e}",
            )
            if dtype == torch.float32:
                check(f"no NaN/Inf in reference forward     cfg={cfg} dtype={dtype}", ref_finite)
                check(
                    f"reference forward stays fp16-representable cfg={cfg} dtype={dtype}",
                    out_ref.float().abs().max().item() < fp16_ceiling,
                    f"max_abs={out_ref.float().abs().max().item():.2e}",
                )
            else:
                print(f"  [INFO] reference forward finite cfg={cfg} dtype={dtype}: {ref_finite} (PyTorch's own autocast path, not our kernel -- informational)")

            if ref_finite:
                diff = (out_ref.float() - out_tri.float()).abs()
                out_scale = out_ref.float().abs().max().clamp_min(1.0).item()
                rel_err = diff.max().item() / out_scale
                check(
                    f"forward tracks reference (rel. to dynamic range) cfg={cfg} dtype={dtype}", rel_err < 0.05,
                    f"max_abs_diff={diff.max().item():.2e}  scale={out_scale:.2e}  rel_err={rel_err:.2e}",
                )

            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            # grad_x's projection-path term (grad_raw0 @ weight^T) is exactly what the
            # gradient-side half of this fix protects -- weight itself is still huge
            # here (300000x), so grad_x staying bounded is the real proof d(packed)/d(raw)
            # (derived from log1p(softplus(raw))) is wired up correctly, not just the
            # forward pass alone. Same reference-vs-Triton reasoning as the forward
            # checks above -- reference required finite only at fp32.
            grad_x_ref_finite = torch.isfinite(x_ref.grad).all().item()
            grad_x_tri_finite = torch.isfinite(x_tri.grad).all().item()
            grad_w_ref_finite = torch.isfinite(ref.weight.grad).all().item()
            grad_w_tri_finite = torch.isfinite(tri.weight.grad).all().item()
            check(f"no NaN/Inf in Triton grad_x         cfg={cfg} dtype={dtype}", grad_x_tri_finite)
            check(f"no NaN/Inf in Triton grad_weight    cfg={cfg} dtype={dtype}", grad_w_tri_finite)
            if dtype == torch.float32:
                check(f"no NaN/Inf in reference grad_x      cfg={cfg} dtype={dtype}", grad_x_ref_finite)
                check(f"no NaN/Inf in reference grad_weight cfg={cfg} dtype={dtype}", grad_w_ref_finite)
            else:
                print(f"  [INFO] reference grad_x/grad_weight finite cfg={cfg} dtype={dtype}: {grad_x_ref_finite}/{grad_w_ref_finite} (informational)")
            if dtype == torch.float32 and grad_x_ref_finite:
                check(
                    f"reference grad_x stays fp16-representable cfg={cfg} dtype={dtype}",
                    x_ref.grad.float().abs().max().item() < fp16_ceiling,
                    f"max_abs={x_ref.grad.float().abs().max().item():.2e}",
                )
            if grad_x_tri_finite:
                check(
                    f"Triton grad_x stays fp16-representable    cfg={cfg} dtype={dtype}",
                    x_tri.grad.float().abs().max().item() < fp16_ceiling,
                    f"max_abs={x_tri.grad.float().abs().max().item():.2e}",
                )

            ref.weight.grad = None
            tri.weight.grad = None

    # =========================================================================
    print()
    print("=" * 78)
    print("2. Backward numerical match: grad_x, grad_weight")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=4, heads=4, S=16, head_dim=64), dict(N=2, heads=16, S=8, head_dim=64)]:
        for dtype in [torch.float32, torch.float16]:
            torch.manual_seed(1)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device).to(dtype)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device).to(dtype)
            _nudge_off_init(ref, tri)

            x_ref = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=dtype, requires_grad=True)
            x_tri = x_ref.detach().clone().requires_grad_(True)

            out_ref = ref(x_ref)
            out_tri = tri(x_tri)
            grad_out = torch.randn_like(out_ref)

            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            rtol, atol = tol[dtype]
            gx_ok = torch.allclose(x_ref.grad.float(), x_tri.grad.float(), rtol=rtol, atol=atol)
            gw_ok = torch.allclose(ref.weight.grad.float(), tri.weight.grad.float(), rtol=rtol, atol=atol)
            check(
                f"grad_x            cfg={cfg} dtype={dtype}", gx_ok,
                f"max_abs_diff={(x_ref.grad.float()-x_tri.grad.float()).abs().max().item():.2e}",
            )
            check(
                f"grad_weight       cfg={cfg} dtype={dtype}", gw_ok,
                f"max_abs_diff={(ref.weight.grad.float()-tri.weight.grad.float()).abs().max().item():.2e}",
            )

    # =========================================================================
    print()
    print("=" * 78)
    print("2b. Mismatched dtypes: x fp16, weight fp32 (mimics real")
    print("    torch.autocast training, where activations get cast but nn.Parameters")
    print("    don't -- this Triton path isn't an autocast-aware op, unlike the plain")
    print("    PyTorch fallback, so it needs its own explicit cast)")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=4, heads=4, S=16, head_dim=64)]:
        torch.manual_seed(1)
        ref = RiemannianMetric(cfg["heads"], cfg["head_dim"]).to(device)  # fp32, untouched
        tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], fused=True).to(device)  # fp32, untouched
        _nudge_off_init(ref, tri)

        x_ref = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=torch.float16, requires_grad=True)
        x_tri = x_ref.detach().clone().requires_grad_(True)

        try:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out_ref = ref(x_ref)
                out_tri = tri(x_tri)

            # backward() runs outside autocast -- matches train.py's own
            # structure (forward+loss inside `with torch.autocast(...)`,
            # `.backward()` after exiting it).
            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            rtol, atol = tol[torch.float16]
            fwd_ok = torch.allclose(out_ref.float(), out_tri.float(), rtol=rtol, atol=atol)
            gx_ok = torch.allclose(x_ref.grad.float(), x_tri.grad.float(), rtol=rtol, atol=atol)
            gw_ok = torch.allclose(ref.weight.grad.float(), tri.weight.grad.float(), rtol=rtol, atol=atol)
            check(f"mismatched-dtype forward cfg={cfg}", fwd_ok, f"max_abs_diff={(out_ref.float()-out_tri.float()).abs().max().item():.2e}")
            check(f"mismatched-dtype grad_x cfg={cfg}", gx_ok, f"max_abs_diff={(x_ref.grad.float()-x_tri.grad.float()).abs().max().item():.2e}")
            check(f"mismatched-dtype grad_weight cfg={cfg}", gw_ok, f"max_abs_diff={(ref.weight.grad.float()-tri.weight.grad.float()).abs().max().item():.2e}")
            check(f"mismatched-dtype weight.grad stayed fp32 cfg={cfg}", tri.weight.grad.dtype == torch.float32, f"got {tri.weight.grad.dtype}")
        except Exception as e:
            check(f"mismatched-dtype fwd+bwd ran cfg={cfg}", False, f"EXCEPTION: {e}")

    # =========================================================================
    print()
    print("=" * 78)
    print("3. End-to-end: swap into an actual GPTmodel, run forward+backward+optimizer step")
    print("=" * 78)
    from config import ModelConfig
    from model import GPTmodel

    CFG = dict(embed_dim=256, n_decoders=3, vocab_size=1000, ff_dim=512, heads=4, dropout=0.0, seq_len=64)
    torch.manual_seed(2)
    m = GPTmodel.build(ModelConfig(riemannian=True, **CFG)).to(device)
    m.train()

    # Monkey-patch: swap each layer's ref RiemannianMetric for the fused RiemannianMetric with matching weights.
    for dec in m.decoders:
        mha = dec.masked_multihead_attention
        old = mha.riemannian_metric
        heads = old.weight.shape[0]
        new = RiemannianMetric(heads, old.head_dim, fused=True).to(device)
        new.weight.data.copy_(old.weight.data)
        mha.riemannian_metric = new

    x = torch.randint(0, CFG["vocab_size"], (4, 64), device=device)
    targets = torch.randint(0, CFG["vocab_size"], (4, 64), device=device)
    mask = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=device))

    try:
        out = m(x, mask)
        loss = F.cross_entropy(out.view(-1, out.shape[-1]), targets.view(-1))
        loss.backward()
        has_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in m.parameters() if p.grad is not None)
        check("full-model forward+backward ran, loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
        check("no NaN/Inf in any gradient", not has_nan)
    except Exception as e:
        check("full-model forward+backward ran", False, f"EXCEPTION: {e}")

    # =========================================================================
    print()
    print("=" * 78)
    print("4. Benchmark: wall-clock and peak memory, at your actual small-test scale")
    print("=" * 78)
    N, heads, S, head_dim = 24, 4, 512, 64
    n_decoders = 6

    def build_stack(use_triton):
        torch.manual_seed(3)
        layers = []
        for _ in range(n_decoders):
            rm = RiemannianMetric(heads, head_dim).to(device).half()
            if use_triton:
                tri = RiemannianMetric(heads, head_dim, fused=True).to(device).half()
                tri.weight.data.copy_(rm.weight.data)
                layers.append(tri)
            else:
                layers.append(rm)
        return layers

    def bench(layers, n=10):
        # A fixed x (key) shared across all n_decoders layers, matching how the
        # real model uses it: each decoder has its own independent
        # RiemannianMetric instance applied to that decoder's own key, not
        # chained output-to-input across layers.
        x = torch.randn(N, heads, S, head_dim, device=device, dtype=torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def run_iter():
            total = None
            for layer in layers:
                out = layer(x)
                total = out.sum() if total is None else total + out.sum()
            total.backward()
            x.grad = None

        for _ in range(2):
            run_iter()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            run_iter()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        return elapsed, peak_mem

    ref_layers = build_stack(use_triton=False)
    t_ref, mem_ref = bench(ref_layers)
    tri_layers = build_stack(use_triton=True)
    t_tri, mem_tri = bench(tri_layers)

    print(f"  reference (PyTorch, {n_decoders} layers): {t_ref*1000:.2f} ms/iter, peak mem {mem_ref:.3f} GB")
    print(f"  triton    (Triton,  {n_decoders} layers): {t_tri*1000:.2f} ms/iter, peak mem {mem_tri:.3f} GB")
    print(f"  speedup: {t_ref/t_tri:.2f}x   memory reduction: {mem_ref/mem_tri:.2f}x")

    # =========================================================================
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
