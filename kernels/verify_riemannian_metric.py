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
    # weight_diag/weight_W both start at exactly 0 -- both required for
    # identity-at-init (see model.py's RiemannianMetric docstring: zero
    # weight_diag -> raw_diag=0 -> packed_diag=0 via the -C shift; zero
    # weight_W -> every gate_k=asinh(0)=0 regardless of weight_U/weight_V).
    # weight_U/weight_V already get a small random init from __init__ --
    # safe at init regardless of value, since weight_W=0 gates them to zero
    # contribution either way. Nudge weight_diag/weight_W off zero (matched
    # between ref/tri) to actually exercise the gated off-diagonal path in
    # these checks; explicitly copy weight_U/weight_V too so ref/tri start
    # from identical values rather than relying on RNG-stream alignment
    # across two separate constructions.
    with torch.no_grad():
        ref.weight_diag.add_(0.3 * torch.randn_like(ref.weight_diag))
        ref.weight_W.add_(0.2 * torch.randn_like(ref.weight_W))
        tri.weight_diag.data.copy_(ref.weight_diag.data)
        tri.weight_W.data.copy_(ref.weight_W.data)
        tri.weight_U.data.copy_(ref.weight_U.data)
        tri.weight_V.data.copy_(ref.weight_V.data)


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
    print("   eager and fused paths. Not bit-exact equality: log1p/softplus/asinh")
    print("   computed via PyTorch tensor ops (eager) vs Triton libdevice calls")
    print("   (fused) aren't guaranteed to round to the identical bit pattern,")
    print("   especially at fp16/bf16 -- a tight tolerance is the correct bar")
    print("   here, not torch.equal.")
    print("=" * 78)
    # fp16/fp32 margins sized off real hardware runs: eager chains native
    # F.softplus/log1p/asinh calls directly on fp16/bf16 tensors, compounding
    # rounding at each step, vs. the fused kernel's single upcast-to-fp32-
    # then-downcast-once -- observed eager residuals landed on exact ULP
    # steps (1/128 at fp16, 1/16 at bf16), not a real divergence.
    identity_tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 8e-2}
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=2, heads=16, S=8, head_dim=64)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.manual_seed(0)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"]).to(device).to(dtype)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"], fused=True).to(device).to(dtype)
            tri.weight_U.data.copy_(ref.weight_U.data)
            tri.weight_V.data.copy_(ref.weight_V.data)
            # No _nudge_off_init here -- weight_diag/weight_W must stay at their real zero init.

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
    print("   rank=head_dim throughout, matching model.py's current wiring")
    print("=" * 78)
    configs = [
        dict(N=2, heads=4, S=8, head_dim=8),
        dict(N=2, heads=4, S=8, head_dim=16),
        dict(N=4, heads=4, S=16, head_dim=64),
        dict(N=2, heads=16, S=8, head_dim=64),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    tol = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2), torch.bfloat16: (2e-2, 2e-2)}

    for cfg in configs:
        for dtype in dtypes:
            torch.manual_seed(0)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"]).to(device)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"], fused=True).to(device)
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
    print("1b. Extreme-magnitude stress test: weight_diag/weight_W nudged large,")
    print("    checking L_diag stays > 0 (non-singularity) and gate/forward/")
    print("    backward stay finite -- the analogous safety properties to the")
    print("    dense design's own diagonal-floor and asinh-self-boundedness")
    print("    checks, re-verified for this construction's own parameters.")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=2, heads=16, S=8, head_dim=64)]:
        torch.manual_seed(4)
        ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"]).to(device)
        tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"], fused=True).to(device)
        _nudge_off_init(ref, tri)
        with torch.no_grad():
            for m in (ref, tri):
                m.weight_diag.data *= 20000.0
                m.weight_W.data *= 20000.0

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (3.0 * torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device)).to(dtype)

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

            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            check(f"no NaN/Inf in Triton grad_x         cfg={cfg} dtype={dtype}", torch.isfinite(x_tri.grad).all().item())
            check(f"no NaN/Inf in Triton grad_weight_diag cfg={cfg} dtype={dtype}", torch.isfinite(tri.weight_diag.grad).all().item())
            check(f"no NaN/Inf in Triton grad_weight_W   cfg={cfg} dtype={dtype}", torch.isfinite(tri.weight_W.grad).all().item())
            check(f"no NaN/Inf in Triton grad_weight_U   cfg={cfg} dtype={dtype}", torch.isfinite(tri.weight_U.grad).all().item())
            check(f"no NaN/Inf in Triton grad_weight_V   cfg={cfg} dtype={dtype}", torch.isfinite(tri.weight_V.grad).all().item())

            ref.weight_diag.grad = None
            ref.weight_W.grad = None
            ref.weight_U.grad = None
            ref.weight_V.grad = None
            tri.weight_diag.grad = None
            tri.weight_W.grad = None
            tri.weight_U.grad = None
            tri.weight_V.grad = None

    # =========================================================================
    print()
    print("=" * 78)
    print("2. Backward numerical match: grad_x, grad_weight_diag, grad_weight_W,")
    print("   grad_weight_U, grad_weight_V")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=4, heads=4, S=16, head_dim=64), dict(N=2, heads=16, S=8, head_dim=64)]:
        for dtype in [torch.float32, torch.float16]:
            torch.manual_seed(1)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"]).to(device).to(dtype)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"], fused=True).to(device).to(dtype)
            _nudge_off_init(ref, tri)

            x_ref = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=dtype, requires_grad=True)
            x_tri = x_ref.detach().clone().requires_grad_(True)

            out_ref = ref(x_ref)
            out_tri = tri(x_tri)
            grad_out = torch.randn_like(out_ref)

            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            rtol, atol = tol[dtype]
            checks = [
                ("grad_x", x_ref.grad, x_tri.grad),
                ("grad_weight_diag", ref.weight_diag.grad, tri.weight_diag.grad),
                ("grad_weight_W", ref.weight_W.grad, tri.weight_W.grad),
                ("grad_weight_U", ref.weight_U.grad, tri.weight_U.grad),
                ("grad_weight_V", ref.weight_V.grad, tri.weight_V.grad),
            ]
            for name, g_ref, g_tri in checks:
                ok = torch.allclose(g_ref.float(), g_tri.float(), rtol=rtol, atol=atol)
                check(
                    f"{name:<18} cfg={cfg} dtype={dtype}", ok,
                    f"max_abs_diff={(g_ref.float()-g_tri.float()).abs().max().item():.2e}",
                )

    # =========================================================================
    print()
    print("=" * 78)
    print("2b. Mismatched dtypes: x fp16, weights fp32 (mimics real")
    print("    torch.autocast training, where activations get cast but nn.Parameters")
    print("    don't -- this Triton path isn't an autocast-aware op, unlike the plain")
    print("    PyTorch fallback, so it needs its own explicit cast)")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=4, heads=4, S=16, head_dim=64)]:
        torch.manual_seed(1)
        ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"]).to(device)  # fp32, untouched
        tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["head_dim"], fused=True).to(device)  # fp32, untouched
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
            check(f"mismatched-dtype forward cfg={cfg}", fwd_ok, f"max_abs_diff={(out_ref.float()-out_tri.float()).abs().max().item():.2e}")

            # grad_x tracks x's own dtype (fp16 here, by normal autograd
            # convention) -- only the weight gradients are expected to stay
            # fp32 (the weights themselves are fp32, untouched by autocast).
            ok = torch.allclose(x_ref.grad.float(), x_tri.grad.float(), rtol=rtol, atol=atol)
            check(f"mismatched-dtype grad_x cfg={cfg}", ok, f"max_abs_diff={(x_ref.grad.float()-x_tri.grad.float()).abs().max().item():.2e}")

            for name, g_ref, g_tri in [
                ("grad_weight_diag", ref.weight_diag.grad, tri.weight_diag.grad),
                ("grad_weight_W", ref.weight_W.grad, tri.weight_W.grad),
                ("grad_weight_U", ref.weight_U.grad, tri.weight_U.grad),
                ("grad_weight_V", ref.weight_V.grad, tri.weight_V.grad),
            ]:
                ok = torch.allclose(g_ref.float(), g_tri.float(), rtol=rtol, atol=atol)
                check(f"mismatched-dtype {name} cfg={cfg}", ok, f"max_abs_diff={(g_ref.float()-g_tri.float()).abs().max().item():.2e}")
                check(f"mismatched-dtype {name} stayed fp32 cfg={cfg}", g_tri.dtype == torch.float32, f"got {g_tri.dtype}")
        except Exception as e:
            check(f"mismatched-dtype fwd+bwd ran cfg={cfg}", False, f"EXCEPTION: {e}")

    # =========================================================================
    print()
    print("=" * 78)
    print("2c. Rank-tiling: rank exceeds BLOCK_R's single-tile cap, forcing")
    print("    the NUM_R_TILES>1 code path in _riemannian_apply_fwd_kernel and")
    print("    _riemannian_grad_per_token_kernel (previously, exceeding that cap")
    print("    silently truncated rank components instead of erroring -- see module")
    print("    docstring's NOTE on tiling). None of the sections above exercise")
    print("    this: they all use rank=head_dim<=64, which fits in one tile.")
    print("=" * 78)
    # head_dim=32, rank=200: at fp32 BLOCK_R caps at 128//2=64, giving
    # NUM_R_TILES=cdiv(200,64)=4 with a partial last tile (200-3*64=8 valid
    # columns) -- exercises both multi-tile accumulation and partial-tile
    # masking in one config.
    for cfg in [dict(N=2, heads=4, S=8, head_dim=32, rank=200)]:
        for dtype in [torch.float32, torch.float16]:
            torch.manual_seed(1)
            ref = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["rank"]).to(device).to(dtype)
            tri = RiemannianMetric(cfg["heads"], cfg["head_dim"], cfg["rank"], fused=True).to(device).to(dtype)
            _nudge_off_init(ref, tri)

            x_ref = torch.randn(cfg["N"], cfg["heads"], cfg["S"], cfg["head_dim"], device=device, dtype=dtype, requires_grad=True)
            x_tri = x_ref.detach().clone().requires_grad_(True)

            out_ref = ref(x_ref)
            out_tri = tri(x_tri)
            rtol, atol = tol[dtype]
            fwd_ok = torch.allclose(out_ref.float(), out_tri.float(), rtol=rtol, atol=atol)
            check(f"tiled-rank forward cfg={cfg} dtype={dtype}", fwd_ok, f"max_abs_diff={(out_ref.float()-out_tri.float()).abs().max().item():.2e}")

            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_tri.backward(grad_out)

            checks = [
                ("grad_x", x_ref.grad, x_tri.grad),
                ("grad_weight_diag", ref.weight_diag.grad, tri.weight_diag.grad),
                ("grad_weight_W", ref.weight_W.grad, tri.weight_W.grad),
                ("grad_weight_U", ref.weight_U.grad, tri.weight_U.grad),
                ("grad_weight_V", ref.weight_V.grad, tri.weight_V.grad),
            ]
            for name, g_ref, g_tri in checks:
                ok = torch.allclose(g_ref.float(), g_tri.float(), rtol=rtol, atol=atol)
                check(
                    f"tiled-rank {name:<18} cfg={cfg} dtype={dtype}", ok,
                    f"max_abs_diff={(g_ref.float()-g_tri.float()).abs().max().item():.2e}",
                )

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
        new = RiemannianMetric(old.heads, old.head_dim, old.rank, fused=True).to(device)
        new.weight_diag.data.copy_(old.weight_diag.data)
        new.weight_W.data.copy_(old.weight_W.data)
        new.weight_U.data.copy_(old.weight_U.data)
        new.weight_V.data.copy_(old.weight_V.data)
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
            rm = RiemannianMetric(heads, head_dim, head_dim).to(device).half()
            if use_triton:
                tri = RiemannianMetric(heads, head_dim, head_dim, fused=True).to(device).half()
                tri.weight_diag.data.copy_(rm.weight_diag.data)
                tri.weight_W.data.copy_(rm.weight_W.data)
                tri.weight_U.data.copy_(rm.weight_U.data)
                tri.weight_V.data.copy_(rm.weight_V.data)
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
    print("  NOTE: rank=head_dim in this wiring. L_offdiag is no longer")
    print("  materialized as a (D,D) matrix per token (see module docstring's")
    print("  updated NOTE on rank / _cumsum_apply_suffix's comment) -- expect")
    print("  a real speedup over the earlier (pre-cumsum) version of this")
    print("  kernel at this same rank=head_dim setting, though not the FULL")
    print("  low-rank compute win over the dense design, which still needs")
    print("  rank<<head_dim to go asymptotically below O(head_dim^2) per token.")

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
