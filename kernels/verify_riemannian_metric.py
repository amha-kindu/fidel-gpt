"""
Run this on a CUDA GPU to validate riemannian_triton.py before trusting it in
real training. It was written and hand-verified for its underlying math
(against torch.autograd.grad, in pure PyTorch) without any GPU access, so the
one thing NOT yet checked is whether the Triton kernel code itself is correct
-- that's what this script checks.

Usage:
    python verify_riemannian_triton.py

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
            ref = RiemannianMetric(cfg["head_dim"], cfg["heads"]).to(device)
            tri = RiemannianMetric(cfg["head_dim"], cfg["heads"], epsilon=ref.epsilon, fused=True).to(device)
            tri.weight.data.copy_(ref.weight.data)

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
    print("2. Backward numerical match: grad_x and grad_weight")
    print("=" * 78)
    for cfg in [dict(N=2, heads=4, S=8, head_dim=8), dict(N=4, heads=4, S=16, head_dim=64), dict(N=2, heads=16, S=8, head_dim=64)]:
        for dtype in [torch.float32, torch.float16]:
            torch.manual_seed(1)
            ref = RiemannianMetric(cfg["head_dim"], cfg["heads"]).to(device).to(dtype)
            
            tri = RiemannianMetric(cfg["head_dim"], cfg["heads"], epsilon=ref.epsilon, fused=True).to(device).to(dtype)
            tri.weight.data.copy_(ref.weight.data)

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
                f"grad_x  cfg={cfg} dtype={dtype}", gx_ok,
                f"max_abs_diff={(x_ref.grad.float()-x_tri.grad.float()).abs().max().item():.2e}",
            )
            check(
                f"grad_weight cfg={cfg} dtype={dtype}", gw_ok,
                f"max_abs_diff={(ref.weight.grad.float()-tri.weight.grad.float()).abs().max().item():.2e}",
            )

    # =========================================================================
    print()
    print("=" * 78)
    print("3. End-to-end: swap into an actual GPTmodel, run forward+backward+optimizer step")
    print("=" * 78)
    from config import ModelConfig
    from model import GPTmodel
    import model as model_module

    CFG = dict(embed_dim=256, n_decoders=3, vocab_size=1000, ff_dim=512, heads=4, dropout=0.0, seq_len=64)
    torch.manual_seed(2)
    m = GPTmodel.build(ModelConfig(riemannian=True, **CFG)).to(device)
    m.train()

    # Monkey-patch: swap each layer's the ref RiemannianMetric for the fused RiemannianMetric with matching weights.
    for dec in m.decoders:
        mha = dec.masked_multihead_attention
        old = mha.riemannian_metric
        new = RiemannianMetric(old.head_dim, mha.heads, epsilon=old.epsilon, fused=True).to(device)
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
            rm = RiemannianMetric(head_dim, heads).to(device).half()
            if use_triton:
                tri = RiemannianMetric(head_dim, heads, epsilon=rm.epsilon, fused=True).to(device).half()
                
                tri.weight.data.copy_(rm.weight.data)
                layers.append(tri)
            else:
                layers.append(rm)
        return layers

    def bench(layers, n=10):
        x = torch.randn(N, heads, S, head_dim, device=device, dtype=torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = x
            for layer in layers:
                out = layer(out)
            out.sum().backward()
            x.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            out = x
            for layer in layers:
                out = layer(out)
            out.sum().backward()
            x.grad = None
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
