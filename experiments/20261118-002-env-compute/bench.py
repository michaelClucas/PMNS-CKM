"""Environment + compute benchmark. Usage: python bench.py [--quick]
Runs at most 8 threads. Prints a plain-text report (redirect to bench.log)."""
import os, sys, time, platform
import numpy as np
import torch
from threadpoolctl import threadpool_limits

QUICK = "--quick" in sys.argv
N_MAT = 200_000 if QUICK else 1_000_000
SEED = 0


def timeit(fn, reps=3, sync=None):
    fn()  # warmup
    if sync: sync()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn()
        if sync: sync()
        best = min(best, time.perf_counter() - t)
    return best


def mlp_bench(device, n_steps=50):
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 8)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(4096, 8, device=device); y = torch.randn(4096, 8, device=device)
    def step():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward(); opt.step()
    sync = (lambda: torch.xpu.synchronize()) if device == "xpu" else None
    for _ in range(10): step()
    if sync: sync()
    t = time.perf_counter()
    for _ in range(n_steps): step()
    if sync: sync()
    return n_steps / (time.perf_counter() - t)


def matrix_bench(threads):
    rng = np.random.default_rng(SEED)
    A = (rng.standard_normal((N_MAT, 3, 3)) + 1j * rng.standard_normal((N_MAT, 3, 3)))
    B = (rng.standard_normal((N_MAT, 3, 3)) + 1j * rng.standard_normal((N_MAT, 3, 3)))
    H = A @ A.conj().transpose(0, 2, 1)
    tA, tB, tH = (torch.from_numpy(m) for m in (A, B, H))
    out = {}
    out["numpy matmul"] = timeit(lambda: A @ B)
    out["torch matmul"] = timeit(lambda: torch.matmul(tA, tB))
    out["numpy eigh"] = timeit(lambda: np.linalg.eigh(H), reps=1)
    out["torch eigh"] = timeit(lambda: torch.linalg.eigh(tH), reps=1)
    out["numpy svd"] = timeit(lambda: np.linalg.svd(A), reps=1)
    out["torch svd"] = timeit(lambda: torch.linalg.svd(tA), reps=1)
    # complex64 variants (what a float32 pipeline would use)
    tA64, tH64 = tA.to(torch.complex64), tH.to(torch.complex64)
    out["torch eigh c64"] = timeit(lambda: torch.linalg.eigh(tH64), reps=1)
    out["torch svd c64"] = timeit(lambda: torch.linalg.svd(tA64), reps=1)
    return out


def xpu_matrix(n):
    sync = torch.xpu.synchronize
    g = torch.Generator().manual_seed(SEED)
    A = torch.randn(n, 3, 3, dtype=torch.complex64, generator=g)
    B = torch.randn(n, 3, 3, dtype=torch.complex64, generator=g)
    H = A @ A.mH
    dA, dB, dH = A.to("xpu"), B.to("xpu"), H.to("xpu")
    res = {"xpu matmul c64": timeit(lambda: dA @ dB, sync=sync)}
    for name, fn in (("xpu eigh c64", lambda: torch.linalg.eigh(dH)), ("xpu svd c64", lambda: torch.linalg.svd(dA))):
        try:
            res[name] = timeit(fn, reps=1, sync=sync)
        except Exception as e:
            res[name] = f"FAILED ({type(e).__name__})"
    print("xpu: " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)


def main():
    print(f"python {platform.python_version()}  numpy {np.__version__}  torch {torch.__version__}")
    print(f"os.cpu_count() = {os.cpu_count()}  default torch.get_num_threads() = {torch.get_num_threads()}")
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"torch.xpu.is_available() = {has_xpu}")
    if has_xpu:
        print(f"xpu device: {torch.xpu.get_device_name(0)}")
    print()
    print("== MLP emulator 8-256-256-256-8, batch 4096, float32, Adam: train steps/sec ==")
    for nt in (1, 2, 4, 8):
        torch.set_num_threads(nt)
        print(f"cpu threads={nt:2d}: {mlp_bench('cpu'):8.1f} steps/s", flush=True)
    if has_xpu:
        torch.set_num_threads(4)
        print(f"xpu           : {mlp_bench('xpu'):8.1f} steps/s", flush=True)
        xpu_matrix(N_MAT)
    print()
    print(f"== Batched 3x3 complex128, N={N_MAT}: seconds (best of reps) ==")
    for nt in (1, 4, 8):
        torch.set_num_threads(nt)
        with threadpool_limits(limits=nt):  # caps numpy's BLAS/LAPACK threads too
            res = matrix_bench(nt)
        print(f"torch threads={nt}: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
