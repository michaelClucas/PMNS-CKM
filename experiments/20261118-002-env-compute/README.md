# 20261118-002 env-compute

## Hypothesis
PyTorch XPU on the Intel iGPU is faster than CPU for our workloads (small MLP emulators, batched 3x3 complex linear algebra). **Result: no.**

## Method
- Repo `.venv` (Python 3.14.7): installed and pinned everything in `requirements.txt`. torch from the PyTorch CPU wheel index.
- XPU: installed `torch==2.15.0+xpu` from `https://download.pytorch.org/whl/xpu` into a throwaway venv outside the repo, so the team venv keeps the CPU build. Same benchmark script.
- `bench.py`: (a) MLP 8-256-256-256-8, SiLU, batch 4096, float32, Adam, MSE: train steps/s after 10 warmup steps, 50 timed steps, at 1/2/4/8 torch threads. (b) 1e6 random 3x3 complex128 matrices: batched matmul, `eigh` (Hermitian), `svd`, numpy vs torch, threads capped with `torch.set_num_threads` + `threadpoolctl` at 1/4/8. Also complex64 torch eigh/svd.
- `xpu_probe_*.py`: per-op first-call timing on XPU after the full XPU benchmark stalled.
- Max 8 threads, per the budget. 12 threads not run. Other agents were running jobs during the benchmark, so CPU numbers carry maybe +-30% noise.
- Rerun: `.venv/Scripts/python experiments/20261118-002-env-compute/bench.py` (add `--quick` for N=2e5).

Machine: 14 logical / 12 physical cores (hybrid P/E design), Intel iGPU. torch defaults to 12 threads.

## Free parameters
N/A.

## Results (raw output in `bench.log`)

MLP training, steps/s (higher is better):

| threads | torch CPU build | torch XPU build, CPU device |
|---|---|---|
| 1 | 9.5 | 4.6 |
| 2 | 23.4 | 5.3 |
| 4 | 17.0 | 6.7 |
| 8 | 24.6 | 10.0 |
| XPU device | - | did not finish 60 steps in >4 min |

Batched 3x3 complex128, N=1e6, seconds (lower is better), CPU build:

| threads | np matmul | torch matmul | np eigh | torch eigh | np svd | torch svd |
|---|---|---|---|---|---|---|
| 1 | 0.245 | 0.133 | 2.84 | 2.63 | 4.98 | 5.91 |
| 4 | 0.269 | 0.037 | 2.37 | 3.49 | 6.29 | 6.16 |
| 8 | 0.413 | 0.023 | 3.04 | 4.12 | 8.27 | 10.87 |

complex64 does not help: torch eigh c64 2.31 s, svd c64 6.01 s at 1 thread.

XPU device: `torch.xpu.is_available() = True`, device "Intel(R) Graphics". A 4096x256 @ 256x256 matmul runs in 0.6 ms once warm. But every new kernel JIT-compiles on first use: an elementwise multiply took 86 s on its first call, `sum` and `mse_loss` did not finish within 2 min. So one MLP training step (dozens of distinct kernels) costs many minutes of compile before any work. Untested possible remedy: `SYCL_CACHE_PERSISTENT=1` to cache compiled kernels on disk across runs. Not worth pursuing given the CPU numbers below.

## Verdict
- **Device: CPU, torch CPU build.** XPU is unusable for us as-is (multi-minute JIT per new op). The XPU build's CPU path is also 2-4x slower than the CPU build, so do not install it in the team venv.
- **MLP emulator training: 2 threads per job.** 2 threads gives ~the same throughput as 8 (23 vs 25 steps/s), 4 threads was slower than 2 (likely hybrid cores: extra threads land on efficiency cores and the slowest thread gates each op; also noisy, see Method). Run up to ~5-6 such jobs in parallel instead of one wide job, keeping the 12-core total. Set `torch.set_num_threads(2)` explicitly; the default of 12 oversubscribes when several jobs run.
- **Batched 3x3 eigh/SVD: 1 thread, parallelize across processes.** These do not get faster with threads (they get slower). Use `threadpoolctl.threadpool_limits(1)` + `torch.set_num_threads(1)` and split the batch across a `multiprocessing` pool. numpy and torch are within ~15% of each other; use whichever the pipeline already holds. torch is 2-10x faster for batched matmul.
- Throughput reference: ~3.5e5 eigh/s and ~2e5 SVD/s per core on 3x3 complex.

## Installed (pinned in `requirements.txt`)
numpy 2.5.5, scipy 1.18.1, pytest 9.2.0 (pre-existing, untouched), torch 2.15.0+cpu, sympy 1.14.0, matplotlib 3.11.2, pandas 3.1.1, scikit-learn 1.9.1, tqdm 4.70.1, threadpoolctl 3.8.0, zuko 1.6.0, sbi 0.27.0 (pulls nflows 0.14 and tensorboard), pysr 2.7.2, juliacall 0.9.36. All install as wheels on Python 3.14 and import cleanly. sbi NPE and zuko NSF smoke tests pass.

PySR works: on first import juliacall downloads Julia 1.13.4 plus SymbolicRegression.jl into `.venv/julia_env` (~1.1 GB, ~6 min, one time). Smoke test recovered `x0*x0 + cos(x1)` exactly. Warm start + 5 iterations: ~40 s, mostly Julia load. PySR writes `outputs/` and sbi writes `sbi-logs/` into the current directory by default: set `output_directory=` (PySR) and a `summary_writer`/log dir (sbi) under `artifacts/` or the experiment folder, or run from the experiment folder.

## Failures
- torch XPU: available but unusable (see above). Not installed in the team venv.
- Nothing failed to install.
