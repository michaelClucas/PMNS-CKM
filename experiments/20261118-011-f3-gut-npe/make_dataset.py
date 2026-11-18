"""Generate F3 datasets into artifacts/f3/ (gitignored). Noiseless z is stored; noise is added at training time.

  python make_dataset.py broad <variant> <n> <seed>
      theta ~ uniform prior box (gen.prior_box), n samples in chunks of 1e5 (seed + chunk index), Pool(4).
  python make_dataset.py local <variant> <n> <seed> [bandwidth]
      theta ~ p~ = Gaussian KDE around the DE-MC posterior p(theta | quark + charged-lepton data)
      (artifacts/f3/mcmc_<variant>_ql.npz, 4000 centres, per-dimension bandwidth = bw * posterior std,
      phases wrapped), truncated to the prior box. Also stores log p~(theta) for importance reweighting.
"""
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "f3"
CHUNK = 100_000


def _broad_chunk(args):
    variant, n, seed = args
    rng = np.random.default_rng(seed)
    th = gen.sample_prior(variant, n, rng)
    z, io, ok = gen.simulate(th, variant)
    return th.astype(np.float32), z.astype(np.float32), io, ok


def kde_spec(variant, n_centres=4000, bw=1.0, seed=0):
    rng = np.random.default_rng(seed)
    m = np.load(ART / f"mcmc_{variant}_ql.npz")
    th = m["theta"]
    lo, hi = gen.prior_box(variant)
    per = np.isclose(hi - lo, 2 * np.pi) & np.isclose(lo, 0)
    c = th[rng.choice(len(th), size=min(n_centres, len(th)), replace=False)]
    # circular std for phases
    sd = np.where(per, np.sqrt(-2 * np.log(np.clip(np.abs(np.exp(1j * th).mean(0)), 1e-12, 1))), th.std(0))
    sd = np.minimum(sd, (hi - lo) / 4)
    return c, bw * sd, per, lo, hi


def kde_logpdf(x, c, h, per, chunk=2000):
    """log of the untruncated product-Gaussian KDE (phases: wrapped differences). Truncation to the box
    multiplies p~ by a theta-independent constant only approximately; the README states this."""
    out = np.empty(len(x))
    lognorm = -np.log(h).sum() - 0.5 * len(h) * np.log(2 * np.pi) - np.log(len(c))
    for i in range(0, len(x), chunk):
        d = x[i:i + chunk, None, :] - c[None]
        d = np.where(per, (d + np.pi) % (2 * np.pi) - np.pi, d)
        e = -0.5 * ((d / h) ** 2).sum(-1)
        mx = e.max(1, keepdims=True)
        out[i:i + chunk] = (mx[:, 0] + np.log(np.exp(e - mx).sum(1))) + lognorm
    return out


def _local_chunk(args):
    variant, n, seed, bw = args
    rng = np.random.default_rng(seed)
    c, h, per, lo, hi = kde_spec(variant, bw=bw)
    th = np.empty((0, len(lo)))
    while len(th) < n:
        x = c[rng.integers(len(c), size=2 * n)] + h * rng.standard_normal((2 * n, len(lo)))
        x = np.where(per, np.mod(x, 2 * np.pi), x)
        x = x[np.all((x >= lo) & (x <= hi), 1)]
        th = np.concatenate([th, x])[:n]
    lq = kde_logpdf(th, c, h, per)
    z, io, ok = gen.simulate(th, variant)
    return th.astype(np.float32), z.astype(np.float32), io, ok, lq


def main():
    mode, variant, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    bw = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nch = (n + CHUNK - 1) // CHUNK
    jobs = [(variant, min(CHUNK, n - i * CHUNK), seed * 1000 + i) for i in range(nch)]
    with Pool(4) as pool:
        if mode == "broad":
            res = pool.map(_broad_chunk, jobs)
        else:
            res = pool.map(_local_chunk, [j + (bw,) for j in jobs])
    out = dict(theta=np.concatenate([r[0] for r in res]), z=np.concatenate([r[1] for r in res]),
               io=np.concatenate([r[2] for r in res]), ok=np.concatenate([r[3] for r in res]))
    if mode == "local":
        out["logq"] = np.concatenate([r[4] for r in res])
    path = ART / f"ds_{mode}_{variant}.npz"
    np.savez(path, **out)
    summ = {"mode": mode, "variant": variant, "n": int(n), "seed": seed, "bandwidth": bw if mode == "local" else None,
            "chunk_seeds": [j[2] for j in jobs], "frac_ok": float(out["ok"].mean()), "frac_IO": float(out["io"].mean()),
            "seconds": round(time.time() - t0, 1),
            "z_quantiles_5_50_95": {nm: [round(float(q), 4) for q in np.quantile(out["z"][:, j], [0.05, 0.5, 0.95])]
                                    for j, nm in enumerate(gen.Z_NAMES)}}
    sp = HERE / "dataset_summary.json"
    allsum = json.loads(sp.read_text()) if sp.exists() else {}
    allsum[f"{mode}_{variant}"] = summ
    sp.write_text(json.dumps(allsum, indent=1))
    print(f"{mode} {variant}: {n} samples in {summ['seconds']}s, ok {summ['frac_ok']:.4f}, IO {summ['frac_IO']:.4f}")


if __name__ == "__main__":
    main()
