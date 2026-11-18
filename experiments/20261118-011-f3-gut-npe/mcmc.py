"""Vectorised differential-evolution MCMC (ter Braak 2006) for p(theta | data) in the SO(10) variants.

target 'ql':  likelihood from the 12 quark + charged-lepton observables only (PMNS held out)
target 'all': all 17 observables (NO data vector)
Uniform prior on the gen.prior_box; phase coordinates are periodic. Chains start from the multi-start
least-squares solutions of fit_ms.py (artifacts/f3/fits_<variant>_<target>.npz), so the relative weight
of disconnected modes is set by the starting points (chains do not hop between distant modes): stated in README.
Usage: python mcmc.py <variant> <target> <n_iter> <seed>
Writes artifacts/f3/mcmc_<variant>_<target>.npz (thinned samples of the second half, z, io, loglike).
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "f3"
REF = gen.gut_reference()
Z0, S0 = np.array(REF["z"]), np.array(REF["sigma"])


def periodic_dims(variant):
    lo, hi = gen.prior_box(variant, REF)
    return np.isclose(hi - lo, 2 * np.pi) & np.isclose(lo, 0)


def loglike(th, variant, idx):
    z, io, ok = gen.simulate(th, variant)
    d = z - Z0
    for j in gen.Z_PHASES:
        d[:, j] = (d[:, j] + np.pi) % (2 * np.pi) - np.pi
    ll = -0.5 * ((d[:, idx] / S0[idx]) ** 2).sum(1)
    if 15 in idx:
        ll = np.where(io == 1, -np.inf, ll)   # NO data vector
    return np.where(ok, ll, -np.inf), z, io


def run(variant, target, n_iter, seed, n_chains=192, thin=50):
    rng = np.random.default_rng(seed)
    idx = list(range(12)) if target == "ql" else list(range(17))
    lo, hi = gen.prior_box(variant, REF)
    per = periodic_dims(variant)
    if target == "all":
        # start from the ql posterior samples with the best full chi2 (the full-data region lies on the ql manifold)
        q = np.load(ART / f"mcmc_{variant}_ql.npz")
        d = q["z"] - Z0
        for j in gen.Z_PHASES:
            d[:, j] = (d[:, j] + np.pi) % (2 * np.pi) - np.pi
        c2 = ((d / S0) ** 2).sum(1) + 1e6 * q["io"]
        good = q["theta"][np.argsort(c2)[:max(n_chains, 1)]]
    else:
        f = np.load(ART / f"fits_{variant}_{target}.npz")
        good = f["theta"][f["chi2"] < max(2 * len(idx), np.sort(f["chi2"])[min(10, len(f['chi2']) - 1)])]
    x = good[rng.integers(len(good), size=n_chains)].copy()
    x += 1e-4 * (hi - lo) * rng.standard_normal(x.shape)
    x = np.where(per, np.mod(x, 2 * np.pi), np.clip(x, lo, hi))
    ll, z, io = loglike(x, variant, idx)
    d = x.shape[1]
    g0 = 2.38 / np.sqrt(2 * d)
    keep_th, keep_z, keep_io, keep_ll = [], [], [], []
    acc = 0
    t0 = time.time()
    for it in range(n_iter):
        a = rng.integers(n_chains, size=n_chains)
        b = (a + 1 + rng.integers(n_chains - 1, size=n_chains)) % n_chains
        gam = np.where(rng.random(n_chains) < 0.1, 1.0, g0 * np.exp(0.5 * rng.standard_normal(n_chains)))[:, None]
        diff = x[a] - x[b]
        diff = np.where(per, (diff + np.pi) % (2 * np.pi) - np.pi, diff)
        # random subspace: update a random subset of coordinates (helps on narrow curved ridges)
        mask = rng.random(x.shape) < np.where(rng.random((n_chains, 1)) < 0.5, 1.0, 0.3)
        prop = x + mask * (gam * diff + 1e-7 * (hi - lo) * rng.standard_normal(x.shape))
        prop = np.where(per, np.mod(prop, 2 * np.pi), prop)
        inbox = np.all((prop >= lo) & (prop <= hi), 1)
        llp = np.full(n_chains, -np.inf)
        zp = z.copy()
        iop = io.copy()
        if inbox.any():
            l_, z_, io_ = loglike(prop[inbox], variant, idx)
            llp[inbox], zp[inbox], iop[inbox] = l_, z_, io_
        accept = np.log(rng.random(n_chains)) < llp - ll
        x[accept], ll[accept], z[accept], io[accept] = prop[accept], llp[accept], zp[accept], iop[accept]
        acc += accept.mean()
        if it >= n_iter // 2 and it % thin == 0:
            keep_th.append(x.copy()); keep_z.append(z.copy()); keep_io.append(io.copy()); keep_ll.append(ll.copy())
        if (it + 1) % max(1, n_iter // 10) == 0:
            print(f"  {variant} {target} it {it + 1}/{n_iter} acc {acc / (it + 1):.3f} "
                  f"median -2lnL {np.median(-2 * ll):.2f} min {np.min(-2 * ll):.2f} t={time.time() - t0:.0f}s", flush=True)
    out = dict(theta=np.concatenate(keep_th), z=np.concatenate(keep_z), io=np.concatenate(keep_io),
               loglike=np.concatenate(keep_ll), chain=np.tile(np.arange(n_chains), len(keep_th)))
    np.savez(ART / f"mcmc_{variant}_{target}.npz", **out)
    return out


if __name__ == "__main__":
    v, t, n, s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    o = run(v, t, n, s)
    print(f"{v} {t}: saved {len(o['theta'])} samples")
