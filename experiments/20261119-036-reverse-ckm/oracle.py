"""Exp 036 oracle power: how much PMNS -> CKM information is detectable under the N2 null at today's precision.

Usage: python oracle.py  (writes oracle.json, oracle.log; see prereg.md "Oracle power")
No networks and no real-data statistic. Measured inputs: NuFIT NO tables and best fit via src.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src import data as sd, evaluate as ev, nulls  # noqa: E402

KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
N_OFF, N_FAKES, N_WORLDS, K_PRIOR = 512, 2000, 30, 20000  # 50 -> 30 worlds: deviation 1
LEVELS = (1, 2, 4, 6, 8, 12)  # 16 -> 6: deviation 1 (README)
LOG = HERE / "oracle.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def box():
    r = sd.load_pmns("NO")
    lo, hi = [], []
    for k in KEYS[:3]:
        a, b = r[k]["range_3sigma"]
        mid, half = (a + b) / 2, nulls.N2_WIDEN * (b - a) / 2
        lo.append(max(mid - half, 1e-6))
        hi.append(min(mid + half, 1 - 1e-6))
    return np.array(lo + [0.0]), np.array(hi + [360.0])


LO, HI = box()
T_BEST = np.array([ev.pmns_best("NO")[k] for k in KEYS])


def centres(rng, n):
    """N2 centres; same draws as src.nulls.n2_centre (checked in main)."""
    return np.array([[c[k] for k in KEYS] for c in (nulls.n2_centre("NO", rng) for _ in range(n))])


def offsets(rng, n):
    return np.array([[o[k] for k in KEYS] for o in (nulls.pmns_noise("NO", rng) for _ in range(n))])


# ---------------------------------------------------------------- exact NuFIT NO delta chi^2 (vectorised)
_P12 = sd._shifted_profile_1d(sd.DEFAULT_ANALYSIS, "NO", "T12")
_P13 = sd._shifted_profile_1d(sd.DEFAULT_ANALYSIS, "NO", "T13")
_G = sd._shifted_grid_2d(sd.DEFAULT_ANALYSIS, "NO", "T23-DCP")


def _interp1(p, x):
    xs, ys = p
    out = np.interp(x, xs, ys)
    return np.where((x < xs[0]) | (x > xs[-1]), np.inf, out)


def _interp2(x, d):
    from scipy.interpolate import RegularGridInterpolator
    xs, ds, z = _G
    f = RegularGridInterpolator((xs, ds), z, bounds_error=False, fill_value=np.inf)
    d = (d + 180.0) % 360.0 - 180.0
    return f(np.stack([x, d], -1))


def dchi2(pt):
    """Delta chi^2 at real-table points pt (..., 4) = (s12sq, s13sq, s23sq, dcp_deg)."""
    return _interp1(_P12, pt[..., 0]) + _interp1(_P13, pt[..., 1]) + _interp2(pt[..., 2], pt[..., 3])


# ---------------------------------------------------------------- link family
def feats(x):
    """x (..., 4) physical -> (..., 5): s^2 scaled to [-1, 1] over B, cos and sin of delta."""
    s = 2 * (x[..., :3] - LO[:3]) / (HI[:3] - LO[:3]) - 1
    d = np.radians(x[..., 3])
    return np.concatenate([s, np.cos(d)[..., None], np.sin(d)[..., None]], -1)


class Link:
    def __init__(self, rng, prior_x):
        self.w = rng.standard_normal((4, 8))
        self.om = 1.5 * rng.standard_normal((4, 8, 5))
        self.ph = rng.uniform(0, 2 * np.pi, (4, 8))
        g = self.raw(prior_x)
        self.mu, self.sd = g.mean(0), g.std(0)

    def raw(self, x):
        f = feats(x)
        arg = np.einsum("...i,kmi->...km", f, self.om) + self.ph
        return np.einsum("...km,km->...k", np.cos(arg), self.w)

    def __call__(self, x):
        return (self.raw(x) - self.mu) / self.sd


def lse_mean(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def lognorm(u, mu):
    return -0.5 * np.sum((u - mu) ** 2, -1) - 2 * np.log(2 * np.pi)


def info(a, link, x, eps, gk, post_g=None):
    """I(u; x) (bits) for u = a g(x) + eps; with post_g (n, J, 4) also I(u; yhat)."""
    u = a * link(x) + eps
    lp_cond = lognorm(u, a * link(x))
    # marginal from K prior points plus the true x (bounded estimator, cap log2(K + 1) bits)
    lp_marg = np.concatenate([lse_mean(np.concatenate([lognorm(u[i:i + 50, None], a * gk[None]), lp_cond[i:i + 50, None]], 1), 1)
                              for i in range(0, len(u), 50)])
    I = float(np.mean(lp_cond - lp_marg) / np.log(2))
    if post_g is None:
        return I
    lp_post = lse_mean(lognorm(u[:, None], a * post_g), 1)
    return I, float(np.mean(lp_post - lp_marg) / np.log(2))


def sample_box(rng, n):
    return LO + (HI - LO) * rng.uniform(size=(n, 4))


def main():
    t0 = time.time()
    # N2 centre check against src.nulls
    c1 = centres(np.random.default_rng([36, 2, 0]), 1)[0]
    log(f"box lo {LO.tolist()} hi {HI.tolist()}; N2 centre check {c1.tolist()}")
    O = offsets(np.random.default_rng(3611), N_OFF)  # posterior offsets for the oracle statistic
    out = {"box_lo": LO.tolist(), "box_hi": HI.tolist(), "n_off": N_OFF, "n_fakes": N_FAKES}

    # dchi2 check against src.data.pmns_delta_chi2
    rng = np.random.default_rng(1)
    pts = T_BEST + 0.5 * np.array([0.02, 0.0015, 0.1, 60]) * rng.standard_normal((5, 4))
    mine = dchi2(pts)
    ref = [sd.pmns_delta_chi2("NO", relative_to="local", **dict(zip(KEYS, map(float, p)))) for p in pts]
    log(f"dchi2 check: max |diff| {np.max(np.abs(mine - np.array(ref))):.2e} ({mine.round(3).tolist()} vs {np.round(ref, 3).tolist()})")

    # ---- information ceiling I_max = log2 vol(B) - h(o)
    def h1(p):
        xs = np.linspace(p[0][0], p[0][-1], 20001)
        w = np.exp(-0.5 * np.interp(xs, *p))
        dx = xs[1] - xs[0]
        q = w / (w.sum() * dx)
        return float(-np.sum(q * np.log(np.maximum(q, 1e-300))) * dx)

    xs = np.linspace(_G[0][0], _G[0][-1], 801)
    ds = np.linspace(-180, 180, 1441)
    X, Dd = np.meshgrid(xs, ds, indexing="ij")
    w = np.exp(-0.5 * _interp2(X, Dd))
    cell = (xs[1] - xs[0]) * (ds[1] - ds[0])
    q = w / (w.sum() * cell)
    h2 = float(-np.sum(q * np.log(np.maximum(q, 1e-300))) * cell)
    hs = {"s12sq": h1(_P12), "s13sq": h1(_P13), "s23sq_dcp": h2}
    vol = np.log(HI - LO)
    I_max = float((vol.sum() - sum(hs.values())) / np.log(2))
    per = {"s12sq": (vol[0] - hs["s12sq"]) / np.log(2), "s13sq": (vol[1] - hs["s13sq"]) / np.log(2),
           "s23sq_dcp": (vol[2] + vol[3] - hs["s23sq_dcp"]) / np.log(2)}
    log(f"I_max = I(x; yhat) under U(B) = {I_max:.2f} bits; per block {{{', '.join(f'{k}: {v:.2f}' for k, v in per.items())}}}")
    log(f"needed rank < {nulls.max_hits_to_pass(N_FAKES) + 1}/{N_FAKES}: -log2 = {-np.log2((nulls.max_hits_to_pass(N_FAKES) + 1) / N_FAKES):.2f} bits")
    out["I_max_bits"] = I_max
    out["I_max_blocks"] = per

    # ---- perfect link ceiling
    # fakes are shared across worlds here (2000 N2 centres, rng [36, 2, i] as the real test)
    F = centres_cached()
    out["perfect"] = {}
    for tag, n in (("uniform", 1000), ("bestfit", 200)):
        rng = np.random.default_rng([36, 5, 0 if tag == "uniform" else 1])
        xt = sample_box(rng, n) if tag == "uniform" else np.tile(T_BEST, (n, 1))
        o = offsets(rng, n)
        T_w = -0.5 * dchi2(xt - (xt + o) + T_BEST)
        ks = []
        for i in range(n):
            T_f = -0.5 * dchi2(xt[i] - F + T_BEST)
            ks.append(int(np.sum(T_f >= T_w[i])))
        ks = np.array(ks)
        det = int(np.sum([nulls.decide(int(k), N_FAKES)["passed"] for k in ks]))
        lo_, hi_ = cp2(det, n)
        out["perfect"][tag] = {"n": n, "detected": det, "rate": det / n, "cp95": [lo_, hi_],
                               "rank_q10_50_90": np.quantile(ks / N_FAKES, [0.1, 0.5, 0.9]).tolist()}
        log(f"perfect link, truth {tag}: detected {det}/{n} = {det / n:.3f} CP95 [{lo_:.3f}, {hi_:.3f}]; rank q10/50/90 "
            f"{np.round(np.quantile(ks / N_FAKES, [0.1, 0.5, 0.9]), 4).tolist()}")

    # ---- smooth links at fixed information levels
    out["levels"] = []
    rng_p = np.random.default_rng([36, 6, 0])
    prior_x = sample_box(rng_p, K_PRIOR)
    xi = sample_box(rng_p, 600)
    eps = rng_p.standard_normal((600, 4))
    oi = offsets(rng_p, 600)
    for L in LEVELS:
        dets, Is, Ims, ranks = [], [], [], []
        for wi in range(N_WORLDS):
            rng = np.random.default_rng([36, 8, L, wi])
            link = Link(rng, prior_x)
            gk = link(prior_x)
            gb = gk[:5000] if L < 8 else gk  # bisection prior subset (cap 12.3 bits below level 8)
            f = lambda a: info(a, link, xi[:100], eps[:100], gb) - L
            lo_a, hi_a = 0.0, 1.0
            while f(hi_a) < 0:
                hi_a *= 2
                if hi_a > 1e6:
                    break
            for _ in range(16):
                mid = 0.5 * (lo_a + hi_a)
                lo_a, hi_a = (mid, hi_a) if f(mid) < 0 else (lo_a, mid)
            a = 0.5 * (lo_a + hi_a)
            yh_i = xi[:200] + oi[:200]
            post_g = link(yh_i[:, None, :] - O[None, :, :])
            I, Im = info(a, link, xi[:200], eps[:200], gk, post_g)
            # world
            x = sample_box(rng, 1)[0]
            yhat = x + offsets(rng, 1)[0]
            u = a * link(x) + rng.standard_normal(4)

            def T(yc):
                return lse_mean(lognorm(u, a * link(yc[:, None, :] - O[None])), 1)

            Tw = T(yhat[None])[0]
            k, n_ev = 0, 0
            for b in range(0, N_FAKES, 200):
                k += int(np.sum(T(F[b:b + 200]) >= Tw))
                n_ev = b + 200
                if k > nulls.max_hits_to_pass(N_FAKES):
                    break
            d = nulls.decide(k, N_FAKES)["passed"] if n_ev == N_FAKES or k > nulls.max_hits_to_pass(N_FAKES) else False
            dets.append(bool(d))
            Is.append(I)
            Ims.append(Im)
            ranks.append(k / n_ev)
        det = int(sum(dets))
        lo_, hi_ = cp2(det, N_WORLDS)
        row = {"level": L, "worlds": N_WORLDS, "I_link_median": float(np.median(Is)), "I_meas_median": float(np.median(Ims)),
               "I_meas_q10_q90": np.quantile(Ims, [0.1, 0.9]).tolist(), "detected": det, "cp95": [lo_, hi_],
               "rank_median": float(np.median(ranks))}
        out["levels"].append(row)
        log(f"level {L} bits: I_link med {row['I_link_median']:.2f}, I_meas med {row['I_meas_median']:.2f}; oracle detected "
            f"{det}/{N_WORLDS} CP95 [{lo_:.3f}, {hi_:.3f}]; rank median {row['rank_median']:.3f} ({time.time() - t0:.0f}s)")
        json.dump(out, open(HERE / "oracle.json", "w"), indent=1)
    json.dump(out, open(HERE / "oracle.json", "w"), indent=1)
    log(f"done {time.time() - t0:.0f}s")


_F = None


def centres_cached():
    global _F
    if _F is None:
        _F = np.array([[c[k] for k in KEYS] for c in (nulls.n2_centre("NO", np.random.default_rng([36, 2, i]))
                                                      for i in range(N_FAKES))])
    return _F


def cp2(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


if __name__ == "__main__":
    main()
