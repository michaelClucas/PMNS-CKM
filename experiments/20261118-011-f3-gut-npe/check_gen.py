"""Consistency checks of gen.py: (1) conventions agree with src/rg.observables, (2) quark block of z reproduces
theta for the quark-anchored SO(10) parametrisation, (3) Clebsch identities, (4) prior acceptance vs real data."""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src import rg
import gen

rng = np.random.default_rng(0)
ref = gen.gut_reference()
zobs, sig = np.array(ref["z"]), np.array(ref["sigma"])

def wrapdiff(a, b):
    d = a - b
    for j in gen.Z_PHASES:
        d[..., j] = (d[..., j] + np.pi) % (2 * np.pi) - np.pi
    return d

# (1) conventions vs rg.observables on 20 so10_I samples
th = gen.sample_prior("so10_I", 20, rng)
Mu, Md, Me, Mn = gen.mats(th, "so10_I")
z, io, ok = gen.observables_from_mats(Mu, Md, Me, Mn)
mx = 0
for i in range(20):
    sc = 1.0 / np.abs(Mu[i]).max()
    st = {"Yu": Mu[i].conj().T * sc, "Yd": Md[i].conj().T * sc, "Ye": Me[i].conj().T * sc,
          "kappa": Mn[i] / np.abs(Mn[i]).max() * 1e-14, "g": np.ones(3), "ordering": "IO" if io[i] else "NO"}
    o = rg.observables(st)
    a = [o["ckm"]["theta12"], o["ckm"]["theta13"], o["ckm"]["theta23"], o["ckm"]["delta"],
         o["pmns"]["theta12"], o["pmns"]["theta13"], o["pmns"]["theta23"], o["pmns"]["delta"]]
    b = [np.arcsin(z[i, 8]), np.arcsin(10 ** z[i, 9]), np.arcsin(10 ** z[i, 10]), z[i, 11],
         np.arcsin(np.sqrt(z[i, 12])), np.arcsin(np.sqrt(z[i, 13])), np.arcsin(np.sqrt(z[i, 14])), z[i, 15]]
    dd = np.array(a) - np.array(b); dd[[3, 7]] = (dd[[3, 7]] + np.pi) % (2 * np.pi) - np.pi
    mx = max(mx, np.abs(dd).max())
print(f"(1) max |angle diff| vs src/rg.observables over 20 samples: {mx:.2e} rad")

# (2) quark block reproduces theta
th = gen.sample_prior("so10_I", 2000, rng)
z, io, ok = gen.simulate(th, "so10_I")
pred = np.stack([th[:, 0], th[:, 1], th[:, 3], th[:, 4], -th[:, 2], th[:, 5], th[:, 6], th[:, 7], th[:, 8]], 1)
got = z[:, [0, 1, 2, 3, 4, 8, 9, 10, 11]]
d = got - pred; d[:, 8] = (d[:, 8] + np.pi) % (2 * np.pi) - np.pi
print(f"(2) quark block |z - theta| max: {np.abs(d).max():.2e}")

# (3) Clebsch identities: Y_d - Y_l = 4F, Y_u/r - Y_d = (s-1) F ; type II m_nu ∝ Y_d - Y_l
Mu, Md, Me, Mn = gen.mats(th[:5], "so10_II")
print("(3) |Mnu - (Md-Me)/4| max:", np.abs(Mn - (Md - Me) / 4).max())

# (4) acceptance and timing per variant
for v in ["so10_I", "so10_II", "so10_mixed", "so10_120", "su5_min", "su5_gj"]:
    t0 = time.time()
    th = gen.sample_prior(v, 200000, rng)
    z, io, ok = gen.simulate(th, v)
    dt = time.time() - t0
    pulls = wrapdiff(z, zobs[None]) / sig
    c_ql = (pulls[:, :12] ** 2).sum(1)
    c_all = (pulls ** 2).sum(1)
    lep = (pulls[:, 5:8] ** 2).sum(1)
    print(f"(4) {v:11s} {dt:5.1f}s/2e5 ok={ok.mean():.4f} IO={io.mean():.3f} "
          f"chi2_lepmass<9: {(lep<9).mean():.2e}  chi2_ql<25: {(c_ql<25).mean():.2e}  chi2_all<40: {(c_all<40).mean():.2e}  min chi2_all {c_all.min():.1f}")
