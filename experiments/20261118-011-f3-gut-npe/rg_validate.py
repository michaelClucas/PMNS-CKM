"""Validation subset: run generated GUT-scale samples down to M_Z with src/rg.py (SM) and fit a linear running map.

  python rg_validate.py <variant> <source npz> <n>
For each sample: Weyl matrices from gen -> rg.py Yukawas Y = M^dagger, normalised so y_b(M_GUT) equals the value
obtained by running the real data up (src.rg); Y_e and Y_u share that normalisation (their ratios are physical
in the model); kappa normalised to a heaviest light mass of 0.05 eV. Gauge couplings and lambda at M_GUT come
from the real-data state run up. Then rg.run(2e16 -> M_Z), observables -> z_MZ.
Linear map z_MZ = a + B z_GUT (B fitted per output on the non-phase inputs, ridge 1e-8) on half of the samples,
residuals on the other half reported in units of the data sigma. Writes rg_map.json.
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
from src import rg  # noqa: E402

MU = 2e16
REF = gen.gut_reference()


def gut_env():
    s = rg.build_mz_state("NO", 0.0)
    up = rg.run_sm(s["mu"], MU, s)
    yb = np.linalg.svd(up["Yd"], compute_uv=False)[0]
    return up["g"], up["lam"], yb, s["mu"]


def z_from_obs(o, ordering):
    y, c, p = o["y"], o["ckm"], o["pmns"]
    m = p["m"]
    if ordering == "NO":
        r = (m[1] ** 2 - m[0] ** 2) / (m[2] ** 2 - m[0] ** 2)
    else:
        r = (m[1] ** 2 - m[0] ** 2) / (m[1] ** 2 - m[2] ** 2)
    return np.array([np.log10(y["y_u"] / y["y_c"]), np.log10(y["y_c"] / y["y_t"]), np.log10(y["y_d"] / y["y_s"]),
                     np.log10(y["y_s"] / y["y_b"]), np.log10(y["y_b"] / y["y_t"]), np.log10(y["y_e"] / y["y_mu"]),
                     np.log10(y["y_mu"] / y["y_tau"]), np.log10(y["y_tau"] / y["y_b"]),
                     np.sin(c["theta12"]), np.log10(np.sin(c["theta13"])), np.log10(np.sin(c["theta23"])), c["delta"],
                     np.sin(p["theta12"]) ** 2, np.sin(p["theta13"]) ** 2, np.sin(p["theta23"]) ** 2, p["delta"],
                     np.log10(r)])


def one(args):
    th, variant, env = args
    g, lam, yb, mz = env
    Mu, Md, Me, Mn = [m[0] for m in gen.mats(th[None], variant)]
    zg, io, ok = gen.observables_from_mats(Mu[None], Md[None], Me[None], Mn[None])
    sc = yb / np.linalg.svd(Md, compute_uv=False)[0]
    v = rg.load_sm_inputs()["ahs2025_msbar_MZ"]["v_MZ"]["value"]
    kap = Mn / np.linalg.svd(Mn, compute_uv=False)[0] * 4 / v ** 2 * 0.05 * rg.EV
    st = {"g": g.copy(), "Yu": Mu.conj().T * sc, "Yd": Md.conj().T * sc, "Ye": Me.conj().T * sc, "lam": lam,
          "kappa": kap, "kscale": 4 / v ** 2 * 0.05 * rg.EV, "mu": MU, "normalisation": "SM",
          "ordering": "IO" if io[0] else "NO"}
    try:
        down = rg.run_sm(MU, mz, st, rtol=1e-8)
        return zg[0], z_from_obs(rg.observables(down), st["ordering"]), int(io[0])
    except Exception:
        return zg[0], np.full(17, np.nan), int(io[0])


def main():
    variant, src, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    th = np.load(HERE.parents[1] / "artifacts" / "f3" / src)["theta"]
    th = th[np.random.default_rng(0).choice(len(th), size=min(n, len(th)), replace=False)].astype(float)
    env = gut_env()
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(one, [(t, variant, env) for t in th], chunksize=4)
    zg = np.array([r[0] for r in res]); zm = np.array([r[1] for r in res]); io = np.array([r[2] for r in res])
    good = np.all(np.isfinite(zm), 1)
    zg, zm, io = zg[good], zm[good], io[good]
    dt = (time.time() - t0) / len(th)
    S = np.array(REF["sigma"])
    d = zm - zg
    for j in gen.Z_PHASES:
        d[:, j] = (d[:, j] + np.pi) % (2 * np.pi) - np.pi
    # linear map on shifts: d = a + B (zg - mean), non-phase inputs
    lin = [j for j in range(17) if j not in gen.Z_PHASES]
    h = len(zg) // 2
    X = np.concatenate([np.ones((len(zg), 1)), zg[:, lin] - zg[:h, lin].mean(0)], 1)
    coef = np.linalg.solve(X[:h].T @ X[:h] + 1e-8 * np.eye(X.shape[1]), X[:h].T @ d[:h])
    resid_lin = (d[h:] - X[h:] @ coef) / S
    resid_const = (d[h:] - d[:h].mean(0)) / S
    out = {"variant": variant, "source": src, "n_run": int(len(zg)), "sec_per_sample": round(dt, 3),
           "mean_shift_GUT_to_MZ": dict(zip(gen.Z_NAMES, np.round(d.mean(0), 5).tolist())),
           "sd_shift_over_sigma": dict(zip(gen.Z_NAMES, np.round(d.std(0) / S, 3).tolist())),
           "const_map_rms_resid_over_sigma": dict(zip(gen.Z_NAMES, np.round(np.sqrt((resid_const ** 2).mean(0)), 3).tolist())),
           "linear_map_rms_resid_over_sigma": dict(zip(gen.Z_NAMES, np.round(np.sqrt((resid_lin ** 2).mean(0)), 3).tolist())),
           "linear_map_max_resid_over_sigma": dict(zip(gen.Z_NAMES, np.round(np.abs(resid_lin).max(0), 3).tolist())),
           "frac_IO": float(io.mean())}
    (HERE / f"rg_map_{variant}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
