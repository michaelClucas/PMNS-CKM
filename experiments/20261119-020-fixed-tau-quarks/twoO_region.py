"""2O lepton island (exp 016 model, copied fit_ours.py / model_2o.py / modforms_2o.py): lepton-only tau
region (gE2/gE1, gE3/gE1 profiled; exp 016 conventions: GUT scale, MSSM tan beta 5, 2% mass floor),
then the 2O model's own quark sector (6 coupling ratios) fitted at fixed tau inside that region.

    python twoO_region.py      -> twoO_region.json, artifacts/fixedtau/twoO_*.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, "..", ".."))]

import numpy as np  # noqa: E402

WORKERS = int(os.environ.get("FT_WORKERS", "5"))
H = 0.001
_C = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def C():
    if "c" not in _C:
        import fit_ours as F
        _C["c"] = F.Chi2(F.build_data("ours"))
    return _C["c"]


def _lep(args):
    import fit_ours as F
    tau, warm = args
    starts = [np.asarray(warm)] + [np.array([F.BASE["rE2"], F.BASE["rE3"]])]
    r = F.best_lsq(F.resid_lep, starts, (tau, C()))
    return tau, 2 * float(r.cost), r.x.tolist()


def _qrk(args):
    import fit_ours as F
    tau, seed = args
    c, x = F.qrk_profile(tau, C(), np.random.default_rng(seed), n_rand=4)
    return tau, float(c), list(map(float, x))


def flood(pool, tau0, warm0, thr):
    done = {}
    frontier = {(round(tau0.real / H), round(tau0.imag / H)): warm0}
    while frontier:
        keys = list(frontier)
        res = pool.map(_lep, [(complex(i * H, j * H), frontier[(i, j)]) for i, j in keys])
        new = {}
        for k, (t, c, x) in zip(keys, res):
            done[k] = (t, c, x)
        for k, (t, c, x) in zip(keys, res):
            if c <= thr:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        kk = (k[0] + di, k[1] + dj)
                        if kk not in done and kk not in new:
                            new[kk] = x
        frontier = new
        if len(done) > 8000:
            break
    return done


if __name__ == "__main__":
    import multiprocessing as mp
    from scipy.stats import chi2 as chi2d
    lo = json.load(open(os.path.join(HERE, "..", "20261118-016-modular-literature", "lepton_only_ours.json")))
    t0 = complex(lo["x"][0], lo["x"][1])
    w0 = lo["x"][2:]
    out = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        _, cmin, _ = _lep((t0, w0))
        log("2O lepton min (exp 016)", t0, cmin)
        pts = {}
        for start in (t0, -np.conj(t0)):
            d = flood(pool, start, w0, cmin + 9.0)
            pts.update(d)
            log("flood from", start, len(d), "points; min", min(v[1] for v in d.values()))
        tau = np.array([v[0] for v in pts.values()])
        chi = np.array([v[1] for v in pts.values()])
        gmin = float(chi.min())
        np.savez(os.path.join(HERE, "..", "..", "artifacts", "fixedtau", "twoO_lepton.npz"), tau=tau, chi2=chi, gmin=gmin)
        rec = {"lepton_chi2_min": gmin, "tau_best": [float(tau[chi.argmin()].real), float(tau[chi.argmin()].imag)],
               "N_lep": 7, "k_lep": 4, "p_lepton": float(chi2d.sf(gmin, 3)), "grid_step": H}
        for lvl in (2.30, 6.18):
            m = chi <= gmin + lvl
            rec[f"n_{lvl}"] = int(m.sum())
            rec[f"area_{lvl}"] = float(m.sum() * H * H)
            for side, sel in (("neg", tau.real < 0), ("pos", tau.real > 0)):
                mm = m & sel
                if mm.any():
                    rec[f"box_{lvl}_{side}"] = {"re": [float(tau[mm].real.min()), float(tau[mm].real.max())],
                                                "im": [float(tau[mm].imag.min()), float(tau[mm].imag.max())],
                                                "chi2_min": float(chi[mm].min())}
        log("2O lepton region", rec)
        m6 = chi <= gmin + 6.18
        qt = tau[m6]
        res = pool.map(_qrk, [(t, i) for i, t in enumerate(qt)])
        qc = np.array([r[1] for r in res])
        j = int(np.argmin(qc))
        rec["quark"] = {"n_tau_points": int(len(qt)), "chi2_q_min": float(qc[j]),
                        "tau": [float(qt[j].real), float(qt[j].imag)], "x": res[j][2],
                        "lepton_dchi2_at_tau": float(chi[m6][j] - gmin),
                        "chi2_q_median_over_region": float(np.median(qc)),
                        "N_q": 8, "k_q": 6, "p_q": float(chi2d.sf(qc[j], 2))}
        m2 = chi[m6] <= gmin + 2.30
        if m2.any():
            rec["quark"]["chi2_q_min_in_2.30"] = float(qc[m2].min())
        # pulls at the best point
        import fit_ours as F
        p = dict(F.BASE, re=qt[j].real, im=qt[j].imag, **dict(zip(F.QRK_N, res[j][2])))
        o = F.observables(p)
        pred, mass, _, cc = C().parts(o)
        from src.evaluate import Scorer
        ev = Scorer(F.CKM_USE, "NO")(pred)
        rec["quark"]["pulls"] = {**{k: float(mass[k]) for k in ("r_uc", "r_ct", "r_ds", "r_sb")},
                                 **{k: float(v) for k, v in ev.pulls.items()}}
        np.savez(os.path.join(HERE, "..", "..", "artifacts", "fixedtau", "twoO_quark.npz"), tau=qt, chi2=qc)
        log("2O quark at fixed tau", rec["quark"])
    json.dump(rec, open(os.path.join(HERE, "twoO_region.json"), "w"), indent=1)
    log("done")
