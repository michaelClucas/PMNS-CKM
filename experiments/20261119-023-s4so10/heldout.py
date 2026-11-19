"""Step 5: does the quark sector (+ charged leptons) constrain PMNS in this model?

Likelihood: quark + charged-lepton data only, official scorer, use = CKM 4 + mr_u_c, mr_c_t, mr_d_s,
mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau (N 11), same GUT scale / floor / eta_b treatment as fit_ours.py.
Mode "ql": no neutrino data at all. Mode "ql_dm": adds dm_ratio (neutrino masses, not mixing).
Sampler: adaptive Metropolis, 4 chains, in u = log|coupling| (signs fixed to the best full fit's
pattern; phases uniform on [0, 2 pi); eta_b uniform on (-0.95, 1) when free).
Prior: log-uniform over +-2 decades around the best full fit for every |coupling| (incl. mu_1..3).
Output: PMNS posterior-predictive quantiles and the fraction inside NuFIT 6.1 3 sigma (1D, NO).
Usage: python heldout.py <fit json, e.g. fit_ebfree.json> <ql|ql_dm> [n_steps] [n_proc]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from fitlib import HERE, ROOT, PHASES, REAL, USE_QL, predict, scorer
from src import data as _data

STEPS = 60000


def setup(fitfile, mode):
    fit = json.loads((HERE / fitfile).read_text())
    p0 = fit["params"]
    eb_free = fit["variant"] in ("ebfree", "real_ebfree")
    eb_fixed = {"eb0": 0.0, "real_eb0": 0.0, "eb08": -0.8}.get(fit["variant"], None)
    use = USE_QL + (("dm_ratio",) if mode == "ql_dm" else ())
    return fit, p0, eb_free, eb_fixed, use


def chain(args):
    fitfile, mode, seed, steps = args
    fit, p0, eb_free, eb_fixed, use = setup(fitfile, mode)
    sc = scorer(use)
    signs = {k: np.sign(p0[k]) or 1.0 for k in REAL}
    centre = np.array([np.log(abs(p0[k])) for k in REAL])
    rng = np.random.default_rng(1000 + seed)

    def unpack(z):
        p = {k: signs[k] * np.exp(z[i]) for i, k in enumerate(REAL)}
        p.update({k: z[13 + i] % (2 * np.pi) for i, k in enumerate(PHASES)})
        eb = z[19] if eb_free else eb_fixed
        return p, eb

    def logpost(z):
        if np.any(np.abs(z[:13] - centre) > 2 * np.log(10)):
            return -np.inf, None
        p, eb = unpack(z)
        if eb_free and not -0.95 < eb < 1.0:
            return -np.inf, None
        pred, o = predict(p, eb)
        c = sc.chi2(pred)
        return (-0.5 * c if np.isfinite(c) else -np.inf), pred

    z = np.concatenate([centre, [p0[k] for k in PHASES], [fit["eta_b"]] if eb_free else []])
    # start: perturb the neutrino parameters (not constrained by the likelihood) from the prior
    for i, k in enumerate(REAL):
        if k.startswith("mu"):
            z[i] = centre[i] + rng.uniform(-2, 2) * np.log(10)
    z[13] = rng.uniform(0, 2 * np.pi) if seed % 2 else z[13]
    lp, pred = logpost(z)
    d = z.size
    cov = np.diag(np.r_[np.full(13, 0.05), np.full(6, 0.05), [0.01] if eb_free else []] ** 2)
    samples, n_acc = [], 0
    hist = []
    for t in range(steps):
        zp = z + rng.multivariate_normal(np.zeros(d), cov * 2.38 ** 2 / d)
        lpp, predp = logpost(zp)
        if np.log(rng.uniform()) < lpp - lp:
            z, lp, pred = zp, lpp, predp
            n_acc += 1
        hist.append(z.copy())
        if t > 2000 and t % 1000 == 0:
            cov = np.cov(np.array(hist[-20000:]).T) + 1e-8 * np.eye(d)
        if t >= steps // 3 and t % 20 == 0:
            samples.append([pred[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")] + [-2 * lp])
    return seed, n_acc / steps, samples


def main():
    fitfile, mode = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    n_proc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    t0 = time.time()
    with Pool(n_proc) as pool:
        out = pool.map(chain, [(fitfile, mode, s, steps) for s in range(4)])
    S = np.array([row for _, _, s in out for row in s])
    names = ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")
    recs = _data.load_pmns("NO")
    q = {}
    for i, n in enumerate(names):
        v = S[:, i]
        q[n] = {"q025": float(np.quantile(v, 0.025)), "q16": float(np.quantile(v, 0.16)),
                "median": float(np.median(v)), "q84": float(np.quantile(v, 0.84)),
                "q975": float(np.quantile(v, 0.975))}
    inside = {}
    for n in ("s12sq", "s13sq", "s23sq"):
        r = recs[n]
        lo, hi = r["value"] - 3 * r["err_minus"], r["value"] + 3 * r["err_plus"]
        inside[n] = float(np.mean((S[:, names.index(n)] > lo) & (S[:, names.index(n)] < hi)))
    joint = np.ones(len(S), bool)
    for n in ("s12sq", "s13sq", "s23sq"):
        r = recs[n]
        v = S[:, names.index(n)]
        joint &= (v > r["value"] - 3 * r["err_minus"]) & (v < r["value"] + 3 * r["err_plus"])
    out_d = {"fit": fitfile, "mode": mode, "use": list(scorer(USE_QL + (("dm_ratio",) if mode == "ql_dm" else ())).use),
             "steps_per_chain": steps, "acceptance": [a for _, a, _ in out], "n_samples": int(len(S)),
             "chi2_ql_quantiles": [float(np.quantile(S[:, -1], x)) for x in (0.0, 0.5, 0.95)],
             "pmns_predictive": q, "frac_inside_nufit_3sigma_1d": inside, "frac_all_three_angles_3sigma": float(joint.mean()),
             "wall_s": time.time() - t0}
    tag = fitfile.replace("fit_", "").replace(".json", "")
    (HERE / f"heldout_{tag}_{mode}.json").write_text(json.dumps(out_d, indent=1))
    np.save(ROOT / "artifacts" / "023" / f"heldout_{tag}_{mode}.npy", S)
    print(json.dumps({k: out_d[k] for k in ("acceptance", "n_samples", "chi2_ql_quantiles",
                                             "frac_inside_nufit_3sigma_1d", "frac_all_three_angles_3sigma")}))
    for n, v in q.items():
        print(n, {a: round(b, 4) for a, b in v.items()})


if __name__ == "__main__":
    main()
