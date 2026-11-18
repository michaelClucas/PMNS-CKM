"""Real-data analysis from the DE-MC posteriors (exact likelihood, no network):

  python analyze_local.py <variant>
1. Held-out PMNS test: p(PMNS | quark + charged-lepton data) from mcmc_<variant>_ql.npz vs measured PMNS.
   Compared with the prior predictive (broad dataset, NO-classified samples).
2. Full fit: mcmc_<variant>_all.npz, best chi2 and pulls, which parameters are constrained.
3. Ridge search: mutual information (sklearn KSG estimator) between each PMNS observable and each theta / each
   quark-sector observable across the ql posterior.
Writes local_<variant>.json and local_<variant>.png.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402
from src import data  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "f3"
REF = gen.gut_reference()
Z0, S0 = np.array(REF["z"]), np.array(REF["sigma"])
PN = ["s12sq_l", "s13sq_l", "s23sq_l", "dl", "lg_dm21_dm3l"]


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def pmns_tests(zp, rng):
    """zp: noiseless PMNS block (N, 5). Returns PIT of measured values under the noisy predictive, 68/95% intervals,
    chi2 summaries (Gaussian sigma model and NuFIT tables)."""
    zn = zp + S0[12:17] * rng.standard_normal(zp.shape)
    zn[:, 3] = np.mod(zn[:, 3], 2 * np.pi)
    out = {}
    for k, n in enumerate(PN):
        if n == "dl":
            dd = wrap(zn[:, 3] - Z0[15])
            pit = float(np.mean(dd < 0))
            q = np.degrees(np.quantile(np.mod(zp[:, 3], 2 * np.pi), [0.025, 0.16, 0.5, 0.84, 0.975]))
        else:
            pit = float(np.mean(zn[:, k] < Z0[12 + k]))
            q = np.quantile(zp[:, k], [0.025, 0.16, 0.5, 0.84, 0.975])
        out[n] = {"measured": float(np.degrees(Z0[15]) if n == "dl" else Z0[12 + k]), "PIT": round(pit, 4),
                  "pred_q2.5_16_50_84_97.5": np.round(q, 4).tolist()}
    d = zp - Z0[12:17]
    d[:, 3] = wrap(d[:, 3])
    c2 = ((d / S0[12:17]) ** 2).sum(1)
    out["chi2_gauss_5obs"] = {"min": round(float(c2.min()), 2), "q05_50": np.round(np.quantile(c2, [0.05, 0.5]), 1).tolist(),
                              "frac_below_11.07": float(np.mean(c2 < 11.07))}
    # NuFIT 6.1 tables (NO), GUT ~ M_Z for PMNS in the SM (<= 0.12 deg)
    sub = zp[np.random.default_rng(0).choice(len(zp), size=min(3000, len(zp)), replace=False)]
    nf = np.array([data.pmns_delta_chi2("NO", s12sq=a, s13sq=b, s23sq=c, dcp_deg=np.degrees(dl) % 360)
                   for a, b, c, dl in sub[:, :4]])
    nf = np.where(np.isfinite(nf), nf, 1e4)
    out["nufit_dchi2_4par"] = {"min": round(float(nf.min()), 2), "median": round(float(np.median(nf)), 1),
                               "frac_below_9.49": float(np.mean(nf < 9.49))}
    return out


def main():
    v = sys.argv[1]
    rng = np.random.default_rng(5)
    res = {"variant": v}
    names = gen.VARIANTS[v]["theta"]
    lo, hi = gen.prior_box(v)
    # --- 1. held-out PMNS
    m = np.load(ART / f"mcmc_{v}_ql.npz")
    zq, ioq, thq = m["z"].astype(float), m["io"], m["theta"].astype(float)
    res["ql_posterior"] = {"n": int(len(zq)), "n_chains": int(m["chain"].max() + 1), "frac_IO": float(ioq.mean()),
                           "chi2_ql_median": round(float(np.median(-2 * m["loglike"])), 2)}
    no = ioq == 0
    res["heldout_pmns_from_ql"] = pmns_tests(zq[no][:, 12:17], rng)
    b = np.load(ART / f"ds_broad_{v}.npz")
    zb = b["z"][:200000].astype(float)
    zb = zb[b["io"][:200000] == 0]
    res["prior_predictive_pmns"] = pmns_tests(zb[:, 12:17], rng)
    # --- 2. full fit
    fa = np.load(ART / f"mcmc_{v}_all.npz")
    za, tha, lla = fa["z"].astype(float), fa["theta"].astype(float), fa["loglike"]
    fp = ART / f"fits_{v}_all_fromql.npz"
    if fp.exists():
        fit = dict(np.load(fp))
        src_fit = "least_squares from ql-posterior starts"
    else:  # fall back to the best DE-MC 'all' sample
        fit = {"theta": tha, "chi2": -2 * lla}
        src_fit = "best DE-MC 'all' sample (no least-squares full fit available)"
    ib = np.argmin(fit["chi2"])
    zbest, iob, _ = gen.simulate(fit["theta"][ib][None], v)
    pl = (zbest[0] - Z0) / S0
    for j in gen.Z_PHASES:
        pl[j] = wrap(zbest[0, j] - Z0[j]) / S0[j]
    per = np.isclose(hi - lo, 2 * np.pi) & np.isclose(lo, 0)

    def spread(t):
        return np.where(per, np.sqrt(-2 * np.log(np.clip(np.abs(np.exp(1j * t).mean(0)), 1e-12, 1))), t.std(0))
    prior_sd = np.where(per, np.inf, (hi - lo) / np.sqrt(12))
    res["full_fit"] = {"source": src_fit, "best_chi2_17obs": round(float(fit["chi2"][ib]), 2), "best_is_IO": int(iob[0]),
                       "pulls_at_best": dict(zip(gen.Z_NAMES, np.round(pl, 2).tolist())),
                       "n_starts_chi2_below_17": int((fit["chi2"] < 17).sum()), "n_starts": int(len(fit["chi2"])),
                       "mcmc_chi2_q05_50_95": np.round(np.quantile(-2 * lla[np.isfinite(lla)], [0.05, 0.5, 0.95]), 2).tolist(),
                       "posterior_spread_over_prior_sd(phases: circular sd in rad)":
                           dict(zip(names, np.round(np.where(per, spread(tha), spread(tha) / prior_sd), 3).tolist())),
                       "ql_posterior_spread_over_prior_sd":
                           dict(zip(names, np.round(np.where(per, spread(thq), spread(thq) / prior_sd), 3).tolist())),
                       "theta_best": dict(zip(names, np.round(fit["theta"][ib], 4).tolist()))}
    # --- 3. MI between PMNS outputs and theta / quark-sector observables across the ql posterior
    from sklearn.feature_selection import mutual_info_regression
    sel = np.random.default_rng(1).choice(no.sum(), size=min(6000, no.sum()), replace=False)
    Zs, Ts = zq[no][sel], thq[no][sel]
    feats_t = np.concatenate([np.where(per, np.cos(Ts), Ts), np.sin(Ts[:, per])], 1)
    fnames_t = names + [n + "_sin" for n, p in zip(names, per) if p]
    mi_t, mi_q = {}, {}
    for k, n in enumerate(PN):
        y = Zs[:, 12 + k] if n != "dl" else np.cos(Zs[:, 15])
        mt = mutual_info_regression(feats_t, y, random_state=0)
        mq = mutual_info_regression(Zs[:, :12], y, random_state=0)
        mi_t[n] = dict(sorted(zip(fnames_t, np.round(mt, 3).tolist()), key=lambda a: -a[1])[:6])
        mi_q[n] = dict(zip(gen.Z_NAMES[:12], np.round(mq, 3).tolist()))
    res["MI_nats_pmns_vs_theta_top6"] = mi_t
    res["MI_nats_pmns_vs_quark_lepton_obs"] = mi_q
    # spread of the ql block across the ql posterior, in sigma (how far the CKM etc. move along the posterior)
    dq = Zs[:, :12] - Z0[:12]
    dq[:, 11] = wrap(dq[:, 11])
    res["ql_obs_posterior_sd_over_sigma"] = dict(zip(gen.Z_NAMES[:12], np.round(dq.std(0) / S0[:12], 2).tolist()))
    (HERE / f"local_{v}.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(14, 2.8), dpi=80)
    lab = ["sin²θ12", "sin²θ13", "sin²θ23", "δ_ℓ [deg]", "log10 Δm²21/Δm²3ℓ"]
    for k in range(5):
        f = (lambda a: np.degrees(np.mod(a, 2 * np.pi))) if k == 3 else (lambda a: a)
        ax[k].hist(f(zb[:, 12 + k]), bins=60, density=True, histtype="step", color="0.5", label="prior pred.")
        ax[k].hist(f(zq[no][:, 12 + k]), bins=60, density=True, alpha=0.6, color="C0", label="| quark+ch.lepton")
        if len(za):
            ax[k].hist(f(za[:, 12 + k]), bins=60, density=True, histtype="step", color="C3", label="| all data")
        ax[k].axvline(np.degrees(Z0[15]) if k == 3 else Z0[12 + k], color="k", lw=1.5)
        ax[k].set_xlabel(lab[k], fontsize=8); ax[k].set_yticks([])
    ax[0].legend(fontsize=6)
    fig.suptitle(f"{v}: PMNS predictive at M_GUT (black = measured, run to M_GUT)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / f"local_{v}.png")
    print(json.dumps({k: res[k] for k in ("ql_posterior", "heldout_pmns_from_ql", "prior_predictive_pmns")}, indent=1))
    print(json.dumps({k: v_ for k, v_ in res["full_fit"].items() if "spread" not in k and k != "theta_best"}, indent=1))
    print("MI pmns vs theta:", json.dumps(mi_t))
    print("MI pmns vs ql obs:", json.dumps(mi_q))


if __name__ == "__main__":
    main()
