"""Held-out PMNS prediction from the quark + charged-lepton posterior (steps 2/3) and the posterior regression of
PMNS on CKM + mass observables (step 4).

Usage: python analyze_heldout.py <model MN|FN> <tag1,tag2,...> <out_name>
Reads artifacts/so10/pt_<tag>.npz (cold samples), draws the prior predictive from the same box, writes
<out_name>.json and <out_name>.png.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import so10fit as S

PM_NAMES = ["s12sq_l", "s13sq_l", "s23sq_l", "dl", "lg_dm21_dm3l"]


def pmns_chi2(z, val, err):
    d = z[:, 12:17] - val[12:17]
    d[:, 3] = S.wrap(d[:, 3])
    return ((d / err[12:17]) ** 2).sum(1)


def features(z):
    q = z[:, :12]
    return np.column_stack([q[:, :11], np.cos(q[:, 11]), np.sin(q[:, 11])])


def targets(z):
    return np.column_stack([z[:, 12], z[:, 13], z[:, 14], np.cos(z[:, 15]), np.sin(z[:, 15]), z[:, 16]])


T_NAMES = ["s12sq_l", "s13sq_l", "s23sq_l", "cos_dl", "sin_dl", "lg_dm21_dm3l"]


def r2_table(Xf, Y, groups, rng, nmax=20000):
    if len(Xf) > nmax:
        sel = rng.choice(len(Xf), nmax, replace=False)
        Xf, Y = Xf[sel], Y[sel]
        groups = None if groups is None else groups[sel]
    cv = GroupKFold(5) if groups is not None and len(np.unique(groups)) >= 5 else KFold(5, shuffle=True, random_state=0)
    models = {"linear": make_pipeline(StandardScaler(), LinearRegression()),
              "quadratic": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(1e-3)),
              "gbm": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1)}
    out = {}
    for mn, m in models.items():
        out[mn] = {}
        for j, tn in enumerate(T_NAMES):
            y = Y[:, j]
            if np.var(y) < 1e-30:
                out[mn][tn] = None
                continue
            pred = cross_val_predict(m, Xf, y, cv=cv, groups=groups if isinstance(cv, GroupKFold) else None)
            out[mn][tn] = float(1 - np.mean((y - pred) ** 2) / np.var(y))
    return out


def main():
    model, tags, out_name = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    rng = np.random.default_rng(5)
    prob = S.Problem(model, "ours")
    val, err = prob.val, prob.err
    th, z, io, grp, c2 = [], [], [], [], []
    for t_i, t in enumerate(tags):
        d = np.load(S.ART / f"pt_{t}.npz")
        th.append(d["theta"]); z.append(d["z"]); io.append(d["io"]); grp.append(d["walker"] + 1000 * t_i)
        c2.append(d["chi2"])
    th, z, io, grp, c2 = map(np.concatenate, (th, z, io, grp, c2))
    fin = np.all(np.isfinite(z), 1)          # degenerate spectra (e.g. a massless neutrino pair) give inf logs
    n_nonfinite = int((~fin).sum())
    th, z, io, grp, c2 = th[fin], z[fin], io[fin], grp[fin], c2[fin]
    lo, hi = (S.mn_box("ours", val) if model == "MN" else S.fn_box("ours"))
    thp = lo + (hi - lo) * rng.random((200000, len(lo)))
    zp, op = prob.predict(thp)
    okp = np.all(np.isfinite(zp), 1)
    zp, iop = zp[okp], op["io"][okp]
    res = dict(model=model, tags=tags, n_post=int(len(z)), n_post_nonfinite_dropped=n_nonfinite, n_prior=int(len(zp)),
               ql_chi2_post_quantiles=dict(zip(["q05", "q50", "q95"], map(float, np.quantile(c2, [.05, .5, .95])))),
               IO_frac_post=float(io.mean()), IO_frac_prior=float(iop.mean()), per_obs={})
    for j, n in zip(range(12, 17), PM_NAMES):
        a, b = z[:, j], zp[:, j]
        if n == "dl":
            m = val[j]
            pit_post = float(np.mean(a < m)); pit_prior = float(np.mean(b < m))
        else:
            pit_post, pit_prior = float(np.mean(a < val[j])), float(np.mean(b < val[j]))
        qa, qb = np.quantile(a, [.025, .16, .5, .84, .975]), np.quantile(b, [.025, .16, .5, .84, .975])
        res["per_obs"][n] = dict(measured=float(val[j]), sigma=float(err[j]), post_q=list(map(float, qa)),
                                 prior_q=list(map(float, qb)), pit_post=pit_post, pit_prior=pit_prior,
                                 width68_ratio_post_over_prior=float((qa[3] - qa[1]) / max(qb[3] - qb[1], 1e-30)),
                                 frac_within_2sigma_post=float(np.mean(np.abs(S.wrap(a - val[j]) if n == "dl" else a - val[j]) < 2 * err[j])),
                                 frac_within_2sigma_prior=float(np.mean(np.abs(S.wrap(b - val[j]) if n == "dl" else b - val[j]) < 2 * err[j])))
    # convergence: the same summaries ladder by ladder (independent PT runs started from the prior)
    res["per_ladder"] = {}
    for t_i, t in enumerate(tags):
        m = (grp // 1000) == t_i
        res["per_ladder"][t] = {n: list(map(float, np.quantile(z[m, j], [.025, .5, .975])))
                                for j, n in zip(range(12, 17), PM_NAMES) if n != "dl"}
        res["per_ladder"][t]["ql_chi2_median"] = float(np.median(c2[m]))
    cp, cq = pmns_chi2(z, val, err), pmns_chi2(zp, val, err)
    cp_no, cq_no = np.where(io, np.inf, cp), np.where(iop, np.inf, cq)
    res["pmns_chi2_5"] = dict(post_min=float(cp_no.min()), prior_min=float(cq_no.min()),
                              post_frac_lt_11p07=float(np.mean(cp_no < 11.07)), prior_frac_lt_11p07=float(np.mean(cq_no < 11.07)),
                              post_frac_lt_50=float(np.mean(cp_no < 50)), prior_frac_lt_50=float(np.mean(cq_no < 50)),
                              post_q=list(map(float, np.quantile(cp_no, [.01, .05, .5]))),
                              prior_q=list(map(float, np.quantile(cq_no, [.01, .05, .5]))))
    # step 4: PMNS on CKM + masses
    res["regression_R2_post"] = r2_table(features(z), targets(z), grp, rng)
    res["regression_R2_prior"] = r2_table(features(zp), targets(zp), None, rng)
    Xth = np.column_stack([np.cos(th), np.sin(th)]) if model == "MN" else th
    res["regression_R2_post_on_theta"] = r2_table(Xth, targets(z), grp, rng)
    (S.HERE / f"{out_name}.json").write_text(json.dumps(res, indent=1))
    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.2))
    for k, (j, n) in enumerate(zip(range(12, 17), PM_NAMES)):
        lo_, hi_ = np.quantile(np.concatenate([z[:, j], zp[:, j]]), [0.001, 0.999])
        lo_, hi_ = min(lo_, val[j] - 3 * err[j]), max(hi_, val[j] + 3 * err[j])
        bins = np.linspace(lo_, hi_, 60)
        ax[k].hist(zp[:, j], bins, density=True, alpha=0.5, color="0.5", label="prior pred.")
        ax[k].hist(z[:, j], bins, density=True, alpha=0.6, color="C0", label="ql posterior pred.")
        ax[k].axvspan(val[j] - err[j], val[j] + err[j], color="C3", alpha=0.3, label="measured ±1σ")
        ax[k].set_xlabel(n)
        ax[k].set_yticks([])
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{model}: PMNS predicted from quark + charged-lepton data only (GUT scale, SM running)")
    fig.tight_layout()
    fig.savefig(S.HERE / f"{out_name}.png", dpi=80)
    print(json.dumps({k: v for k, v in res.items() if k != "per_obs"}, indent=1)[:3000])
    for n, v in res["per_obs"].items():
        print(f"{n:14s} meas {v['measured']:.4f} post 95% [{v['post_q'][0]:.4f}, {v['post_q'][4]:.4f}] "
              f"prior 95% [{v['prior_q'][0]:.4f}, {v['prior_q'][4]:.4f}] PIT post {v['pit_post']:.3f} prior "
              f"{v['pit_prior']:.3f} width ratio {v['width68_ratio_post_over_prior']:.2f}")


if __name__ == "__main__":
    main()
