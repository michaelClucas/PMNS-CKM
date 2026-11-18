"""Calibration (SBC ranks, marginal + HPD coverage) of a trained theta-NPE on its held-out synthetic samples,
and the NPE posterior at the real-data vector.

  python analyze_npe.py <name> [n_sbc] [importance]
importance = 1 for a model trained on the local KDE proposal: real-data posterior samples are reweighted by
pi(theta)/p~(theta) (uniform prior -> 1/p~) with SIR, and the effective sample size is reported.
Writes <name>_calib.png, <name>_calib.json in the experiment folder and prints a summary.
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402
from train_npe import ART, Model, features, SIG  # noqa: E402

torch.set_num_threads(2)
REF = gen.gut_reference()


def main():
    name = sys.argv[1]
    n_sbc = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    importance = len(sys.argv) > 3 and sys.argv[3] == "1"
    M = Model.load(name)
    ck = torch.load(ART / f"npe_{name}.pt", weights_only=False)
    val_idx = np.asarray(ck["val_idx"])
    meta = M.meta
    variant = meta["variant"]
    d = np.load(ART / meta["ds"])
    th, z, io = d["theta"].astype(float), d["z"].astype(float), d["io"]
    names = gen.VARIANTS[variant]["theta"]
    lo, hi = gen.prior_box(variant)
    zv = gen.add_noise(z[val_idx], SIG, np.random.default_rng(12345))
    cv = features(zv, io[val_idx], meta["cond"])
    sel = np.random.default_rng(1).choice(len(val_idx), size=min(n_sbc, len(val_idx)), replace=False)
    L = 300
    ranks, hpd = [], []
    levels = np.linspace(0.05, 0.95, 19)
    cover = np.zeros((len(levels), len(names)))
    width_ratio = []
    for i in sel:
        s = M.sample(cv[i], L)
        t = th[val_idx[i]]
        ranks.append((s < t).sum(0))
        lq = M.log_prob(np.vstack([t[None], s]), np.repeat(cv[i][None], L + 1, 0))
        hpd.append((lq[1:] > lq[0]).mean())
        q = np.quantile(s, np.concatenate([(1 - levels) / 2, (1 + levels) / 2]), axis=0)
        cover += (t >= q[:len(levels)]) & (t <= q[len(levels):])
        width_ratio.append(s.std(0) / ((hi - lo) / np.sqrt(12)))
    ranks, hpd, cover = np.array(ranks), np.array(hpd), cover / len(sel)
    width_ratio = np.median(np.array(width_ratio), 0)
    # chi^2-type uniformity test per dim for the ranks (10 bins)
    from scipy.stats import chisquare
    pv = [chisquare(np.histogram(ranks[:, j], bins=10, range=(0, L + 1))[0]).pvalue for j in range(len(names))]
    hpd_cov = [(np.mean(hpd <= a)) for a in levels]  # fraction of truths inside the a-HPD region
    res = {"name": name, "n_sbc": len(sel), "L": L,
           "rank_uniformity_p": dict(zip(names, np.round(pv, 4).tolist())),
           "marginal_coverage_at_65": dict(zip(names, np.round(cover[np.argmin(abs(levels - 0.65))], 3).tolist())),
           "hpd_coverage": dict(zip(np.round(levels, 2).astype(str).tolist(), np.round(hpd_cov, 3).tolist())),
           "median_post_sd_over_prior_sd": dict(zip(names, np.round(width_ratio, 3).tolist()))}

    # real data
    for o, key in (("NO", ""), ("IO", "_IO")):
        z0 = np.array(REF["z" + key])[None]
        c0 = features(z0, np.array([1 if o == "IO" else 0]), meta["cond"])[0]
        s = M.sample(c0, 20000)
        s = s[np.all((s >= lo) & (s <= hi), 1)]
        info = {"frac_in_box": float(len(s) / 20000)}
        if importance:
            from make_dataset import kde_spec, kde_logpdf
            c, h, per, _, _ = kde_spec(variant)
            lw = -kde_logpdf(s, c, h, per)
            w = np.exp(lw - lw.max()); w /= w.sum()
            info["ess"] = float(1 / (w ** 2).sum())
            s = s[np.random.default_rng(2).choice(len(s), size=len(s), p=w)]
        zz, iio, ok = gen.simulate(s, variant)
        dd = zz - z0
        for j in gen.Z_PHASES:
            dd[:, j] = (dd[:, j] + np.pi) % (2 * np.pi) - np.pi
        S0 = np.array(REF["sigma" + key])
        c2 = ((dd / S0) ** 2).sum(1)
        c2ql = ((dd[:, :12] / S0[:12]) ** 2).sum(1)
        info.update({"post_pred_chi2_all_quantiles_5_50_95": np.round(np.quantile(c2, [0.05, 0.5, 0.95]), 1).tolist(),
                     "post_pred_chi2_ql_quantiles_5_50_95": np.round(np.quantile(c2ql, [0.05, 0.5, 0.95]), 1).tolist(),
                     "post_pred_frac_IO": float(iio.mean()),
                     "post_mean": dict(zip(names, np.round(s.mean(0), 3).tolist())),
                     "post_sd_over_prior_sd": dict(zip(names, np.round(s.std(0) / ((hi - lo) / np.sqrt(12)), 3).tolist()))})
        res["real_" + o] = info
        np.save(ART / f"post_{name}_{o}.npy", s[:5000].astype(np.float32))
    (HERE / f"{name}_calib.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = len(names)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), dpi=80)
    for j in range(nd):
        ax[0].plot(levels, cover[:, j], color="C0", alpha=0.35, lw=1)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="marginal central-interval coverage", xlabel="credible level", ylabel="empirical coverage")
    ax[1].plot(levels, hpd_cov, "o-", ms=3)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set(title="joint HPD coverage", xlabel="credible level")
    ax[2].bar(range(nd), width_ratio)
    ax[2].set_xticks(range(nd)); ax[2].set_xticklabels(names, rotation=90, fontsize=7)
    ax[2].set(title="median posterior sd / prior sd (synthetic)")
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_calib.png")
    print(json.dumps({k: v for k, v in res.items() if k in ("name", "hpd_coverage", "real_NO", "real_IO")}, indent=1))
    print("rank p-values:", res["rank_uniformity_p"])
    print("65% marginal coverage:", res["marginal_coverage_at_65"])


if __name__ == "__main__":
    main()
