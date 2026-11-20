"""Step 4: held-out test. Quark sector only (CKM 4 + quark mass ratios 4, k_cont 5) fixes
(b_u, b_d, beta_d, phi1, phi2); xi_R is then fitted to the Dm^2 ratio alone; s12sq, s13sq, s23sq, delta
are predictions.

    python heldout.py <quark fit json> <n_steps> <seed>

Samples the quark-sector posterior exp(-chi2_q / 2) (flat priors on the fit box) with a
random-walk Metropolis chain started at the quark best fit, step scale from a finite-difference
Hessian. Each kept sample: xi_R = argmin dm_ratio Delta chi^2 (bounded scalar search over a log grid
+ refine, both signs), then the PMNS prediction is scored with the official scorer (NO tables).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = list(F.BOUNDS_Q)
SC_Q = ev.Scorer(F.CKM + F.MQ, "NO", inputs=F.INPUTS, mass_floor=F.FLOOR)  # YK_FLOOR env, as fit.py
SC_DM = ev.Scorer(["dm_ratio"], "NO")
SC_P = ev.Scorer(["s12sq", "s13sq", "s23sq", "dcp_deg"], "NO")
PHI0 = F.phi0()


def pred(x, xi):
    t = dict(zip(NAMES, x))
    t["xi_R"] = xi
    return F.M.predict(F.theta_full(t), PHI0)


def chi2_q(x):
    for n, v in zip(NAMES, x):
        lo, hi = F.BOUNDS_Q[n]
        if n not in F.PERIODIC and not lo <= v <= hi:
            return np.inf
    try:
        c = SC_Q.chi2_soft(pred(x, 2000.0), include_finite=True)  # xi_R does not touch quark keys
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.inf
    return c if np.isfinite(c) else np.inf


def fit_xi(x):
    best = (np.inf, None)
    for s in (1.0, -1.0):
        grid = s * np.logspace(0, 5, 41)
        vals = []
        for g in grid:
            try:
                vals.append(SC_DM.chi2_soft(pred(x, g), include_finite=True))
            except Exception:
                vals.append(np.inf)
        i = int(np.argmin(vals))
        lo, hi = sorted((grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]))
        r = minimize_scalar(lambda g: SC_DM.chi2_soft(pred(x, g), include_finite=True), bounds=(lo, hi),
                            method="bounded", options={"xatol": 1e-6 * abs(grid[i])})
        if r.fun < best[0]:
            best = (float(r.fun), float(r.x))
    return best


def hessian_scale(x0, h=1e-4):
    n = len(x0)
    H = np.zeros((n, n))
    f0 = chi2_q(x0)
    step = h * np.maximum(1.0, np.abs(x0))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * step[i], np.eye(n)[j] * step[j]
            H[i, j] = H[j, i] = (chi2_q(x0 + ei + ej) - chi2_q(x0 + ei - ej) - chi2_q(x0 - ei + ej)
                                 + chi2_q(x0 - ei - ej)) / (4 * step[i] * step[j])
    w, v = np.linalg.eigh(0.5 * H)  # chi2/2
    w = np.clip(w, 1e-8, None)
    return v @ np.diag(1 / np.sqrt(w)) @ v.T, f0


def main(fit_json, n_steps, seed):
    rng = np.random.default_rng(seed)
    q = json.loads(Path(fit_json).read_text())
    x = np.array([q["params"][n] for n in NAMES])
    L, f = hessian_scale(x)
    scale = 2.38 / np.sqrt(len(x))
    chain, acc = [], 0
    t0 = time.time()
    for it in range(n_steps):
        y = x + scale * L @ rng.standard_normal(len(x))
        g = chi2_q(y)
        if np.log(rng.random()) < -(g - f) / 2:
            x, f, acc = y, g, acc + 1
        chain.append((x.copy(), f))
        if it == n_steps // 5 and acc / (it + 1) < 0.1:
            scale *= 0.5
    burn = n_steps // 5
    keep = chain[burn::max(1, (n_steps - burn) // 400)]
    rows = []
    for xs, fq in keep:
        cdm, xi = fit_xi(xs)
        p = pred(xs, xi)
        rp = SC_P(p)
        rows.append({"chi2_q": fq, "xi_R": xi, "chi2_dm": cdm, "chi2_pmns": rp.chi2,
                     **{k: p[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")},
                     **dict(zip(NAMES, map(float, xs)))})
    out = {"fit_json": Path(fit_json).name, "n_steps": n_steps, "seed": seed, "acceptance": acc / n_steps,
           "chi2_q_best": q["chi2"], "seconds": time.time() - t0, "samples": rows}
    tag = Path(fit_json).stem
    (HERE / f"heldout_{tag}_s{seed}.json").write_text(json.dumps(out, default=float))
    a = {k: np.array([r[k] for r in rows]) for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "chi2_pmns", "chi2_q")}
    print(f"acceptance {acc / n_steps:.3f}, kept {len(rows)}, chi2_q range {a['chi2_q'].min():.2f}-{a['chi2_q'].max():.2f}")
    for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "chi2_pmns"):
        print(f"{k:10s} median {np.median(a[k]):.4g}  2.5-97.5% [{np.percentile(a[k], 2.5):.4g}, {np.percentile(a[k], 97.5):.4g}]")
    nb = ev.pmns_best("NO")
    print("NuFIT NO best fit:", nb)
    print("fraction with PMNS Delta chi2 (4 params, joint) < 9.49 (95% for 4 dof):", float(np.mean(a["chi2_pmns"] < 9.49)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
