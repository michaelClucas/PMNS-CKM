"""Exp 028 analysis: held-out information, calibration, real T, N1-lambda and N1 ranks, Dirichlet
prior sensitivity, per-seed spread. Implements the README pre-registration.

Usage: python analyze.py <ordering> [n_fakes] [n_dirichlet]   -> results_<ordering>.json
Measured values only through src loaders (CKM, masses, NuFIT via src.data / src.evaluate /
src.nulls). Per-(seed, context, PMNS point, family) log densities are cached in
artifacts/028/evals_<ordering>.npz.
"""
from __future__ import annotations

import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import math
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from src import data as sd  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import nulls  # noqa: E402
from src.infogain import cp_interval  # noqa: E402
from train import ART, Clf, Flow, load, to_target  # noqa: E402

LN2 = math.log(2)
MR_KEYS = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau")  # exp 025 gen.MR_KEYS
N_CONV = 256
SEEDS = range(5)


def log(msg):
    print(msg, flush=True)
    with open(HERE / "run.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


# ---------------------------------------------------------------- inputs
def ckm_ctx(c):
    s12, s13, s23, d = (c[k]["value"] for k in ("s12", "s13", "s23", "delta_rad"))
    return [np.log10(s12), np.log10(s13), np.log10(s23), np.cos(d), np.sin(d)]


def real_inputs(ordering):
    ckm = np.array([ckm_ctx(sd.load_ckm())])
    mr, _, _ = ev.mass_ratio_targets()
    m = np.array([[np.log10(mr[k]) for k in MR_KEYS]])
    p = sd.load_pmns(ordering)
    r = abs(p["dm21sq_1e-5eV2"]["value"] * 1e-5 / (p["dm3l_sq_1e-3eV2"]["value"] * 1e-3))
    best = ev.pmns_best(ordering)
    pts = [[best["s12sq"], best["s13sq"], best["s23sq"], np.radians(best["dcp_deg"]) % (2 * np.pi), np.log10(r)]]
    rng = np.random.default_rng(2525)
    for _ in range(N_CONV):
        off = nulls.pmns_noise(ordering, rng)
        pts.append([best["s12sq"] - off["s12sq"], best["s13sq"] - off["s13sq"], best["s23sq"] - off["s23sq"],
                    np.radians(best["dcp_deg"] - off["dcp_deg"]) % (2 * np.pi), np.log10(r)])
    return ckm, m, to_target(np.array(pts)), dict(best=best, dm_ratio=r)


def split_normal(rng, rec):
    z = rng.standard_normal()
    return rec["value"] + z * (rec["err_plus"] if z > 0 else rec["err_minus"])


def fakes(kind, n, ordering):
    base = nulls.Universe.real(ordering)
    real = sd.load_ckm()
    ctx, w = [], []
    for i in range(n):
        if kind == "N1lam":
            rng = np.random.default_rng([28, 2, i])
            lam, A = split_normal(rng, real["lambda"]), split_normal(rng, real["A"])
            u = nulls.n1_universe(base, rng, prior={"lambda": (lam, lam), "A": (A, A)})
        else:
            u = nulls.n1_universe(base, np.random.default_rng([28, 1, i]))
        ctx.append(ckm_ctx(u.load_ckm()))
        w.append([u.truth["wolfenstein"][k] for k in ("lambda", "A", "rhobar", "etabar")])
    return np.array(ctx), np.array(w)


# ---------------------------------------------------------------- networks
def models(ordering):
    out = []
    for s in SEEDS:
        p = {k: ART / "models" / f"{ordering}_{k}_s{s}.pt" for k in ("q1", "q0", "c1", "c0")}
        if all(x.exists() for x in p.values()):
            out.append((s, Flow.load(p["q1"]), Flow.load(p["q0"]), Clf.load(p["c1"]), Clf.load(p["c0"])))
    return out


def all_family_logq(flow, y, c, F):
    """(n, F) log q(y_i | c_i, f) for every family f."""
    n = len(y)
    return np.stack([flow.log_prob(y, c, np.tile(np.eye(F)[f], (n, 1))) for f in range(F)], 1)


def mixture_logq(L, logpf, logw_eff):
    """L (..., F) family log densities, logpf (..., F) classifier log p(f|c) broadcastable,
    logw_eff (F,) = log w_f - log p_train(f). Returns log sum_f pi_f q_f."""
    a = logpf + logw_eff
    a = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    b = a + L
    return np.logaddexp.reduce(b, axis=-1)


def lme(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


# ---------------------------------------------------------------- held-out information and calibration
def heldout(D, M, ptrain):
    te = D["split"] == 2
    y, c1, c0, fam, oh = D["y"][te], D["c1"][te], D["c0"][te], D["fam"][te], D["onehot"][te]
    F = len(D["fams"])
    lw = -np.log(ptrain)
    cond, mix, clf = [], [], []
    for s, q1, q0, k1, k0 in M:
        cond.append((q1.log_prob(y, c1, oh) - q0.log_prob(y, c0, oh)) / LN2)
        lp1, lp0 = k1.log_proba(c1), k0.log_proba(c0)
        m1 = mixture_logq(all_family_logq(q1, y, c1, F), lp1, lw)
        m0 = mixture_logq(all_family_logq(q0, y, c0, F), lp0, lw)
        mix.append((m1, m0))
        clf.append(dict(seed=s, c1_nll=float(-lp1[np.arange(len(fam)), fam].mean()),
                        c1_acc=float((lp1.argmax(1) == fam).mean()),
                        c0_nll=float(-lp0[np.arange(len(fam)), fam].mean()), c0_acc=float((lp0.argmax(1) == fam).mean())))
    g_cond = np.mean(cond, 0)
    g_mix = (np.mean([m[0] for m in mix], 0) - np.mean([m[1] for m in mix], 0)) / LN2
    out = {}
    for i, f in enumerate(D["fams"] + ["ALL"]):
        sel = fam == i if f != "ALL" else np.ones(len(fam), bool)
        out[f] = dict(n=int(sel.sum()),
                      I_cond_bits=float(g_cond[sel].mean()), se_cond=float(g_cond[sel].std() / np.sqrt(sel.sum())),
                      I_mix_bits=float(g_mix[sel].mean()), se_mix=float(g_mix[sel].std() / np.sqrt(sel.sum())),
                      I_cond_per_seed=[float(c[sel].mean()) for c in cond])
    test_nll = {f"seed{s}": dict(q1_mix=float(-m[0].mean()), q0_mix=float(-m[1].mean())) for (s, *_), m in zip(M, mix)}
    return out, clf, test_nll


def calibration(D, M, ptrain, rng, n=500, m=200):
    """HPD-rank coverage. cond: per-seed family-conditional flows. mix: per-seed mixture at equal
    weights (family drawn from pi(c), then y from q(.|c,f); mixture density of the samples)."""
    te = np.flatnonzero(D["split"] == 2)
    idx = rng.choice(te, n, replace=False)
    F = len(D["fams"])
    lw = -np.log(ptrain)
    levels = (0.5, 0.68, 0.9, 0.95)
    out = {}
    for s, q1, q0, k1, k0 in M:
        for name, q, k, c in (("q1", q1, k1, D["c1"][idx]), ("q0", q0, k0, D["c0"][idx])):
            y, oh, fam = D["y"][idx], D["onehot"][idx], D["fam"][idx]
            # conditional
            S = q.sample(c, oh, m)  # (m, n, 5)
            ls = q.log_prob(S.reshape(-1, 5), np.tile(c, (m, 1)), np.tile(oh, (m, 1))).reshape(m, n)
            ly = q.log_prob(y, c, oh)
            r_cond = (ls > ly[None]).mean(0)
            # mixture
            lp = k.log_proba(c)
            a = lp + lw
            pi = np.exp(a - np.logaddexp.reduce(a, 1, keepdims=True))
            cum = np.cumsum(pi / pi.sum(1, keepdims=True), 1)
            fs = np.minimum((rng.uniform(size=(m, n, 1)) > cum[None]).sum(-1), F - 1)  # (m, n) family draws
            Sm = np.empty((m, n, 5))
            for f in range(F):
                sel = fs == f
                if sel.any():
                    cols = np.nonzero(sel)
                    Sm[cols] = q.sample(c[cols[1]], np.tile(np.eye(F)[f], (len(cols[1]), 1)), 1)[0]
            cc = np.tile(c, (m, 1))
            lsm = mixture_logq(all_family_logq(q, Sm.reshape(-1, 5), cc, F), np.tile(lp, (m, 1)), lw).reshape(m, n)
            lym = mixture_logq(all_family_logq(q, y, c, F), lp, lw)
            r_mix = (lsm > lym[None]).mean(0)
            out[f"seed{s}_{name}"] = dict(cond={f"{l:.2f}": float(np.mean(r_cond <= l)) for l in levels},
                                          mix={f"{l:.2f}": float(np.mean(r_mix <= l)) for l in levels})
    return out


# ---------------------------------------------------------------- real-data evaluations (cached)
def evals(ordering, M, F, real, ctx_all):
    """L1 (S, n_ctx, J, F), lp1 (S, n_ctx, F), L0 (S, J, F), lp0 (S, F)."""
    path = ART / f"evals_{ordering}.npz"
    ckm, m, ys, _ = real
    J = len(ys)
    n_ctx = len(ctx_all)
    c1 = np.concatenate([ctx_all, np.repeat(m, n_ctx, 0)], 1)
    L1, LP1, L0, LP0 = [], [], [], []
    for s, q1, q0, k1, k0 in M:
        t0 = time.time()
        cc = np.repeat(c1, J, 0)
        yy = np.tile(ys, (n_ctx, 1))
        L1.append(all_family_logq(q1, yy, cc, F).reshape(n_ctx, J, F).astype(np.float32))
        LP1.append(k1.log_proba(c1))
        L0.append(all_family_logq(q0, ys, np.repeat(m, J, 0), F))
        LP0.append(k0.log_proba(m)[0])
        log(f"[{ordering}] evals seed {s} done ({time.time() - t0:.0f} s)")
    out = dict(L1=np.stack(L1), LP1=np.stack(LP1), L0=np.stack(L0), LP0=np.stack(LP0))
    np.savez(path, **out)
    return out


def statistic(E, logw_eff, seeds=None):
    """T_conv and T_point (bits) for every context row; ensemble = mean over seeds of log q."""
    sel = slice(None) if seeds is None else seeds
    L1, LP1, L0, LP0 = E["L1"][sel], E["LP1"][sel], E["L0"][sel], E["LP0"][sel]
    # mixture weights pi (S, n_ctx, F)
    a = LP1 + logw_eff
    a = a - np.logaddexp.reduce(a, -1, keepdims=True)
    if seeds is None:
        if "_X1" not in E:
            E["_mx1"] = L1.max(-1, keepdims=True)
            E["_X1"] = np.exp(L1 - E["_mx1"])
        mx, X1 = E["_mx1"], E["_X1"]
    else:
        mx = L1.max(-1, keepdims=True)
        X1 = np.exp(L1 - mx)
    q1 = np.log(np.einsum("scf,scjf->scj", np.exp(a).astype(np.float32), X1)) + mx[..., 0]  # (S, n_ctx, J)
    q0 = mixture_logq(L0, LP0[:, None, :], logw_eff)  # (S, J)
    e1, e0 = q1.mean(0), q0.mean(0)
    conv = (lme(e1[:, 1:], 1) - lme(e0[None, 1:], 1)) / LN2
    point = (e1[:, 0] - e0[0]) / LN2
    return conv, point


def rank_block(T_real, T_f):
    h = int(np.sum(T_f >= T_real))
    d = nulls.decide(h, len(T_f))
    lo, hi = cp_interval(h, len(T_f))
    return dict(n=len(T_f), n_ge_real=h, rank=h / len(T_f), cp95_two_sided=[lo, hi], cp95_upper_one_sided=d["upper"],
                passed=d["passed"], fake_T_median=float(np.median(T_f)),
                fake_T_q05_q95=[float(np.quantile(T_f, .05)), float(np.quantile(T_f, .95))])


def evaluate_weights(E, logw_eff, idx, seeds=None):
    conv, point = statistic(E, logw_eff, seeds)
    out = dict(T_conv=float(conv[0]), T_point=float(point[0]))
    for nm, sl in idx.items():
        out[f"{nm}_conv"] = rank_block(conv[0], conv[sl])
        out[f"{nm}_point"] = rank_block(point[0], point[sl])
    return out, conv, point


def main(ordering="NO", n_fakes=2000, n_dir=500):
    torch.set_num_threads(int(os.environ.get("EXP028_THREADS", 6)))
    D = load(ordering)
    F = len(D["fams"])
    ptrain = np.bincount(D["fam"][D["split"] == 0], minlength=F) / np.sum(D["split"] == 0)
    M = models(ordering)
    log(f"[{ordering}] analysis start: {len(M)} seeds, families {D['fams']}, p_train {np.round(ptrain, 4).tolist()}")
    res = dict(ordering=ordering, families=D["fams"], seeds=[m[0] for m in M], n_fakes=n_fakes, n_conv=N_CONV,
               p_train=ptrain.tolist())
    # training summaries
    res["training"] = {}
    for k in ("q1", "q0", "c1", "c0"):
        for s in res["seeds"]:
            j = json.load(open(HERE / "curves" / f"{ordering}_{k}_s{s}.json"))
            res["training"][f"{k}_s{s}"] = dict(epochs=j["epochs"], best_epoch=j["best_epoch"], stop=j["stop_reason"],
                                                best_val_nll=j["best_val_nll"], seconds=round(j["seconds"]))
    # 1. held-out information and calibration (synthetic only)
    rng = np.random.default_rng(280)
    res["heldout"], res["classifiers"], res["test_nll_mix"] = heldout(D, M, ptrain)
    log(f"[{ordering}] held-out I_cond (bits): " + ", ".join(f"{f} {v['I_cond_bits']:.2f}" for f, v in res["heldout"].items()))
    log(f"[{ordering}] held-out I_mix (bits): " + ", ".join(f"{f} {v['I_mix_bits']:.2f}" for f, v in res["heldout"].items()))
    res["calibration"] = calibration(D, M, ptrain, rng)
    log(f"[{ordering}] calibration done")
    # 2. real data and nulls
    real = real_inputs(ordering)
    f_lam, w_lam = fakes("N1lam", n_fakes, ordering)
    f_n1, w_n1 = fakes("N1", n_fakes, ordering)
    ctx_all = np.concatenate([real[0], f_lam, f_n1])
    idx = {"N1lam": slice(1, 1 + n_fakes), "N1": slice(1 + n_fakes, 1 + 2 * n_fakes)}
    res["real"] = dict(ckm_ctx=real[0][0].tolist(), mass_ctx=real[1][0].tolist(), **real[3])
    E = evals(ordering, M, F, real, ctx_all)
    eq = -np.log(ptrain) + np.log(1.0 / F)
    res["equal"], conv_eq, point_eq = evaluate_weights(E, eq, idx)
    r = res["equal"]
    log(f"[{ordering}] equal weights: T_conv {r['T_conv']:.3f} bits, T_point {r['T_point']:.3f}; "
        f"N1lam conv rank {r['N1lam_conv']['rank']:.4f} (upper {r['N1lam_conv']['cp95_upper_one_sided']:.4f}, "
        f"passed {r['N1lam_conv']['passed']}); N1 conv rank {r['N1_conv']['rank']:.4f} "
        f"(upper {r['N1_conv']['cp95_upper_one_sided']:.4f}, passed {r['N1_conv']['passed']})")
    res["fake_T_conv"] = {k: conv_eq[sl].round(4).tolist() for k, sl in idx.items()}
    res["fake_wolfenstein"] = {"N1lam": w_lam.round(5).tolist(), "N1": w_n1.round(5).tolist()}
    # diagnostic: N1lam fake T vs |rhobar + i etabar| and vs the real (rhobar, etabar) neighbourhood
    Tl, wl = conv_eq[idx["N1lam"]], w_lam
    rad = np.hypot(wl[:, 2], wl[:, 3])
    ck = sd.load_ckm()
    rr, er = ck["rhobar"]["value"], ck["etabar"]["value"]
    diag = []
    for a, b in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5)):
        sel = (rad >= a) & (rad < b)
        if sel.any():
            diag.append(dict(bin=[a, b], n=int(sel.sum()), median_T=float(np.median(Tl[sel])),
                             frac_ge_real=float(np.mean(Tl[sel] >= conv_eq[0]))))
    near = np.hypot(wl[:, 2] - rr, wl[:, 3] - er) < 0.15
    res["diag_N1lam"] = dict(by_radius=diag, near_real_rho_eta=dict(n=int(near.sum()), median_T=float(np.median(Tl[near])) if near.any() else None,
                                                                    frac_ge_real=float(np.mean(Tl[near] >= conv_eq[0])) if near.any() else None),
                             corr_T_rhobar=float(np.corrcoef(Tl, wl[:, 2])[0, 1]), corr_T_etabar=float(np.corrcoef(Tl, wl[:, 3])[0, 1]))
    # 3. per-seed spread
    res["per_seed"] = {}
    for i, s in enumerate(res["seeds"]):
        o, _, _ = evaluate_weights(E, eq, idx, seeds=[i])
        res["per_seed"][f"seed{s}"] = dict(T_conv=o["T_conv"], T_point=o["T_point"], N1lam_conv_rank=o["N1lam_conv"]["rank"],
                                           N1_conv_rank=o["N1_conv"]["rank"])
    log(f"[{ordering}] per-seed T_conv: " + ", ".join(f"{k} {v['T_conv']:.2f}" for k, v in res["per_seed"].items()))
    # 4. Dirichlet prior sensitivity
    rd = np.random.default_rng(2828)
    W = rd.dirichlet(np.ones(F), n_dir)
    rows = []
    for w in W:
        o, _, _ = evaluate_weights(E, -np.log(ptrain) + np.log(w), idx)
        rows.append([o["T_conv"], o["N1lam_conv"]["rank"], o["N1lam_conv"]["passed"], o["N1_conv"]["rank"],
                     o["N1_conv"]["passed"]])
    rows = np.array(rows, float)
    res["dirichlet"] = dict(n=n_dir, alpha=1.0,
                            N1lam_pass_fraction=float(rows[:, 2].mean()), N1_pass_fraction=float(rows[:, 4].mean()),
                            T_conv_q05_q50_q95=np.quantile(rows[:, 0], [.05, .5, .95]).tolist(),
                            N1lam_rank_q05_q50_q95=np.quantile(rows[:, 1], [.05, .5, .95]).tolist(),
                            N1_rank_q05_q50_q95=np.quantile(rows[:, 3], [.05, .5, .95]).tolist(),
                            N1lam_rank_min=float(rows[:, 1].min()),
                            corr_rank_N1lam_with_weight={f: float(np.corrcoef(W[:, i], rows[:, 1])[0, 1])
                                                         for i, f in enumerate(D["fams"])})
    log(f"[{ordering}] Dirichlet: N1lam pass fraction {res['dirichlet']['N1lam_pass_fraction']:.3f}, "
        f"N1 pass fraction {res['dirichlet']['N1_pass_fraction']:.3f}, T_conv q05/50/95 "
        f"{np.round(res['dirichlet']['T_conv_q05_q50_q95'], 2).tolist()}, N1lam rank q05/50/95 "
        f"{np.round(res['dirichlet']['N1lam_rank_q05_q50_q95'], 3).tolist()}")
    # 5. decision (pre-registered)
    p_eq = res["equal"]["N1lam_conv"]["passed"]
    p_dir = res["dirichlet"]["N1lam_pass_fraction"] >= 0.9
    res["decision"] = dict(primary_null_passed_equal_weights=p_eq, dirichlet_pass_fraction_ge_0p9=p_dir,
                           verdict="candidate signal" if (p_eq and p_dir) else "no evidence")
    log(f"[{ordering}] decision: {res['decision']}")
    json.dump(res, open(HERE / f"results_{ordering}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "NO", int(a[1]) if len(a) > 1 else 2000, int(a[2]) if len(a) > 2 else 500)
