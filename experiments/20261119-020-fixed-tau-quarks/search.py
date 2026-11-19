"""Common-tau modular search: synthetic data, emulators, emulator ranking, exact fits, joint fits.

    python search.py gen       # synthetic samples per group and sector -> artifacts/modular/
    python search.py train     # one MLP emulator per group and sector (5 processes x 2 threads)
    python search.py rank      # emulator-ranked assignments (sector-wise best chi2)
    python search.py exact     # exact multi-start fits: emulator top-K + random subset
    python search.py joint     # common-tau joint fits of top quark x top lepton assignments, scored
Run from the repo root or this folder; reads data only through src.nulls.Universe.
Scale: M_Z (SM). Model Yukawas are compared with M_Z Yukawa ratios from src.rg and with the
low-scale CKM/NuFIT data; the model is not run (stated in README).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402

import models as md  # noqa: E402
import modforms as mf  # noqa: E402

ART = os.path.join(ROOT, "artifacts", "modular")
os.makedirs(ART, exist_ok=True)
USE8 = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg")
MASS_INPUTS = ("m_u", "m_d", "m_s", "m_c", "m_b", "m_t_direct", "m_e", "m_mu", "m_tau")
RATIO_FLOOR = 0.01
LA_RANGE = (-14.0, 4.0)
G_RANGE = (-6.0, 6.0)
ORDERINGS = ("NO", "IO")
WORKERS = int(os.environ.get("MODSEARCH_WORKERS", "5"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- targets (through the universe)


def targets(universe, ordering):
    from src import rg
    masses = universe.load_masses()
    y = rg.derived_yukawas_mz(masses=None if universe.masses is None else masses)

    def rel(k):
        return masses[k]["err_plus"] / masses[k]["value"]

    def lr(a, b, ka, kb):
        return (float(np.log(y[a] / y[b])), max(float(np.hypot(rel(ka), rel(kb))), RATIO_FLOOR))

    T = {"lr_uc": lr("y_u", "y_c", "m_u", "m_c"), "lr_ct": lr("y_c", "y_t", "m_c", "m_t_direct"),
         "lr_ds": lr("y_d", "y_s", "m_d", "m_s"), "lr_sb": lr("y_s", "y_b", "m_s", "m_b"),
         "lr_emu": lr("y_e", "y_mu", "m_e", "m_mu"), "lr_mutau": lr("y_mu", "y_tau", "m_mu", "m_tau")}
    G = {}
    ckm = universe.load_ckm()
    for k in ("s12", "s13", "s23", "delta_rad"):
        r = ckm[k]
        G["ckm_" + k] = (r["value"], r["err_plus"], abs(r["err_minus"]))
    pm = universe.load_pmns(ordering)
    for k in ("s12sq", "s13sq", "s23sq", "dcp_deg"):
        r = pm[k]
        G[k] = (r["value"], r["err_plus"], abs(r["err_minus"]))
    d21, d3l = pm["dm21sq_1e-5eV2"], pm["dm3l_sq_1e-3eV2"]
    rv = 0.01 * d21["value"] / abs(d3l["value"])
    re = rv * np.hypot(d21["err_plus"] / d21["value"], d3l["err_plus"] / abs(d3l["value"]))
    T["r_dm"] = (float(rv), float(re))  # absolute error (not log)
    return {"ratio": T, "gauss": G}


def _pull(pred, val, ep, em):
    d = pred - val
    return d / np.where(d > 0, ep, em)


def _wrap(d, period):
    return (d + period / 2) % period - period / 2


def chi2_quark_gauss(o, T):
    c = 0.0
    for k in ("lr_uc", "lr_ct", "lr_ds", "lr_sb"):
        v, e = T["ratio"][k]
        c = c + ((o[k] - v) / e) ** 2
    for k in ("ckm_s12", "ckm_s13", "ckm_s23"):
        v, ep, em = T["gauss"][k]
        c = c + _pull(o[k], v, ep, em) ** 2
    v, ep, em = T["gauss"]["ckm_delta_rad"]
    c = c + _pull(v + _wrap(o["ckm_delta_rad"] - v, 2 * np.pi), v, ep, em) ** 2
    return np.nan_to_num(c, nan=1e9, posinf=1e9)


def chi2_lepton_gauss(o, T):
    c = 0.0
    for k in ("lr_emu", "lr_mutau"):
        v, e = T["ratio"][k]
        c = c + ((o[k] - v) / e) ** 2
    v, e = T["ratio"]["r_dm"]
    c = c + ((o["r_dm"] - v) / e) ** 2
    for k in ("s12sq", "s13sq", "s23sq"):
        v, ep, em = T["gauss"][k]
        c = c + _pull(o[k], v, ep, em) ** 2
    v, ep, em = T["gauss"]["dcp_deg"]
    c = c + _pull(v + _wrap(o["dcp_deg"] - v, 360.0), v, ep, em) ** 2
    return np.nan_to_num(c, nan=1e9, posinf=1e9)


# ---------------------------------------------------------------- parameter vectors


def bounds(a):
    """Box for z = (u_tau (2), params)."""
    lo, hi = [0.0, 0.0], [1.0, 1.0]
    if isinstance(a, md.QuarkAssign):
        n = 4
        ex = md.n_extra(a.u) + md.n_extra(a.d)
    else:
        n = 2
        ex = md.n_extra(a.e) + a.nu[1] - 1
    lo += [LA_RANGE[0]] * n + [G_RANGE[0]] * ex
    hi += [LA_RANGE[1]] * n + [G_RANGE[1]] * ex
    return np.array(lo), np.array(hi)


def sector_obs(a, z, ordering=None):
    z = np.atleast_2d(z)
    tau = md.tau_from_unit(np.clip(z[:, :2], 0, 1))
    if isinstance(a, md.QuarkAssign):
        return md.quark_obs(a, tau, z[:, 2:])
    return md.lepton_obs(a, tau, z[:, 2:], ordering)


# ---------------------------------------------------------------- 1. synthetic data


def _gen_task(args):
    N, kind, idx, n, seed = args
    rng = np.random.default_rng(seed)
    cat = md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N)
    a = cat[idx]
    lo, hi = bounds(a)
    z = rng.uniform(lo, hi, size=(n, len(lo)))
    if kind == "q":
        o = sector_obs(a, z)
        y = np.stack([o["lr_uc"], o["lr_ct"], o["lr_ds"], o["lr_sb"], np.log(o["ckm_s12"]),
                      np.log(o["ckm_s13"]), np.log(o["ckm_s23"]), np.cos(o["ckm_delta_rad"]),
                      np.sin(o["ckm_delta_rad"])], 1)
    else:
        cols = []
        for oo in ORDERINGS:
            o = sector_obs(a, z, oo)
            if not cols:
                cols += [o["lr_emu"], o["lr_mutau"]]
            d = np.radians(o["dcp_deg"])
            cols += [o["s12sq"], o["s13sq"], o["s23sq"], np.cos(d), np.sin(d), np.log(o["r_dm"])]
        y = np.stack(cols, 1)
    y = np.nan_to_num(y, nan=-40.0, neginf=-40.0, posinf=10.0)
    y = np.clip(y, -40.0, 10.0)
    zp = np.zeros((n, 8 if kind == "q" else 7))
    zp[:, : z.shape[1]] = z
    return idx, zp.astype(np.float32), y.astype(np.float32)


def gen_sizes(N, kind):
    cat = md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N)
    total = {"q": 300_000, "l": 150_000}[kind]
    return len(cat), max(total // len(cat), 200)


def cmd_gen():
    import multiprocessing as mp
    tasks = []
    for N in mf.LEVELS:
        for kind in ("q", "l"):
            n_a, per = gen_sizes(N, kind)
            tasks += [(N, kind, i, per, [17, N, ord(kind), i]) for i in range(n_a)]
    log("gen tasks", len(tasks))
    out = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for t, res in zip(tasks, pool.imap(_gen_task, tasks, chunksize=8)):
            out.setdefault((t[0], t[1]), []).append(res)
    for (N, kind), lst in out.items():
        idx = np.concatenate([np.full(len(r[1]), r[0]) for r in lst])
        z = np.concatenate([r[1] for r in lst])
        y = np.concatenate([r[2] for r in lst])
        np.savez(os.path.join(ART, f"samples_{N}_{kind}.npz"), idx=idx, z=z, y=y)
        log("saved", N, kind, z.shape, y.shape)


# ---------------------------------------------------------------- 2. emulator


def embed(N, kind):
    """Assignment embedding matrix (n_assign, E): multi-hot row types per sector + rep / nu one-hot."""
    cat = md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N)
    vocab = sorted({(q, t) for q in mf.TRIPLETS[N] for t in md.row_types(N, q)})
    vi = {v: i for i, v in enumerate(vocab)}
    reps = list(mf.TRIPLETS[N])
    rows = []
    for a in cat:
        if kind == "q":
            e1, e2 = np.zeros(len(vocab)), np.zeros(len(vocab))
            for t in a.u:
                e1[vi[(a.qrep, t)]] = 1
            for t in a.d:
                e2[vi[(a.qrep, t)]] = 1
            rows.append(np.concatenate([e1, e2, np.eye(len(reps))[reps.index(a.qrep)]]))
        else:
            e1 = np.zeros(len(vocab))
            for t in a.e:
                e1[vi[(a.lrep, t)]] = 1
            nu = np.zeros(len(md.NU_WEIGHTS))
            nu[md.NU_WEIGHTS.index(a.nu[0])] = 1
            rows.append(np.concatenate([e1, nu, np.eye(len(reps))[reps.index(a.lrep)]]))
    return np.array(rows, np.float32)


def features_torch(zpad, emb, N):
    import torch
    u = zpad[:, :2].clamp(0, 1)
    x = u[:, 0] - 0.5
    ylo = torch.sqrt(1 - x ** 2)
    y = ylo + u[:, 1] * (md.YMAX - ylo)
    qa = torch.exp(-2 * np.pi * y / N)
    tf = torch.stack([x, 1 / y, qa * torch.cos(2 * np.pi * x / N), qa * torch.sin(2 * np.pi * x / N),
                      qa ** 2 * torch.cos(4 * np.pi * x / N), qa ** 2 * torch.sin(4 * np.pi * x / N)], 1)
    p = zpad[:, 2:] / 6.0
    return torch.cat([tf, p, emb], 1)


def make_mlp(n_in, n_out, width=384, depth=4):
    import torch.nn as nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def _train_task(args):
    N, kind, epochs = args
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(N * 10 + (kind == "q"))
    d = np.load(os.path.join(ART, f"samples_{N}_{kind}.npz"))
    idx, z, y = d["idx"], d["z"], d["y"]
    if len(z) > 600_000:  # A4 quarks: 200 per assignment x 12100; train on a random 600k subset
        keep = np.random.default_rng(0).choice(len(z), 600_000, replace=False)
        idx, z, y = idx[keep], z[keep], y[keep]
    zmax = 8 if kind == "q" else 7
    zp = z
    emb = torch.tensor(embed(N, kind))
    Z, Y, I = torch.tensor(zp), torch.tensor(y), torch.tensor(idx, dtype=torch.long)
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    n = len(Z)
    perm = torch.randperm(n)
    ntest = n // 20
    te, tr = perm[:ntest], perm[ntest:]
    model = make_mlp(6 + zmax - 2 + emb.shape[1], Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = epochs * (len(tr) // 1024)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        p = tr[torch.randperm(len(tr))]
        for b in range(len(tr) // 1024):
            j = p[b * 1024:(b + 1) * 1024]
            f = features_torch(Z[j], emb[I[j]], N)
            loss = (((model(f) - (Y[j] - ym) / ys)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            f = features_torch(Z[te], emb[I[te]], N)
            tl = (((model(f) - (Y[te] - ym) / ys)) ** 2).mean(0)
        print(f"[{N}{kind}] epoch {ep} train {loss.item():.4f} test {tl.mean().item():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    rmse = (tl.sqrt() * ys).tolist()
    torch.save({"state": model.state_dict(), "ym": ym, "ys": ys, "zmax": zmax, "rmse": rmse},
               os.path.join(ART, f"emu_{N}_{kind}.pt"))
    return N, kind, rmse


def cmd_train(epochs=8):
    import multiprocessing as mp
    tasks = [(N, k, epochs) for N in mf.LEVELS for k in ("q", "l")]
    res = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for N, kind, rmse in pool.imap_unordered(_train_task, tasks):
            res[f"{N}{kind}"] = rmse
            log("trained", N, kind, "test RMSE per output", np.round(rmse, 3).tolist())
    json.dump(res, open(os.path.join(HERE, "emulator_rmse.json"), "w"), indent=1)


# ---------------------------------------------------------------- 3. emulator ranking


def emu_chi2_torch(kind, out, T, ordering):
    """Search chi2 from (de-standardised) emulator outputs."""
    import torch

    def pull(pred, key):
        v, ep, em = T["gauss"][key]
        d = pred - v
        return torch.where(d > 0, d / ep, d / em)

    c = 0
    if kind == "q":
        for i, k in enumerate(("lr_uc", "lr_ct", "lr_ds", "lr_sb")):
            v, e = T["ratio"][k]
            c = c + ((out[:, i] - v) / e) ** 2
        for i, k in zip((4, 5, 6), ("ckm_s12", "ckm_s13", "ckm_s23")):
            c = c + pull(torch.exp(out[:, i]), k) ** 2
        dl = torch.atan2(out[:, 8], out[:, 7])
        v, ep, em = T["gauss"]["ckm_delta_rad"]
        w = torch.remainder(dl - v + np.pi, 2 * np.pi) - np.pi
        c = c + torch.where(w > 0, w / ep, w / em) ** 2
    else:
        for i, k in enumerate(("lr_emu", "lr_mutau")):
            v, e = T["ratio"][k]
            c = c + ((out[:, i] - v) / e) ** 2
        b = 2 if ordering == "NO" else 8
        for j, k in enumerate(("s12sq", "s13sq", "s23sq")):
            c = c + pull(out[:, b + j], k) ** 2
        dl = torch.rad2deg(torch.atan2(out[:, b + 4], out[:, b + 3]))
        v, ep, em = T["gauss"]["dcp_deg"]
        w = torch.remainder(dl - v + 180, 360) - 180
        c = c + torch.where(w > 0, w / ep, w / em) ** 2
        v, e = T["ratio"]["r_dm"]
        c = c + ((torch.exp(out[:, b + 5]) - v) / e) ** 2
    return c


def _rank_task(args):
    N, kind, ordering, starts, steps = args
    import torch
    torch.set_num_threads(2)
    from src.nulls import Universe
    T = targets(Universe.real(ordering), ordering)
    ck = torch.load(os.path.join(ART, f"emu_{N}_{kind}.pt"))
    emb_all = torch.tensor(embed(N, kind))
    zmax = ck["zmax"]
    model = make_mlp(6 + zmax - 2 + emb_all.shape[1], len(ck["ym"]))
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    cat = md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N)
    nA = len(cat)
    lo = np.zeros((nA, zmax), np.float32)
    hi = np.zeros((nA, zmax), np.float32)
    for i, a in enumerate(cat):
        l, h = bounds(a)
        lo[i, : len(l)], hi[i, : len(h)] = l, h
    best = np.full(nA, np.inf)
    bestz = np.zeros((nA, zmax), np.float32)
    g = torch.Generator().manual_seed(5)
    chunk = max(1, 40000 // starts)
    for c0 in range(0, nA, chunk):
        ids = np.arange(c0, min(nA, c0 + chunk))
        rep = np.repeat(ids, starts)
        L, H = torch.tensor(lo[rep]), torch.tensor(hi[rep])
        w = torch.logit(torch.rand(len(rep), zmax, generator=g).clamp(0.02, 0.98)).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=0.05)
        emb = emb_all[rep]
        for s in range(steps):
            z = L + (H - L) * torch.sigmoid(w)
            out = model(features_torch(z, emb, N)) * ck["ys"] + ck["ym"]
            c = emu_chi2_torch(kind, out, T, ordering)
            opt.zero_grad()
            torch.log1p(c).sum().backward()
            opt.step()
        with torch.no_grad():
            z = L + (H - L) * torch.sigmoid(w)
            c = emu_chi2_torch(kind, model(features_torch(z, emb, N)) * ck["ys"] + ck["ym"], T, ordering)
        c = c.numpy().reshape(len(ids), starts)
        j = c.argmin(1)
        best[ids] = c[np.arange(len(ids)), j]
        bestz[ids] = z.numpy().reshape(len(ids), starts, zmax)[np.arange(len(ids)), j]
    return N, kind, ordering, best, bestz


def cmd_rank(starts=4, steps=150):
    import multiprocessing as mp
    tasks = [(N, "q", "NO", starts, steps) for N in mf.LEVELS]
    tasks += [(N, "l", o, starts, steps) for N in mf.LEVELS for o in ORDERINGS]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for N, kind, o, best, bestz in pool.imap_unordered(_rank_task, tasks):
            np.savez(os.path.join(ART, f"rank_{N}_{kind}_{o}.npz"), chi2=best, z=bestz)
            log("ranked", N, kind, o, "n", len(best), "min emu chi2", float(best.min()))


# ---------------------------------------------------------------- 4. exact fits


def exact_chi2_fn(a, T, scorer, ordering):
    """Exact sector chi2: Gaussian ratio terms + the src.evaluate Scorer (chi2_soft) on mixing."""
    keys = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad") if isinstance(a, md.QuarkAssign) \
        else ("s12sq", "s13sq", "s23sq", "dcp_deg")
    rkeys = ("lr_uc", "lr_ct", "lr_ds", "lr_sb") if isinstance(a, md.QuarkAssign) else ("lr_emu", "lr_mutau")

    def f(z):
        o = sector_obs(a, z, ordering)
        c = 0.0
        for k in rkeys:
            v, e = T["ratio"][k]
            c += ((o[k][0] - v) / e) ** 2
        if not isinstance(a, md.QuarkAssign):
            v, e = T["ratio"]["r_dm"]
            c += ((o["r_dm"][0] - v) / e) ** 2
        pred = {k: float(o[k][0]) for k in keys}
        if not all(np.isfinite(list(pred.values()))) or not np.isfinite(c):
            return 1e9
        return c + scorer.chi2_soft(pred)

    return f


def fit_exact(a, T, scorer, ordering, rng, n_samp=4000, n_starts=3, extra_starts=(), maxfev=1200):
    from scipy.optimize import minimize
    lo, hi = bounds(a)
    z = rng.uniform(lo, hi, size=(n_samp, len(lo)))
    o = sector_obs(a, z, ordering)
    cg = chi2_quark_gauss(o, T) if isinstance(a, md.QuarkAssign) else chi2_lepton_gauss(o, T)
    starts = list(z[np.argsort(cg)[:n_starts]]) + [np.asarray(s, float) for s in extra_starts]
    f = exact_chi2_fn(a, T, scorer, ordering)
    fb = lambda x: f(np.clip(x, lo, hi)) + 10.0 * float(((x - np.clip(x, lo, hi)) ** 2).sum())
    best = (np.inf, None)
    for s in starts:
        r = minimize(fb, s, method="Nelder-Mead",
                     options={"maxfev": maxfev, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True})
        r2 = minimize(fb, r.x, method="Nelder-Mead",
                      options={"maxfev": maxfev // 2, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True})
        if r2.fun < best[0]:
            best = (float(r2.fun), np.clip(r2.x, lo, hi))
    return best


def _exact_task(args):
    N, kind, ordering, i, seed, z_emu = args
    from src.nulls import Universe
    u = Universe.real(ordering)
    T = targets(u, ordering)
    a = (md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N))[i]
    use = USE8[:4] if kind == "q" else USE8[4:]
    sc = u.scorer(use, ordering)
    rng = np.random.default_rng(seed)
    lo, _ = bounds(a)
    extra = [z_emu[: len(lo)]] if z_emu is not None else []
    c, z = fit_exact(a, T, sc, ordering, rng, extra_starts=extra)
    return N, kind, ordering, i, c, None if z is None else z.tolist()


def cmd_exact(top=15, rand=15):
    import multiprocessing as mp
    tasks, meta = [], []
    rng = np.random.default_rng(2026)
    for N in mf.LEVELS:
        for kind, ords in (("q", ("NO",)), ("l", ORDERINGS)):
            for o in ords:
                r = np.load(os.path.join(ART, f"rank_{N}_{kind}_{o}.npz"))
                c, z = r["chi2"], r["z"]
                order = np.argsort(c)
                topi = list(order[:top])
                rest = [i for i in range(len(c)) if i not in set(topi)]
                randi = list(rng.choice(rest, size=min(rand, len(rest)), replace=False)) if rest else []
                for i in topi + randi:
                    tasks.append((N, kind, o, int(i), [17, N, ord(kind), int(i)], z[i]))
                    meta.append({"N": N, "kind": kind, "ordering": o, "i": int(i), "emu_chi2": float(c[i]),
                                 "emu_rank": int(np.where(order == i)[0][0]),
                                 "subset": "top" if i in topi else "random"})
    log("exact tasks", len(tasks))
    res = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for N, kind, o, i, c, z in pool.imap_unordered(_exact_task, tasks, chunksize=1):
            res[(N, kind, o, i)] = (c, z)
    rows = []
    for m in meta:
        c, z = res[(m["N"], m["kind"], m["ordering"], m["i"])]
        cat = md.quark_catalog(m["N"]) if m["kind"] == "q" else md.lepton_catalog(m["N"])
        rows.append(dict(m, exact_chi2=c, z=z, label=cat[m["i"]].label(), k_cont=cat[m["i"]].k_cont))
    json.dump(rows, open(os.path.join(HERE, "exact_sector_fits.json"), "w"), indent=0)
    log("exact done", len(rows))


# ---------------------------------------------------------------- 5. joint common-tau fits


def tau_profile(a, T, scorer, ordering, grid, z0, rng, n_samp=400):
    """Profile chi2 over sector params at fixed tau grid points (u coordinates)."""
    from scipy.optimize import minimize
    lo, hi = bounds(a)
    f = exact_chi2_fn(a, T, scorer, ordering)
    out = np.empty(len(grid))
    zs = []
    for gi, ug in enumerate(grid):
        zz = rng.uniform(lo, hi, size=(n_samp, len(lo)))
        zz[:, :2] = ug
        zz[0, 2:] = z0[2:]
        o = sector_obs(a, zz, ordering)
        cg = chi2_quark_gauss(o, T) if isinstance(a, md.QuarkAssign) else chi2_lepton_gauss(o, T)
        s = zz[np.argmin(cg), 2:]
        fp = lambda p: f(np.concatenate([ug, np.clip(p, lo[2:], hi[2:])]))
        r = minimize(fp, s, method="Nelder-Mead", options={"maxfev": 300, "adaptive": True})
        out[gi] = r.fun
        zs.append(np.concatenate([ug, np.clip(r.x, lo[2:], hi[2:])]))
    return out, zs


def _profile_task(args):
    N, kind, ordering, i, z0, grid = args
    from src.nulls import Universe
    u = Universe.real(ordering)
    T = targets(u, ordering)
    a = (md.quark_catalog(N) if kind == "q" else md.lepton_catalog(N))[i]
    sc = u.scorer(USE8[:4] if kind == "q" else USE8[4:], ordering)
    prof, zs = tau_profile(a, T, sc, ordering, grid, np.array(z0), np.random.default_rng([N, i]))
    return N, kind, ordering, i, prof, [z.tolist() for z in zs]


def tau_grid(n=32, seed=11):
    """Quasi-uniform u grid over the fundamental domain (Sobol), denser at low Im tau by the map."""
    from scipy.stats import qmc
    return qmc.Sobol(2, seed=seed).random(n)


def cmd_joint(kq=6, kl=6):
    import multiprocessing as mp
    rows = json.load(open(os.path.join(HERE, "exact_sector_fits.json")))
    grid = tau_grid()
    tasks = []
    sel = {}
    for N in mf.LEVELS:
        q = sorted([r for r in rows if r["N"] == N and r["kind"] == "q"], key=lambda r: r["exact_chi2"])[:kq]
        sel[(N, "q", "NO")] = q
        for o in ORDERINGS:
            l_ = sorted([r for r in rows if r["N"] == N and r["kind"] == "l" and r["ordering"] == o],
                        key=lambda r: r["exact_chi2"])[:kl]
            sel[(N, "l", o)] = l_
        for (n_, kind, o), lst in sel.items():
            if n_ != N:
                continue
            for r in lst:
                g = np.concatenate([grid, [r["z"][:2]]])
                tasks.append((N, kind, o, r["i"], r["z"], g))
    log("profile tasks", len(tasks))
    prof = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for N, kind, o, i, p, zs in pool.imap_unordered(_profile_task, tasks, chunksize=1):
            prof[(N, kind, o, i)] = (np.array(p), zs)
    import pickle
    pickle.dump({"prof": prof, "sel": sel}, open(os.path.join(ART, "profiles.pkl"), "wb"))
    log("profiles done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"gen": cmd_gen, "train": cmd_train, "rank": cmd_rank, "exact": cmd_exact, "joint": cmd_joint}[cmd]()
