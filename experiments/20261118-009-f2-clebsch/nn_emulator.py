"""MLP emulator of continuous F2 and likelihood scan / inversion against the real data.

Continuous F2: U = V_e(th12e, th13e, th23e, delta_e; order)^dag D(phi) U_nu, U_nu one-hot over
the 5 LO patterns. Emulator: 13 inputs (th12e, th23e, th13e, cos/sin phi, cos/sin delta_e,
one-hot(5), order flag) -> 5 outputs (s12sq, s13sq, s23sq, n_cos, n_sin), where
delta = atan2(n_sin, n_cos) (the smooth invariants of src.mixing.extract_angles).
Training data: synthetic, exact model, priors th12e ~ U[0, 0.5], th23e ~ U[0, 0.3],
th13e ~ U[0, 0.03], phi, delta_e ~ U[0, 2pi). No measured number is used in training.

Then (per scale, delta_e = delta_q of that scale):
 1. dense emulator scan of (th12e, th23e, phi) per (pattern, order), th13e = 0, with the
    NuFIT tables (NuFIT chi^2 via f2lib.TableChi2): allowed regions chi2 <= 4 and the
    continuous best fit (verified with the exact model);
 2. lattice comparison: for each Clebsch lattice point (c12, c23) of tiers A and B, the
    emulator min over phi; compare to the exact enumeration in fits_<scale>_<O>.csv;
 3. inversion of 400 shifted-data PMNS targets per ordering (split-normal NuFIT draws,
    Gaussian chi^2) for TBM std with the emulator (grid argmin + Adam on the inputs),
    saved to artifacts/009/inversions.csv for pysr_distill.py.
Writes emulator_report.json, emulator_lattice_<scale>.csv.
"""
import json
import os
import time

import numpy as np
import torch

import f2lib as L
from src import data
from src.haar import split_normal

torch.set_num_threads(int(os.environ.get("F2_TORCH_THREADS", "2")))
torch.manual_seed(0)
ART = L.ROOT / "artifacts" / "009"
ART.mkdir(parents=True, exist_ok=True)
PM = ("s12sq", "s13sq", "s23sq", "dcp_deg")
t0 = time.time()
UNU = np.array([L.u_nu(p) for p in L.PATTERNS])


def log(*a):
    print(f"[nn {time.time() - t0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- exact batched model

def rot(i, j, t, ph=None):
    n = len(t)
    R = np.zeros((n, 3, 3), complex)
    k = 3 - i - j
    R[:, k, k] = 1
    c, s = np.cos(t), np.sin(t)
    e = np.exp(-1j * ph) if ph is not None else 1.0
    R[:, i, i] = c
    R[:, j, j] = c
    R[:, i, j] = s * e
    R[:, j, i] = -s * np.conj(e)
    return R


def exact(t12, t23, t13, phi, de, pat, order):
    R12, R23, R13 = rot(0, 1, t12), rot(1, 2, t23), rot(0, 2, t13, de)
    std = R23 @ R13 @ R12
    rev = R12 @ R13 @ R23
    Ve = np.where(order[:, None, None] == 0, std, rev)
    D = np.zeros((len(t12), 3, 3), complex)
    D[:, 0, 0] = 1
    D[:, 1, 1] = D[:, 2, 2] = np.exp(1j * phi)
    U = np.conj(np.swapaxes(Ve, 1, 2)) @ D @ UNU[pat]
    E = np.stack([U[:, i, j] for i, j in L.ENTRIES], -1)
    e1, e2, e3, m1, m2, m3, t3 = (E[:, k] for k in range(7))
    s12, s13, s23, d = L.fast_obs(E)
    c13sq = np.abs(e1) ** 2 + np.abs(e2) ** 2
    J = np.imag(e1 * m2 * np.conj(e2) * np.conj(m1))
    n_sin = J / c13sq
    n_cos = 0.5 * (np.abs(m1) ** 2 - s12 * (1 - s23) - (1 - s12) * s23 * s13)
    return np.stack([s12, s13, s23, n_cos, n_sin], -1), d


def features(t12, t23, t13, phi, de, pat, order):
    oh = np.eye(5)[pat]
    return np.column_stack([t12, t23, t13, np.cos(phi), np.sin(phi), np.cos(de), np.sin(de), oh, order])


def sample(rng, n):
    t12 = rng.uniform(0, 0.5, n)
    t23 = rng.uniform(0, 0.3, n)
    t13 = rng.uniform(0, 0.03, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    de = rng.uniform(0, 2 * np.pi, n)
    pat = rng.integers(0, 5, n)
    order = rng.integers(0, 2, n)
    y, _ = exact(t12, t23, t13, phi, de, pat, order)
    return features(t12, t23, t13, phi, de, pat, order), y


# ---------------------------------------------------------------- emulator

class MLP(torch.nn.Module):
    def __init__(self, d_in=13, d_out=5, w=128, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, w), torch.nn.SiLU()]
            d = w
        layers.append(torch.nn.Linear(d, d_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(rng):
    X, Y = sample(rng, 300_000)
    Xv, Yv = sample(rng, 50_000)
    xm, xs = X.mean(0), X.std(0) + 1e-9
    ym, ys = Y.mean(0), Y.std(0)
    f = lambda a, m, s: torch.tensor((a - m) / s, dtype=torch.float32)  # noqa: E731
    Xt, Yt, Xvt, Yvt = f(X, xm, xs), f(Y, ym, ys), f(Xv, xm, xs), f(Yv, ym, ys)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    epochs, bs = 30, 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xt) // bs))
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for k in range(len(Xt) // bs):
            idx = perm[k * bs:(k + 1) * bs]
            loss = torch.mean((net(Xt[idx]) - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad():
                vl = torch.mean((net(Xvt) - Yvt) ** 2).item()
            log(f"epoch {ep + 1} train {loss.item():.2e} val {vl:.2e}")
    norm = dict(xm=xm, xs=xs, ym=ym, ys=ys)
    return net, norm


class Emu:
    def __init__(self, net, norm):
        self.net, self.n = net, norm

    def raw(self, Xf):
        with torch.no_grad():
            z = self.net(torch.tensor((Xf - self.n["xm"]) / self.n["xs"], dtype=torch.float32)).numpy()
        return z * self.n["ys"] + self.n["ym"]

    def obs(self, Xf, chunk=200_000):
        out = np.concatenate([self.raw(Xf[a:a + chunk]) for a in range(0, len(Xf), chunk)])
        d = np.degrees(np.mod(np.arctan2(out[:, 4], out[:, 3]), 2 * np.pi))
        return out[:, 0], out[:, 1], out[:, 2], d


def accuracy(emu, rng):
    X, Y = sample(rng, 100_000)
    s12, s13, s23, d = emu.obs(X)
    dd = np.degrees(np.mod(np.arctan2(Y[:, 4], Y[:, 3]), 2 * np.pi))
    ddel = np.abs((d - dd + 180) % 360 - 180)
    D = Y[:, 0] * (1 - Y[:, 0]) * Y[:, 2] * (1 - Y[:, 2]) * Y[:, 1]
    sel = (Y[:, 1] > 0.015) & (Y[:, 1] < 0.03)  # near the measured theta13
    return {"rmse_s12sq": float(np.sqrt(np.mean((s12 - Y[:, 0]) ** 2))),
            "rmse_s13sq": float(np.sqrt(np.mean((s13 - Y[:, 1]) ** 2))),
            "rmse_s23sq": float(np.sqrt(np.mean((s23 - Y[:, 2]) ** 2))),
            "median_abs_ddelta_deg": float(np.median(ddel)),
            "rmse_s13sq_near_data": float(np.sqrt(np.mean((s13[sel] - Y[sel, 1]) ** 2))),
            "rmse_s12sq_near_data": float(np.sqrt(np.mean((s12[sel] - Y[sel, 0]) ** 2))),
            "median_abs_ddelta_deg_near_data": float(np.median(ddel[sel & (D > 1e-4)])),
            "nufit_1sigma_NO": {k: data.load_pmns("NO")[k]["err_plus"] for k in PM}}


# ---------------------------------------------------------------- scans

def scan(emu, scale, ckm, rows_exact):
    """Dense scan of (th12e, th23e, phi), th13e = 0, delta_e = delta_q, per pattern/order."""
    n12, n23, nph = 251, 61, 120
    t12 = np.linspace(0, 0.5, n12)
    t23 = np.linspace(0, 0.3, n23)
    ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    G12, G23, GP = (a.ravel() for a in np.meshgrid(t12, t23, ph, indexing="ij"))
    rep = {}
    lattice = []
    tabs = {o: L.TableChi2(o) for o in ("NO", "IO")}
    for p, pat in enumerate(L.PATTERNS):
        for order in (0, 1):
            n = len(G12)
            Xf = features(G12, G23, np.zeros(n), GP, np.full(n, ckm[3]), np.full(n, p), np.full(n, order))
            obs_grid = emu.obs(Xf)
            for o in ("NO", "IO"):
                T = tabs[o]
                chi = T(*obs_grid).reshape(n12, n23, nph)
                cmin = chi.min(axis=2)  # profile over phi
                key = f"{scale}|{o}|{pat}|{'std' if order == 0 else 'rev'}"
                i, j = np.unravel_index(np.argmin(cmin), cmin.shape)
                k = int(np.argmin(chi[i, j]))
                ex, d = exact(np.array([t12[i]]), np.array([t23[j]]), np.zeros(1), np.array([ph[k]]),
                              np.array([ckm[3]]), np.array([p]), np.array([order]))
                chi_ex = float(T(ex[0, 0], ex[0, 1], ex[0, 2], d[0]))
                allowed = cmin <= 4.0
                r = {"emu_best_chi2": float(cmin[i, j]), "exact_chi2_at_emu_best": chi_ex,
                     "th12e": float(t12[i]), "th23e": float(t23[j]), "phi_deg": float(np.degrees(ph[k])),
                     "th12e_over_thetaC": float(t12[i] / ckm[0]), "th23e_over_thetacb": float(t23[j] / ckm[2]),
                     "allowed_frac_chi2le4": float(allowed.mean())}
                if allowed.any():
                    ii, jj = np.where(allowed)
                    r["allowed_th12e_over_thetaC"] = [float(t12[ii].min() / ckm[0]), float(t12[ii].max() / ckm[0])]
                    r["allowed_th23e_over_thetacb"] = [float(t23[jj].min() / ckm[2]), float(t23[jj].max() / ckm[2])]
                rep[key] = r
                # lattice points of tiers A and B
                for c12 in L.C0:
                    for c23 in L.C0:
                        a12, a23 = float(c12) * ckm[0], float(c23) * ckm[2]
                        if a12 > 0.5 or a23 > 0.3 or (order == 1 and c23 == 0):
                            continue
                        m = len(ph)
                        Xl = features(np.full(m, a12), np.full(m, a23), np.zeros(m), ph, np.full(m, ckm[3]),
                                      np.full(m, p), np.full(m, order))
                        ce = float(T(*emu.obs(Xl)).min())
                        nm = f"{pat}|c12={c12}|c23={c23}|c13=0|{'std' if order == 0 else 'rev'}"
                        lattice.append(dict(scale=scale, ordering=o, name=nm, emu_chi2=ce,
                                            exact_chi2=rows_exact[o].get(nm, np.nan)))
    return rep, lattice


def invert(emu, ckm, rng, n=400):
    """Shifted-data targets -> best (th12e, th23e, phi) for TBM std via the emulator."""
    out = []
    for o in ("NO", "IO"):
        P = data.load_pmns(o)
        C = data.load_ckm()
        tg = np.column_stack([split_normal(rng, P[k]["value"], P[k]["err_plus"], abs(P[k]["err_minus"]), n) for k in PM])
        ckm_s = np.column_stack([np.arcsin(split_normal(rng, C[k]["value"], C[k]["err_plus"], C[k]["err_minus"], n))
                                 for k in ("s12", "s23")])
        # coarse grid argmin
        t12 = np.linspace(0, 0.5, 201)
        t23 = np.linspace(0, 0.3, 31)
        ph = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        G12, G23, GP = (a.ravel() for a in np.meshgrid(t12, t23, ph, indexing="ij"))
        m = len(G12)
        Xf = features(G12, G23, np.zeros(m), GP, np.full(m, ckm[3]), np.zeros(m, int), np.zeros(m))
        ob = np.column_stack(emu.obs(Xf))
        P_ = data.load_pmns(o)
        ep = np.array([P_[k]["err_plus"] for k in PM])
        em = np.array([abs(P_[k]["err_minus"]) for k in PM])
        x0 = []
        for t in tg:
            r = ob - t
            r[:, 3] = (r[:, 3] + 180) % 360 - 180
            c = np.sum((r / np.where(r >= 0, ep, em)) ** 2, 1)
            i = int(np.argmin(c))
            x0.append((G12[i], G23[i], GP[i]))
        x0 = np.array(x0)
        # Adam on the emulator inputs (batched over targets)
        prm = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        tgt = torch.tensor(tg, dtype=torch.float32)
        nm = {k: torch.tensor(v, dtype=torch.float32) for k, v in emu.n.items()}
        ept, emt = torch.tensor(ep, dtype=torch.float32), torch.tensor(em, dtype=torch.float32)
        opt = torch.optim.Adam([prm], 2e-3)
        de = float(ckm[3])
        for _ in range(400):
            t12_, t23_, ph_ = prm[:, 0], prm[:, 1], prm[:, 2]
            z = torch.zeros_like(t12_)
            X = torch.stack([t12_, t23_, z, torch.cos(ph_), torch.sin(ph_), z + np.cos(de), z + np.sin(de),
                             z + 1, z, z, z, z, z], 1)
            y = emu.net((X - nm["xm"]) / nm["xs"]) * nm["ys"] + nm["ym"]
            d = torch.rad2deg(torch.atan2(y[:, 4], y[:, 3]))
            pr = torch.stack([y[:, 0], y[:, 1], y[:, 2]], 1)
            r = pr - tgt[:, :3]
            rd = torch.remainder(d - tgt[:, 3] + 180, 360) - 180
            r = torch.cat([r, rd[:, None]], 1)
            loss_i = ((r / torch.where(r >= 0, ept, emt)) ** 2).sum(1)
            opt.zero_grad()
            loss_i.sum().backward()
            opt.step()
        x = prm.detach().numpy()
        ex, dex = exact(x[:, 0], x[:, 1], np.zeros(n), x[:, 2], np.full(n, de), np.zeros(n, int), np.zeros(n, int))
        for k in range(n):
            out.append(dict(ordering=o, s12sq=tg[k, 0], s13sq=tg[k, 1], s23sq=tg[k, 2], dcp_deg=tg[k, 3],
                            theta_C=ckm_s[k, 0], theta_cb=ckm_s[k, 1],
                            th12e=x[k, 0], th23e=x[k, 1], phi=float(np.mod(x[k, 2], 2 * np.pi)),
                            emu_loss=float(loss_i[k].item()),
                            exact_s13sq=ex[k, 1], exact_s12sq=ex[k, 0], exact_s23sq=ex[k, 2], exact_dcp=dex[k]))
    return out


def main():
    import csv
    rng = np.random.default_rng(7009)
    net, norm = train(rng)
    torch.save({"state": net.state_dict(), "norm": norm}, ART / "emulator.pt")
    emu = Emu(net, norm)
    report = {"accuracy": accuracy(emu, rng)}
    log("accuracy", json.dumps(report["accuracy"]))
    for scale in ("MZ", "GUT"):
        ckm = L.ckm_at_scale(scale)
        rows_exact = {}
        for o in ("NO", "IO"):
            with open(L.HERE / f"fits_{scale}_{o}.csv") as fh:
                rows_exact[o] = {r["name"]: float(r["chi2_fit"]) for r in csv.DictReader(fh)}
        rep, lat = scan(emu, scale, ckm, rows_exact)
        report[f"scan_{scale}"] = rep
        with open(L.HERE / f"emulator_lattice_{scale}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lat[0]))
            w.writeheader()
            w.writerows(lat)
        e = np.array([[r["emu_chi2"], r["exact_chi2"]] for r in lat])
        ok = np.isfinite(e).all(1)
        both = lambda a, b: int(np.sum(a & b))  # noqa: E731
        report[f"lattice_agreement_{scale}"] = {
            "n_points": int(ok.sum()),
            "exact_chi2_le4": int(np.sum(e[ok, 1] <= 4)), "emu_chi2_le4": int(np.sum(e[ok, 0] <= 4)),
            "both_le4": both(e[ok, 1] <= 4, e[ok, 0] <= 4),
            "exact_chi2_le9": int(np.sum(e[ok, 1] <= 9)), "emu_chi2_le9": int(np.sum(e[ok, 0] <= 9)),
            "both_le9": both(e[ok, 1] <= 9, e[ok, 0] <= 9),
            "median_abs_diff_where_exact_lt25": float(np.median(np.abs(np.diff(e[ok & (e[:, 1] < 25)], axis=1))))}
        log(scale, json.dumps(report[f"lattice_agreement_{scale}"]))
    inv = invert(emu, L.ckm_at_scale("MZ"), rng)
    with open(ART / "inversions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)
    a = np.array([[r["s13sq"], r["exact_s13sq"]] for r in inv])
    report["inversion_s13sq_rmse_exact_vs_target"] = float(np.sqrt(np.mean((a[:, 0] - a[:, 1]) ** 2)))
    with open(L.HERE / "emulator_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    log("done")


if __name__ == "__main__":
    main()
