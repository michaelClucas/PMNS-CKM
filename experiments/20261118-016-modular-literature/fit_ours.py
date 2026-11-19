"""Refit the minimal 2O quark-lepton model (arXiv:2307.14926 / 2409.15823) to our data.

Observables (all dimensionless, N = 15):
  PMNS s12sq, s13sq, s23sq, dcp_deg  -> src.evaluate (NuFIT 6.1 tables, 2D s23-delta), NO
  CKM  ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad -> src.evaluate (PDG global fit)
  mass ratios at M_GUT: m_u/m_c, m_c/m_t, m_d/m_s, m_s/m_b, m_e/m_mu, m_mu/m_tau (6)
  Delta m^2_21 / Delta m^2_31 (1)
The model lives at M_GUT = 2e16 GeV (SUSY, tan beta = 5, M_SUSY = 10 TeV as in 2409.15823). Data are
run up with src.rg (MSSM, one-loop Yukawas). Mixing predictions are brought to M_Z by the data's
own running shift (pred_MZ = pred_GUT + data_MZ - data_GUT; angles via sin, delta additive), then
scored by src.evaluate. Mass ratios and the Delta m^2 ratio are scored at M_GUT with exp 014's noise
model (each mass max(PDG, 2%), in quadrature, log10), because src.evaluate has no mass-ratio keys
for in-sector ratios, no lepton-mass ratios, no Delta m^2, and no theory floor (src gap, README).
Error variants: "ours" (2% floor) and "loose" (2409.15823's relative errors on r_uc, r_ct, r_ds,
r_sb: 31%, 4.3%, 12%, 5.5%; diagnostic only).

Free parameters (k_continuous = 10): Re tau, Im tau, gE2/gE1, gE3/gE1, gu2/gu1, gu3/gu1,
gd2/gd1, gd3/gd1, gd4/gd1, gd5/gd1 (real, either sign). Overall scales drop out of every scored
observable (the neutrino scale only sets Delta m^2_31, which is not scored).

Usage: python fit_ours.py [ours|loose] [n_proc]
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src import data, rg, stats  # noqa: E402
from src.evaluate import Scorer, evaluate  # noqa: E402
from model_2o import observables  # noqa: E402

MU_GUT, TB, MS = 2e16, 5.0, 1e4
PMNS_USE = ("s12sq", "s13sq", "s23sq", "dcp_deg")
CKM_USE = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")
RATIOS = ("r_uc", "r_ct", "r_ds", "r_sb", "r_emu", "r_mutau")
LOOSE = {"r_uc": 0.31, "r_ct": 0.043, "r_ds": 0.12, "r_sb": 0.055}


def rel(r):
    return 0.5 * (r["err_plus"] + r["err_minus"]) / r["value"]


def build_data(variant="ours"):
    c_g = rg.ckm_at(MU_GUT, model="MSSM", tan_beta=TB, m_susy=MS)
    c_z = rg.ckm_at(91.1876, model="MSSM", tan_beta=TB, m_susy=MS)
    p_g = rg.pmns_at(MU_GUT, "NO", model="MSSM", tan_beta=TB, m_susy=MS)
    p_z = rg.pmns_at(91.1876, "NO", model="MSSM", tan_beta=TB, m_susy=MS)
    m = data.load_masses()
    em = {f: max(rel(m["m_" + f]), 0.02) for f in ("u", "d", "s", "c", "b", "e", "mu", "tau")}
    em["t"] = max(rel(m["m_t_direct"]), 0.02)
    r = c_g["ratios"]
    centre = {"r_uc": r["mu_over_mc"], "r_ct": r["mc_over_mt"], "r_ds": r["md_over_ms"], "r_sb": r["ms_over_mb"],
              "r_emu": r["me_over_mmu"], "r_mutau": r["mmu_over_mtau"]}
    pairs = {"r_uc": ("u", "c"), "r_ct": ("c", "t"), "r_ds": ("d", "s"), "r_sb": ("s", "b"),
             "r_emu": ("e", "mu"), "r_mutau": ("mu", "tau")}
    sig = {k: np.hypot(em[a], em[b]) for k, (a, b) in pairs.items()}  # relative error of the ratio
    if variant == "loose":
        sig.update(LOOSE)
    pn = data.load_pmns("NO")
    dm_rel = np.hypot(rel(pn["dm21sq_1e-5eV2"]), rel(pn["dm3l_sq_1e-3eV2"]))
    shift = {"pmns_s12": np.sin(p_z["theta12"]) - np.sin(p_g["theta12"]),
             "pmns_s13": np.sin(p_z["theta13"]) - np.sin(p_g["theta13"]),
             "pmns_s23": np.sin(p_z["theta23"]) - np.sin(p_g["theta23"]),
             "pmns_d": p_z["delta"] - p_g["delta"],
             "ckm_s12": np.sin(c_z["theta12"]) / np.sin(c_g["theta12"]),
             "ckm_s13": np.sin(c_z["theta13"]) / np.sin(c_g["theta13"]),
             "ckm_s23": np.sin(c_z["theta23"]) / np.sin(c_g["theta23"]),
             "ckm_d": c_z["delta"] - c_g["delta"]}
    return {"centre": centre, "sig": sig, "dm_ratio": (p_g["dm21_over_dm3l"], float(dm_rel)), "shift": shift,
            "gut_ckm": {k: c_g[k] for k in ("theta12", "theta13", "theta23", "delta")},
            "gut_pmns": {k: p_g[k] for k in ("theta12", "theta13", "theta23", "delta")}}


def to_mz(o, D):
    s = D["shift"]
    pred = {}
    for i, k in (("12", "s12sq"), ("13", "s13sq"), ("23", "s23sq")):
        pred[k] = (np.sqrt(o[k]) + s["pmns_s" + i]) ** 2
    pred["dcp_deg"] = np.degrees(o["dl"] + s["pmns_d"]) % 360.0
    pred["ckm_s12"] = np.sin(o["th12q"]) * s["ckm_s12"]
    pred["ckm_s13"] = np.sin(o["th13q"]) * s["ckm_s13"]
    pred["ckm_s23"] = np.sin(o["th23q"]) * s["ckm_s23"]
    pred["ckm_delta_rad"] = (o["dq"] + s["ckm_d"] + np.pi) % (2 * np.pi) - np.pi
    return pred


class Chi2:
    def __init__(self, D):
        self.D = D
        self.sp = Scorer(PMNS_USE, "NO")
        self.sc = Scorer(CKM_USE, "NO")

    def parts(self, o):
        D = self.D
        pred = to_mz(o, D)
        mass = {k: np.log(o[k] / D["centre"][k]) / D["sig"][k] for k in RATIOS}
        v, e = D["dm_ratio"]
        mass["dm_ratio"] = np.log(o["dm_ratio"] / v) / e
        return pred, mass, self.sp.chi2_soft(pred), self.sc.chi2_soft(pred)


LEP_N = ["rE2", "rE3"]
QRK_N = ["ru2", "ru3", "rd2", "rd3", "rd4", "rd5"]
BASE = dict(rE2=0.698, rE3=86.4, ru2=121.9, ru3=0.487, rd2=45.1, rd3=1.16, rd4=0.0132, rd5=0.00361)


def _obs(p):
    try:
        o = observables(p)
    except np.linalg.LinAlgError:
        return None
    if not all(np.isfinite(o[k]) and o[k] > 0 for k in RATIOS + ("dm_ratio",)):
        return None
    return o


def resid_lep(x, tau, C):
    p = dict(BASE, re=tau.real, im=tau.imag, rE2=x[0], rE3=x[1])
    o = _obs(p)
    if o is None:
        return np.full(4, 1e3)
    _, mass, cp, _ = C.parts(o)
    return np.array([mass["r_emu"], mass["r_mutau"], mass["dm_ratio"], np.sqrt(max(cp, 0))])


def resid_qrk(x, tau, C):
    p = dict(BASE, re=tau.real, im=tau.imag, **dict(zip(QRK_N, x)))
    o = _obs(p)
    if o is None:
        return np.full(5, 1e3)
    _, mass, _, cc = C.parts(o)
    return np.array([mass[k] for k in ("r_uc", "r_ct", "r_ds", "r_sb")] + [np.sqrt(max(cc, 0))])


def resid_all(x, C):
    p = dict(BASE, re=x[0], im=x[1], **dict(zip(LEP_N + QRK_N, x[2:])))
    if p["im"] < 0.6 or abs(p["re"]) > 0.75:
        return np.full(10, 1e3)
    o = _obs(p)
    if o is None:
        return np.full(10, 1e3)
    _, mass, cp, cc = C.parts(o)
    return np.array([mass[k] for k in RATIOS] + [mass["dm_ratio"], np.sqrt(max(cp, 0)), np.sqrt(max(cc, 0)), 0.0])


def best_lsq(fun, starts, args):
    best = None
    for x0 in starts:
        try:
            r = least_squares(fun, x0, args=args, method="trf", max_nfev=1500, x_scale="jac")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def lep_profile(tau, C, rng):
    starts = [np.array([BASE["rE2"], BASE["rE3"]])]
    for _ in range(2):
        starts.append(rng.choice([-1, 1], 2) * 10 ** rng.uniform([-2, 0], [1, 3]))
    r = best_lsq(resid_lep, starts, (tau, C))
    return 2 * r.cost, r.x


def qrk_profile(tau, C, rng, n_rand=2):
    b = np.array([BASE[n] for n in QRK_N])
    starts = [b, np.array([121.1, 0.441, 38.8, 0.870, 0.0101, 0.00277])]
    for _ in range(n_rand):
        starts.append(b * rng.choice([-1, 1], 6) * 10 ** rng.uniform(-0.7, 0.7, 6))
    r = best_lsq(resid_qrk, starts, (tau, C))
    return 2 * r.cost, r.x


def worker(args):
    taus, variant, seed, which = args
    D = build_data(variant)
    C = Chi2(D)
    rng = np.random.default_rng(seed)
    out = []
    for t in taus:
        if which == "lep":
            c, x = lep_profile(t, C, rng)
        else:
            c, x = qrk_profile(t, C, rng)
        out.append((t.real, t.imag, float(c), [float(v) for v in x]))
    return out


def tau_grid():
    pts = []
    for im in np.concatenate([np.arange(0.86, 1.5, 0.04), np.arange(1.5, 3.01, 0.1)]):
        for re in np.arange(-0.5, 0.5001, 0.04):
            if abs(re + 1j * im) >= 0.999:
                pts.append(complex(round(re, 4), round(im, 4)))
    return pts


def run(variant, nproc):
    from multiprocessing import Pool
    t0 = time.time()
    grid = tau_grid()
    print(f"[{variant}] lepton profile on {len(grid)} tau points", flush=True)
    chunks = [(grid[i::nproc], variant, 100 + i, "lep") for i in range(nproc)]
    with Pool(nproc) as pool:
        lep = [r for part in pool.map(worker, chunks) for r in part]
    lep.sort(key=lambda r: r[2])
    print(f"[{variant}] lepton min chi2 {lep[0][2]:.2f} at tau {lep[0][0]:+.3f}+{lep[0][1]:.3f}i "
          f"({time.time() - t0:.0f}s)", flush=True)
    lmin = lep[0][2]
    sel = [complex(r[0], r[1]) for r in lep if r[2] < lmin + 30][:24]
    print(f"[{variant}] quark profile on {len(sel)} tau points with lepton chi2 < min + 30", flush=True)
    chunks = [(sel[i::nproc], variant, 200 + i, "qrk") for i in range(nproc)]
    with Pool(nproc) as pool:
        qrk = [r for part in pool.map(worker, chunks) for r in part]
    lepd = {(r[0], r[1]): r for r in lep}
    comb = sorted(((lepd[(q[0], q[1])][2] + q[2], q, lepd[(q[0], q[1])]) for q in qrk), key=lambda z: z[0])
    print(f"[{variant}] profile-sum min {comb[0][0]:.2f} at tau {comb[0][1][0]:+.3f}+{comb[0][1][1]:.3f}i "
          f"({time.time() - t0:.0f}s)", flush=True)
    # joint polish of the 15 best profile points plus the published combined point
    D = build_data(variant)
    C = Chi2(D)
    starts = [np.array([z[1][0], z[1][1]] + z[2][3] + z[1][3]) for z in comb[:15]]
    starts.append(np.array([-0.202, 1.071] + [BASE[n] for n in LEP_N + QRK_N]))
    best = None
    for x0 in starts:
        r = least_squares(resid_all, x0, args=(C,), method="trf", max_nfev=4000, x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    p = dict(BASE, re=x[0], im=x[1], **dict(zip(LEP_N + QRK_N, x[2:])))
    o = observables(p)
    pred, mass, cp, cc = C.parts(o)
    ev = evaluate(pred, "NO", k=10, use=PMNS_USE + CKM_USE)
    chi2_mass = float(sum(v ** 2 for v in mass.values()))
    chi2_tot = float(ev.chi2 + chi2_mass)
    chi2_glob = float(ev.chi2_global + chi2_mass)
    N, kc = 15, 10
    res = {"variant": variant, "tau": [float(x[0]), float(x[1])], "params": {k: float(v) for k, v in p.items()},
           "chi2_evaluate_part": float(ev.chi2), "chi2_mass_part": chi2_mass, "chi2": chi2_tot,
           "chi2_global": chi2_glob, "N": N, "k_continuous": kc, "dof": N - kc,
           "p": float(stats.p_value(chi2_tot, N, kc)), "p_global": float(stats.p_value(chi2_glob, N, kc)),
           "evaluate_pulls": {k: float(v) for k, v in ev.pulls.items()},
           "mass_pulls": {k: float(v) for k, v in mass.items()},
           "pred_gut": {k: float(o[k]) for k in RATIOS + ("dm_ratio", "s12sq", "s13sq", "s23sq", "dl", "th12q",
                                                          "th13q", "th23q", "dq", "m1", "m2", "m3")},
           "pred_mz": {k: float(v) for k, v in pred.items()},
           "lepton_profile_min": lep[0][2], "lepton_profile_min_tau": lep[0][:2],
           "profile_sum_min": comb[0][0], "data_gut": {k: (float(v), float(D["sig"][k])) for k, v in D["centre"].items()},
           "dm_ratio_data": [float(v) for v in D["dm_ratio"]], "wall_s": time.time() - t0}
    res["max_abs_pull"] = max(abs(v) for v in list(res["evaluate_pulls"].values()) + list(res["mass_pulls"].values()))
    (HERE / f"fit_ours_{variant}.json").write_text(json.dumps(res, indent=1))
    (HERE / f"profile_{variant}.json").write_text(json.dumps({"lep": lep, "qrk": qrk}, indent=0))
    print(json.dumps({k: res[k] for k in ("tau", "chi2", "chi2_global", "N", "k_continuous", "p", "p_global",
                                          "max_abs_pull")}), flush=True)
    print("evaluate pulls:", {k: round(v, 2) for k, v in res["evaluate_pulls"].items()})
    print("mass pulls:", {k: round(v, 2) for k, v in res["mass_pulls"].items()}, flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ours", int(sys.argv[2]) if len(sys.argv) > 2 else 2)
