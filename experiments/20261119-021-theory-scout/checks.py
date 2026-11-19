"""Quick viability checks for the proposals in proposals.md. Writes checks.json. Single process, ~2-5 min.

Run from this folder: ../../.venv/Scripts/python checks.py

C1  Clebsch double-ratio filter (P1, P2, P3): SU(5)/Pati-Salam with fixed Clebsch factors in the 1-2 block of
    Y_d and Y_e = (C o Y_d)^T. R = (m_e/m_mu)/(m_d/m_s) is RG-invariant to < 0.1% in the SM (light Yukawas only)
    and the tan(beta)-enhanced SUSY thresholds are generation-universal for the first two generations, so they
    cancel in m_d/m_s. Scan the texture family, keep points matching m_d/m_s, ask which Clebsch triples give R
    within 2 sigma, and what charged-lepton angle theta12e they imply.
C2  Universal four-zero Hermitian texture (P4): one shape ratio r = Btilde/A shared by u, d, e, nu; quark side
    (r, 2 phases, 6 masses as nuisances with a 2% floor) against CKM; lepton side (same r, m1, 2 phases)
    against NuFIT.
C3  QLC with symmetry-fixed phases (P5): U = V_CKM^dag diag(1, e^{ia}, e^{ib}) U_BM with a, b in {0, pi/2, pi, 3pi/2}.
"""
import itertools
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_ckm, load_masses, load_pmns, pmns_delta_chi2  # noqa: E402
from src.mixing import extract_angles, standard_matrix  # noqa: E402
from src.rg import derived_yukawas_mz  # noqa: E402

OUT = {}
Y = derived_yukawas_mz()
FLOOR = 0.02


def ckm_data():
    c = load_ckm()
    return {k: (c[k]["value"], 0.5 * (c[k]["err_plus"] + c[k]["err_minus"])) for k in ("s12", "s13", "s23", "delta_rad")}


# ------------------------------------------------------------------ C1
def c1():
    rho = Y["y_d"] / Y["y_s"]
    rho_err = rho * math.sqrt(2) * FLOOR
    ms = load_masses()
    pdg_rel = math.hypot(ms["m_d"]["err_plus"] / ms["m_d"]["value"], ms["m_s"]["err_plus"] / ms["m_s"]["value"])
    ratio_e = Y["y_e"] / Y["y_mu"]
    R = ratio_e / rho
    clebsch = [1 / 6, 2 / 9, 1 / 3, 1 / 2, 2 / 3, 1, 3 / 2, 2, 3, 9 / 2, 6]
    # Y_d = [[0, b], [a, c]] (rows = left-handed index), c = 1; left rotation from Y Y^dag
    g = np.logspace(-3, 0.3, 500)
    A, B = np.meshgrid(g, g, indexing="ij")
    def sv_angle(M):
        H = M @ np.swapaxes(M, -1, -2)
        w, v = np.linalg.eigh(H)
        sv = np.sqrt(np.clip(w, 0, None))
        return sv[..., 0] / sv[..., 1], np.abs(np.arctan2(v[..., 0, 1], v[..., 1, 1]))  # angle of heavy eigvec
    Md = np.zeros(A.shape + (2, 2)); Md[..., 0, 1] = B; Md[..., 1, 0] = A; Md[..., 1, 1] = 1.0
    r_d, t_d = sv_angle(Md)
    keep = np.abs(r_d - rho) < 2 * rho_err
    tC = math.asin(ckm_data()["s12"][0])
    win = keep & (t_d > tC - 0.05) & (t_d < tC + 0.05)   # theta12d^L near theta_C (up-sector shift <= sqrt(mu/mc))
    res = dict(rho_md_ms_MZ=rho, rho_err_2pct_floor=rho_err, rho_rel_err_pdg=pdg_rel, me_mmu_MZ=ratio_e,
               R_double_ratio=R, R_err=R * math.sqrt(2) * FLOOR, R_err_pdg=R * pdg_rel,
               n_texture_points_matching_rho=int(keep.sum()), n_in_thetaC_window=int(win.sum()), triples={})
    a, b = A[win], B[win]
    tdw = t_d[win]
    surv = []
    for c12, c21, c22 in itertools.product(clebsch, repeat=3):
        Me = np.zeros(a.shape + (2, 2)); Me[..., 0, 1] = c12 * b; Me[..., 1, 0] = c21 * a; Me[..., 1, 1] = c22
        Me = np.swapaxes(Me, -1, -2)            # Y_e = (C o Y_d)^T
        r_e, t_e = sv_angle(Me)
        pull = (r_e - ratio_e) / ratio_e / (math.sqrt(2) * FLOOR)  # m_e/m_mu at the texture's m_d/m_s, i.e. a pull on R
        ok = np.abs(pull) < 2
        if ok.any():
            surv.append(dict(c12=c12, c21=c21, c22=c22, n_ok=int(ok.sum()), min_abs_pull=float(np.abs(pull).min()),
                             theta12d_range=[float(tdw[ok].min()), float(tdw[ok].max())],
                             theta12e_range=[float(t_e[ok].min()), float(t_e[ok].max())]))
    res["n_triples"] = len(clebsch) ** 3
    res["n_surviving_triples"] = len(surv)
    # what U_nu needs: TBM/GR-type needs s13 = sin(theta12e) s23 -> theta12e ~ 0.22; TM1 (CSD3) needs theta12e small
    s13 = math.sqrt(load_pmns("NO")["s13sq"]["value"]); s23 = math.sqrt(load_pmns("NO")["s23sq"]["value"])
    res["theta12e_needed_TBM"] = math.asin(s13 / s23)
    res["n_surv_theta12e_TBM_window"] = sum(1 for s in surv if s["theta12e_range"][0] < res["theta12e_needed_TBM"] * 1.05
                                            and s["theta12e_range"][1] > res["theta12e_needed_TBM"] * 0.95)
    res["n_surv_theta12e_small_lt_0p05"] = sum(1 for s in surv if s["theta12e_range"][0] < 0.05)
    res["surviving"] = sorted(surv, key=lambda s: s["min_abs_pull"])
    # well-known fixed sets
    named = {"GJ (1,1,3)": (1, 1, 3), "minimal SU(5) (1,1,1)": (1, 1, 1), "(1/2,1/2,6)": (1 / 2, 1 / 2, 6),
             "(1/2,1/2,9/2)": (1 / 2, 1 / 2, 9 / 2), "(1,1/2,9/2)": (1, 1 / 2, 9 / 2), "(1,1/2,6)": (1, 1 / 2, 6), "1305.6612 AGMS (c21=6 in this convention)": (1 / 2, 6, 6)}
    nm = {}
    for lab, (c12, c21, c22) in named.items():
        Me = np.zeros(a.shape + (2, 2)); Me[..., 0, 1] = c12 * b; Me[..., 1, 0] = c21 * a; Me[..., 1, 1] = c22
        r_e, t_e = sv_angle(np.swapaxes(Me, -1, -2))
        pull = (r_e - ratio_e) / ratio_e / (math.sqrt(2) * FLOOR)
        i = int(np.argmin(np.abs(pull)))
        nm[lab] = dict(best_pull=float(pull[i]), pull_range=[float(pull.min()), float(pull.max())],
                       theta12e_at_best=float(t_e[i]), theta12d_at_best=float(tdw[i]))
    res["named"] = nm
    return res


# ------------------------------------------------------------------ C2
def fourzero_O(lams, r):
    """Real orthogonal O with M = O diag(lams) O^T, M = [[0,C,0],[C,Bt,Bm],[0,Bm,Am]], Bt = r Am. None if invalid."""
    l1, l2, l3 = lams
    Am = (l1 + l2 + l3) / (1 + r)
    if Am <= 0:
        return None
    Bt = r * Am
    C2 = -l1 * l2 * l3 / Am
    e2 = l1 * l2 + l1 * l3 + l2 * l3
    B2 = Bt * Am - C2 - e2
    if C2 < 0 or B2 < 0:
        return None
    M = np.array([[0, math.sqrt(C2), 0], [math.sqrt(C2), Bt, math.sqrt(B2)], [0, math.sqrt(B2), Am]])
    w, v = np.linalg.eigh(M)
    return v  # ascending eigenvalues: l1 < 0 < l2 < l3


def mix(Oa, Ob, p1, p2):
    return Oa.T @ np.diag([1, np.exp(1j * p1), np.exp(1j * p2)]) @ Ob


def c2(n_starts=60, seed=21):
    rng = np.random.default_rng(seed)
    ck = ckm_data()
    yq = np.array([Y["y_u"], Y["y_c"], Y["y_t"], Y["y_d"], Y["y_s"], Y["y_b"]])
    yq = yq / yq[2]
    def resid_q(x, shared=True):
        r_u, r_d = (x[0], x[0]) if shared else (x[0], x[1])
        o = 1 if shared else 2
        p1, p2 = x[o], x[o + 1]
        lm = x[o + 2:o + 8]
        m = yq * np.exp(lm)
        Ou = fourzero_O((-m[0], m[1], m[2]), r_u); Od = fourzero_O((-m[3], m[4], m[5]), r_d)
        if Ou is None or Od is None:
            return np.full(10, 1e3)
        t12, t13, t23, d = extract_angles(mix(Ou, Od, p1, p2))
        pred = [math.sin(t12), math.sin(t13), math.sin(t23)]
        rr = [(pred[i] - ck[k][0]) / ck[k][1] for i, k in enumerate(("s12", "s13", "s23"))]
        dd = (d - ck["delta_rad"][0] + math.pi) % (2 * math.pi) - math.pi
        rr.append(dd / ck["delta_rad"][1])
        rr += list(lm / FLOOR)
        return np.array(rr)
    out = {}
    for shared in (True, False):
        best = None
        for _ in range(n_starts):
            x0 = ([rng.uniform(0.01, 3)] if shared else list(rng.uniform(0.01, 3, 2))) + list(rng.uniform(0, 2 * math.pi, 2)) + [0.0] * 6
            try:
                f = least_squares(lambda x: resid_q(x, shared), x0, method="lm", max_nfev=3000)
            except Exception:
                continue
            c = float(np.sum(f.fun ** 2))
            if best is None or c < best[0]:
                best = (c, f.x.tolist(), f.fun.tolist())
        out["quark_shared_r" if shared else "quark_separate_r"] = dict(chi2=best[0], x=best[1], pulls=best[2],
                                                                     N=10, k=(9 if shared else 10), dof=(1 if shared else 0))
    # lepton side, NO, r fixed to the quark shared-r best fit
    r_q = out["quark_shared_r"]["x"][0]
    yl = np.array([Y["y_e"], Y["y_mu"], Y["y_tau"]])
    p = load_pmns("NO")
    ms = load_masses()["neutrino_splittings"]["IC23_woSKatm"]["NO"]
    dm21, dm31 = ms["dm21sq"]["value"], ms["dm31sq"]["value"]
    def chi_l(x, r):
        m1, p1, p2 = 10 ** x[0], x[1], x[2]
        mn = np.array([m1, math.sqrt(m1 ** 2 + dm21), math.sqrt(m1 ** 2 + dm31)])
        Oe = fourzero_O((-yl[0], yl[1], yl[2]), r); On = fourzero_O((-mn[0], mn[1], mn[2]), r)
        if Oe is None or On is None:
            return 1e4
        t12, t13, t23, d = extract_angles(mix(Oe, On, p1, p2))
        return pmns_delta_chi2("NO", s12sq=math.sin(t12) ** 2, s13sq=math.sin(t13) ** 2, s23sq=math.sin(t23) ** 2,
                               dcp_deg=math.degrees(d) % 360)
    from scipy.optimize import minimize
    res_l = {}
    for lab, r in (("r_from_quarks", r_q), ("r_free", None)):
        best = None
        for _ in range(n_starts):
            if r is None:
                x0 = [rng.uniform(-4, -1), rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0.01, 3)]
                f = minimize(lambda x: chi_l(x[:3], x[3]), x0, method="Nelder-Mead", options=dict(maxiter=3000))
            else:
                x0 = [rng.uniform(-4, -1), rng.uniform(0, 6.28), rng.uniform(0, 6.28)]
                f = minimize(lambda x: chi_l(x, r), x0, method="Nelder-Mead", options=dict(maxiter=3000))
            if best is None or f.fun < best[0]:
                best = (float(f.fun), f.x.tolist())
        res_l[lab] = dict(chi2=best[0], x=best[1], N=4, k=(3 if r is not None else 4))
    out["lepton_NO"] = res_l
    return out


# ------------------------------------------------------------------ C3
def c3():
    ck = ckm_data()
    V = standard_matrix(math.asin(ck["s12"][0]), math.asin(ck["s13"][0]), math.asin(ck["s23"][0]), ck["delta_rad"][0])
    s = 1 / math.sqrt(2)
    UBM = np.array([[s, s, 0], [-0.5, 0.5, s], [0.5, -0.5, s]])
    rows = []
    for a, b in itertools.product([0, 0.5, 1, 1.5], repeat=2):
        for form, U in (("Vdag.P.UBM", V.conj().T @ np.diag([1, np.exp(1j * math.pi * a), np.exp(1j * math.pi * b)]) @ UBM),
                        ("UBM.P.Vdag", UBM @ np.diag([1, np.exp(1j * math.pi * a), np.exp(1j * math.pi * b)]) @ V.conj().T)):
            t12, t13, t23, d = extract_angles(U)
            obs = dict(s12sq=math.sin(t12) ** 2, s13sq=math.sin(t13) ** 2, s23sq=math.sin(t23) ** 2, dcp_deg=math.degrees(d) % 360)
            for o in ("NO", "IO"):
                rows.append(dict(form=form, a_pi=a, b_pi=b, ordering=o, **obs,
                                 chi2_global=pmns_delta_chi2(o, **obs, relative_to="global")))
    rows.sort(key=lambda r: r["chi2_global"])
    return dict(n_configs=len(rows), best=rows[:6])


if __name__ == "__main__":
    OUT["C1_double_ratio"] = c1()
    r = OUT["C1_double_ratio"]
    print({k: r[k] for k in r if k not in ("surviving", "triples", "named")})
    print(json.dumps(r["named"], indent=1)); print(json.dumps(r["surviving"][:12], indent=0))
    OUT["C3_qlc_fixed_phase"] = c3()
    print(json.dumps(OUT["C3_qlc_fixed_phase"]["best"][:4], indent=0))
    OUT["C2_universal_4zero"] = c2()
    print(json.dumps({k: (v if k != "lepton_NO" else v) for k, v in OUT["C2_universal_4zero"].items()}, indent=0)[:3000])
    (HERE / "checks.json").write_text(json.dumps(OUT, indent=1))
