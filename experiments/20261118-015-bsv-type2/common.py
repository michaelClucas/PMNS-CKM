"""BSV type-II SO(10) model (hep-ph/0210207) at M_GUT: data-to-GUT map, model matrices, chi^2.

Conventions (same as exp 011 gen.py, verified there against src.rg): Weyl mass matrices f^c M f,
symmetric for 10_H and 126bar_H. src.rg's Yukawa is Y = M^dagger, left rotations come from
eigh(M^dagger M), kappa = M_nu (holomorphic), U_PMNS = U_e^dagger U_nu, V_CKM = U_u^dagger U_d.

SO(10) 10_H + 126bar_H, BSV Eqs. (9)-(13), (17), (18) (equivalently 1306.4468 Eq. (4) with x = s, y = 1/r):
    M_u = diag(u)                       (basis choice: SO(10) flavour rotation makes M_u real diagonal)
    M_d = (1 + eta) U^* diag(d_i e^{i phi_i}) U^dagger,   U = diag(1, e^{i a2}, e^{i a3}) V_std(CKM at M_GUT)
    M_e = [(3 + x) M_d - 4 y M_u] / (x - 1)
    M_nu = c (M_d - M_e)                (type-II dominance, BSV Eq. (13), m_nu ∝ Y_126)
eta is 0 except in variant R0eta (a universal down-type GUT/threshold factor).
All matrices are in the normalisation of the src.rg state at M_GUT (MSSM normalisation above M_SUSY);
x and y absorb the vev ratios, so the normalisation does not matter.
"""
from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import rg  # noqa: E402
from src.mixing import extract_angles, standard_matrix  # noqa: E402

M_GUT = 2e16
TANB = (5, 10, 20, 30, 40, 50, 55)
QUARK_KEYS = ("m_u", "m_d", "m_s", "m_c", "m_b", "m_t_direct")
CKM_KEYS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")
NUIS_KEYS = QUARK_KEYS + CKM_KEYS
PMNS_KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
OFFICIAL_USE = NUIS_KEYS + PMNS_KEYS          # scored by src.evaluate (N_official = 14)
LEPTON_INPUTS = ("m_e", "m_mu", "m_tau")      # consumed; enter as GUT relation with a theory floor
SIG_TH = 0.02                                  # theory floor on GUT charged-lepton eigenvalues (as exp 011)


def scenario_name(model, tanb=None, msusy=3000.0):
    return "SM" if model == "SM" else f"MSSM_tb{tanb}_ms{msusy / 1000:g}TeV"


def _svals(Y):
    return np.sqrt(np.clip(np.linalg.eigvalsh(Y @ Y.conj().T), 0, None))  # ascending


def run_up(model, tanb, msusy, yukawas=None, ordering="NO"):
    s = rg.build_mz_state(ordering=ordering, yukawas=yukawas)
    if model == "SM":
        return rg.run_sm(s["mu"], M_GUT, s)
    return rg.run_mssm(s["mu"], M_GUT, s, tanb, msusy)


@functools.lru_cache(maxsize=None)
def gut_reference(model, tanb=None, msusy=3000.0):
    """Data run to M_GUT plus a log-linear map from low-scale nuisances to GUT inputs."""
    y0 = rg.derived_yukawas_mz()
    s = run_up(model, tanb, msusy)
    yu, yd, ye = _svals(s["Yu"]), _svals(s["Yd"]), _svals(s["Ye"])
    obs = rg.observables(s, tanb)
    ck = obs["ckm"]
    ckd = D.load_ckm()
    low = (np.arcsin(ckd["s12"]["value"]), np.arcsin(ckd["s13"]["value"]), np.arcsin(ckd["s23"]["value"]),
           ckd["delta_rad"]["value"])
    # Jacobian d ln y_GUT(t, b, tau) / d ln y_MZ(t, b) by finite differences (1%)
    J = np.zeros((3, 2))
    for j, key in enumerate(("y_t", "y_b")):
        yy = dict(y0)
        yy[key] *= 1.01
        s2 = run_up(model, tanb, msusy, yukawas=yy)
        new = np.array([_svals(s2["Yu"])[2], _svals(s2["Yd"])[2], _svals(s2["Ye"])[2]])
        J[:, j] = (np.log(new) - np.log([yu[2], yd[2], ye[2]])) / np.log(1.01)
    return {"state": s, "yu": yu, "yd": yd, "ye": ye,
            "ckm_gut": (ck["theta12"], ck["theta13"], ck["theta23"], ck["delta"]),
            "ckm_low": low, "J": J, "model": model, "tanb": tanb, "msusy": msusy,
            "mb_over_mtau": yd[2] / ye[2]}


# ------------------------------------------------------------------ data records

@functools.lru_cache(maxsize=None)
def nuisance_records():
    m, c = D.load_masses(), D.load_ckm()
    rec = {k: m[k] for k in QUARK_KEYS}
    rec.update({"ckm_" + k: c[k] for k in ("s12", "s13", "s23", "delta_rad")})
    return {k: (r["value"], r["err_plus"], r["err_minus"]) for k, r in rec.items()}


def nuis_from_z(z):
    """Pull-space nuisances z (10) -> low-scale values (asymmetric errors, side by sign)."""
    rec = nuisance_records()
    out = {}
    for zi, k in zip(z, NUIS_KEYS):
        v, ep, em = rec[k]
        out[k] = v + zi * (ep if zi > 0 else em)
    return out


@functools.lru_cache(maxsize=None)
def ratio_data(ordering):
    sp = D.load_masses()["neutrino_splittings"][D.DEFAULT_ANALYSIS][ordering]
    a = sp["dm21sq"]
    b = sp["dm31sq"] if ordering == "NO" else sp["dm32sq"]
    r = a["value"] / abs(b["value"])
    e = r * np.hypot(0.5 * (a["err_plus"] + a["err_minus"]) / a["value"],
                     0.5 * (b["err_plus"] + b["err_minus"]) / abs(b["value"]))
    return r, e


# ------------------------------------------------------------------ model

def gut_inputs(ref, nu):
    """Low-scale nuisance values -> GUT eigenvalues (u, d, e targets) and CKM angles."""
    rec = nuisance_records()
    f = {k: nu[k] / rec[k][0] for k in QUARK_KEYS}
    dl = np.log([f["m_t_direct"], f["m_b"]])
    u = ref["yu"] * np.array([f["m_u"], f["m_c"], 1.0])
    d = ref["yd"] * np.array([f["m_d"], f["m_s"], 1.0])
    e = ref["ye"].copy()
    lt, lb, ltau = ref["J"] @ dl
    u[2] *= np.exp(lt); d[2] *= np.exp(lb); e[2] *= np.exp(ltau)
    g, l = ref["ckm_gut"], ref["ckm_low"]
    th = (np.arcsin(nu["ckm_s12"]) * g[0] / l[0], np.arcsin(nu["ckm_s13"]) * g[1] / l[1],
          np.arcsin(nu["ckm_s23"]) * g[2] / l[2], nu["ckm_delta_rad"] + (g[3] - l[3]))
    return u, d, e, th


def matrices(u, d, th, x, y, phases=(0, 0, 0, 0, 0), eta=0.0):
    a2, a3, p1, p2, p3 = phases
    V = standard_matrix(*th)
    U = np.diag([1, np.exp(1j * a2), np.exp(1j * a3)]) @ V
    Mu = np.diag(u).astype(complex)
    Md = (1 + eta) * U.conj() @ np.diag(d * np.exp(1j * np.array([p1, p2, p3]))) @ U.conj().T
    Me = ((3 + x) * Md - 4 * y * Mu) / (x - 1)
    Mnu = Md - Me
    return Mu, Md, Me, Mnu


def left_basis(M):
    w, U = np.linalg.eigh(M.conj().T @ M)
    return np.sqrt(np.clip(w, 0, None)), U


def lepton_obs(Me, Mnu, ordering):
    """Charged-lepton eigenvalues, PMNS (s12sq, s13sq, s23sq, dcp_deg) and dm21/|dm3l| for a fixed ordering label."""
    me, Ue = left_basis(Me)
    mn, Un = left_basis(Mnu)
    a2, b2, c2 = mn ** 2
    if ordering == "IO":
        Un = Un[:, [1, 2, 0]]
        r = (c2 - b2) / (c2 - a2)
    else:
        r = (b2 - a2) / (c2 - a2)
    t12, t13, t23, dl = extract_angles(Ue.conj().T @ Un)
    pm = {"s12sq": np.sin(t12) ** 2, "s13sq": np.sin(t13) ** 2, "s23sq": np.sin(t23) ** 2,
          "dcp_deg": np.degrees(dl) % 360.0}
    return me, pm, r, mn


# ------------------------------------------------------------------ variants

VARIANTS = {
    # name: (continuous model params, discrete bits)
    "R0": (["x", "yhat"], 2),            # BSV's own choice: no phases beyond the CKM's, x, y real; 4 sign patterns
    "R0eta": (["x", "yhat", "eta"], 2),  # R0 + free universal down-type factor (threshold / b-tau mismatch)
    "C": (["xr", "xi", "yhat", "a2", "a3", "p1", "p2", "p3"], 0),  # general minimal 10+126 type II (x complex, y real WLOG)
}
SIGNS = [(0.0, 0.0), (np.pi, 0.0), (0.0, np.pi), (np.pi, np.pi)]  # (phi1, phi2), phi3 = 0


def unpack(variant, p, sign=(0.0, 0.0)):
    if variant == "C":
        xr, xi, yh, a2, a3, p1, p2, p3 = p
        return complex(xr, xi), yh, (a2, a3, p1, p2, p3), 0.0
    eta = p[2] if variant == "R0eta" else 0.0
    return p[0], p[1], (0.0, 0.0, sign[0], sign[1], 0.0), eta


def predict(ref, variant, p, z, ordering, sign=(0.0, 0.0), shift=None):
    """Full prediction at M_GUT (+ optional additive GUT->M_Z shift on PMNS and ratio)."""
    nu = nuis_from_z(z)
    u, d, e_t, th = gut_inputs(ref, nu)
    x, yh, ph, eta = unpack(variant, p, sign)
    y = yh * d[2] / u[2]
    Mu, Md, Me, Mnu = matrices(u, d, th, x, y, ph, eta)
    me, pm, r, mn = lepton_obs(Me, Mnu, ordering)
    if shift is not None:
        pm = {k: pm[k] + shift[k] for k in PMNS_KEYS}
        r = r + shift["ratio"]
    return {"nu": nu, "me": me, "e_target": e_t, "pmns": pm, "ratio": r, "mats": (Mu, Md, Me, Mnu), "mn": mn,
            "mb_over_mtau_model": d[2] * (1 + eta) / me[2] if np.isscalar(eta) else np.nan}


# ------------------------------------------------------------------ chi^2 pieces

def _pmns_residuals(pm, ordering):
    """Signed sqrt of NuFIT Delta chi^2: 1D for s12, s13; 2D (s23, delta) as one residual."""
    best = {k: D.load_pmns(ordering)[k]["value"] for k in PMNS_KEYS}
    res = []
    for k in ("s12sq", "s13sq"):
        c = D.pmns_delta_chi2(ordering, **{k: pm[k]})
        if not np.isfinite(c):
            rec = D.load_pmns(ordering)[k]
            c = 100 + ((pm[k] - rec["value"]) / rec["err_plus"]) ** 2
        res.append(np.sign(pm[k] - best[k]) * np.sqrt(c))
    c = D.pmns_delta_chi2(ordering, s23sq=pm["s23sq"], dcp_deg=pm["dcp_deg"])
    if not np.isfinite(c):
        c = 100 + ((pm["s23sq"] - best["s23sq"]) / 0.02) ** 2
    res.append(np.sqrt(c))
    return res


def residuals(ref, variant, p, z, ordering, sign=(0.0, 0.0), shift=None, sig_th=SIG_TH):
    try:
        pr = predict(ref, variant, p, z, ordering, sign, shift)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return np.full(17, 1e3)
    lep = (np.log(np.clip(pr["me"], 1e-30, None)) - np.log(pr["e_target"])) / sig_th
    r0, re = ratio_data(ordering)
    res = np.concatenate([np.asarray(z, float), lep, _pmns_residuals(pr["pmns"], ordering), [(pr["ratio"] - r0) / re]])
    return np.where(np.isfinite(res), res, 1e3)
