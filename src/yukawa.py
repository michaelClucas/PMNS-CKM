"""Shared Yukawa-matrices -> observables engine (exp 030).

    observables(Y_u, Y_d, Y_e, m_nu=None, *, Y_nu=None, M_R=None, ordering="auto",
                nu_mass_scale_eV=None, details=False)            -> dict (src.evaluate keys)
    left_rotation(Y)          -> (U, y)     U^dag Y Y^dag U = diag(y^2), y ascending
    takagi(m)                 -> (U, m_i)   U^T m U = diag(m_i) real >= 0, ascending
    seesaw_type1(Y_nu, M_R)   -> m_nu = -Y_nu^* M_R^-1 Y_nu^dag
    majorana_phases(U, delta) -> (alpha21, alpha31) in radians, [0, 2 pi)
    observables_batch(Y_u, Y_d, Y_e, m_nu, ordering="auto")  -> dict of arrays (exp 033; same
        convention and keys as observables, batched over a leading axis; rows with non-finite
        input give nan; used by src.theoryspace)

Convention (one, used everywhere in this module)
------------------------------------------------
Dirac Yukawas are LR: -L = conj(f_L)_i (Y_f)_ij f_R,j H + h.c. Rows are the left-handed (doublet)
index, columns the right-handed one. The mass matrix is M_f = Y_f v; v cancels in every output.
The left rotation U_f diagonalises Y_f Y_f^dag (U_f^dag Y_f Y_f^dag U_f = diag(y^2), ascending);
U_f is unique up to column phases, which no output depends on.

    V_CKM  = U_u^dag U_d          U_PMNS = U_e^dag U_nu

Light Majorana neutrinos: two-component mass term -L = 1/2 nu^T m_nu nu + h.c. (equivalently
1/2 conj(nu_L^c) m_nu nu_L), so m_nu is complex symmetric and

    m_nu = U_nu^* diag(m_1, m_2, m_3) U_nu^dag,   m_i >= 0     (PDG convention)

i.e. U_nu is the Takagi factor (U_nu^T m_nu U_nu = diag(m_i)): its columns are eigenvectors of
m_nu^dag m_nu, with column phases fixed so the diagonal is real and non-negative.
Type-I seesaw: Y_nu is LR like the others (conj(L) Y_nu nu_R), M_R is the two-component
Majorana matrix of the conjugate fields N = nu_R^c (-L = 1/2 N^T M_R N), and integrating N out
gives m_nu = -Y_nu^* M_R^-1 Y_nu^dag (times v^2; the overall factor and sign are unphysical).

Angles, delta and Majorana phases: src.mixing (PDG standard parametrisation, rephasing-invariant
extraction). delta is the PDG delta in [0, 2 pi) (dcp_deg in [0, 360)), so U_e3 = s13 e^{-i delta}
up to rephasing. Majorana phases follow PDG, U = U_std diag(1, e^{i alpha21/2}, e^{i alpha31/2})
up to charged-lepton rephasing, from the invariants of the Takagi-fixed U_PMNS:
    alpha21 = 2 arg(U_e2 U_e1^*),   alpha31 = 2 [arg(U_e3 U_e1^*) + delta]   (mod 2 pi)
They are well defined when m_i > 0 are non-degenerate; a vanishing m_i makes its phase
unphysical (returned, but meaningless).

Mass labels and ordering. m_nu singular values ascending a < b < c. ordering="auto": NO when
b^2 - a^2 < c^2 - b^2 (m1, m2, m3 = a, b, c), else IO (m3, m1, m2 = a, b, c; columns of U_nu
reordered (1, 2, 0) so that column 1, 2 carry the two quasi-degenerate states). ordering "NO" /
"IO" forces the labelling. dm_ratio = dm21 / |dm3l| with dm3l = dm31 (NO) or dm32 (IO), the
src.evaluate definition; dm21sq_eV2 and dm3l_sq_eV2 are added when nu_mass_scale_eV (eV per
unit of m_nu) is given.

Output keys (src.evaluate names, plain floats): ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad,
s12sq, s13sq, s23sq, dcp_deg, mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau,
R_clebsch, dm_ratio [, dm21sq_eV2, dm3l_sq_eV2], and alpha21_deg, alpha31_deg (not scored by
src.evaluate; unused keys are ignored there). The mass ratios are ratios of the singular values
of the matrices given, at the scale where they are given; for a high-scale model pass
scale=... to src.evaluate. details=True also returns a dict with V, U_pmns, U_e, U_nu, the
singular values, the neutrino masses and the ordering.

Mapping of earlier experiments (tests/test_yukawa.py reproduces each from its stored inputs):
  exp 014 (SO(10) FN/MN, so10fit.py)  Weyl RL matrices f^c M f, left rotation from M^dag M,
      m_nu = M_D^T M_R^-1 M_D, U_nu from M_nu^dag M_nu.  ->  Y_f = M_f^dag, m_nu = M_nu as given
      (or Y_nu = M_D^dag, M_R = F for the seesaw path; the sign is unphysical).
  exp 022 (SU(5)+CSD3, model.py)  LR Yukawas (Y Y^dag), U_nu = eigenvectors of m m^dag.
      ->  Y_f = Y_f, m_nu = conj(m) (exp 022's U_nu is the Takagi factor of conj(m)).
  exp 023 (S4xSO(10), model.py)  LR Yukawas, PMNS = Takagi factor of U_e^T conj(m) U_e.
      ->  Y_f = Y_f, m_nu = conj(m).
  Exps 022 and 023 picked their conjugation by reproducing published CP-phase signs, so in this
  convention their papers' m_nu is the complex conjugate of the engine's.
"""
from __future__ import annotations

import numpy as np

from .mixing import extract_angles

TWO_PI = 2.0 * np.pi

__all__ = ["observables", "observables_batch", "left_rotation", "takagi", "seesaw_type1", "majorana_phases",
           "OUTPUT_KEYS"]

OUTPUT_KEYS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg",
               "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau", "R_clebsch",
               "dm_ratio", "alpha21_deg", "alpha31_deg")


def _mat(Y, name):
    Y = np.asarray(Y, dtype=complex)
    if Y.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {Y.shape}")
    if not np.all(np.isfinite(Y)):
        raise ValueError(f"{name} has non-finite entries")
    return Y


def left_rotation(Y):
    """(U, y): U^dag Y Y^dag U = diag(y^2), y ascending (LR convention: left = row index)."""
    Y = _mat(Y, "Y")
    w, U = np.linalg.eigh(Y @ Y.conj().T)
    return U, np.sqrt(np.clip(w, 0.0, None))


def takagi(m, sym_tol=1e-8):
    """(U, m_i) with U^T m U = diag(m_i), m_i >= 0 ascending, for complex symmetric m.

    Columns of U: eigenvectors of m^dag m; each column's phase is then fixed so that
    (U^T m U)_ii is real and non-negative. Exact for non-degenerate m_i; for degenerate
    m_i the off-diagonal block is not rotated away (not needed for any output here: the
    mixing angles of a degenerate pair are unphysical)."""
    m = _mat(m, "m_nu")
    scale = np.max(np.abs(m))
    if scale > 0 and np.max(np.abs(m - m.T)) > sym_tol * scale:
        raise ValueError("m_nu must be symmetric")
    w, U = np.linalg.eigh(m.conj().T @ m)
    d = np.diag(U.T @ m @ U)
    U = U * np.exp(-0.5j * np.angle(d))[None, :]
    return U, np.sqrt(np.clip(w, 0.0, None))


def seesaw_type1(Y_nu, M_R):
    """Light-neutrino matrix m_nu = -Y_nu^* M_R^-1 Y_nu^dag (module docstring; units of v^2 / M_R)."""
    Y_nu, M_R = _mat(Y_nu, "Y_nu"), _mat(M_R, "M_R")
    m = -Y_nu.conj() @ np.linalg.solve(M_R, Y_nu.conj().T)
    return 0.5 * (m + m.T)


def majorana_phases(U, delta):
    """PDG (alpha21, alpha31) in [0, 2 pi) of a Takagi-fixed U_PMNS with Dirac phase delta (rad)."""
    U = np.asarray(U)
    a21 = 2 * np.angle(U[0, 1] * np.conj(U[0, 0]))
    a31 = 2 * (np.angle(U[0, 2] * np.conj(U[0, 0])) + delta)
    return float(np.mod(a21, TWO_PI)), float(np.mod(a31, TWO_PI))


def _label(m, ordering):
    """Permutation of the ascending singular values into (m1, m2, m3) and the ordering name."""
    a2, b2, c2 = m[0] ** 2, m[1] ** 2, m[2] ** 2
    if ordering == "auto":
        ordering = "NO" if (b2 - a2) < (c2 - b2) else "IO"
    if ordering == "NO":
        return np.array([0, 1, 2]), "NO"
    if ordering == "IO":
        return np.array([1, 2, 0]), "IO"
    raise ValueError("ordering must be 'auto', 'NO' or 'IO'")


def observables(Y_u, Y_d, Y_e, m_nu=None, *, Y_nu=None, M_R=None, ordering="auto",
                nu_mass_scale_eV=None, details=False):
    """CKM, PMNS, mass ratios and dm ratio from Yukawa matrices (module docstring for conventions).

    Give either m_nu (light Majorana mass matrix, any units) or Y_nu and M_R (type-I seesaw).
    Returns a dict of floats keyed as src.evaluate expects (plus alpha21_deg, alpha31_deg);
    with details=True returns (dict, details).
    """
    if (m_nu is None) == (Y_nu is None and M_R is None):
        raise ValueError("give either m_nu or (Y_nu, M_R)")
    if m_nu is None:
        if Y_nu is None or M_R is None:
            raise ValueError("the seesaw path needs both Y_nu and M_R")
        m_nu = seesaw_type1(Y_nu, M_R)
    Uu, yu = left_rotation(Y_u)
    Ud, yd = left_rotation(Y_d)
    Ue, ye = left_rotation(Y_e)
    Un, mn = takagi(m_nu)
    perm, o = _label(mn, ordering)
    Un, mn = Un[:, perm], mn[perm]
    V = Uu.conj().T @ Ud
    U = Ue.conj().T @ Un
    q12, q13, q23, qd = (float(x) for x in extract_angles(V))
    l12, l13, l23, ld = (float(x) for x in extract_angles(U))
    a21, a31 = majorana_phases(U, ld)
    dm21 = mn[1] ** 2 - mn[0] ** 2
    dm3l = mn[2] ** 2 - (mn[0] ** 2 if o == "NO" else mn[1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "ckm_s12": np.sin(q12), "ckm_s13": np.sin(q13), "ckm_s23": np.sin(q23), "ckm_delta_rad": qd,
            "s12sq": np.sin(l12) ** 2, "s13sq": np.sin(l13) ** 2, "s23sq": np.sin(l23) ** 2,
            "dcp_deg": np.degrees(ld),
            "mr_u_c": yu[0] / yu[1], "mr_c_t": yu[1] / yu[2], "mr_d_s": yd[0] / yd[1], "mr_s_b": yd[1] / yd[2],
            "mr_e_mu": ye[0] / ye[1], "mr_mu_tau": ye[1] / ye[2], "mr_b_tau": yd[2] / ye[2],
            "dm_ratio": dm21 / abs(dm3l),
            "alpha21_deg": np.degrees(a21), "alpha31_deg": np.degrees(a31),
        }
        out["R_clebsch"] = out["mr_e_mu"] / out["mr_d_s"]
    if nu_mass_scale_eV is not None:
        f = float(nu_mass_scale_eV) ** 2
        out["dm21sq_eV2"], out["dm3l_sq_eV2"] = dm21 * f, dm3l * f
    out = {k: float(v) for k, v in out.items()}
    if not details:
        return out
    return out, {"V_ckm": V, "U_pmns": U, "U_u": Uu, "U_d": Ud, "U_e": Ue, "U_nu": Un, "y_u": yu, "y_d": yd,
                 "y_e": ye, "m_nu": mn, "ordering": o, "m_nu_matrix": np.asarray(m_nu, complex)}


# ---------------------------------------------------------------- batched path (exp 033)

def _batch(Y, name):
    Y = np.asarray(Y, dtype=complex)
    if Y.ndim != 3 or Y.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (B, 3, 3), got {Y.shape}")
    return Y


def _dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def observables_batch(Y_u, Y_d, Y_e, m_nu, *, ordering="auto", nu_mass_scale_eV=None, sym_tol=1e-8):
    """observables() over a batch: Y_u, Y_d, Y_e, m_nu of shape (B, 3, 3) -> dict of (B,) arrays.

    Same convention, keys, ordering rule and Takagi phase fixing as observables(); agrees with it
    row by row to rounding (tests/test_theoryspace.py). Rows where any input entry is non-finite,
    or where m_nu is not symmetric (relative tolerance sym_tol), give nan in every key. Also returns
    "io" (1.0 for IO, 0.0 for NO, nan for bad rows). No details; for the seesaw path pass
    m_nu = seesaw_type1 applied per row (or compute -Y_nu^* M_R^-1 Y_nu^dag in batch yourself).
    """
    Yu, Yd, Ye, m = (_batch(Y, n) for Y, n in ((Y_u, "Y_u"), (Y_d, "Y_d"), (Y_e, "Y_e"), (m_nu, "m_nu")))
    B = len(Yu)
    if not (len(Yd) == len(Ye) == len(m) == B):
        raise ValueError("all inputs need the same batch size")
    good = np.all(np.isfinite(np.stack([Yu, Yd, Ye, m])), axis=(0, 2, 3))
    scale = np.max(np.abs(np.where(good[:, None, None], m, 0)), axis=(1, 2))
    asym = np.max(np.abs(np.where(good[:, None, None], m - np.swapaxes(m, 1, 2), 0)), axis=(1, 2))
    good &= ~((scale > 0) & (asym > sym_tol * scale))
    keys = OUTPUT_KEYS + ("io",) + (("dm21sq_eV2", "dm3l_sq_eV2") if nu_mass_scale_eV is not None else ())
    out = {k: np.full(B, np.nan) for k in keys}
    if not good.any():
        return out
    Yu, Yd, Ye, m = Yu[good], Yd[good], Ye[good], m[good]

    def lrot(Y):
        w, U = np.linalg.eigh(Y @ _dag(Y))
        return U, np.sqrt(np.clip(w, 0.0, None))

    Uu, yu = lrot(Yu)
    Ud, yd = lrot(Yd)
    Ue, ye = lrot(Ye)
    w, Un = np.linalg.eigh(_dag(m) @ m)
    d = np.einsum("bji,bjk,bki->bi", Un, m, Un)  # diag(U^T m U)
    Un = Un * np.exp(-0.5j * np.angle(d))[:, None, :]
    mn = np.sqrt(np.clip(w, 0.0, None))
    a2, b2, c2 = mn[:, 0] ** 2, mn[:, 1] ** 2, mn[:, 2] ** 2
    if ordering == "auto":
        io = ~((b2 - a2) < (c2 - b2))
    elif ordering in ("NO", "IO"):
        io = np.full(len(mn), ordering == "IO")
    else:
        raise ValueError("ordering must be 'auto', 'NO' or 'IO'")
    Un = np.where(io[:, None, None], Un[:, :, [1, 2, 0]], Un)
    mn = np.where(io[:, None], mn[:, [1, 2, 0]], mn)
    V = _dag(Uu) @ Ud
    U = _dag(Ue) @ Un
    q12, q13, q23, qd = extract_angles(V)
    l12, l13, l23, ld = extract_angles(U)
    a21 = np.mod(2 * np.angle(U[:, 0, 1] * np.conj(U[:, 0, 0])), TWO_PI)
    a31 = np.mod(2 * (np.angle(U[:, 0, 2] * np.conj(U[:, 0, 0])) + ld), TWO_PI)
    dm21 = mn[:, 1] ** 2 - mn[:, 0] ** 2
    dm3l = mn[:, 2] ** 2 - np.where(io, mn[:, 1] ** 2, mn[:, 0] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = {
            "ckm_s12": np.sin(q12), "ckm_s13": np.sin(q13), "ckm_s23": np.sin(q23), "ckm_delta_rad": qd,
            "s12sq": np.sin(l12) ** 2, "s13sq": np.sin(l13) ** 2, "s23sq": np.sin(l23) ** 2,
            "dcp_deg": np.degrees(ld),
            "mr_u_c": yu[:, 0] / yu[:, 1], "mr_c_t": yu[:, 1] / yu[:, 2], "mr_d_s": yd[:, 0] / yd[:, 1],
            "mr_s_b": yd[:, 1] / yd[:, 2], "mr_e_mu": ye[:, 0] / ye[:, 1], "mr_mu_tau": ye[:, 1] / ye[:, 2],
            "mr_b_tau": yd[:, 2] / ye[:, 2], "dm_ratio": dm21 / np.abs(dm3l),
            "alpha21_deg": np.degrees(a21), "alpha31_deg": np.degrees(a31), "io": io.astype(float),
        }
        g["R_clebsch"] = g["mr_e_mu"] / g["mr_d_s"]
    if nu_mass_scale_eV is not None:
        f = float(nu_mass_scale_eV) ** 2
        g["dm21sq_eV2"], g["dm3l_sq_eV2"] = dm21 * f, dm3l * f
    for k in keys:
        out[k][good] = g[k]
    return out
