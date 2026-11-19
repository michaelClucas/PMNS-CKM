"""Mass matrices from modular forms with one common modulus tau, the assignment catalog, and
batched observables.

Conventions (as src.mixing): superpotential W = f^c_i Y^f_ij(tau) Q_j H (rows = conjugate RH
fields f^c, columns = left-handed doublet components). Mass matrix M_f = Y_f (overall scale
dropped). Left rotation U_f from M_f^dag M_f = U_f diag(m^2) U_f^dag, masses ascending.
CKM = U_u^dag U_d, PMNS = U_e^dag U_nu, with U_nu from M_nu^dag M_nu (Weinberg operator
(1/2) L M_nu L). Majorana phases are dropped (not observable here).

Charged sector f in {u, d, e}: Q (or L) is a triplet r_Q, each f^c_i is a singlet r_i with
Yukawa form weight w_i (= k_Q + k_{f^c_i}, any value reachable by choosing k_{f^c_i}). Row i =
alpha_i * sum_a g_ia R^(a)(tau), with R^(a) the n_c real invariant structures of
modforms.yukawa_row_structures. A "row type" is (w, r_i); a sector assignment is 3 distinct
row types. Couplings real (generalized CP, Novichkov et al. 1905.11970): alpha_i > 0 (the sign
is a field rephasing), the first g of each row is 1, extra g's real.
Neutrinos: Weinberg operator of weight 2 k_L: M_nu = sum_a g_a W^(a)(tau), g_0 = 1, others real.

Catalog (CATALOG RULE, fixed before any fit):
  row types: w in {2,4,6,8}, singlet r, n_c in {1, 2} couplings;
  sector: 3 distinct row types with at most one n_c = 2 row (<= 1 extra real coupling);
  neutrinos: Weinberg weight in {2, 4, 6} (1, 3, 3 couplings: 0, 2, 2 real ratios);
  Q and L independently any triplet of the group; one tau for everything.
"""
from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass

import numpy as np

import modforms as mf

NU_WEIGHTS = (2, 4, 6)
ROW_WEIGHTS = (2, 4, 6, 8)
MAX_EXTRA_PER_SECTOR = 1


# ---------------------------------------------------------------- catalog


@functools.lru_cache(maxsize=None)
def row_types(N, qrep):
    out = []
    for w in ROW_WEIGHTS:
        for r in mf.SINGLETS[N]:
            n = mf.yukawa_row_structures(N, w, r, qrep).shape[0]
            if 1 <= n <= 2:
                out.append((w, r, n))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def sector_sets(N, qrep, max_extra=MAX_EXTRA_PER_SECTOR):
    out = []
    for combo in itertools.combinations(row_types(N, qrep), 3):
        extra = sum(n - 1 for (_, _, n) in combo)
        if extra <= max_extra:
            out.append(combo)
    return tuple(out)


def n_extra(rowset):
    return sum(n - 1 for (_, _, n) in rowset)


@functools.lru_cache(maxsize=None)
def nu_options(N, lrep):
    out = []
    for w in NU_WEIGHTS:
        n = mf.weinberg_structures(N, w, lrep).shape[0]
        if n >= 1:
            out.append((w, n))
    return tuple(out)


@dataclass(frozen=True)
class QuarkAssign:
    N: int
    qrep: str
    u: tuple
    d: tuple

    @property
    def k_cont(self):  # 2 log-alpha ratios per sector + extras (tau counted separately)
        return 4 + n_extra(self.u) + n_extra(self.d)

    def label(self):
        f = lambda s: "+".join(f"{w}{r}{'*' if n == 2 else ''}" for w, r, n in s)
        return f"{mf.GROUP[self.N]} Q~{self.qrep} u[{f(self.u)}] d[{f(self.d)}]"


@dataclass(frozen=True)
class LeptonAssign:
    N: int
    lrep: str
    e: tuple
    nu: tuple  # (weight, n_couplings)

    @property
    def k_cont(self):
        return 2 + n_extra(self.e) + (self.nu[1] - 1)

    def label(self):
        f = lambda s: "+".join(f"{w}{r}{'*' if n == 2 else ''}" for w, r, n in s)
        return f"{mf.GROUP[self.N]} L~{self.lrep} e[{f(self.e)}] nu[W{self.nu[0]}]"


@functools.lru_cache(maxsize=None)
def quark_catalog(N):
    return tuple(QuarkAssign(N, q, u, d) for q in mf.TRIPLETS[N]
                 for u in sector_sets(N, q) for d in sector_sets(N, q))


@functools.lru_cache(maxsize=None)
def lepton_catalog(N):
    return tuple(LeptonAssign(N, l, e, nu) for l in mf.TRIPLETS[N]
                 for e in sector_sets(N, l) for nu in nu_options(N, l))


def catalog_size(N):
    return len(quark_catalog(N)) * len(lepton_catalog(N))


# ---------------------------------------------------------------- matrices


def charged_matrix(N, qrep, rowset, tau, logalpha, extra):
    """tau (B,), logalpha (B, 2) = ln(alpha_2/alpha_1), ln(alpha_3/alpha_1), extra (B, n_extra).
    Returns M (B, 3, 3)."""
    B = tau.shape[0]
    rows = []
    ie = 0
    la = np.concatenate([np.zeros((B, 1)), logalpha], axis=1)
    for i, (w, r, n) in enumerate(rowset):
        S = mf.yukawa_row_structures(N, w, r, qrep)  # (n, 3, M)
        v = mf.eval_series(S, tau, N)  # (B, n, 3)
        row = v[:, 0]
        if n == 2:
            row = row + extra[:, ie: ie + 1] * v[:, 1]
            ie += 1
        rows.append(np.exp(la[:, i: i + 1]) * row)
    return np.stack(rows, axis=1)


def nu_matrix(N, lrep, nu, tau, g):
    w, n = nu
    W = mf.weinberg_structures(N, w, lrep)  # (n, 3, 3, M)
    v = mf.eval_series(W, tau, N)  # (B, n, 3, 3)
    M = v[:, 0]
    for a in range(1, n):
        M = M + g[:, a - 1, None, None] * v[:, a]
    return M


def left_diag(M):
    """U (B,3,3) and masses (B,3) ascending from M^dag M."""
    H = np.conj(np.swapaxes(M, 1, 2)) @ M
    ev, U = np.linalg.eigh(H)
    return U, np.sqrt(np.clip(ev, 0, None))


def mixing_obs(U):
    """(s12, s13, s23, delta) arrays via src.mixing.extract_angles."""
    from src.mixing import extract_angles
    t12, t13, t23, d = extract_angles(U)
    return np.sin(t12), np.sin(t13), np.sin(t23), d


def quark_obs(a: QuarkAssign, tau, p):
    """p (B, k_cont): [la_u(2), la_d(2), extras_u, extras_d]. Returns dict of arrays."""
    nu_, nd_ = n_extra(a.u), n_extra(a.d)
    Mu = charged_matrix(a.N, a.qrep, a.u, tau, p[:, 0:2], p[:, 4:4 + nu_])
    Md = charged_matrix(a.N, a.qrep, a.d, tau, p[:, 2:4], p[:, 4 + nu_:4 + nu_ + nd_])
    Uu, mu = left_diag(Mu)
    Ud, md = left_diag(Md)
    s12, s13, s23, d = mixing_obs(np.conj(np.swapaxes(Uu, 1, 2)) @ Ud)
    with np.errstate(divide="ignore"):
        return {"lr_uc": np.log(mu[:, 0] / mu[:, 1]), "lr_ct": np.log(mu[:, 1] / mu[:, 2]),
                "lr_ds": np.log(md[:, 0] / md[:, 1]), "lr_sb": np.log(md[:, 1] / md[:, 2]),
                "ckm_s12": s12, "ckm_s13": s13, "ckm_s23": s23, "ckm_delta_rad": d}


def lepton_obs(a: LeptonAssign, tau, p, ordering):
    """p (B, k_cont): [la_e(2), extras_e, g_nu (n-1)]."""
    ne = n_extra(a.e)
    Me = charged_matrix(a.N, a.lrep, a.e, tau, p[:, 0:2], p[:, 2:2 + ne])
    Mn = nu_matrix(a.N, a.lrep, a.nu, tau, p[:, 2 + ne:])
    Ue, me = left_diag(Me)
    Un, mn = left_diag(Mn)
    if ordering == "IO":  # m3 lightest: columns (1,2,3) = eigen (1,2,0)
        Un = Un[:, :, [1, 2, 0]]
        mn = mn[:, [1, 2, 0]]
    s12, s13, s23, d = mixing_obs(np.conj(np.swapaxes(Ue, 1, 2)) @ Un)
    m2 = mn ** 2
    d21 = m2[:, 1] - m2[:, 0]
    d3l = (m2[:, 2] - m2[:, 0]) if ordering == "NO" else (m2[:, 2] - m2[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"lr_emu": np.log(me[:, 0] / me[:, 1]), "lr_mutau": np.log(me[:, 1] / me[:, 2]),
                "s12sq": s12 ** 2, "s13sq": s13 ** 2, "s23sq": s23 ** 2, "dcp_deg": np.degrees(d),
                "r_dm": d21 / np.abs(d3l)}


# ---------------------------------------------------------------- tau domain


def tau_from_unit(u):
    """Map u in [0,1]^2 to the fundamental domain |Re| <= 1/2, |tau| >= 1, Im <= YMAX."""
    x = u[:, 0] - 0.5
    ylo = np.sqrt(1 - x ** 2)
    y = ylo + u[:, 1] * (YMAX - ylo)
    return x + 1j * y


YMAX = 4.0
