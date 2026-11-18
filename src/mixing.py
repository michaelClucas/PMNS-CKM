"""Mixing-matrix math: PDG standard parametrization, angle extraction, Jarlskog, rephasing.

Conventions
-----------
Rows are flavour (e, mu, tau) / (u, c, t), columns are mass states (1, 2, 3) / (d, s, b).
PDG standard parametrization (radians):

    U = [[ c12 c13,                     s12 c13,                    s13 e^{-i d} ],
         [-s12 c23 - c12 s23 s13 e^{i d}, c12 c23 - s12 s23 s13 e^{i d}, s23 c13 ],
         [ s12 s23 - c12 c23 s13 e^{i d},-c12 s23 - s12 c23 s13 e^{i d}, c23 c13 ]]

Angles live in [0, pi/2], delta in [0, 2 pi). Majorana phases are ignored: any
diagonal phase matrix on the right is treated as unphysical.

Every function accepts a single matrix (3, 3) or a batch (..., 3, 3). Angle
arguments broadcast.

Jarlskog: J = Im(U_e1 U_mu2 U_e2^* U_mu1^*) = c12 s12 c23 s23 c13^2 s13 sin(delta).
For the CKM matrix this equals PDG's Im(V_us V_cb V_ub^* V_cs^*).
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

__all__ = [
    "standard_matrix",
    "extract_angles",
    "jarlskog",
    "rephase_to_standard",
    "unitarity_defect",
    "wolfenstein_to_standard",
    "wolfenstein_matrix",
    "standard_to_wolfenstein",
]


def standard_matrix(t12, t13, t23, delta):
    """PDG standard-parametrization matrix. Broadcasts over inputs; returns (..., 3, 3) complex."""
    t12, t13, t23, delta = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (t12, t13, t23, delta))
    )
    s12, c12 = np.sin(t12), np.cos(t12)
    s13, c13 = np.sin(t13), np.cos(t13)
    s23, c23 = np.sin(t23), np.cos(t23)
    ep = np.exp(1j * delta)
    U = np.empty(t12.shape + (3, 3), dtype=complex)
    U[..., 0, 0] = c12 * c13
    U[..., 0, 1] = s12 * c13
    U[..., 0, 2] = s13 / ep
    U[..., 1, 0] = -s12 * c23 - c12 * s23 * s13 * ep
    U[..., 1, 1] = c12 * c23 - s12 * s23 * s13 * ep
    U[..., 1, 2] = s23 * c13
    U[..., 2, 0] = s12 * s23 - c12 * c23 * s13 * ep
    U[..., 2, 1] = -c12 * s23 - s12 * c23 * s13 * ep
    U[..., 2, 2] = c23 * c13
    return U


def jarlskog(U):
    """J = Im(U_e1 U_mu2 U_e2^* U_mu1^*). Shape (...)."""
    U = np.asarray(U)
    return np.imag(U[..., 0, 0] * U[..., 1, 1] * np.conj(U[..., 0, 1]) * np.conj(U[..., 1, 0]))


def unitarity_defect(U):
    """max |U U^dagger - 1| per matrix. Shape (...)."""
    U = np.asarray(U)
    M = U @ np.conj(np.swapaxes(U, -1, -2)) - np.eye(3)
    return np.max(np.abs(M), axis=(-2, -1))


def extract_angles(U, tol=1e-12, delta_tol=1e-15):
    """Rephasing-invariant extraction of (t12, t13, t23, delta) from a 3x3 unitary (or batch).

    t13 = atan2(|U_e3|, sqrt(|U_e1|^2 + |U_e2|^2))
    t12 = atan2(|U_e2|, |U_e1|)
    t23 = atan2(|U_mu3|, |U_tau3|)
    delta = atan2(N_sin, N_cos), where, with D = s12 c12 s23 c23 s13,
        N_sin = J / c13^2                                   = D sin(delta)
        N_cos = (|U_mu1|^2 - s12^2 c23^2 - c12^2 s23^2 s13^2) / 2 = D cos(delta)
    Both are rephasing invariant, so delta is too.

    Edge cases (delta is unphysical whenever D = 0):
      * D < delta_tol: delta = 0 (delta is unphysical when D = 0 exactly; exact zeros
        from float trig, e.g. cos(pi/2) ~ 6e-17, land below delta_tol).
      * c13 < tol (|U_e3| = 1): t12 is undefined, set t12 = 0, and t23 is read from
        the 2x2 block: t23 = atan2(|U_mu1|, |U_mu2|), which is exact for t12 = 0.
    Numerical precision of delta degrades roughly as 1e-16 / D, so for D below ~1e-12
    delta is only meaningful to a few digits.

    Returns four arrays of shape (...).
    """
    U = np.asarray(U)
    a = np.abs(U)
    a2 = a * a
    row_e12 = np.sqrt(a2[..., 0, 0] + a2[..., 0, 1])
    t13 = np.arctan2(a[..., 0, 2], row_e12)
    c13 = np.cos(t13)
    generic = row_e12 > tol
    t12 = np.where(generic, np.arctan2(a[..., 0, 1], a[..., 0, 0]), 0.0)
    t23 = np.where(
        generic,
        np.arctan2(a[..., 1, 2], a[..., 2, 2]),
        np.arctan2(a[..., 1, 0], a[..., 1, 1]),
    )
    s12, c12 = np.sin(t12), np.cos(t12)
    s23, c23 = np.sin(t23), np.cos(t23)
    s13 = np.sin(t13)

    J = jarlskog(U)
    c13sq = np.where(generic, c13 * c13, 1.0)
    n_sin = np.where(generic, J / c13sq, 0.0)
    n_cos = 0.5 * (a2[..., 1, 0] - s12**2 * c23**2 - c12**2 * s23**2 * s13**2)
    D = s12 * c12 * s23 * c23 * s13
    delta = np.mod(np.arctan2(n_sin, n_cos), TWO_PI)
    delta = np.where(delta >= TWO_PI, delta - TWO_PI, delta)
    delta = np.where(D > delta_tol, delta, 0.0)
    return t12, t13, t23, delta


def rephase_to_standard(U, tol=1e-12):
    """Write U = P @ U_std @ Q with P, Q diagonal phase matrices.

    Returns (U_std, P, Q), where U_std = standard_matrix(*extract_angles(U)) and P, Q are
    (..., 3, 3) diagonal unitary matrices. Q carries what would be Majorana phases plus
    the column rephasing; P carries the row rephasing and the overall phase (so this
    works for U(3), not only SU(3)). P[0,0] = 1 by convention.

    The phases (a_i, b_j) solve a_i + b_j = arg(U_ij conj(U_std_ij)) on entries where
    both are non-zero (|U_ij||U_std_ij| > tol). Method: maximum-weight spanning tree of
    the bipartite row/column graph (weights |U_ij||U_std_ij|) fixes the phases exactly
    on tree edges, then one weighted least-squares step over all entries averages the
    rounding noise. Disconnected components (block-diagonal U) get their own gauge.
    """
    U = np.asarray(U, dtype=complex)
    batch_shape = U.shape[:-2]
    Uf = U.reshape(-1, 3, 3)
    n = Uf.shape[0]
    Ustd = standard_matrix(*extract_angles(Uf, tol=tol))
    Z = (Uf * np.conj(Ustd)).reshape(n, 9)
    W = np.abs(Z)
    phi = np.angle(Z)
    live = W > 1e-30  # keep tiny-but-nonzero entries; exact zeros (~1e-33) drop out
    ar = np.arange(n)
    edge_i = np.arange(9) // 3  # row node 0..2
    edge_j = 3 + np.arange(9) % 3  # column node 3..5

    # Kruskal on K_{3,3}, vectorized over the batch.
    order = np.argsort(-W, axis=1)
    comp = np.tile(np.arange(6), (n, 1))
    tree = np.zeros((n, 9), dtype=bool)
    for k in range(9):
        e = order[:, k]
        ci = comp[ar, edge_i[e]]
        cj = comp[ar, edge_j[e]]
        merge = live[ar, e] & (ci != cj)
        relabel = merge[:, None] & (comp == cj[:, None])
        comp = np.where(relabel, ci[:, None], comp)
        tree[ar[merge], e[merge]] = True

    # Roots: smallest node index in each component gets phase 0 (row e is always a root).
    known = np.ones((n, 6), dtype=bool)
    for v in range(1, 6):
        known[:, v] = ~np.any(comp[:, :v] == comp[:, v : v + 1], axis=1)
    x = np.zeros((n, 6))
    for _ in range(5):
        for e in range(9):
            i, j = edge_i[e], edge_j[e]
            fwd = tree[:, e] & known[:, i] & ~known[:, j]
            bwd = tree[:, e] & known[:, j] & ~known[:, i]
            x[:, j] = np.where(fwd, phi[:, e] - x[:, i], x[:, j])
            x[:, i] = np.where(bwd, phi[:, e] - x[:, j], x[:, i])
            known[:, j] |= fwd
            known[:, i] |= bwd

    # One weighted least-squares refinement on the wrapped residuals (all small now).
    inc = np.zeros((9, 6))
    inc[np.arange(9), edge_i] = 1.0
    inc[np.arange(9), edge_j] = 1.0
    sw = np.sqrt(np.where(live, W, 0.0))
    Apinv = np.linalg.pinv(sw[:, :, None] * inc, rcond=1e-14)
    R = np.angle(np.exp(1j * (phi - x @ inc.T)))
    x = x + np.einsum("nke,ne->nk", Apinv, sw * np.where(live, R, 0.0))
    g = x[:, :1].copy()  # gauge: a_e = 0
    x[:, :3] -= g
    x[:, 3:] += g
    eye = np.eye(3)
    P = (np.exp(1j * x[:, :3])[:, :, None] * eye).reshape(batch_shape + (3, 3))
    Q = (np.exp(1j * x[:, 3:])[:, :, None] * eye).reshape(batch_shape + (3, 3))
    return Ustd.reshape(batch_shape + (3, 3)), P, Q


# ---------------------------------------------------------------- Wolfenstein


def wolfenstein_to_standard(lam, A, rhobar, etabar):
    """Exact PDG map (lambda, A, rhobar, etabar) -> (t12, t13, t23, delta).

    s12 = lambda, s23 = A lambda^2,
    s13 e^{i delta} = A lambda^3 (rhobar + i etabar) sqrt(1 - A^2 lambda^4)
                      / ( sqrt(1 - lambda^2) [1 - A^2 lambda^4 (rhobar + i etabar)] ).
    Unitary to all orders in lambda.
    """
    lam, A, rhobar, etabar = (np.asarray(x, dtype=float) for x in (lam, A, rhobar, etabar))
    z = rhobar + 1j * etabar
    Al4 = A**2 * lam**4
    w = A * lam**3 * z * np.sqrt(1 - Al4) / (np.sqrt(1 - lam**2) * (1 - Al4 * z))
    t12 = np.arcsin(lam)
    t23 = np.arcsin(A * lam**2)
    t13 = np.arcsin(np.abs(w))
    delta = np.mod(np.angle(w), TWO_PI)
    return t12, t13, t23, delta


def wolfenstein_matrix(lam, A, rhobar, etabar):
    """Exact CKM matrix from Wolfenstein parameters (standard-form phases)."""
    return standard_matrix(*wolfenstein_to_standard(lam, A, rhobar, etabar))


def standard_to_wolfenstein(U):
    """Inverse map from any CKM-like unitary: (lambda, A, rhobar, etabar).

    lambda = |V_us| / sqrt(|V_ud|^2 + |V_us|^2), A lambda = |V_cb/V_us|,
    rhobar + i etabar = -V_ud V_ub^* / (V_cd V_cb^*)  (rephasing invariant).
    """
    V = np.asarray(U)
    a = np.abs(V)
    lam = a[..., 0, 1] / np.sqrt(a[..., 0, 0] ** 2 + a[..., 0, 1] ** 2)
    A = a[..., 1, 2] / (lam * a[..., 0, 1])
    z = -V[..., 0, 0] * np.conj(V[..., 0, 2]) / (V[..., 1, 0] * np.conj(V[..., 1, 2]))
    return lam, A, np.real(z), np.imag(z)
