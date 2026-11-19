"""Modular forms of level N = 3, 4, 5 (finite modular groups Gamma_3 = A4, Gamma_4 = S4, Gamma_5 = A5)
up to weight 8, as q-expansions in q^(1/N), plus the group representations and numerically
computed invariant (equivariant) couplings used to build mass matrices.

Construction
------------
Weight 2. M_2(Gamma(N)) is spanned by the logarithmic derivatives of the Siegel functions
g_a(tau), a = (A/N, B/N) != (0, 0) (Kubert-Lang, "Modular Units"). g_a^(12N) is a modular unit
for Gamma(N) (no zeros or poles in H), so D log g_a with D = q d/dq = (1/2 pi i) d/dtau is a
holomorphic weight-2 form for Gamma(N). From the product formula
    g_a = -q^{B2(a1)/2} e^{i pi a2 (a1-1)} (1 - q_z) prod_{n>=1} (1 - q^n q_z)(1 - q^n / q_z),
    q_z = e^{2 pi i a2} q^{a1},  B2(t) = t^2 - t + 1/6,
    D log g_a = B2(a1)/2 - sum_{n>=0} sum_{k>=1} (n+a1) x^k q^{k(n+a1)}
                          - sum_{n>=1} sum_{k>=1} (n-a1) x^-k q^{k(n-a1)},   x = e^{2 pi i a2}
(the n = 0 term of the first sum is dropped when a1 = 0: it is the constant (1 - x)).
The span has the Eisenstein dimension (#cusps - 1) = 3, 5, 11 for N = 3, 4, 5, which equals
dim M_2(Gamma(N)) (Gamma(3), Gamma(4), Gamma(5) have genus 0, no weight-2 cusp forms).

Weight 2k > 2. Products of k weight-2 forms; a linearly independent subset is chosen by pivoted
QR on values at random points. dim M_w(Gamma(N)) = w + 1 (N=3), 2w + 1 (N=4), 5w + 1 (N=5)
(Feruglio 1706.08749 Table 1 / Novichkov et al. 1812.02158); the products reach it (tested).

Representations. rho_V(T) is exact (q^(m/N) -> zeta_N^m q^(m/N)); rho_V(S) is the least-squares
solution of f(-1/tau) = tau^w rho_V(S) f(tau) on random points, verified on fresh points.
Convention: Y(gamma tau) = (c tau + d)^w rho(gamma) Y(tau).
Field irreps are explicit matrices (`IRREPS`), checked against the presentation
S^2 = (ST)^3 = T^N = 1 in tests:
  A4 (Feruglio 1706.08749 App. / Eq. (8)-(9) basis): 3: S = (1/3)[[-1,2,2],[2,-1,2],[2,2,-1]],
     T = diag(1, w, w^2); 1, 1', 1'': T = 1, w, w^2.
  S4 (Penedo-Petcov 1806.11040 basis up to an overall sign): triplet S = (1/2)[[0,r2,r2],[r2,-1,1],
     [r2,1,-1]], T = diag(1, i, -i) and its negative. The one present at weight 2 is called 3'
     (Penedo-Petcov: weight 2 = 2 + 3'). 1': S = T = -1.
  A5 (Novichkov et al. 1812.02158 App. A basis): 3: S = (1/sqrt5)[[1,-r2,-r2],[-r2,-phi,1/phi],
     [-r2,1/phi,-phi]], T = diag(1, z, z^4); 3': S = (1/sqrt5)[[-1,r2,r2],[r2,-1/phi,phi],
     [r2,phi,-1/phi]], T = diag(1, z^2, z^3).
Characters of the other irreps (S4: 2; A5: 4, 5) are obtained from tensor products.

Couplings. An invariant term (fields x modular forms) is an equivariant linear map, found as
the null space of (rho(g) action - 1) for g = S, T. With generalized CP (Novichkov, Penedo,
Petcov, Titov 1811.04933 and 1905.11970) in these symmetric bases, couplings are real when
the coupling structures have real q-expansion coefficients; we pick that real form
(`real_form`) and check its dimension equals the complex one.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np

QMAX = 20           # keep powers q^(m/N) for m < N*QMAX + 1
LEVELS = (3, 4, 5)
GROUP = {3: "A4", 4: "S4", 5: "A5"}
GROUP_ORDER = {3: 12, 4: 24, 5: 60}
WEIGHTS = (2, 4, 6, 8)


def dim_formula(N, w):
    return {3: w + 1, 4: 2 * w + 1, 5: 5 * w + 1}[N]


def n_terms(N):
    return N * QMAX + 1


# ---------------------------------------------------------------- weight 2: Siegel log-derivatives


def siegel_logderiv(A, B, N, M=None):
    """q^(1/N)-series coefficients (length M) of D log g_(A/N, B/N)."""
    M = n_terms(N) if M is None else M
    a1 = A / N
    x = np.exp(2j * np.pi * B / N)
    c = np.zeros(M, complex)
    c[0] += 0.5 * (a1 * a1 - a1 + 1.0 / 6.0)
    # first sum: exponent (in units 1/N) k (n N + A)
    for n in range(0, M):
        e = n * N + A
        if e == 0:
            continue
        if e >= M:
            break
        for k in range(1, M):
            if k * e >= M:
                break
            c[k * e] -= (n + a1) * x ** k
    for n in range(1, M):
        e = n * N - A
        if e >= M:
            break
        for k in range(1, M):
            if k * e >= M:
                break
            c[k * e] -= (n - a1) * x ** (-k)
    return c


def eval_series(coef, tau, N):
    """coef (..., M) complex, tau (B,) complex -> values (B, ...)."""
    tau = np.atleast_1d(np.asarray(tau, complex))
    M = coef.shape[-1]
    qn = np.exp(2j * np.pi * np.outer(tau, np.arange(M)) / N)  # (B, M)
    return np.tensordot(qn, coef, axes=([1], [-1]))


def mul_series(a, b):
    M = a.shape[-1]
    return np.convolve(a, b)[:M]


def _rand_tau(rng, n):
    """Random points near |tau| = 1 with Im tau and Im(-1/tau) both >= ~0.75."""
    r = rng.uniform(0.92, 1.08, n)
    th = rng.uniform(np.radians(55), np.radians(125), n)
    return r * np.exp(1j * th)


# ---------------------------------------------------------------- spaces of modular forms


class FormSpace:
    """Basis of M_w(Gamma(N)) as q^(1/N) coefficient rows, with rho(S), rho(T)."""

    def __init__(self, N, w, coef, rhoS=None, S_resid=None, mons=None):
        self.N, self.w, self.coef = N, w, coef  # coef (D, M)
        self.D = coef.shape[0]
        self.mons = mons
        self.rhoT = self._rho_T()
        if rhoS is None:
            self.rhoS, self.S_fit_resid = self._rho_S()
        else:
            self.rhoS, self.S_fit_resid = rhoS, S_resid

    def __call__(self, tau):
        return eval_series(self.coef, tau, self.N)  # (B, D)

    def _rho_T(self):
        M = self.coef.shape[1]
        ph = np.exp(2j * np.pi * np.arange(M) / self.N)
        # f(tau+1) = coef * ph ; express in basis: rho_T coef = coef * ph
        X, _ = _solve_in_basis(self.coef, self.coef * ph)
        return X

    def _rho_S(self, npts=None, seed=1):
        rng = np.random.default_rng(seed)
        npts = npts or 3 * self.D + 8
        t = _rand_tau(rng, npts)
        F = self(t)
        FS = self(-1 / t) / (t ** self.w)[:, None]
        X, *_ = np.linalg.lstsq(F, FS, rcond=None)
        resid = np.abs(F @ X - FS).max() / np.abs(FS).max()
        return X.T, resid


@functools.lru_cache(maxsize=None)
def weight2_space(N):
    M = n_terms(N)
    labels = [(A, B) for A in range(N) for B in range(N) if (A, B) != (0, 0)]
    R = np.array([siegel_logderiv(A, B, N, M) for A, B in labels])
    sel = _independent_rows(R)
    # Exact rho(S): g_a(gamma tau) = eps g_{a gamma}(tau) (a a row vector; Kubert-Lang), so
    # f_a(-1/tau) = tau^2 f_{a S}(tau) with a S = (a2, -a1); shifts of a by Z^2 and a -> -a only
    # change g_a by a constant factor, which D log removes.
    idx = {lab: i for i, lab in enumerate(labels)}
    img = np.array([R[idx[(labels[i][1] % N, (-labels[i][0]) % N)]] for i in sel])
    rhoS, resid = _solve_in_basis(R[sel], img)
    return FormSpace(N, 2, R[sel], rhoS=rhoS, S_resid=resid)


def _independent_rows(R, tol=1e-10):
    """Indices of a maximal linearly independent subset of coefficient rows (pivoted QR on the
    row-normalised coefficient matrix; values at points are too ill-conditioned for this)."""
    from scipy.linalg import qr
    Rn = R / np.abs(R).max(axis=1, keepdims=True)
    q, r, piv = qr(Rn.T, mode="economic", pivoting=True)
    d = int((np.abs(np.diag(r)) > tol * np.abs(r[0, 0])).sum())
    return np.sort(piv[:d])


def _solve_in_basis(basis, targets):
    """Coefficients X with targets = X @ basis in q-coefficient space (column-scaled lstsq)."""
    sc = np.abs(basis).max(axis=0) + 1e-300
    X, *_ = np.linalg.lstsq((basis / sc).T, (targets / sc).T, rcond=None)
    resid = np.abs(X.T @ basis - targets).max() / np.abs(targets).max()
    return X.T, resid


@functools.lru_cache(maxsize=None)
def space(N, w):
    if w == 2:
        return weight2_space(N)
    if w % 2 or w < 2:
        raise ValueError("even weights >= 2 only")
    base = weight2_space(N)
    k = w // 2
    mons = list(itertools.combinations_with_replacement(range(base.D), k))

    def prod(factors):
        c = factors[0]
        for f in factors[1:]:
            c = mul_series(c, f)
        return c

    R = np.array([prod([base.coef[j] for j in m]) for m in mons])
    sel = _independent_rows(R)
    coef = R[sel]
    # rho(S) exactly from rho_2(S): f_i(-1/tau) = tau^2 sum_j S_ij f_j, so a product of k
    # weight-2 forms maps to the product of the transformed factors.
    S2 = base.rhoS
    img = np.array([prod([S2[j] @ base.coef for j in mons[i]]) for i in sel])
    rhoS, resid = _solve_in_basis(coef, img)
    return FormSpace(N, w, coef, rhoS=rhoS, S_resid=resid, mons=[mons[i] for i in sel])


# ---------------------------------------------------------------- group representations

_w3 = np.exp(2j * np.pi / 3)
_z5 = np.exp(2j * np.pi / 5)
_r2 = np.sqrt(2.0)
_phi = (1 + np.sqrt(5.0)) / 2

_S4_TRIP = (np.array([[0, _r2, _r2], [_r2, -1, 1], [_r2, 1, -1]]) / 2, np.diag([1, 1j, -1j]))


def _one(s, t):
    return (np.array([[s]], complex), np.array([[t]], complex))


IRREPS = {
    3: {
        "1": _one(1, 1), "1p": _one(1, _w3), "1pp": _one(1, _w3 ** 2),
        "3": (np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3.0, np.diag([1, _w3, _w3 ** 2])),
    },
    4: {
        "1": _one(1, 1), "1p": _one(-1, -1),
        "2": (np.array([[-1, np.sqrt(3)], [np.sqrt(3), 1]]) / 2, np.diag([1.0, -1.0]).astype(complex)),
        # with these signs the triplet present at weight 2 is 3p, matching Penedo-Petcov's 2 + 3'
        "3": (-_S4_TRIP[0], -_S4_TRIP[1]),
        "3p": (_S4_TRIP[0], _S4_TRIP[1]),
    },
    5: {
        "1": _one(1, 1),
        "3": (np.array([[1, -_r2, -_r2], [-_r2, -_phi, 1 / _phi], [-_r2, 1 / _phi, -_phi]]) / np.sqrt(5),
              np.diag([1, _z5, _z5 ** 4])),
        "3p": (np.array([[-1, _r2, _r2], [_r2, -1 / _phi, _phi], [_r2, _phi, -1 / _phi]]) / np.sqrt(5),
               np.diag([1, _z5 ** 2, _z5 ** 3])),
    },
}
for _N in IRREPS:
    for _k, (_S, _T) in IRREPS[_N].items():
        IRREPS[_N][_k] = (np.asarray(_S, complex), np.asarray(_T, complex))

SINGLETS = {3: ("1", "1p", "1pp"), 4: ("1", "1p"), 5: ("1",)}
TRIPLETS = {3: ("3",), 4: ("3", "3p"), 5: ("3", "3p")}


@functools.lru_cache(maxsize=None)
def group_words(N):
    """Enumerate group elements as words in S, T using the faithful triplet."""
    S, T = IRREPS[N]["3"]
    found = {}
    frontier = [((), np.eye(3, dtype=complex))]

    def key(m):
        return tuple(np.round(np.concatenate([m.real.ravel(), m.imag.ravel()]) * 1e5).astype(int))

    found[key(np.eye(3))] = ()
    while frontier:
        nxt = []
        for w, m in frontier:
            for g, G in (("S", S), ("T", T)):
                m2 = m @ G
                k = key(m2)
                if k not in found:
                    found[k] = w + (g,)
                    nxt.append((w + (g,), m2))
        frontier = nxt
    return list(found.values())


def word_matrix(word, S, T):
    m = np.eye(S.shape[0], dtype=complex)
    for g in word:
        m = m @ (S if g == "S" else T)
    return m


@functools.lru_cache(maxsize=None)
def characters(N):
    """Character table rows (one per irrep) over group_words(N)."""
    words = group_words(N)
    ch = {}
    for name, (S, T) in IRREPS[N].items():
        ch[name] = np.array([np.trace(word_matrix(w, S, T)) for w in words])
    if N == 5:
        c33 = ch["3"] ** 2
        ch["5"] = c33 - ch["1"] - ch["3"]
        ch["4"] = ch["3"] * ch["3p"] - ch["5"]
    return ch


def decompose(N, rhoS, rhoT):
    """Irrep multiplicities of a representation given by rho(S), rho(T)."""
    words = group_words(N)
    chiV = np.array([np.trace(word_matrix(w, rhoS, rhoT)) for w in words])
    out = {}
    for name, chi in characters(N).items():
        m = (chiV * np.conj(chi)).sum() / len(words)
        out[name] = m
    return out


# ---------------------------------------------------------------- invariant couplings


def _nullspace(A, tol=1e-8):
    u, s, vh = np.linalg.svd(A)
    rank = int((s > tol * max(s.max(), 1.0)).sum())
    return vh[rank:].conj()


def real_form(structs, coefs):
    """Given complex basis structures (n, ...) with q-coefficient arrays coefs (n, ..., M),
    return real combinations (n_real, ...) whose coefficient arrays are real."""
    n = coefs.shape[0]
    X = coefs.reshape(n, -1).real
    Y = coefs.reshape(n, -1).imag
    # Im(sum (u + i v)(X + iY)) = sum u Y + v X = 0
    A = np.concatenate([Y, X], axis=0).T  # (P, 2n)
    scale = np.abs(A).max() or 1.0
    ns = _nullspace(A / scale, tol=1e-9)
    if ns.shape[0] == 0:
        return np.zeros((0,) + structs.shape[1:], complex), 0
    ns = ns.real if np.abs(ns.imag).max() < 1e-12 else ns
    # orthonormalise the real null vectors
    Q, _ = np.linalg.qr(np.real(ns).T)
    c = Q[:, : ns.shape[0]].T[:, :n] + 1j * Q[:, : ns.shape[0]].T[:, n:]
    out = np.tensordot(c, structs, axes=([1], [0]))
    return out, ns.shape[0]


@functools.lru_cache(maxsize=None)
def yukawa_row_structures(N, w, r_sing, r_trip):
    """Invariant couplings f^c (r_sing) Q (r_trip) Y^(w): returns real-coefficient row structures
    as q-coefficient arrays (n_c, 3, M) with R(tau)_j = sum_m coef[j, m] q^(m/N)."""
    V = space(N, w)
    rS, rT = IRREPS[N][r_sing]
    qS, qT = IRREPS[N][r_trip]
    D = V.D
    eqs = []
    for (a, Q, F) in ((rS[0, 0], qS, V.rhoS), (rT[0, 0], qT, V.rhoT)):
        # map C -> a Q^T C F - C on C (3, D)
        L = a * np.kron(Q.T, F.T) - np.eye(3 * D)
        eqs.append(L)
    ns = _nullspace(np.concatenate(eqs, axis=0))
    if ns.shape[0] == 0:
        return np.zeros((0, 3, V.coef.shape[1]))
    C = ns.reshape(-1, 3, D)
    coefs = np.einsum("njd,dm->njm", C, V.coef)
    R, nr = real_form(C, coefs)
    assert nr == C.shape[0], (N, w, r_sing, r_trip, nr, C.shape[0])
    out = np.einsum("njd,dm->njm", R, V.coef)
    assert np.abs(out.imag).max() < 1e-8 * max(np.abs(out).max(), 1)
    return _normalise(out.real, N)


@functools.lru_cache(maxsize=None)
def weinberg_structures(N, w, r_trip):
    """Invariant couplings L L Y^(w), L ~ r_trip: real symmetric structures (n, 3, 3, M)."""
    V = space(N, w)
    qS, qT = IRREPS[N][r_trip]
    D = V.D
    eqs = []
    for Q, F in ((qS, V.rhoS), (qT, V.rhoT)):
        # C (3,3,D) -> sum Q[j',j] Q[k',k] C[j',k',b'] F[b',b] - C
        L = np.kron(np.kron(Q.T, Q.T), F.T) - np.eye(9 * D)
        eqs.append(L)
    # symmetry C[j,k,b] = C[k,j,b]
    P = np.zeros((9 * D, 9 * D))
    for j in range(3):
        for k in range(3):
            for b in range(D):
                P[(j * 3 + k) * D + b, (j * 3 + k) * D + b] += 1
                P[(j * 3 + k) * D + b, (k * 3 + j) * D + b] -= 1
    eqs.append(P)
    ns = _nullspace(np.concatenate(eqs, axis=0))
    if ns.shape[0] == 0:
        return np.zeros((0, 3, 3, V.coef.shape[1]))
    C = ns.reshape(-1, 3, 3, D)
    coefs = np.einsum("njkd,dm->njkm", C, V.coef)
    R, nr = real_form(C, coefs)
    assert nr == C.shape[0], (N, w, r_trip, nr, C.shape[0])
    out = np.einsum("njkd,dm->njkm", R, V.coef)
    assert np.abs(out.imag).max() < 1e-8 * max(np.abs(out).max(), 1)
    return _normalise(out.real, N)


def _normalise(coef, N, tau0=2j):
    """Scale each structure to unit Frobenius norm at tau = 2i (coupling priors are O(1))."""
    v = eval_series(coef, np.array([tau0]), N)[0]  # (n, ...)
    nrm = np.sqrt((np.abs(v) ** 2).reshape(v.shape[0], -1).sum(1))
    return coef / nrm.reshape((-1,) + (1,) * (coef.ndim - 1))


def multiplet(N, w, irrep):
    """Equivariant maps V_w -> irrep (one per copy): q-coefficient arrays (copies, dim, M), for
    irreps with explicit matrices. Normalised so the first nonzero coefficient of the first
    component with one is 1; real-form chosen as for couplings."""
    V = space(N, w)
    S, T = IRREPS[N][irrep]
    d = S.shape[0]
    D = V.D
    eqs = []
    for G, F in ((S, V.rhoS), (T, V.rhoT)):
        # P F = G P for P (d, D):  vec: (F^T kron I) - (I kron G)
        eqs.append(np.kron(np.eye(d), F.T) - np.kron(G, np.eye(D)))
    ns = _nullspace(np.concatenate(eqs, axis=0))
    if ns.shape[0] == 0:
        return np.zeros((0, d, V.coef.shape[1]))
    P = ns.reshape(-1, d, D)
    coefs = np.einsum("nid,dm->nim", P, V.coef)
    R, nr = real_form(P, coefs)
    out = np.einsum("nid,dm->nim", R, V.coef).real
    for i in range(out.shape[0]):
        flat = out[i].ravel()
        j = np.argmax(np.abs(flat) > 1e-9 * np.abs(flat).max())
        out[i] /= flat[j]
    return out
