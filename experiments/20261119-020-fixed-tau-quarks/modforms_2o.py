"""Vector-valued modular forms of the finite modular group 2O used by the 2O quark-lepton model.

q-expansions transcribed from arXiv:2409.15823 App. A (identical to arXiv:2307.14926 Eq. 2.17),
q = exp(2 pi i tau), truncated at q^6 (|q| <= 1.9e-3 for Im tau >= 1, so the truncation error is
< 1e-12 relative for the fits in this experiment; check_s() verifies the S transformation).

Each component is (prefactor, r, [c0, c1, ..., c6]) meaning prefactor * q^r * sum_n c_n q^n.
"""
import functools

import numpy as np

S2, S3 = np.sqrt(2.0), np.sqrt(3.0)

FORMS = {
    # weight 2
    ("2", 2): [(1, 0, [1, 24, 24, 96, 24, 144, 96]),
               (8 * S3, 0.5, [1, 4, 6, 8, 13, 12, 14])],
    ("3", 2): [(1, 0, [1, -8, 24, -32, 24, -48, 96]),
               (-4 * S2, 0.25, [1, 6, 13, 14, 18, 32, 31]),
               (-16 * S2, 0.75, [1, 2, 3, 6, 5, 6, 10])],
    # weight 3
    ("4h", 3): [(4 * S3, 5 / 8, [1, -1, -4, 3, 1, 3, 13]),
                (2 * S3, 3 / 8, [1, 1, -7, -6, 16, 9, -6]),
                (-8, 7 / 8, [1, -3, 3, -4, 3, 6, -3]),
                (1, 1 / 8, [1, 3, -6, -23, 12, 66, -15])],
    # weight 4
    ("1", 4): [(1, 0, [1, 240, 2160, 6720, 17520, 30240, 60480])],
    ("3p", 4): [(2 * S2, 0.5, [1, -4, -2, 24, -11, -44, 22]),
                (4, 0.75, [1, -6, 11, -2, -11, 14, -38]),
                (1, 0.25, [-1, 2, 11, -22, -50, 96, 121])],
    # weight 5
    ("2h", 5): [(1, 3 / 8, [1, -7, 9, 42, -112, -63, 378]),
                (2, 5 / 8, [1, -9, 28, -21, -63, 123, -35])],
    ("2hp", 5): [(1, 1 / 8, [1, 123, 378, -191, -1428, -1134, -735]),
                 (8, 7 / 8, [7, 51, -27, -28, -459, 378, -357])],
    ("4h", 5): [(4 * S3, 5 / 8, [-3, -37, -20, 63, 253, 207, -471]),
                (-2 * S3, 3 / 8, [1, 57, 73, -150, -240, -447, -198]),
                (8, 7 / 8, [5, 9, 63, -212, 111, -306, 513]),
                (1, 1 / 8, [1, -69, -198, 193, 684, 18, 801])],
    # weight 6
    ("1", 6): [(1, 0, [1, -504, -504 * 33, -504 * 244, -504 * 1057, -504 * 3126, -504 * 8052])],
    ("1p", 6): [(1, 0.5, [1, -12, 54, -88, -99, 540, -418])],
    ("2", 6): [(1, 0, [1, 264, 7944, 64416, 253704, 825264, 1938336]),
               (8 * S3, 0.5, [1, 244, 3126, 16808, 59293, 161052, 371294])],
}

xi = np.exp(1j * np.pi / 4)
RHO_S = {
    "1": np.array([[1.0]]),
    "1p": np.array([[-1.0]]),
    "2": -0.5 * np.array([[1, S3], [S3, -1]]),
    "2h": 1j / S2 * np.array([[1, 1], [1, -1]]),
    "2hp": 1j / S2 * np.array([[1, 1], [1, -1]]),
    "3": 0.5 * np.array([[0, S2, S2], [S2, -1, 1], [S2, 1, -1]]),
    "3p": -0.5 * np.array([[0, S2, S2], [S2, -1, 1], [S2, 1, -1]]),
    "4h": 1j / (2 * S2) * np.array([[1, 1, S3, -S3], [1, -1, -S3, -S3],
                                     [S3, -S3, 1, 1], [-S3, -S3, 1, -1]]),
}


@functools.lru_cache(maxsize=4096)
def _Y(rep, k, tau):
    """Modular form multiplet Y^(k)_rep(tau) as a complex numpy vector."""
    out = []
    for pref, r, c in FORMS[(rep, k)]:
        n = np.arange(len(c))
        out.append(pref * np.sum(np.array(c, float) * np.exp(2j * np.pi * tau * (n + r))))
    return tuple(out)


def Y(rep, k, tau):
    """Modular form multiplet Y^(k)_rep(tau) (memoised per tau)."""
    return np.array(_Y(rep, k, complex(tau)))


def check_s(tau=1.15j + 0.1):
    """Max relative violation of Y(-1/tau) = (-tau)^k rho(S) Y(tau) over all multiplets."""
    res = {}
    for (rep, k) in FORMS:
        lhs = Y(rep, k, -1 / tau)
        rhs = (-tau) ** k * RHO_S[rep] @ Y(rep, k, tau)
        res[(rep, k)] = float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)))
    return res


if __name__ == "__main__":
    for key, v in check_s().items():
        print(f"S check {key}: rel. violation {v:.2e}")
