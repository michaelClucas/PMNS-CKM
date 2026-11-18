"""Haar-random unitaries and fake-universe generators.

Haar sampling uses Mezzadri's method (arXiv:math-ph/0609050): QR-decompose a matrix of
i.i.d. standard complex Gaussians and multiply column k of Q by r_kk / |r_kk|, which
removes the phase ambiguity of the QR factorisation and makes Q exactly Haar on U(n).

Known analytic fact used as a check: in the PDG standard parametrization the Haar
measure on U(3) induces, after dropping the 5 unphysical phases,

    dmu  proportional to  d(sin^2 t12) d(cos^4 t13) d(sin^2 t23) d(delta),

so sin^2 t12, cos^4 t13, sin^2 t23 and delta / (2 pi) are independent and uniform on
[0, 1]. `haar_angle_uniforms` maps angles to these four uniforms.
"""

from __future__ import annotations

import numpy as np

from .mixing import extract_angles

__all__ = [
    "haar_unitary",
    "haar_special_unitary",
    "haar_angle_uniforms",
    "sample_haar_angles",
    "fake_universe_pair",
    "shifted_data_universe",
    "split_normal",
]


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def haar_unitary(rng=None, size=None, n=3):
    """Haar-random U(n). size=None returns (n, n); size=m returns (m, n, n)."""
    rng = _rng(rng)
    shape = (() if size is None else (size,) if np.isscalar(size) else tuple(size)) + (n, n)
    Z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R, axis1=-2, axis2=-1)
    return Q * (d / np.abs(d))[..., None, :]


def haar_special_unitary(rng=None, size=None, n=3):
    """Haar-random SU(n): Haar U(n) divided by a branch of det^(1/n)."""
    U = haar_unitary(rng, size, n)
    det = np.linalg.det(U)
    return U * np.exp(-1j * np.angle(det) / n)[..., None, None]


def haar_angle_uniforms(U):
    """Map a (batch of) unitary to (sin^2 t12, cos^4 t13, sin^2 t23, delta/2pi).

    Each is Uniform[0, 1] if U is Haar-distributed.
    """
    t12, t13, t23, d = extract_angles(U)
    return np.sin(t12) ** 2, np.cos(t13) ** 4, np.sin(t23) ** 2, d / (2 * np.pi)


def sample_haar_angles(rng=None, size=None):
    """Draw (t12, t13, t23, delta) directly from the Haar-induced distribution.

    Much cheaper than sampling matrices when only the physical parameters are needed.
    """
    rng = _rng(rng)
    u = rng.random((4,) + (() if size is None else (size,) if np.isscalar(size) else tuple(size)))
    t12 = np.arcsin(np.sqrt(u[0]))
    t13 = np.arccos(u[1] ** 0.25)
    t23 = np.arcsin(np.sqrt(u[2]))
    delta = 2 * np.pi * u[3]
    return t12, t13, t23, delta


def fake_universe_pair(rng=None, size=None):
    """Independent Haar U(3) pair (fake CKM, fake PMNS).

    Under this null hypothesis there is no relation between the two matrices, so any
    relation a search finds in such pairs is a look-elsewhere artefact. Note the fake
    CKM is not hierarchical (typical angles are large); for a null that keeps the
    observed hierarchy use `shifted_data_universe` or a conditioned sampler.
    """
    rng = _rng(rng)
    return haar_unitary(rng, size), haar_unitary(rng, size)


def split_normal(rng, value, err_plus, err_minus, size=None):
    """Draw from a split (two-piece) normal: sigma = err_plus above value, err_minus below.

    Both halves carry probability 1/2, so `value` is the median. This is the usual
    reading of an asymmetric +err_plus / -err_minus quote.
    """
    rng = _rng(rng)
    z = rng.standard_normal(size)
    return value + np.where(z >= 0, z * err_plus, z * err_minus)


def shifted_data_universe(rng, *datasets, keys=None):
    """Fake universe made by shifting measured observables within their errors.

    Each dataset is a dict name -> {value, err_plus, err_minus, ...} (the format returned
    by src.data.load_ckm / load_pmns / load_masses). Returns copies of the datasets with
    'value' replaced by a split-normal draw around the measured value and the original
    stored under 'value_measured'; errors are unchanged. Draws are independent across
    observables (correlations ignored), and values are not clipped to physical ranges.

    keys: optional iterable restricting which names get shifted (others are copied).

    Caveat: the PMNS likelihood is non-Gaussian (NuFIT chi^2 tables, octant ambiguity
    in theta23, delta_CP poorly constrained). For PMNS, sampling from the tabulated
    Delta chi^2 profile is more faithful than split-normal draws; this function is the
    simple first-pass version.
    """
    rng = _rng(rng)
    keys = None if keys is None else set(keys)
    out = []
    for data in datasets:
        new = {}
        for name, entry in data.items():
            e = dict(entry)
            if keys is None or name in keys:
                ep, em = e.get("err_plus"), e.get("err_minus")
                if e.get("value") is not None and ep is not None and em is not None:
                    e["value_measured"] = e["value"]
                    e["value"] = float(split_normal(rng, e["value"], abs(ep), abs(em)))
            new[name] = e
        out.append(new)
    return out[0] if len(out) == 1 else tuple(out)
