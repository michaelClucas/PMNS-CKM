"""Loader for the measured-data layer in data/.

Every measured number used by the program comes through this module. Sources and
versions are listed in data/SOURCES.md:
    CKM and masses: PDG Review of Particle Physics 2026
    PMNS and neutrino splittings: NuFIT 6.1 (2025)

Record format: {"value", "err_plus", "err_minus", "scale", "scheme", "source", ...}.
err_plus and err_minus are positive 1 sigma errors.
"""
import csv
import functools
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
NUFIT = DATA / "nufit"
ORDERINGS = ("NO", "IO")
ANALYSES = {"IC23_woSKatm": "TBoff", "IC24_wSKatm": "TByes"}
DEFAULT_ANALYSIS = "IC23_woSKatm"


@functools.lru_cache(maxsize=None)
def _json(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def _check_ordering(ordering):
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")


def load_ckm() -> dict:
    """CKM records from PDG 2026, flat name -> record.

    Names: lambda, A, rhobar, etabar (Wolfenstein); s12, s13, s23, delta_rad (standard
    parametrization); J; Vud ... Vtb (global-fit magnitudes, unitarity imposed);
    direct_Vud ... direct_Vtb (direct determinations, no unitarity).
    'correlations' is None: PDG does not publish the global-fit correlation matrix.
    'meta' holds the source description; 'alternatives' holds the UTfit-method and
    tree-level-only Wolfenstein fits.
    """
    d = _json("ckm.json")
    out = {}
    for group in ("wolfenstein", "standard_parametrization", "jarlskog", "magnitudes_global_fit"):
        out.update(d[group])
    for k, v in d["magnitudes_direct"].items():
        if k != "note":
            out["direct_" + k] = v
    out["correlations"] = None
    out["alternatives"] = d["wolfenstein_alternatives"]
    out["meta"] = d["meta"]
    return out


def load_pmns(ordering: str, analysis: str = DEFAULT_ANALYSIS) -> dict:
    """PMNS records from NuFIT 6.1 for one mass ordering ("NO" or "IO").

    Names: s12sq, s13sq, s23sq, theta12_deg, theta13_deg, theta23_deg, dcp_deg,
    dm21sq_1e-5eV2, dm3l_sq_1e-3eV2 (Delta m^2_31 > 0 for NO, Delta m^2_32 < 0 for IO).
    Each record also has range_3sigma. Errors are relative to the local minimum of the
    requested ordering. 'delta_chi2_IO_minus_NO' is the ordering penalty for this analysis.
    analysis: "IC23_woSKatm" (default) or "IC24_wSKatm".
    """
    _check_ordering(ordering)
    d = _json("pmns.json")["analyses"][analysis]
    out = dict(d[ordering])
    out["delta_chi2_IO_minus_NO"] = d["delta_chi2_IO_minus_NO"]
    out["analysis"] = analysis
    return out


def load_masses() -> dict:
    """Fermion masses in GeV (PDG 2026) and neutrino splittings in eV^2 (NuFIT 6.1).

    Keys: m_u, m_d, m_s, m_c, m_b, m_t_direct, m_t_pole_xsec, m_t_msbar_xsec, m_e, m_mu,
    m_tau, the ratios (m_u_over_m_d, ...), and 'neutrino_splittings' =
    {analysis: {"NO": {"dm21sq", "dm31sq"}, "IO": {"dm21sq", "dm32sq"}}}.
    Every record carries its own scale and scheme; they differ between quarks.
    """
    d = _json("masses.json")
    out = {}
    for group in ("quarks", "charged_leptons", "quark_mass_ratios"):
        out.update(d[group])
    out["neutrino_splittings"] = d["neutrino_splittings"]
    out["meta"] = d["meta"]
    return out


# ---------------------------------------------------------------- NuFIT chi^2 tables

@functools.lru_cache(maxsize=None)
def _table(analysis, ordering, tag):
    path = NUFIT / f"v61_{ANALYSES[analysis]}_{ordering}_{tag}.csv"
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return rows[0], np.array(rows[1:], dtype=float)


@functools.lru_cache(maxsize=None)
def _profile_1d(analysis, ordering, tag):
    _, a = _table(analysis, ordering, tag)
    order = np.argsort(a[:, 0])
    return a[order, 0], a[order, 1]


@functools.lru_cache(maxsize=None)
def _grid_2d(analysis, ordering, tag):
    _, a = _table(analysis, ordering, tag)
    x, y = np.unique(a[:, 0]), np.unique(a[:, 1])
    z = np.full((len(x), len(y)), np.nan)
    z[np.searchsorted(x, a[:, 0]), np.searchsorted(y, a[:, 1])] = a[:, 2]
    if np.isnan(z).any():
        raise ValueError(f"{tag} table is not a full rectangular grid")
    return x, y, z


def _interp_1d(x, y, v):
    if v < x[0] or v > x[-1]:
        return np.inf
    return float(np.interp(v, x, y))


def _interp_2d(x, y, z, u, v):
    if u < x[0] or u > x[-1] or v < y[0] or v > y[-1]:
        return np.inf
    i = min(max(np.searchsorted(x, u) - 1, 0), len(x) - 2)
    j = min(max(np.searchsorted(y, v) - 1, 0), len(y) - 2)
    tx = (u - x[i]) / (x[i + 1] - x[i])
    ty = (v - y[j]) / (y[j + 1] - y[j])
    return float((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
                 + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])


def _wrap_dcp(d):
    """Map an angle in degrees to the table range [-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def ordering_offset(ordering: str, analysis: str = DEFAULT_ANALYSIS) -> float:
    """Delta chi^2 of the ordering's local minimum above the global minimum, from the tables."""
    _check_ordering(ordering)
    return float(min(_profile_1d(analysis, ordering, t)[1].min() for t in ("T12", "T13", "T23", "DCP")))


def pmns_delta_chi2(ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None,
                    dm21sq_eV2=None, dm3l_sq_eV2=None, analysis=DEFAULT_ANALYSIS,
                    method="auto", relative_to="local") -> float:
    """Approximate NuFIT 6.1 Delta chi^2 at the given oscillation parameters.

    NuFIT publishes 1D, 2D and one 3D (s23sq, dm3l, dcp) projection of the 6D chi^2, not
    the full 6D table, so the exact joint chi^2 cannot be rebuilt. This function sums
    marginal (profiled) Delta chi^2 tables, each shifted so its own minimum is zero:

      method="auto": if s23sq and dcp_deg are both given, use the 2D (s23sq, dcp) table
          for that pair (keeps the strongest correlation, the octant/delta_CP one), and
          add the 1D profiles for every other given parameter.
      method="1d": sum the 1D profiles of all given parameters.

    Both ignore correlations between the summed pieces (e.g. s13sq-s23sq, s13sq-dcp,
    dm3l-s23sq). Each profile already minimizes over the other parameters, so the sum
    is not the joint chi^2; treat it as an approximation that is exact only when
    the parameters are uncorrelated.

    Parameters left as None are profiled out and contribute 0.
    dcp_deg is periodic (any real value accepted). dm21sq_eV2 is Delta m^2_21 in eV^2;
    dm3l_sq_eV2 is Delta m^2_31 (NO, positive) or Delta m^2_32 (IO, negative) in eV^2.
    Points outside the tabulated grid return inf.

    relative_to="local": 0 at the best fit of the requested ordering.
    relative_to="global": adds the ordering penalty (0 for NO, about 1.5 for IO in the
        default analysis), i.e. Delta chi^2 relative to the global minimum.
    """
    _check_ordering(ordering)
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {tuple(ANALYSES)}")
    if method not in ("auto", "1d"):
        raise ValueError("method must be 'auto' or '1d'")
    if relative_to not in ("local", "global"):
        raise ValueError("relative_to must be 'local' or 'global'")

    one_d = {"T12": s12sq, "T13": s13sq, "T23": s23sq,
             "DCP": None if dcp_deg is None else _wrap_dcp(dcp_deg),
             "DMS": None if dm21sq_eV2 is None else float(np.log10(dm21sq_eV2)) if dm21sq_eV2 > 0 else -np.inf,
             "DMA": None if dm3l_sq_eV2 is None else dm3l_sq_eV2 * 1e3}
    total = 0.0
    if method == "auto" and s23sq is not None and dcp_deg is not None:
        x, y, z = _grid_2d(analysis, ordering, "T23-DCP")
        total += _interp_2d(x, y, z - z.min(), s23sq, one_d["DCP"])
        one_d["T23"] = one_d["DCP"] = None
    for tag, v in one_d.items():
        if v is None:
            continue
        if not np.isfinite(v):
            return np.inf
        x, y = _profile_1d(analysis, ordering, tag)
        total += _interp_1d(x, y - y.min(), v)
    if relative_to == "global":
        total += ordering_offset(ordering, analysis)
    return total
