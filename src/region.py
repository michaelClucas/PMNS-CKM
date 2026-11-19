"""Fixed-parameter regions: restrict a fit to the region another dataset allows (exp 024, for exp 020's tau).

A Region is a set of grid points in d parameters with their Delta chi^2 from the dataset that
fixed them (e.g. tau from the lepton fit, Delta chi^2_lep <= 6.18):

    reg = Region.from_grid(points, chi2, level=6.18, names=("re_tau", "im_tau"), step=0.001)
    reg.save("region.json"); reg = Region.load("region.json")
    reg.contains(x); reg.dchi2_at(x)          membership / Delta chi^2 of the nearest grid cell
    reg.bounds()                              {name: (lo, hi)}, the bounding box (for param_bounds)
    reg.representatives(k)                    best point + farthest-point sample
    profile_over_region(fit_fn, reg, ...)     min over region points of fit_fn(point) (+ Delta chi^2)

Points may be given as a complex array (tau): they are stored as (Re, Im) columns.

JSON format: {"names": [...], "points": [[...], ...], "dchi2": [...], "level": L,
"chi2_min": c, "step": s or null, "meta": {...}}. dchi2 is relative to the minimum over the
full grid (chi2_min), not over the stored points.

Counting: a parameter restricted to a region fixed by an independent dataset is not a fitted
parameter of the new fit (it is not in k), but the joint chi^2 of the two datasets must carry
the region's Delta chi^2 (add_dchi2=True), otherwise the region's edge is used for free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["Region", "profile_over_region"]


def _as_points(points):
    p = np.asarray(points)
    if np.iscomplexobj(p):
        p = np.column_stack([p.real.ravel(), p.imag.ravel()])
    p = np.asarray(p, float)
    return p[:, None] if p.ndim == 1 else p


def _single(x):
    a = np.asarray(x)
    return a.ndim == 0 if np.iscomplexobj(a) else a.ndim <= 1


@dataclass
class Region:
    points: np.ndarray                 # (n, d)
    dchi2: np.ndarray                  # (n,), relative to chi2_min
    names: tuple
    level: float
    chi2_min: float = 0.0
    step: float | Sequence[float] | None = None   # grid step (scalar or per axis); None = nearest point only
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = _as_points(self.points)
        self.dchi2 = np.asarray(self.dchi2, float).ravel()
        self.names = tuple(self.names)
        if len(self.points) != len(self.dchi2):
            raise ValueError("points and dchi2 differ in length")
        if len(self.points) == 0:
            raise ValueError("empty region")
        if self.points.shape[1] != len(self.names):
            raise ValueError(f"{self.points.shape[1]} columns but names {self.names}")

    @classmethod
    def from_grid(cls, points, chi2, level: float, names: Sequence[str] = ("re_tau", "im_tau"),
                  step=None, chi2_min: float | None = None, meta: dict | None = None) -> "Region":
        """Keep the grid points with chi2 <= chi2_min + level (chi2_min defaults to min(chi2))."""
        P, c = _as_points(points), np.asarray(chi2, float).ravel()
        cmin = float(np.nanmin(c)) if chi2_min is None else float(chi2_min)
        m = c <= cmin + level
        return cls(P[m], c[m] - cmin, names, float(level), cmin, step, dict(meta or {}))

    @classmethod
    def load(cls, path) -> "Region":
        d = json.loads(open(path, encoding="utf-8").read())
        return cls(np.array(d["points"], float), np.array(d["dchi2"], float), d["names"], d["level"],
                   d.get("chi2_min", 0.0), d.get("step"), d.get("meta", {}))

    def save(self, path):
        step = None if self.step is None else np.atleast_1d(self.step).astype(float).tolist()
        d = dict(names=list(self.names), points=self.points.tolist(), dchi2=self.dchi2.tolist(),
                 level=self.level, chi2_min=self.chi2_min, step=step, meta=self.meta)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)

    def __len__(self):
        return len(self.points)

    def restrict(self, level: float) -> "Region":
        """Sub-region with dchi2 <= level (level <= self.level)."""
        m = self.dchi2 <= level
        return Region(self.points[m], self.dchi2[m], self.names, float(min(level, self.level)), self.chi2_min,
                      self.step, dict(self.meta))

    @property
    def best(self) -> np.ndarray:
        return self.points[int(np.argmin(self.dchi2))]

    def bounds(self) -> dict:
        """{name: (lo, hi)} bounding box of the points, widened by half a grid step."""
        h = np.zeros(len(self.names)) if self.step is None else np.broadcast_to(np.asarray(self.step, float),
                                                                               (len(self.names),)) / 2
        lo, hi = self.points.min(0) - h, self.points.max(0) + h
        return {n: (float(a), float(b)) for n, a, b in zip(self.names, lo, hi)}

    def _nearest(self, x):
        x = _as_points(x) if np.iscomplexobj(np.asarray(x)) else np.atleast_2d(np.asarray(x, float))
        if self.step is None:
            d = ((x[:, None, :] - self.points[None]) ** 2).sum(-1)
            return np.argmin(d, 1), np.ones(len(x), bool)
        s = np.broadcast_to(np.asarray(self.step, float), (len(self.names),))
        u = np.abs(x[:, None, :] - self.points[None]) / s          # distance in grid steps per axis
        j = np.argmin(u.max(-1), 1)
        inside = u[np.arange(len(x)), j].max(-1) <= 0.5 + 1e-9
        return j, inside

    def contains(self, x) -> np.ndarray | bool:
        """True where x lies in a grid cell of the region (without step: always True). x is one point
        (d,), a batch (n, d), or complex tau value(s)."""
        _, inside = self._nearest(x)
        return bool(inside[0]) if _single(x) else inside

    def dchi2_at(self, x, outside: float = np.inf):
        """Delta chi^2 of the grid cell containing x (nearest point), `outside` if x is not in the region."""
        j, inside = self._nearest(x)
        v = np.where(inside, self.dchi2[j], outside)
        return float(v[0]) if _single(x) else v

    def representatives(self, k: int) -> np.ndarray:
        """Best point plus farthest-point sample (exp 020 rep_points), k points (or all if fewer)."""
        pts = [int(np.argmin(self.dchi2))]
        while len(pts) < min(k, len(self.points)):
            d = np.min(((self.points[:, None, :] - self.points[pts][None]) ** 2).sum(-1), 1)
            pts.append(int(np.argmax(d)))
        return self.points[pts]


def profile_over_region(fit_fn: Callable[[np.ndarray], float], region: Region, points: np.ndarray | None = None,
                        add_dchi2: bool = True) -> dict:
    """Minimize over the region: fit_fn(point) -> chi2 of the new data with the region parameters
    fixed at `point` (the caller profiles its own free parameters inside fit_fn).

    points: subset of region points to visit (default all; e.g. region.representatives(k)).
    add_dchi2: add the region's Delta chi^2 at each point (joint chi^2 of both datasets).
    Returns best_point, best_chi2 (as returned by fit_fn), best_total, best_dchi2, and per-point
    arrays points, chi2, dchi2, total.
    """
    P = region.points if points is None else _as_points(points)
    dch = np.array([region.dchi2_at(p, outside=np.nan) for p in P], float)
    if np.any(np.isnan(dch)):
        raise ValueError("some points are outside the region")
    c = np.array([float(fit_fn(p)) for p in P])
    tot = c + dch if add_dchi2 else c.copy()
    j = int(np.nanargmin(tot))
    return dict(best_point=P[j], best_chi2=float(c[j]), best_total=float(tot[j]), best_dchi2=float(dch[j]),
                points=P, chi2=c, dchi2=dch, total=tot)
