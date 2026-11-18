"""The search pipeline shared by the real run and every fake universe.

For each (target, ordering): cumulative best |pull| for complexity <= c, number of expressions
with |pull| < 1 and < 2, and number of valid (predictive) expressions at complexity <= c.
Only valid expressions (symlib.predictions rules) are scored.
"""
from __future__ import annotations

import functools

import numpy as np

import symlib as S

ORDERINGS = ("NO", "IO")


@functools.lru_cache(maxsize=None)
def profile_scorer(target, ordering):
    """Built once and cached as a pickle under artifacts/exp010 (pure function of data/)."""
    import pickle
    f = S.ART / f"profile_{target}_{ordering}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    sc = S.ProfileScorer(target, ordering)
    S.ART.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(sc))
    return sc


def level_slices(comp):
    cmax = int(comp.max())
    edges = np.searchsorted(comp, np.arange(1, cmax + 2))
    return [slice(edges[c - 1], edges[c]) for c in range(1, cmax + 1)]


def valid_preds(v, T, comp):
    """target -> (idx of valid expressions, p, s, comp) ; comp stays sorted."""
    out = {}
    for tgt in S.TARGETS:
        p, s = S.predictions(v, T, tgt)
        idx = np.nonzero(np.isfinite(p))[0]
        out[tgt] = (idx, p[idx], None if s is None else s[idx], comp[idx])
    return out


def summarize(absP, comp_valid, cmax):
    starts = np.searchsorted(comp_valid, np.arange(1, cmax + 1))
    best = np.full(cmax, np.inf)
    n1 = np.zeros(cmax, int)
    n2 = np.zeros(cmax, int)
    nv = np.diff(np.append(starts, len(comp_valid)))
    x = np.where(np.isfinite(absP), absP, np.inf)
    for c in range(cmax):
        a, b = starts[c], (starts[c + 1] if c + 1 < cmax else len(x))
        if b > a:
            seg = x[a:b]
            best[c] = seg.min()
            n1[c] = int((seg < 1).sum())
            n2[c] = int((seg < 2).sum())
    return np.minimum.accumulate(best), np.cumsum(n1), np.cumsum(n2), np.cumsum(nv)


def run_profile(v, T, comp, keep_pulls=False):
    """Official-table scoring (real targets): dict[(target, ordering)] -> (best, n1, n2, nvalid[, pulls])."""
    cmax = int(comp.max())
    vp = valid_preds(v, T, comp)
    out = {}
    for tgt, (idx, p, s, cv) in vp.items():
        for o in ORDERINGS:
            pull = profile_scorer(tgt, o).pull(p, s)
            r = summarize(np.abs(pull), cv, cmax)
            if keep_pulls:
                full = np.full(len(v), np.nan)
                full[idx] = pull
                r = r + (full,)
            out[(tgt, o)] = r
    return out


def run_gauss_many(v, T, comp, centers_list):
    """Gaussian scoring with NuFIT widths around arbitrary centres, for many fake target sets
    at once (inputs fixed). centers_list: list of dict[(target, ordering)] -> centre."""
    cmax = int(comp.max())
    vp = valid_preds(v, T, comp)
    gs = {(t, o): S.GaussScorer(t, o) for t in S.TARGETS for o in ORDERINGS}
    outs = []
    for centers in centers_list:
        out = {}
        for (tgt, o), c in centers.items():
            idx, p, s, cv = vp[tgt]
            out[(tgt, o)] = summarize(np.abs(gs[(tgt, o)].pull(p, s, c)), cv, cmax)
        outs.append(out)
    return outs
