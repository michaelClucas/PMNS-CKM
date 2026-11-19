"""Leak probe: detect measured data that a model reads without declaring it (exp 012 H3).

    leak_probe(model_fn, declared_inputs=(), files=("ckm.json", "pmns.json", "masses.json"))
        -> list of LeakFlag(key, record, moved)

model_fn() takes no arguments and returns a dict of predictions (bind fitted parameters
with a lambda). The probe calls it once on the real data, then once per measured record
in data/<files> with that record's value shifted by +n_sigma * err_plus (plus a 1e-9
relative bump, so a record with zero error still moves), one record at a time. The shift
goes through a patched src.data._json, so the model sees it through the official loaders
(load_ckm, load_pmns, load_masses, and everything built on them, such as
src.baselines.data_context and src.rg). Every lru_cache in loaded src.* modules is cleared
before each call, so cached derived quantities (src.rg Yukawas, ckm_at results, the
evaluate registry) are recomputed from the shifted data, and cleared again at the end so
no shifted value survives the probe.

A record is flagged if shifting it moves any output by more than rtol (relative). Records
whose key is in declared_inputs are skipped. Keys are src.evaluate observable names:
  ckm.json     ckm_<name> (global fit and the alternative Wolfenstein fits), ckm_direct_<V>
  pmns.json    the record name (s12sq, theta12_deg, dcp_deg, dm21sq_1e-5eV2, ...), for
               every analysis and ordering
  masses.json  the record name (m_c, m_c_over_m_s, ...); neutrino splittings dm21sq, dm31sq, dm32sq
Matching is exact: declaring ckm_s12 does not license a read of ckm_Vus (an alias is a
different record and is flagged).

Not caught: reads of the NuFIT CSV tables, reads that bypass src.data (open() on a data
file), numbers hardcoded in the model. Candidate code still needs a read for literals.
Cost: (number of records + 1) model calls, about 150 for the default files.
"""
from __future__ import annotations

import copy
import json
import sys
from typing import Callable, Mapping, NamedTuple, Sequence

import numpy as np

from . import data as _data

DEFAULT_FILES = ("ckm.json", "pmns.json", "masses.json")

# lru caches that depend only on the NuFIT CSV tables (not on data/*.json): kept, for speed.
_KEEP_CACHES = {("src.data", "_table"), ("src.data", "_profile_1d"), ("src.data", "_grid_2d"),
                ("src.data", "_shifted_profile_1d"), ("src.data", "_shifted_grid_2d"),
                ("src.data", "_grid_2d_uniform")}


class LeakFlag(NamedTuple):
    key: str          # src.evaluate-style key of the shifted record
    record: str       # file/path/of/record
    moved: tuple      # model outputs that moved


def _records(obj, path=()):
    if isinstance(obj, dict):
        if "value" in obj and "err_plus" in obj and isinstance(obj["value"], (int, float)):
            yield path, obj
            return
        for k, v in obj.items():
            yield from _records(v, path + (k,))


def record_key(fname: str, path: Sequence[str]) -> str:
    """src.evaluate-style key for a record at `path` inside data/<fname>."""
    leaf = path[-1]
    if fname == "ckm.json":
        return ("ckm_direct_" if path[0] == "magnitudes_direct" else "ckm_") + leaf
    return leaf


def _clear_src_caches():
    for name, mod in list(sys.modules.items()):
        if not (name == "src" or name.startswith("src.")) or mod is None:
            continue
        for attr, obj in list(vars(mod).items()):
            if (name, attr) in _KEEP_CACHES:
                continue
            if callable(getattr(obj, "cache_clear", None)) and getattr(obj, "__module__", None) == name:
                obj.cache_clear()


def _outputs(model_fn):
    out = {}
    for k, v in dict(model_fn()).items():
        try:
            a = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            continue
        out[k] = a
    return out


def leak_probe(model_fn: Callable[[], Mapping], declared_inputs: Sequence[str] = (),
               files: Sequence[str] = DEFAULT_FILES, rtol: float = 1e-12, n_sigma: float = 1.0):
    """Return a LeakFlag for every undeclared measured record that moves model_fn()'s output."""
    declared = set(declared_inputs)
    base_raw = {}
    for f in files:
        with open(_data.DATA / f) as fh:
            base_raw[f] = json.load(fh)
    orig_json = _data._json
    flagged = []
    try:
        _clear_src_caches()
        ref = _outputs(model_fn)
        for f in files:
            for path, _ in list(_records(base_raw[f])):
                key = record_key(f, path)
                if key in declared:
                    continue
                mod = copy.deepcopy(base_raw[f])
                node = mod
                for p in path:
                    node = node[p]
                node["value"] = (node["value"] + n_sigma * abs(node["err_plus"])
                                 + 1e-9 * abs(node["value"]) + 1e-300)

                def patched(name, _f=f, _mod=mod):
                    return _mod if name == _f else orig_json(name)

                _data._json = patched
                try:
                    _clear_src_caches()
                    out = _outputs(model_fn)
                finally:
                    _data._json = orig_json
                moved = tuple(k for k in ref
                              if k not in out or out[k].shape != ref[k].shape
                              or not np.allclose(out[k], ref[k], rtol=rtol, atol=0, equal_nan=True))
                if moved:
                    flagged.append(LeakFlag(key, f + "/" + "/".join(path), moved))
    finally:
        _data._json = orig_json
        _clear_src_caches()
    return flagged
