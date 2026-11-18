"""Prototype leakage detector for candidate models (proposed fix for finding F3).

leak_probe(model_fn, declared_inputs) reruns model_fn() with each measured record in
data/*.json shifted by +1 sigma, one record at a time, skipping the declared inputs (by
src.evaluate key, e.g. "ckm_s12" or "m_e"), and returns the records whose shift moves the
prediction. A model that reads measured data it did not declare, including an alias of a
declared input (ckm_Vus when only ckm_s12 is declared), is flagged. Shifts go through a
patched src.data._json, so the model sees them through the official loader.
Not caught: reads of the NuFIT CSV tables, reads that bypass src.data, hardcoded numbers.
"""
from __future__ import annotations

import copy
import json

import numpy as np

import src.data as _data


def _records(obj, path=()):
    if isinstance(obj, dict):
        if "value" in obj and "err_plus" in obj and isinstance(obj["value"], (int, float)):
            yield path, obj
            return
        for k, v in obj.items():
            yield from _records(v, path + (k,))


def _key(fname, path):
    """src.evaluate observable key for a record path (best effort; used to skip declared inputs)."""
    leaf = path[-1]
    if fname == "ckm.json":
        if path[0] == "magnitudes_direct":
            return "ckm_direct_" + leaf
        return "ckm_" + leaf
    if fname == "pmns.json":
        return f"pmns_{path[-2]}_{leaf}"  # analyses/<analysis>/<ordering>/<name>
    return leaf


def leak_probe(model_fn, declared_inputs=(), files=("ckm.json", "pmns.json", "masses.json"), rtol=1e-12):
    base_raw = {}
    for f in files:
        with open(_data.DATA / f) as fh:
            base_raw[f] = json.load(fh)
    orig_json = _data._json
    ref = {k: float(v) for k, v in model_fn().items()}
    flagged = []
    try:
        for f in files:
            for path, _ in list(_records(base_raw[f])):
                key = _key(f, path)
                if key in declared_inputs:
                    continue
                mod = copy.deepcopy(base_raw[f])
                node = mod
                for p in path:
                    node = node[p]
                node["value"] = node["value"] + abs(node["err_plus"]) + 1e-9 * abs(node["value"])

                def patched(name, _f=f, _mod=mod):
                    return _mod if name == _f else orig_json(name)

                _data._json = patched
                try:
                    out = {k: float(v) for k, v in model_fn().items()}
                finally:
                    _data._json = orig_json
                moved = [k for k in ref if not np.isclose(out.get(k, np.nan), ref[k], rtol=rtol, atol=0)]
                if moved and key not in [k for k, _ in flagged]:
                    flagged.append((key, moved))
    finally:
        _data._json = orig_json
    return flagged
