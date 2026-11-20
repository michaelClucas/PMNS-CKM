"""Deviation (run.log): GUT-scale fits only for MZ fits with chi2 < CUT plus a random check sample of the
rest. Writes artifacts/044/gut_ids_<g>.json.  python gut_subset.py A 200 600"""
import json
import sys

import numpy as np

from main import ART, log

g, cut, nchk = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
fits = [json.loads(l) for l in open(ART / f"fits_{g}_MZ.jsonl")]
near = [f["id"] for f in fits if f["chi2"] < cut]
rest = [f["id"] for f in fits if not f["chi2"] < cut]
rng = np.random.default_rng(4407)
chk = list(rng.choice(rest, size=min(nchk, len(rest)), replace=False))
json.dump(near + chk, open(ART / f"gut_ids_{g}.json", "w"))
json.dump(chk, open(ART / f"gut_check_ids_{g}.json", "w"))
log(f"gut_subset {g}: MZ chi2 < {cut}: {len(near)}; random check sample of the rest: {len(chk)}")
