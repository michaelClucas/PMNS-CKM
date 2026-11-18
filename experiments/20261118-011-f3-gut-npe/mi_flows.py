"""Amortised information estimate from the predictive flows (broad prior, NO-classified samples):
  I(PMNS ; CKM | masses) ~= E[ log q(y | masses, CKM) - log q(y | masses) ]   over held-out synthetic samples
plus the predictive coverage of the ql flow on held-out samples (HPD rank test) and the predictive at the real data.
  python mi_flows.py <variant>       -> mi_<variant>.json
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402
from train_npe import ART, Model, features, SIG  # noqa: E402

torch.set_num_threads(2)
REF = gen.gut_reference()


def main():
    v = sys.argv[1]
    Mq, Mm = Model.load(f"pred_ql_{v}"), Model.load(f"pred_mass_{v}")
    ck = torch.load(ART / f"npe_pred_ql_{v}.pt", weights_only=False)
    vi = np.asarray(ck["val_idx"])
    d = np.load(ART / f"ds_broad_{v}.npz")
    z, io = d["z"][vi].astype(float), d["io"][vi]
    zn = gen.add_noise(z, SIG, np.random.default_rng(12345))   # same fixed noise as training validation
    y = zn[:, gen.PMNS_IDX]
    lq = Mq.log_prob(y, features(zn, io, "ql"))
    lm = Mm.log_prob(y, features(zn, io, "mass"))
    diff = lq - lm
    res = {"variant": v, "n_heldout": int(len(y)),
           "I_pmns_ckm_given_masses_nats": round(float(diff.mean()), 4),
           "stderr": round(float(diff.std() / np.sqrt(len(diff))), 4),
           "mean_logq_ql": round(float(lq.mean()), 3), "mean_logq_mass": round(float(lm.mean()), 3)}
    # HPD coverage of the ql predictive on 300 held-out samples
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), 300, replace=False)
    ranks = []
    for i in sel:
        c = features(zn[i:i + 1], io[i:i + 1], "ql")[0]
        s = Mq.sample(c, 300)
        l = Mq.log_prob(np.vstack([y[i:i + 1], s]), np.repeat(c[None], 301, 0))
        ranks.append((l[1:] > l[0]).mean())
    ranks = np.array(ranks)
    res["ql_predictive_hpd_coverage_at_0.5_0.68_0.95"] = [round(float((ranks <= a).mean()), 3) for a in (0.5, 0.68, 0.95)]
    # predictive at real data (NO)
    z0 = np.array(REF["z"])[None]
    c0 = features(z0, np.array([0]), "ql")[0]
    s = Mq.sample(c0, 5000)
    l0 = Mq.log_prob(np.vstack([z0[:, gen.PMNS_IDX], s]), np.repeat(c0[None], 5001, 0))
    res["real_data_ql_predictive"] = {
        "median": dict(zip(["s12sq", "s13sq", "s23sq", "dl", "lg_dm"], np.round(np.median(s, 0), 4).tolist())),
        "q16_84": np.round(np.quantile(s, [0.16, 0.84], axis=0), 4).tolist(),
        "measured_hpd_level": round(float((l0[1:] > l0[0]).mean()), 4)}
    (HERE / f"mi_{v}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
