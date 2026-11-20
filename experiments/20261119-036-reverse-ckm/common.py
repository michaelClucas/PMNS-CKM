"""Exp 036 shared code: features, data split, flow wrapper (adapted from exp 028 train.py)."""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import pathlib
import sys

import numpy as np
import torch
import zuko

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts" / "036"
DATA = ROOT / "artifacts" / "025" / "data.npz"
FAMS = ("GEN", "CTRL", "F1", "F2", "MN")
SPLIT_SEED = 3601
ARCH = dict(transforms=4, hidden_features=(96, 96), bins=8)
SEEDS = (0, 1, 2)


def logit(p):
    return np.log(p) - np.log1p(-p)


def ckm_target(s12, s13, s23, delta_rad):
    d = np.clip(np.mod(delta_rad, 2 * np.pi) / (2 * np.pi), 1e-6, 1 - 1e-6)
    return np.stack([np.log10(s12), np.log10(s13), np.log10(s23), logit(d)], -1)


def pmns_ctx(s12sq, s13sq, s23sq, dcp_rad):
    c = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
    return np.stack([c(s12sq), c(s13sq), c(s23sq), np.cos(dcp_rad), np.sin(dcp_rad)], -1)


def load():
    """exp 025 data restricted to FAMS; returns dict with c (target), p, m, fam, split."""
    D = np.load(DATA)
    allf = [str(f) for f in D["families"]]
    keep = np.isin(D["fam"].astype(int), [allf.index(f) for f in FAMS])
    fam = np.array([FAMS.index(allf[i]) for i in D["fam"][keep].astype(int)])
    ck, cm, y = D["ctx_ckm"][keep], D["ctx_m"][keep], D["y"][keep]
    c = ckm_target(10 ** ck[:, 0], 10 ** ck[:, 1], 10 ** ck[:, 2], np.arctan2(ck[:, 4], ck[:, 3]))
    p = pmns_ctx(y[:, 0], y[:, 1], y[:, 2], y[:, 3])
    m = np.concatenate([cm, y[:, 4:5]], 1)
    n = len(fam)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    split = np.empty(n, int)
    split[perm[: int(0.8 * n)]] = 0
    split[perm[int(0.8 * n): int(0.9 * n)]] = 1
    split[perm[int(0.9 * n):]] = 2
    ok = np.all(np.isfinite(c), 1) & np.all(np.isfinite(p), 1) & np.all(np.isfinite(m), 1)
    return dict(c=c, p=p, m=m, fam=fam, onehot=np.eye(len(FAMS))[fam], split=np.where(ok, split, -1))


class Flow:
    """Family-conditional NSF on the 4D CKM target; context = standardised physical ++ one-hot."""

    def __init__(self, n_phys, n_fam, cm, cs, ym, ys):
        self.n_fam = n_fam
        self.flow = zuko.flows.NSF(features=4, context=n_phys + n_fam, **ARCH)
        self.cm, self.cs, self.ym, self.ys = (np.asarray(a, np.float32) for a in (cm, cs, ym, ys))

    def _ctx(self, c, onehot):
        return torch.as_tensor(np.concatenate([(c - self.cm) / self.cs, onehot], 1).astype(np.float32))

    def log_prob(self, y, c, onehot, chunk=65536):
        out = []
        self.flow.eval()
        with torch.no_grad():
            for i in range(0, len(y), chunk):
                yy = torch.as_tensor(((y[i:i + chunk] - self.ym) / self.ys).astype(np.float32))
                out.append(self.flow(self._ctx(c[i:i + chunk], onehot[i:i + chunk])).log_prob(yy).numpy())
        return np.concatenate(out).astype(float)

    def save(self, path):
        torch.save(dict(state=self.flow.state_dict(), cm=self.cm, cs=self.cs, ym=self.ym, ys=self.ys,
                        n_phys=len(self.cm), n_fam=self.n_fam), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(d["n_phys"], d["n_fam"], d["cm"], d["cs"], d["ym"], d["ys"])
        m.flow.load_state_dict(d["state"])
        return m
