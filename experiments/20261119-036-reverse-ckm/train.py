"""Exp 036 training. Usage: python train.py <q1|q0> <seed>
q1(CKM | PMNS, masses, family), q0(CKM | masses, family); settings in prereg.md.
Writes artifacts/036/models/<kind>_s<seed>.pt and curves/<kind>_s<seed>.json."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from common import ART, HERE, Flow, load

BATCH, LR, MAX_EPOCHS, PATIENCE, MIN_DELTA, MAX_SECONDS = 1024, 1e-3, 60, 5, 1e-3, 900


def train(kind, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = load()
    ctx = np.concatenate([D["p"], D["m"]], 1) if kind == "q1" else D["m"]
    tr, va = D["split"] == 0, D["split"] == 1
    y = D["c"]
    m = Flow(ctx.shape[1], D["onehot"].shape[1], ctx[tr].mean(0), ctx[tr].std(0), y[tr].mean(0), y[tr].std(0))
    X = m._ctx(ctx[tr], D["onehot"][tr])
    Y = torch.as_tensor(((y[tr] - m.ym) / m.ys).astype(np.float32))
    opt = torch.optim.Adam(m.flow.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, bad, curve, reason, best_state, t0 = np.inf, 0, [], "max_epochs", None, time.time()
    for ep in range(MAX_EPOCHS):
        m.flow.train()
        perm = torch.randperm(len(Y))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = -m.flow(X[idx]).log_prob(Y[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.flow.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        v = float(-m.log_prob(y[va], ctx[va], D["onehot"][va]).mean())
        sched.step(v)
        lr = opt.param_groups[0]["lr"]
        curve.append((ep, tot / len(perm), v, lr, time.time() - t0))
        print(f"[{kind} s{seed}] ep {ep} train {tot / len(perm):.4f} val {v:.4f} lr {lr:.1e} {time.time() - t0:.0f}s", flush=True)
        if v < best - MIN_DELTA:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in m.flow.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            reason = "early_stop"
            break
        if time.time() - t0 > MAX_SECONDS:
            reason = "wall_cap"
            break
    m.flow.load_state_dict(best_state)
    (ART / "models").mkdir(parents=True, exist_ok=True)
    m.save(ART / "models" / f"{kind}_s{seed}.pt")
    out = dict(kind=kind, seed=seed, n_train=int(tr.sum()), n_val=int(va.sum()), best_val_nll=float(best),
               best_epoch=int(np.argmin([r[2] for r in curve])), epochs=len(curve), stop_reason=reason,
               seconds=time.time() - t0, curve=curve)
    (HERE / "curves").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "curves" / f"{kind}_s{seed}.json", "w"), indent=1)
    print(f"[{kind} s{seed}] done best val {best:.4f} ({reason}, {len(curve)} epochs)", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]))
