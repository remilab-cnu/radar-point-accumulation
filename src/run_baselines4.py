"""Baselines round 4 — FAITHFUL Xia & Xu (2022) raw-spectrum row, Infineon ONLY
(equal-HPO protocol docs/EQUAL_HPO_PROTOCOL.md, baseline queue #5).

The row is feasible only on the Infineon BGT60TR13C set (raw ADC cubes available);
mHomeGes ships detected point clouds only -> cite-and-position, no runnable row.

Representation: their Sec. III raw-spectrum features RTA+DTA+ATA+ETA (incoherent
superposition of per-channel RD spectra, CFAR target box, 2-element DBF angle spectra)
+ Sec. V-A spatial position alignment + 64x64 rescale (see baselines_xiaxu.py).
Model: MyNetV2 (Fig. 13) in its published dimensions (~17.2M params).
Instances: frozen 2400-instance manifest replayed from the RAW cubes (same selection
logic as rep_variants.infineon_recs, verified against data/infineon_recs.pkl).
Grid: lr {3e-4, 1e-3} x ep {30, 40} x seeds {0,1,2}; their published default lr 1e-3
is inside the protocol pair. Folds: the frozen seed-0 4-fold over users.
Outputs: docs/baselines4.json (+ per-instance preds docs/baselines4_preds.npz).
"""
import hashlib
import json
import os
import time

import numpy as np

from baselines_xiaxu import (CACHE, DEVIATIONS, XiaXuNet, train_eval_xiaxu,
                             verify_against_pkl, xiaxu_dataset)
from rep_round3 import kfold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2)
EPOCH_BUDGETS = (30, 40)
LRS = (("3e-4", 3e-4), ("1e-3", 1e-3))


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


if __name__ == "__main__":
    t0 = time.time()
    X, y, subj, rec_idx, n_dets = xiaxu_dataset()
    print(f"Infineon Xia&Xu maps: {X.shape} ({time.time() - t0:.0f}s)", flush=True)
    replay_check = verify_against_pkl(y, subj, n_dets)
    print("frozen-manifest cross-check:", replay_check, flush=True)

    folds = kfold(subj, 4)
    b2 = os.path.join(DOCS, "baselines2.json")
    if os.path.exists(b2):                       # folds must equal the frozen protocol's
        frozen = json.load(open(b2))["datasets"]["Infineon"]["folds"]
        assert [sorted(map(str, f)) for f in folds] == frozen, "fold mismatch vs baselines2"

    results, preds = {}, {}
    tag, ncls = "Infineon", 5
    for lrname, lr in LRS:
        for ep in EPOCH_BUDGETS:
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                for sd in SEEDS:
                    a, yt, yp, ta = train_eval_xiaxu(X[tr], y[tr], X[te], y[te], ncls,
                                                     epochs=ep, lr=lr, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{tag}|XiaXu2022|lr{lrname}|ep{ep}|f{fi}|s{sd}"] = \
                        np.stack([te_idx, yt, yp])
            key = f"{tag}|XiaXu2022-MyNetV2|RTA+DTA+ATA+ETA|lr{lrname}|ep{ep}"
            results[key] = {"mean": float(np.mean(accs)) * 100,
                            "std": float(np.std(accs)) * 100,
                            "accs": [round(float(a) * 100, 2) for a in accs],
                            "min_train_acc": round(float(min(tr_accs)) * 100, 2),
                            "underfit": bool(min(tr_accs) < 0.95)}
            print(f"  {key:56s}: {results[key]['mean']:6.2f}% (+-{results[key]['std']:.1f})"
                  f"{'  UNDERFIT' if results[key]['underfit'] else ''}", flush=True)

    import torch
    out = {
        "manifest": {"infineon_pkl_md5": _md5(os.path.join(DATA, "infineon_recs.pkl")),
                     "xiaxu_maps_md5": _md5(CACHE),
                     "n_instances": {tag: int(len(y))},
                     "replay_check": replay_check},
        "datasets": {tag: {"n_instances": int(len(y)), "n_cls": ncls,
                           "chance": round(1 / ncls, 4),
                           "folds": [sorted(map(str, f)) for f in folds],
                           "test_N": [int(np.isin(subj, list(f)).sum()) for f in folds]}},
        "protocol": {
            "epochs": list(EPOCH_BUDGETS), "batch": 64, "seeds": list(SEEDS),
            "optimizer": "adam", "lrs": [lr for _, lr in LRS], "grad_clip": None,
            "aug": "none",
            "input": "Xia&Xu 2022 Sec. III raw-spectrum features RTA+DTA+ATA+ETA "
                     "(4ch 64x64), spatial-position-aligned (Sec. V-A), per-map "
                     "max-norm; built from RAW ADC cubes, no detected point clouds",
            "lr_policy": "published default 1e-3 (Adam, constant) is inside the "
                         "protocol pair {3e-4, 1e-3}; both reported",
            "models": {"XiaXu2022-MyNetV2":
                       "3x [conv3x3(s1,p1)-BN-ReLU-maxpool2] depths 64/128/256 -> "
                       "FC1024-ReLU-dropout0.5 -> FC5 "
                       f"({sum(p.numel() for p in XiaXuNet(4, ncls).parameters()):,} "
                       "params)"},
            "scope": "Infineon ONLY: the faithful row requires raw radar cubes; "
                     "mHomeGes distributes detected point clouds only "
                     "(cite-and-position, no runnable row)",
            "deviations": DEVIATIONS},
        "results": results}
    json.dump(out, open(os.path.join(DOCS, "baselines4.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, "baselines4_preds.npz"), **preds)
    print(f"\nwrote docs/baselines4.json + baselines4_preds.npz in {time.time() - t0:.0f}s",
          flush=True)
