"""Baselines round 2 — PointLSTM parity anchor (Min et al., CVPR 2020) under the frozen
equal-HPO protocol (docs/EQUAL_HPO_PROTOCOL.md; repositioning memo queue #3).

Arms: PointLSTM x frame-input (x,y,z,v,A; T=40,K=24) x lr{3e-4,1e-3} x ep{30,40} x
seeds {0,1,2}, batch 64, Adam, grad-clip 5.0, no augmentation.
BOTH LRs are reported for every dataset: the round-1 baselines collapsed on mHomeGes at
lr=1e-3 (constant-output signature, see baselines1_mhfix) so LR sensitivity is live —
do not pre-select a single LR.
Datasets: mHomeGes 5-fold, Infineon frozen pkl 4-fold, MM-Fi documented S2 split.
Outputs: docs/baselines2.json (+ per-instance preds docs/baselines2_preds.npz).

Input note: PointLSTM's published form is a per-frame point-set SEQUENCE, so it consumes
build_frame_tensors (same normalization as instance_to_points; time = bin index) instead
of the whole-instance 384-point common input, which is structurally inapplicable to a
recurrent point model. Published default LR is 1e-4 (pointlstm.yaml); this pass runs the
protocol pair {3e-4, 1e-3} — the published-default arm remains a TODO (same status as
round 1's pending 3-point LR selection).
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_variants import infineon_recs
from rep_round3 import kfold
from pointset_models import build_frame_tensors
from baselines_pointnets import train_eval_set_preds_tr
from baselines_pointlstm import PointLSTM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40); LRS = (("3e-4", 3e-4), ("1e-3", 1e-3))

results, preds, datasets = {}, {}, {}


def run_set(tag, insts, folds, ncls):
    ranges = fit_ranges([t[0] for t in insts])
    X, M, y, s = build_frame_tensors(insts, ranges)                     # (B,40,24,5)+mask
    datasets[tag] = {"n_instances": len(insts), "n_cls": ncls, "chance": round(1 / ncls, 4),
                     "folds": [sorted(map(str, f)) for f in folds],
                     "test_N": [int(np.isin(s, list(f)).sum()) for f in folds]}
    for lrname, lr in LRS:
        for ep in EPOCH_BUDGETS:
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(s, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                for sd in SEEDS:
                    a, yt, yp, ta = train_eval_set_preds_tr(
                        PointLSTM, X[tr], M[tr], y[tr], X[te], M[te], y[te], ncls, 5,
                        epochs=ep, lr=lr, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{tag}|PointLSTM|lr{lrname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
            key = f"{tag}|PointLSTM|frames(xyzvA)|lr{lrname}|ep{ep}"
            results[key] = {"mean": float(np.mean(accs)) * 100, "std": float(np.std(accs)) * 100,
                            "accs": [round(float(a) * 100, 2) for a in accs],
                            "min_train_acc": round(float(min(tr_accs)) * 100, 2),
                            "underfit": bool(min(tr_accs) < 0.95)}
            print(f"  {key:48s}: {results[key]['mean']:6.2f}% (+-{results[key]['std']:.1f})"
                  f"{'  UNDERFIT' if results[key]['underfit'] else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    print(f"mHomeGes {len(mh)} inst", flush=True)
    run_set("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10)

    inf = infineon_recs()
    print(f"Infineon {len(inf)} inst", flush=True)
    run_set("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5)

    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
    print(f"MM-Fi {len(mf)} inst (S2 split)", flush=True)
    run_set("MM-Fi", mf, [S2], 27)

    out = {"manifest": {"infineon_pkl_md5": hashlib.md5(
                            open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest(),
                        "n_instances": {k: v["n_instances"] for k, v in datasets.items()}},
           "datasets": datasets,
           "protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "seeds": list(SEEDS),
                        "optimizer": "adam", "lrs": [lr for _, lr in LRS], "grad_clip": 5.0,
                        "aug": "none",
                        "input": "published-form frame sequence: build_frame_tensors "
                                 "(x,y,z,v,A), T=40 bins, K=24 pts/bin, masked",
                        "lr_policy": "protocol pair {3e-4,1e-3} both reported (round-1 "
                                     "mHomeGes collapse at 1e-3 keeps LR sensitivity live); "
                                     "published default 1e-4 arm pending (pointlstm.yaml)",
                        "models": {"PointLSTM": "phi MLP[in,64,128] -> PointLSTM cell "
                                                "(k=4 masked kNN t->t-1, shared gates on "
                                                "[feat_i; rel_xyz; h_j], neighbor max-pool "
                                                "of h and c, hidden 128) -> masked "
                                                "spatio-temporal max+mean pool -> FC"},
                        "deviations": [
                            "compact backbone: pointwise MLP + 1 PointLSTM layer (hidden 128) "
                            "+ masked ST max+mean pool; no Motion-net stages/MotionBlocks/"
                            "point downsampling (published: 4-stage, hidden 256)",
                            "grouping k=4 of <=24 pts/bin (published topk=16 of 128/frame); "
                            "direct grouping only (offsets=False)",
                            "3-D xyz offsets (published 4-D x,y,z,d positions)",
                            "point features [x,y,z,v,A] common columns (published xyz(d) only)",
                            "masked kNN + pair-validity has-guard pooling; empty bins carry "
                            "state from last non-empty bin (published dense 32x128, no padding)",
                            "whole-instance 384-pt common input not run: structurally "
                            "inapplicable to a per-frame recurrent point model",
                            "published-default lr 1e-4 arm not yet run"]},
           "results": results}
    json.dump(out, open(os.path.join(DOCS, "baselines2.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, "baselines2_preds.npz"), **preds)
    print(f"\nwrote docs/baselines2.json + baselines2_preds.npz in {time.time()-t0:.0f}s", flush=True)
