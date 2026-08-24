"""Baselines round 1 — published point-cloud baselines under the frozen equal-HPO
protocol (docs/EQUAL_HPO_PROTOCOL.md; repositioning memo queue #1 PointNet++, #2 DGCNN).

Arms: {PointNetPP, DGCNNTemporal} x common-input full(x,y,z,v,A,t) x ep{30,40} x
seeds {0,1,2}, batch 64, Adam, grad-clip 5.0, no augmentation.
Datasets: mHomeGes 5-fold, Infineon frozen pkl 4-fold, MM-Fi documented S2 split.
Outputs: docs/baselines1.json (+ per-instance preds docs/baselines1_preds.npz).

LR policy: published default only (both papers' default Adam LR = 1e-3).

SUPERSEDED (2026-07-12): this file is obsolete — run_baselines1_fix2.py replaces it
(post BatchNorm track_running_stats fix; produces the results actually used,
docs/baselines1_fix2.json) and sweeps LR {1e-3, 3e-4}. The earlier validation-subject
3-point LR-selection plan was RESOLVED by EQUAL_HPO amendment A1 (Table-III re-audit):
the main table freezes LR = 1e-3 and demotes the {3e-4} arm to a supplementary
LR-sensitivity note, so no per-arm validation LR selection is added here. Kept only for
provenance of the initial (superseded) baselines1.json run.
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_variants import infineon_recs
from rep_round3 import kfold
from pointset_models import build_point_tensors
from baselines_pointnets import PointNetPP, DGCNNTemporal, train_eval_set_preds_tr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40); LR = 1e-3
MODELS = (("PointNetPP", PointNetPP), ("DGCNNTemporal", DGCNNTemporal))

results, preds, datasets = {}, {}, {}


def run_set(tag, insts, folds, ncls, lr=LR):
    ranges = fit_ranges([t[0] for t in insts])
    X, M, y, s = build_point_tensors(insts, ranges)
    datasets[tag] = {"n_instances": len(insts), "n_cls": ncls, "chance": round(1 / ncls, 4),
                     "folds": [sorted(map(str, f)) for f in folds],
                     "test_N": [int(np.isin(s, list(f)).sum()) for f in folds]}
    for ep in EPOCH_BUDGETS:
        for mname, mcls in MODELS:
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(s, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                for sd in SEEDS:
                    a, yt, yp, ta = train_eval_set_preds_tr(
                        mcls, X[tr], M[tr], y[tr], X[te], M[te], y[te], ncls, 6,
                        epochs=ep, lr=lr, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{tag}|{mname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
            key = f"{tag}|{mname}|full(xyzvAt)|ep{ep}"
            results[key] = {"mean": float(np.mean(accs)) * 100, "std": float(np.std(accs)) * 100,
                            "accs": [round(float(a) * 100, 2) for a in accs],
                            "min_train_acc": round(float(min(tr_accs)) * 100, 2),
                            "underfit": bool(min(tr_accs) < 0.95)}
            print(f"  {key:44s}: {results[key]['mean']:6.2f}% (+-{results[key]['std']:.1f})"
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
                        "optimizer": "adam", "lr": LR, "grad_clip": 5.0, "aug": "none",
                        "input": "common-input full(x,y,z,v,A,t), n_points=384 masked",
                        "lr_policy": "published-default 1e-3 only; SUPERSEDED by run_baselines1_fix2.py "
                                     "(sweeps {1e-3,3e-4}); main-table policy frozen to 1e-3 per EQUAL_HPO amendment A1",
                        "models": {"PointNetPP": "SSG SA(128,r=0.2,32,[64,64,128])->"
                                                 "SA(32,r=0.4,64,[128,128,256])->global[256,512]->FC",
                                   "DGCNNTemporal": "EdgeConv k=16 [64,64]->[128], static xyz graph, "
                                                    "full-feature edges, masked max+mean pool->FC"},
                        "deviations": ["masked FPS/ball-query/kNN (padding excluded)",
                                       "compact depth: 2 SA levels / 2 EdgeConv blocks",
                                       "DGCNN static xyz graph (no dynamic feature re-kNN)",
                                       "LR selection pass not yet run (default-LR arm only)"]},
           "results": results}
    json.dump(out, open(os.path.join(DOCS, "baselines1.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, "baselines1_preds.npz"), **preds)
    print(f"\nwrote docs/baselines1.json + baselines1_preds.npz in {time.time()-t0:.0f}s", flush=True)
