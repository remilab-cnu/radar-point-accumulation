"""Uniform budget extension for the point-domain ladder (PI directive, 2026-08-17).

The ep120 primary estimate rests on a gated reference arm (0.958) with ablated
arms at 0.918-0.948. Reviewers can ask whether the residual fit gap drives the
8.3 pp contrast. Per-arm gating would select on the treatment, so the honest
probe is a UNIFORM extension: every arm, same budgets, ep160 and ep200 on
mHomeGes, frozen protocol otherwise (lr 1e-3, batch 64, seeds {0,1,2},
kfold(seed=0) == p1/converge folds). Two readouts:
  1) do the ablated arms approach/clear the 0.95 gate as budget grows;
  2) does the dimensionality-matched contrast stay near 8.3 pp
     (ep80 8.47 -> ep120 8.27 -> ?).
MM-Fi is not rerun: the primary claim rests on mHomeGes, and MM-Fi ep240
already sits at 0.911-0.970.

Out: docs/converge_point_ext.json + docs/converge_point_ext_preds.npz
(the frozen converge_point.json artifacts are not touched).
"""
import os, json, time
import numpy as np
from rep_converge_point import run, SMOKE, SEEDS, UF_GATE
from spectra_dataset import mhomeges_instances
from rep_round3 import kfold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")

if __name__ == "__main__":
    t0 = time.time()
    budgets = (3,) if SMOKE else (160, 200)
    print(f"CONVERGE-POINT-EXT  SMOKE={SMOKE}  seeds={SEEDS}  budgets={budgets}", flush=True)
    results, preds = {}, {}

    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    subj_mh = np.array([t[2] for t in mh])
    if SMOKE:
        keep = sorted(set(subj_mh))[:3]
        mh = [t for t in mh if t[2] in keep]; subj_mh = np.array([t[2] for t in mh])
    folds_mh = kfold(subj_mh, 2 if SMOKE else 5)
    run("mHomeGes", mh, folds_mh, 10, budgets, results, preds)

    out = {"purpose": "uniform budget extension ep160/ep200 (PI directive 2026-08-17); "
                      "same arms/folds/protocol as converge_point.json",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(SEEDS), "uf_gate": UF_GATE,
                        "aug": "none", "model": "DeepSets", "smoke": SMOKE,
                        "budgets": {"mHomeGes": list(budgets)},
                        "folds": "kfold(seed=0) == p1_crossparadigm/converge_point"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"converge_point_ext{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"converge_point_ext{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/converge_point_ext{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
