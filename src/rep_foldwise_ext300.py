"""E8 (cold review r12): push the mHomeGes ladder to ep300 so the cross-instance
shuffle arm can clear the 0.95 criterion, closing the last underfit objection on
the headline contrast. Same four arms and machinery as rep_foldwise_ext."""
import os, json, time
import numpy as np
import rep_foldwise_ladder as base
from spectra_dataset import mhomeges_instances
from rep_round3 import kfold

base.ARMS = {k: v for k, v in base.ARMS.items() if k in
             ("full(xyzvAt)", "v-inshuffle", "v-crossshuffle", "no-velocity(xyzAt)")}

if __name__ == "__main__":
    t0 = time.time()
    SMOKE = base.SMOKE
    print(f"FOLDWISE-EXT300 SMOKE={SMOKE} seeds={base.SEEDS}", flush=True)
    results, preds = {}, {}
    mh = mhomeges_instances(os.path.join(base.DATA, "mhomeges_full"))
    subj = np.array([t[2] for t in mh])
    if SMOKE:
        keep = sorted(set(subj))[:3]
        mh = [t for t in mh if t[2] in keep]; subj = np.array([t[2] for t in mh])
    folds = kfold(subj, 2 if SMOKE else 5)
    base.run("mHomeGes", mh, folds, 10, 2 if SMOKE else 300, results, preds)
    out = {"purpose": "E8: ep300 fold-wise extension (cold review r12)",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(base.SEEDS),
                        "uf_gate": 0.95, "model": "DeepSets", "smoke": SMOKE,
                        "ranges": "fit_ranges on training subjects per fold"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(base.DOCS, f"foldwise_ext300{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(base.DOCS, f"foldwise_ext300{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/foldwise_ext300{sfx}.json in {time.time()-t0:.0f}s", flush=True)
