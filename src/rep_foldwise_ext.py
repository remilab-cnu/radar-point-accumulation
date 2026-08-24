"""E1x: uniform budget extension (ep160/200) under the FOLD-WISE leak-free preprocessing,
so the budget-sensitivity check and the primary ladder share one pipeline.
Arms: the four the bound argument needs. 5 seeds, mHomeGes only.
Out: docs/foldwise_ext.json + preds. SMOKE=1 -> tiny plumbing check."""
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
    print(f"FOLDWISE-EXT SMOKE={SMOKE} seeds={base.SEEDS} arms={list(base.ARMS)}", flush=True)
    results, preds = {}, {}
    mh = mhomeges_instances(os.path.join(base.DATA, "mhomeges_full"))
    subj = np.array([t[2] for t in mh])
    if SMOKE:
        keep = sorted(set(subj))[:3]
        mh = [t for t in mh if t[2] in keep]; subj = np.array([t[2] for t in mh])
    folds = kfold(subj, 2 if SMOKE else 5)
    for ep in ((2,) if SMOKE else (160, 200)):
        base.run("mHomeGes", mh, folds, 10, ep, results, preds)
    out = {"purpose": "E1x: ep160/200 budget extension under fold-wise ranges (2026-08-19)",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(base.SEEDS), "uf_gate": 0.95,
                        "model": "DeepSets", "smoke": SMOKE,
                        "ranges": "fit_ranges on training subjects per fold"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(base.DOCS, f"foldwise_ext{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(base.DOCS, f"foldwise_ext{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/foldwise_ext{sfx}.json in {time.time()-t0:.0f}s", flush=True)
