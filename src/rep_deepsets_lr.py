"""N6 — DeepSets learning-rate sensitivity (reviewer round, 2026-07-20).

Fills the S1 gap flagged in review: DeepSets (the primary point-domain attribution model)
was only ever trained at lr=1e-3, so the frozen-rate robustness check in SUPPLEMENT S1
covered the published baselines but not the central model. This runs DeepSets at lr=3e-4
for all four point-feature arms, ep{30,40}, frozen protocol (batch 64, seeds 0-2, no aug),
per-instance preds saved. The lr=1e-3 counterparts already exist in p1_crossparadigm_preds.npz
(same loader/splits/instance ids -> pairs directly), so we only run the 3e-4 arm here.

Outputs: docs/deepsets_lr.json + docs/deepsets_lr_preds.npz
  key = "{tag}|DeepSets|{arm}|lr3e-4|ep{E}|f{F}|s{S}" -> stack([te_idx, y_true, y_pred])

Lets us check (a) DeepSets full-model accuracy 1e-3 vs 3e-4 (the S1 row) and
(b) whether the velocity contribution (full - intensity) is lr-robust.
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_variants import infineon_recs
from rep_round3 import kfold
from pointset_models import DeepSets, train_eval_set_preds, build_point_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40); LRS = (("3e-4", 3e-4),)
SMOKE = os.environ.get("SMOKE", "") == "1"
if SMOKE:
    SEEDS = (0,); EPOCH_BUDGETS = (1,)

# feature columns in build_point_tensors: [x,y,z,v,a,t]
SETS_DS = {"occupancy(xyzt)": [0, 1, 2, 5], "intensity(xyzAt)": [0, 1, 2, 4, 5],
           "velocity(xyzvt)": [0, 1, 2, 3, 5], "full(xyzvAt)": [0, 1, 2, 3, 4, 5]}

results = {}; preds = {}


def run_ds(tag, insts, folds, ncls):
    ranges = fit_ranges([t[0] for t in insts])
    Xp, Mp, y, s = build_point_tensors(insts, ranges)
    print(f"{tag}: {len(y)} inst", flush=True)
    for lrname, lr in LRS:
        for ep in EPOCH_BUDGETS:
            for fname, cols in SETS_DS.items():
                accs = []
                for fi, te_s in enumerate(folds):
                    te = np.isin(s, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                    Xc = np.ascontiguousarray(Xp[..., cols])
                    for sd in SEEDS:
                        a, yt, yp = train_eval_set_preds(DeepSets, Xc[tr], Mp[tr], y[tr],
                                                         Xc[te], Mp[te], y[te], ncls, len(cols),
                                                         epochs=ep, seed=sd, lr=lr)
                        accs.append(a)
                        preds[f"{tag}|DeepSets|{fname}|lr{lrname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
                key = f"{tag}|DeepSets|{fname}|lr{lrname}|ep{ep}"
                results[key] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100,
                                [round(float(a) * 100, 2) for a in accs])
                print(f"  {key:52s}: {results[key][0]:6.2f}% (+-{results[key][1]:.1f})", flush=True)


t0 = time.time()
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
run_ds("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10)

inf = infineon_recs()
run_ds("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5)

mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
run_ds("MM-Fi", mf, [S2], 27)

suffix = "_smoke" if SMOKE else ""
out = {"protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "seeds": list(SEEDS),
                    "lrs": [l for l, _ in LRS], "aug": "none", "model": "DeepSets",
                    "note": "3e-4 arm; 1e-3 counterparts in p1_crossparadigm_preds.npz"},
       "manifest": {"infineon_pkl_md5": hashlib.md5(open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest()},
       "results": results}
json.dump(out, open(os.path.join(DOCS, f"deepsets_lr{suffix}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(DOCS, f"deepsets_lr{suffix}_preds.npz"), **preds)
print(f"\nwrote docs/deepsets_lr{suffix}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
