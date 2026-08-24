"""P1 — Cross-paradigm quantity factorial (RUN FIRST, repositioning memo §4).

Pre-registered, gesture-scoped hypothesis: the quantity ordering velocity > occupancy >
intensity observed in MAP arms also holds when quantities enter as PER-POINT FEATURES
in point networks: {xyz(,t)} =~ occupancy, {xyz+A} =~ intensity, {xyz+v} =~ velocity,
{xyz+v+A} = full. MM-Fi cells are exploratory bonus.
Audit protocol: frozen manifests, single kfold, ep30 AND ep40, batch 64, seeds {0,1,2},
no aug, per-instance predictions saved.
Models: DeepSets (primary; t coordinate included), FramePointGRU (secondary; time via bins).
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_variants import infineon_recs
from rep_round3 import kfold
from pointset_models import (DeepSets, FramePointGRU, train_eval_set_preds,
                             build_point_tensors, build_frame_tensors)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40)

# feature columns in build_point_tensors: [x,y,z,v,a,t]; frame tensors: [x,y,z,v,a]
SETS_DS = {"occupancy(xyzt)": [0, 1, 2, 5], "intensity(xyzAt)": [0, 1, 2, 4, 5],
           "velocity(xyzvt)": [0, 1, 2, 3, 5], "full(xyzvAt)": [0, 1, 2, 3, 4, 5]}
SETS_GRU = {"occupancy(xyz)": [0, 1, 2], "intensity(xyzA)": [0, 1, 2, 4],
            "velocity(xyzv)": [0, 1, 2, 3], "full(xyzvA)": [0, 1, 2, 3, 4]}

results = {}; preds = {}

def run_ds(tag, insts, folds, ncls):
    ranges = fit_ranges([t[0] for t in insts])
    Xp, Mp, y, s = build_point_tensors(insts, ranges)
    Xf, Mf, yf, sf = build_frame_tensors(insts, ranges)
    for ep in EPOCH_BUDGETS:
        for fam, model, X, M, yy, ss, sets in (("DeepSets", DeepSets, Xp, Mp, y, s, SETS_DS),
                                               ("GRU", FramePointGRU, Xf, Mf, yf, sf, SETS_GRU)):
            for fname, cols in sets.items():
                accs = []
                for fi, te_s in enumerate(folds):
                    te = np.isin(ss, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                    Xc = np.ascontiguousarray(X[..., cols])
                    for sd in SEEDS:
                        a, yt, yp = train_eval_set_preds(model, Xc[tr], M[tr], yy[tr],
                                                         Xc[te], M[te], yy[te], ncls, len(cols),
                                                         epochs=ep, seed=sd)
                        accs.append(a)
                        preds[f"{tag}|{fam}|{fname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
                key = f"{tag}|{fam}|{fname}|ep{ep}"
                results[key] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100,
                                [round(float(a) * 100, 2) for a in accs])
                print(f"  {key:48s}: {results[key][0]:6.2f}% (+-{results[key][1]:.1f})", flush=True)

t0 = time.time()
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
print(f"mHomeGes {len(mh)} inst", flush=True)
run_ds("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10)

inf = infineon_recs()
print(f"Infineon {len(inf)} inst", flush=True)
run_ds("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5)

mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
print(f"MM-Fi {len(mf)} inst (exploratory)", flush=True)
run_ds("MM-Fi", mf, [S2], 27)

out = {"protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "seeds": list(SEEDS), "aug": "none"},
       "manifest": {"infineon_pkl_md5": hashlib.md5(open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest()},
       "results": results}
json.dump(out, open(os.path.join(DOCS, "p1_crossparadigm.json"), "w"), indent=1)
np.savez_compressed(os.path.join(DOCS, "p1_crossparadigm_preds.npz"), **preds)
print(f"\nwrote docs/p1_crossparadigm.json (+preds) in {time.time()-t0:.0f}s total", flush=True)
