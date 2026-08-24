"""DECISIVE frozen-protocol run (campaign-audit spec) — settles the last open claim:
Infineon maps vs point-input, apples-to-apples.

Audit rules implemented:
- frozen manifest: data/infineon_recs.pkl (md5 + counts recorded in output JSON)
- single kfold (seed-0 permutation over sorted users); fold membership in JSON
- pre-registered arms (no test-fold selection):
    map_v_sum          3ch  SUM velocity (campaign-standard arm)
    map_composite      10ch signed velocity + V-T + occupancy ("velocity+occupancy composite")
    REF_occupancy      3ch  per-bin count
    REF_int_sum        3ch  SUM intensity (standard arm)
    REF_int_mean       3ch  MEAN intensity (tagged mean-arm)
    DeepSets_full      point-input bar (x,y,z,v,A,t)
    FramePointGRU      secondary point bar (post-fix)
- matched budgets: ALL arms at epochs 30 AND 40, batch 64, width 16, seeds {0,1,2}
- no augmentation
- per-instance predictions saved (docs/final_infineon_preds.npz) -> paired tests computable
"""
import os, json, time, hashlib
import numpy as np
from rep_variants import cell_stats, norm, CAXES, infineon_recs
from spectra_dataset import fit_ranges
from rep_round3 import kfold
from cnn import train_eval_preds
from pointset_models import (DeepSets, FramePointGRU, train_eval_set_preds,
                             build_point_tensors, build_frame_tensors)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
PKL = os.path.join(DATA, "infineon_recs.pkl")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40)

recs = infineon_recs()
manifest = {"pkl_md5": hashlib.md5(open(PKL, "rb").read()).hexdigest(),
            "n_instances": len(recs),
            "class_counts": {int(c): int(n) for c, n in
                             zip(*np.unique([t[1] for t in recs], return_counts=True))}}
print("manifest:", manifest, flush=True)

ranges = fit_ranges([t[0] for t in recs])
y = np.array([t[1] for t in recs]); subj = np.array([t[2] for t in recs])
folds = kfold(subj, 4)
print("folds:", folds, flush=True)

# ---- build map arms once ----
t0 = time.time()
stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in recs]
def stack(spec_fn):
    return np.stack([spec_fn(st) for st in stats]).astype(np.float32)
ARMS_MAP = {
    "map_v_sum":     stack(lambda st: np.stack([norm(st[ax]["sum"]) for ax in CAXES])),
    "map_composite": stack(lambda st: np.stack([norm(st[ax]["pos_mean"]) for ax in CAXES] +
                                               [norm(st[ax]["neg_mean"]) for ax in CAXES] +
                                               [norm(st["vt"])] +
                                               [norm(st[ax]["cnt"]) for ax in CAXES])),
    "REF_occupancy": stack(lambda st: np.stack([norm(st[ax]["cnt"]) for ax in CAXES])),
    "REF_int_sum":   stack(lambda st: np.stack([norm(st[ax]["int_mean"] * st[ax]["cnt"]) for ax in CAXES])),
    "REF_int_mean":  stack(lambda st: np.stack([norm(st[ax]["int_mean"]) for ax in CAXES])),
}
del stats
print(f"map arms built in {time.time()-t0:.0f}s:", {k: v.shape for k, v in ARMS_MAP.items()}, flush=True)
Xp, Mp, yp, sp = build_point_tensors(recs, ranges)
Xf, Mf, yf, sf = build_frame_tensors(recs, ranges)

results = {}; preds_store = {}
def record(arm, ep, fold_i, seed, yt, ypred, te_idx):
    preds_store[f"{arm}|ep{ep}|fold{fold_i}|seed{seed}"] = np.stack([te_idx, yt, ypred])

for ep in EPOCH_BUDGETS:
    for arm, X in ARMS_MAP.items():
        accs = []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            for s in SEEDS:
                a, yt, ypred = train_eval_preds(X[tr], y[tr], X[te], y[te], 5, epochs=ep, seed=s)
                accs.append(a); record(arm, ep, fi, s, yt, ypred, te_idx)
        results[f"{arm}|ep{ep}"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100,
                                    [round(float(a) * 100, 2) for a in accs])
        print(f"  {arm:16s} ep{ep}: {results[f'{arm}|ep{ep}'][0]:6.2f}% (+-{results[f'{arm}|ep{ep}'][1]:.1f})", flush=True)
    for arm, model, XX, MM, yy, ss_, ind in (("DeepSets_full", DeepSets, Xp, Mp, yp, sp, 6),
                                             ("FramePointGRU", FramePointGRU, Xf, Mf, yf, sf, 5)):
        accs = []
        for fi, te_s in enumerate(folds):
            te = np.isin(ss_, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            for s in SEEDS:
                a, yt, ypred = train_eval_set_preds(model, XX[tr], MM[tr], yy[tr], XX[te], MM[te], yy[te],
                                                    5, ind, epochs=ep, seed=s)
                accs.append(a); record(arm, ep, fi, s, yt, ypred, te_idx)
        results[f"{arm}|ep{ep}"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100,
                                    [round(float(a) * 100, 2) for a in accs])
        print(f"  {arm:16s} ep{ep}: {results[f'{arm}|ep{ep}'][0]:6.2f}% (+-{results[f'{arm}|ep{ep}'][1]:.1f})", flush=True)

out = {"manifest": manifest, "folds": [list(map(str, f)) for f in folds],
       "test_N": [int(np.isin(subj, list(f)).sum()) for f in folds],
       "protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "width": 16,
                    "seeds": list(SEEDS), "aug": "none", "arm_standard": "sum"},
       "results": results}
json.dump(out, open(os.path.join(DOCS, "final_infineon.json"), "w"), indent=1)
np.savez_compressed(os.path.join(DOCS, "final_infineon_preds.npz"), **preds_store)
print("\nwrote docs/final_infineon.json + final_infineon_preds.npz", flush=True)
