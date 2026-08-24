"""mRI (2nd whole-body dataset) — pre-registered replication of the F-C interaction.

mRI (NeurIPS D&B 2022, CC0): 20 subjects, 10 rehab exercises (pose_1..pose_10) with
per-subject video-frame segments in *_all_labels.cpl['video_label']; aligned radar
CSV per subject (per-point x,y,z,Doppler,Intensity + Camera Frame alignment).

PRE-REGISTERED PREDICTIONS (from F-B/F-C, stated before any result):
  P-map  : in MAP form, geometry/intensity >= velocity (the MM-Fi reversal replicates)
  P-point: in POINT networks, velocity remains the best single quantity
Protocol: frozen (kfold seed-0 over 20 subjects, k=5; ep30 AND ep40; batch 64;
seeds 0-2; no aug; per-instance predictions saved).
Instances: sliding windows (40 camera frames, stride 20) inside each exercise segment.
"""
import os, json, glob, pickle, time
import numpy as np
import pandas as pd
from rep_variants import cell_stats, norm, CAXES
from spectra_dataset import fit_ranges
from rep_round3 import kfold
from cnn import train_eval_preds
from pointset_models import (DeepSets, FramePointGRU, train_eval_set_preds,
                             build_point_tensors, build_frame_tensors)

HERE = os.path.dirname(os.path.abspath(__file__))
MRI = os.path.join(HERE, "..", "data", "mri_sample", "mri_data")
DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPS = (30, 40)
CLASSES = [f"pose_{i}" for i in range(1, 11)]
WIN, STRIDE = 40, 20

print("PRE-REGISTERED: P-map = geometry/intensity >= velocity in maps (reversal replicates); "
      "P-point = velocity best in point networks.", flush=True)

recs = []
for csvf in sorted(glob.glob(os.path.join(MRI, "subject*.csv"))):
    sid = os.path.basename(csvf).replace(".csv", "")
    if "_all_labels" in sid:
        continue
    cpl = os.path.join(MRI, f"{sid}_all_labels.cpl")
    if not os.path.exists(cpl):
        continue
    df = pd.read_csv(csvf); df.columns = [c.strip() for c in df.columns]
    can = pd.DataFrame({"frame": df["Camera Frame"].astype(int),
                        "x": df["X"], "y": df["Y"], "z": df["Z"],
                        "doppler": df["Doppler"], "intensity": df["Intensity"]})
    vl = pickle.load(open(cpl, "rb"))["video_label"]
    for ci, cname in enumerate(CLASSES):
        if cname not in vl:
            continue
        a, b = vl[cname]
        seg = can[(can.frame >= a) & (can.frame <= b)]
        t0 = a
        while t0 + WIN <= b:
            w = seg[(seg.frame >= t0) & (seg.frame < t0 + WIN)]
            if w["frame"].nunique() >= 6 and len(w) >= 30:
                recs.append((w.reset_index(drop=True), ci, sid))
            t0 += STRIDE
print(f"mRI instances: {len(recs)} | subjects {len(set(t[2] for t in recs))} | "
      f"per-class {np.bincount([t[1] for t in recs])}", flush=True)

ranges = fit_ranges([t[0] for t in recs])
y = np.array([t[1] for t in recs]); subj = np.array([t[2] for t in recs])
folds = kfold(subj, 5)

t0 = time.time()
stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in recs]
ARMS = {
    "map_v_sum":     np.stack([np.stack([norm(st[ax]["sum"]) for ax in CAXES]) for st in stats]).astype(np.float32),
    "map_occupancy": np.stack([np.stack([norm(st[ax]["cnt"]) for ax in CAXES]) for st in stats]).astype(np.float32),
    "map_int_sum":   np.stack([np.stack([norm(st[ax]["int_mean"] * st[ax]["cnt"]) for ax in CAXES]) for st in stats]).astype(np.float32),
}
del stats
print(f"map arms built in {time.time()-t0:.0f}s", flush=True)
Xp, Mp, yp, sp = build_point_tensors(recs, ranges)
Xf, Mf, yf, sf = build_frame_tensors(recs, ranges)
SETS_DS = {"occupancy(xyzt)": [0, 1, 2, 5], "intensity(xyzAt)": [0, 1, 2, 4, 5],
           "velocity(xyzvt)": [0, 1, 2, 3, 5], "full(xyzvAt)": [0, 1, 2, 3, 4, 5]}
SETS_GRU = {"occupancy(xyz)": [0, 1, 2], "intensity(xyzA)": [0, 1, 2, 4],
            "velocity(xyzv)": [0, 1, 2, 3], "full(xyzvA)": [0, 1, 2, 3, 4]}

results = {}; preds = {}
def rec_p(key, fi, sd, te_idx, yt, ypd):
    preds[f"{key}|f{fi}|s{sd}"] = np.stack([te_idx, yt, ypd])

for ep in EPS:
    for arm, X in ARMS.items():
        accs = []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            for sd in SEEDS:
                a, yt, ypd = train_eval_preds(X[tr], y[tr], X[te], y[te], 10, epochs=ep, seed=sd)
                accs.append(a); rec_p(f"{arm}|ep{ep}", fi, sd, te_idx, yt, ypd)
        results[f"{arm}|ep{ep}"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
        print(f"  {arm:16s} ep{ep}: {results[f'{arm}|ep{ep}'][0]:6.2f}% (+-{results[f'{arm}|ep{ep}'][1]:.1f})", flush=True)
    for fam, model, XX, MM, yy, ss, sets in (("DeepSets", DeepSets, Xp, Mp, yp, sp, SETS_DS),
                                             ("GRU", FramePointGRU, Xf, Mf, yf, sf, SETS_GRU)):
        for fname, cols in sets.items():
            accs = []
            Xc = np.ascontiguousarray(XX[..., cols])
            for fi, te_s in enumerate(folds):
                te = np.isin(ss, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                for sd in SEEDS:
                    a, yt, ypd = train_eval_set_preds(model, Xc[tr], MM[tr], yy[tr], Xc[te], MM[te], yy[te],
                                                      10, len(cols), epochs=ep, seed=sd)
                    accs.append(a); rec_p(f"{fam}|{fname}|ep{ep}", fi, sd, te_idx, yt, ypd)
            results[f"{fam}|{fname}|ep{ep}"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
            print(f"  {fam}|{fname:18s} ep{ep}: {results[f'{fam}|{fname}|ep{ep}'][0]:6.2f}% "
                  f"(+-{results[f'{fam}|{fname}|ep{ep}'][1]:.1f})", flush=True)

out = {"protocol": {"epochs": list(EPS), "batch": 64, "seeds": list(SEEDS), "aug": "none",
                    "window": WIN, "stride": STRIDE, "classes": CLASSES},
       "folds": [list(map(str, f)) for f in folds],
       "prereg": {"P-map": "geometry/intensity >= velocity", "P-point": "velocity best"},
       "results": results}
json.dump(out, open(os.path.join(DOCS, "mri_replication.json"), "w"), indent=1)
np.savez_compressed(os.path.join(DOCS, "mri_replication_preds.npz"), **preds)
print("\nwrote docs/mri_replication.json (+preds)", flush=True)
