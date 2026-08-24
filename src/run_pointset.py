"""RawSet control (blueprint) — point-input baselines on the open datasets.

Questions:
 Q1 information cost of the 2D-map compression (point-input vs map CNN accuracy)
 Q2 is the map-domain ranking (velocity wins gestures / reversal on HAR) a
    rasterization artifact? (check ranking persists with raw points)
 Q3 point-feature ablation = point-domain analog of what-to-accumulate:
    full(x,y,z,v,A,t) vs no-velocity vs xyz-t only.

Models: DeepSets (whole-window set) and FramePointGRU (temporal point net).
Protocols identical to the map gate (same subject folds / S2 split).
"""
import os, io, json, time, zipfile, re, pickle
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from pointset_models import (build_point_tensors, build_frame_tensors,
                             DeepSets, FramePointGRU, train_eval_set)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2)

# map-CNN reference numbers (same folds) for the Q1 comparison, printed alongside
MAP_REF = {"mHomeGes": {"velocity": 67.49, "occupancy": 54.38, "intensity": 44.60},
           "Infineon": {"velocity": 86.35, "occupancy": 84.30, "intensity": 89.81},
           "MM-Fi": {"velocity": 70.37, "occupancy": 77.47, "intensity": 77.01}}

DS_VARIANTS = {"full(xyzvAt)": [0, 1, 2, 3, 4, 5],
               "no-velocity": [0, 1, 2, 4, 5],
               "xyz-t only": [0, 1, 2, 5]}
GRU_VARIANTS = {"full(xyzvA)": [0, 1, 2, 3, 4],
                "no-velocity": [0, 1, 2, 4]}


def infineon_recs():
    cache = os.path.join(DATA, "infineon_recs.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import infineon_detection as ifx
    ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
    LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    recs = []
    for m in members:
        user = "u" + re.search(r"user(\d+)", m).group(1)
        with zf.open(m) as f:
            d = np.load(io.BytesIO(f.read())); inputs, targets = d["inputs"], d["targets"]
        by = {}
        for r in np.random.RandomState(0).permutation(len(inputs)):
            g = np.where(targets[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
            if cls not in LABELMAP or by.get(cls, 0) >= 40:
                continue
            df = ifx.process_recording(inputs[r, max(0, g.min() - 6):g.max() + 7])
            if len(df) < 8:
                continue
            recs.append((df, LABELMAP[cls], user)); by[cls] = by.get(cls, 0) + 1
    pickle.dump(recs, open(cache, "wb"))
    return recs


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


def eval_model(model_cls, X, M, y, subj, folds, ncls, in_dim, epochs, seeds=SEEDS):
    accs = []
    for te_s in folds:
        te = np.isin(subj, list(te_s)); tr = ~te
        accs += [train_eval_set(model_cls, X[tr], M[tr], y[tr], X[te], M[te], y[te],
                                ncls, in_dim, epochs=epochs, seed=s) for s in seeds]
    return float(np.mean(accs)) * 100, float(np.std(accs)) * 100


results = {}


def run_dataset(name, insts, folds, ncls, epochs, do_gru=True):
    t0 = time.time()
    ranges = fit_ranges([t[0] for t in insts])
    Xp, Mp, y, subj = build_point_tensors(insts, ranges)
    print(f"\n######## {name}: {len(y)} inst, {len(set(subj.tolist()))} subj, {ncls} cls "
          f"(tensors {time.time()-t0:.0f}s) ########", flush=True)
    print(f"  [map-CNN reference: {MAP_REF.get(name, {})}]", flush=True)
    res = {}
    for vname, cols in DS_VARIANTS.items():
        m, s = eval_model(DeepSets, Xp[..., cols].copy(), Mp, y, subj, folds, ncls, len(cols), epochs)
        res[f"DeepSets {vname}"] = m
        print(f"  DeepSets {vname:14s}: {m:6.2f}% (+-{s:.1f})", flush=True)
    if do_gru:
        Xf, Mf, y2, s2 = build_frame_tensors(insts, ranges)
        for vname, cols in GRU_VARIANTS.items():
            m, s = eval_model(FramePointGRU, Xf[..., cols].copy(), Mf, y2, s2, folds, ncls, len(cols), epochs)
            res[f"FramePointGRU {vname}"] = m
            print(f"  FramePointGRU {vname:12s}: {m:6.2f}% (+-{s:.1f})", flush=True)
    results[name] = res


mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
run_dataset("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10, epochs=30)

inf = infineon_recs()
run_dataset("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5, epochs=30)

mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
run_dataset("MM-Fi", mf, [S2], 27, epochs=40)

os.makedirs(DOCS, exist_ok=True)
json.dump({"map_ref": MAP_REF, "pointset": results}, open(os.path.join(DOCS, "pointset_results.json"), "w"), indent=1)
print("\nwrote docs/pointset_results.json", flush=True)
