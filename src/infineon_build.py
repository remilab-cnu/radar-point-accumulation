"""CROSS-HARDWARE gate: build labeled point clouds from the Infineon BGT60TR13C 60 GHz
raw cubes via our detection chain, then run the 3-arm (velocity/geometry/intensity)
cross-subject comparison. Different vendor/band from the TI IWR1443 gesture data —
if velocity still beats geometry here, the result replicates across hardware.

Infineon npz: inputs (R,100,3,32,64) float32, targets (R,100) int64.
5 gesture classes are labeled {1,2,3,6,7}; 0 = background. ~8 gesture frames/recording.
Uses MEAN accumulation (audit M1 fix: decoupled from point density).
"""
import os, io, zipfile, re, time
import numpy as np
from preprocess import SpecConfig, build_spectrum, max_norm
from spectra_dataset import fit_ranges
from cnn import train_eval
import infineon_detection as ifx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
CACHE = os.path.join(DATA, "infineon_gate.npz")
AXES = ["x", "y", "z"]
LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}     # 5 gesture classes -> 0..4
MARGIN = 6
PER_USER_CAP = 300                             # subsample recordings/user for tractable first run
ARMS = {"velocity": ("doppler", "mean"), "intensity": ("intensity", "mean"), "occupancy": ("count", "sum")}


def gesture_window(cube_rec, tgt_row):
    g = np.where(tgt_row > 0)[0]
    if len(g) < 2:
        return None, None
    nz = tgt_row[tgt_row > 0]
    cls = int(np.bincount(nz).argmax())
    lo, hi = max(0, g.min() - MARGIN), min(len(tgt_row), g.max() + MARGIN + 1)
    return cube_rec[lo:hi], cls


def build_instances():
    zf = zipfile.ZipFile(ZIP)
    members = [m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m) and "_fast" not in m and "_slow" not in m and "_wrist" not in m]
    members = sorted(members, key=lambda m: int(re.search(r"user(\d+)_e1", m).group(1)))
    recs = []
    for m in members:
        user = "u" + re.search(r"user(\d+)_e1", m).group(1)
        t0 = time.time()
        with zf.open(m) as f:
            d = np.load(io.BytesIO(f.read()))
            inputs, targets = d["inputs"], d["targets"]
        # balanced subsample per user
        by_cls = {}
        order = np.random.RandomState(0).permutation(len(inputs))
        for r in order:
            cube, cls = gesture_window(inputs[r], targets[r])
            if cube is None or cls not in LABELMAP:
                continue
            if by_cls.get(cls, 0) >= PER_USER_CAP // len(LABELMAP):
                continue
            df = ifx.process_recording(cube)
            if len(df) < 8:
                continue
            recs.append((df, LABELMAP[cls], user)); by_cls[cls] = by_cls.get(cls, 0) + 1
        print(f"  {user}: {sum(by_cls.values())} instances in {time.time()-t0:.0f}s", flush=True)
    return recs


def build_arms(recs):
    insts = [t[0] for t in recs]
    cfg = SpecConfig(32, 40, fit_ranges(insts))
    out = {k: [] for k in ARMS}; y = []; subj = []
    for inst, lab, s in recs:
        for k, (val, agg) in ARMS.items():
            out[k].append(np.stack([max_norm(build_spectrum(inst, a, val, cfg, agg=agg)) for a in AXES]).astype(np.float32))
        y.append(lab); subj.append(s)
    return {k: np.stack(v) for k, v in out.items()}, np.array(y, np.int64), np.array(subj)


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    arms = {"velocity": z["velocity"], "intensity": z["intensity"], "occupancy": z["occupancy"]}; y = z["y"]; subj = z["subj"]
else:
    t0 = time.time(); print("building Infineon instances (detection chain)...", flush=True)
    recs = build_instances()
    print(f"total {len(recs)} instances in {time.time()-t0:.0f}s", flush=True)
    arms, y, subj = build_arms(recs)
    np.savez_compressed(CACHE, y=y, subj=subj, **arms)

users = sorted(set(subj.tolist()))
print(f"\nInfineon: {len(y)} instances, {len(users)} users {users}, 5 classes")
import collections
print("class counts:", dict(sorted(collections.Counter(y.tolist()).items())))

# cross-subject: 4-fold over users
rng = np.random.RandomState(0)
folds = [list(g) for g in np.array_split(rng.permutation(users), 4)]
print(f"\n#### CROSS-HARDWARE gate (Infineon 60GHz, user 4-fold, mean accumulation) ####")
acc = {}
for k, X in arms.items():
    a = []
    for te_s in folds:
        te = np.isin(subj, list(te_s)); tr = ~te
        a += [train_eval(X[tr], y[tr], X[te], y[te], 5, epochs=30, seed=s) for s in (0, 1, 2)]
    acc[k] = float(np.mean(a)); print(f"   {k:10s}: {acc[k]*100:6.2f}%", flush=True)
print(f"   >> velocity - occupancy(geometry) = {(acc['velocity']-acc['occupancy'])*100:+.2f}   (chance=20%)")
print(f"   >> velocity - intensity           = {(acc['velocity']-acc['intensity'])*100:+.2f}")
