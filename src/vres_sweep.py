"""H4 direct test: does velocity RESOLUTION drive the Infineon result?
 - mHomeGes: artificially COARSEN per-point doppler (native ~0.356 m/s) to 0.712 /
   1.068 / 1.424 m/s grids -> if velocity-arm accuracy barely drops, resolution is not
   what velocity needs; if it collapses toward the geometry arm, resolution matters.
 - Infineon: coarsen native 0.258 to 0.516 / 1.032 -> slope comparison.
Velocity arm = MEAN accumulation; occupancy reference printed for the gap.
Intermediate result only — no conclusion drawn here.
"""
import os, io, zipfile, re, time
import numpy as np
from preprocess import SpecConfig, build_spectrum, max_norm
from spectra_dataset import fit_ranges, mhomeges_instances
from cnn import train_eval
import infineon_detection as ifx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
AXES = ["x", "y", "z"]


def build_vel_arm(recs, cfg, q):
    X = []
    for inst, lab, s in recs:
        d = inst.copy()
        if q > 0:
            d["doppler"] = np.round(d["doppler"] / q) * q
        X.append(np.stack([max_norm(build_spectrum(d, a, "doppler", cfg, agg="mean")) for a in AXES]).astype(np.float32))
    return np.stack(X)


def build_occ_arm(recs, cfg):
    return np.stack([np.stack([max_norm(build_spectrum(i, a, "count", cfg, agg="sum")) for a in AXES]).astype(np.float32)
                     for i, _, _ in recs])


def xsub(X, y, subj, folds, ncls, epochs, seeds=(0, 1)):
    a = []
    for te_s in folds:
        te = np.isin(subj, list(te_s)); tr = ~te
        a += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs, seed=s) for s in seeds]
    return float(np.mean(a)) * 100


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


# ---------------- mHomeGes sweep ----------------
print("collecting mHomeGes...", flush=True)
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
cfg = SpecConfig(32, 40, fit_ranges([t[0] for t in mh]))
y = np.array([t[1] for t in mh]); subj = np.array([t[2] for t in mh])
folds = kfold(subj, 3)
print(f"mHomeGes {len(y)} inst; occupancy reference:", flush=True)
occ = xsub(build_occ_arm(mh, cfg), y, subj, folds, 10, 15)
print(f"  occupancy: {occ:.2f}%", flush=True)
for q in (0.0, 0.712, 1.068, 1.424):
    acc = xsub(build_vel_arm(mh, cfg, q), y, subj, folds, 10, 15)
    print(f"  velocity q={q:5.3f} m/s: {acc:6.2f}%  (vel-occ {acc-occ:+.2f})", flush=True)

# ---------------- Infineon sweep ----------------
print("\nrebuilding Infineon instances...", flush=True)
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
cfg2 = SpecConfig(32, 40, fit_ranges([t[0] for t in recs]))
y2 = np.array([t[1] for t in recs]); s2 = np.array([t[2] for t in recs])
folds2 = kfold(s2, 4)
print(f"Infineon {len(y2)} inst; occupancy reference:", flush=True)
occ2 = xsub(build_occ_arm(recs, cfg2), y2, s2, folds2, 5, 30)
print(f"  occupancy: {occ2:.2f}%", flush=True)
for q in (0.0, 0.516, 1.032):
    acc = xsub(build_vel_arm(recs, cfg2, q), y2, s2, folds2, 5, 30)
    print(f"  velocity q={q:5.3f} m/s: {acc:6.2f}%  (vel-occ {acc-occ2:+.2f})", flush=True)
