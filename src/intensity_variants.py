"""Range-compensation REVIEW experiment: was int_rc = |RD|*r^2 a fair 'clean intensity'?
Compare intensity variants under identical protocol (Infineon, MEAN accumulation):
  int_mag    raw |RD| magnitude            (range-leaky? strong in prior runs)
  int_logmag log1p(|RD|)                   (dB-like compression, TI-ish scale)
  int_r2     |RD| * r^2                    (Opus's compensation — noise-amplifying?)
  int_snr    |RD| / local CFAR noise floor (semantically closest to TI 'SNR')
Plus velocity & occupancy references. EASY (4-fold) + HARD (train 3 users -> test 9).
Intermediate results only.
"""
import os, io, zipfile, re, time
import numpy as np
from preprocess import SpecConfig, build_spectrum, max_norm
from spectra_dataset import fit_ranges
from cnn import train_eval
import infineon_detection as ifx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
AXES = ["x", "y", "z"]
ARMS = {
    "velocity":   ("doppler", "mean"),
    "occupancy":  ("count", "sum"),
    "int_mag":    ("intensity", "mean"),
    "int_logmag": ("intensity_log", "mean"),
    "int_r2":     ("intensity_rc", "mean"),
    "int_snr":    ("intensity_snr", "mean"),
}

t0 = time.time(); print("building Infineon instances (with SNR column)...", flush=True)
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
        df = df.copy()
        rr = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
        df["intensity_rc"] = df.intensity * rr ** 2
        df["intensity_log"] = np.log1p(df.intensity)
        recs.append((df, LABELMAP[cls], user)); by[cls] = by.get(cls, 0) + 1
print(f"{len(recs)} instances in {time.time()-t0:.0f}s", flush=True)

cfg = SpecConfig(32, 40, fit_ranges([t[0] for t in recs]))
y = np.array([t[1] for t in recs]); subj = np.array([t[2] for t in recs])
arms = {}
for k, (val, agg) in ARMS.items():
    arms[k] = np.stack([np.stack([max_norm(build_spectrum(i, a, val, cfg, agg=agg)) for a in AXES]).astype(np.float32)
                        for i, _, _ in recs])

users = sorted(set(subj.tolist()))
rng = np.random.RandomState(0); perm = rng.permutation(users)


def run(folds, tag, seeds):
    print(f"\n#### {tag} ####", flush=True)
    acc = {}
    for k, X in arms.items():
        a = []
        for te_s in folds:
            te = np.isin(subj, list(te_s)); tr = ~te
            a += [train_eval(X[tr], y[tr], X[te], y[te], 5, epochs=30, seed=s) for s in seeds]
        acc[k] = float(np.mean(a)) * 100
        print(f"   {k:11s}: {acc[k]:6.2f}%", flush=True)
    return acc

run([list(g) for g in np.array_split(perm, 4)], "EASY: user 4-fold", (0,))
run([list(perm[3:])], "HARD: train 3 users -> test 9", (0, 1, 2))
print("\n[intermediate results only — interpretation deferred]")
