"""Intensity-arm audit, Step 2 (decisive): Step 1 showed Infineon |RD| intensity is not
motion-coupled but DOES leak range/geometry (corr with range -0.37 vs TI -0.11). Two tests:
 (1) FAIRNESS: range-compensate intensity (amplitude ~ 1/r^2 two-way -> multiply by r^2) so it
     becomes a reflectivity-like quantity comparable to TI's AGC'd SNR. Does its lead vanish?
 (2) SATURATION: re-run at an EASY (4-fold) and a HARD/data-starved (train 3 users -> test 9)
     protocol. Do arms separate, and does velocity emerge over intensity when de-saturated?
Self-contained (does not import infineon_build to avoid re-running its module code).
"""
import os, io, zipfile, re, time
import numpy as np
from preprocess import SpecConfig, build_spectrum, max_norm
from spectra_dataset import fit_ranges
from cnn import train_eval
import infineon_detection as ifx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}; CAP = 200; MARGIN = 6
AXES = ["x", "y", "z"]
ARMS = {  # (column, agg)
    "velocity":  ("doppler", "mean"),
    "occupancy": ("count", "sum"),
    "int_raw":   ("intensity", "mean"),
    "int_rc":    ("intensity_rc", "mean"),   # range-compensated
}


def build_instances():
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
            if cls not in LABELMAP or by.get(cls, 0) >= CAP // 5:
                continue
            df = ifx.process_recording(inputs[r, max(0, g.min() - MARGIN):g.max() + MARGIN + 1])
            if len(df) < 8:
                continue
            rr = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
            df = df.copy(); df["intensity_rc"] = df.intensity * (rr ** 2)   # 1/r^2 amplitude comp
            recs.append((df, LABELMAP[cls], user)); by[cls] = by.get(cls, 0) + 1
    return recs


def build_arms(recs):
    cfg = SpecConfig(32, 40, fit_ranges([t[0] for t in recs]))
    out = {k: [] for k in ARMS}; y = []; subj = []
    for inst, lab, s in recs:
        for k, (val, agg) in ARMS.items():
            out[k].append(np.stack([max_norm(build_spectrum(inst, a, val, cfg, agg=agg)) for a in AXES]).astype(np.float32))
        y.append(lab); subj.append(s)
    return {k: np.stack(v) for k, v in out.items()}, np.array(y, np.int64), np.array(subj)


t0 = time.time(); print("reprocessing Infineon (range-comp intensity)...", flush=True)
recs = build_instances()
arms, y, subj = build_arms(recs)
users = sorted(set(subj.tolist()))
print(f"{len(y)} instances, {len(users)} users, in {time.time()-t0:.0f}s", flush=True)


def run(folds, tag):
    print(f"\n#### {tag} ####")
    acc = {}
    for k, X in arms.items():
        a = []
        for te_s in folds:
            te = np.isin(subj, list(te_s)); tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            a += [train_eval(X[tr], y[tr], X[te], y[te], 5, epochs=30, seed=s) for s in (0, 1)]
        acc[k] = float(np.mean(a)); print(f"   {k:10s}: {acc[k]*100:6.2f}%", flush=True)
    print(f"   >> velocity-int_raw={ (acc['velocity']-acc['int_raw'])*100:+.2f}  "
          f"velocity-int_rc={(acc['velocity']-acc['int_rc'])*100:+.2f}  "
          f"velocity-occupancy={(acc['velocity']-acc['occupancy'])*100:+.2f}", flush=True)

rng = np.random.RandomState(0); perm = rng.permutation(users)
run([list(g) for g in np.array_split(perm, 4)], "EASY: user 4-fold (near-saturated)")
run([list(perm[3:])], "HARD: train 3 users -> test 9 (data-starved / de-saturated)")
