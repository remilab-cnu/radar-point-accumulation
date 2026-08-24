"""Diagnose: WHAT differs between Infineon and mHomeGes that makes geometry win there?
Intermediate results only — no conclusions. Four hypotheses probed:
 H1 motion-gated occupancy: our chain removes static clutter -> all Infineon points MOVE
    -> occupancy = spatial trace of the moving hand (a motion map). TI clouds contain
    statics. Probe: % near-zero-velocity points per dataset.
 H2 tangential motion: swipes are mostly tangential at boresight -> weak radial velocity
    (only push is radial). Probe: per-class mean |v| on Infineon.
 H3 gesture-set geometric separability: swipe L/R/U/D differ in spatial trajectory ->
    geometry discriminative. Probe: occupancy cos-overlap + NCM per arm, Infineon vs
    mHomeGes; NCM per-class recall (velocity vs occupancy) on Infineon.
 H4 velocity resolution: SPEC check only here (sweep in vres_sweep.py):
    Infineon v_res=0.258 m/s is FINER than mHomeGes observed 0.356 m/s.
"""
import os, io, zipfile, glob
import numpy as np
from preprocess import load_mgesture_csv, segment_instances
import infineon_detection as ifx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def ncm_xsub_preds(X, y, subj, folds):
    F = X.reshape(len(X), -1).astype(np.float64)
    yt_all, yp_all = [], []
    for te_s in folds:
        te = np.isin(subj, list(te_s)); tr = ~te
        cls = np.unique(y[tr])
        cents = np.stack([F[tr][y[tr] == c].mean(0) for c in cls])
        d = ((F[te][:, None, :] - cents[None]) ** 2).sum(-1)
        yp_all.append(cls[d.argmin(1)]); yt_all.append(y[te])
    return np.concatenate(yt_all), np.concatenate(yp_all)


def cos_overlap(X, y):
    cls = np.unique(y)
    cm = np.stack([X[y == c].reshape((y == c).sum(), -1).mean(0) for c in cls])
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    S = cm @ cm.T
    return float(S[np.triu_indices(len(cls), 1)].mean())


def kfold(users, k, seed=0):
    rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(sorted(users)), k)]


print("=" * 70)
print("SPEC TABLE (H4 pre-check)")
print(f"  Infineon : v_max=4.13 m/s, 32 Doppler bins -> v_res=0.258 m/s (FINER)")
print(f"  mHomeGes : observed per-point quantization ~0.356 m/s")
print(f"  -> velocity RESOLUTION per se cannot explain a velocity handicap on Infineon")

# ---------- H3: separability per arm, both datasets ----------
print("\n" + "=" * 70)
print("H3: per-arm NCM cross-subject + occupancy cos-overlap (gesture-set geometry)")
for name, cache, keys, foldk in (
        ("mHomeGes", "mhomeges_gate.npz", {"velocity": "Xv", "intensity": "Xa", "occupancy": "Xo"}, 5),
        ("Infineon", "infineon_gate.npz", {"velocity": "velocity", "intensity": "intensity", "occupancy": "occupancy"}, 4)):
    z = np.load(os.path.join(DATA, cache), allow_pickle=True)
    y, subj = z["y"], z["subj"]
    folds = kfold(set(subj.tolist()), foldk)
    print(f"\n  {name} ({len(y)} inst):  [note: mHomeGes cache arms are SUM-accum, Infineon vel/int are MEAN]")
    for arm, k in keys.items():
        X = z[k]
        yt, yp = ncm_xsub_preds(X, y, subj, folds)
        acc = (yt == yp).mean() * 100
        print(f"    {arm:10s}: NCM={acc:6.2f}%   cos-overlap={cos_overlap(X, y):.3f}")
    # per-class recall velocity vs occupancy
    Xv, Xo = z[keys["velocity"]], z[keys["occupancy"]]
    ytv, ypv = ncm_xsub_preds(Xv, y, subj, folds)
    yto, ypo = ncm_xsub_preds(Xo, y, subj, folds)
    cls = np.unique(y)
    rv = [float((ypv[ytv == c] == c).mean()) for c in cls]
    ro = [float((ypo[yto == c] == c).mean()) for c in cls]
    print(f"    per-class NCM recall (velocity vs occupancy):")
    for c in cls:
        print(f"      class {c}: vel={rv[c]*100:5.1f}  occ={ro[c]*100:5.1f}  diff={(rv[c]-ro[c])*100:+6.1f}")

# ---------- H1 + H2: point-level stats ----------
print("\n" + "=" * 70)
print("H1/H2: point-level velocity stats")

# mHomeGes points (TI-reported clouds)
vs = []
for f in sorted(glob.glob(os.path.join(DATA, "mhomeges_full", "longGes_1.2m", "805", "point_*.csv"))):
    for seg in segment_instances(load_mgesture_csv(f))[0][:5]:
        vs.append(seg.doppler.values)
v = np.concatenate(vs)
steps = np.diff(np.unique(np.round(np.abs(v), 4)))
print(f"  mHomeGes: n={len(v)} pts | %v==0 = {(v==0).mean()*100:.1f}%  %|v|<=0.36 = {(np.abs(v)<=0.36).mean()*100:.1f}%"
      f"  mean|v|={np.abs(v).mean():.3f}  median unique step={np.median(steps[steps>0]):.3f}")

# Infineon points via chain, with per-class labels (user1, 100 recordings)
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
zf = zipfile.ZipFile(ZIP)
with zf.open("fulldataset/user1_e1.npz") as f:
    d = np.load(io.BytesIO(f.read())); inputs, targets = d["inputs"], d["targets"]
percls = {c: [] for c in range(5)}
allv = []
n = 0
for r in np.random.RandomState(0).permutation(len(inputs)):
    g = np.where(targets[r] > 0)[0]
    if len(g) < 2:
        continue
    cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
    if cls not in LABELMAP:
        continue
    df = ifx.process_recording(inputs[r, max(0, g.min() - 6):g.max() + 7])
    if not len(df):
        continue
    percls[LABELMAP[cls]].append(np.abs(df.doppler.values).mean())
    allv.append(df.doppler.values)
    n += 1
    if n >= 100:
        break
v2 = np.concatenate(allv)
print(f"  Infineon: n={len(v2)} pts | %v==0 = {(v2==0).mean()*100:.1f}%  %|v|<=0.36 = {(np.abs(v2)<=0.36).mean()*100:.1f}%"
      f"  mean|v|={np.abs(v2).mean():.3f}")
print(f"  Infineon per-class mean|v| (H2; one should be radial=push):")
for c in range(5):
    print(f"    class {c}: mean|v|={np.mean(percls[c]):.3f} m/s  (n={len(percls[c])} recs)")
print("\n[intermediate results only — interpretation deferred]")
