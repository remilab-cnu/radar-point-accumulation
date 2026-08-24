"""Calibrate the Infineon detection chain so its point density matches the other
datasets. (1) measure points/frame on mHomeGes, MM-Fi, M-Gesture; (2) sweep Infineon
CFAR configs on the sample recording and report points/frame + spatial/velocity spread.
"""
import os, glob
import numpy as np
from preprocess import load_mgesture_csv, segment_instances
from spectra_dataset import load_mmfi_action
import infineon_detection as ifx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def ppf(df):
    """points per frame distribution -> (median, p25, p75)."""
    c = df.groupby("frame").size().values
    return np.median(c), np.percentile(c, 25), np.percentile(c, 75)


print("=== TARGET: points/frame in existing datasets ===")
targets = {}

# mHomeGes: a few gesture files
mh = []
for f in glob.glob(os.path.join(DATA, "mhomeges_full", "longGes_1.2m", "805", "point_*.csv"))[:6]:
    for seg in segment_instances(load_mgesture_csv(f))[0][:5]:
        mh.append(ppf(seg)[0])
if mh:
    targets["mHomeGes"] = np.median(mh); print(f"  mHomeGes : median {np.median(mh):.1f} pts/frame (n_inst={len(mh)})")

# MM-Fi: a few actions of S01
mf = []
base = os.path.join(DATA, "mmfi_extracted", "filtered_mmwave")
sdir = (sorted(glob.glob(os.path.join(base, "E*", "S01"))) or [None])[0]
if sdir:
    for adir in sorted(glob.glob(os.path.join(sdir, "A*")))[:6]:
        df = load_mmfi_action(adir)
        if df is not None and len(df):
            mf.append(ppf(df)[0])
if mf:
    targets["MM-Fi"] = np.median(mf); print(f"  MM-Fi    : median {np.median(mf):.1f} pts/frame (n_act={len(mf)})")

# M-Gesture sample
mg = []
for f in glob.glob(os.path.join(DATA, "mgesture_sample", "long_SEP", "007", "long_point_*.csv")):
    for seg in segment_instances(load_mgesture_csv(f))[0][:5]:
        mg.append(ppf(seg)[0])
if mg:
    targets["M-Gesture"] = np.median(mg); print(f"  M-Gesture: median {np.median(mg):.1f} pts/frame (n_inst={len(mg)})")

tgt = np.median(list(targets.values())) if targets else 15
print(f"  --> overall target ~= {tgt:.0f} pts/frame")

print("\n=== SWEEP: Infineon detection configs on the sample recording ===")
cube = np.load(os.path.join(DATA, "infineon_raw_sample", "user10_e1_recording0.npy"))
F = cube.shape[0]

configs = []
for a in (2.0, 2.5, 3.0, 4.0, 6.0):
    configs.append(("ca", dict(method="ca", alpha=a)))
for a in (1.5, 2.0, 3.0):
    configs.append(("os", dict(method="os", alpha=a)))
for k in (8, 12, 16, 20, 25):
    configs.append(("topk", dict(method="topk", topk=k)))

print(f"  {'config':28s} {'pts/frame':>10s} {'std_x':>7s} {'std_y':>7s} {'std_z':>7s} {'|v|>0.3':>8s}")
best = None
for name, kw in configs:
    df = ifx.process_recording(cube, **kw)
    n = len(df) / F
    if len(df):
        sx, sy, sz = df.x.std(), df.y.std(), df.z.std()
        mov = np.mean(np.abs(df.doppler) > 0.3)
    else:
        sx = sy = sz = mov = 0
    tag = f"{name} {kw}".replace("method=", "").replace("'", "")
    print(f"  {tag:28s} {n:10.1f} {sx:7.3f} {sy:7.3f} {sz:7.3f} {mov:8.2f}")
    score = abs(n - tgt)
    if best is None or score < best[0]:
        best = (score, name, kw, n)

print(f"\n  closest to target ({tgt:.0f}): {best[1]} {best[2]} -> {best[3]:.1f} pts/frame")
