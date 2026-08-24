"""Intensity-arm fairness audit, Step 1: is Infineon 'intensity' (=|RD| after static-
clutter removal) a motion-energy quantity (coupled to velocity) rather than a pure
reflectivity/geometry control like TI's reported intensity?

Pull intermediate results:
 (A) per-point correlations intensity~|velocity|, intensity~range, intensity~|z| on
     Infineon vs mHomeGes (TI).
 (B) range-Doppler magnitude map BEFORE vs AFTER static-clutter removal (one gesture frame).
"""
import os, io, zipfile, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import infineon_detection as ifx
from preprocess import load_mgesture_csv, segment_instances

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DOCS = os.path.join(DATA, "..", "docs")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")


def corrs(intensity, vel, x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    def c(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
    return c(intensity, np.abs(vel)), c(intensity, r), c(intensity, np.abs(z))


# ---- Infineon per-point ----
zf = zipfile.ZipFile(ZIP)
with zf.open("fulldataset/user1_e1.npz") as f:
    d = np.load(io.BytesIO(f.read())); inputs, targets = d["inputs"], d["targets"]
I, V, X, Y, Z = [], [], [], [], []
n = 0
for r in range(len(inputs)):
    g = np.where(targets[r] > 0)[0]
    if len(g) < 2:
        continue
    cube = inputs[r, max(0, g.min() - 6):g.max() + 7]
    df = ifx.process_recording(cube)
    if len(df):
        I += df.intensity.tolist(); V += df.doppler.tolist()
        X += df.x.tolist(); Y += df.y.tolist(); Z += df.z.tolist()
    n += 1
    if n >= 60:
        break
I, V, X, Y, Z = map(np.array, (I, V, X, Y, Z))
ci_v, ci_r, ci_z = corrs(I, V, X, Y, Z)
print("=== (A) per-point correlations ===")
print(f"Infineon (|RD|, clutter-removed): corr(intensity,|vel|)={ci_v:+.3f}  corr(intensity,range)={ci_r:+.3f}  corr(intensity,|z|)={ci_z:+.3f}  (n={len(I)} pts)")

# ---- mHomeGes (TI) per-point ----
I2, V2, X2, Y2, Z2 = [], [], [], [], []
for fpath in glob.glob(os.path.join(DATA, "mhomeges_full", "longGes_1.2m", "805", "point_*.csv")):
    for seg in segment_instances(load_mgesture_csv(fpath))[0][:6]:
        I2 += seg.intensity.tolist(); V2 += seg.doppler.tolist()
        X2 += seg.x.tolist(); Y2 += seg.y.tolist(); Z2 += seg.z.tolist()
I2, V2, X2, Y2, Z2 = map(np.array, (I2, V2, X2, Y2, Z2))
c2_v, c2_r, c2_z = corrs(I2, V2, X2, Y2, Z2)
print(f"mHomeGes (TI reported intensity): corr(intensity,|vel|)={c2_v:+.3f}  corr(intensity,range)={c2_r:+.3f}  corr(intensity,|z|)={c2_z:+.3f}  (n={len(I2)} pts)")

# ---- (B) RD map before vs after static-clutter removal (one gesture frame) ----
rec = 0
g = np.where(targets[rec] > 0)[0]
frame = inputs[rec, g[len(g) // 2]]                    # a labeled gesture frame (3,32,64)
xw = frame.astype(np.float64); xw = xw - xw.mean(axis=2, keepdims=True)
Rrng = np.fft.rfft(xw * np.hanning(64)[None, None, :], axis=2)     # (3,32,33)
dwin = np.hanning(32)[None, :, None]
RD_with = np.fft.fftshift(np.fft.fft(Rrng * dwin, axis=1), axes=1)               # clutter kept
RD_no = np.fft.fftshift(np.fft.fft((Rrng - Rrng.mean(1, keepdims=True)) * dwin, axis=1), axes=1)  # clutter removed
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, RD, t in ((ax[0], RD_with, "RD |mag| WITH static clutter"), (ax[1], RD_no, "RD |mag| clutter REMOVED (what we use)")):
    a.imshow(np.abs(RD).sum(0), aspect="auto", origin="lower", cmap="viridis")
    a.set_title(t, fontsize=10); a.set_xlabel("range bin"); a.set_ylabel("Doppler bin (0=static center)")
fig.suptitle("Infineon: static-clutter removal makes |RD| a MOVING-target power (nonzero-Doppler)")
fig.tight_layout(); out = os.path.join(DOCS, "intensity_audit_rd.png"); fig.savefig(out, dpi=115)
print(f"\nsaved {out}")
print("\nReading: high Infineon corr(intensity,|vel|) vs low TI corr would confirm our intensity is")
print("motion-coupled (moving-power), NOT a pure reflectivity control -> not apples-to-apples with TI.")
