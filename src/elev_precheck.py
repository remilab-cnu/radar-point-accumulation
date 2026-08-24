"""SphereBin pre-check (gesture / mHomeGes): is the ELEVATION axis (z -> el) reliable
enough to include in spherical velocity maps, or is IWR1443 elevation too coarse/noisy
(azimuth-mostly array) so that (r-el)/(el) channels ADD NOISE?

Test: cross-subject velocity accuracy with spherical axes {r,az,el} vs {r,az} (drop
elevation) vs {el} alone. If dropping el does not hurt (or helps), restrict SphereBin
to (r,az) and report the caveat. Also report z/el descriptive stats (quantization).
"""
import os, numpy as np
from compare_coords import build, cross_subject, kfold, add_sph
from spectra_dataset import mhomeges_instances

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
print(f"mHomeGes instances: {len(mh)}")

# --- descriptive: z / elevation quantization & spread ---
sph = [add_sph(t[0]) for t in mh[:400]]
allz = np.concatenate([d.z.values for d in sph]); alle = np.concatenate([d.el.values for d in sph])
allr = np.concatenate([d.r.values for d in sph])
print("\n=== elevation reliability descriptives ===")
print(f"  z : std={allz.std():.3f}m  unique-values={len(np.unique(np.round(allz,3)))}  "
      f"p1={np.percentile(allz,1):.2f} p99={np.percentile(allz,99):.2f}")
print(f"  el: std={np.degrees(alle.std()):.2f}deg  p1={np.degrees(np.percentile(alle,1)):.1f} "
      f"p99={np.degrees(np.percentile(alle,99)):.1f}")
print(f"  (for reference) r std={allr.std():.3f}m")
# per-instance z quantization: median number of distinct z levels
qz = np.median([len(np.unique(np.round(d.z.values, 3))) / max(len(d), 1) for d in sph])
print(f"  median distinct-z-ratio per instance = {qz:.2f}  (low => degenerate/quantized elevation)")

# --- cross-subject velocity accuracy with different spherical axis sets ---
folds = kfold(np.array([t[2] for t in mh]), 3)
print("\n=== cross-subject velocity acc (3-fold, 2 seeds) by spherical axis set ===")
for axes in (["r", "az", "el"], ["r", "az"], ["r", "el"], ["el"], ["az"]):
    Xv, Xo, y, subj = build(mh, axes)
    v, o = cross_subject(Xv, Xo, y, subj, folds, 10, 15)
    print(f"  axes={str(axes):22s} velocity={v*100:6.2f}%  geometry={o*100:6.2f}%")
