"""Validate the M-Gesture -> XTD/YTD/ZTD & XTA/YTA/ZTA pipeline on the in-repo sample.

Run: python3 validate_mgesture.py
Outputs: instance counts, spectrum shapes, and a comparison figure per gesture.
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (load_mgesture_csv, segment_by_gap, make_channels,
                        max_norm, SpecConfig)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "mgesture_sample", "long_SEP")
OUT = os.path.join(HERE, "..", "docs")
os.makedirs(OUT, exist_ok=True)

cfg = SpecConfig()

files = sorted(glob.glob(os.path.join(DATA, "*", "long_point_*.csv")))
print(f"sample files: {len(files)}")

# 1) segmentation stats across all sample files
inst_counts = {}
by_class = {}
for f in files:
    m = re.search(r"long_point_(\d+)_([a-z]+)\.csv", os.path.basename(f))
    sid, cls = m.group(1), m.group(2)
    df = load_mgesture_csv(f)
    segs = segment_by_gap(df)
    inst_counts[(sid, cls)] = len(segs)
    by_class.setdefault(cls, []).append(len(segs))

print("\n=== instances per (subject, class) ===")
for k in sorted(inst_counts):
    print(f"  subj {k[0]} {k[1]:8s}: {inst_counts[k]} instances")
print("\n=== instances per class (mean across subjects) ===")
for cls, v in sorted(by_class.items()):
    print(f"  {cls:8s}: mean={np.mean(v):.1f} min={min(v)} max={max(v)} (n_subj={len(v)})")

# 2) build spectra for one instance of each gesture (subject 007) and render
subj = "007"
classes = ["knock", "lswipe", "rswipe", "rotate"]
fig, axes = plt.subplots(len(classes), 6, figsize=(15, 2.4 * len(classes)))
chan_order = ["XTD", "YTD", "ZTD", "XTA", "YTA", "ZTA"]
shapes_printed = False
for r, cls in enumerate(classes):
    f = os.path.join(DATA, subj, f"long_point_{subj}_{cls}.csv")
    if not os.path.exists(f):
        continue
    segs = segment_by_gap(load_mgesture_csv(f))
    inst = segs[len(segs) // 2]  # a middle instance
    ch = make_channels(inst, cfg)
    if not shapes_printed:
        print(f"\n=== spectrum shape: {ch['XTD'].shape} (n_bins x n_frames) ===")
        shapes_printed = True
    for c, name in enumerate(chan_order):
        ax = axes[r, c]
        img = max_norm(ch[name])
        # amplitude channels: log for visibility; doppler: signed
        if name.endswith("A"):
            disp = np.log1p(np.abs(img)); cmap = "viridis"
        else:
            disp = img; cmap = "RdBu_r"
        ax.imshow(disp, aspect="auto", origin="lower", cmap=cmap)
        if r == 0:
            ax.set_title(name, fontsize=10)
        if c == 0:
            ax.set_ylabel(cls, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"M-Gesture subj {subj}: Doppler (XTD/YTD/ZTD) vs Amplitude (XTA/YTA/ZTA)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = os.path.join(OUT, "mgesture_spectra_check.png")
fig.savefig(out_png, dpi=110)
print(f"\nsaved figure: {out_png}")

# 3) quick discriminability sanity: mean energy per channel per class (subject 007)
print("\n=== mean |value| per channel (subj 007, middle instance) ===")
print(f"  {'class':8s} " + " ".join(f"{n:>7s}" for n in chan_order))
for cls in classes:
    f = os.path.join(DATA, subj, f"long_point_{subj}_{cls}.csv")
    if not os.path.exists(f):
        continue
    inst = segment_by_gap(load_mgesture_csv(f))[0]
    ch = make_channels(inst, cfg)
    print(f"  {cls:8s} " + " ".join(f"{np.abs(ch[n]).mean():7.3f}" for n in chan_order))
