"""Render Infineon sample velocity spectra (XTD/YTD/ZTD) under a few CFAR configs
to judge point-cloud QUALITY (coherent motion vs noise), not just count."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import SpecConfig, make_channels, max_norm
from spectra_dataset import fit_ranges
import infineon_detection as ifx

HERE = os.path.dirname(os.path.abspath(__file__))
cube = np.load(os.path.join(HERE, "..", "data", "infineon_raw_sample", "user10_e1_recording0.npy"))
F = cube.shape[0]

CONFIGS = [("CA a=2.5", dict(method="ca", alpha=2.5)),
           ("CA a=2.0", dict(method="ca", alpha=2.0)),
           ("topK=20", dict(method="topk", topk=20)),
           ("topK=30", dict(method="topk", topk=30))]

fig, axes = plt.subplots(len(CONFIGS), 4, figsize=(11, 2.4 * len(CONFIGS)))
for r, (name, kw) in enumerate(CONFIGS):
    df = ifx.process_recording(cube, **kw)
    ppf = len(df) / F
    ranges = fit_ranges([df]) if len(df) else {"x": (-.3, .3), "y": (0, 1), "z": (-.7, .5)}
    cfg = SpecConfig(100, 40, ranges)
    ch = make_channels(df, cfg)
    # scatter x vs frame (spatial-temporal motion)
    ax = axes[r, 0]
    if len(df):
        ax.scatter(df.frame, df.x, s=4, c=df.doppler, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_ylabel(f"{name}\n{ppf:.1f} pts/fr", fontsize=9); ax.set_xlabel("frame->x" if r == len(CONFIGS)-1 else "")
    for c, key in enumerate(["XTD", "YTD", "ZTD"]):
        ax = axes[r, c + 1]
        ax.imshow(max_norm(ch[key]), aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(key, fontsize=10)
fig.suptitle("Infineon sample: point-cloud quality across CFAR configs (col1: x-vs-frame colored by velocity)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "..", "docs", "infineon_cfar_compare.png")
fig.savefig(out, dpi=115); print("saved", out, "| ppf:", {n: round(len(ifx.process_recording(cube, **kw))/F, 1) for n, kw in CONFIGS})
