"""Turn segmented radar point-cloud instances into stacked spectra arrays.

Produces TWO parallel representations per instance from the SAME points:
  - Doppler:  [XTD, YTD, ZTD]  (accumulate per-point radial velocity)
  - Amplitude:[XTA, YTA, ZTA]  (accumulate per-point intensity)  -- the control
Each channel is max-normalized. Both are 3-channel; identical downstream model.
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import pandas as pd
from preprocess import (load_mgesture_csv, segment_instances, segment_sliding,
                        make_channels, max_norm, SpecConfig)


def fit_ranges(insts, pad=0.05):
    """Robust per-axis spatial ranges (1st-99th percentile) from a list of instances."""
    xs = np.concatenate([i["x"].values for i in insts])
    ys = np.concatenate([i["y"].values for i in insts])
    zs = np.concatenate([i["z"].values for i in insts])
    r = {}
    for name, a in (("x", xs), ("y", ys), ("z", zs)):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        span = hi - lo
        r[name] = (float(lo - pad * span), float(hi + pad * span))
    return r


def stack_record(inst, cfg: SpecConfig):
    ch = make_channels(inst, cfg)
    Xd = np.stack([max_norm(ch["XTD"]), max_norm(ch["YTD"]), max_norm(ch["ZTD"])], 0)
    Xa = np.stack([max_norm(ch["XTA"]), max_norm(ch["YTA"]), max_norm(ch["ZTA"])], 0)
    return Xd.astype(np.float32), Xa.astype(np.float32)


def build_arrays(records, cfg: SpecConfig):
    """records: iterable of (inst_df, label_int, subject_str)."""
    Xd, Xa, y, subj = [], [], [], []
    for inst, lab, s in records:
        d, a = stack_record(inst, cfg)
        Xd.append(d); Xa.append(a); y.append(lab); subj.append(s)
    return (np.stack(Xd), np.stack(Xa), np.array(y, dtype=np.int64), np.array(subj))


# ---------------- M-Gesture (long_SEP CSV: Frame#,#Obj,X,Y,Z,Doppler,Intensity) --------
MGESTURE_CLASSES = {"knock": 0, "lswipe": 1, "rswipe": 2, "rotate": 3, "unex": 4}


def mgesture_records(sample_dir, cfg: SpecConfig, classes=("knock", "lswipe", "rswipe", "rotate")):
    cls_map = {c: i for i, c in enumerate(classes)}
    for f in sorted(glob.glob(os.path.join(sample_dir, "*", "long_point_*.csv"))):
        m = re.search(r"long_point_(\d+)_([a-z]+)\.csv", os.path.basename(f))
        sid, cls = m.group(1), m.group(2)
        if cls not in cls_map:
            continue
        segs, _ = segment_instances(load_mgesture_csv(f))
        for seg in segs:
            yield seg, cls_map[cls], sid


def build_mgesture(sample_dir, cfg: SpecConfig, cache=None, classes=("knock", "lswipe", "rswipe", "rotate")):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["Xd"], z["Xa"], z["y"], z["subj"]
    Xd, Xa, y, subj = build_arrays(mgesture_records(sample_dir, cfg, classes), cfg)
    if cache:
        np.savez_compressed(cache, Xd=Xd, Xa=Xa, y=y, subj=subj)
    return Xd, Xa, y, subj


# ---------------- generic build with data-driven ranges + caching -----------------
def build_cached(insts_labeled, base_cfg: SpecConfig, cache=None):
    """insts_labeled: list of (inst_df, label_int, subject_str). Auto-fits spatial ranges."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["Xd"], z["Xa"], z["y"], z["subj"]
    ranges = fit_ranges([t[0] for t in insts_labeled])
    cfg = SpecConfig(n_bins=base_cfg.n_bins, n_frames=base_cfg.n_frames, ranges=ranges)
    Xd, Xa, y, subj = build_arrays(insts_labeled, cfg)
    if cache:
        np.savez_compressed(cache, Xd=Xd, Xa=Xa, y=y, subj=subj, ranges=str(ranges))
    return Xd, Xa, y, subj


# ---------------- mHomeGes (same 7-col CSV, short-range arm gestures) --------------
MHOMEGES_CLASSES = ("circle", "clap", "down", "knock", "lift", "pull", "push", "up", "yawn", "z")


def mhomeges_instances(root, classes=MHOMEGES_CLASSES, distances=None):
    cls_map = {c: i for i, c in enumerate(classes)}
    out = []
    for f in sorted(glob.glob(os.path.join(root, "longGes_*", "*", "point_*.csv"))):
        m = re.search(r"point_(\d+)_([0-9.]+)m_([A-Za-z]+)\.csv", os.path.basename(f))
        if not m:
            continue
        sid, dist, cls = m.group(1), m.group(2), m.group(3).lower()
        if cls not in cls_map:
            continue
        if distances and dist not in distances:
            continue
        try:
            segs, _ = segment_instances(load_mgesture_csv(f))
        except Exception:
            continue
        for seg in segs:
            out.append((seg, cls_map[cls], sid))
    return out


# ---------------- MM-Fi (float64 bins: x,y,z,intensity,Doppler ; HAR, 27 classes) ---
def load_mmfi_action(folder):
    """Concatenate all frame*.bin of one action into a canonical point df.
    On-disk column order is [x, y, z, intensity, Doppler] (Doppler = index 4)."""
    frames = sorted(glob.glob(os.path.join(folder, "frame*.bin")))
    rows = []
    for fi, fp in enumerate(frames):
        a = np.frombuffer(open(fp, "rb").read(), dtype=np.float64).reshape(-1, 5)
        if a.size == 0:
            continue
        n = a.shape[0]
        blk = np.empty((n, 6))
        blk[:, 0] = fi            # frame index
        blk[:, 1] = a[:, 0]       # x
        blk[:, 2] = a[:, 1]       # y
        blk[:, 3] = a[:, 2]       # z
        blk[:, 4] = a[:, 4]       # doppler (signed col)
        blk[:, 5] = a[:, 3]       # intensity
        rows.append(blk)
    if not rows:
        return None
    arr = np.concatenate(rows, 0)
    return pd.DataFrame(arr, columns=["frame", "x", "y", "z", "doppler", "intensity"])


def mmfi_instances(root, window=None, stride=None, actions=None):
    """One instance per (subject, action) folder; optional sliding-window augmentation."""
    out = []
    for sdir in sorted(glob.glob(os.path.join(root, "filtered_mmwave", "E*", "S*"))):
        sid = os.path.basename(sdir)  # S01..S40
        for adir in sorted(glob.glob(os.path.join(sdir, "A*"))):
            act = os.path.basename(adir)  # A01..A27
            if actions and act not in actions:
                continue
            df = load_mmfi_action(adir)
            if df is None or df["frame"].nunique() < 6:
                continue
            lab = int(act[1:]) - 1
            segs = segment_sliding(df, window, stride) if window else [df]
            for seg in segs:
                out.append((seg, lab, sid))
    return out
