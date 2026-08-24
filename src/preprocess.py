"""
M-Gesture (long_SEP point clouds) -> Doppler-accumulation (XTD/YTD/ZTD) and
amplitude-accumulation (XTA/YTA/ZTA) spatiotemporal spectra.

Per-point CSV columns (verified from real data + M-Gesture README line 106):
    Frame # , # Obj , X , Y , Z , Doppler , Intensity
Each CSV holds >=50 instances of ONE gesture class, separated by large frame gaps.

This module is dataset-agnostic at the core: build_spectrum() takes an instance
(a DataFrame of points with columns x,y,z,doppler,intensity,frame) and a config,
so the same code will apply to mHomeGes / MM-Fi / mRI once their loaders map to
the same canonical columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

CANONICAL = ["frame", "x", "y", "z", "doppler", "intensity"]

# gesture class from M-Gesture long-range filename token
MGESTURE_CLASSES = ["knock", "lswipe", "rswipe", "rotate", "unex"]


def load_mgesture_csv(path: str) -> pd.DataFrame:
    """Load one long_SEP CSV into canonical columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "frame": df["Frame #"].astype(int),
        "x": df["X"].astype(float),
        "y": df["Y"].astype(float),
        "z": df["Z"].astype(float),
        "doppler": df["Doppler"].astype(float),
        "intensity": df["Intensity"].astype(float),
    })
    return out


def segment_by_gap(df: pd.DataFrame, gap_thresh: int = 20, min_frames: int = 6) -> list[pd.DataFrame]:
    """Split a continuous stream into gesture instances on large frame gaps.

    README: instances are 'divided by a large interval between each two of them'.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    frames = df["frame"].values
    # boundary where the jump to the next *distinct* frame exceeds gap_thresh
    boundaries = [0]
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > gap_thresh:
            boundaries.append(i)
    boundaries.append(len(frames))
    segs = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = df.iloc[a:b]
        if seg["frame"].nunique() >= min_frames:
            segs.append(seg.reset_index(drop=True))
    return segs


def segment_sliding(df: pd.DataFrame, window: int = 40, stride: int = 20) -> list[pd.DataFrame]:
    """Fixed-length sliding-window segmentation over a *continuous* stream.

    Used for gestures performed continuously without inter-instance gaps
    (e.g. M-Gesture 'rotate', whose max frame gap is ~6). Windows are cut in
    frame-number space so the temporal scale matches gap-segmented classes.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    f = df["frame"].values
    f0, f1 = f.min(), f.max()
    segs = []
    start = f0
    while start + window <= f1 + stride:
        m = (f >= start) & (f < start + window)
        seg = df.iloc[np.where(m)[0]]
        if seg["frame"].nunique() >= max(4, window // 8):
            segs.append(seg.reset_index(drop=True))
        start += stride
    return segs


def segment_instances(df: pd.DataFrame, gap_thresh: int = 20, min_frames: int = 6,
                      window: int = 40, stride: int = 20, min_gap_segments: int = 5) -> tuple[list[pd.DataFrame], str]:
    """Segment into gesture instances, choosing gap-based (discrete gestures) or
    sliding-window (continuous gestures) automatically.

    Returns (segments, method) where method is 'gap' or 'sliding'.
    """
    gap_segs = segment_by_gap(df, gap_thresh=gap_thresh, min_frames=min_frames)
    if len(gap_segs) >= min_gap_segments:
        return gap_segs, "gap"
    return segment_sliding(df, window=window, stride=stride), "sliding"


@dataclass
class SpecConfig:
    n_bins: int = 32               # spatial bins per axis
    n_frames: int = 40             # fixed temporal length
    ranges: dict = field(default_factory=lambda: {  # fixed spatial ranges (m), data-driven
        "x": (-1.0, 1.0),
        "y": (0.3, 3.0),
        "z": (-1.5, 2.5),
    })


def build_spectrum(inst: pd.DataFrame, axis: str, value: str, cfg: SpecConfig, agg: str = "sum") -> np.ndarray:
    """2D (n_bins x n_frames) accumulation of `value` binned along `axis` over time.

    agg="sum": E(w_i,t) = sum_p value_p  (original; couples the value to point density).
    agg="mean": per-bin mean value (sum / per-bin count) -> DECOUPLES the value channel
    from point density/geometry, so velocity/amplitude no longer superset occupancy.
    `value="count"` (occupancy) is always a raw count regardless of agg.
    Variable-length instances are mapped onto a fixed n_frames time grid.
    """
    n_bins, T = cfg.n_bins, cfg.n_frames
    lo, hi = cfg.ranges[axis]
    w = inst[axis].values.astype(float)
    if value == "count":                       # occupancy / geometry arm: accumulate 1 per point
        val = np.ones(len(inst), dtype=float)
    else:
        val = inst[value].values.astype(float)
    fr = inst["frame"].values.astype(float)

    # spatial bin index
    bi = np.floor((w - lo) / (hi - lo) * n_bins).astype(int)
    # temporal bin index: map instance frame span onto 0..T-1
    f0, f1 = fr.min(), fr.max()
    if f1 > f0:
        ti = np.floor((fr - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
    else:
        ti = np.zeros_like(fr, dtype=int)

    m = (bi >= 0) & (bi < n_bins) & (ti >= 0) & (ti < T)
    spec = np.zeros((n_bins, T), dtype=np.float32)
    np.add.at(spec, (bi[m], ti[m]), val[m])
    if agg == "mean" and value != "count":
        cnt = np.zeros((n_bins, T), dtype=np.float32)
        np.add.at(cnt, (bi[m], ti[m]), 1.0)
        spec = np.divide(spec, cnt, out=np.zeros_like(spec), where=cnt > 0)
    return spec


def make_channels(inst: pd.DataFrame, cfg: SpecConfig) -> dict[str, np.ndarray]:
    """Nine maps for one instance, three arms:
       velocity  X/Y/Z T-D  (XTD/YTD/ZTD),
       amplitude X/Y/Z T-A  (XTA/YTA/ZTA),
       occupancy X/Y/Z T-O  (XTO/YTO/ZTO)  <- geometry control (per-bin point count).
    """
    out = {}
    for ax, name in (("x", "X"), ("y", "Y"), ("z", "Z")):
        out[f"{name}TD"] = build_spectrum(inst, ax, "doppler", cfg)
        out[f"{name}TA"] = build_spectrum(inst, ax, "intensity", cfg)
        out[f"{name}TO"] = build_spectrum(inst, ax, "count", cfg)
    return out


def max_norm(a: np.ndarray) -> np.ndarray:
    m = np.abs(a).max()
    return a / m if m > 0 else a
