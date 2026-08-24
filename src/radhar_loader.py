"""RadHAR loader (5th radar source for the cross-sensor program).

RadHAR (Singh et al., mmNets 2019; nesl/RadHAR, BSD-3) ships per-point ROS ti_mmwave
dumps: each point block carries seq/stamp/point_id + x,y,z,range,velocity,doppler_bin,
bearing,intensity. A frame boundary is where point_id resets to 0 (ti_mmwave RadarScan).
We parse to the project's canonical per-point schema [frame,x,y,z,doppler,intensity]
(doppler = the radar 'velocity' field), then window into fixed-frame instances exactly
as mRI/MM-Fi, honoring the repo's official Train/Test split.

Classes: boxing, jack, jump, squats, walk (gross whole-body). TI IWR1443 77 GHz.
Verified per-point fields present natively (2026-07-13): x,y,z,velocity,intensity.
"""
import os, glob
import numpy as np
import pandas as pd

CLASSES = ["boxing", "jack", "jump", "squats", "walk"]


def parse_file(path):
    """Single pass -> canonical DataFrame [frame,x,y,z,doppler,intensity]."""
    frame = -1
    fr, xs, ys, zs, dp, it = [], [], [], [], [], []
    cur = {}
    have = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("point_id:"):
                if int(s.split(":", 1)[1]) == 0:
                    frame += 1
                cur = {"frame": frame}; have = 0
            elif s.startswith("x:"):
                cur["x"] = float(s.split(":", 1)[1]); have += 1
            elif s.startswith("y:"):
                cur["y"] = float(s.split(":", 1)[1]); have += 1
            elif s.startswith("z:"):
                cur["z"] = float(s.split(":", 1)[1]); have += 1
            elif s.startswith("velocity:"):
                cur["doppler"] = float(s.split(":", 1)[1]); have += 1
            elif s.startswith("intensity:"):          # last field of a point block
                cur["intensity"] = float(s.split(":", 1)[1]); have += 1
                if have == 5 and "frame" in cur:
                    fr.append(cur["frame"]); xs.append(cur["x"]); ys.append(cur["y"])
                    zs.append(cur["z"]); dp.append(cur["doppler"]); it.append(cur["intensity"])
    return pd.DataFrame({"frame": fr, "x": xs, "y": ys, "z": zs,
                         "doppler": dp, "intensity": it})


def radhar_instances(root, split="Train", window=40, stride=20, min_frames=6, min_pts=30):
    """Windowed instances from RadHAR Data/<split>/<class>/*.txt.
    Returns list of (df, label_idx, subject) with subject = file stem (file-disjoint)."""
    base = os.path.join(root, "Data", split)
    recs = []
    for ci, cls in enumerate(CLASSES):
        for path in sorted(glob.glob(os.path.join(base, cls, "*.txt"))):
            df = parse_file(path)
            if len(df) == 0:
                continue
            sid = os.path.basename(path).replace(".txt", "")
            f0, f1 = int(df.frame.min()), int(df.frame.max())
            t0 = f0
            while t0 + window <= f1 + 1:
                w = df[(df.frame >= t0) & (df.frame < t0 + window)]
                if w["frame"].nunique() >= min_frames and len(w) >= min_pts:
                    recs.append((w.reset_index(drop=True), ci, f"{split}:{sid}"))
                t0 += stride
    return recs


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.join(HERE, "..", "data", "radhar")
    # smoke: parse one file per class, report format sanity
    for cls in CLASSES:
        p = sorted(glob.glob(os.path.join(ROOT, "Data", "Train", cls, "*.txt")))[0]
        df = parse_file(p)
        print(f"{cls:8s} {os.path.basename(p):22s} pts={len(df):6d} frames={df.frame.nunique():4d} "
              f"v[{df.doppler.min():+.2f},{df.doppler.max():+.2f}] "
              f"xyz~[{df.x.min():.1f},{df.x.max():.1f}]x[{df.y.min():.1f},{df.y.max():.1f}]x[{df.z.min():.1f},{df.z.max():.1f}] "
              f"int[{df.intensity.min():.0f},{df.intensity.max():.0f}]", flush=True)
    # smoke: windowed instance counts (Train subset = first 2 files/class handled inside)
    import time
    t0 = time.time()
    tr = radhar_instances(ROOT, "Test")                    # Test is smaller -> quick smoke
    from collections import Counter
    print(f"\nTest instances: {len(tr)} in {time.time()-t0:.0f}s; class dist "
          f"{dict(Counter(t[1] for t in tr))}; subjects {len(set(t[2] for t in tr))}", flush=True)
