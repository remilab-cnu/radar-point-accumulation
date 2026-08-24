"""Synthetic-wrap sensitivity of C (panel medium-ask #2, 2026-07-19).

Question: could the LOW C of the whole-body sets be produced by Doppler WRAPPING
(aliasing) rather than genuine antiphase motion? Two probes:

(A) Forward simulation: take the high-C mHomeGes points (full dataset, garbage
    outliers |v|>15 m/s removed) and synthetically wrap them at narrower spans
    V in {4.8, 2.1, 1.5} m/s via v -> ((v+V) mod 2V) - V. If wrapping at the
    whole-body sets' spans leaves C high, wrap alone cannot explain C~0.5.
(B) Exposure bound: on MM-Fi and mRI, the fraction of |v| mass within one
    resolution step of the span boundary (wrapped mass enters from the boundary,
    so near-boundary mass bounds the plausible wrap contamination).

C definition identical to c_recompute_full.py (nb=32, T=40, per-instance
percentile ranges, per-instance-mean). Out: docs/c_wrap_check.json
"""
import os, glob, json, pickle
import numpy as np
import pandas as pd
from spectra_dataset import mmfi_instances, mhomeges_instances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
NB, T = 32, 40


def c_mean(insts, wrap_V=None, clip=15.0):
    rs = []
    for inst, _, _ in insts:
        f = inst["frame"].values.astype(float); f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
        ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
        v = inst["doppler"].values.astype(float)
        v = np.clip(v, -clip, clip)
        if wrap_V is not None:
            v = np.mod(v + wrap_V, 2 * wrap_V) - wrap_V
        num = den = 0.0
        for ax in ("x", "y", "z"):
            lo, hi = np.percentile(inst[ax], 1), np.percentile(inst[ax], 99)
            bi = np.floor((inst[ax].values - lo) / max(hi - lo, 1e-9) * NB).astype(int)
            m = (bi >= 0) & (bi < NB)
            sv = np.zeros((NB, T)); sa = np.zeros((NB, T))
            np.add.at(sv, (bi[m], ti[m]), v[m]); np.add.at(sa, (bi[m], ti[m]), np.abs(v[m]))
            num += np.abs(sv).sum(); den += sa.sum()
        if den > 0:
            rs.append(num / den)
    return round(float(np.mean(rs)), 3)


def boundary_mass(insts, span, res):
    v = np.abs(np.concatenate([t[0]["doppler"].values for t in insts]).astype(float))
    thr = span - res
    return {"span": span, "res": res,
            "frac_points_near_boundary": round(float((v >= thr).mean()), 4),
            "frac_mass_near_boundary": round(float(v[v >= thr].sum() / max(v.sum(), 1e-9)), 4)}


def mri_windows():
    CL = [f"pose_{i}" for i in range(1, 11)]; recs = []
    MRI = os.path.join(DATA, "mri_sample", "mri_data")
    for csvf in sorted(glob.glob(os.path.join(MRI, "subject*.csv"))):
        sid = os.path.basename(csvf).replace(".csv", "")
        if "_all_labels" in sid:
            continue
        df = pd.read_csv(csvf); df.columns = [x.strip() for x in df.columns]
        can = pd.DataFrame({"frame": df["Camera Frame"].astype(int), "x": df["X"], "y": df["Y"],
                            "z": df["Z"], "doppler": df["Doppler"], "intensity": df["Intensity"]})
        vl = pickle.load(open(os.path.join(MRI, f"{sid}_all_labels.cpl"), "rb"))["video_label"]
        for cn in CL:
            if cn not in vl:
                continue
            a, b = vl[cn]; t0 = a
            while t0 + 40 <= b:
                w = can[(can.frame >= t0) & (can.frame < t0 + 40)]
                if w["frame"].nunique() >= 6 and len(w) >= 30:
                    recs.append((w, 0, "x"))
                t0 += 120
    return recs


if __name__ == "__main__":
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    out["A_forward_wrap_mHomeGes_full"] = {
        "C_original": c_mean(mh),
        "C_wrap_at_4.8": c_mean(mh, wrap_V=4.8),
        "C_wrap_at_2.1": c_mean(mh, wrap_V=2.1),
        "C_wrap_at_1.5": c_mean(mh, wrap_V=1.5),
    }
    print("A:", out["A_forward_wrap_mHomeGes_full"], flush=True)
    del mh
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    out["B_boundary_mass"] = {
        "MM-Fi": boundary_mass(mf, span=4.835, res=0.604),
        "mRI": boundary_mass(mri_windows(), span=2.136, res=0.356),
    }
    print("B:", out["B_boundary_mass"], flush=True)
    json.dump(out, open(os.path.join(DOCS, "c_wrap_check.json"), "w"), indent=1)
    print("wrote docs/c_wrap_check.json", flush=True)
