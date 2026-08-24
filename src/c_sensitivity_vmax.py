"""Referee items W3 (2026-07-19): C bin-size sensitivity + empirical v_max table.

(1) C(nb) for nb in {16,32,64} (T=40 fixed), 400-instance subsample per dataset
    (rng seed 0), same per-instance 1-99th-percentile ranges as cancel_stat.py.
(2) Per-dataset Doppler span/resolution measured from the point clouds themselves:
    max|v|, 99.9th percentile |v|, and the minimum positive spacing of unique
    values (quantization step ~ Doppler resolution).
Out: docs/c_sensitivity_vmax.json
"""
import os, glob, json, pickle
import numpy as np
import pandas as pd
from spectra_dataset import mmfi_instances, mhomeges_instances
from rep_variants import infineon_recs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
CAP = 400; T = 40


def cancellation(insts, nb):
    ratios = []
    for inst, _, _ in insts[:CAP]:
        f = inst["frame"].values.astype(float); f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
        ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
        v = inst["doppler"].values.astype(float)
        num = den = 0.0
        for ax in ("x", "y", "z"):
            lo, hi = np.percentile(inst[ax], 1), np.percentile(inst[ax], 99)
            bi = np.floor((inst[ax].values - lo) / max(hi - lo, 1e-9) * nb).astype(int)
            m = (bi >= 0) & (bi < nb)
            sv = np.zeros((nb, T)); sa = np.zeros((nb, T))
            np.add.at(sv, (bi[m], ti[m]), v[m]); np.add.at(sa, (bi[m], ti[m]), np.abs(v[m]))
            num += np.abs(sv).sum(); den += sa.sum()
        if den > 0:
            ratios.append(num / den)
    return round(float(np.mean(ratios)), 3), len(ratios)


def vstats(insts):
    v = np.concatenate([t[0]["doppler"].values for t in insts[:CAP]]).astype(float)
    av = np.abs(v)
    u = np.unique(np.round(v, 6))
    d = np.diff(u); step = float(d[d > 1e-9].min()) if (d > 1e-9).any() else None
    return {"n_points": int(len(v)), "max_abs_v": round(float(av.max()), 3),
            "p999_abs_v": round(float(np.percentile(av, 99.9)), 3),
            "min_unique_spacing": round(step, 4) if step else None,
            "n_unique_values": int(len(u))}


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
    rng = np.random.RandomState(0)
    out = {"cap": CAP, "T": T, "C_by_nb": {}, "vstats": {}}
    loaders = {}
    inf = infineon_recs(); loaders["Infineon"] = [inf[i] for i in rng.permutation(len(inf))[:CAP]]
    loaders["MM-Fi"] = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    loaders["mRI"] = mri_windows()
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    loaders["mHomeGes"] = [mh[i] for i in np.random.RandomState(0).permutation(len(mh))[:CAP]]
    for name, insts in loaders.items():
        out["vstats"][name] = vstats(insts)
        out["C_by_nb"][name] = {nb: cancellation(insts, nb) for nb in (16, 32, 64)}
        print(name, out["vstats"][name], out["C_by_nb"][name], flush=True)
    json.dump(out, open(os.path.join(DOCS, "c_sensitivity_vmax.json"), "w"), indent=1)
    print("wrote docs/c_sensitivity_vmax.json", flush=True)
