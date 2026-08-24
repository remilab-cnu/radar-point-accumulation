"""Definitive C recomputation (PI directive, 2026-07-19).

Motive: the quoted mHomeGes C=0.764 could not be reproduced (400-inst subsample,
nb=32 -> 0.845) and its computation config is undocumented ("computed earlier").
This run recomputes C for ALL FOUR datasets on the FULL instance sets (no
subsampling anywhere) under the documented cancel_stat definition:
  per-instance ratio C_i = sum_axes |sum_cell v| / sum_axes sum_cell |v|,
  cells = per-axis (x,y,z) nb x T grids, nb=32, T=40,
  per-instance 1-99th percentile spatial ranges, dataset C = mean_i C_i.
Diagnostics per dataset:
  - pooled variant (sum_i num_i / sum_i den_i)  [mass-weighted across instances]
  - dataset-fitted-ranges variant (fit_ranges, as the trained maps use)
  - per-instance C distribution stats (std, quartiles)
to identify which variant the historical 0.764 corresponds to.
Out: docs/c_recompute_full.json
"""
import os, glob, json, pickle
import numpy as np
import pandas as pd
from spectra_dataset import mmfi_instances, mhomeges_instances, fit_ranges
from rep_variants import infineon_recs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
NB, T = 32, 40


def inst_ratio(inst, ranges=None):
    f = inst["frame"].values.astype(float); f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
    v = inst["doppler"].values.astype(float)
    num = den = 0.0
    for ax in ("x", "y", "z"):
        if ranges is None:
            lo, hi = np.percentile(inst[ax], 1), np.percentile(inst[ax], 99)
        else:
            lo, hi = ranges[ax]
        bi = np.floor((inst[ax].values - lo) / max(hi - lo, 1e-9) * NB).astype(int)
        m = (bi >= 0) & (bi < NB)
        sv = np.zeros((NB, T)); sa = np.zeros((NB, T))
        np.add.at(sv, (bi[m], ti[m]), v[m]); np.add.at(sa, (bi[m], ti[m]), np.abs(v[m]))
        num += np.abs(sv).sum(); den += sa.sum()
    return num, den


def dataset_c(insts, name, out):
    # per-instance percentile ranges (cancel_stat definition)
    nums, dens = [], []
    for inst, _, _ in insts:
        n, d = inst_ratio(inst)
        if d > 0:
            nums.append(n); dens.append(d)
    r = np.array(nums) / np.array(dens)
    # dataset-fitted ranges variant (map geometry)
    rng_fit = fit_ranges([t[0] for t in insts])
    nums2, dens2 = [], []
    for inst, _, _ in insts:
        n, d = inst_ratio(inst, rng_fit)
        if d > 0:
            nums2.append(n); dens2.append(d)
    r2 = np.array(nums2) / np.array(dens2)
    out[name] = {
        "n_instances": len(r),
        "C_mean_per_instance": round(float(r.mean()), 3),
        "C_pooled": round(float(np.sum(nums) / np.sum(dens)), 3),
        "C_fitted_ranges_mean": round(float(r2.mean()), 3),
        "C_fitted_ranges_pooled": round(float(np.sum(nums2) / np.sum(dens2)), 3),
        "C_std": round(float(r.std()), 3),
        "C_quartiles": [round(float(q), 3) for q in np.percentile(r, [25, 50, 75])],
    }
    print(name, out[name], flush=True)


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
    out = {"definition": "per-instance C_i = sum_ax |sum_cell v| / sum_ax sum_cell |v|; "
                         "nb=32, T=40; FULL datasets, no subsampling",
           }
    dataset_c(infineon_recs(), "Infineon", out)
    dataset_c(mmfi_instances(os.path.join(DATA, "mmfi_extracted")), "MM-Fi", out)
    dataset_c(mri_windows(), "mRI", out)
    dataset_c(mhomeges_instances(os.path.join(DATA, "mhomeges_full")), "mHomeGes", out)
    json.dump(out, open(os.path.join(DOCS, "c_recompute_full.json"), "w"), indent=1)
    print("wrote docs/c_recompute_full.json", flush=True)
