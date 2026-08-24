"""Full-dataset C(nb) grid sweep (provenance gap closed 2026-08-11).

Section V-D of the manuscript quotes MM-Fi C = 0.43 / 0.49 / 0.55 at 16 / 32 / 64
bins. Those values came from an inline computation that was never archived, so a
referee recomputing from the release could not reproduce them: the only stored
sweep, docs/c_sensitivity_vmax.json, uses a 400-instance cap and reads
0.436 / 0.498 / 0.560. This script recomputes the sweep on the FULL datasets,
using the same per-instance definition as cancel_stat.py / c_recompute_full.py.
Out: docs/c_grid_full.json
"""
import os, sys, json, glob, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectra_dataset import mmfi_instances, mhomeges_instances
from measurand_table import infineon_recs_direct, mri_windows

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
T = 40


def cancellation(insts, nb):
    """Mean over instances of |sum v| / sum |v|, summed over the three axis maps."""
    ratios = []
    for tup in insts:
        inst = tup[0]
        f = inst["frame"].values.astype(float)
        f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
        ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
        v = inst["doppler"].values.astype(float)
        num = den = 0.0
        for ax in ("x", "y", "z"):
            lo, hi = np.percentile(inst[ax], 1), np.percentile(inst[ax], 99)
            bi = np.floor((inst[ax].values - lo) / max(hi - lo, 1e-9) * nb).astype(int)
            m = (bi >= 0) & (bi < nb)
            sv = np.zeros((nb, T)); sa = np.zeros((nb, T))
            np.add.at(sv, (bi[m], ti[m]), v[m])
            np.add.at(sa, (bi[m], ti[m]), np.abs(v[m]))
            num += np.abs(sv).sum(); den += sa.sum()
        if den > 0:
            ratios.append(num / den)
    return round(float(np.mean(ratios)), 4), len(ratios)


if __name__ == "__main__":
    which = sys.argv[1:] or ["MM-Fi", "mRI", "BGT60TR13C", "mHomeGes"]
    loaders = {}
    if "MM-Fi" in which:
        loaders["MM-Fi"] = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if "mRI" in which:
        loaders["mRI"] = mri_windows()
    if "BGT60TR13C" in which:
        loaders["BGT60TR13C"] = infineon_recs_direct()
    if "mHomeGes" in which:
        loaders["mHomeGes"] = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    out = {"_definition": __doc__.strip(), "T": T, "full_datasets": True, "C_by_nb": {}}
    fn = os.path.join(DOCS, "c_grid_full.json")
    if os.path.exists(fn):
        out = json.load(open(fn))
    for name, insts in loaders.items():
        out["C_by_nb"][name] = {str(nb): cancellation(insts, nb) for nb in (16, 32, 64)}
        print(name, out["C_by_nb"][name], flush=True)
        json.dump(out, open(fn, "w"), indent=1)
    print("wrote docs/c_grid_full.json", flush=True)
