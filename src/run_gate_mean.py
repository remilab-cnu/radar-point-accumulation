"""M1 FIX / decisive control: does MEAN-accumulated velocity (decoupled from point
density) still beat the occupancy (geometry) control cross-subject? If yes, the
motion-vs-geometry dissociation is clean; the SUM confound (velocity_sum superset
occupancy) is not what drives the +13.

Arms (identical fixed CNN, identical cross-subject splits):
  vel_mean  = per-bin MEAN radial velocity   (pure motion, density-decoupled)
  vel_sum   = per-bin SUM radial velocity     (original; density-coupled, for reference)
  int_mean  = per-bin MEAN intensity          (density-decoupled amplitude)
  occupancy = per-bin point count             (pure geometry / density)
"""
import os, numpy as np
from preprocess import SpecConfig, build_spectrum, max_norm
from spectra_dataset import fit_ranges, mhomeges_instances, mmfi_instances
from cnn import train_eval

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
AXES = ["x", "y", "z"]
ARMS = {
    "vel_mean":  ("doppler", "mean"),
    "vel_sum":   ("doppler", "sum"),
    "int_mean":  ("intensity", "mean"),
    "occupancy": ("count", "sum"),
}


def build(insts_labeled):
    insts = [t[0] for t in insts_labeled]
    cfg = SpecConfig(32, 40, fit_ranges(insts))
    out = {k: [] for k in ARMS}
    y, subj = [], []
    for inst, lab, s in insts_labeled:
        for k, (val, agg) in ARMS.items():
            out[k].append(np.stack([max_norm(build_spectrum(inst, a, val, cfg, agg=agg)) for a in AXES]).astype(np.float32))
        y.append(lab); subj.append(s)
    return {k: np.stack(v) for k, v in out.items()}, np.array(y, np.int64), np.array(subj)


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


def run(name, insts, folds, ncls, epochs, seeds=(0, 1, 2)):
    arms, y, subj = build(insts)
    print(f"\n#### {name}: {len(y)} inst, {len(set(subj.tolist()))} subj, {ncls} cls", flush=True)
    acc = {}
    for k, X in arms.items():
        a = []
        for te_s in folds:
            te = np.isin(subj, list(te_s)); tr = ~te
            a += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs, seed=s) for s in seeds]
        acc[k] = float(np.mean(a))
        print(f"   {k:10s}: {acc[k]*100:6.2f}%", flush=True)
    print(f"   >> vel_mean - occupancy = {(acc['vel_mean']-acc['occupancy'])*100:+.2f}  "
          f"(clean motion-vs-geometry); vel_mean - int_mean = {(acc['vel_mean']-acc['int_mean'])*100:+.2f}", flush=True)
    return acc


mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
run("mHomeGes (gesture, 5-fold)", mh, kfold(np.array([t[2] for t in mh]), 5), 10, 20)

mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
run("MM-Fi (HAR, S2)", mf, [S2], 27, 40)
