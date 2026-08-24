"""GO/NO-GO GATE: 3-arm cross-subject comparison on OPEN datasets (M-Gesture dropped).

Arms (same fixed CNN, same splits, only the accumulated per-point quantity changes):
  velocity  = XTD/YTD/ZTD  (proposed)
  intensity = XTA/YTA/ZTA  (amplitude control, = prior-art axis image)
  occupancy = XTO/YTO/ZTO  (GEOMETRY control: per-bin point count)  <- the decisive control

Decision: the paper's thesis ("accumulate velocity, not geometry") survives only if
velocity beats BOTH intensity AND occupancy cross-subject on gestures. Reports
per-arm accuracy + macro-F1, gaps, and a fold-level bootstrap 95% CI on the gaps.
"""
import os, sys, time
import numpy as np
from sklearn.metrics import f1_score
from preprocess import SpecConfig, make_channels, max_norm
from spectra_dataset import fit_ranges, mhomeges_instances, mmfi_instances
from cnn import train_eval_preds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SEEDS = [0, 1, 2]
ARMS = ["velocity", "intensity", "occupancy"]
ARM_KEYS = {"velocity": ["XTD", "YTD", "ZTD"],
            "intensity": ["XTA", "YTA", "ZTA"],
            "occupancy": ["XTO", "YTO", "ZTO"]}


def build_three_arms(insts_labeled, base_cfg, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["Xv"], z["Xa"], z["Xo"], z["y"], z["subj"]
    ranges = fit_ranges([t[0] for t in insts_labeled])
    cfg = SpecConfig(n_bins=base_cfg.n_bins, n_frames=base_cfg.n_frames, ranges=ranges)
    Xv, Xa, Xo, y, subj = [], [], [], [], []
    for inst, lab, s in insts_labeled:
        ch = make_channels(inst, cfg)
        Xv.append(np.stack([max_norm(ch[k]) for k in ARM_KEYS["velocity"]], 0).astype(np.float32))
        Xa.append(np.stack([max_norm(ch[k]) for k in ARM_KEYS["intensity"]], 0).astype(np.float32))
        Xo.append(np.stack([max_norm(ch[k]) for k in ARM_KEYS["occupancy"]], 0).astype(np.float32))
        y.append(lab); subj.append(s)
    Xv, Xa, Xo = np.stack(Xv), np.stack(Xa), np.stack(Xo)
    y = np.array(y, np.int64); subj = np.array(subj)
    if cache:
        np.savez_compressed(cache, Xv=Xv, Xa=Xa, Xo=Xo, y=y, subj=subj)
    return Xv, Xa, Xo, y, subj


def kfold(subjects, k, seed=0):
    subs = sorted(set(subjects)); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(subs), k)]


def eval_arm(X, y, subj, folds, n_cls, epochs):
    """Return per-fold (acc, macroF1) averaged over seeds."""
    accs, f1s = [], []
    for test_subs in folds:
        te = np.isin(subj, list(test_subs)); tr = ~te
        a_s, f_s = [], []
        for s in SEEDS:
            a, yt, yp = train_eval_preds(X[tr], y[tr], X[te], y[te], n_cls, epochs=epochs, seed=s)
            a_s.append(a); f_s.append(f1_score(yt, yp, average="macro", zero_division=0))
        accs.append(np.mean(a_s)); f1s.append(np.mean(f_s))
    return np.array(accs), np.array(f1s)


def bootstrap_ci(fold_gaps, n=4000, seed=0):
    g = np.asarray(fold_gaps); m = len(g)
    if m < 2:                                  # single fold -> no interval (audit: was zero-width)
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    boots = [g[rng.randint(0, m, m)].mean() for _ in range(n)]
    # NOTE: with ~5 folds this is an indicative fold-spread interval, not a rigorous 95% CI;
    # back the gate with a fold-level sign test (see run_gate reporting).
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_dataset(name, insts, folds, n_cls, epochs, cache):
    t0 = time.time()
    Xv, Xa, Xo, y, subj = build_three_arms(insts, SpecConfig(32, 40), cache)
    print(f"\n############ {name} ############")
    print(f"built {len(y)} instances, {len(set(subj.tolist()))} subjects, {n_cls} classes in {time.time()-t0:.0f}s")
    arm_acc = {}
    for arm, X in (("velocity", Xv), ("intensity", Xa), ("occupancy", Xo)):
        acc, f1 = eval_arm(X, y, subj, folds, n_cls, epochs)
        arm_acc[arm] = acc
        print(f"  {arm:10s}: acc={acc.mean()*100:5.2f}%  macroF1={f1.mean()*100:5.2f}%  "
              f"per-fold={['%.1f'%(a*100) for a in acc]}")
    for ctrl in ("intensity", "occupancy"):
        gap = arm_acc["velocity"] - arm_acc[ctrl]
        lo, hi = bootstrap_ci(gap)
        print(f"  GAP velocity-{ctrl:9s}: +{gap.mean()*100:5.2f} pts  "
              f"95%CI[{lo*100:+.1f},{hi*100:+.1f}]  (chance={100/n_cls:.1f}%)")
    return arm_acc


# ---- mHomeGes: sole large-N gesture positive (M-Gesture dropped) ----
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
run_dataset("mHomeGes (25 subj, 10 gestures)", mh, kfold([s for _, _, s in mh], 5),
            n_cls=10, epochs=20, cache=os.path.join(DATA, "mhomeges_gate.npz"))

# ---- MM-Fi: whole-body HAR boundary ----
mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
S2_TEST = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
run_dataset("MM-Fi (40 subj, 27 actions, S2)", mf, [S2_TEST],
            n_cls=27, epochs=40, cache=os.path.join(DATA, "mmfi_gate.npz"))
