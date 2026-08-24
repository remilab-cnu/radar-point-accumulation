"""Round 4 (mHomeGes-focused): close the remaining -7.7 to the point-input bar.

Levers: 3-crop sliding-window aug (original paper's trick, train-only), intensity
channels added to the composite (DeepSets 'full' uses A; our maps didn't), batch 128.
FAIRNESS: also re-run the DeepSets bar WITH the same aug (aug-parity bar a reviewer
will demand) and the grad-clip-FIXED FramePointGRU as a secondary bar.
Infineon: transfer check of the composite (winner there already: axis+planes_w48 96.7).
"""
import os, json, time
import numpy as np
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_variants import cell_stats, infineon_recs, CAXES, norm
from rep_round3 import plane_maps, kfold
from cnn import train_eval
from pointset_models import (build_point_tensors, build_frame_tensors,
                             DeepSets, FramePointGRU, train_eval_set)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
BAR = {"mHomeGes": 79.87, "Infineon": 92.18}


def axis_chs(inst, ranges, with_int=False):
    st = cell_stats(inst, CAXES, ranges, nb=32)
    keys = ("mean", "std") + (("int_mean",) if with_int else ())
    out = [norm(st[ax][k]) for k in keys for ax in CAXES]
    out.append(norm(st["vt"]))
    return out


def build_X(insts, ranges, with_int):
    X = []
    for inst, _, _ in insts:
        X.append(np.stack(axis_chs(inst, ranges, with_int) + plane_maps(inst, ranges)))
    return np.stack(X)


def crops3(inst):
    f = inst["frame"].values; f0, f1 = f.min(), f.max(); span = max(f1 - f0, 1)
    outs = []
    for lo, hi in ((0.0, 0.7), (0.15, 0.85), (0.3, 1.0)):
        sub = inst[(f >= f0 + lo * span) & (f <= f0 + hi * span)]
        if sub["frame"].nunique() >= 6:
            outs.append(sub.reset_index(drop=True))
    return outs


def run_maps(name, orig, folds, ncls, epochs=40, seeds=(0, 1)):
    ranges = fit_ranges([t[0] for t in orig])
    aug = [(c, lab, s) for inst, lab, s in orig for c in crops3(inst)]
    y = np.array([t[1] for t in orig] + [t[1] for t in aug])
    subj = np.array([t[2] for t in orig] + [t[2] for t in aug])
    is_aug = np.zeros(len(y), bool); is_aug[len(orig):] = True
    print(f"\n######## {name} maps: {len(orig)} orig + {len(aug)} aug ########", flush=True)
    res = {}
    for vname, with_int in (("axis+planes_aug3", False), ("axis+planes+int_aug3", True)):
        t0 = time.time()
        X = build_X(orig + aug, ranges, with_int)
        accs = []
        for te_s in folds:
            in_te = np.isin(subj, list(te_s))
            tr = ~in_te; te = in_te & ~is_aug
            accs += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs,
                                seed=s, width=16, batch=128) for s in seeds]
        res[vname] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
        print(f"  {vname:22s} ({X.shape[1]}ch): {res[vname][0]:6.2f}% (+-{res[vname][1]:.1f})"
              f"  vs bar {res[vname][0]-BAR[name]:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
        del X
    return res, (orig, aug, y, subj, is_aug, ranges)


def run_point_bars(name, packed, folds, ncls, seeds=(0, 1, 2)):
    orig, aug, y, subj, is_aug, ranges = packed
    print(f"\n######## {name} point bars ########", flush=True)
    res = {}
    # DeepSets WITH the same aug (parity bar)
    Xp, Mp, yp, sp = build_point_tensors(orig + aug, ranges)
    accs = []
    for te_s in folds:
        in_te = np.isin(sp, list(te_s))
        tr = ~in_te; te = in_te & ~is_aug
        accs += [train_eval_set(DeepSets, Xp[tr], Mp[tr], yp[tr], Xp[te], Mp[te], yp[te],
                                ncls, 6, epochs=30, seed=s) for s in seeds]
    res["DeepSets_full_aug3"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
    print(f"  DeepSets_full_aug3    : {res['DeepSets_full_aug3'][0]:6.2f}% (+-{res['DeepSets_full_aug3'][1]:.1f})"
          f"  (un-aug bar was {BAR[name]})", flush=True)
    del Xp, Mp
    # fixed FramePointGRU (grad-clip, lower lr), no aug — secondary bar validity
    Xf, Mf, yf, sf = build_frame_tensors(orig, ranges)
    accs = []
    for te_s in folds:
        te = np.isin(sf, list(te_s)); tr = ~te
        accs += [train_eval_set(FramePointGRU, Xf[tr], Mf[tr], yf[tr], Xf[te], Mf[te], yf[te],
                                ncls, 5, epochs=40, lr=3e-4, seed=s) for s in (0, 1)]
    res["FramePointGRU_fixed"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
    print(f"  FramePointGRU_fixed   : {res['FramePointGRU_fixed'][0]:6.2f}% (+-{res['FramePointGRU_fixed'][1]:.1f})", flush=True)
    return res


if __name__ == "__main__":
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    folds = kfold(np.array([t[2] for t in mh]), 5)
    m, packed = run_maps("mHomeGes", mh, folds, 10)
    out["mHomeGes"] = m
    out["mHomeGes"].update(run_point_bars("mHomeGes", packed, folds, 10))

    inf = infineon_recs()
    foldsI = kfold(np.array([t[2] for t in inf]), 4)
    mi, packedI = run_maps("Infineon", inf, foldsI, 5, epochs=40, seeds=(0, 1, 2))
    out["Infineon"] = mi
    out["Infineon"].update(run_point_bars("Infineon", packedI, foldsI, 5))

    json.dump(out, open(os.path.join(DOCS, "repsweep_round4.json"), "w"), indent=1)
    print("\nwrote docs/repsweep_round4.json", flush=True)
