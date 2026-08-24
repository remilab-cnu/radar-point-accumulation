"""Round 3: close the mHomeGes gap to the point-input bar (DeepSets 79.9).

Hypothesis: DeepSets wins because it sees JOINT (x,y,z) structure; our (axis,time)
maps discard cross-axis correlation (64-bin didn't help -> not resolution).
Levers (representation/training only):
  planes4      : xy/xz/yz plane projections x 4 time chunks x [v-mean, occupancy]
                 = 24 joint-structure channels of (32,40)
  axis+planes  : round-2 anchor (mean/std per axis + V-T) + planes4  (31ch)
  width 48     : wider (still lightweight) SmallCNN
  AUG          : sliding-window crops ([0,.75],[.25,1]) of each TRAIN instance
                 (original paper's augmentation; test = originals only)
Bar (same folds): DeepSets full 79.87 (mHomeGes) / 92.18 (Infineon).
"""
import os, json, time
import numpy as np
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_variants import cell_stats, infineon_recs, CAXES, norm
from cnn import train_eval

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
HB, WB, CHUNKS = 32, 40, 4
BAR = {"mHomeGes": 79.87, "Infineon": 92.18}


def plane_maps(inst, ranges):
    f = inst["frame"].values.astype(float); f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    ci = np.floor((f - f0) / (f1 - f0) * (CHUNKS - 1e-9)).astype(int)
    v = inst["doppler"].values.astype(float)
    chs = []
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        la, ha = ranges[a]; lb, hb = ranges[b]
        bi = np.floor((inst[a].values - la) / (ha - la) * HB).astype(int)
        bj = np.floor((inst[b].values - lb) / (hb - lb) * WB).astype(int)
        ok = (bi >= 0) & (bi < HB) & (bj >= 0) & (bj < WB)
        for c in range(CHUNKS):
            m = ok & (ci == c)
            sv = np.zeros((HB, WB), np.float32); cnt = np.zeros((HB, WB), np.float32)
            np.add.at(sv, (bi[m], bj[m]), v[m]); np.add.at(cnt, (bi[m], bj[m]), 1.0)
            vm = np.divide(sv, cnt, out=np.zeros_like(sv), where=cnt > 0)
            chs += [norm(vm), norm(cnt)]
    return chs                                                     # 24 channels


def axis_chs(inst, ranges):
    st = cell_stats(inst, CAXES, ranges, nb=HB)
    out = [norm(st[ax][k]) for k in ("mean", "std") for ax in CAXES]
    out.append(norm(st["vt"]))
    return out                                                     # 7 channels


def build_X(insts, ranges, spec):
    X = []
    for inst, _, _ in insts:
        chs = []
        if "axis" in spec:
            chs += axis_chs(inst, ranges)
        if "planes" in spec:
            chs += plane_maps(inst, ranges)
        X.append(np.stack(chs))
    return np.stack(X)


def crops(inst):
    f = inst["frame"].values; f0, f1 = f.min(), f.max(); span = max(f1 - f0, 1)
    outs = []
    for lo, hi in ((0.0, 0.75), (0.25, 1.0)):
        sub = inst[(f >= f0 + lo * span) & (f <= f0 + hi * span)]
        if sub["frame"].nunique() >= 6:
            outs.append(sub.reset_index(drop=True))
    return outs


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


def run(name, orig, folds, ncls, variants, epochs):
    ranges = fit_ranges([t[0] for t in orig])
    y_o = np.array([t[1] for t in orig]); s_o = np.array([t[2] for t in orig])
    aug = []
    for inst, lab, s in orig:
        aug += [(c, lab, s) for c in crops(inst)]
    print(f"\n######## {name}: {len(orig)} orig + {len(aug)} aug | bar(DeepSets)={BAR[name]} ########", flush=True)
    res = {}
    for vname, cfg in variants.items():
        t0 = time.time()
        insts = orig + aug if cfg["aug"] else orig
        X = build_X(insts, ranges, cfg["spec"])
        y = np.concatenate([y_o, np.array([t[1] for t in aug])]) if cfg["aug"] else y_o
        subj = np.concatenate([s_o, np.array([t[2] for t in aug])]) if cfg["aug"] else s_o
        is_aug = np.zeros(len(y), bool); is_aug[len(orig):] = cfg["aug"]
        accs = []
        for te_s in folds:
            in_te = np.isin(subj, list(te_s))
            tr = ~in_te                                  # train subjects incl. their crops
            te = in_te & ~is_aug                         # test = originals only
            accs += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs,
                                seed=s, width=cfg["width"]) for s in cfg["seeds"]]
        res[vname] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
        d = res[vname][0] - BAR[name]
        print(f"  {vname:22s} ({X.shape[1]:2d}ch,w{cfg['width']},aug={int(cfg['aug'])}): "
              f"{res[vname][0]:6.2f}% (+-{res[vname][1]:.1f})  vs bar {d:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
        del X
    return res


V_MH = {
    "axis(anchor)":       dict(spec=["axis"], aug=False, width=16, seeds=(0, 1, 2)),
    "planes4":            dict(spec=["planes"], aug=False, width=16, seeds=(0, 1, 2)),
    "axis+planes":        dict(spec=["axis", "planes"], aug=False, width=16, seeds=(0, 1, 2)),
    "axis+planes_w48":    dict(spec=["axis", "planes"], aug=False, width=48, seeds=(0, 1, 2)),
    "axis+planes_aug":    dict(spec=["axis", "planes"], aug=True, width=16, seeds=(0, 1)),
    "axis+planes_w48aug": dict(spec=["axis", "planes"], aug=True, width=48, seeds=(0, 1)),
}
V_INF = {
    "planes4":            dict(spec=["planes"], aug=False, width=16, seeds=(0, 1, 2)),
    "axis+planes":        dict(spec=["axis", "planes"], aug=False, width=16, seeds=(0, 1, 2)),
    "axis+planes_w48":    dict(spec=["axis", "planes"], aug=False, width=48, seeds=(0, 1, 2)),
}

if __name__ == "__main__":
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    out["mHomeGes"] = run("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10, V_MH, epochs=30)
    inf = infineon_recs()
    out["Infineon"] = run("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5, V_INF, epochs=40)
    json.dump(out, open(os.path.join(DOCS, "repsweep_round3.json"), "w"), indent=1)
    print("\nwrote docs/repsweep_round3.json", flush=True)
    print("NOTE: aug variants use the original paper's sliding-window trick (train-only);"
          " the DeepSets bar is un-augmented — report with/without aug separately.", flush=True)
