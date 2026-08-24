"""Round 5 (final representation push + diagnosis):
 (1) last map levers on mHomeGes: 8-chunk planes; multiscale planes (16-up + 32)
 (2) ERROR ANALYSIS: per-class recall, best map (axis+planes+int_aug3) vs DeepSets_aug3
     -> what exactly drives the remaining ~7.6-pt gap
 (3) FIXED FramePointGRU (empty-bin saturation bug) rerun on both datasets
Decision after this round: if the gap persists, adopt the honest density-regime story
(maps win sparse/Infineon incl. parity bar; points win dense/mHomeGes) and finalize.
"""
import os, json, time
import numpy as np
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_variants import cell_stats, infineon_recs, CAXES, norm
from rep_round3 import kfold
from rep_round4 import crops3, axis_chs
from cnn import train_eval, train_eval_preds
from pointset_models import (build_point_tensors, build_frame_tensors,
                             DeepSets, FramePointGRU, train_eval_set, train_eval_set_preds)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
PARITY_BAR = {"mHomeGes": 82.12, "Infineon": 94.06}


def plane_maps_p(inst, ranges, chunks=4, hb=32, wb=40, upsample_to=None):
    f = inst["frame"].values.astype(float); f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    ci = np.floor((f - f0) / (f1 - f0) * (chunks - 1e-9)).astype(int)
    v = inst["doppler"].values.astype(float)
    chs = []
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        la, ha = ranges[a]; lb, hb_ = ranges[b]
        bi = np.floor((inst[a].values - la) / (ha - la) * hb).astype(int)
        bj = np.floor((inst[b].values - lb) / (hb_ - lb) * wb).astype(int)
        ok = (bi >= 0) & (bi < hb) & (bj >= 0) & (bj < wb)
        for c in range(chunks):
            m = ok & (ci == c)
            sv = np.zeros((hb, wb), np.float32); cnt = np.zeros((hb, wb), np.float32)
            np.add.at(sv, (bi[m], bj[m]), v[m]); np.add.at(cnt, (bi[m], bj[m]), 1.0)
            vm = np.divide(sv, cnt, out=np.zeros_like(sv), where=cnt > 0)
            for ch in (norm(vm), norm(cnt)):
                if upsample_to and hb != upsample_to:
                    ch = np.repeat(ch, upsample_to // hb, axis=0)
                chs.append(ch)
    return chs


def build_X5(insts, ranges, kind):
    X = []
    for inst, _, _ in insts:
        chs = axis_chs(inst, ranges, with_int=True)                      # 10ch
        if kind == "planes8":
            chs += plane_maps_p(inst, ranges, chunks=8)                  # +48
        elif kind == "multiscale":
            chs += plane_maps_p(inst, ranges, chunks=4)                  # +24 @32
            chs += plane_maps_p(inst, ranges, chunks=4, hb=16, upsample_to=32)  # +24 @16->32
        X.append(np.stack(chs))
    return np.stack(X)


def per_class(yt, yp, ncls):
    return [float((yp[yt == c] == c).mean()) if (yt == c).sum() else 0.0 for c in range(ncls)]


if __name__ == "__main__":
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    folds = kfold(np.array([t[2] for t in mh]), 5)
    ranges = fit_ranges([t[0] for t in mh])
    aug = [(c, lab, s) for inst, lab, s in mh for c in crops3(inst)]
    y = np.array([t[1] for t in mh] + [t[1] for t in aug])
    subj = np.array([t[2] for t in mh] + [t[2] for t in aug])
    is_aug = np.zeros(len(y), bool); is_aug[len(mh):] = True

    # ---- (1) final map levers ----
    res = {}
    for kind in ("planes8", "multiscale"):
        t0 = time.time()
        X = build_X5(mh + aug, ranges, kind)
        accs = []
        for te_s in folds:
            in_te = np.isin(subj, list(te_s)); tr = ~in_te; te = in_te & ~is_aug
            accs += [train_eval(X[tr], y[tr], X[te], y[te], 10, epochs=40, seed=s,
                                width=16, batch=128) for s in (0, 1)]
        res[f"axis+int+{kind}_aug3"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
        print(f"mHomeGes {kind:11s} ({X.shape[1]}ch): {res[f'axis+int+{kind}_aug3'][0]:6.2f}% "
              f"(+-{res[f'axis+int+{kind}_aug3'][1]:.1f}) vs parity-bar {res[f'axis+int+{kind}_aug3'][0]-PARITY_BAR['mHomeGes']:+.2f} [{time.time()-t0:.0f}s]", flush=True)
        del X

    # ---- (2) error analysis: best r4 map vs DeepSets_aug3, aggregated preds ----
    from rep_round4 import build_X as build_X4
    Xm = build_X4(mh + aug, ranges, with_int=True)
    Xp, Mp, yp_, sp = build_point_tensors(mh + aug, ranges)
    yt_m, pp_m, yt_d, pp_d = [], [], [], []
    for te_s in folds:
        in_te = np.isin(subj, list(te_s)); tr = ~in_te; te = in_te & ~is_aug
        _, a, b = train_eval_preds(Xm[tr], y[tr], Xm[te], y[te], 10, epochs=40, seed=0)
        yt_m.append(a); pp_m.append(b)
        _, c, d = train_eval_set_preds(DeepSets, Xp[tr], Mp[tr], yp_[tr], Xp[te], Mp[te], yp_[te], 10, 6, epochs=30, seed=0)
        yt_d.append(c); pp_d.append(d)
    yt_m, pp_m = np.concatenate(yt_m), np.concatenate(pp_m)
    yt_d, pp_d = np.concatenate(yt_d), np.concatenate(pp_d)
    rm, rd = per_class(yt_m, pp_m, 10), per_class(yt_d, pp_d, 10)
    classes = ["circle", "clap", "down", "knock", "lift", "pull", "push", "up", "yawn", "z"]
    print("\nPER-CLASS recall: map(axis+planes+int_aug3) vs DeepSets_aug3 (seed0, all folds)", flush=True)
    for i, cname in enumerate(classes):
        print(f"  {cname:8s} map={rm[i]*100:5.1f}  pts={rd[i]*100:5.1f}  diff={ (rm[i]-rd[i])*100:+6.1f}", flush=True)
    res["error_analysis"] = {"classes": classes, "map_recall": rm, "deepsets_recall": rd}

    # ---- (3) fixed GRU bars ----
    Xf, Mf, yf, sf = build_frame_tensors(mh, ranges)
    accs = []
    for te_s in folds:
        te = np.isin(sf, list(te_s)); tr = ~te
        accs += [train_eval_set(FramePointGRU, Xf[tr], Mf[tr], yf[tr], Xf[te], Mf[te], yf[te], 10, 5,
                                epochs=40, seed=s) for s in (0, 1)]
    res["FramePointGRU_FIXED"] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
    print(f"\nmHomeGes FramePointGRU_FIXED: {res['FramePointGRU_FIXED'][0]:6.2f}% (+-{res['FramePointGRU_FIXED'][1]:.1f})", flush=True)
    out["mHomeGes"] = res

    inf = infineon_recs()
    foldsI = kfold(np.array([t[2] for t in inf]), 4)
    rI = fit_ranges([t[0] for t in inf])
    Xf, Mf, yf, sf = build_frame_tensors(inf, rI)
    accs = []
    for te_s in foldsI:
        te = np.isin(sf, list(te_s)); tr = ~te
        accs += [train_eval_set(FramePointGRU, Xf[tr], Mf[tr], yf[tr], Xf[te], Mf[te], yf[te], 5, 5,
                                epochs=40, seed=s) for s in (0, 1)]
    out["Infineon"] = {"FramePointGRU_FIXED": (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)}
    print(f"Infineon FramePointGRU_FIXED: {out['Infineon']['FramePointGRU_FIXED'][0]:6.2f}%", flush=True)

    json.dump(out, open(os.path.join(DOCS, "repsweep_round5.json"), "w"), indent=1)
    print("\nwrote docs/repsweep_round5.json", flush=True)
