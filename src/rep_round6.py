"""Round 6: FAITHFUL original-paper augmentation.

Per the original manuscript: spectrogram on a FIXED native-time axis (absolute frames,
no per-instance stretching), then a sliding WINDOW over the map's time axis -> phase
shift only (no time-warp). Multiplier reduced (PI: original x10 was for tiny data):
  mHomeGes: T_full=56 native frames, W=40, t0 in {0,5,10,16}  (x4 train windows)
  Infineon: T_full=20, W=16, t0 in {0,2,4}                    (x3 train windows)
Test = center window of the original instance only (deterministic single view).
Axis maps are built ONCE at T_full and sliced per window (exactly map-domain sliding);
plane maps are recomputed from the window's points (they have no time axis).
Parity: DeepSets bar gets the SAME windows (points within window, t normalized in-window).
Native axis also preserves absolute duration/speed (fixes the earlier stretch critique).
"""
import os, json, time
import numpy as np
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_variants import infineon_recs, norm, CAXES
from rep_round3 import kfold
from rep_round5 import plane_maps_p
from cnn import train_eval
from pointset_models import DeepSets, train_eval_set

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
NB = 32
VT_MAX = 3.0


def native_axis_maps(inst, ranges, T_full, with_int, margin=0):
    """Channels on the NATIVE time axis (frame offsets), shape (C, NB, T_full).
    C = 3*mean + 3*std (+3*int_mean) + 1*VT."""
    f = inst["frame"].values.astype(float)
    ti = (f - f.min()).astype(int) + margin   # margin: reproduce the original recordings' pre/post-gesture margin
    keep = ti < T_full
    v = inst["doppler"].values.astype(float)
    a = np.abs(inst["intensity"].values.astype(float))
    chs_mean, chs_std, chs_int = [], [], []
    for ax in CAXES:
        lo, hi = ranges[ax]
        bi = np.floor((inst[ax].values - lo) / (hi - lo) * NB).astype(int)
        m = keep & (bi >= 0) & (bi < NB)
        b, t, vv, aa = bi[m], ti[m], v[m], a[m]
        S = lambda: np.zeros((NB, T_full), np.float32)
        sv, s2, sa, cnt = S(), S(), S(), S()
        np.add.at(sv, (b, t), vv); np.add.at(s2, (b, t), vv ** 2)
        np.add.at(sa, (b, t), aa); np.add.at(cnt, (b, t), 1.0)
        div = lambda x: np.divide(x, cnt, out=np.zeros_like(x), where=cnt > 0)
        mean = div(sv)
        chs_mean.append(mean)
        chs_std.append(np.sqrt(np.clip(div(s2) - mean ** 2, 0, None)))
        chs_int.append(div(sa))
    vt = np.zeros((NB, T_full), np.float32)
    vi = np.floor((v + VT_MAX) / (2 * VT_MAX) * (NB - 1e-9)).astype(int)
    mv = keep & (vi >= 0) & (vi < NB)
    np.add.at(vt, (vi[mv], ti[mv]), 1.0)
    chs = chs_mean + chs_std + (chs_int if with_int else []) + [vt]
    return np.stack(chs)                                   # un-normalized; slice then norm


def window_stack(inst, full_maps, ranges, t0, W, with_int, margin=0):
    """One training/test sample: sliced native axis maps + window plane maps."""
    sl = full_maps[:, :, t0:t0 + W]
    axis = np.stack([norm(c) for c in sl])
    f = inst["frame"].values.astype(float); off = f - f.min() + margin
    sub = inst[(off >= t0) & (off < t0 + W)]
    if len(sub) < 3:
        planes = np.zeros((24, NB, W), np.float32)
    else:
        planes = np.stack(plane_maps_p(sub.reset_index(drop=True), ranges, chunks=4, hb=NB, wb=W))
    return np.concatenate([axis, planes], 0)


def build_windowed(insts, ranges, T_full, W, t0s, with_int, margin=0):
    """Returns X (N*k, C, NB, W), y, subj, win_id, is_center for all windows."""
    X, y, subj, is_center = [], [], [], []
    center = t0s[len(t0s) // 2]
    for inst, lab, s in insts:
        fm = native_axis_maps(inst, ranges, T_full, with_int, margin=margin)
        off = inst["frame"].values - inst["frame"].values.min() + margin
        for t0 in t0s:
            n_in = int(((off >= t0) & (off < t0 + W)).sum())
            # variable-length instances (original paper had fixed-length recordings):
            # skip degenerate off-center windows that see almost none of the gesture
            if n_in < 3 or (t0 != center and n_in < max(10, 0.15 * len(inst))):
                continue
            X.append(window_stack(inst, fm, ranges, t0, W, with_int, margin=margin))
            y.append(lab); subj.append(s); is_center.append(t0 == center)
    return (np.stack(X), np.array(y, np.int64), np.array(subj), np.array(is_center))


def points_windowed(insts, ranges, T_full, W, t0s, n_points=384, margin=0):
    """Parity DeepSets tensors for the same windows."""
    center = t0s[len(t0s) // 2]
    Xs, Ms, ys, ss, ic = [], [], [], [], []
    for k, (inst, lab, s) in enumerate(insts):
        f = inst["frame"].values.astype(float); off = f - f.min() + margin
        for t0 in t0s:
            sub = inst[(off >= t0) & (off < t0 + W)]
            n = len(sub)
            if n < 3 or (t0 != center and n < max(10, 0.15 * len(inst))):
                continue
            out = np.zeros((n_points, 6), np.float32); m = np.zeros((n_points,), bool)
            if n >= 3:
                rng = np.random.RandomState(k + t0)
                idx = rng.choice(n, size=min(n, n_points), replace=False)
                sub2 = sub.iloc[idx]
                a = np.abs(sub2["intensity"].values); amax = max(a.max(), 1e-9)
                out[:len(idx), 0] = 2 * (sub2["x"].values - ranges["x"][0]) / (ranges["x"][1] - ranges["x"][0]) - 1
                out[:len(idx), 1] = 2 * (sub2["y"].values - ranges["y"][0]) / (ranges["y"][1] - ranges["y"][0]) - 1
                out[:len(idx), 2] = 2 * (sub2["z"].values - ranges["z"][0]) / (ranges["z"][1] - ranges["z"][0]) - 1
                out[:len(idx), 3] = sub2["doppler"].values / 2.0
                out[:len(idx), 4] = a / amax
                out[:len(idx), 5] = (sub2["frame"].values - sub2["frame"].min()) / max(W, 1)
                m[:len(idx)] = True
            Xs.append(out); Ms.append(m); ys.append(lab); ss.append(s); ic.append(t0 == center)
    return np.stack(Xs), np.stack(Ms), np.array(ys, np.int64), np.array(ss), np.array(ic)


def eval_folds(train_fn, folds, subj, is_center, seeds):
    accs = []
    for te_s in folds:
        in_te = np.isin(subj, list(te_s))
        tr = ~in_te                       # all windows of train subjects
        te = in_te & is_center            # center window only for test
        accs += [train_fn(tr, te, s) for s in seeds]
    return float(np.mean(accs)) * 100, float(np.std(accs)) * 100


def run(name, insts, folds, ncls, T_full, W, t0s, with_int, epochs=40, seeds=(0, 1), margin=0):
    print(f"\n######## {name}: faithful windowing T_full={T_full} W={W} t0s={t0s} ########", flush=True)
    ranges = fit_ranges([t[0] for t in insts])
    res = {}
    t0_ = time.time()
    X, y, subj, ic = build_windowed(insts, ranges, T_full, W, t0s, with_int, margin=margin)
    print(f"  maps built {X.shape} in {time.time()-t0_:.0f}s", flush=True)
    fn = lambda tr, te, s: train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs,
                                      seed=s, width=16, batch=128)
    res["map_faithful_winaug"] = eval_folds(fn, folds, subj, ic, seeds)
    print(f"  map_faithful_winaug   : {res['map_faithful_winaug'][0]:6.2f}% (+-{res['map_faithful_winaug'][1]:.1f})", flush=True)
    del X
    Xp, Mp, yp, sp, icp = points_windowed(insts, ranges, T_full, W, t0s, margin=margin)
    fnp = lambda tr, te, s: train_eval_set(DeepSets, Xp[tr], Mp[tr], yp[tr], Xp[te], Mp[te], yp[te],
                                           ncls, 6, epochs=30, seed=s)
    res["DeepSets_faithful_winaug"] = eval_folds(fnp, folds, sp, icp, seeds)
    print(f"  DeepSets_faithful_win : {res['DeepSets_faithful_winaug'][0]:6.2f}% (+-{res['DeepSets_faithful_winaug'][1]:.1f})", flush=True)
    return res


if __name__ == "__main__":
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    folds = kfold(np.array([t[2] for t in mh]), 5)
    out["mHomeGes"] = run("mHomeGes", mh, folds, 10, T_full=56, W=40, t0s=(0, 4, 8, 12, 16), with_int=True, margin=8)
    inf = infineon_recs()
    foldsI = kfold(np.array([t[2] for t in inf]), 4)
    out["Infineon"] = run("Infineon", inf, foldsI, 5, T_full=20, W=16, t0s=(0, 2, 4), with_int=False,
                          seeds=(0, 1, 2))
    json.dump(out, open(os.path.join(DOCS, "repsweep_round6b.json"), "w"), indent=1)
    print("\nwrote docs/repsweep_round6b.json", flush=True)
    print("compare: prev best mH map 74.5 / parity bar 82.1 | inf map 96.3 / bar 94.1", flush=True)
