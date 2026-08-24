"""E3 (major-revision campaign, 2026-08-18): complete the map-domain attribution ladder
for the sign-resolved histogram on MM-Fi.

Review finding R3 (internal inconsistency): the manuscript attributed
hist4 - hist4_vshuffled to the "velocity distribution", but the within-instance shuffle
PRESERVES the instance-level distribution and destroys the velocity-to-point (hence
velocity-to-cell) association - by the paper's own Section III-D definitions that contrast
is the ASSOCIATION component. rep_converge.py's docstring carried the same mislabel.
This run adds the missing rungs so the map-domain ladder mirrors the point-domain one:

  v_sum               scalar signed sum (the failing arm)
  v_hist4             4-bin sign-resolved histogram (the recovery arm)
  v_hist4_vshuffled   within-instance doppler permutation before binning
                      -> destroys association, keeps instance distribution + count
  v_hist4_vcross      doppler values resampled from a donor instance before binning  (NEW)
                      -> destroys association AND distribution, keeps count + marginals
  cnt4rand            each point assigned to one of 4 channels uniformly at random   (NEW)
                      -> same width, same per-cell total count, zero velocity info
                      (the dimensionality/width control)
  v_hist2 / v_hist8   bin-count sensitivity of the recovery arm                      (NEW)

Ladder readout: hist4 - vshuffled = association; vshuffled - vcross = distribution;
vcross - cnt4rand = width/nuisance. One shuffle realization per instance (seed = instance
index), the rep_converge convention.

MM-Fi only: the one whole-body set whose histogram arms all meet the 0.95 training
criterion (mRI's shuffle arm trains to 0.497 and is not evaluable; documented, not rerun).
Frozen: SmallCNN width 32, lr 1e-3, batch 64, 5-fold kfold(seed=0) == converge_body folds,
seeds {0,1,2}, budget 120, per-instance preds saved.

Out: docs/hist_ladder.json + docs/hist_ladder_preds.npz.  SMOKE=1 -> subset, ep4, seed 0.
"""
import os, json, time
import numpy as np
from rep_variants import cell_stats, compose, norm, CAXES, kfold, T
from spectra_dataset import fit_ranges, mmfi_instances
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
BUDGET = 4 if SMOKE else 120
WIDTH = 8 if SMOKE else 32
NB = 32
EDGES4 = [-99, -0.6, 0, 0.6, 99]
EDGES2 = [-99, 0, 99]
EDGES8 = [-99, -1.2, -0.6, -0.2, 0, 0.2, 0.6, 1.2, 99]


def axis_bins(inst, ranges):
    """Shared spatial/temporal binning; returns per-axis (b, valid-mask) plus time index."""
    f = inst["frame"].values.astype(float)
    f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
    out = {}
    for ax in CAXES:
        lo, hi = ranges[ax]
        bi = np.floor((inst[ax].values - lo) / (hi - lo) * NB).astype(int)
        out[ax] = (bi, (bi >= 0) & (bi < NB))
    return out, ti


def hist_channels(inst, ranges, edges, values):
    """Per-axis sign histogram of `values` with `edges`; norm() per channel like compose."""
    ab, ti = axis_bins(inst, ranges)
    ch = []
    for ax in CAXES:
        bi, m = ab[ax]
        b, t, vv = bi[m], ti[m], values[m]
        h = np.zeros((len(edges) - 1, NB, T), np.float32)
        for k in range(len(edges) - 1):
            sel = (vv > edges[k]) & (vv <= edges[k + 1])
            np.add.at(h[k], (b[sel], t[sel]), 1.0)
        ch += [norm(h[k]) for k in range(len(edges) - 1)]
    return np.stack(ch)


def cnt4rand_channels(inst, ranges, rng):
    """4 channels, each point assigned to one channel uniformly at random: width- and
    count-matched to hist4, velocity-free."""
    ab, ti = axis_bins(inst, ranges)
    assign = rng.integers(0, 4, size=len(inst))
    ch = []
    for ax in CAXES:
        bi, m = ab[ax]
        b, t, k = bi[m], ti[m], assign[m]
        h = np.zeros((4, NB, T), np.float32)
        np.add.at(h, (k, b, t), 1.0)
        ch += [norm(h[j]) for j in range(4)]
    return np.stack(ch)


def build_arms(insts, ranges):
    n = len(insts)
    stats = [cell_stats(t[0], CAXES, ranges, nb=NB) for t in insts]
    X = {"v_sum": np.stack([compose(st, CAXES, ["sum"]) for st in stats]).astype(np.float32),
         "v_hist4": np.stack([compose(st, CAXES, ["hist"]) for st in stats]).astype(np.float32)}
    del stats
    hs, hc, cr, h2, h8 = [], [], [], [], []
    for i, (inst, _, _) in enumerate(insts):
        v = inst["doppler"].values.astype(float)
        vs = np.random.RandomState(i).permutation(v)                    # within-instance
        rngc = np.random.default_rng(20_000 + i)                        # cross-instance donor
        donor = insts[(i + 1 + int(rngc.integers(0, n - 1))) % n][0]
        vc = rngc.choice(donor["doppler"].values.astype(float), size=len(v), replace=True)
        hs.append(hist_channels(inst, ranges, EDGES4, vs))
        cr.append(hist_channels(inst, ranges, EDGES4, vc))
        hc.append(cnt4rand_channels(inst, ranges, np.random.default_rng(40_000 + i)))
        h2.append(hist_channels(inst, ranges, EDGES2, v))
        h8.append(hist_channels(inst, ranges, EDGES8, v))
    X["v_hist4_vshuffled"] = np.stack(hs).astype(np.float32)
    X["v_hist4_vcross"] = np.stack(cr).astype(np.float32)
    X["cnt4rand"] = np.stack(hc).astype(np.float32)
    X["v_hist2"] = np.stack(h2).astype(np.float32)
    X["v_hist8"] = np.stack(h8).astype(np.float32)
    return X


if __name__ == "__main__":
    t0 = time.time()
    print(f"HIST-LADDER SMOKE={SMOKE} width={WIDTH} ep={BUDGET} seeds={SEEDS}", flush=True)
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE:
        idx = np.random.RandomState(0).permutation(len(mf))[:200]
        mf = [mf[i] for i in idx]
    subj = np.array([str(t[2]) for t in mf])
    folds = ([[str(s) for s in g] for g in kfold(subj, 5)]
             if not SMOKE else [sorted(set(subj))[:2]])
    y = np.array([t[1] for t in mf])
    # Fold-wise leak-free form (2026-08-19): ranges fit on training subjects per fold.
    # The dataset-global run is archived as hist_ladder.json.
    results, preds = {}, {"SUBJ__MM-Fi": subj}
    accs_all = {}
    for fi, te_s in enumerate(folds):
        te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ranges = fit_ranges([t[0] for t, keep in zip(mf, tr) if keep])
        X = build_arms(mf, ranges)
        for name, Xa in X.items():
            for s in SEEDS:
                a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], 27,
                                                epochs=BUDGET, seed=s, width=WIDTH)
                accs_all.setdefault(name, ([], []))
                accs_all[name][0].append(a); accs_all[name][1].append(ta)
                preds[f"MM-Fi|{name}|ep{BUDGET}|fold{fi}|seed{s}"] = np.stack([te_idx, yt, yp])
        del X
    for name, (accs, taccs) in accs_all.items():
        k = f"MM-Fi|{name}|ep{BUDGET}"
        results[k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                      "std": round(float(np.std(accs)) * 100, 2),
                      "min_train_acc": round(float(np.min(taccs)), 4),
                      "mean_train_acc": round(float(np.mean(taccs)), 4),
                      "underfit": bool(np.min(taccs) < 0.95)}
        r = results[k]
        print(f"  {k:38s}: {r['acc']:6.2f} (+-{r['std']:.1f}) train={r['mean_train_acc']:.3f}"
              f"{' UF' if r['underfit'] else ''}", flush=True)

    out = {"purpose": "E3 major-revision: complete map-domain sign-histogram ladder on MM-Fi (2026-08-18)",
           "protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budget": BUDGET, "nb": NB, "edges4": EDGES4, "edges2": EDGES2,
                        "edges8": EDGES8, "folds": "kfold(seed=0) 5-fold", "smoke": SMOKE},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"hist_ladder_fw{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"hist_ladder_fw{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/hist_ladder_fw{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
