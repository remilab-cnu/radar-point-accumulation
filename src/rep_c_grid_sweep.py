"""Within-dataset test of the cancellation mechanism: move C by changing the map grid.

Section V-B establishes the sign-resolution recovery against C across datasets, which
confounds C with everything else that differs between corpora (sensor, task, subjects,
class count). This run varies C inside ONE dataset. The spatial grid sets how many points
share a cell, so it sets how much opposing Doppler mass cancels: coarser grids collide
more and lower C. Dataset, subjects, classes, folds, classifier, capacity, epoch count and
seeds are all held fixed; only the grid moves.

  nb = 16 / 32 / 64 spatial bins per axis (temporal bins fixed at T = 40)
  configurations: v_sum (scalar signed sum) and v_hist4 (four-bin sign-resolved histogram)

For each grid we also recompute C on the SAME geometry the maps use (fold-fitted ranges,
matching nb), so the abscissa and the ordinate of the resulting plot come from one binning
rather than two. NOTE ON INTERPRETATION: a coarser grid lowers C but also lowers spatial
resolution, so the across-grid comparison is not a C-only manipulation; within each grid
the paired difference is still matched (identical geometry on both configurations).

Protocol identical to rep_hist_ladder.py: SmallCNN width 32, lr 1e-3, batch 64,
5-fold kfold(seed=0), seeds {0,1,2}, 120 epochs, per-instance preds saved. nb=32 is rerun
here so all three points share one code path; it must reproduce hist_ladder_fw.json.

Out: docs/c_grid_sweep.json + docs/c_grid_sweep_preds.npz.  SMOKE=1 -> subset, ep4, seed 0.
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
GRIDS = (16,) if SMOKE else (16, 32, 64)
EDGES4 = [-99, -0.6, 0, 0.6, 99]


def build_configs(insts, ranges, nb):
    """Scalar signed sum and four-bin sign-resolved histogram, as rep_hist_ladder
    builds them (compose() already applies per-channel norm)."""
    stats = [cell_stats(t[0], CAXES, ranges, nb=nb) for t in insts]
    return {
        "v_sum": np.stack([compose(st, CAXES, ["sum"]) for st in stats]).astype(np.float32),
        "v_hist4": np.stack([compose(st, CAXES, ["hist"]) for st in stats]).astype(np.float32),
    }


def c_index(insts, ranges, nb):
    """Mean per-instance sign-coherence index on the maps' own geometry."""
    vals = []
    for t in insts:
        inst = t[0]
        f = inst["frame"].values.astype(float)
        f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
        ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
        v = inst["doppler"].values.astype(float)
        num = den = 0.0
        for ax in CAXES:
            lo, hi = ranges[ax]
            bi = np.floor((inst[ax].values - lo) / max(hi - lo, 1e-9) * nb).astype(int)
            m = (bi >= 0) & (bi < nb)
            sv = np.zeros((nb, T)); sa = np.zeros((nb, T))
            np.add.at(sv, (bi[m], ti[m]), v[m])
            np.add.at(sa, (bi[m], ti[m]), np.abs(v[m]))
            num += np.abs(sv).sum(); den += sa.sum()
        if den > 0:
            vals.append(num / den)
    return float(np.mean(vals)), len(vals)


if __name__ == "__main__":
    t0 = time.time()
    print(f"C-GRID-SWEEP SMOKE={SMOKE} grids={GRIDS} width={WIDTH} ep={BUDGET} seeds={SEEDS}",
          flush=True)
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE:
        idx = np.random.RandomState(0).permutation(len(mf))[:200]
        mf = [mf[i] for i in idx]
    subj = np.array([str(t[2]) for t in mf])
    folds = ([[str(s) for s in g] for g in kfold(subj, 5)]
             if not SMOKE else [sorted(set(subj))[:2]])
    y = np.array([t[1] for t in mf])

    results, preds = {}, {"SUBJ__MM-Fi": subj}
    cvals = {}
    for nb in GRIDS:
        # C on the dataset-fitted ranges at this grid, matching the maps' geometry
        cm, nc = c_index(mf, fit_ranges([t[0] for t in mf]), nb)
        cvals[nb] = {"C_mean_per_instance": round(cm, 4), "n_instances": nc}
        print(f"  nb={nb}: C = {cm:.4f} over {nc} instances", flush=True)

        accs_all = {}
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            if te.sum() == 0 or tr.sum() == 0:
                continue
            ranges = fit_ranges([t[0] for t, keep in zip(mf, tr) if keep])
            X = build_configs(mf, ranges, nb)
            for name, Xa in X.items():
                for s in SEEDS:
                    a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], 27,
                                                    epochs=BUDGET, seed=s, width=WIDTH)
                    accs_all.setdefault(name, ([], []))
                    accs_all[name][0].append(a); accs_all[name][1].append(ta)
                    preds[f"MM-Fi|{name}|nb{nb}|ep{BUDGET}|fold{fi}|seed{s}"] = \
                        np.stack([te_idx, yt, yp])
            del X
        for name, (accs, taccs) in accs_all.items():
            k = f"MM-Fi|{name}|nb{nb}|ep{BUDGET}"
            results[k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                          "std": round(float(np.std(accs)) * 100, 2),
                          "min_train_acc": round(float(np.min(taccs)), 4),
                          "mean_train_acc": round(float(np.mean(taccs)), 4),
                          "underfit": bool(np.min(taccs) < 0.95)}
            r = results[k]
            print(f"  {k:40s}: {r['acc']:6.2f} (+-{r['std']:.1f}) "
                  f"train={r['mean_train_acc']:.3f}{' UF' if r['underfit'] else ''}",
                  flush=True)
        print(f"  [nb={nb} done at {time.time()-t0:.0f}s]", flush=True)

    out = {"purpose": "within-dataset C manipulation via map grid (MM-Fi, 2026-08-26)",
           "protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budget": BUDGET, "grids": list(GRIDS), "T": T, "edges4": EDGES4,
                        "folds": "kfold(seed=0) 5-fold", "ranges": "fit_ranges per fold",
                        "smoke": SMOKE},
           "C_by_grid": cvals, "results": results,
           "caveat": "coarser grid lowers C but also lowers spatial resolution; "
                     "within-grid paired differences are geometry-matched"}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"c_grid_sweep{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"c_grid_sweep{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/c_grid_sweep{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
