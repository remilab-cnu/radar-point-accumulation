"""E1 (major-revision campaign, 2026-08-18): fold-wise preprocessing refit + point ladder.

Two review findings drive this rerun:
  R1  fit_ranges() was computed on the FULL instance list, so held-out subjects entered
      the 1-99 percentile spatial ranges. Label-free and arm-symmetric, but a strict
      subject-disjointness violation. Here the ranges are fit PER FOLD on the training
      subjects only; test tensors are built with the train-fitted ranges.
  R5  the ladder had matched-width shuffle controls for velocity only. Intensity gets the
      symmetric treatment (A-inshuffle / A-crossshuffle) so the conditional contribution
      of both physical quantities is measured the same way.
Also: seeds 3 -> 5 on every arm (review: three seed clusters are too few to bootstrap),
and the [x,y,z,t] arm is named geometry (it is the coordinate support, not an occupancy
accumulation - the map-domain occupancy arm keeps its name).

Frozen otherwise: DeepSets, lr 1e-3, batch 64, kfold(subj,5,seed=0) == p1/converge folds,
UF criterion 0.95, per-instance preds saved. Budgets: the primary near-converged points
only (mHomeGes ep120, MM-Fi ep240; fixed-budget cells stay owned by converge_point.json).

Out: docs/foldwise_ladder.json + docs/foldwise_ladder_preds.npz
SMOKE=1 -> 3 subjects, ep2, seed 0, plumbing check.
"""
import os, json, time
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_round3 import kfold
from pointset_models import DeepSets, train_eval_set_full, build_point_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2, 3, 4)
UF_GATE = 0.95
COL = {"v": 3, "A": 4}  # columns of [x,y,z,v,A,t]

# arm -> (cols, (shuffled column, mode) or None)
ARMS = {
    "full(xyzvAt)":       ([0, 1, 2, 3, 4, 5], None),
    "no-velocity(xyzAt)": ([0, 1, 2, 4, 5],    None),
    "v-inshuffle":        ([0, 1, 2, 3, 4, 5], ("v", "in")),
    "v-crossshuffle":     ([0, 1, 2, 3, 4, 5], ("v", "cross")),
    "A-inshuffle":        ([0, 1, 2, 3, 4, 5], ("A", "in")),
    "A-crossshuffle":     ([0, 1, 2, 3, 4, 5], ("A", "cross")),
    "velocity(xyzvt)":    ([0, 1, 2, 3, 5],    None),
    "geometry(xyzt)":     ([0, 1, 2, 5],       None),
}


def shuffled_col(X, M, col, mode, seed):
    """Copy of column `col` under the control; valid points only, mask untouched."""
    c = COL[col]
    rng = np.random.default_rng((30_000 if col == "A" else 10_000) + seed)
    v = X[:, :, c].copy()
    valid = M > 0.5
    n = X.shape[0]
    if mode == "in":
        for i in range(n):
            idx = np.where(valid[i])[0]
            v[i, idx] = v[i, idx][rng.permutation(len(idx))]
    else:
        donor = rng.permutation(n)
        clash = donor == np.arange(n)
        donor[clash] = (donor[clash] + 1) % n
        for i in range(n):
            idx = np.where(valid[i])[0]
            dv = X[donor[i], :, c][valid[donor[i]]]
            if len(dv) == 0:
                v[i, idx] = 0.0
            else:
                v[i, idx] = rng.choice(dv, size=len(idx), replace=True)
    return v


def run(tag, insts, folds, ncls, ep, results, preds):
    subj = np.array([t[2] for t in insts])
    y_all = np.array([t[1] for t in insts], dtype=np.int64)
    accs_by_arm = {a: [] for a in ARMS}
    tr_by_arm = {a: [] for a in ARMS}
    for fi, te_s in enumerate(folds):
        te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # THE FIX: percentile ranges from training subjects only
        ranges = fit_ranges([t[0] for t, keep in zip(insts, tr) if keep])
        X, M, y, _ = build_point_tensors(insts, ranges)
        assert np.array_equal(y, y_all)
        print(f"{tag} fold{fi}: train-fit ranges {ranges}", flush=True)
        for aname, (cols, shuf) in ARMS.items():
            for sd in SEEDS:
                Xa = X
                if shuf is not None:
                    scol, mode = shuf
                    Xa = X.copy(); Xa[:, :, COL[scol]] = shuffled_col(X, M, scol, mode, sd)
                Xc = np.ascontiguousarray(Xa[..., cols])
                a, yt, yp, ta = train_eval_set_full(
                    DeepSets, Xc[tr], M[tr], y[tr], Xc[te], M[te], y[te],
                    ncls, len(cols), epochs=ep, seed=sd)
                accs_by_arm[aname].append(a); tr_by_arm[aname].append(ta)
                preds[f"{tag}|{aname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
        del X, M
    preds[f"SUBJ__{tag}"] = subj
    for aname in ARMS:
        accs, tr_accs = accs_by_arm[aname], tr_by_arm[aname]
        key = f"{tag}|{aname}|ep{ep}"
        results[key] = {"acc": round(float(np.mean(accs)) * 100, 2),
                        "std": round(float(np.std(accs)) * 100, 2),
                        "accs": [round(float(a) * 100, 2) for a in accs],
                        "min_train_acc": round(float(np.min(tr_accs)), 4),
                        "mean_train_acc": round(float(np.mean(tr_accs)), 4),
                        "underfit": bool(np.min(tr_accs) < UF_GATE)}
        r = results[key]
        print(f"  {key:44s}: {r['acc']:6.2f}% (+-{r['std']:.1f})  min_train={r['min_train_acc']:.3f}"
              f"{'  UF' if r['underfit'] else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(f"FOLDWISE-LADDER  SMOKE={SMOKE}  seeds={SEEDS}", flush=True)
    results, preds = {}, {}

    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    subj_mh = np.array([t[2] for t in mh])
    if SMOKE:
        keep = sorted(set(subj_mh))[:3]
        mh = [t for t in mh if t[2] in keep]; subj_mh = np.array([t[2] for t in mh])
    folds_mh = kfold(subj_mh, 2 if SMOKE else 5)
    run("mHomeGes", mh, folds_mh, 10, 2 if SMOKE else 120, results, preds)
    del mh

    if not SMOKE:
        mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
        folds_mf = kfold(np.array([t[2] for t in mf]), 5)
        run("MM-Fi", mf, folds_mf, 27, 240, results, preds)
        del mf

    out = {"purpose": "E1 major-revision: fold-wise range refit + intensity shuffles + 5 seeds (2026-08-18)",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(SEEDS), "uf_gate": UF_GATE,
                        "model": "DeepSets", "smoke": SMOKE,
                        "ranges": "fit_ranges on training subjects per fold (leak-free)",
                        "budgets": {"mHomeGes": 120, "MM-Fi": 240},
                        "folds": "kfold(seed=0); MM-Fi 5-fold over all subjects"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"foldwise_ladder{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"foldwise_ladder{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/foldwise_ladder{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
