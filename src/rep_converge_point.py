"""CONVERGENCE RERUN for the point-domain velocity ablation (cold-review S1 + P5, 2026-07-17).

Why: the Phase-1 headline (+10.6 pp velocity contribution, mHomeGes DeepSets full vs
no-velocity) sits on cells that are themselves under the 0.95 train-acc gate
(rerun_audit_ds: mHomeGes min_train 0.919/0.929 mild; MM-Fi 0.513/0.593 severe), and the
ablation is channel-REMOVAL (drops input dimensionality). This run answers both:
  S1  does the velocity contribution survive at a converged budget?
  P5  is the gap velocity INFORMATION rather than channel count? (shuffle controls,
      dimensionality preserved)

Design (frozen otherwise: lr 1e-3, batch 64, seeds {0,1,2}, subject-disjoint folds via the
single kfold module, per-instance preds + train-acc saved, UF gate 0.95):
  arms (DeepSets, cols of [x,y,z,v,A,t]):
    full(xyzvAt)        [0,1,2,3,4,5]
    no-velocity(xyzAt)  [0,1,2,4,5]      channel-removal ablation (the current headline)
    v-inshuffle         full cols, v permuted among VALID points within each instance
                        -> destroys point-level (x,y,z,t)<->v coupling, KEEPS the
                        instance-level v distribution AND dimensionality
    v-crossshuffle      full cols, v of instance i replaced by resampled v values from a
                        random OTHER instance -> destroys class-relevant v info entirely,
                        keeps marginal stats + dimensionality (the P5 dimensionality control)
    velocity(xyzvt)     [0,1,2,3,5]
    occupancy(xyzt)     [0,1,2,5]
  ladder: full - inshuffle   = per-point coupling value
          inshuffle - cross  = instance-level v-distribution value
          cross - no-velocity ~ 0 expected (pure dimensionality)
  budgets: mHomeGes ep80/ep120 (0.93 at ep40 -> expect convergence);
           MM-Fi ep120/ep240 (0.51-0.59 at ep40; may STILL underfit -> report honestly,
           MM-Fi stays direction-only regardless per the cold-review disposition).
  folds: mHomeGes kfold(subj,5) seed=0 == p1_crossparadigm folds (bit-identical module);
         MM-Fi = the S2 split (single split; no fold CI claims).

Known pitfalls injected: DeepSets empty-mask finfo.min guard already fixed in
pointset_models (round-4/6 collapse); __main__ guard; shuffles respect the padding mask;
SMOKE=1 -> 3 subjects, ep2, seed 0 only, plumbing check.

Out: docs/converge_point.json + docs/converge_point_preds.npz.
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_round3 import kfold
from pointset_models import DeepSets, train_eval_set_full, build_point_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
UF_GATE = 0.95
V = 3  # velocity column in [x,y,z,v,A,t]

# arm -> (cols, v_variant in {None,'in','cross'})
ARMS = {
    "full(xyzvAt)":       ([0, 1, 2, 3, 4, 5], None),
    "no-velocity(xyzAt)": ([0, 1, 2, 4, 5],    None),
    "v-inshuffle":        ([0, 1, 2, 3, 4, 5], "in"),
    "v-crossshuffle":     ([0, 1, 2, 3, 4, 5], "cross"),
    "velocity(xyzvt)":    ([0, 1, 2, 3, 5],    None),
    "occupancy(xyzt)":    ([0, 1, 2, 5],       None),
}


def shuffled_v(X, M, mode, seed):
    """Return a copy of the v column under the control. Shuffles touch VALID points only
    (M > 0.5); padded slots keep their original (masked-out) value."""
    rng = np.random.default_rng(10_000 + seed)
    v = X[:, :, V].copy()
    valid = M > 0.5
    n = X.shape[0]
    if mode == "in":            # permute v among an instance's own valid points
        for i in range(n):
            idx = np.where(valid[i])[0]
            v[i, idx] = v[i, idx][rng.permutation(len(idx))]
    else:                       # 'cross': donor v values from a random other instance
        donor = rng.permutation(n)
        donor[donor == np.arange(n)] = (donor[donor == np.arange(n)] + 1) % n
        for i in range(n):
            idx = np.where(valid[i])[0]
            dv = X[donor[i], :, V][valid[donor[i]]]
            if len(dv) == 0:    # degenerate donor: zero out (info-free either way)
                v[i, idx] = 0.0
            else:
                v[i, idx] = rng.choice(dv, size=len(idx), replace=True)
    return v


def run(tag, insts, folds, ncls, budgets, results, preds):
    ranges = fit_ranges([t[0] for t in insts])
    X, M, y, s = build_point_tensors(insts, ranges)
    print(f"{tag}: {len(y)} inst, X{X.shape}", flush=True)
    for ep in budgets:
        for aname, (cols, vmode) in ARMS.items():
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(s, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                for sd in SEEDS:
                    Xa = X
                    if vmode is not None:               # per-seed shuffle realization
                        Xa = X.copy(); Xa[:, :, V] = shuffled_v(X, M, vmode, sd)
                    Xc = np.ascontiguousarray(Xa[..., cols])
                    a, yt, yp, ta = train_eval_set_full(
                        DeepSets, Xc[tr], M[tr], y[tr], Xc[te], M[te], y[te],
                        ncls, len(cols), epochs=ep, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{tag}|{aname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
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
    print(f"CONVERGE-POINT  SMOKE={SMOKE}  seeds={SEEDS}", flush=True)
    results, preds = {}, {}

    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    subj_mh = np.array([t[2] for t in mh])
    if SMOKE:
        keep = sorted(set(subj_mh))[:3]
        mh = [t for t in mh if t[2] in keep]; subj_mh = np.array([t[2] for t in mh])
    folds_mh = kfold(subj_mh, 2 if SMOKE else 5)      # seed=0 -> identical to p1 folds
    run("mHomeGes", mh, folds_mh, 10, (2,) if SMOKE else (80, 120), results, preds)
    del mh

    if not SMOKE:
        mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
        S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
        run("MM-Fi", mf, [S2], 27, (120, 240), results, preds)
        del mf

    out = {"purpose": "cold-review S1 convergence + P5 shuffle-control (2026-07-17)",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(SEEDS), "uf_gate": UF_GATE,
                        "aug": "none", "model": "DeepSets", "smoke": SMOKE,
                        "budgets": {"mHomeGes": [80, 120], "MM-Fi": [120, 240]},
                        "folds": "kfold(seed=0) == p1_crossparadigm; MM-Fi = S2 single split"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"converge_point{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"converge_point{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/converge_point{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
