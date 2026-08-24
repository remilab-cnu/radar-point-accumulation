"""Panel medium asks #1/#3/#4 in one job (2026-07-19).

A) MM-Fi point-domain ladder, FIVE subject-disjoint folds (matching the map-domain
   treatment; kfold(subj,5) seed=0 == converge_body folds), DeepSets at ep240
   (the budget that converges the full arm). Arms: full / no-velocity /
   v-inshuffle / v-crossshuffle. Upgrades the single-split "direction-only"
   corroboration to fold-replicated boundary-side evidence (or refutes it).
B) mHomeGes map arms at width 64, ep200: attempt to clear the 0.95 training gate
   that width-32/ep120 fails (min_train 0.62). Arms: v_sum / amplitude(int_mean)
   / occupancy. Same 5-fold, seeds {0,1,2}. Reports whether the velocity>occupancy
   map ordering holds where (if) the gate clears.
C) CPDP split-driver probe: same CPDP reimplementation on mHomeGes under a RANDOM
   instance-level 5-fold split (everything else frozen: lr 1e-3, ep30/40, seeds
   {0,1,2}). Frozen subject-disjoint result is 63.2/63.3; if the random split
   recovers most of the ~30 pp gap to the original paper's ~95, the driver is
   cross-subject generalization, and the V-B footnote can say so with evidence.

Out: docs/medium_asks.json + docs/medium_asks_preds.npz. SMOKE=1: tiny subsets, ep2.
"""
import os, json, time
import numpy as np
from spectra_dataset import fit_ranges, mhomeges_instances, mmfi_instances
from rep_variants import cell_stats, norm, CAXES
from rep_round3 import kfold
from cnn import train_eval_full
from pointset_models import DeepSets, train_eval_set_full, build_point_tensors
from rep_converge_point import shuffled_v
from baselines_pointnets import train_eval_set_preds_tr
from baselines_cpdp import MGesNetCPDP, build_cpdp_tensors, fit_cpdp_ranges

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
UF_GATE = 0.95
V = 3

results, preds = {}, {}


def summarize(key, accs, tr_accs):
    results[key] = {"acc": round(float(np.mean(accs)) * 100, 2),
                    "std": round(float(np.std(accs)) * 100, 2),
                    "accs": [round(float(a) * 100, 2) for a in accs],
                    "min_train_acc": round(float(np.min(tr_accs)), 4),
                    "underfit": bool(np.min(tr_accs) < UF_GATE)}
    r = results[key]
    print(f"  {key:44s}: {r['acc']:6.2f}% (+-{r['std']:.1f})  min_train={r['min_train_acc']:.3f}"
          f"{'  UF' if r['underfit'] else ''}", flush=True)


def part_a():
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE:
        keep = sorted(set(t[2] for t in mf))[:6]; mf = [t for t in mf if t[2] in keep]
    ranges = fit_ranges([t[0] for t in mf])
    y = np.array([t[1] for t in mf]); subj = np.array([t[2] for t in mf])
    folds = kfold(subj, 2 if SMOKE else 5)
    X, M, _, _ = build_point_tensors(mf, ranges)
    print(f"A) MM-Fi {len(y)} inst, 27 cls, X{X.shape}", flush=True)
    ARMS = {"full(xyzvAt)": ([0, 1, 2, 3, 4, 5], None),
            "no-velocity(xyzAt)": ([0, 1, 2, 4, 5], None),
            "v-inshuffle": ([0, 1, 2, 3, 4, 5], "in"),
            "v-crossshuffle": ([0, 1, 2, 3, 4, 5], "cross")}
    ep = 2 if SMOKE else 240
    for aname, (cols, vmode) in ARMS.items():
        accs, tr_accs = [], []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            if te.sum() == 0 or tr.sum() == 0:
                continue
            for sd in SEEDS:
                Xa = X
                if vmode is not None:
                    Xa = X.copy(); Xa[:, :, V] = shuffled_v(X, M, vmode, sd)
                Xc = np.ascontiguousarray(Xa[..., cols])
                a, yt, yp, ta = train_eval_set_full(DeepSets, Xc[tr], M[tr], y[tr],
                                                    Xc[te], M[te], y[te], 27, len(cols),
                                                    epochs=ep, seed=sd)
                accs.append(a); tr_accs.append(ta)
                preds[f"MMFi5f|{aname}|ep{ep}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, yp])
        summarize(f"MMFi5f|{aname}|ep{ep}", accs, tr_accs)


def part_b(mh, folds, y, subj):
    ranges = fit_ranges([t[0] for t in mh])
    stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in mh]
    ARMS = {"v_sum": "sum", "amplitude": "int_mean", "occupancy": "cnt"}
    ep, width = (2, 16) if SMOKE else (200, 64)
    for aname, statkey in ARMS.items():
        X = np.stack([np.stack([norm(st[ax][statkey]) for ax in CAXES]) for st in stats]).astype(np.float32)
        accs, tr_accs = [], []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            if te.sum() == 0 or tr.sum() == 0:
                continue
            for sd in SEEDS:
                a, yt, yp, ta = train_eval_full(X[tr], y[tr], X[te], y[te], 10,
                                                epochs=ep, seed=sd, width=width)
                accs.append(a); tr_accs.append(ta)
                preds[f"mHw64|{aname}|ep{ep}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, yp])
        summarize(f"mHw64|{aname}|ep{ep}", accs, tr_accs)
        del X
    del stats


def part_c(mh, y_subj_folds):
    y, subj, subj_folds = y_subj_folds
    ranges = fit_cpdp_ranges([t[0] for t in mh])
    X, M, yy, s = build_cpdp_tensors(mh, ranges, 32, 32, 30, 15, 16)
    n = len(yy)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    rfolds = np.array_split(perm, 2 if SMOKE else 5)
    print(f"C) CPDP random-split probe: {n} inst, tensors {X.shape}", flush=True)
    for ep in ((2,) if SMOKE else (30, 40)):
        accs, tr_accs = [], []
        for fi, te_idx in enumerate(rfolds):
            te = np.zeros(n, bool); te[te_idx] = True; tr = ~te
            for sd in SEEDS:
                a, yt, yp, ta = train_eval_set_preds_tr(MGesNetCPDP, X[tr], M[tr], yy[tr],
                                                        X[te], M[te], yy[te], 10, 1,
                                                        epochs=ep, lr=1e-3, seed=sd)
                accs.append(a); tr_accs.append(ta)
                preds[f"CPDPrand|ep{ep}|fold{fi}|seed{sd}"] = np.stack([np.where(te)[0], yt, yp])
        summarize(f"CPDPrand|mHomeGes|randomsplit|ep{ep}", accs, tr_accs)


if __name__ == "__main__":
    t0 = time.time()
    print(f"MEDIUM ASKS  SMOKE={SMOKE}", flush=True)
    part_a()
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    if SMOKE:
        keep = sorted(set(t[2] for t in mh))[:3]; mh = [t for t in mh if t[2] in keep]
    y = np.array([t[1] for t in mh]); subj = np.array([t[2] for t in mh])
    folds = kfold(subj, 2 if SMOKE else 5)
    part_b(mh, folds, y, subj)
    part_c(mh, (y, subj, folds))
    out = {"purpose": "panel medium asks: (A) MM-Fi 5-fold point ladder ep240; "
                      "(B) mHomeGes map width-64 ep200 gate attempt; "
                      "(C) CPDP random-split driver probe (frozen subj-disjoint ref: 63.2/63.3)",
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(SEEDS), "uf_gate": UF_GATE,
                        "smoke": SMOKE},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"medium_asks{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"medium_asks{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/medium_asks{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
