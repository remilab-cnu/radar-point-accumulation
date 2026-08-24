"""E2 (major-revision campaign, 2026-08-18): quantity x reducer factorial for map arms.

Review finding R2: the map arms confound the physical quantity with the reducer
(velocity=signed SUM, intensity=MEAN, occupancy=COUNT). vmean_arm.json already shows the
reducer can flip an ordering on BGT60TR13C. This run crosses them:

    quantity  {v (signed doppler), A (|intensity|), 1 (presence)}
  x reducer   {sum, mean}

  v_sum   = per-cell signed doppler sum          (the paper's velocity map)
  v_mean  = count-normalized signed doppler
  A_sum   = per-cell |intensity| sum             (NEW: sum-reduced intensity)
  A_mean  = count-normalized |intensity|         (the paper's intensity map)
  cnt     = per-cell point count                 (the paper's occupancy map; sum of 1)
  occ_ind = nonempty-cell indicator              (NEW: mean-reduction of 1, saturating)

Same max-normalization (rep_variants.norm) for every arm, same SmallCNN width 32, same
folds, budgets {60,120}, seeds {0,1,2}. Datasets: Infineon/BGT60TR13C (the set whose map
arms meet the training criterion) and MM-Fi 5-fold (whole-body regime where map arms also
train; mHomeGes map arms stay out - they do not reach the criterion at any tested budget).

Out: docs/reducer_factorial.json + docs/reducer_factorial_preds.npz
SMOKE=1 -> subset, ep4, seed 0.
"""
import os, json, time
import numpy as np
from rep_variants import cell_stats, compose, norm, CAXES, infineon_recs, kfold
from spectra_dataset import fit_ranges, mmfi_instances
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
BUDGETS = (4,) if SMOKE else (60, 120)
WIDTH = 8 if SMOKE else 32

ARMS = {"v_sum": ["sum"], "v_mean": ["mean"], "A_sum": ["int_sum"], "A_mean": ["int_mean"],
        "cnt": ["cnt"], "occ_ind": ["occ_ind"]}


def extend_stats(st):
    """Add the two missing factorial cells to a cell_stats dict."""
    for ax in CAXES:
        s = st[ax]
        s["int_sum"] = (s["int_mean"] * s["cnt"]).astype(np.float32)
        s["occ_ind"] = (s["cnt"] > 0).astype(np.float32)
    return st


def build_arms(insts, ranges):
    stats = [extend_stats(cell_stats(t[0], CAXES, ranges, nb=32)) for t in insts]
    return {name: np.stack([compose(st, CAXES, spec) for st in stats]).astype(np.float32)
            for name, spec in ARMS.items()}


def run(tag, insts, folds, ncls, out, preds):
    """Fold-wise leak-free form (2026-08-19): ranges fit on training subjects per fold.
    The dataset-global run is archived as reducer_factorial.json."""
    y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
    preds[f"SUBJ__{tag}"] = subj
    res = {"n_instances": len(insts), "n_cls": ncls, "folds": [list(f) for f in folds],
           "width": WIDTH, "budgets": list(BUDGETS), "results": {},
           "ranges": "fit_ranges on training subjects per fold"}
    maxb = max(BUDGETS)
    accs_all = {f"{n}|ep{ep}": ([], []) for n in ARMS for ep in BUDGETS}
    for fi, te_s in enumerate(folds):
        te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ranges = fit_ranges([t[0] for t, keep in zip(insts, tr) if keep])
        X = build_arms(insts, ranges)
        for name, Xa in X.items():
            for ep in BUDGETS:
                for s in SEEDS:
                    a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], ncls,
                                                    epochs=ep, seed=s, width=WIDTH)
                    accs_all[f"{name}|ep{ep}"][0].append(a)
                    accs_all[f"{name}|ep{ep}"][1].append(ta)
                    if ep == maxb:
                        preds[f"{tag}|{name}|ep{ep}|fold{fi}|seed{s}"] = np.stack([te_idx, yt, yp])
        del X
    for k, (accs, taccs) in accs_all.items():
        res["results"][k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                             "std": round(float(np.std(accs)) * 100, 2),
                             "min_train_acc": round(float(np.min(taccs)), 4),
                             "mean_train_acc": round(float(np.mean(taccs)), 4),
                             "underfit": bool(np.min(taccs) < 0.95)}
        r = res["results"][k]
        print(f"  {tag} {k:18s}: {r['acc']:6.2f} (+-{r['std']:.1f}) "
              f"train={r['mean_train_acc']:.3f}{' UF' if r['underfit'] else ''}", flush=True)
    out[tag] = res


def subset(insts, n=200):
    idx = np.random.RandomState(0).permutation(len(insts))[:n]
    return [insts[i] for i in idx]


if __name__ == "__main__":
    t0 = time.time()
    print(f"REDUCER-FACTORIAL SMOKE={SMOKE} width={WIDTH} budgets={BUDGETS}", flush=True)
    FROZEN = json.load(open(os.path.join(DOCS, "baselines2.json")))["datasets"]
    out = {"purpose": "E2 major-revision: quantity x reducer factorial, uniform max-norm (2026-08-18)",
           "protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budgets": list(BUDGETS), "smoke": SMOKE}}
    preds = {}

    recs = infineon_recs()
    if SMOKE: recs = subset(recs)
    folds_if = (FROZEN["Infineon"]["folds"] if not SMOKE
                else [sorted(set(str(t[2]) for t in recs))[:2]])
    run("Infineon", recs, folds_if, 5, out, preds); del recs

    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE: mf = subset(mf)
    folds_mf = ([[str(s) for s in g] for g in kfold(np.array([str(t[2]) for t in mf]), 5)]
                if not SMOKE else [sorted(set(str(t[2]) for t in mf))[:2]])
    run("MM-Fi", mf, folds_mf, 27, out, preds); del mf

    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"reducer_factorial_fw{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"reducer_factorial_fw{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/reducer_factorial_fw{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
