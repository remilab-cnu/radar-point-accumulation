"""E6+E7 (cold-review round 12, 2026-08-24): the direct cancellation controls the
referee asked for, plus mRI on the fold-wise pipeline so every C-narrative point
shares one preprocessing.

E6 (MM-Fi, fold-wise, ep120): does removing ONLY the cancellation explain the
histogram's gain, with nothing else changed?
  v_possum_negsum  2ch: per-cell sum of positive v, sum of |negative v|
                   -> cancellation removed; same reducer (sum), no binning
  v_abssum         1ch: per-cell sum of |v| -> sign REMOVED but magnitude kept
                   (if cancellation is the failure, this should NOT recover)
  v_sum_2ch        2ch: the scalar signed sum duplicated -> pure width control
  v_sum, v_hist4   reference arms, same build

E7 (mRI, fold-wise, ep120): scalar sum and hist4 only (its shuffle arm is known
not to train), so the three-dataset C-vs-recovery series is single-pipeline.

Fold-wise ranges (train subjects only), SmallCNN w32, lr 1e-3, batch 64,
seeds {0,1,2}, per-instance preds saved. SMOKE=1 tiny check.
Out: docs/sign_controls.json + _preds.npz
"""
import os, json, time
import numpy as np
from rep_variants import cell_stats, compose, norm, CAXES, kfold, T
from spectra_dataset import fit_ranges, mmfi_instances
from rep_converge import mri_records
from rep_hist_ladder import axis_bins, NB, EDGES4, hist_channels
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
BUDGET = 4 if SMOKE else 120
WIDTH = 8 if SMOKE else 32


def signed_sum_channels(inst, ranges, mode):
    """mode: 'posneg' (2ch: sum v>0, sum |v<0|), 'abs' (1ch), 'dup' (2ch signed sum)."""
    ab, ti = axis_bins(inst, ranges)
    v = inst["doppler"].values.astype(float)
    ch = []
    for ax in CAXES:
        bi, m = ab[ax]
        b, t, vv = bi[m], ti[m], v[m]
        def acc(vals, sel):
            h = np.zeros((NB, T), np.float32)
            np.add.at(h, (b[sel], t[sel]), vals[sel])
            return h
        if mode == "posneg":
            ch += [norm(acc(vv, vv > 0)), norm(acc(-vv, vv < 0))]
        elif mode == "abs":
            ch += [norm(acc(np.abs(vv), np.ones_like(vv, bool)))]
        else:  # dup
            s = norm(acc(vv, np.ones_like(vv, bool)))
            ch += [s, s.copy()]
    return np.stack(ch)


def build(insts, ranges, arms):
    X = {}
    if "v_sum" in arms or "v_hist4" in arms:
        stats = [cell_stats(t[0], CAXES, ranges, nb=NB) for t in insts]
        if "v_sum" in arms:
            X["v_sum"] = np.stack([compose(st, CAXES, ["sum"]) for st in stats]).astype(np.float32)
        if "v_hist4" in arms:
            X["v_hist4"] = np.stack([compose(st, CAXES, ["hist"]) for st in stats]).astype(np.float32)
        del stats
    for mode, name in (("posneg", "v_possum_negsum"), ("abs", "v_abssum"), ("dup", "v_sum_2ch")):
        if name in arms:
            X[name] = np.stack([signed_sum_channels(t[0], ranges, mode)
                                for t in insts]).astype(np.float32)
    return X


def run(tag, insts, ncls, arms, results, preds):
    subj = np.array([str(t[2]) for t in insts])
    folds = ([[str(x) for x in g] for g in kfold(subj, 5)]
             if not SMOKE else [sorted(set(subj))[:2]])
    y = np.array([t[1] for t in insts])
    preds[f"SUBJ__{tag}"] = subj
    accs_all = {}
    for fi, te_s in enumerate(folds):
        te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ranges = fit_ranges([t[0] for t, keep in zip(insts, tr) if keep])
        X = build(insts, ranges, arms)
        for name, Xa in X.items():
            for sd in SEEDS:
                a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], ncls,
                                                epochs=BUDGET, seed=sd, width=WIDTH)
                accs_all.setdefault(name, ([], []))
                accs_all[name][0].append(a); accs_all[name][1].append(ta)
                preds[f"{tag}|{name}|ep{BUDGET}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, yp])
        del X
    for name, (accs, taccs) in accs_all.items():
        k = f"{tag}|{name}|ep{BUDGET}"
        results[k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                      "std": round(float(np.std(accs)) * 100, 2),
                      "min_train_acc": round(float(np.min(taccs)), 4),
                      "mean_train_acc": round(float(np.mean(taccs)), 4),
                      "underfit": bool(np.min(taccs) < 0.95)}
        r = results[k]
        print(f"  {k:36s}: {r['acc']:6.2f} (+-{r['std']:.1f}) train={r['mean_train_acc']:.3f}"
              f"{' UF' if r['underfit'] else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(f"SIGN-CONTROLS SMOKE={SMOKE} w={WIDTH} ep={BUDGET}", flush=True)
    results, preds = {}, {}
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE:
        idx = np.random.RandomState(0).permutation(len(mf))[:200]; mf = [mf[i] for i in idx]
    run("MM-Fi", mf, 27, ("v_sum", "v_hist4", "v_possum_negsum", "v_abssum", "v_sum_2ch"),
        results, preds)
    del mf
    if not SMOKE:
        mr = mri_records()
        run("mRI", mr, 10, ("v_sum", "v_hist4"), results, preds)
    out = {"purpose": "E6 cancellation-isolating controls + E7 mRI fold-wise (cold review r12)",
           "protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budget": BUDGET, "nb": NB, "smoke": SMOKE,
                        "ranges": "fit_ranges on training subjects per fold"},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"sign_controls{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"sign_controls{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/sign_controls{sfx}.json in {time.time()-t0:.0f}s", flush=True)
