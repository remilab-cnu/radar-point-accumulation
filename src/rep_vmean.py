"""MEAN-FORM VELOCITY MAP ARM (referee W1, 2026-07-19).

The map velocity arm accumulates a signed SUM, which grows with the cell point count
and hence partially encodes occupancy for sign-coherent motion; the intensity arm is
count-normalized (mean). This run adds the count-normalized velocity arm (per-cell
MEAN of signed Doppler) under the same protocols as the existing arms, so the map
ordering can be read with the count coupling removed:
  Infineon: frozen protocol (frozen pkl md5-checked, ep30+ep40, width 16, 4-fold)
            -> compare with final_infineon.json map_v_sum / REF_occupancy.
  mHomeGes: converge protocol (ep120, width 32, 5-fold)
            -> compare with converge_mh.json v_sum|ep120 / occupancy|ep120.
seeds {0,1,2}, batch 64, train-acc gate 0.95, per-instance preds saved.
Out: docs/vmean_arm.json + docs/vmean_arm_preds.npz. SMOKE=1: 3 subjects, ep2.
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import fit_ranges, mhomeges_instances
from rep_variants import cell_stats, norm, CAXES, infineon_recs
from rep_round3 import kfold
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
UF_GATE = 0.95


def build_vmean(recs, ranges):
    return np.stack([np.stack([norm(cell_stats(t[0], CAXES, ranges, nb=32)[ax]["mean"])
                               for ax in CAXES]) for t in recs]).astype(np.float32)


def run(tag, recs, k, ncls, epochs, width, results, preds):
    ranges = fit_ranges([t[0] for t in recs])
    y = np.array([t[1] for t in recs]); subj = np.array([t[2] for t in recs])
    folds = kfold(subj, k)
    X = build_vmean(recs, ranges)
    print(f"{tag}: {len(y)} inst, X{X.shape}", flush=True)
    for ep in epochs:
        accs, tr_accs = [], []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            if te.sum() == 0 or tr.sum() == 0:
                continue
            for sd in SEEDS:
                a, yt, yp, ta = train_eval_full(X[tr], y[tr], X[te], y[te], ncls,
                                                epochs=ep, seed=sd, width=width)
                accs.append(a); tr_accs.append(ta)
                preds[f"{tag}|map_v_mean|ep{ep}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, yp])
        key = f"{tag}|map_v_mean|ep{ep}"
        results[key] = {"acc": round(float(np.mean(accs)) * 100, 2),
                        "std": round(float(np.std(accs)) * 100, 2),
                        "accs": [round(float(a) * 100, 2) for a in accs],
                        "min_train_acc": round(float(np.min(tr_accs)), 4),
                        "underfit": bool(np.min(tr_accs) < UF_GATE)}
        r = results[key]
        print(f"  {key:28s}: {r['acc']:6.2f}% (+-{r['std']:.1f})  min_train={r['min_train_acc']:.3f}"
              f"{'  UF' if r['underfit'] else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(f"V-MEAN ARM  SMOKE={SMOKE}", flush=True)
    results, preds = {}, {}
    inf = infineon_recs()
    md5 = hashlib.md5(open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest()
    run("Infineon", inf, 2 if SMOKE else 4, 5, (2,) if SMOKE else (30, 40), 16, results, preds)
    del inf
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    if SMOKE:
        keep = sorted(set(t[2] for t in mh))[:3]; mh = [t for t in mh if t[2] in keep]
    run("mHomeGes", mh, 2 if SMOKE else 5, 10, (2,) if SMOKE else (120,), 32, results, preds)
    out = {"purpose": "count-normalized (mean) velocity map arm — referee W1 sum-vs-mean asymmetry",
           "manifest": {"infineon_pkl_md5": md5},
           "protocol": {"lr": 1e-3, "batch": 64, "seeds": list(SEEDS), "uf_gate": UF_GATE,
                        "widths": {"Infineon": 16, "mHomeGes": 32},
                        "budgets": {"Infineon": [30, 40], "mHomeGes": [120]}, "smoke": SMOKE},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"vmean_arm{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"vmean_arm{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/vmean_arm{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
