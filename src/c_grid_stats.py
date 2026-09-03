"""Paired statistics for the within-dataset C manipulation (docs/c_grid_sweep.json).

For each map grid, the subject-clustered paired difference v_hist4 - v_sum, computed from
the saved per-instance predictions with the same estimator and rng seeds as
foldwise_stats.py (B = 5000). Writes docs/c_grid_stats.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
B = 5000
NPZ = "c_grid_sweep_preds.npz"
KFMT = "MM-Fi|{config}|nb{nb}|ep{ep}|fold{fi}|seed{sd}"


def load(z, config, nb, ep, folds, seeds):
    per = {}
    for sd in range(seeds):
        c = {}
        for fi in range(folds):
            k = KFMT.format(config=config, nb=nb, ep=ep, fi=fi, sd=sd)
            if k in z:
                te, yt, yp = z[k]
                for i, t, p in zip(te, yt, yp):
                    c[int(i)] = int(t == p)
        per[sd] = c
    return per


def contrast(z, nb, ep=120, folds=5, seeds=3):
    subj = z["SUBJ__MM-Fi"]
    A, Bc = load(z, "v_hist4", nb, ep, folds, seeds), load(z, "v_sum", nb, ep, folds, seeds)
    sds = sorted(set(A) & set(Bc))
    idx = sorted(set(A[sds[0]]) & set(Bc[sds[0]]))
    s = np.array([subj[i] for i in idx]); us = np.unique(s)
    D = np.zeros((len(us), len(sds)))
    for j, sd in enumerate(sds):
        d = np.array([A[sd][i] - Bc[sd][i] for i in idx], float)
        for k, u in enumerate(us):
            D[k, j] = d[s == u].mean()
    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (2.5, 97.5)]
    rng = np.random.default_rng(0)
    bs = [D[rng.integers(0, len(us), len(us))].mean() * 100 for _ in range(B)]
    rng = np.random.default_rng(1)
    bt = []
    for _ in range(B):
        ui = rng.integers(0, len(us), len(us)); si = rng.integers(0, len(sds), len(sds))
        bt.append(D[np.ix_(ui, si)].mean() * 100)
    return {"mean_pp": round(float(D.mean() * 100), 2), "ci_subject": q(bs),
            "ci_two_level": q(bt), "n_subjects": int(len(us)), "n_seeds": len(sds), "B": B}


if __name__ == "__main__":
    sweep = json.load(open(os.path.join(DOCS, "c_grid_sweep.json")))
    z = np.load(os.path.join(DOCS, NPZ), allow_pickle=True)
    out = {}
    for nb in sweep["protocol"]["grids"]:
        st = contrast(z, nb)
        st["C"] = sweep["C_by_grid"][str(nb)]["C_mean_per_instance"]
        for config in ("v_sum", "v_hist4"):
            k = f"MM-Fi|{config}|nb{nb}|ep120"
            st[f"{config}_acc"] = sweep["results"][k]["acc"]
            st[f"{config}_min_train"] = sweep["results"][k]["min_train_acc"]
        out[f"hist4-v_sum|nb{nb}"] = st
        print(f"nb={nb}  C={st['C']:.3f}  gain={st['mean_pp']:+.2f} pp "
              f"CI {st['ci_subject']}  (train {st['v_sum_min_train']:.3f}/"
              f"{st['v_hist4_min_train']:.3f})")
    out["_note"] = ("within-dataset C manipulation via map grid; coarser grid lowers C but "
                    "also lowers spatial resolution, so across-grid comparison is not a "
                    "C-only manipulation")
    json.dump(out, open(os.path.join(DOCS, "c_grid_stats.json"), "w"), indent=1)
    print("\nwrote docs/c_grid_stats.json")
