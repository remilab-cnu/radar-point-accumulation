"""Cluster-unit check for the primary point-domain estimate (mHomeGes, ep300).

Section III-C of the manuscript states that resampling the five folds rather than
the twenty-five subjects does not widen the interval of the primary estimate.
This script computes both intervals from the saved per-instance predictions and
writes docs/cluster_unit_check.json. Same estimator and rng seeds as
foldwise_stats.py (B = 5000); CPU-only, runs from preds alone.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
B = 5000
NPZ, KFMT = "foldwise_ext300_preds.npz", "{tag}|{arm}|ep{ep}|f{fi}|s{sd}"


def load(tag, arm, ep, folds, seeds):
    z = np.load(os.path.join(DOCS, NPZ), allow_pickle=True)
    subj = z[f"SUBJ__{tag}"]
    per_seed, fold_of = {}, {}
    for sd in range(seeds):
        c = {}
        for fi in range(folds):
            k = KFMT.format(tag=tag, arm=arm, ep=ep, fi=fi, sd=sd)
            if k not in z:
                continue
            te, yt, yp = z[k]
            for i, t, p in zip(te, yt, yp):
                c[int(i)] = int(t == p)
                fold_of[int(i)] = fi
        per_seed[sd] = c
    return subj, per_seed, fold_of


def check(tag, armA, armB, ep=300, folds=5, seeds=5):
    subj, A, fo = load(tag, armA, ep, folds, seeds)
    _, Bc, _ = load(tag, armB, ep, folds, seeds)
    sds = sorted(set(A) & set(Bc))
    idx = sorted(set(A[sds[0]]) & set(Bc[sds[0]]))
    s = np.array([subj[i] for i in idx])
    us = np.unique(s)
    D = np.zeros((len(us), len(sds)))
    for j, sd in enumerate(sds):
        d = np.array([A[sd][i] - Bc[sd][i] for i in idx], float)
        for k, u in enumerate(us):
            D[k, j] = d[s == u].mean()
    sub_fold = {}
    for i in idx:
        sub_fold.setdefault(subj[i], fo[i])
    ff = np.array([sub_fold[u] for u in us])

    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (2.5, 97.5)]
    rng = np.random.default_rng(0)
    bs = [D[rng.integers(0, len(us), len(us))].mean() * 100 for _ in range(B)]
    uf = np.unique(ff)
    rng = np.random.default_rng(2)
    bf = []
    for _ in range(B):
        drawn = rng.integers(0, len(uf), len(uf))
        rows = np.concatenate([np.where(ff == uf[f])[0] for f in drawn])
        bf.append(D[rows].mean() * 100)
    return {"mean_pp": round(float(D.mean() * 100), 2),
            "ci_subject": q(bs), "ci_fold": q(bf),
            "n_subjects": int(len(us)), "n_folds": int(len(uf)),
            "n_seeds": len(sds), "B": B}


out = {"dim_matched(full-vcross)|mHomeGes|ep300":
       check("mHomeGes", "full(xyzvAt)", "v-crossshuffle"),
       "_note": "cluster-unit sensitivity for Sec. III-C; subject vs fold resampling"}
json.dump(out, open(os.path.join(DOCS, "cluster_unit_check.json"), "w"), indent=1)
for k, v in out.items():
    if not k.startswith("_"):
        print(k, v)
