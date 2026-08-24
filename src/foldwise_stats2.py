"""Clustered CIs for the fold-wise map reruns (E2fw reducer factorial, E3fw hist ladder).
Same estimator as foldwise_stats.py; preds exist at the max budget (ep120) only.
Cross-file contrasts (hist arms vs factorial arms) are valid: identical folds
(kfold seed 0), subjects, seeds, and pipeline. Out: docs/foldwise_stats2.json."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
B = 5000
KF = "{tag}|{arm}|ep{ep}|fold{fi}|seed{sd}"

def load(npz, tag, arm, ep, folds, seeds):
    z = np.load(os.path.join(DOCS, npz), allow_pickle=True)
    subj = z[f"SUBJ__{tag}"]
    per_seed = {}
    for sd in range(seeds):
        c = {}
        for fi in range(folds):
            k = KF.format(tag=tag, arm=arm, ep=ep, fi=fi, sd=sd)
            if k not in z:
                continue
            te, yt, yp = z[k]
            for i, t, p in zip(te, yt, yp):
                c[int(i)] = int(t == p)
        per_seed[sd] = c
    return subj, per_seed

def contrast(fA, armA, fB, armB, tag="MM-Fi", ep=120, folds=5, seeds=3):
    subj, A = load(fA, tag, armA, ep, folds, seeds)
    _,    Bc = load(fB, tag, armB, ep, folds, seeds)
    sds = sorted(set(A) & set(Bc))
    idx = sorted(set(A[sds[0]]) & set(Bc[sds[0]]))
    s = np.array([subj[i] for i in idx]); us = np.unique(s)
    D = np.zeros((len(us), len(sds)))
    for j, sd in enumerate(sds):
        d = np.array([A[sd][i] - Bc[sd][i] for i in idx], float)
        for k, u in enumerate(us):
            D[k, j] = d[s == u].mean()
    rng = np.random.default_rng(0)
    bs = [D[rng.integers(0, len(us), len(us))].mean() * 100 for _ in range(B)]
    rng = np.random.default_rng(1)
    bt = []
    for _ in range(B):
        ui = rng.integers(0, len(us), len(us)); si = rng.integers(0, len(sds), len(sds))
        bt.append(D[np.ix_(ui, si)].mean() * 100)
    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (2.5, 97.5)]
    return {"mean_pp": round(float(D.mean() * 100), 2), "ci_subject": q(bs),
            "ci_two_level": q(bt), "n_subjects": int(len(us)), "n_seeds": len(sds)}

RF, HL = "reducer_factorial_fw_preds.npz", "hist_ladder_fw_preds.npz"
out = {"MM-Fi_ep120": {
    "cnt-v_sum": contrast(RF, "cnt", RF, "v_sum"),
    "A_sum-v_sum": contrast(RF, "A_sum", RF, "v_sum"),
    "cnt-occ_ind(reducer, occupancy)": contrast(RF, "cnt", RF, "occ_ind"),
    "A_sum-A_mean(reducer, intensity)": contrast(RF, "A_sum", RF, "A_mean"),
    "hist4-cnt(cross-run)": contrast(HL, "v_hist4", RF, "cnt"),
    "hist4-A_sum(cross-run)": contrast(HL, "v_hist4", RF, "A_sum"),
    "hist4-v_sum": contrast(HL, "v_hist4", HL, "v_sum"),
    "hist4-vshuffled(association)": contrast(HL, "v_hist4", HL, "v_hist4_vshuffled"),
    "vshuffled-v_sum(distribution+width)": contrast(HL, "v_hist4_vshuffled", HL, "v_sum"),
    "hist2-v_sum(sign_only)": contrast(HL, "v_hist2", HL, "v_sum"),
    "hist8-hist4": contrast(HL, "v_hist8", HL, "v_hist4"),
}, "Infineon_ep120": {
    "v_sum-occ_ind": contrast(RF, "v_sum", RF, "occ_ind", tag="Infineon", folds=12),
}}
out["_note"] = ("fold-wise map reruns 2026-08-19; A_mean/occ_ind cells carry a dagger at ep120 "
                "(A_mean 0.701 is below the 0.90 floor: its reducer contrast is context only)")
json.dump(out, open(os.path.join(DOCS, "foldwise_stats2.json"), "w"), indent=1)
for g, d in out.items():
    if g.startswith("_"): continue
    print("==", g)
    for k, v in d.items():
        print(f"  {k:36s} {v['mean_pp']:6.2f}  subj{v['ci_subject']}  two{v['ci_two_level']}")
