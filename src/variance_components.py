"""Subject-by-seed variance decomposition of the primary paired difference.

Section III-C of the manuscript reports the subject, seed and residual-cell
standard deviations of the primary point-domain difference (mHomeGes, ep300,
full vs cross-instance value replacement) and the share each contributes to the
variance of the mean. Both are computed here from the saved per-instance
predictions, so the two-level-bootstrap justification is auditable.

Components are the standard deviations of the additive decomposition's marginal
means (subject means, seed means, interaction residual cells); the variance of
the mean is then s_subj^2/n_subj + s_seed^2/n_seed + s_resid^2/(n_subj*n_seed).
CPU-only, runs from preds alone.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")


def paired_matrix(npz, kfmt, tag, armA, armB, ep, folds=5, seeds=5):
    """D[subject, seed] = per-subject mean paired difference, in percentage points."""
    z = np.load(os.path.join(DOCS, npz), allow_pickle=True)
    subj = z[f"SUBJ__{tag}"]

    def load(arm):
        per = {}
        for sd in range(seeds):
            c = {}
            for fi in range(folds):
                k = kfmt.format(tag=tag, arm=arm, ep=ep, fi=fi, sd=sd)
                if k in z:
                    te, yt, yp = z[k]
                    for i, t, p in zip(te, yt, yp):
                        c[int(i)] = int(t == p)
            per[sd] = c
        return per

    A, B = load(armA), load(armB)
    sds = sorted(set(A) & set(B))
    idx = sorted(set(A[sds[0]]) & set(B[sds[0]]))
    s = np.array([subj[i] for i in idx])
    us = np.unique(s)
    D = np.zeros((len(us), len(sds)))
    for j, sd in enumerate(sds):
        d = np.array([A[sd][i] - B[sd][i] for i in idx], float)
        for k, u in enumerate(us):
            D[k, j] = d[s == u].mean()
    return D * 100


def decompose(D):
    ns, nk = D.shape
    g = D.mean()
    rm, cm = D.mean(1), D.mean(0)
    resid = D - rm[:, None] - cm[None, :] + g
    s_subj, s_seed = float(rm.std(ddof=1)), float(cm.std(ddof=1))
    s_res = float(resid.std(ddof=1))
    v = [s_subj**2 / ns, s_seed**2 / nk, s_res**2 / (ns * nk)]
    tot = sum(v)
    return {"mean_pp": round(float(g), 2), "n_subjects": ns, "n_seeds": nk,
            "sd_subject_pp": round(s_subj, 2), "sd_seed_pp": round(s_seed, 2),
            "sd_residual_cell_pp": round(s_res, 2),
            "share_of_var_of_mean_pct": {
                "subject": round(100 * v[0] / tot), "seed": round(100 * v[1] / tot),
                "residual": round(100 * v[2] / tot)}}


KF = "{tag}|{arm}|ep{ep}|f{fi}|s{sd}"
out = {
    "primary|mHomeGes|ep300|full-vcross":
        decompose(paired_matrix("foldwise_ext300_preds.npz", KF, "mHomeGes",
                                "full(xyzvAt)", "v-crossshuffle", 300)),
    "earlier|mHomeGes|ep120|full-vcross":
        decompose(paired_matrix("foldwise_ladder_preds.npz", KF, "mHomeGes",
                                "full(xyzvAt)", "v-crossshuffle", 120)),
    "_note": "components are SDs of the additive decomposition's marginal means; "
             "the ep120 entry is the earlier primary, kept for comparison",
}
json.dump(out, open(os.path.join(DOCS, "variance_components.json"), "w"), indent=1)
for k, v in out.items():
    if not k.startswith("_"):
        print(k, json.dumps(v))
