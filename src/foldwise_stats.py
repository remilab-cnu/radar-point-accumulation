"""Manuscript statistics from the fold-wise (leak-free) reruns: every contrast §V quotes,
with subject-clustered and two-level (crossed-seed) bootstrap CIs, written to
docs/foldwise_stats.json. Bit-reproducible (fixed rng). CPU-only, runs from preds alone."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
B = 5000

def load(npz, tag, arm, ep, folds, seeds, kfmt):
    """per-instance correctness, per seed: dict seed -> {inst: 0/1}"""
    z = np.load(os.path.join(DOCS, npz), allow_pickle=True)
    subj = z[f"SUBJ__{tag}"]
    per_seed = {}
    for sd in range(seeds):
        c = {}
        for fi in range(folds):
            k = kfmt.format(tag=tag, arm=arm, ep=ep, fi=fi, sd=sd)
            if k not in z:
                continue
            te, yt, yp = z[k]
            for i, t, p in zip(te, yt, yp):
                c[int(i)] = int(t == p)
        per_seed[sd] = c
    return subj, per_seed

def contrast(npz, tag, armA, armB, ep, folds, seeds, kfmt):
    subj, A = load(npz, tag, armA, ep, folds, seeds, kfmt)
    _,    Bc = load(npz, tag, armB, ep, folds, seeds, kfmt)
    sds = sorted(set(A) & set(Bc))
    idx = sorted(set(A[sds[0]]) & set(Bc[sds[0]]))
    s = np.array([subj[i] for i in idx]); us = np.unique(s)
    # D[u, sd] = per-subject mean diff at seed sd
    D = np.zeros((len(us), len(sds)))
    for j, sd in enumerate(sds):
        d = np.array([A[sd][i] - Bc[sd][i] for i in idx], float)
        for k, u in enumerate(us):
            D[k, j] = d[s == u].mean()
    mean = D.mean() * 100
    rng = np.random.default_rng(0)
    bs_subj = [D[rng.integers(0, len(us), len(us))].mean() * 100 for _ in range(B)]
    rng = np.random.default_rng(1)
    bs_two = []
    for _ in range(B):
        ui = rng.integers(0, len(us), len(us)); si = rng.integers(0, len(sds), len(sds))
        bs_two.append(D[np.ix_(ui, si)].mean() * 100)
    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (2.5, 97.5)]
    return {"mean_pp": round(float(mean), 2), "ci_subject": q(bs_subj),
            "ci_two_level": q(bs_two), "n_subjects": int(len(us)), "n_seeds": len(sds)}

E1 = ("foldwise_ladder_preds.npz", "{tag}|{arm}|ep{ep}|f{fi}|s{sd}")
E3 = ("hist_ladder_preds.npz", "{tag}|{arm}|ep{ep}|fold{fi}|seed{sd}")

out = {}
for tag, ep in (("mHomeGes", 120), ("MM-Fi", 240)):
    fw = {}
    fw["velocity_contribution(full-novel)"] = contrast(E1[0], tag, "full(xyzvAt)", "no-velocity(xyzAt)", ep, 5, 5, E1[1])
    fw["dim_matched(full-vcross)"] = contrast(E1[0], tag, "full(xyzvAt)", "v-crossshuffle", ep, 5, 5, E1[1])
    fw["association(full-vin)"] = contrast(E1[0], tag, "full(xyzvAt)", "v-inshuffle", ep, 5, 5, E1[1])
    fw["distribution(vin-vcross)"] = contrast(E1[0], tag, "v-inshuffle", "v-crossshuffle", ep, 5, 5, E1[1])
    fw["dim_control(vcross-novel)"] = contrast(E1[0], tag, "v-crossshuffle", "no-velocity(xyzAt)", ep, 5, 5, E1[1])
    fw["intensity_destroyed(full-Across)"] = contrast(E1[0], tag, "full(xyzvAt)", "A-crossshuffle", ep, 5, 5, E1[1])
    fw["intensity_assoc(full-Ain)"] = contrast(E1[0], tag, "full(xyzvAt)", "A-inshuffle", ep, 5, 5, E1[1])
    fw["vel_vs_int(velocity-novel)"] = contrast(E1[0], tag, "velocity(xyzvt)", "no-velocity(xyzAt)", ep, 5, 5, E1[1])
    fw["vel_vs_geom(velocity-geometry)"] = contrast(E1[0], tag, "velocity(xyzvt)", "geometry(xyzt)", ep, 5, 5, E1[1])
    out[tag] = fw
h = {}
h["recovery(hist4-vsum)"] = contrast(E3[0], "MM-Fi", "v_hist4", "v_sum", 120, 5, 3, E3[1])
h["association(hist4-vshuf)"] = contrast(E3[0], "MM-Fi", "v_hist4", "v_hist4_vshuffled", 120, 5, 3, E3[1])
h["shuffled_retains(vshuf-vsum)"] = contrast(E3[0], "MM-Fi", "v_hist4_vshuffled", "v_sum", 120, 5, 3, E3[1])
h["sign_only(hist2-vsum)"] = contrast(E3[0], "MM-Fi", "v_hist2", "v_sum", 120, 5, 3, E3[1])
h["bins8_vs_4(hist8-hist4)"] = contrast(E3[0], "MM-Fi", "v_hist8", "v_hist4", 120, 5, 3, E3[1])
out["hist_ladder_MM-Fi"] = h
out["_note"] = "fold-wise leak-free reruns 2026-08-18; ci_two_level = crossed subject x seed"
json.dump(out, open(os.path.join(DOCS, "foldwise_stats.json"), "w"), indent=1)
for tag, d in out.items():
    if tag.startswith("_"): continue
    print("==", tag)
    for k, v in d.items():
        print(f"  {k:38s} {v['mean_pp']:6.2f}  subj{v['ci_subject']}  two{v['ci_two_level']}")
