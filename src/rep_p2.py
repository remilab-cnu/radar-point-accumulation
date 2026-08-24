"""P2 — Boundary predictor (repositioning memo §4, RUN-FIRST pair with P1).

Pre-registered hypothesis: per-class CNN recall deltas (velocity - occupancy) are
rank-correlated with a classifier-free per-class separability statistic
    predictor_c = overlap_c(occupancy) - overlap_c(velocity)
(overlap_c(q) = mean cosine similarity of class-c's mean map to other class means;
higher predictor => geometry more confusable for c => velocity should win more),
pooled across mHomeGes + Infineon + MM-Fi classes; and the predictor calls the
(velocity - occupancy) SIGN on MM-Fi FINE vs GROSS subsets IN ADVANCE (the prediction
is printed BEFORE the subset training results).

Protocol: frozen caches/manifests, single kfold, ep30 (correlation part; final-run
budget), batch 64, seeds {0,1,2}, preds saved. Subset test at ep30 AND ep40.
"""
import os, json
import numpy as np
from cnn import train_eval_preds
from rep_variants import cell_stats, norm, CAXES, infineon_recs
from spectra_dataset import fit_ranges
from rep_round3 import kfold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2)
FINE = [0, 3, 4, 6, 7, 8, 9, 14, 15, 16, 17, 18, 24, 26]     # from mmfi_why data-driven split
GROSS = [1, 2, 5, 10, 11, 12, 13, 19, 20, 21, 22, 23, 25]
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]


def overlap_per_class(X, y):
    """overlap_c = mean cosine similarity of class-c mean map to the other class means."""
    cls = np.unique(y)
    cm = np.stack([X[y == c].reshape((y == c).sum(), -1).mean(0) for c in cls])
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    S = cm @ cm.T
    return {int(c): float((S[i].sum() - 1.0) / (len(cls) - 1)) for i, c in enumerate(cls)}


def cnn_perclass_delta(Xv, Xo, y, subj, folds, ncls, ep=30):
    """per-class recall delta (velocity - occupancy) from trained CNNs, preds pooled."""
    rec = {}
    for arm, X in (("vel", Xv), ("occ", Xo)):
        yt_all, yp_all = [], []
        for te_s in folds:
            te = np.isin(subj, list(te_s)); tr = ~te
            for s in SEEDS:
                _, yt, yp = train_eval_preds(X[tr], y[tr], X[te], y[te], ncls, epochs=ep, seed=s)
                yt_all.append(yt); yp_all.append(yp)
        yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
        rec[arm] = {c: float((yp[yt == c] == c).mean()) for c in range(ncls)}
    return {c: rec["vel"][c] - rec["occ"][c] for c in range(ncls)}


def spearman_boot(x, y, n=4000, seed=0):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    rng = np.random.RandomState(seed); m = len(x); boots = []
    x = np.asarray(x); y = np.asarray(y)
    for _ in range(n):
        i = rng.randint(0, m, m)
        boots.append(spearmanr(x[i], y[i]).correlation)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(r), float(lo), float(hi)


out = {"protocol": {"ep_corr": 30, "ep_subsets": [30, 40], "batch": 64, "seeds": list(SEEDS)}}
pred_x, resp_y, tags = [], [], []

# ---------- mHomeGes (sum-arm cache) ----------
z = np.load(os.path.join(DATA, "mhomeges_gate.npz"), allow_pickle=True)
Xv, Xo, y, subj = z["Xv"], z["Xo"], z["y"], z["subj"]
folds = kfold(subj, 5)
ov_v, ov_o = overlap_per_class(Xv, y), overlap_per_class(Xo, y)
d = cnn_perclass_delta(Xv, Xo, y, subj, folds, 10)
out["mHomeGes"] = {"predictor": {c: ov_o[c] - ov_v[c] for c in ov_v}, "delta": d}
for c in ov_v:
    pred_x.append(ov_o[c] - ov_v[c]); resp_y.append(d[c]); tags.append(f"mH:{c}")
print("mHomeGes per-class done", flush=True)

# ---------- Infineon (rebuild sum arms from frozen pkl; response from decisive preds) ----------
recs = infineon_recs()
ranges = fit_ranges([t[0] for t in recs])
stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in recs]
Xv = np.stack([np.stack([norm(st[ax]["sum"]) for ax in CAXES]) for st in stats]).astype(np.float32)
Xo = np.stack([np.stack([norm(st[ax]["cnt"]) for ax in CAXES]) for st in stats]).astype(np.float32)
del stats
yI = np.array([t[1] for t in recs]); ov_v, ov_o = overlap_per_class(Xv, yI), overlap_per_class(Xo, yI)
pz = np.load(os.path.join(DOCS, "final_infineon_preds.npz"))
def perclass_from_final(arm):
    yt_all, yp_all = [], []
    for k in pz.files:
        if k.startswith(f"{arm}|ep30|"):
            _, yt, yp = pz[k]; yt_all.append(yt); yp_all.append(yp)
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    return {c: float((yp[yt == c] == c).mean()) for c in range(5)}
rv, ro = perclass_from_final("map_v_sum"), perclass_from_final("REF_occupancy")
dI = {c: rv[c] - ro[c] for c in range(5)}
out["Infineon"] = {"predictor": {c: ov_o[c] - ov_v[c] for c in ov_v}, "delta": dI}
for c in ov_v:
    pred_x.append(ov_o[c] - ov_v[c]); resp_y.append(dI[c]); tags.append(f"INF:{c}")
print("Infineon per-class done (response from decisive-run preds)", flush=True)
del Xv, Xo

# ---------- MM-Fi (sum-arm cache) ----------
z = np.load(os.path.join(DATA, "mmfi_gate.npz"), allow_pickle=True)
Xv, Xa, Xo, y, subj = z["Xv"], z["Xa"], z["Xo"], z["y"], z["subj"]
ov_v, ov_o = overlap_per_class(Xv, y), overlap_per_class(Xo, y)
dM = cnn_perclass_delta(Xv, Xo, y, subj, [S2], 27)
out["MM-Fi"] = {"predictor": {c: ov_o[c] - ov_v[c] for c in ov_v}, "delta": dM}
for c in ov_v:
    pred_x.append(ov_o[c] - ov_v[c]); resp_y.append(dM[c]); tags.append(f"MMFi:{c}")
print("MM-Fi per-class done", flush=True)

# ---------- pooled rank correlation ----------
r, lo, hi = spearman_boot(pred_x, resp_y)
out["pooled_spearman"] = {"r": r, "ci": [lo, hi], "n_classes": len(pred_x)}
print(f"\nPOOLED Spearman r = {r:.3f}  95%CI[{lo:.3f},{hi:.3f}]  (n={len(pred_x)} classes)", flush=True)

# ---------- PRE-REGISTERED prediction for FINE/GROSS (stated BEFORE training) ----------
def subset_predictor(classes):
    m = np.isin(y, classes)
    yy = y[m]
    return (np.mean(list(overlap_per_class(Xo[m], yy).values())) -
            np.mean(list(overlap_per_class(Xv[m], yy).values())))
pf, pg = subset_predictor(FINE), subset_predictor(GROSS)
pred_sign = {"FINE": "velocity>occupancy" if pf > 0 else "occupancy>velocity",
             "GROSS": "velocity>occupancy" if pg > 0 else "occupancy>velocity"}
out["subset_prediction"] = {"FINE_predictor": pf, "GROSS_predictor": pg, "predicted_sign": pred_sign}
print(f"PRE-REGISTERED prediction: FINE predictor={pf:+.4f} -> {pred_sign['FINE']}; "
      f"GROSS predictor={pg:+.4f} -> {pred_sign['GROSS']}", flush=True)

# ---------- held-out test: FINE/GROSS subset retrains ----------
sub_res = {}
for name, classes in (("FINE", FINE), ("GROSS", GROSS)):
    m = np.isin(y, classes); remap = {c: i for i, c in enumerate(sorted(classes))}
    yy = np.array([remap[c] for c in y[m]]); ss = subj[m]
    for ep in (30, 40):
        accs = {}
        for arm, X in (("vel", Xv[m]), ("occ", Xo[m]), ("int", Xa[m])):
            a = []
            for s in SEEDS:
                te = np.isin(ss, S2); tr = ~te
                acc, _, _ = train_eval_preds(X[tr], yy[tr], X[te], yy[te], len(classes), epochs=ep, seed=s)
                a.append(acc)
            accs[arm] = (float(np.mean(a)) * 100, float(np.std(a)) * 100)
        sub_res[f"{name}|ep{ep}"] = accs
        print(f"  {name} ep{ep}: vel={accs['vel'][0]:.2f} occ={accs['occ'][0]:.2f} int={accs['int'][0]:.2f} "
              f"-> observed {'velocity>occupancy' if accs['vel'][0]>accs['occ'][0] else 'occupancy>velocity'}", flush=True)
out["subset_results"] = sub_res
json.dump(out, open(os.path.join(DOCS, "p2_boundary_predictor.json"), "w"), indent=1)
print("\nwrote docs/p2_boundary_predictor.json", flush=True)
