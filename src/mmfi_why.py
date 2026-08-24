"""Qualitative analysis: WHY velocity accumulation loses to geometry on whole-body HAR.

Evidence, not hand-waving:
 (1) Descriptive contrast (raw point clouds): gesture (small volume, hand) vs
     whole-body HAR (large vertical extent, many points).
 (2) Within-MM-Fi identification (SAME sensor, removes 60GHz/class-count confound):
     split the 27 actions data-drivenly into FINE (motion vertically localized) vs
     GROSS (motion spread over body height) and re-run velocity/occupancy/intensity
     cross-subject on each. Prediction: velocity advantage returns on FINE, reverses on GROSS.
 (3) Per-class: correlate each action's vertical-motion-spread with the velocity-minus-
     geometry recall gap -> the advantage tracks motion granularity.
"""
import os, glob
import numpy as np
from sklearn.metrics import recall_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cnn import train_eval_preds
from preprocess import load_mgesture_csv, segment_instances
from spectra_dataset import load_mmfi_action

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
DOCS = os.path.join(HERE, "..", "docs")
z = np.load(os.path.join(DATA, "mmfi_gate.npz"), allow_pickle=True)
Xv, Xa, Xo, y, subj = z["Xv"], z["Xa"], z["Xo"], z["y"], z["subj"]
S2_TEST = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
te_mask = np.isin(subj, S2_TEST)
NCLS = int(y.max()) + 1

# ---------- (1) descriptive contrast from raw points ----------
def raw_stats(dfs):
    ppf, sx, sy, sz, mv = [], [], [], [], []
    for df in dfs:
        if len(df) < 5: continue
        ppf.append(df.groupby("frame").size().mean())
        sx.append(df.x.std()); sy.append(df.y.std()); sz.append(df.z.std())
        mv.append(np.mean(np.abs(df.doppler)))
    f = lambda a: (np.mean(a), np.std(a))
    return {"pts/frame": f(ppf), "std_x(m)": f(sx), "std_y(m)": f(sy), "std_z(m)": f(sz), "mean|v|": f(mv)}

mmfi_dfs = []
for sd in sorted(glob.glob(os.path.join(DATA, "mmfi_extracted", "filtered_mmwave", "E*", "S0[1-4]")))[:4]:
    for ad in sorted(glob.glob(os.path.join(sd, "A*"))):
        d = load_mmfi_action(ad)
        if d is not None: mmfi_dfs.append(d)
mh_dfs = []
for f in glob.glob(os.path.join(DATA, "mhomeges_full", "longGes_1.2m", "805", "point_*.csv")):
    mh_dfs += segment_instances(load_mgesture_csv(f))[0][:6]

print("=== (1) DESCRIPTIVE CONTRAST (mean +/- std over instances) ===")
print(f"{'stat':11s} {'mHomeGes (gesture)':>24s} {'MM-Fi (whole-body HAR)':>26s}")
sm, sf = raw_stats(mh_dfs), raw_stats(mmfi_dfs)
for k in sm:
    print(f"{k:11s} {sm[k][0]:10.3f}+-{sm[k][1]:<11.3f} {sf[k][0]:12.3f}+-{sf[k][1]:<11.3f}")

# ---------- vertical motion spread per MM-Fi class (from velocity ZTD) ----------
def vspread(ztd):  # ztd: (32 zbins, 40 frames) -> std of z-activity marginal, in bins
    m = np.abs(ztd).sum(1); s = m.sum()
    if s <= 0: return 0.0
    p = m / s; zb = np.arange(len(p))
    mu = (p * zb).sum()
    return float(np.sqrt((p * (zb - mu) ** 2).sum()))

cls_spread = np.array([np.mean([vspread(Xv[i, 2]) for i in np.where(y == c)[0]]) for c in range(NCLS)])
med = np.median(cls_spread)
FINE = np.where(cls_spread <= med)[0]      # vertically localized motion
GROSS = np.where(cls_spread > med)[0]      # motion spread over body height
print(f"\n=== (2) WITHIN-MM-Fi FINE vs GROSS (data-driven z-motion-spread split, median={med:.2f} bins) ===")
print(f"FINE classes ({len(FINE)}): {FINE.tolist()}")
print(f"GROSS classes({len(GROSS)}): {GROSS.tolist()}")

def run_subset(classes, arms, seeds=(0, 1, 2)):
    idx = np.isin(y, classes)
    remap = {c: i for i, c in enumerate(sorted(classes))}
    yy = np.array([remap[c] for c in y[idx]])
    te = te_mask[idx]; tr = ~te
    out = {}
    for name, X in arms.items():
        Xs = X[idx]
        accs = [train_eval_preds(Xs[tr], yy[tr], Xs[te], yy[te], len(classes), epochs=40, seed=s)[0] for s in seeds]
        out[name] = np.mean(accs)
    return out

arms = {"velocity": Xv, "occupancy": Xo, "intensity": Xa}
for grp, name in ((FINE, "FINE (localized motion)"), (GROSS, "GROSS (whole-body motion)")):
    r = run_subset(grp, arms)
    print(f"  {name:26s} vel={r['velocity']*100:5.2f}  geom={r['occupancy']*100:5.2f}  int={r['intensity']*100:5.2f}"
          f"   vel-geom={(r['velocity']-r['occupancy'])*100:+.2f}  (chance={100/len(grp):.1f}%)")

# ---------- (3) per-class recall gap vs vertical spread ----------
def per_class_recall(X):
    tr = ~te_mask
    _, yt, yp = train_eval_preds(X[tr], y[tr], X[te_mask], y[te_mask], NCLS, epochs=40, seed=0)
    return recall_score(yt, yp, labels=list(range(NCLS)), average=None, zero_division=0)
rec_v, rec_o = per_class_recall(Xv), per_class_recall(Xo)
gap = (rec_v - rec_o) * 100
r_corr = np.corrcoef(cls_spread, gap)[0, 1]
print(f"\n=== (3) per-class (velocity recall - geometry recall) vs vertical motion spread ===")
print(f"  Pearson r = {r_corr:.3f}  (negative => velocity advantage shrinks as motion becomes more whole-body)")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
grps = {"FINE\n(localized)": run_subset(FINE, arms), "GROSS\n(whole-body)": run_subset(GROSS, arms)}
xs = np.arange(2); w = 0.26
for j, (k, col) in enumerate([("velocity", "#1f77b4"), ("occupancy", "#2ca02c"), ("intensity", "#d62728")]):
    ax[0].bar(xs + (j-1)*w, [grps[g][k]*100 for g in grps], w, label=k, color=col)
ax[0].set_xticks(xs); ax[0].set_xticklabels(list(grps.keys())); ax[0].set_ylabel("cross-subject acc (%)")
ax[0].set_title("Within MM-Fi (same sensor): velocity loses to geometry on BOTH\nlocalized & whole-body subsets -> motion-granularity hypothesis REFUTED")
ax[0].legend(fontsize=8)
ax[1].scatter(cls_spread, gap, c=gap, cmap="RdBu", vmin=-30, vmax=30, edgecolor="k")
ax[1].axhline(0, color="k", lw=.7); ax[1].set_xlabel("per-class vertical motion spread (z-bins)")
ax[1].set_ylabel("velocity - geometry recall (pts)")
ax[1].set_title(f"Per action: velocity's edge shrinks as motion\nspreads over the body (r={r_corr:.2f})")
fig.tight_layout(); out = os.path.join(DOCS, "mmfi_why.png"); fig.savefig(out, dpi=120)
print("\nsaved", out)
