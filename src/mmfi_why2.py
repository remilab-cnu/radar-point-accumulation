"""The REAL mechanism (motion-granularity was refuted): does geometry become
class-discriminative on whole-body HAR while staying uninformative for co-located
hand gestures? Measure per-arm class SEPARABILITY on the raw maps (no training) for
both datasets, and visualize class-mean geometry maps (overlap vs distinctness).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")


def load(name):
    z = np.load(os.path.join(DATA, name), allow_pickle=True)
    return z["Xv"], z["Xa"], z["Xo"], z["y"]


def separability(X, y):
    """Fisher-style ratio: mean inter-class centroid distance / mean within-class scatter.
    Higher => classes are easier to tell apart in this representation."""
    F = X.reshape(len(X), -1).astype(np.float64)
    cls = np.unique(y)
    cents = np.stack([F[y == c].mean(0) for c in cls])
    intra = np.mean([np.linalg.norm(F[y == c] - cents[i], axis=1).mean() for i, c in enumerate(cls)])
    inter = np.mean([np.linalg.norm(cents[i] - cents[j]) for i in range(len(cls)) for j in range(i + 1, len(cls))])
    return inter / (intra + 1e-9)


def class_overlap(Xo, y):
    """Mean pairwise cosine SIMILARITY of class-mean geometry maps.
    High => geometry maps look alike across classes (co-located) => geometry cannot separate them."""
    cls = np.unique(y)
    cm = np.stack([Xo[y == c].reshape((y == c).sum(), -1).mean(0) for c in cls])
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    S = cm @ cm.T
    iu = np.triu_indices(len(cls), 1)
    return S[iu].mean()


print("=== SEPARABILITY (inter/intra ratio; higher = more class-separable) ===")
print(f"{'dataset':20s} {'velocity':>10s} {'geometry':>10s} {'intensity':>10s} {'geom class-overlap':>20s}")
for name, tag in (("mhomeges_gate.npz", "mHomeGes (gesture)"), ("mmfi_gate.npz", "MM-Fi (whole-body)")):
    Xv, Xa, Xo, y = load(name)
    sv, sg, si = separability(Xv, y), separability(Xo, y), separability(Xa, y)
    ov = class_overlap(Xo, y)
    print(f"{tag:20s} {sv:10.3f} {sg:10.3f} {si:10.3f} {ov:20.3f}")

print("\n[AUDIT CORRECTION] The Fisher inter/intra ratio above is NOT a reliable mechanism"
      "\nmetric: it ties velocity=geometry (0.171) on gestures yet the CNN shows +13 for velocity."
      "\nUse mmfi_mechanism_fix.py instead: nearest-class-mean (NCM) cross-subject accuracy"
      "\nreproduces the CNN direction on BOTH datasets (velocity best on gestures, geometry best"
      "\non whole-body HAR). The cos-overlap metric explains only the gesture side (velocity"
      "\nclass-means near-orthogonal at 0.100 vs geometry 0.917), not the HAR reversal.")

print("\n=== sensor velocity resolution (secondary confound) ===")
print("  mHomeGes / M-Gesture : TI IWR1443, per-point Doppler step ~0.356 m/s (finer)")
print("  MM-Fi                : TI IWR6843 60GHz, Doppler step ~0.604 m/s (coarser -> blunter velocity)")

# ---- visualize class-mean z-occupancy (geometry) maps: overlap (gesture) vs distinct (HAR) ----
fig, axs = plt.subplots(2, 1, figsize=(11, 6))
for row, (name, tag, ncol) in enumerate([("mhomeges_gate.npz", "mHomeGes (gesture): class-mean Z-occupancy", 10),
                                          ("mmfi_gate.npz", "MM-Fi (whole-body): class-mean Z-occupancy", 12)]):
    Xv, Xa, Xo, y = load(name)
    cls = np.unique(y)[:ncol]
    strip = np.concatenate([np.abs(Xo[y == c, 2]).mean(0) for c in cls], axis=1)  # ZTO per class, side by side
    axs[row].imshow(strip, aspect="auto", origin="lower", cmap="magma")
    axs[row].set_title(tag, fontsize=11)
    axs[row].set_yticks([]); axs[row].set_xticks([])
    axs[row].set_xlabel("classes side-by-side (each block = one class' mean height-occupancy over time)")
fig.suptitle("Geometry maps overlap across hand-gesture classes but are distinct across whole-body classes", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(DOCS, "mmfi_why_mechanism.png"); fig.savefig(out, dpi=120)
print("\nsaved", out)
