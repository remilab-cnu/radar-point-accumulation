"""M2 FIX: replace the Fisher inter/intra 'mechanism' (which tied velocity=geometry on
gestures, failing to reproduce the +13 CNN win) with honest, classifier-grounded
separability measures computed identically for every arm on BOTH datasets:
  - cosine class-overlap of class-mean maps (low overlap => arm can separate classes)
  - nearest-class-mean (NCM) cross-subject accuracy on the raw flattened maps
    (a simple, parameter-free linear-ish separability that should track the CNN direction)
Report alongside the deep-CNN accuracy so the reader sees which simple metric tracks it.
"""
import os, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
CNN = {  # deep-CNN cross-subject accuracy already measured (for reference)
    "mHomeGes": {"velocity": 67.49, "geometry": 54.38, "intensity": 44.60},
    "MM-Fi":    {"velocity": 70.37, "geometry": 77.47, "intensity": 77.01},
}


def cos_overlap(X, y):
    cls = np.unique(y)
    cm = np.stack([X[y == c].reshape((y == c).sum(), -1).mean(0) for c in cls])
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-9)
    S = cm @ cm.T
    return float(S[np.triu_indices(len(cls), 1)].mean())


def ncm_xsub(X, y, subj, folds):
    F = X.reshape(len(X), -1).astype(np.float64)
    accs = []
    for te_s in folds:
        te = np.isin(subj, list(te_s)); tr = ~te
        cls = np.unique(y[tr])
        cents = np.stack([F[tr][y[tr] == c].mean(0) for c in cls])
        d = ((F[te][:, None, :] - cents[None]) ** 2).sum(-1)   # [Nte, C]
        pred = cls[d.argmin(1)]
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs) * 100)


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


for name, cache, folds_fn in (("mHomeGes", "mhomeges_gate.npz", lambda subj: kfold(subj, 5)),
                              ("MM-Fi", "mmfi_gate.npz", lambda subj: [S2])):
    z = np.load(os.path.join(DATA, cache), allow_pickle=True)
    arms = {"velocity": z["Xv"], "geometry": z["Xo"], "intensity": z["Xa"]}
    y, subj = z["y"], z["subj"]; folds = folds_fn(subj)
    print(f"\n=== {name} ===")
    print(f"  {'arm':10s} {'CNN acc':>8s} {'NCM xsub':>9s} {'cos-overlap':>12s}")
    for arm, X in arms.items():
        print(f"  {arm:10s} {CNN[name][arm]:8.1f} {ncm_xsub(X, y, subj, folds):9.2f} {cos_overlap(X, y):12.3f}")

print("\nHonest reading: report the metric that tracks the CNN direction; if none fully does,")
print("state that the advantage requires the nonlinear CNN and rely on the class-mean VISUAL")
print("(gesture geometry maps overlap; HAR geometry maps distinct) as qualitative mechanism.")
