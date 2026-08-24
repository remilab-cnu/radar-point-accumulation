"""End-to-end: M-Gesture sample -> Doppler vs amplitude, cross-subject (LOSO).

The headline claim: under cross-subject (user-independent) evaluation, the
Doppler-accumulation representation degrades far less than amplitude-accumulation.
"""
import os, time
import numpy as np
from spectra_dataset import build_mgesture
from cnn import train_eval
from preprocess import SpecConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "mgesture_sample", "long_SEP")
CACHE = os.path.join(HERE, "..", "data", "mgesture_sample_cache.npz")

cfg = SpecConfig(n_bins=32, n_frames=40)
CLASSES = ("knock", "lswipe", "rswipe", "rotate")

t0 = time.time()
Xd, Xa, y, subj = build_mgesture(DATA, cfg, cache=CACHE, classes=CLASSES)
print(f"built {len(y)} instances in {time.time()-t0:.1f}s | Xd {Xd.shape} | classes {CLASSES}")
subjects = sorted(set(subj.tolist()))
print("subjects:", subjects)
for c, name in enumerate(CLASSES):
    print(f"  class {name:8s}: {(y==c).sum()} instances")

n_cls = len(CLASSES)
EPOCHS = 30
SEEDS = [0, 1, 2]


def loso(X, tag):
    """Leave-one-subject-out cross-subject accuracy, averaged over seeds."""
    fold_acc = {}
    for held in subjects:
        te = subj == held
        tr = ~te
        accs = [train_eval(X[tr], y[tr], X[te], y[te], n_cls, epochs=EPOCHS, seed=s)
                for s in SEEDS]
        fold_acc[held] = float(np.mean(accs))
    mean = float(np.mean(list(fold_acc.values())))
    print(f"\n[{tag}] LOSO cross-subject acc = {mean*100:.2f}%")
    for h in subjects:
        print(f"    held-out {h}: {fold_acc[h]*100:.2f}%")
    return mean, fold_acc


print("\n" + "=" * 60)
md, fd = loso(Xd, "Doppler  (XTD/YTD/ZTD)")
ma, fa = loso(Xa, "Amplitude(XTA/YTA/ZTA)")
print("\n" + "=" * 60)
print(f"SUMMARY  Doppler={md*100:.2f}%  Amplitude={ma*100:.2f}%  gap=+{(md-ma)*100:.2f} pts")
print(f"(chance = {100/n_cls:.1f}%)")
