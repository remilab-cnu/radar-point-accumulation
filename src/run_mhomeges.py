"""mHomeGes (25 subjects, 10 arm gestures, short-range) cross-subject study.

Best regime match to the original in-cabin setting (short range). MIT license.
"""
import os, time, sys
import numpy as np
from spectra_dataset import mhomeges_instances, build_cached, MHOMEGES_CLASSES
from experiment import compare, subject_kfold_folds
from preprocess import SpecConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "mhomeges_full")
CACHE = os.path.join(HERE, "..", "data", "mhomeges_cache.npz")

# subset of distances keeps it tractable while spanning the short-range regime
DISTANCES = None if "--all-dist" in sys.argv else {"1.2", "1.5", "1.8", "2.1", "2.4"}

t0 = time.time()
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); Xd, Xa, y, subj = z["Xd"], z["Xa"], z["y"], z["subj"]
else:
    insts = mhomeges_instances(ROOT, distances=DISTANCES)
    print(f"collected {len(insts)} instances in {time.time()-t0:.1f}s (distances={DISTANCES})")
    Xd, Xa, y, subj = build_cached(insts, SpecConfig(n_bins=32, n_frames=40), cache=CACHE)

print(f"dataset: Xd {Xd.shape} | subjects {len(set(subj.tolist()))} | classes {len(MHOMEGES_CLASSES)}")
for c, name in enumerate(MHOMEGES_CLASSES):
    print(f"  {name:8s}: {(y==c).sum()}")

folds = subject_kfold_folds(subj, k=5, seed=0)
print("subject folds:", [len(f) for f in folds])
compare(Xd, Xa, y, subj, n_cls=len(MHOMEGES_CLASSES), folds=folds,
        epochs=25, seeds=(0, 1), label="mHomeGes (25 subj, 10 gestures)")
