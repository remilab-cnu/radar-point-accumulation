"""MM-Fi (40 subjects, 27 daily/rehab actions) cross-subject study — HAR extension.

Uses the dataset's documented S2 cross-subject split (32 train / 8 test) as the
primary protocol, plus a subject 5-fold for robustness.
"""
import os, time
import numpy as np
from spectra_dataset import mmfi_instances, build_cached
from experiment import compare, subject_kfold_folds
from preprocess import SpecConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "mmfi_extracted")
CACHE = os.path.join(HERE, "..", "data", "mmfi_cache.npz")

# documented S2 split test subjects (every 5th) -> hold-out fold
S2_TEST = [f"S{ i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]

t0 = time.time()
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); Xd, Xa, y, subj = z["Xd"], z["Xa"], z["y"], z["subj"]
else:
    insts = mmfi_instances(ROOT)  # one instance per (subject, action)
    print(f"collected {len(insts)} instances in {time.time()-t0:.1f}s")
    Xd, Xa, y, subj = build_cached(insts, SpecConfig(n_bins=32, n_frames=40), cache=CACHE)

n_cls = int(y.max()) + 1
print(f"dataset: Xd {Xd.shape} | subjects {len(set(subj.tolist()))} | classes {n_cls}")

# primary: documented S2 cross-subject split
compare(Xd, Xa, y, subj, n_cls=n_cls, folds=[S2_TEST],
        epochs=40, seeds=(0, 1, 2), label="MM-Fi S2 split (32 train / 8 test)")

# robustness: subject 5-fold
folds = subject_kfold_folds(subj, k=5, seed=0)
compare(Xd, Xa, y, subj, n_cls=n_cls, folds=folds,
        epochs=40, seeds=(0, 1), label="MM-Fi subject 5-fold")
