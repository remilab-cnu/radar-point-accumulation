"""Reusable cross-subject evaluation comparing Doppler vs amplitude representations.

The claim under test: under user-independent (cross-subject) evaluation, Doppler
accumulation (XTD/YTD/ZTD) generalizes better than amplitude accumulation
(XTA/YTA/ZTA). Same fixed CNN and splits for both -> gap is due to representation.
"""
from __future__ import annotations
import numpy as np
from cnn import train_eval


def _eval_folds(X, y, subj, folds, n_cls, epochs, seeds):
    accs = []
    for test_subjects in folds:
        te = np.isin(subj, list(test_subjects))
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        fa = [train_eval(X[tr], y[tr], X[te], y[te], n_cls, epochs=epochs, seed=s) for s in seeds]
        accs.append(np.mean(fa))
    return float(np.mean(accs)), [float(a) for a in accs]


def subject_kfold_folds(subjects, k, seed=0):
    subs = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subs)
    return [list(g) for g in np.array_split(perm, k)]


def compare(Xd, Xa, y, subj, n_cls, folds, epochs=30, seeds=(0, 1, 2), label=""):
    """Run both representations over the SAME subject folds; return summary dict."""
    md, fd = _eval_folds(Xd, y, subj, folds, n_cls, epochs, seeds)
    ma, fa = _eval_folds(Xa, y, subj, folds, n_cls, epochs, seeds)
    print(f"\n=== {label} | cross-subject ({len(folds)} folds, seeds={list(seeds)}) ===")
    print(f"  Doppler  (XTD/YTD/ZTD): {md*100:.2f}%")
    print(f"  Amplitude(XTA/YTA/ZTA): {ma*100:.2f}%")
    print(f"  GAP (Doppler-Amplitude): +{(md-ma)*100:.2f} pts   | chance={100/n_cls:.1f}%")
    print(f"  per-fold Doppler:   {[f'{a*100:.1f}' for a in fd]}")
    print(f"  per-fold Amplitude: {[f'{a*100:.1f}' for a in fa]}")
    return {"doppler": md, "amplitude": ma, "gap": md - ma, "fold_d": fd, "fold_a": fa,
            "n_cls": n_cls, "label": label}
