"""WINDOWING SENSITIVITY (autonomous, 2026-07-14): how much does the way you CUT the
continuous stream into a network input change gesture-recognition accuracy?

M2 showed prep-only (73%) ~ gesture-only (76.6%). This asks the general question the PI
raised: is radar gesture recognition (in)sensitive to the input segmentation/windowing?
Systematically vary the windowing, hold the classifier fixed (SmallCNN velocity-sum map,
subj-disjoint 4-fold, Infineon), and report the accuracy SPREAD across windowings.

Windowings (each -> one input per instance; cell_stats time-normalizes to T bins, so this
isolates WHICH frames/phase, with duration partly normalized by construction):
  gesture      [g0, g1]                 tight labeled gesture (the standard target)
  prep         [g0-P, g0)               approach only
  retract      (g1, g1+P]               withdraw only
  full_env     [g0-P, g1+P]             whole reach->gesture->retract envelope
  fixed_naive  [C-M, C+M], C=mid        LABEL-AGNOSTIC fixed window (ignores gesture location)
  whole        [0, F)                   entire recording, no cutting

Small spread => cutting barely matters (signal temporally diffuse/robust); large spread =>
segmentation matters. Env SMOKE=1: 3 users.
"""
import os, json
import numpy as np
from epenthesis_char import load, P
from rep_variants import cell_stats, compose, CAXES, kfold
from spectra_dataset import fit_ranges
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
NB = 32; SEEDS = (0,) if SMOKE else (0, 1, 2); EP = 4 if SMOKE else 40; WIDTH = 16; M = 12


def sub(df, lo, hi):
    w = df[(df.frame >= lo) & (df.frame < hi)]
    return w.reset_index(drop=True) if len(w) >= 3 else None


def windowings(df, g0, g1):
    F = int(df.frame.max()) + 1
    C = F // 2
    far = df[(df.frame < g0 - P) | (df.frame >= g1 + 1 + P)]   # LEAKAGE CONTROL: motion-free idle only
    far = far.reset_index(drop=True) if len(far) >= 3 else None
    return {
        "gesture":     sub(df, g0, g1 + 1),
        "prep":        sub(df, max(0, g0 - P), g0),
        "retract":     sub(df, g1 + 1, g1 + 1 + P),
        "full_env":    sub(df, max(0, g0 - P), g1 + 1 + P),
        "fixed_naive": sub(df, max(0, C - M), C + M),
        "whole":       sub(df, 0, F),
        "far_idle":    far,                                     # ~chance => no recording-structure leakage
    }


if __name__ == "__main__":
    print(f"WINDOWING SENSITIVITY  SMOKE={SMOKE}  P={P} M={M}", flush=True)
    recs = load()
    print(f"instances: {len(recs)}, users {sorted(set(t[4] for t in recs))}", flush=True)
    ranges = fit_ranges([t[0] for t in recs])
    names = ["gesture", "prep", "retract", "full_env", "fixed_naive", "whole", "far_idle"]
    # build per-windowing arrays (shared instance order; drop instances missing any window)
    per = {n: [] for n in names}; y = []; s = []
    for df, g0, g1, c, u in recs:
        ws = windowings(df, g0, g1)
        if any(ws[n] is None for n in names):
            continue
        for n in names:
            per[n].append(compose(cell_stats(ws[n], CAXES, ranges, nb=NB), CAXES, ["sum"]).astype(np.float32))
        y.append(c); s.append(u)
    y = np.array(y); s = np.array(s)
    print(f"usable instances (all windows present): {len(y)}", flush=True)
    folds = kfold(s, 2 if SMOKE else 4)
    res = {}
    for n in names:
        X = np.stack(per[n]); accs = []
        for te in folds:
            m = np.isin(s, [str(x) for x in te]); tr = ~m
            if tr.sum() == 0 or m.sum() == 0:
                continue
            for sd in SEEDS:
                a, _, _, _ = train_eval_full(X[tr], y[tr], X[m], y[m], 5, epochs=EP, seed=sd, width=WIDTH)
                accs.append(a)
        res[n] = (round(float(np.mean(accs)) * 100, 2), round(float(np.std(accs)) * 100, 2))
        print(f"  {n:12s}: {res[n][0]:6.2f}% (+-{res[n][1]:.1f})", flush=True)
    choice = [n for n in names if n != "far_idle"]            # far_idle is a leakage control, not a choice
    accs_only = [res[n][0] for n in choice]
    spread = max(accs_only) - min(accs_only)
    print(f"\nSPREAD across windowing CHOICES = {spread:.2f} pp  (small => cutting barely matters)", flush=True)
    print(f"  best={max(choice,key=lambda k:res[k][0])} worst={min(choice,key=lambda k:res[k][0])}", flush=True)
    print(f"  LEAKAGE CONTROL far_idle acc = {res['far_idle'][0]:.2f}% (chance 20; ~chance => whole-win is real motion signal)", flush=True)
    out = {"n_instances": len(y), "epochs": EP, "seeds": list(SEEDS), "smoke": SMOKE,
           "results": {n: {"acc": res[n][0], "std": res[n][1]} for n in names},
           "spread_pp": round(spread, 2)}
    json.dump(out, open(os.path.join(DOCS, f"windowing_sensitivity{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"wrote docs/windowing_sensitivity{'_smoke' if SMOKE else ''}.json", flush=True)
