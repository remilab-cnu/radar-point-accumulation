"""Fit-to-interpolation test for the 8.3 pp claim (2026-08-11).

WHY. The paper's primary number is the dimensionality-matched velocity
contribution: full(xyzvAt) minus v-crossshuffle on mHomeGes. Both arms sit
just under the 0.95 training-accuracy gate (0.9576 vs 0.9374), so a referee can
ask whether the 8.3 pp is a fitting artifact rather than an information effect.
The gate cannot answer that, because fit quality and information content are
confounded across arms by construction.

DESIGN (agreed with an independent reviewer after a literature pass).
E1  Paired fit-to-interpolation. Same two arms, dropout removed, a decaying
    schedule out to 360 epochs, and PAIRED randomness: the two arms share
    initialization, minibatch order and the frozen point tensors, so the only
    difference is the velocity content. Selection is on TRAINING accuracy only;
    test accuracy is recorded but never used to stop or choose.
      - both arms interpolate and the paired gap stays near 8.3 pp
        -> the gap is not caused by failure to fit. This is the claim-protecting
           outcome.
      - the shuffled arm climbs and the gap shrinks materially
        -> the published number was optimization-confounded and must be replaced.
      - the shuffled arm plateaus while full interpolates
        -> an information/architecture limit, and "UNDERFIT" is the wrong label
           for that cell.
E2  Grouped random labels (Zhang et al., CACM 2021), which asks the narrow
    question "can this architecture separate these particular inputs at all?".
    Successful fitting is strong evidence of empirical separability; FAILURE is
    weak evidence, because arbitrary labels are harder to optimize than real
    structure (Arpit et al., ICML 2017). Exact-duplicate inputs are grouped and
    given one shared label, otherwise the diagnostic manufactures impossible
    conflicts.
E3  Exact collision ceiling, computed on CPU from the frozen tensors:
        A_collision = (1/N) * sum_g max_c n_{g,c}
    over groups g of identical post-preprocessing inputs. This is the highest
    training accuracy ANY deterministic classifier on that representation can
    reach, so it separates an exact non-separability ceiling from slow fitting.

Folds are chosen from the existing 5x3 run table rather than arbitrarily:
fold 1 is representative (76-79% across seeds) and fold 2 is the hard fold
(40-54% across all three seeds, i.e. fold-driven, not seed collapse).

Out: docs/interpolation_test.json  (+ per-epoch curves)
Run: python3 rep_interpolation.py            # full
     SMOKE=1 python3 rep_interpolation.py     # 1 fold, 1 seed, 40 epochs
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectra_dataset import mhomeges_instances, fit_ranges
from rep_round3 import kfold
from pointset_models import build_point_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"

V = 3                                   # velocity column in [x,y,z,v,A,t]
ARMS = {"full(xyzvAt)": ([0, 1, 2, 3, 4, 5], None),
        "v-crossshuffle": ([0, 1, 2, 3, 4, 5], "cross")}
FOLDS_USED = [0] if SMOKE else [1, 2]
SEEDS = (0,) if SMOKE else (0, 1, 2)
# 1e-3 -> 120, 1e-4 -> 240, 1e-5 -> 360; decay so the run can actually interpolate
SCHEDULE = [(40, 1e-3)] if SMOKE else [(120, 1e-3), (120, 1e-4), (120, 1e-5)]
EVAL_EVERY = 5


class DeepSetsNoDrop(nn.Module):
    """Identical to the paper's DeepSets except dropout p=0, so a training-accuracy
    ceiling cannot be blamed on the regularizer."""

    def __init__(self, in_dim=6, n_cls=10, w=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, w), nn.ReLU(True),
                                 nn.Linear(w, 2 * w), nn.ReLU(True))
        self.rho = nn.Sequential(nn.Linear(4 * w, 2 * w), nn.ReLU(True),
                                 nn.Linear(2 * w, n_cls))

    def forward(self, x, m):
        h = self.phi(x)
        hmax = h.masked_fill(~m[..., None], -1e9).max(1).values
        has = m.any(1, keepdim=True)
        hmax = torch.where(has, hmax, torch.zeros_like(hmax))
        hmean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.rho(torch.cat([hmax, hmean], -1))


def shuffled_v(X, M, mode, seed):
    """Byte-identical control to rep_converge_point.shuffled_v, so the arms match
    the published ladder."""
    rng = np.random.default_rng(10_000 + seed)
    v = X[:, :, V].copy(); valid = M > 0.5; n = X.shape[0]
    if mode == "in":
        for i in range(n):
            idx = np.where(valid[i])[0]
            v[i, idx] = v[i, idx][rng.permutation(len(idx))]
    else:
        donor = rng.permutation(n)
        donor[donor == np.arange(n)] = (donor[donor == np.arange(n)] + 1) % n
        for i in range(n):
            idx = np.where(valid[i])[0]
            dv = X[donor[i], :, V][valid[donor[i]]]
            v[i, idx] = 0.0 if len(dv) == 0 else rng.choice(dv, size=len(idx), replace=True)
    return v


def collision_ceiling(Xc, M, y):
    """Exact-duplicate ceiling on the representation actually fed to the network."""
    key = np.ascontiguousarray(np.concatenate([Xc.reshape(len(y), -1),
                                              M.reshape(len(y), -1)], 1)).view(np.uint8)
    import hashlib
    h = [hashlib.blake2b(k.tobytes(), digest_size=16).digest() for k in key]
    groups = {}
    for i, hh in enumerate(h):
        groups.setdefault(hh, []).append(i)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    correct = 0
    for idx in groups.values():
        lbl, cnt = np.unique(y[idx], return_counts=True)
        correct += cnt.max()
    conflict = sum(1 for idx in dup.values() if len(np.unique(y[idx])) > 1)
    return dict(ceiling=float(correct / len(y)), n_groups=len(groups),
                n_duplicate_groups=len(dup), n_conflicting_groups=conflict)


def eval_pass(model, loader):
    model.eval(); lossf = nn.CrossEntropyLoss(reduction="sum")
    c = n = 0; tot = 0.0
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            o = model(xb, mb)
            tot += lossf(o, yb).item(); c += (o.argmax(1) == yb).sum().item(); n += yb.numel()
    return c / max(n, 1), tot / max(n, 1)


def train_to_interpolation(Xc, M, y, tr, te, n_cls, in_dim, seed, tag, curves):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DeepSetsNoDrop(in_dim=in_dim, n_cls=n_cls).to(DEVICE)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)          # paired minibatch order
    dl = DataLoader(TensorDataset(torch.from_numpy(Xc[tr]), torch.from_numpy(M[tr]),
                                  torch.from_numpy(y[tr])), batch_size=64, shuffle=True, generator=g)
    tr_eval = DataLoader(TensorDataset(torch.from_numpy(Xc[tr]), torch.from_numpy(M[tr]),
                                       torch.from_numpy(y[tr])), batch_size=256)
    te_eval = DataLoader(TensorDataset(torch.from_numpy(Xc[te]), torch.from_numpy(M[te]),
                                       torch.from_numpy(y[te])), batch_size=256)
    ep = 0; hist = []
    for n_ep, lr in SCHEDULE:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_ep):
            model.train()
            for xb, mb, yb in dl:
                xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); lossf(model(xb, mb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ep += 1
            if ep % EVAL_EVERY == 0 or ep == 1:
                tra, trl = eval_pass(model, tr_eval)
                tea, _ = eval_pass(model, te_eval)
                hist.append(dict(epoch=ep, lr=lr, train_acc=round(tra, 5),
                                 train_loss=round(trl, 5), test_acc=round(tea, 5)))
                print(f"    [{tag}] ep{ep:3d} lr{lr:g} train={tra:.4f} loss={trl:.4f} test={tea:.4f}",
                      flush=True)
    curves[tag] = hist
    tra, trl = eval_pass(model, tr_eval); tea, _ = eval_pass(model, te_eval)
    return dict(final_train_acc=round(tra, 5), final_train_loss=round(trl, 5),
                final_test_acc=round(tea, 5), epochs=ep)


if __name__ == "__main__":
    t0 = time.time()
    insts = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    subj_all = np.array([t[2] for t in insts])
    if SMOKE:                                    # 3 subjects, mirroring rep_converge_point
        keep = sorted(set(subj_all))[:3]
        insts = [t for t in insts if t[2] in keep]
        subj_all = np.array([t[2] for t in insts])
    ranges = fit_ranges([t[0] for t in insts])
    X, M, y, s = build_point_tensors(insts, ranges)
    # kfold(subj, 5) with seed=0 reproduces the published fold membership exactly,
    # so fold indices here mean the same thing as in converge_point.json
    folds = kfold(subj_all, 2 if SMOKE else 5)
    print(f"mHomeGes: {len(y)} instances, X{X.shape}, {len(folds)} folds", flush=True)

    out = {"_purpose": __doc__.split("\n")[0], "_device": DEVICE, "_smoke": SMOKE,
           "_schedule": SCHEDULE, "_folds_used": FOLDS_USED, "_seeds": list(SEEDS),
           "E1_interpolation": {}, "E2_random_labels": {}, "E3_collision_ceiling": {}}
    curves = {}

    for fi in FOLDS_USED:
        te = np.isin(s, list(folds[fi])); tr = ~te
        for aname, (cols, vmode) in ARMS.items():
            # E3 first: the ceiling is a property of the representation, not of training
            Xa = X if vmode is None else X.copy()
            if vmode is not None:
                Xa[:, :, V] = shuffled_v(X, M, vmode, SEEDS[0])
            Xc = np.ascontiguousarray(Xa[..., cols])
            ck = f"fold{fi}|{aname}"
            out["E3_collision_ceiling"][ck] = collision_ceiling(Xc[tr], M[tr], y[tr])
            print(f"  E3 {ck}: {out['E3_collision_ceiling'][ck]}", flush=True)

            for sd in SEEDS:
                Xa = X if vmode is None else X.copy()
                if vmode is not None:
                    Xa[:, :, V] = shuffled_v(X, M, vmode, sd)
                Xc = np.ascontiguousarray(Xa[..., cols])
                tag = f"fold{fi}|{aname}|seed{sd}"
                print(f"  E1 {tag}", flush=True)
                out["E1_interpolation"][tag] = train_to_interpolation(
                    Xc, M, y, tr, te, 10, len(cols), sd, tag, curves)
                json.dump(out, open(os.path.join(DOCS, "interpolation_test.json"), "w"), indent=1)
                json.dump(curves, open(os.path.join(DOCS, "interpolation_curves.json"), "w"))

    # E2: grouped random labels on the representative fold, one seed per arm
    fi = FOLDS_USED[0]
    te = np.isin(s, list(folds[fi])); tr = ~te
    for aname, (cols, vmode) in ARMS.items():
        Xa = X if vmode is None else X.copy()
        if vmode is not None:
            Xa[:, :, V] = shuffled_v(X, M, vmode, SEEDS[0])
        Xc = np.ascontiguousarray(Xa[..., cols])
        rng = np.random.default_rng(999)
        yr = y.copy()
        yr[tr] = rng.permutation(y[tr])          # class-count preserving
        tag = f"fold{fi}|{aname}|randomlabels"
        print(f"  E2 {tag}", flush=True)
        out["E2_random_labels"][tag] = train_to_interpolation(
            Xc, M, yr, tr, te, 10, len(cols), SEEDS[0], tag, curves)
        json.dump(out, open(os.path.join(DOCS, "interpolation_test.json"), "w"), indent=1)
        json.dump(curves, open(os.path.join(DOCS, "interpolation_curves.json"), "w"))

    out["_wall_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(os.path.join(DOCS, "interpolation_test.json"), "w"), indent=1)
    print(f"\nwrote docs/interpolation_test.json in {out['_wall_minutes']} min", flush=True)
