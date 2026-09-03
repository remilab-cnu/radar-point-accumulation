"""E9+E10 (PI-endorsed external review, 2026-08-24).

E9 — shared-normalization positive/negative-sum control. The per-channel
max-abs normalization means the pos and neg channels are rescaled by
different factors, so the earlier claim "changes nothing but the
cancellation" was too strong: independent per-sign rescaling rides along.
Here both channels are divided by ONE shared scale (per axis, per instance:
max of the two channels' absolute maxima), so the scalar map is recoverable
from the pair by a fixed [1,-1] combination and the ONLY change vs the
scalar sum is that cancellation no longer occurs before the network sees
the cells.

E10 — within-dataset C sweep. Coherence is enforced synthetically: for a
fraction f of each instance's points, the velocity sign is flipped to match
the sign of its cell's signed sum (majority direction), pushing C upward
with f on MM-Fi (natural C = 0.488). At each f we train the scalar sum and
the shared-normalized sign split; if cancellation exposure drives the gap,
the gap must shrink monotonically as C rises WITHIN one dataset, one task,
one sensor, one pipeline.

Fold-wise ranges, SmallCNN w32, lr 1e-3, batch 64, seeds {0,1,2}, ep120,
per-instance preds saved; per-f C values written alongside.
Out: docs/shared_norm_sweep.json + _preds.npz.  SMOKE=1 tiny check.
"""
import os, json, time
import numpy as np
from rep_variants import cell_stats, compose, norm, CAXES, kfold, T
from spectra_dataset import fit_ranges, mmfi_instances
from rep_hist_ladder import axis_bins, NB
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
BUDGET = 4 if SMOKE else 120
WIDTH = 8 if SMOKE else 32
FLIPS = (0.0,) if SMOKE else (0.0, 0.25, 0.5, 0.75, 1.0)


def coherence_enforce(inst, f, seed):
    """Flip the sign of a fraction f of points toward their cell's majority
    Doppler direction (computed on the x-axis map geometry of the instance)."""
    if f <= 0:
        return inst
    df = inst.copy()
    v = df["doppler"].values.astype(float).copy()
    rng = np.random.default_rng(50_000 + seed)
    # cell majority via the instance's own x-t binning (ranges per instance)
    ranges = fit_ranges([inst])
    ab, ti = axis_bins(inst, ranges)
    bi, m = ab["x"]
    key = np.where(m, bi * 1000 + ti, -1)
    sums = {}
    for k, vv in zip(key, v):
        if k >= 0:
            sums[k] = sums.get(k, 0.0) + vv
    pick = rng.random(len(v)) < f
    for i in range(len(v)):
        if pick[i] and key[i] >= 0 and sums[key[i]] != 0:
            v[i] = abs(v[i]) * np.sign(sums[key[i]])
    df["doppler"] = v
    return df


def C_of(insts):
    vals = []
    for inst, _, _ in insts:
        ranges = fit_ranges([inst])
        ab, ti = axis_bins(inst, ranges)
        num = den = 0.0
        for ax in CAXES:
            bi, m = ab[ax]
            v = inst["doppler"].values.astype(float)[m]
            b, t = bi[m], ti[m]
            s = {}; a = {}
            for k, vv in zip(b * 1000 + t, v):
                s[k] = s.get(k, 0.0) + vv; a[k] = a.get(k, 0.0) + abs(vv)
            num += sum(abs(x) for x in s.values()); den += sum(a.values())
        if den > 0:
            vals.append(num / den)
    return float(np.mean(vals))


def shared_signed_channels(inst, ranges):
    """pos-sum and |neg|-sum per axis, both divided by ONE shared per-axis scale."""
    ab, ti = axis_bins(inst, ranges)
    v = inst["doppler"].values.astype(float)
    ch = []
    for ax in CAXES:
        bi, m = ab[ax]
        b, t, vv = bi[m], ti[m], v[m]
        P = np.zeros((NB, T), np.float32); Nn = np.zeros((NB, T), np.float32)
        sel = vv > 0; np.add.at(P, (b[sel], t[sel]), vv[sel])
        sel = vv < 0; np.add.at(Nn, (b[sel], t[sel]), -vv[sel])
        s = max(np.abs(P).max(), np.abs(Nn).max(), 1e-12)
        ch += [(P / s).astype(np.float32), (Nn / s).astype(np.float32)]
    return np.stack(ch)


def scalar_channels(inst, ranges):
    st = cell_stats(inst, CAXES, ranges, nb=NB)
    return compose(st, CAXES, ["sum"]).astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    print(f"SHARED-NORM SWEEP SMOKE={SMOKE} flips={FLIPS}", flush=True)
    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    if SMOKE:
        idx = np.random.RandomState(0).permutation(len(mf))[:200]; mf = [mf[i] for i in idx]
    subj = np.array([str(t[2]) for t in mf])
    folds = ([[str(x) for x in g] for g in kfold(subj, 5)]
             if not SMOKE else [sorted(set(subj))[:2]])
    y = np.array([t[1] for t in mf])
    results, preds, cvals = {}, {"SUBJ__MM-Fi": subj}, {}
    for f in FLIPS:
        insts_f = [(coherence_enforce(t[0], f, i), t[1], t[2]) for i, t in enumerate(mf)]
        cvals[str(f)] = C_of(insts_f)
        print(f"flip={f}: C={cvals[str(f)]:.3f}", flush=True)
        accs_all = {}
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
            if te.sum() == 0 or tr.sum() == 0:
                continue
            ranges = fit_ranges([t[0] for t, keep in zip(insts_f, tr) if keep])
            X = {"v_sum": np.stack([scalar_channels(t[0], ranges) for t in insts_f]),
                 "v_signsplit_shared": np.stack([shared_signed_channels(t[0], ranges)
                                                 for t in insts_f])}
            for name, Xa in X.items():
                for sd in SEEDS:
                    a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], 27,
                                                    epochs=BUDGET, seed=sd, width=WIDTH)
                    accs_all.setdefault(name, ([], []))
                    accs_all[name][0].append(a); accs_all[name][1].append(ta)
                    preds[f"MM-Fi|{name}|f{f}|ep{BUDGET}|fold{fi}|seed{sd}"] = \
                        np.stack([te_idx, yt, yp])
            del X
        for name, (accs, taccs) in accs_all.items():
            k = f"MM-Fi|{name}|flip{f}|ep{BUDGET}"
            results[k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                          "std": round(float(np.std(accs)) * 100, 2),
                          "min_train_acc": round(float(np.min(taccs)), 4),
                          "underfit": bool(np.min(taccs) < 0.95)}
            r = results[k]
            print(f"  {k:44s} {r['acc']:6.2f} (+-{r['std']:.1f}) "
                  f"min_tr={r['min_train_acc']:.3f}", flush=True)
    out = {"purpose": "E9 shared-normalization sign split + E10 within-dataset C sweep",
           "protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budget": BUDGET, "flips": list(FLIPS), "smoke": SMOKE,
                        "ranges": "fold-fitted"},
           "C_per_flip": cvals, "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"shared_norm_sweep{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"shared_norm_sweep{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/shared_norm_sweep{sfx}.json in {time.time()-t0:.0f}s", flush=True)
