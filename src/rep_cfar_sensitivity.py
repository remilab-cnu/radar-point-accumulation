"""CFAR OPERATING-POINT SENSITIVITY (referee request, 2026-07-18).

Question (automotive-radar referee): does the accumulation-arm ordering on the 60 GHz
set survive a different detector operating point / CFAR variant? A lower threshold
admits more points (geometry-rich, could favor occupancy); a higher threshold gates
detection toward high-SNR movers (couples occupancy to velocity); OS-CFAR changes the
noise-estimate rule itself.

Design: rebuild the point clouds at three NEW detector configs with the SAME
deterministic instance selection as the frozen baseline (RandomState(0) permutation,
cap 40/class/user, gesture window +-6 frames, >=8-point filter), then run the frozen
protocol per config:
  maps  : map_v_sum / REF_int_mean / REF_occupancy  (cell_stats nb=32, width-16 CNN)
  points: DeepSets full(xyzvAt) vs intensity(xyzAt) = no-velocity (velocity contribution)
  ep30 AND ep40, batch 64, seeds {0,1,2}, kfold(subj,4) seed=0, per-instance preds
  saved, train accuracy recorded (UF gate 0.95).

BASELINE ca/alpha=2.5 is NOT recomputed: the frozen pkl's results live in
final_infineon.json (maps + DeepSets_full) and p1_crossparadigm.json (DeepSets
intensity(xyzAt)). Variant clouds cache to data/infineon_recs_cfar_<name>.pkl; the
frozen infineon_recs.pkl is NEVER touched. One zip pass builds all variants (each
per-user npz is loaded once).

Out: docs/cfar_sensitivity.json + docs/cfar_sensitivity_preds.npz.
SMOKE=1: first 2 users, cap 4/class, ep2, seed 0, 2 folds.
"""
import os, io, re, json, time, pickle, zipfile
import numpy as np
from spectra_dataset import fit_ranges
from rep_variants import cell_stats, norm, CAXES
from rep_round3 import kfold
from cnn import train_eval_full
from pointset_models import DeepSets, train_eval_set_full, build_point_tensors
import infineon_detection as ifx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPOCH_BUDGETS = (2,) if SMOKE else (30, 40)
CAP = 4 if SMOKE else 40           # per class per user (baseline: 40 -> 200/user)
LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
UF_GATE = 0.95

# detector variants (baseline ca/2.5 comes from the frozen pkl, not rebuilt here)
VARIANTS = {
    "ca_a2.0": dict(method="ca", alpha=2.0),   # looser threshold -> more points
    "ca_a3.5": dict(method="ca", alpha=3.5),   # tighter -> high-SNR movers only
    "os_a2.5": dict(method="os", alpha=2.5),   # ordered-statistic CFAR variant
}
SETS_DS = {"full(xyzvAt)": [0, 1, 2, 3, 4, 5], "intensity(xyzAt)": [0, 1, 2, 4, 5]}


def build_all_variants():
    """One pass over the zip; per user npz loaded once, all variants detected from the
    same gesture windows with the same deterministic selection as the frozen build."""
    caches = {v: os.path.join(DATA, f"infineon_recs_cfar_{v}{'_smoke' if SMOKE else ''}.pkl")
              for v in VARIANTS}
    if all(os.path.exists(c) for c in caches.values()):
        return {v: pickle.load(open(c, "rb")) for v, c in caches.items()}
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    if SMOKE:
        members = members[:2]
    per = {v: [] for v in VARIANTS}
    for m in members:
        user = "u" + re.search(r"user(\d+)", m).group(1)
        t0 = time.time()
        with zf.open(m) as f:
            d = np.load(io.BytesIO(f.read())); inputs, targets = d["inputs"], d["targets"]
        by = {v: {} for v in VARIANTS}
        for r in np.random.RandomState(0).permutation(len(inputs)):
            g = np.where(targets[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
            if cls not in LM:
                continue
            win = inputs[r, max(0, g.min() - 6):g.max() + 7]
            for v, cfg in VARIANTS.items():
                if by[v].get(cls, 0) >= CAP:
                    continue
                df = ifx.process_recording(win, **cfg)
                if len(df) < 8:
                    continue
                per[v].append((df, LM[cls], user)); by[v][cls] = by[v].get(cls, 0) + 1
            if all(len(b) == len(LM) and min(b.values()) >= CAP for b in by.values()):
                break
        print(f"  {user}: " + " ".join(f"{v}={sum(by[v].values())}" for v in VARIANTS)
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del inputs, targets
    for v, c in caches.items():
        pickle.dump(per[v], open(c, "wb"))
    return per


def build_stats(recs):
    npts = np.array([len(t[0]) for t in recs])
    nfr = np.array([int(t[0].frame.max()) + 1 for t in recs])
    cls, cnt = np.unique([t[1] for t in recs], return_counts=True)
    return {"n_instances": len(recs),
            "points_per_instance_mean": round(float(npts.mean()), 1),
            "points_per_frame_mean": round(float((npts / nfr).mean()), 2),
            "class_counts": {int(c): int(n) for c, n in zip(cls, cnt)}}


def run_variant(vname, recs, results, preds):
    ranges = fit_ranges([t[0] for t in recs])
    y = np.array([t[1] for t in recs]); subj = np.array([t[2] for t in recs])
    folds = kfold(subj, 2 if SMOKE else 4)
    stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in recs]
    ARMS_MAP = {
        "map_v_sum":     np.stack([np.stack([norm(st[ax]["sum"]) for ax in CAXES]) for st in stats]).astype(np.float32),
        "REF_int_mean":  np.stack([np.stack([norm(st[ax]["int_mean"]) for ax in CAXES]) for st in stats]).astype(np.float32),
        "REF_occupancy": np.stack([np.stack([norm(st[ax]["cnt"]) for ax in CAXES]) for st in stats]).astype(np.float32),
    }
    del stats
    Xp, Mp, yp, sp = build_point_tensors(recs, ranges)

    def summarize(key, accs, tr_accs):
        results[key] = {"acc": round(float(np.mean(accs)) * 100, 2),
                        "std": round(float(np.std(accs)) * 100, 2),
                        "accs": [round(float(a) * 100, 2) for a in accs],
                        "min_train_acc": round(float(np.min(tr_accs)), 4),
                        "underfit": bool(np.min(tr_accs) < UF_GATE)}
        r = results[key]
        print(f"  {key:38s}: {r['acc']:6.2f}% (+-{r['std']:.1f})  min_train={r['min_train_acc']:.3f}"
              f"{'  UF' if r['underfit'] else ''}", flush=True)

    for ep in EPOCH_BUDGETS:
        for arm, X in ARMS_MAP.items():
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(subj, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                for sd in SEEDS:
                    a, yt, ypred, ta = train_eval_full(X[tr], y[tr], X[te], y[te], 5,
                                                       epochs=ep, seed=sd, width=16)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{vname}|{arm}|ep{ep}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, ypred])
            summarize(f"{vname}|{arm}|ep{ep}", accs, tr_accs)
        for fname, cols in SETS_DS.items():
            Xc = np.ascontiguousarray(Xp[..., cols])
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(sp, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                for sd in SEEDS:
                    a, yt, ypred, ta = train_eval_set_full(DeepSets, Xc[tr], Mp[tr], yp[tr],
                                                           Xc[te], Mp[te], yp[te], 5, len(cols),
                                                           epochs=ep, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{vname}|DeepSets_{fname}|ep{ep}|fold{fi}|seed{sd}"] = np.stack([te_idx, yt, ypred])
            summarize(f"{vname}|DeepSets_{fname}|ep{ep}", accs, tr_accs)
    del ARMS_MAP, Xp, Mp


if __name__ == "__main__":
    t0 = time.time()
    print(f"CFAR SENSITIVITY  SMOKE={SMOKE}  variants={list(VARIANTS)}", flush=True)
    per = build_all_variants()
    results, preds, bstats = {}, {}, {}
    for v in VARIANTS:
        bstats[v] = build_stats(per[v])
        print(f"\n#### {v}: {bstats[v]} ####", flush=True)
        run_variant(v, per[v], results, preds)
    out = {"purpose": "CFAR operating-point sensitivity (referee R1, 2026-07-18)",
           "baseline_note": "ca/alpha=2.5 = frozen infineon_recs.pkl; its results are in "
                            "final_infineon.json (maps, DeepSets_full) and "
                            "p1_crossparadigm.json (DeepSets intensity(xyzAt)); not recomputed",
           "detector_common": {"train": 9, "guard": 3, "window": "+-6 frames",
                               "min_points": 8, "cap_per_class_per_user": CAP},
           "variants": {v: cfg for v, cfg in VARIANTS.items()},
           "build_stats": bstats,
           "protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "width": 16,
                        "seeds": list(SEEDS), "aug": "none", "uf_gate": UF_GATE,
                        "folds": "kfold(subj,4) seed=0", "smoke": SMOKE},
           "results": results}
    sfx = "_smoke" if SMOKE else ""
    json.dump(out, open(os.path.join(DOCS, f"cfar_sensitivity{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"cfar_sensitivity{sfx}_preds.npz"), **preds)
    print(f"\nwrote docs/cfar_sensitivity{sfx}.json (+preds) in {time.time()-t0:.0f}s", flush=True)
