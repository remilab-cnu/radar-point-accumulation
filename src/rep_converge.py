"""Cold-review gating experiments (2026-07-12) — the two blockers all reviewers named.

Blocker #1 (UNDERFIT CONFOUND): frozen ep30/40 + width16 left most map arms UF, and the
  Infineon hist-vs-sum ordering flips between budgets. => train map arms to CONVERGENCE
  (width 32; budgets 60 AND 120; per-arm train acc logged) and test whether the
  quantity ordering (velocity vs amplitude vs occupancy) and the scalar-vs-histogram
  ordering are STABLE once the models fit their training sets.
Blocker #2 (C3 on ONE MM-Fi fold, pseudoreplicated): run MM-Fi as multi-fold (5-fold,
  subject-disjoint over all subjects) and mRI 5-fold, save per-instance preds + subject
  IDs => subject-cluster bootstrap for the within-representation contrasts (the paper's
  own significance standard, not per-instance McNemar on one fold). Add a COUNT-MATCHED
  control `v_hist4_vshuffled`: per-point doppler permuted within each instance before
  binning => identical spatial occupancy and identical global velocity marginal, but the
  point->velocity association is destroyed. If hist4 > vshuffled, the recovery is the
  velocity DISTRIBUTION, not the embedded occupancy (addresses the "hist4 embeds
  occupancy" confound, non-claim #6).

Arms per regime:
  gesture (high-C: mHomeGes, Infineon)      : v_sum, v_hist4, occupancy, amplitude
  whole-body (low-C: MM-Fi, mRI)            : v_sum, v_hist4, v_hist4_vshuffled, occupancy, amplitude
Model: SmallCNN width=32 (capacity to converge; the width-16 24k efficiency numbers stay
  in rerun_audit_*). Adam lr 1e-3, batch 64, seeds {0,1,2}. Budgets {60,120}, both printed.
Every run saves test preds + subject IDs (at the max budget) + min train acc (UF gate).

Env RERUN_PART: mh | gest | body   ; SMOKE=1 tiny plumbing check.
"""
import os, json, glob, pickle
import numpy as np
import pandas as pd
from rep_variants import cell_stats, compose, norm, CAXES, infineon_recs, kfold
from spectra_dataset import fit_ranges, mhomeges_instances, mmfi_instances
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
PART = os.environ.get("RERUN_PART", "body")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
BUDGETS = (4,) if SMOKE else (60, 120)
WIDTH = 8 if SMOKE else 32
FROZEN = json.load(open(os.path.join(DOCS, "baselines2.json")))["datasets"]

GEST_ARMS = {"v_sum": ["sum"], "v_hist4": ["hist"], "occupancy": ["cnt"], "amplitude": ["int_mean"]}
BODY_ARMS = {"v_sum": ["sum"], "v_hist4": ["hist"], "v_hist4_vshuffled": ["hist"],
             "occupancy": ["cnt"], "amplitude": ["int_mean"]}


def shuffle_doppler(inst, seed=0):
    df = inst.copy()
    rng = np.random.RandomState(seed)
    df["doppler"] = rng.permutation(df["doppler"].values.astype(float))
    return df


def build_arms(insts, arm_specs):
    ranges = fit_ranges([t[0] for t in insts])
    stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in insts]
    X = {}
    for name, spec in arm_specs.items():
        if name == "v_hist4_vshuffled":
            sh = [cell_stats(shuffle_doppler(t[0], seed=i), CAXES, ranges, nb=32)
                  for i, t in enumerate(insts)]
            X[name] = np.stack([compose(st, CAXES, ["hist"]) for st in sh]).astype(np.float32)
        else:
            X[name] = np.stack([compose(st, CAXES, spec) for st in stats]).astype(np.float32)
    del stats
    return X


def run(tag, insts, folds, ncls, arm_specs, out, preds):
    y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
    X = build_arms(insts, arm_specs)
    preds[f"SUBJ__{tag}"] = subj
    res = {"n_instances": len(insts), "n_cls": ncls, "folds": folds, "width": WIDTH,
           "budgets": list(BUDGETS), "results": {}}
    maxb = max(BUDGETS)
    for name, Xa in X.items():
        for ep in BUDGETS:
            accs, taccs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(subj, te_s); tr = ~te; te_idx = np.where(te)[0]
                for s in SEEDS:
                    a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], ncls,
                                                    epochs=ep, seed=s, width=WIDTH)
                    accs.append(a); taccs.append(ta)
                    if ep == maxb:
                        preds[f"{tag}|{name}|ep{ep}|fold{fi}|seed{s}"] = np.stack([te_idx, yt, yp])
            k = f"{name}|ep{ep}"
            res["results"][k] = {"acc": round(float(np.mean(accs)) * 100, 2),
                                 "std": round(float(np.std(accs)) * 100, 2),
                                 "min_train_acc": round(float(np.min(taccs)), 4),
                                 "mean_train_acc": round(float(np.mean(taccs)), 4),
                                 "underfit": bool(np.min(taccs) < 0.95)}
            r = res["results"][k]
            uf = " UF" if r["underfit"] else ""
            print(f"  {tag} {name:18s} ep{ep:<3d}: {r['acc']:6.2f} (+-{r['std']:.1f}) "
                  f"train={r['mean_train_acc']:.3f}{uf}", flush=True)
    out[tag] = res


def mri_records():
    CL = [f"pose_{i}" for i in range(1, 11)]; recs = []
    MRI = os.path.join(DATA, "mri_sample", "mri_data")
    for csvf in sorted(glob.glob(os.path.join(MRI, "subject*.csv"))):
        sid = os.path.basename(csvf).replace(".csv", "")
        if "_all_labels" in sid:
            continue
        df = pd.read_csv(csvf); df.columns = [x.strip() for x in df.columns]
        can = pd.DataFrame({"frame": df["Camera Frame"].astype(int), "x": df["X"], "y": df["Y"],
                            "z": df["Z"], "doppler": df["Doppler"], "intensity": df["Intensity"]})
        vl = pickle.load(open(os.path.join(MRI, f"{sid}_all_labels.cpl"), "rb"))["video_label"]
        for ci, cn in enumerate(CL):
            if cn not in vl:
                continue
            a, b = vl[cn]; t0 = a
            while t0 + 40 <= b:
                w = can[(can.frame >= t0) & (can.frame < t0 + 40)]
                if w["frame"].nunique() >= 6 and len(w) >= 30:
                    recs.append((w.reset_index(drop=True), ci, sid))
                t0 += 20
    return recs


def subset(insts, n=200):
    idx = np.random.RandomState(0).permutation(len(insts))[:n]
    return [insts[i] for i in idx]


if __name__ == "__main__":
    print(f"PART={PART} SMOKE={SMOKE} width={WIDTH} budgets={BUDGETS} seeds={SEEDS}", flush=True)
    out = {"protocol": {"lr": 1e-3, "batch": 64, "width": WIDTH, "seeds": list(SEEDS),
                        "budgets": list(BUDGETS), "aug": "none", "part": PART, "smoke": SMOKE,
                        "note": "convergence + MM-Fi multi-fold + count-matched vshuffled control"}}
    preds = {}
    if PART == "mh":
        mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
        if SMOKE: mh = subset(mh, 400)
        folds = FROZEN["mHomeGes"]["folds"] if not SMOKE else [sorted(set(str(t[2]) for t in mh))[:2]]
        run("mHomeGes", mh, folds, 10, GEST_ARMS, out, preds)
    elif PART == "gest":
        recs = infineon_recs()
        if SMOKE: recs = subset(recs)
        folds = FROZEN["Infineon"]["folds"] if not SMOKE else [sorted(set(str(t[2]) for t in recs))[:2]]
        run("Infineon", recs, folds, 5, GEST_ARMS, out, preds)
    elif PART == "body":
        mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
        if SMOKE: mf = subset(mf)
        # NEW: MM-Fi multi-fold (5-fold subject-disjoint over ALL subjects) vs the old single S2 fold
        mf_folds = ([[str(s) for s in g] for g in kfold(np.array([str(t[2]) for t in mf]), 5)]
                    if not SMOKE else [sorted(set(str(t[2]) for t in mf))[:2]])
        run("MM-Fi", mf, mf_folds, 27, BODY_ARMS, out, preds); del mf
        mr = mri_records()
        if SMOKE: mr = subset(mr)
        mr_folds = ([[str(s) for s in g] for g in kfold(np.array([str(t[2]) for t in mr]), 5)]
                    if not SMOKE else [sorted(set(str(t[2]) for t in mr))[:2]])
        run("mRI", mr, mr_folds, 10, BODY_ARMS, out, preds)
    else:
        raise SystemExit(f"unknown RERUN_PART={PART}")

    suff = f"_{PART}" + ("_smoke" if SMOKE else "")
    json.dump(out, open(os.path.join(DOCS, f"converge{suff}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"converge{suff}_preds.npz"), **preds)
    print(f"\nwrote docs/converge{suff}.json + converge{suff}_preds.npz", flush=True)
