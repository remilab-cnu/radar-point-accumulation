"""Audit-compliance reruns R1-R5 (Table III re-audit, 2026-07-12).

Implements the lead-reviewer rerun list with ZERO baseline reruns (baselines are
already protocol-compliant; all debt is on our side):
  R1  MM-Fi + mRI map arms rerun WITH per-instance preds (rep_pc2 discarded them)
  R2  mHomeGes map arms under the frozen protocol (no aug, batch 64, seeds 0-2,
      BOTH ep30/40; round-4's aug3/batch128/2-seed cells are supplement-only)
  R3  velocity-family member per dataset selected on the TRAIN-subject validation
      split (protocol Splits), never on test
  R4  DeepSets missing cells (mH ep40, MM-Fi ep30) + train-acc instrumentation
  R5  train acc re-logged for our Infineon arms (symmetric underfit gate)

PRE-REGISTERED SELECTION RULE (R3): for each dataset the quotable family member is
the arm with the highest MEAN ep40 VALIDATION accuracy over folds x seeds {0,1,2}.
Validation subjects = last fold of a seed-1 permutation of the sorted TRAIN subjects
of each test fold (VAL_K below). v_sum is ALWAYS also reported (pre-declared
standard arm). Test folds are touched once per (arm, budget, seed).

Frozen protocol (docs/EQUAL_HPO_PROTOCOL.md): Adam lr 1e-3, batch 64, width 16,
seeds {0,1,2}, epochs BOTH 30 and 40, no augmentation, frozen manifests + folds,
per-instance preds + subject IDs + final train acc saved for EVERY run.

Parts (env RERUN_PART):
  small : MM-Fi + mRI + Infineon map arms (R1, R3, R5)
  mh    : mHomeGes map arms (R2, R3)
  ds    : DeepSets on mH + MM-Fi + Infineon (R4, symmetric UF)
Env SMOKE=1: tiny plumbing check (subset, 2 epochs, seed 0 only) - not results.
"""
import os, json, glob, pickle, hashlib
import numpy as np
import pandas as pd
from rep_variants import cell_stats, compose, norm, CAXES, infineon_recs, kfold
from spectra_dataset import fit_ranges, mhomeges_instances, mmfi_instances
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
PART = os.environ.get("RERUN_PART", "small")
SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPS = (2,) if SMOKE else (30, 40)
VAL_EP = 2 if SMOKE else 40                     # member selection budget (declared)
FAMILY = {                                       # velocity-family candidates (R3)
    "v_sum":        ["sum"],
    "v_signed":     ["pos_mean", "neg_mean"],
    "v_hist4":      ["hist"],
    "v_meanstd+vt": ["mean", "std", "vt"],
}
CONTROL = {"occupancy": ["cnt"]}                 # geometry control, test-only
VAL_K = {"mHomeGes": 5, "Infineon": 4, "MM-Fi": 5, "mRI": 5}

FROZEN = json.load(open(os.path.join(DOCS, "baselines2.json")))["datasets"]


def frozen_folds(tag, subj, k):
    """Frozen kfold; verified against baselines2.json where stored (housekeeping #6)."""
    folds = [list(map(str, f)) for f in kfold(subj, k)]
    if tag in FROZEN and not SMOKE:
        ref = [sorted(f) for f in FROZEN[tag]["folds"]]
        assert [sorted(f) for f in folds] == ref, f"{tag}: fold mismatch vs baselines2.json"
        print(f"  [{tag}] folds verified == baselines2.json", flush=True)
    return folds


def val_split(train_subjects, k):
    """Protocol Splits: last fold of a seed-1 permutation of sorted train subjects."""
    rng = np.random.RandomState(1)
    return [str(s) for s in np.array_split(rng.permutation(sorted(train_subjects)), k)[-1]]


def run_maps(tag, insts, folds, ncls, out, preds):
    y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
    ranges = fit_ranges([t[0] for t in insts])
    stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in insts]
    arm_specs = dict(FAMILY); arm_specs.update(CONTROL)
    X = {n: np.stack([compose(st, CAXES, spec) for st in stats]).astype(np.float32)
         for n, spec in arm_specs.items()}
    del stats
    preds[f"SUBJ__{tag}"] = subj
    res = {"n_instances": len(insts), "folds": folds, "val_k": VAL_K[tag],
           "val_budget_ep": VAL_EP, "results": {}, "val": {}}
    # ---- test runs (both budgets, every arm) ----
    for name, Xa in X.items():
        for ep in EPS:
            accs, taccs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(subj, te_s); tr = ~te; te_idx = np.where(te)[0]
                for s in SEEDS:
                    a, yt, yp, ta = train_eval_full(Xa[tr], y[tr], Xa[te], y[te], ncls,
                                                    epochs=ep, seed=s)
                    accs.append(a); taccs.append(ta)
                    preds[f"{tag}|{name}|ep{ep}|fold{fi}|seed{s}"] = np.stack([te_idx, yt, yp])
            k = f"{name}|ep{ep}"
            res["results"][k] = {"acc": float(np.mean(accs)) * 100, "std": float(np.std(accs)) * 100,
                                 "accs": [round(float(a) * 100, 2) for a in accs],
                                 "min_train_acc": round(float(np.min(taccs)), 4),
                                 "underfit": bool(np.min(taccs) < 0.95)}
            uf = " UF" if res["results"][k]["underfit"] else ""
            print(f"  {tag} {name:14s} ep{ep}: {res['results'][k]['acc']:6.2f}% "
                  f"(+-{res['results'][k]['std']:.1f}) min_tr={np.min(taccs):.3f}{uf}", flush=True)
    # ---- validation runs (family only, VAL_EP) ----
    for name in FAMILY:
        Xa = X[name]; vaccs = []
        for te_s in folds:
            tr_subj = [s_ for s_ in sorted(set(subj.tolist())) if s_ not in set(te_s)]
            vs = val_split(tr_subj, VAL_K[tag])
            va = np.isin(subj, vs); trn = (~np.isin(subj, te_s)) & (~va)
            for s in SEEDS:
                a, _, _, _ = train_eval_full(Xa[trn], y[trn], Xa[va], y[va], ncls,
                                             epochs=VAL_EP, seed=s)
                vaccs.append(a)
        res["val"][name] = {"acc": float(np.mean(vaccs)) * 100, "std": float(np.std(vaccs)) * 100,
                            "accs": [round(float(a) * 100, 2) for a in vaccs]}
        print(f"  {tag} VAL {name:14s} ep{VAL_EP}: {res['val'][name]['acc']:6.2f}% "
              f"(+-{res['val'][name]['std']:.1f})", flush=True)
    sel = max(FAMILY, key=lambda n: res["val"][n]["acc"])
    res["selected_family_member"] = sel
    print(f"  {tag} SELECTED (val ep{VAL_EP}): {sel}", flush=True)
    out[tag] = res


def run_deepsets(tag, insts, folds, ncls, out, preds):
    from pointset_models import DeepSets, build_point_tensors, train_eval_set_full
    y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
    ranges = fit_ranges([t[0] for t in insts])
    Xp, Mp, yp_, sp = build_point_tensors(insts, ranges)
    sp = np.array([str(s) for s in sp])
    assert np.array_equal(yp_, y) and np.array_equal(sp, subj)
    preds[f"SUBJ__{tag}"] = subj
    res = {"n_instances": len(insts), "folds": folds, "results": {}}
    for ep in EPS:
        accs, taccs = [], []
        for fi, te_s in enumerate(folds):
            te = np.isin(subj, te_s); tr = ~te; te_idx = np.where(te)[0]
            for s in SEEDS:
                a, yt, yp, ta = train_eval_set_full(DeepSets, Xp[tr], Mp[tr], y[tr],
                                                    Xp[te], Mp[te], y[te], ncls, 6,
                                                    epochs=ep, seed=s)
                accs.append(a); taccs.append(ta)
                preds[f"{tag}|DeepSets_full|ep{ep}|fold{fi}|seed{s}"] = np.stack([te_idx, yt, yp])
        k = f"DeepSets_full|ep{ep}"
        res["results"][k] = {"acc": float(np.mean(accs)) * 100, "std": float(np.std(accs)) * 100,
                             "accs": [round(float(a) * 100, 2) for a in accs],
                             "min_train_acc": round(float(np.min(taccs)), 4),
                             "underfit": bool(np.min(taccs) < 0.95)}
        uf = " UF" if res["results"][k]["underfit"] else ""
        print(f"  {tag} DeepSets_full ep{ep}: {res['results'][k]['acc']:6.2f}% "
              f"(+-{res['results'][k]['std']:.1f}) min_tr={np.min(taccs):.3f}{uf}", flush=True)
    out[tag] = res


def mri_records():
    """mRI window-40/stride-20 build, identical to rep_pc2/rep_mri."""
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
    print(f"PART={PART} SMOKE={SMOKE} seeds={SEEDS} eps={EPS}", flush=True)
    out, preds = {"protocol": {"lr": 1e-3, "batch": 64, "width": 16, "seeds": list(SEEDS),
                               "epochs": list(EPS), "aug": "none", "val_budget_ep": VAL_EP,
                               "uf_gate": 0.95, "part": PART, "smoke": SMOKE}}, {}
    if PART == "small":
        mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
        if SMOKE: mf = subset(mf)
        run_maps("MM-Fi", mf, [FROZEN["MM-Fi"]["folds"][0]] if not SMOKE
                 else [sorted(set(str(t[2]) for t in mf))[:2]], 27, out, preds)
        del mf
        mr = mri_records()
        if SMOKE: mr = subset(mr)
        run_maps("mRI", mr, frozen_folds("mRI", np.array([str(t[2]) for t in mr]), 5), 10, out, preds)
        del mr
        recs = infineon_recs()
        out["infineon_manifest_md5"] = hashlib.md5(
            open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest()
        if SMOKE: recs = subset(recs)
        run_maps("Infineon", recs,
                 frozen_folds("Infineon", np.array([str(t[2]) for t in recs]), 4), 5, out, preds)
    elif PART == "mh":
        mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
        if SMOKE: mh = subset(mh, 400)
        run_maps("mHomeGes", mh,
                 frozen_folds("mHomeGes", np.array([str(t[2]) for t in mh]), 5), 10, out, preds)
    elif PART == "ds":
        mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
        if SMOKE: mf = subset(mf)
        run_deepsets("MM-Fi", mf, [FROZEN["MM-Fi"]["folds"][0]] if not SMOKE
                     else [sorted(set(str(t[2]) for t in mf))[:2]], 27, out, preds)
        del mf
        recs = infineon_recs()
        if SMOKE: recs = subset(recs)
        run_deepsets("Infineon", recs,
                     frozen_folds("Infineon", np.array([str(t[2]) for t in recs]), 4), 5, out, preds)
        del recs
        mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
        if SMOKE: mh = subset(mh, 400)
        run_deepsets("mHomeGes", mh,
                     frozen_folds("mHomeGes", np.array([str(t[2]) for t in mh]), 5), 10, out, preds)
    else:
        raise SystemExit(f"unknown RERUN_PART={PART}")

    suff = f"_{PART}" + ("_smoke" if SMOKE else "")
    json.dump(out, open(os.path.join(DOCS, f"rerun_audit{suff}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, f"rerun_audit{suff}_preds.npz"), **preds)
    print(f"\nwrote docs/rerun_audit{suff}.json + rerun_audit{suff}_preds.npz", flush=True)
