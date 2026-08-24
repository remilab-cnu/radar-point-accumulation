"""paired_stats.py — source-of-truth for every paired statistic in the paper.

Reads the saved per-instance predictions (experiments/docs/*_preds.npz) and the
aggregate result JSONs, recomputes the paired contrasts (McNemar, subject/fold-cluster
bootstrap CIs, TOST equivalence), and writes figures/paired_stats.json. The figure
scripts and the manuscript cite THIS file's output — no number is hand-copied.

Preds npz layout (verified): key "arm|ep|foldK|seedS" -> int64 array (3, N):
  row0 = instance id (stable across methods within a fold -> enables pairing)
  row1 = y_true
  row2 = y_pred
Protocol: frozen equal-HPO, lr 1e-3, batch 64, seeds {0,1,2}, subject-disjoint folds.
The frozen-protocol Infineon run (final_infineon_preds.npz) has no lr in its keys
(lr is the standard 1e-3); the published baselines carry an explicit lr tag, and we
select lr1e-3 to match the frozen protocol (lr3e-4 is a supplementary sensitivity arm).

Run:  cd figures && python3 paired_stats.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, "..", "experiments", "docs"))
BOOT = 20000
SEED = 0  # fixed so the CI is reproducible


def load_preds(fname):
    return np.load(os.path.join(DOCS, fname), allow_pickle=True)


def per_instance_correct(z, key_prefix, folds=4, seeds=3):
    """Return dict inst_id -> mean-over-seeds correctness (float in [0,1]), pooled over folds.
    key_prefix is everything before |foldK|seedS (or |fK|sS for the baseline npz)."""
    # auto-detect fold/seed token spelling
    sample = [k for k in z.files if k.startswith(key_prefix + "|")]
    if not sample:
        raise KeyError(f"no keys for prefix {key_prefix!r}")
    ft, st = ("fold", "seed") if any("|fold" in k for k in sample) else ("f", "s")
    acc = {}
    for f in range(folds):
        for s in range(seeds):
            k = f"{key_prefix}|{ft}{f}|{st}{s}"
            if k not in z.files:
                continue
            a = z[k]
            ids, ytrue, ypred = a[0], a[1], a[2]
            for i, yt, yp in zip(ids, ytrue, ypred):
                acc.setdefault(int(i), []).append(1.0 if yt == yp else 0.0)
    return {i: float(np.mean(v)) for i, v in acc.items()}


def mcnemar(correct_a, correct_b):
    """Continuity-corrected McNemar on majority-vote (>0.5) per-instance correctness."""
    ids = sorted(set(correct_a) & set(correct_b))
    a = np.array([correct_a[i] > 0.5 for i in ids])
    b = np.array([correct_b[i] > 0.5 for i in ids])
    n01 = int(np.sum(~a & b))   # b right, a wrong
    n10 = int(np.sum(a & ~b))   # a right, b wrong
    from scipy.stats import chi2
    if n01 + n10 == 0:
        return dict(n=len(ids), n10=n10, n01=n01, stat=0.0, p=1.0)
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return dict(n=len(ids), n10=n10, n01=n01, stat=float(stat), p=float(chi2.sf(stat, 1)))


def _paired_diffs(correct_a, correct_b, clusters=None):
    """Per-unit paired differences (a-b). Unit = instance, or cluster-mean if a
    clusters mapping (inst_id -> cluster key, e.g. user) is given. Instance-level CIs on
    Infineon are anti-conservative (12 users; frames within a recording are correlated),
    so CLUSTERED is the primary analysis wherever a mapping exists."""
    ids = sorted(set(correct_a) & set(correct_b))
    if clusters is None:
        return np.array([correct_a[i] - correct_b[i] for i in ids]), len(ids), None
    by = {}
    for i in ids:
        by.setdefault(clusters[i], []).append(correct_a[i] - correct_b[i])
    d = np.array([np.mean(v) for k, v in sorted(by.items())])
    return d, len(ids), len(d)


def paired_diff_ci(correct_a, correct_b, boot=BOOT, seed=SEED, clusters=None):
    """Mean paired accuracy difference (a - b) in pp with a 95% bootstrap CI.
    clusters=None -> instance bootstrap; clusters given -> cluster (user) bootstrap."""
    d, n_inst, n_cl = _paired_diffs(correct_a, correct_b, clusters)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(boot)]) * 100.0
    return dict(n=n_inst, n_clusters=n_cl, unit=("user" if clusters else "instance"),
                mean_pp=float(d.mean() * 100.0),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def tost(correct_a, correct_b, margin=3.0, boot=BOOT, seed=SEED, clusters=None):
    """Two one-sided tests for equivalence of paired accuracies within +/- margin pp.
    Equivalent if the 90% CI (alpha=0.05 per side) lies inside (-margin, +margin).
    Pass clusters (inst_id -> user) to make the user the resampling unit (primary)."""
    d, n_inst, n_cl = _paired_diffs(correct_a, correct_b, clusters)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(boot)]) * 100.0
    ci90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    ci95 = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    equivalent = (ci90[0] > -margin) and (ci90[1] < margin)
    return dict(n=n_inst, n_clusters=n_cl, unit=("user" if clusters else "instance"),
                mean_pp=float(d.mean() * 100.0), margin_pp=margin,
                ci90=ci90, ci95=ci95, equivalent=bool(equivalent))


def load_infineon_clusters():
    """inst_id -> user, from the frozen pkl (recs[i] = (df, class, user)); ids are rec indices."""
    import pickle
    pkl = os.path.abspath(os.path.join(DOCS, "..", "data", "infineon_recs.pkl"))
    recs = pickle.load(open(pkl, "rb"))
    return {i: str(t[2]) for i, t in enumerate(recs)}


def cluster_ci_from_folds(arr15, n_folds=5, n_seeds=3, boot=BOOT, seed=SEED):
    """Fold-cluster bootstrap CI of a per-config mean, from a flat [folds*seeds] accuracy array.
    Used for the aggregate arm contrasts stored as [mean,std,[vals]] in the result JSONs."""
    a = np.array(arr15).reshape(n_folds, n_seeds).mean(axis=1)
    rng = np.random.default_rng(seed)
    bs = np.array([a[rng.integers(0, n_folds, n_folds)].mean() for _ in range(boot)])
    return float(a.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    out = {"_provenance": "recomputed by figures/paired_stats.py from experiments/docs/*_preds.npz",
           "_protocol": "frozen equal-HPO, lr1e-3, seeds{0,1,2}, subject-disjoint folds, ep40 unless noted"}

    # ---- Infineon frozen-protocol contrasts (n=2400 instances, 12 users, 4 folds x 600) ----
    # PRIMARY analysis is USER-CLUSTERED (12 clusters): instance-level CIs are anti-conservative
    # (frames within a recording correlate; the narrow-core label makes instances non-independent).
    # McNemar is kept as a descriptive statistic only — its independence assumption fails here.
    zf = load_preds("final_infineon_preds.npz")
    zpl = load_preds("baselines2_preds.npz")
    zxx = load_preds("baselines4_preds.npz")
    cl = load_infineon_clusters()
    ours = per_instance_correct(zf, "map_v_sum|ep40")
    ours30 = per_instance_correct(zf, "map_v_sum|ep30")
    comp = per_instance_correct(zf, "map_composite|ep40")
    occ = per_instance_correct(zf, "REF_occupancy|ep40")
    ds = per_instance_correct(zf, "DeepSets_full|ep40")
    pl = per_instance_correct(zpl, "Infineon|PointLSTM|lr1e-3|ep40")
    pl30 = per_instance_correct(zpl, "Infineon|PointLSTM|lr1e-3|ep30")
    xx = per_instance_correct(zxx, "Infineon|XiaXu2022|lr1e-3|ep40")

    inf = {}
    # TOST at BOTH frozen budgets — equivalence is budget-conditional and must be reported so.
    inf["ours_vs_PointLSTM_TOST"] = {**tost(ours, pl, clusters=cl), "mcnemar": mcnemar(ours, pl),
                                     "instance_level_ci95": tost(ours, pl)["ci95"]}
    inf["ours_vs_PointLSTM_TOST_ep30"] = {**tost(ours30, pl30, clusters=cl),
                                          "mcnemar": mcnemar(ours30, pl30)}
    inf["ours_vs_XiaXu"] = {**paired_diff_ci(ours, xx, clusters=cl), "mcnemar": mcnemar(ours, xx)}
    inf["ours_vs_DeepSets"] = {**paired_diff_ci(ours, ds, clusters=cl), "mcnemar": mcnemar(ours, ds)}
    inf["ours_vs_REFocc"] = {**paired_diff_ci(ours, occ, clusters=cl), "mcnemar": mcnemar(ours, occ)}
    inf["composite_vs_REFocc"] = {**paired_diff_ci(comp, occ, clusters=cl), "mcnemar": mcnemar(comp, occ)}
    out["infineon_ep40"] = inf

    # ---- point-domain velocity contribution (the significant, binning-free claim) ----
    # from p1_crossparadigm.json aggregate arrays (per-fold), DeepSets, mHomeGes
    p1 = json.load(open(os.path.join(DOCS, "p1_crossparadigm.json")))["results"]

    def foldmeans(k):
        """Per-fold means from the flat [folds*seeds] array (n_seeds=3 fixed);
        MM-Fi here is a single S2 split x 3 seeds -> 1 fold (its 5-fold data is in converge_body)."""
        v = np.array(p1[k][2], dtype=float)
        nf = max(1, len(v) // 3)
        return v.reshape(nf, 3).mean(1), nf
    pt = {}
    for dset in ["mHomeGes", "MM-Fi", "Infineon"]:
        full, nf = foldmeans(f"{dset}|DeepSets|full(xyzvAt)|ep40")
        novel, _ = foldmeans(f"{dset}|DeepSets|intensity(xyzAt)|ep40")  # no-velocity featureset
        occp, _ = foldmeans(f"{dset}|DeepSets|occupancy(xyzt)|ep40")
        velp, _ = foldmeans(f"{dset}|DeepSets|velocity(xyzvt)|ep40")
        d_drop = full - novel
        entry = dict(n_folds=nf, full=float(full.mean()), no_velocity=float(novel.mean()),
                     velocity=float(velp.mean()), occupancy=float(occp.mean()),
                     velocity_contribution_pp=float(d_drop.mean()))
        if nf >= 3:  # fold-cluster bootstrap only meaningful with >=3 clusters
            rng = np.random.default_rng(SEED)
            bs = np.array([d_drop[rng.integers(0, nf, nf)].mean() for _ in range(BOOT)])
            entry["velocity_contribution_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        else:
            entry["velocity_contribution_ci"] = None  # single split -> point estimate only
        pt[dset] = entry
    out["point_domain_velocity"] = pt

    # The fold-level CI above resamples 4 designed folds, which contradicts the
    # declared protocol (Sec. III-C resamples users). Recompute the BGT60TR13C
    # contrast on the 12 users; this is the value the manuscript quotes.
    zp1 = load_preds("p1_crossparadigm_preds.npz")
    cl_inf = load_infineon_clusters()
    out["point_domain_velocity_BGT60TR13C_userclustered"] = {
        ep: paired_diff_ci(
            per_instance_correct(zp1, f"Infineon|DeepSets|full(xyzvAt)|{ep}", folds=4, seeds=3),
            per_instance_correct(zp1, f"Infineon|DeepSets|intensity(xyzAt)|{ep}", folds=4, seeds=3),
            clusters=cl_inf)
        for ep in ("ep30", "ep40")}
    out["point_domain_velocity_BGT60TR13C_userclustered"]["_note"] = (
        "supersedes point_domain_velocity.Infineon.velocity_contribution_ci, which "
        "bootstraps 4 folds rather than the 12 users the protocol declares")

    # ---- mHomeGes deficit: compact velocity map vs PointLSTM (honest "point nets win on dense") ----
    # ours = converged map v_sum (ep120, width32); PointLSTM = ep40; both subject-disjoint 5-fold.
    # subject-cluster bootstrap over the 25 mHomeGes subjects (SUBJ__mHomeGes maps inst-id -> subject).
    zc = load_preds("converge_mh_preds.npz")
    subj = zc["SUBJ__mHomeGes"]
    ours_mh = per_instance_correct(zc, "mHomeGes|v_sum|ep120", folds=5, seeds=3)
    pl_mh = per_instance_correct(zpl, "mHomeGes|PointLSTM|lr1e-3|ep40", folds=5, seeds=3)
    ids = sorted(set(ours_mh) & set(pl_mh))
    offset = min(ids)  # inst-id -> SUBJ index
    by_subj = {}
    for i in ids:
        s = str(subj[i - offset])
        by_subj.setdefault(s, []).append(ours_mh[i] - pl_mh[i])
    subs = sorted(by_subj)
    subj_diff = np.array([np.mean(by_subj[s]) for s in subs]) * 100.0  # per-subject mean diff (pp)
    rng = np.random.default_rng(SEED)
    ns = len(subs)
    bs = np.array([subj_diff[rng.integers(0, ns, ns)].mean() for _ in range(BOOT)])
    inst_mean = float(np.array([ours_mh[i] - pl_mh[i] for i in ids]).mean() * 100.0)
    out["mHomeGes_ours_vs_PointLSTM"] = dict(
        n_subjects=ns, n_instances=len(ids),
        mean_pp=float(subj_diff.mean()),          # subject-balanced macro (matches the clustered CI)
        mean_pp_instance_weighted=inst_mean,       # instance-weighted, for reference (subject imbalance 218-2411)
        subject_cluster_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        mcnemar=mcnemar(ours_mh, pl_mh),
        note="ours=map v_sum ep120 (width32; UNDERFIT, min train acc 0.62 -- the deficit is if anything understated); PointLSTM ep40; macro-mean over 25 subjects")

    # ---- convergence rerun (cold-review S1 + P5), mHomeGes ep120, subject-clustered ----
    # converge_point_preds instance indices share the mhomeges_instances() ordering with
    # converge_mh_preds (same loader/path), so SUBJ__mHomeGes maps both.
    zcp = load_preds("converge_point_preds.npz")
    cl_mh = {i: str(subj[i]) for i in range(len(subj))}
    arm = lambda a: per_instance_correct(zcp, f"mHomeGes|{a}|ep120", folds=5, seeds=3)
    full, novl = arm("full(xyzvAt)"), arm("no-velocity(xyzAt)")
    insh, cros = arm("v-inshuffle"), arm("v-crossshuffle")
    velo, occu = arm("velocity(xyzvt)"), arm("occupancy(xyzt)")
    out["converge_point_mHomeGes_ep120"] = {
        "_note": "S1 resolution: full arm converged (min_train 0.958). Ladder: coupling + "
                 "distribution + dimensionality; cross-shuffle keeps marginal v stats + dims "
                 "but destroys class-relevant v info (P5 control).",
        "velocity_contribution (full - no-velocity)": paired_diff_ci(full, novl, clusters=cl_mh),
        "velocity_info_dim-matched (full - crossshuffle)": paired_diff_ci(full, cros, clusters=cl_mh),
        "per-point_coupling (full - inshuffle)": paired_diff_ci(full, insh, clusters=cl_mh),
        "instance_v-distribution (inshuffle - crossshuffle)": paired_diff_ci(insh, cros, clusters=cl_mh),
        "dimensionality_control (crossshuffle - no-velocity)": paired_diff_ci(cros, novl, clusters=cl_mh),
        "velocity_vs_occupancy": paired_diff_ci(velo, occu, clusters=cl_mh),
    }

    # ---- histogram-recovery contrasts (referee request 2026-07-18): converge_body ep120,
    # subject-clustered (MM-Fi 40 subj, mRI 20 subj, 5 folds x 3 seeds each) ----
    zcb = load_preds("converge_body_preds.npz")
    for ds in ("MM-Fi", "mRI"):
        sb = zcb[f"SUBJ__{ds}"]
        cl_ds = {i: str(sb[i]) for i in range(len(sb))}
        arm_ds = lambda a: per_instance_correct(zcb, f"{ds}|{a}|ep120", folds=5, seeds=3)
        vs, vh, sh = arm_ds("v_sum"), arm_ds("v_hist4"), arm_ds("v_hist4_vshuffled")
        out[f"hist_recovery_{ds}_ep120"] = {
            "hist_minus_scalar": paired_diff_ci(vh, vs, clusters=cl_ds),
            "hist_minus_shuffled (distribution component)": paired_diff_ci(vh, sh, clusters=cl_ds),
            "shuffled_minus_scalar": paired_diff_ci(sh, vs, clusters=cl_ds),
        }

    # ---- CFAR operating-point sensitivity (referee R1, 2026-07-18): per detector variant,
    # user-clustered velocity-vs-occupancy (map) and velocity-contribution (point) at ep40.
    # Baseline ca/alpha=2.5 rows for comparison: infineon_ep40.ours_vs_REFocc (+3.5 [0.9,6.5])
    # and point_domain_velocity.Infineon (+0.6 [-0.3,+1.5]).
    import pickle
    zcf = load_preds("cfar_sensitivity_preds.npz")
    cfar = {}
    for vn in ("ca_a2.0", "ca_a3.5", "os_a2.5"):
        pklv = os.path.abspath(os.path.join(DOCS, "..", "data", f"infineon_recs_cfar_{vn}.pkl"))
        recs_v = pickle.load(open(pklv, "rb"))
        cl_v = {i: str(t[2]) for i, t in enumerate(recs_v)}
        armv = lambda a, ep: per_instance_correct(zcf, f"{vn}|{a}|ep{ep}", folds=4, seeds=3)
        entry = {}
        for ep in (30, 40):
            entry[f"map_velocity_minus_occupancy_ep{ep}"] = paired_diff_ci(
                armv("map_v_sum", ep), armv("REF_occupancy", ep), clusters=cl_v)
            entry[f"point_velocity_contribution_ep{ep}"] = paired_diff_ci(
                armv("DeepSets_full(xyzvAt)", ep), armv("DeepSets_intensity(xyzAt)", ep), clusters=cl_v)
        cfar[vn] = entry
    out["cfar_sensitivity"] = cfar

    # ---- mean-form velocity arm (referee W1, 2026-07-19): map ordering with the count
    # coupling removed (per-cell MEAN of signed Doppler instead of SUM). vmean_arm.json.
    zvm = load_preds("vmean_arm_preds.npz")
    vm_inf = per_instance_correct(zvm, "Infineon|map_v_mean|ep40", folds=4, seeds=3)
    vm_mh = per_instance_correct(zvm, "mHomeGes|map_v_mean|ep120", folds=5, seeds=3)
    occ_mh = per_instance_correct(zc, "mHomeGes|occupancy|ep120", folds=5, seeds=3)
    out["vmean_W1"] = {
        "_note": "v_mean underfit on Infineon (min_train 0.896/0.907) - read cautiously there",
        "mHomeGes_vmean_minus_occupancy_ep120": paired_diff_ci(vm_mh, occ_mh, clusters=cl_mh),
        "Infineon_vmean_minus_occupancy_ep40": paired_diff_ci(vm_inf, occ, clusters=cl),
    }

    # ---- MM-Fi 5-fold point-domain ladder (panel ask #1, 2026-07-19): medium_asks.json.
    # Instance ordering matches converge_body's loader, so SUBJ__MM-Fi maps both.
    zma = load_preds("medium_asks_preds.npz")
    sb_mf = zcb["SUBJ__MM-Fi"]
    cl_mf = {i: str(sb_mf[i]) for i in range(len(sb_mf))}
    arm_mf = lambda a: per_instance_correct(zma, f"MMFi5f|{a}|ep240", folds=5, seeds=3)
    f_, n_, i_, c_ = (arm_mf("full(xyzvAt)"), arm_mf("no-velocity(xyzAt)"),
                      arm_mf("v-inshuffle"), arm_mf("v-crossshuffle"))
    out["MMFi_5fold_ladder_ep240"] = {
        "_note": "40-subject clustered; full arm converged (0.970), inshuffle 0.951; "
                 "no-vel/crossshuffle mildly under gate (0.911/0.922)",
        "velocity_contribution (full - no-velocity)": paired_diff_ci(f_, n_, clusters=cl_mf),
        "velocity_info_dim-matched (full - crossshuffle)": paired_diff_ci(f_, c_, clusters=cl_mf),
        "per-point_coupling (full - inshuffle)": paired_diff_ci(f_, i_, clusters=cl_mf),
        "instance_v-distribution (inshuffle - crossshuffle)": paired_diff_ci(i_, c_, clusters=cl_mf),
        "dimensionality_control (crossshuffle - no-velocity)": paired_diff_ci(c_, n_, clusters=cl_mf),
    }

    # ---- N1: point-domain single-quantity ordering (reviewer ask, 2026-07-20) ----
    # velocity-only=velocity(xyzvt), intensity-only=no-velocity(xyzAt), occupancy-only=occupancy(xyzt).
    # Gives the DIRECT velocity-vs-intensity point-domain contrast (not only velocity-vs-occupancy).
    # intensity-only featureset is named "no-velocity(xyzAt)" in converge_point but
    # "intensity(xyzAt)" in p1_crossparadigm (same [x,y,z,A,t] tuple, different npz tag).
    def _singleq(z, ds, ep, folds, seeds, clusters, intkey="no-velocity(xyzAt)"):
        vel = per_instance_correct(z, f"{ds}|velocity(xyzvt)|ep{ep}", folds, seeds)
        itn = per_instance_correct(z, f"{ds}|{intkey}|ep{ep}", folds, seeds)
        ocp = per_instance_correct(z, f"{ds}|occupancy(xyzt)|ep{ep}", folds, seeds)
        return {"velocity_minus_intensity": paired_diff_ci(vel, itn, clusters=clusters),
                "velocity_minus_occupancy": paired_diff_ci(vel, ocp, clusters=clusters),
                "intensity_minus_occupancy": paired_diff_ci(itn, ocp, clusters=clusters)}
    zp1 = load_preds("p1_crossparadigm_preds.npz")
    cl_mf_cp = {i: str(zcb["SUBJ__MM-Fi"][i]) for i in range(len(zcb["SUBJ__MM-Fi"]))}
    out["single_quantity_mHomeGes_ep120"] = _singleq(zcp, "mHomeGes", 120, 5, 3, cl_mh)
    out["single_quantity_MMFi_ep240_8subj"] = {
        "_note": "converge_point MM-Fi = single 8-subject split (direction-only); "
                 "5-fold single-quantity arms were not run",
        **_singleq(zcp, "MM-Fi", 240, 5, 3, cl_mf_cp)}
    out["single_quantity_Infineon_ep40"] = _singleq(
        zp1, "Infineon|DeepSets", 40, 4, 3, cl, intkey="intensity(xyzAt)")

    # ---- N3: map vs DeepSets at ep30 (ep40 already in infineon_ep40.ours_vs_DeepSets) ----
    ds30 = per_instance_correct(zf, "DeepSets_full|ep30")
    out["infineon_ep40"]["ours_vs_DeepSets_ep30"] = {**paired_diff_ci(ours30, ds30, clusters=cl),
                                                     "mcnemar": mcnemar(ours30, ds30)}

    # ---- N7: class-balanced (macro-recall) accuracy, reviewer ask (per-class robustness) ----
    def _balanced(z, prefix, folds, seeds):
        ks = [k for k in z.files if k.startswith(prefix + "|")]
        ft, st = ("fold", "seed") if any("|fold" in k for k in ks) else ("f", "s")
        yts, yps = [], []
        for f in range(folds):
            for s in range(seeds):
                k = f"{prefix}|{ft}{f}|{st}{s}"
                if k in z.files:
                    a = z[k]; yts.append(a[1]); yps.append(a[2])
        yt = np.concatenate(yts); yp = np.concatenate(yps)
        rec = {int(c): float((yp[yt == c] == c).mean()) for c in np.unique(yt)}
        return dict(micro_top1=round(float((yt == yp).mean()) * 100, 2),
                    macro_balanced=round(float(np.mean(list(rec.values()))) * 100, 2),
                    n_classes=len(rec), worst_class_recall=round(min(rec.values()) * 100, 1))
    out["class_balanced_mHomeGes_point_ep120"] = {
        arm: _balanced(zcp, f"mHomeGes|{key}|ep120", 5, 3) for arm, key in
        [("full", "full(xyzvAt)"), ("velocity_only", "velocity(xyzvt)"),
         ("intensity_only", "no-velocity(xyzAt)"), ("occupancy_only", "occupancy(xyzt)")]}
    out["class_balanced_MMFi_map_ep120"] = {
        arm: _balanced(zcb, f"MM-Fi|{key}|ep120", 5, 3) for arm, key in
        [("v_sum", "v_sum"), ("v_hist4", "v_hist4")]}

    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "paired_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote figures/paired_stats.json")


if __name__ == "__main__":
    main()
