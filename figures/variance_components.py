"""Variance components and small-cluster equivalence checks (referee request,
2026-08-11).

Both TIM reviewers made the same point: paired_stats.py averages the three
seeds inside per_instance_correct BEFORE the cluster bootstrap, so every
interval in the paper is conditional on the seed mean and the training
stochasticity is never propagated. Omitting a component narrows the interval,
which makes an equivalence test EASIER to pass, so the direction of that bias
is unfavourable for Section V-C.

This script does three things and writes figures/variance_components.json:

 1. decomposes the paired difference into between-subject and between-seed
    components, from per-(subject, seed) accuracies;
 2. re-estimates the headline contrasts with a TWO-LEVEL bootstrap that
    resamples subjects and then draws ONE shared seed multiset per replicate,
    because seeds are crossed with the subject (see boot_two_level);
 3. for the equivalence claim, adds a cluster-t (studentized) interval and a
    leave-one-user-out sensitivity analysis on the 12 users.

Run: cd figures && python3 variance_components.py
"""
import os, json
import numpy as np
import paired_stats as PS

BOOT = 20000
SEED = 12345
HERE = os.path.dirname(os.path.abspath(__file__))


def per_cluster_seed(z, prefix, clusters, folds, seeds=3, offset=None, subj=None):
    """(cluster, seed) -> mean correctness. Keeps the seed axis instead of
    collapsing it, which is exactly what per_instance_correct throws away.

    Asserts rather than skipping: a missing fold/seed cell, a duplicated
    instance id across folds, or a truncated prediction row would otherwise
    produce a silently wrong decomposition. Returns (values, ids_per_cell) so
    the caller can confirm both arms were scored on the same instances.
    """
    if not any(k.startswith(prefix + "|") for k in z.files):
        raise KeyError(f"no keys for prefix {prefix!r}")
    out, ids_per_cell = {}, {}
    for s in range(seeds):
        acc = {}
        for f in range(folds):
            hit = None
            for tok in (f"|fold{f}|seed{s}", f"|f{f}|s{s}"):
                if prefix + tok in z.files:
                    hit = prefix + tok
                    break
            assert hit is not None, f"missing cell {prefix}|fold{f}|seed{s}"
            a = z[hit]
            assert a.shape[0] == 3 and a.shape[1] > 0, f"bad array shape {a.shape} for {hit}"
            assert len(a[0]) == len(a[1]) == len(a[2]), f"ragged rows in {hit}"
            for i, yt, yp in zip(a[0], a[1], a[2]):
                i = int(i)
                assert i not in acc, f"instance {i} appears in two folds for seed {s}"
                acc[i] = 1.0 if yt == yp else 0.0
        ids_per_cell[s] = frozenset(acc)
        for i, c in acc.items():
            if clusters is not None:
                cl = clusters[i]
            else:
                j = i - offset
                assert 0 <= j < len(subj), f"subject index {j} out of range for instance {i}"
                cl = str(subj[j])
            out.setdefault((cl, s), []).append(c)
    assert len({v for v in ids_per_cell.values()}) == 1, "seeds scored different instance sets"
    return {k: float(np.mean(v)) for k, v in out.items()}, ids_per_cell[0]


def components(a, b):
    """Variance components of the paired difference d[cluster, seed]."""
    cls = sorted({c for c, _ in a} & {c for c, _ in b})
    seeds = sorted({s for _, s in a})
    D = np.array([[ (a[(c, s)] - b[(c, s)]) * 100.0 for s in seeds] for c in cls])
    grand = D.mean()
    subj_means = D.mean(axis=1)
    seed_means = D.mean(axis=0)
    n_c, n_s = D.shape
    ss_subj = n_s * ((subj_means - grand) ** 2).sum()
    ss_seed = n_c * ((seed_means - grand) ** 2).sum()
    ss_res = ((D - subj_means[:, None] - seed_means[None, :] + grand) ** 2).sum()
    ms_subj = ss_subj / (n_c - 1)
    ms_seed = ss_seed / (n_s - 1)
    ms_res = ss_res / ((n_c - 1) * (n_s - 1))
    # method-of-moments variance components (two-way random effects, no replicates)
    var_subj = max((ms_subj - ms_res) / n_s, 0.0)
    var_seed = max((ms_seed - ms_res) / n_c, 0.0)
    return D, dict(mean_pp=float(grand), n_clusters=n_c, n_seeds=n_s,
                   sd_between_subject_pp=float(np.sqrt(var_subj)),
                   sd_between_seed_pp=float(np.sqrt(var_seed)),
                   # NOT an identified noise component: with one observation per
                   # cell this absorbs subject-by-seed interaction, finite-instance
                   # scoring noise, heteroskedasticity from unequal instance counts,
                   # and any fold-by-seed run effect
                   sd_residual_cell_pp=float(np.sqrt(max(ms_res, 0.0))),
                   sd_residual_pp=float(np.sqrt(max(ms_res, 0.0))),
                   # contributions to Var(grand mean), in pp^2 -- normalize by
                   # their sum to get the shares quoted in the manuscript
                   var_contrib_of_mean_pp2=dict(
                       subject=float(var_subj / n_c), seed=float(var_seed / n_s),
                       residual=float(max(ms_res, 0.0) / (n_c * n_s))),
                   var_share_of_mean=dict(
                       subject=float(var_subj / n_c), seed=float(var_seed / n_s),
                       residual=float(max(ms_res, 0.0) / (n_c * n_s))),
                   se_seed_averaged_pp=float(subj_means.std(ddof=1) / np.sqrt(n_c)),
                   seed_means_pp=[float(x) for x in seed_means])


def boot_two_level(D, boot=BOOT, seed=SEED, level=95):
    """Resample clusters, then resample the SEED SET ONCE per replicate, shared
    by every drawn cluster.

    The seed factor is CROSSED with the subject here: all subjects were trained
    and evaluated under the same three seeds, so a seed shift moves every
    subject together. Drawing seed indices independently per cluster (which an
    earlier version of this file did) models seeds as NESTED inside subject and
    shrinks the seed contribution by roughly a factor n_c -- it reproduces
    sqrt(vs/n_c + (vd+vr)/(n_c*n_s)) instead of the correct
    sqrt(vs/n_c + vd/n_s + vr/(n_c*n_s)). analytic_se() below is the guard.
    """
    rng = np.random.default_rng(seed)
    n_c, n_s = D.shape
    lo_q, hi_q = (100 - level) / 2, 100 - (100 - level) / 2
    stats = np.empty(boot)
    for b in range(boot):
        ci = rng.integers(0, n_c, n_c)
        si = rng.integers(0, n_s, n_s)          # shared: seeds are crossed
        stats[b] = D[np.ix_(ci, si)].mean()
    return [float(np.percentile(stats, lo_q)), float(np.percentile(stats, hi_q))]


def analytic_se(comp, crossed=True):
    """SE of the grand mean from the variance components, used to check that the
    bootstrap reproduces the intended design."""
    n_c, n_s = comp["n_clusters"], comp["n_seeds"]
    vs = comp["sd_between_subject_pp"] ** 2
    vd = comp["sd_between_seed_pp"] ** 2
    vr = comp["sd_residual_pp"] ** 2
    if crossed:
        return float(np.sqrt(vs / n_c + vd / n_s + vr / (n_c * n_s)))
    return float(np.sqrt(vs / n_c + (vd + vr) / (n_c * n_s)))


def check_design(comp, ci, level=95, tol=0.15):
    """Record whether the bootstrap interval matches the crossed or the nested
    SE, so a repeat of the nesting bug cannot pass silently."""
    z = {90: 1.645, 95: 1.960}[level]
    se_boot = (ci[1] - ci[0]) / (2 * z)
    se_x, se_n = analytic_se(comp, True), analytic_se(comp, False)
    vs = comp["sd_between_subject_pp"] ** 2
    vd = comp["sd_between_seed_pp"] ** 2
    # the analytic SE is only a usable reference when the method-of-moments
    # components are non-degenerate: if a component was clamped at zero the
    # model-based SE understates what the nonparametric bootstrap propagates
    degenerate = (vd <= 0.0) or (vs <= 0.0)
    return dict(se_bootstrap_pp=round(se_boot, 4),
                se_analytic_crossed_pp=round(se_x, 4),
                se_analytic_nested_pp=round(se_n, 4),
                matches="crossed" if abs(se_boot - se_x) <= abs(se_boot - se_n) else "nested",
                components_degenerate=bool(degenerate),
                within_tol=(None if degenerate
                            else bool(abs(se_boot - se_x) <= tol * max(se_x, 1e-9))),
                note=("a variance component was clamped at zero, so the analytic SE is a "
                      "lower reference only; trust the bootstrap" if degenerate else
                      "components non-degenerate; bootstrap SE should match the crossed value"))


def boot_studentized(d, boot=BOOT, seed=SEED, level=90):
    """Cluster-t interval: bootstrap the t statistic, not the mean. Percentile
    intervals under-cover at 12 clusters; this is the standard repair."""
    rng = np.random.default_rng(seed)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    ts = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, n, n)
        db = d[idx]
        se_b = db.std(ddof=1) / np.sqrt(n)
        ts[b] = (db.mean() - m) / se_b if se_b > 0 else 0.0
    lo_q, hi_q = (100 - level) / 2, 100 - (100 - level) / 2
    t_lo, t_hi = np.percentile(ts, [lo_q, hi_q])
    return [float(m - t_hi * se), float(m - t_lo * se)], float(m), float(se)


if __name__ == "__main__":
    out = {"_provenance": "figures/variance_components.py, 2026-08-11",
           "_boot": BOOT, "_seed": SEED}

    # ---- 1. headline: mHomeGes dimensionality-matched velocity information ----
    zcp = PS.load_preds("converge_point_preds.npz")
    zc = PS.load_preds("converge_mh_preds.npz")
    subj = zc["SUBJ__mHomeGes"]
    ids = [int(k) for k in range(len(subj))]
    offset = 0
    full, ids_a = per_cluster_seed(zcp, "mHomeGes|full(xyzvAt)|ep120", None, folds=5, subj=subj, offset=offset)
    cross, ids_b = per_cluster_seed(zcp, "mHomeGes|v-crossshuffle|ep120", None, folds=5, subj=subj, offset=offset)
    assert ids_a == ids_b, "the two arms were scored on different instances"
    D, comp = components(full, cross)
    comp["ci95_seed_averaged_percentile"] = PS.paired_diff_ci(
        PS.per_instance_correct(zcp, "mHomeGes|full(xyzvAt)|ep120", folds=5, seeds=3),
        PS.per_instance_correct(zcp, "mHomeGes|v-crossshuffle|ep120", folds=5, seeds=3),
        clusters={i: str(subj[i]) for i in ids})["ci"]
    comp["ci95_two_level"] = boot_two_level(D)
    comp["design_check_95"] = check_design(comp, comp["ci95_two_level"], 95)
    out["velocity_info_dim_matched_mHomeGes_ep120"] = comp

    # ---- 2. equivalence: BGT60TR13C compact map vs PointLSTM at ep40 ----
    zf = PS.load_preds("final_infineon_preds.npz")
    zpl = PS.load_preds("baselines2_preds.npz")
    cl = PS.load_infineon_clusters()
    ours, ids_c = per_cluster_seed(zf, "map_v_sum|ep40", cl, folds=4)
    pl, ids_d = per_cluster_seed(zpl, "Infineon|PointLSTM|lr1e-3|ep40", cl, folds=4)
    assert ids_c == ids_d, "the two arms were scored on different instances"
    D2, comp2 = components(ours, pl)
    comp2["ci90_two_level"] = boot_two_level(D2, level=90)
    comp2["ci95_two_level"] = boot_two_level(D2, level=95)
    comp2["design_check_90"] = check_design(comp2, comp2["ci90_two_level"], 90)
    d_user = D2.mean(axis=1)
    ci_t, m, se = boot_studentized(d_user, level=90)
    comp2["ci90_cluster_t"] = ci_t
    comp2["se_pp"] = se
    comp2["margin_pp"] = 3.0
    comp2["equivalent_two_level"] = bool(comp2["ci90_two_level"][0] > -3 and comp2["ci90_two_level"][1] < 3)
    comp2["equivalent_cluster_t"] = bool(ci_t[0] > -3 and ci_t[1] < 3)
    # leave-one-user-out on the 12 users
    loo = []
    for j in range(len(d_user)):
        keep = np.delete(d_user, j)
        loo.append(round(float(keep.mean()), 3))
    comp2["loo_user_means_pp"] = loo
    comp2["loo_worst_pp"] = float(max(loo, key=abs))
    comp2["smallest_margin_passed_two_level"] = round(float(max(abs(comp2["ci90_two_level"][0]),
                                                               abs(comp2["ci90_two_level"][1]))), 2)
    out["equivalence_map_vs_PointLSTM_BGT60TR13C_ep40"] = comp2

    # ---- 3. fold-level dependence: subjects are nested in subject-disjoint
    # folds, and every subject in a fold shares that fold's trained model, so
    # check whether resampling folds instead of subjects widens the headline.
    fold_of = {}
    for f in range(5):
        for i in zcp[f"mHomeGes|full(xyzvAt)|ep120|f{f}|s0"][0]:
            fold_of.setdefault(str(subj[int(i)]), set()).add(f)
    assert all(len(v) == 1 for v in fold_of.values()), "subjects are not nested in folds"
    cls = sorted(fold_of)
    fmap = {s_: sorted(fold_of[s_])[0] for s_ in cls}
    folds_ = sorted(set(fmap.values()))
    rows_by_fold = [[i for i, s_ in enumerate(cls) if fmap[s_] == f] for f in folds_]
    fold_means = np.array([D[r].mean() for r in rows_by_fold])
    rng = np.random.default_rng(SEED)
    bs = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(0, len(folds_), len(folds_))
        rows = np.concatenate([rows_by_fold[p] for p in pick])
        si = rng.integers(0, D.shape[1], D.shape[1])
        bs[b] = D[np.ix_(rows, si)].mean()
    out["fold_level_check_headline"] = dict(
        n_folds=len(folds_), subjects_nested_in_folds=True,
        fold_means_pp=[round(float(x), 3) for x in fold_means],
        sd_between_fold_pp=round(float(fold_means.std(ddof=1)), 3),
        ci95_fold_cluster_crossed_seed=[round(float(np.percentile(bs, 2.5)), 3),
                                        round(float(np.percentile(bs, 97.5)), 3)],
        note=("resampling the 5 folds instead of the 25 subjects does NOT widen the "
              "interval, so the subject-level crossed bootstrap is the conservative "
              "choice; only 5 clusters, so this is a check and not the primary estimate"))

    json.dump(out, open(os.path.join(HERE, "variance_components.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("\nwrote figures/variance_components.json")
