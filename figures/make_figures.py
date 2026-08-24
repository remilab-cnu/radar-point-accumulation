"""make_figures.py — generate F1-F4 for the Phase-1 measurement paper.

Reads ONLY audit-clean sources in experiments/docs/ plus figures/paired_stats.json
(produced by paired_stats.py). No hand-typed accuracies except the C-index (which is
a derived scene statistic, sourced from cancel_stat.py via THEORY_SECTION.md) and the
parameter counts (measured, provenance noted). Writes fig{1..4}_*.pdf and .png here.

Design notes / honesty guardrails baked in:
  * F1 uses the CLEAN converged map runs (converge_mh/converge_body) and the frozen
    Infineon run (final_infineon) — NOT the banned map_ref gate constants.
  * F1 map bars that are underfit (UF, min train acc < 0.95) are hatched, so the reader
    sees the compact map did not converge on dense mHomeGes.
  * F3 plots ALL point-net baselines honestly: DGCNN/PointNet++ sit ABOVE ours. The claim
    is Pareto efficiency at the low-parameter end (TOST-equal to PointLSTM at 1/7 params),
    not the accuracy frontier.
  * Every paired number in F2 comes from paired_stats.json.

Run:  cd figures && python3 paired_stats.py && python3 make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, "..", "experiments", "docs"))
plt.rcParams.update({"font.size": 9, "axes.grid": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

# colour-blind-safe (Okabe-Ito): velocity=blue, intensity=orange, occupancy=green
C_VEL, C_INT, C_OCC = "#0072B2", "#E69F00", "#009E73"
DATASETS = ["mHomeGes", "MM-Fi", "Infineon"]


def jload(name):
    return json.load(open(os.path.join(DOCS, name)))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


# ----------------------------------------------------------------------------- F1
def fig1_arms():
    """Accumulation-quantity comparison (velocity/intensity/occupancy) x dataset,
    two paradigms. Point panel at CONVERGED budgets (mH ep120 / MM-Fi ep240 from
    converge_point; Infineon ep40 from p1, converged at the short budget).
    Map panel adds mRI (converge_body). Hatching = under the 0.95 train gate."""
    p1 = jload("p1_crossparadigm.json")["results"]
    cp = jload("converge_point.json")["results"]

    def cpv(key):
        r = cp[key]; return (r["acc"], r["std"], r.get("underfit", False))

    def ptv(dset, arm):  # p1 ep40 (no train-acc recorded there -> unhatched)
        v = p1[f"{dset}|DeepSets|{arm}|ep40"]
        return (v[0], v[1], False)
    point = {
        "mHomeGes": {"vel": cpv("mHomeGes|velocity(xyzvt)|ep120"),
                     "int": cpv("mHomeGes|no-velocity(xyzAt)|ep120"),
                     "occ": cpv("mHomeGes|occupancy(xyzt)|ep120")},
        "MM-Fi": {"vel": cpv("MM-Fi|velocity(xyzvt)|ep240"),
                  "int": cpv("MM-Fi|no-velocity(xyzAt)|ep240"),
                  "occ": cpv("MM-Fi|occupancy(xyzt)|ep240")},
        "Infineon": {"vel": ptv("Infineon", "velocity(xyzvt)"),
                     "int": ptv("Infineon", "intensity(xyzAt)"),
                     "occ": ptv("Infineon", "occupancy(xyzt)")},
    }

    # map paradigm: clean per-dataset runs (NOT map_ref). (mean, std, underfit)
    cmh = jload("converge_mh.json")["mHomeGes"]["results"]
    cbd = jload("converge_body.json")
    fin = jload("final_infineon.json")["results"]

    def mv(res, k):
        r = res[k]; return (r["acc"], r["std"], r.get("underfit", False))
    mapd = {
        "mHomeGes": {"vel": mv(cmh, "v_sum|ep120"), "int": mv(cmh, "amplitude|ep120"),
                     "occ": mv(cmh, "occupancy|ep120")},
        "MM-Fi": {"vel": mv(cbd["MM-Fi"]["results"], "v_sum|ep120"),
                  "int": mv(cbd["MM-Fi"]["results"], "amplitude|ep120"),
                  "occ": mv(cbd["MM-Fi"]["results"], "occupancy|ep120")},
        "mRI": {"vel": mv(cbd["mRI"]["results"], "v_sum|ep120"),
                "int": mv(cbd["mRI"]["results"], "amplitude|ep120"),
                "occ": mv(cbd["mRI"]["results"], "occupancy|ep120")},
        "Infineon": {"vel": (fin["map_v_sum|ep40"][0], fin["map_v_sum|ep40"][1], False),
                     "int": (fin["REF_int_mean|ep40"][0], fin["REF_int_mean|ep40"][1], False),
                     "occ": (fin["REF_occupancy|ep40"][0], fin["REF_occupancy|ep40"][1], False)},
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    w = 0.26
    for ax, data, title in [
        (axes[0], point, "(a) Point-set paradigm (DeepSets, converged budgets)"),
        (axes[1], mapd, "(b) Compact-map paradigm")]:
        dsets = list(data.keys())
        x = np.arange(len(dsets))
        for j, (arm, col) in enumerate([("vel", C_VEL), ("int", C_INT), ("occ", C_OCC)]):
            means = [data[d][arm][0] for d in dsets]
            errs = [data[d][arm][1] for d in dsets]
            uf = [data[d][arm][2] for d in dsets]
            bars = ax.bar(x + (j-1)*w, means, w, yerr=errs, capsize=2.5, color=col,
                          edgecolor="black", linewidth=0.4,
                          label={"vel": "velocity", "int": "intensity", "occ": "occupancy"}[arm])
            for b, isuf in zip(bars, uf):
                if isuf:
                    b.set_hatch("///"); b.set_alpha(0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(dsets)
        ax.set_ylim(0, 100)
        ax.axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("cross-subject top-1 accuracy (%)")
    handles = [Patch(fc=C_VEL, ec="k"), Patch(fc=C_INT, ec="k"), Patch(fc=C_OCC, ec="k"),
               Patch(fc="white", ec="k", hatch="///")]
    labels = ["velocity", "intensity", "occupancy", "underfit (train acc < 0.95)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("Velocity is the most class-informative quantity to accumulate — "
                 "clearest in the binning-free point domain", fontsize=9.5, y=1.02)
    save(fig, "fig1_arms")


# ----------------------------------------------------------------------------- F2
def fig2_forest():
    """Forest plot of the key paired contrasts (all from paired_stats.json)."""
    ps = json.load(open(os.path.join(HERE, "paired_stats.json")))
    inf = ps["infineon_ep40"]
    rows = []  # (label, mean, lo, hi, kind)
    m = ps["mHomeGes_ours_vs_PointLSTM"]
    rows.append(("mHomeGes: compact map − PointLSTM\n(dense data; point nets win)",
                 m["mean_pp"], m["subject_cluster_ci"][0], m["subject_cluster_ci"][1], "deficit"))
    cp = ps["converge_point_mHomeGes_ep120"]
    v = cp["velocity_info_dim-matched (full - crossshuffle)"]
    rows.append(("mHomeGes point-domain: velocity info at\nmatched dims (full − v-shuffled, ep120 conv.)",
                 v["mean_pp"], v["ci"][0], v["ci"][1], "gain"))
    dc = cp["dimensionality_control (crossshuffle - no-velocity)"]
    rows.append(("mHomeGes: dimensionality control\n(v-shuffled − no-vel; ≈0 expected)",
                 dc["mean_pp"], dc["ci"][0], dc["ci"][1], "equiv"))
    t = inf["ours_vs_PointLSTM_TOST"]
    rows.append(("Infineon: ours − PointLSTM\n(TOST-equivalent, ±3 pp)",
                 t["mean_pp"], t["ci95"][0], t["ci95"][1], "equiv"))
    xx = inf["ours_vs_XiaXu"]
    rows.append(("Infineon: ours − Xia&Xu-style (17.2M)\n(directional; n.s. user-clustered)",
                 xx["mean_pp"], xx["ci"][0], xx["ci"][1], "gain"))
    oc = inf["ours_vs_REFocc"]
    rows.append(("Infineon: ours − occupancy map",
                 oc["mean_pp"], oc["ci"][0], oc["ci"][1], "gain"))

    colmap = {"deficit": "#D55E00", "gain": "#0072B2", "equiv": "#555555"}
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, mean, lo, hi, kind) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=colmap[kind], lw=2)
        ax.plot([mean], [yi], "o", color=colmap[kind], ms=6)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(-3, 3, color="0.9", zorder=0)  # ±3 pp equivalence band
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("paired accuracy difference (percentage points)")
    ax.set_title("Paired contrasts, cluster 95% CIs — subjects (mHomeGes) / users (Infineon, ep40)\n"
                 "(grey = ±3 pp equivalence band)", fontsize=9)
    leg = [Patch(fc="#0072B2"), Patch(fc="#D55E00"), Patch(fc="#555555")]
    ax.legend(leg, ["favours ours", "point nets win (honest)", "equivalent"],
              loc="lower right", frameon=False, fontsize=8)
    save(fig, "fig2_forest")


# ----------------------------------------------------------------------------- F3
def fig3_efficiency():
    """Accuracy vs parameters on the sparse Infineon sensor (ep40, lr1e-3).
    Honest: DGCNN/PointNet++ score above ours; the claim is efficiency, not the frontier."""
    fin = jload("final_infineon.json")["results"]
    b1 = jload("baselines1_fix2.json")["results"]
    b2 = jload("baselines2.json")["results"]
    b3 = jload("baselines3.json")["results"]
    b4 = jload("baselines4.json")["results"]

    def g(res, needle):
        for k, v in res.items():
            if "Infineon" in k and "lr1e-3" in k and "ep40" in k and needle in k:
                return v["mean"]
        raise KeyError(needle)
    # (label, params, accuracy, is_ours) — params measured at the 5-class config
    # (docs/complexity.json, tools_complexity.py)
    pts = [
        ("ours: velocity map", 24_133, fin["map_v_sum|ep40"][0], True),
        ("DeepSets", 42_309, fin["DeepSets_full|ep40"][0], False),
        ("CPDP-reimpl", 57_061, g(b3, "CPDP"), False),
        ("DGCNN", 121_861, g(b1, "DGCNN"), False),
        ("PointLSTM", 175_365, g(b2, "PointLSTM"), False),
        ("PointNet++", 413_189, g(b1, "PointNetPP"), False),
        ("Xia&Xu-style", 17_155_653, g(b4, "XiaXu"), False),
    ]
    ours_acc = pts[0][2]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhspan(ours_acc - 3, ours_acc + 3, color="#0072B2", alpha=0.10,
               label="±3 pp equivalence band (ours)")
    for lab, p, acc, is_ours in pts:
        ax.scatter(p, acc, s=90 if is_ours else 55,
                   color="#D55E00" if is_ours else "#333333",
                   marker="*" if is_ours else "o", zorder=3,
                   edgecolor="black", linewidth=0.5)
        dy = 0.9 if lab not in ("DeepSets", "CPDP-reimpl") else -1.6
        ax.annotate(lab, (p, acc), xytext=(0, 8 if dy > 0 else -12),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("Infineon cross-subject top-1 (%)  ·  ep40")
    ax.set_ylim(60, 100)
    ps = json.load(open(os.path.join(HERE, "paired_stats.json")))["infineon_ep40"]
    t, t30 = ps["ours_vs_PointLSTM_TOST"], ps["ours_vs_PointLSTM_TOST_ep30"]
    ax.set_title(f"Compact velocity map is TOST-equivalent to PointLSTM at 1/7 the params (ep40)\n"
                 f"(user-clustered: {t['mean_pp']:+.2f} pp, 95% CI [{t['ci95'][0]:.2f}, {t['ci95'][1]:.2f}], "
                 f"equivalent within ±3 pp; NOT at ep30: {t30['mean_pp']:+.1f} pp)", fontsize=8.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, "fig3_efficiency")


# ----------------------------------------------------------------------------- F4
def fig4_cancellation():
    """(a) sign-cancellation index C per dataset; (b) sign-resolved histogram recovery on low-C data."""
    # C-index: full-dataset recomputation (docs/c_recompute_full.json, 2026-07-19;
    # per-instance mean, no subsampling). The historical mHomeGes 0.764 was a
    # timed-out n=200 unshuffled-prefix artifact - do not resurrect it.
    C = {"Infineon": 0.958, "mHomeGes": 0.840, "mRI": 0.649, "MM-Fi": 0.488}
    cbd = jload("converge_body.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # (a) C bars
    ax = axes[0]
    names = list(C.keys()); vals = list(C.values())
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, 1.0); ax.set_ylabel(r"cancellation index  $C=|\Sigma v|/\Sigma|v|$")
    ax.set_title("(a) Sign-cancellation by scene type\n(1 = coherent, 0 = fully cancelling)", fontsize=8.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)

    # (b) histogram recovery on low-C datasets (converged ep120)
    ax = axes[1]
    arms = [("v_sum", "scalar sum", "#888888"),
            ("v_hist4_vshuffled", "hist. (Doppler shuffled)", "#E69F00"),
            ("v_hist4", "sign-resolved hist.", "#0072B2")]
    lowC = ["MM-Fi", "mRI"]
    x = np.arange(len(lowC)); w = 0.26
    for j, (arm, lab, col) in enumerate(arms):
        means = [cbd[d]["results"][f"{arm}|ep120"]["acc"] for d in lowC]
        ax.bar(x + (j-1)*w, means, w, color=col, edgecolor="black", linewidth=0.4, label=lab)
    # annotate recovery
    for i, d in enumerate(lowC):
        vs = cbd[d]["results"]["v_sum|ep120"]["acc"]
        vh = cbd[d]["results"]["v_hist4|ep120"]["acc"]
        ax.annotate(f"+{vh-vs:.1f} pp", (i + w, vh + 1.5), ha="center", fontsize=7.5, color="#0072B2")
    ax.set_xticks(x); ax.set_xticklabels(lowC)
    ax.set_ylim(0, 100); ax.set_ylabel("cross-subject top-1 (%)")
    ax.set_title("(b) Sign-resolved histogram recovers low-C data\n(count-matched shuffle isolates the distribution)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig4_cancellation")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HERE, "paired_stats.json")):
        raise SystemExit("run paired_stats.py first (figures/paired_stats.json missing)")
    fig1_arms()
    fig2_forest()
    fig3_efficiency()
    fig4_cancellation()
    print("\nall figures written to", HERE)
