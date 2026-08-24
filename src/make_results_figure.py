"""Aggregate the 3-arm cross-subject results into one figure + RESULTS.md.

Reads docs/results.json (velocity / intensity / occupancy per dataset) and renders a
grouped bar chart with the velocity-vs-geometry and velocity-vs-intensity gaps.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
res = json.load(open(os.path.join(DOCS, "results.json")))

labels = [f"{r['dataset']}\n{r['subjects']} subj / {r['classes']} cls\n{r['task']}" for r in res]
vel = [r["velocity"] for r in res]
inten = [r["intensity"] for r in res]
occ = [r["occupancy"] for r in res]
chance = [r["chance"] for r in res]

x = np.arange(len(res)); w = 0.26
fig, ax = plt.subplots(figsize=(2.9 * len(res) + 3, 5.4))
b1 = ax.bar(x - w, vel, w, label="Velocity (XTD/YTD/ZTD)", color="#1f77b4")
b2 = ax.bar(x, occ, w, label="Occupancy = geometry (XTO/YTO/ZTO)", color="#2ca02c")
b3 = ax.bar(x + w, inten, w, label="Amplitude (XTA/YTA/ZTA)", color="#d62728")
for i, r in enumerate(res):
    ax.plot([x[i]-0.45, x[i]+0.45], [chance[i], chance[i]], "k--", lw=1)
    g = r["gap_vel_occ"]
    ax.annotate(f"vel-geom {g:+.1f}", (x[i], max(vel[i], occ[i], inten[i]) + 2.5),
                ha="center", fontsize=10, fontweight="bold",
                color=("#1f77b4" if g >= 0 else "#d62728"))
for b in (b1, b2, b3):
    ax.bar_label(b, fmt="%.1f", padding=1, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cross-subject accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("What to accumulate? Velocity vs geometry vs amplitude, cross-subject\n"
             "(same fixed CNN per bar; dashed = chance)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out = os.path.join(DOCS, "crosssubject_3arm.png")
fig.savefig(out, dpi=130); print("saved", out)

# RESULTS.md
L = ["# Cross-subject results: what per-point quantity to accumulate\n",
     "Same fixed lightweight CNN per bar; only the accumulated per-point quantity changes.",
     "Velocity = radial-velocity accumulation; Amplitude = intensity; Occupancy = per-bin point count (geometry control).\n",
     "| Dataset | Task | Subj | Cls | Protocol | Velocity % | Occupancy % | Amplitude % | vel-geom | vel-amp | Chance % |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    L.append(f"| {r['dataset']} | {r['task']} | {r['subjects']} | {r['classes']} | {r['protocol']} | "
             f"**{r['velocity']:.2f}** | {r['occupancy']:.2f} | {r['intensity']:.2f} | "
             f"**{r['gap_vel_occ']:+.2f}** | {r['gap_vel_int']:+.2f} | {r['chance']:.1f} |")
L += ["", "Notes:"]
for r in res:
    L.append(f"- **{r['dataset']}**: {r['note']}")
open(os.path.join(DOCS, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("wrote RESULTS.md")
