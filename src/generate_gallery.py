"""Build a per-class spectra GALLERY (self-contained HTML) for PI inspection.

Faithful to the original paper's representation: 100 spatial bins x 40 time frames,
per Cartesian axis, max-normalized. For each dataset and each class we render one
representative instance across the three accumulation arms x three axes (9 panels):
  Velocity  (XTD/YTD/ZTD) - signed, diverging colormap
  Amplitude (XTA/YTA/ZTA) - intensity control
  Occupancy (XTO/YTO/ZTO) - geometry control (per-bin point count)
"""
import os, io, base64, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import SpecConfig, make_channels, max_norm, load_mgesture_csv, segment_instances
from spectra_dataset import fit_ranges, load_mmfi_action

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "docs", "gallery.html")
CFG = SpecConfig(n_bins=100, n_frames=40)          # faithful to original 100x40

ARMS = [("Velocity", ["XTD", "YTD", "ZTD"], "RdBu_r"),
        ("Amplitude", ["XTA", "YTA", "ZTA"], "viridis"),
        ("Occupancy", ["XTO", "YTO", "ZTO"], "magma")]
AXES = ["X", "Y", "Z"]


def panel_png(inst, cfg, title):
    ch = make_channels(inst, cfg)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6))
    for r, (arm, keys, cmap) in enumerate(ARMS):
        for c, (ax_name, key) in enumerate(zip(AXES, keys)):
            ax = axes[r, c]
            img = max_norm(ch[key])
            disp = img if arm == "Velocity" else np.log1p(np.abs(img))
            ax.imshow(disp, aspect="auto", origin="lower", cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{arm[0]}T{ax_name}", fontsize=9)
            if c == 0:
                ax.set_ylabel(arm, fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.text(0.5, 0.005, "rows: accumulation arm | cols: X/Y/Z axis | 100 bins x 40 frames, max-norm",
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=95); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def collect_mhomeges():
    """One representative instance per gesture class (subject 805, 1.2 m)."""
    classes = ["circle", "clap", "down", "knock", "lift", "pull", "push", "up", "yawn", "z"]
    root = os.path.join(DATA, "mhomeges_full", "longGes_1.2m", "805")
    recs, insts = [], []
    for cls in classes:
        f = os.path.join(root, f"point_805_1.2m_{cls}.csv")
        if not os.path.exists(f):
            continue
        segs, _ = segment_instances(load_mgesture_csv(f))
        if segs:
            inst = segs[len(segs) // 2]; insts.append(inst); recs.append((cls, inst))
    ranges = fit_ranges(insts)
    return recs, SpecConfig(100, 40, ranges), "mHomeGes", "TI IWR1443 (77 GHz), short-range arm gestures, 25 subjects; subject 805 @1.2 m shown"


def collect_mmfi():
    """One representative instance per action (subject S01)."""
    base = os.path.join(DATA, "mmfi_extracted", "filtered_mmwave")
    sdir = None
    for e in sorted(glob.glob(os.path.join(base, "E*", "S01"))):
        sdir = e; break
    recs, insts = [], []
    if sdir:
        for adir in sorted(glob.glob(os.path.join(sdir, "A*"))):
            act = os.path.basename(adir)
            df = load_mmfi_action(adir)
            if df is not None and df["frame"].nunique() >= 6:
                insts.append(df); recs.append((act, df))
    ranges = fit_ranges(insts) if insts else CFG.ranges
    return recs, SpecConfig(100, 40, ranges), "MM-Fi", "TI IWR6843 (60 GHz), whole-body activities (HAR), 40 subjects; subject S01 shown (scope-boundary dataset)"


def build():
    sections = []
    for collector in (collect_mhomeges, collect_mmfi):
        recs, cfg, name, desc = collector()
        cards = []
        for cls, inst in recs:
            b64 = panel_png(inst, cfg, f"{name}: {cls}")
            cards.append(f'<figure class="card"><img src="data:image/png;base64,{b64}"/>'
                         f'<figcaption>{cls} &nbsp;({inst["frame"].nunique()} frames, {len(inst)} pts)</figcaption></figure>')
        sections.append((name, desc, len(recs), "\n".join(cards)))
    body = []
    for name, desc, n, cards in sections:
        body.append(f'<section><h2>{name} <span class="badge">{n} classes</span></h2>'
                    f'<p class="desc">{desc}</p><div class="grid">{cards}</div></section>')
    html = f"""<h1>Radar gesture datasets — per-class spectra gallery</h1>
<p class="lead">Preprocessing inspection. Each panel = one representative instance rendered in three accumulation
arms (rows: <b>Velocity</b> XTD/YTD/ZTD, <b>Amplitude</b> XTA/YTA/ZTA, <b>Occupancy</b>=geometry control XTO/YTO/ZTO)
across the three Cartesian axes (cols X/Y/Z). Faithful to the original representation: <b>100 spatial bins x 40 time
frames</b>, per axis, max-normalized. Velocity uses a signed diverging colormap; amplitude/occupancy are log-scaled.</p>
{''.join(body)}
<style>
:root{{color-scheme:light dark}}
body,h1,h2,p,figure{{margin:0}}
.lead,.desc{{max-width:70rem;line-height:1.5;color:#444}}
@media(prefers-color-scheme:dark){{.lead,.desc{{color:#bbb}}}}
h1{{margin:.4rem 0 1rem}} h2{{margin:1.6rem 0 .3rem}}
.badge{{font-size:.7em;background:#2b6cb0;color:#fff;border-radius:1em;padding:.1em .7em;vertical-align:middle}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem;margin-top:.8rem}}
.card{{border:1px solid #ddd;border-radius:8px;padding:.4rem;background:#fafafa}}
@media(prefers-color-scheme:dark){{.card{{background:#1a1a1a;border-color:#333}}}}
.card img{{width:100%;height:auto;display:block}}
figcaption{{font-size:.82rem;text-align:center;color:#555;margin-top:.3rem}}
</style>"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print("wrote", OUT, "size", f"{os.path.getsize(OUT)/1e6:.1f} MB")


if __name__ == "__main__":
    build()
