"""EPENTHESIS CHARACTERIZATION (autonomous exploration, 2026-07-14).

Establishes, rigorously, the phenomenon behind the continuous-spotting failure:
gestures are embedded in a preparation -> gesture -> retraction Doppler envelope where the
non-gesture motion is (i) LARGE (often larger |Doppler| than the gesture), (ii) CLASS-
INDISCRIMINATE, and (iii) frequently ANTI-DIRECTIONAL. This is movement epenthesis, and it
is why energy-gating + velocity representations misfire on a continuous stream.

Infineon BGT60 full recordings + per-frame targets (0=idle, id=active gesture). Phases per
instance: prep=[g0-P,g0), gesture=[g0,g1], retr=(g1,g1+P]. P=8 frames.

Metrics (subject-disjoint where trained):
  M1 phase Doppler structure: per-class mean & |mean| Doppler in prep/gesture/retr (all users).
  M2 CLASS-INDISCRIMINABILITY (the key result): classify the 5 gestures from PREPARATION-only
     windows vs from GESTURE-only windows (same SmallCNN map, subj-disjoint 4-fold). If
     prep-acc ~ chance (20%) << gesture-acc, the discriminative signal is in the gesture, and
     the dominant-Doppler preparation is a pure confounder.
  M3 ENERGY-GATE FALSE-TRIGGER: an onset gate calibrated to fire on gesture frames (thr =
     median gesture-frame motion-energy); report the fraction of prep / retr / far-idle frames
     that also exceed it => how badly a standard energy gate cannot isolate the gesture.
  M4 ANTI-DIRECTIONAL fraction: per instance, sign(mean prep Doppler) != sign(mean gesture Doppler).
Env SMOKE=1: 3 users.
"""
import os, io, re, zipfile, json
import numpy as np
import pandas as pd
import infineon_detection as ifx
from rep_variants import cell_stats, compose, CAXES, kfold
from spectra_dataset import fit_ranges
from cnn import train_eval_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
SMOKE = os.environ.get("SMOKE") == "1"
LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
P = 8; NB = 32; SEEDS = (0,) if SMOKE else (0, 1, 2); EP = 4 if SMOKE else 40; WIDTH = 16


def load():
    zf = zipfile.ZipFile(ZIP)
    mem = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                  and not re.search(r"_(fast|slow|wrist)", m)],
                 key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    if SMOKE:
        mem = mem[:3]
    recs = []
    for m in mem:
        u = "u" + re.search(r"user(\d+)", m).group(1)
        d = np.load(io.BytesIO(zf.open(m).read())); inp, tg = d["inputs"], d["targets"]
        by = {}; cap = 6 if SMOKE else 40
        for r in np.random.RandomState(0).permutation(len(inp)):
            g = np.where(tg[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(tg[r][tg[r] > 0]).argmax())
            if cls not in LM or by.get(cls, 0) >= cap:
                continue
            df = ifx.process_recording(inp[r])
            if len(df):
                recs.append((df, int(g.min()), int(g.max()), LM[cls], u))
                by[cls] = by.get(cls, 0) + 1
    return recs


def phase_df(df, lo, hi):
    w = df[(df.frame >= lo) & (df.frame < hi)]
    return w.reset_index(drop=True) if len(w) else None


def energy_per_frame(df):
    """motion energy per frame = #points * mean|Doppler| (proxy for an onset gate signal)."""
    if not len(df):
        return {}
    g = df.groupby("frame")
    return (g.size() * g["doppler"].apply(lambda v: np.abs(v).mean())).to_dict()


if __name__ == "__main__":
    print(f"EPENTHESIS CHARACTERIZATION  SMOKE={SMOKE}  P={P}", flush=True)
    recs = load()
    print(f"instances: {len(recs)}, users {sorted(set(t[4] for t in recs))}", flush=True)
    ranges = fit_ranges([t[0] for t in recs])

    # ---- M1 phase Doppler structure + M4 anti-directional ----
    from collections import defaultdict
    ph = defaultdict(lambda: {"prep": [], "ges": [], "retr": []})
    anti = []
    for df, g0, g1, c, u in recs:
        pr = phase_df(df, max(0, g0 - P), g0); ge = phase_df(df, g0, g1 + 1); rt = phase_df(df, g1 + 1, g1 + 1 + P)
        mp = pr.doppler.mean() if pr is not None else np.nan
        mg = ge.doppler.mean() if ge is not None else np.nan
        mr = rt.doppler.mean() if rt is not None else np.nan
        ph[c]["prep"].append(mp); ph[c]["ges"].append(mg); ph[c]["retr"].append(mr)
        if not (np.isnan(mp) or np.isnan(mg)) and abs(mg) > 0.03:
            anti.append(mp * mg < 0)
    print("\n[M1] phase mean Doppler (all instances) + |Doppler| ratio prep/ges:", flush=True)
    for c in sorted(ph):
        pp = np.nanmean(ph[c]["prep"]); gg = np.nanmean(ph[c]["ges"]); rr = np.nanmean(ph[c]["retr"])
        ratio = abs(pp) / max(abs(gg), 1e-6)
        print(f"  class{c}: prep={pp:+.3f} ges={gg:+.3f} retr={rr:+.3f}  |prep|/|ges|={ratio:.2f}", flush=True)
    anti_frac = float(np.mean(anti)) * 100 if anti else 0.0
    print(f"[M4] anti-directional (sign prep != sign ges): {anti_frac:.1f}% of instances", flush=True)

    # ---- M2 class-indiscriminability: prep-only vs gesture-only classification ----
    def build(kind):
        X, y, s = [], [], []
        for df, g0, g1, c, u in recs:
            w = phase_df(df, max(0, g0 - P), g0) if kind == "prep" else phase_df(df, g0, g1 + 1)
            if w is None or len(w) < 3:
                continue
            X.append(compose(cell_stats(w, CAXES, ranges, nb=NB), CAXES, ["sum"]).astype(np.float32))
            y.append(c); s.append(u)
        return np.stack(X), np.array(y), np.array(s)

    m2 = {}
    for kind in ("prep", "ges"):
        X, y, s = build(kind)
        folds = kfold(s, 2 if SMOKE else 4)
        accs = []
        for te in folds:
            m = np.isin(s, [str(x) for x in te]); tr = ~m
            if tr.sum() == 0 or m.sum() == 0:
                continue
            for sd in SEEDS:
                a, _, _, _ = train_eval_full(X[tr], y[tr], X[m], y[m], 5, epochs=EP, seed=sd, width=WIDTH)
                accs.append(a)
        m2[kind] = round(float(np.mean(accs)) * 100, 2)
        print(f"[M2] {kind}-only 5-way classification acc: {m2[kind]:.2f}% (chance 20)", flush=True)

    # ---- M3 energy-gate false-trigger ----
    thr_pool = []
    for df, g0, g1, c, u in recs:
        e = energy_per_frame(df)
        thr_pool += [e[f] for f in range(g0, g1 + 1) if f in e]
    thr = float(np.median(thr_pool)) if thr_pool else 0.0
    fire = {"prep": [], "retr": [], "far_idle": []}
    for df, g0, g1, c, u in recs:
        e = energy_per_frame(df); F = int(df.frame.max()) + 1
        for f in range(max(0, g0 - P), g0):
            if f in e: fire["prep"].append(e[f] > thr)
        for f in range(g1 + 1, min(F, g1 + 1 + P)):
            if f in e: fire["retr"].append(e[f] > thr)
        for f in range(0, F):
            if f < g0 - P or f > g1 + P:
                if f in e: fire["far_idle"].append(e[f] > thr)
    print(f"[M3] energy-gate (thr=median gesture energy) false-trigger fraction:", flush=True)
    m3 = {}
    for k in ("prep", "retr", "far_idle"):
        m3[k] = round(float(np.mean(fire[k])) * 100, 1) if fire[k] else 0.0
        print(f"     {k:9s}: {m3[k]:.1f}% of frames exceed gate", flush=True)

    out = {"n_instances": len(recs), "P": P, "epochs": EP, "seeds": list(SEEDS), "smoke": SMOKE,
           "M1_phase_doppler": {int(c): {"prep": round(float(np.nanmean(ph[c]["prep"])), 3),
                                         "ges": round(float(np.nanmean(ph[c]["ges"])), 3),
                                         "retr": round(float(np.nanmean(ph[c]["retr"])), 3)} for c in ph},
           "M2_prep_vs_ges_acc": m2, "M3_energy_gate_falsetrigger": m3, "M4_anti_directional_pct": round(anti_frac, 1)}
    json.dump(out, open(os.path.join(DOCS, f"epenthesis_char{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"\nwrote docs/epenthesis_char{'_smoke' if SMOKE else ''}.json", flush=True)
    print("READ: M2 prep~chance<<ges => preparation is class-indiscriminate confounder; "
          "M3 high prep/retr => energy-gate cannot isolate gesture; M4 => anti-directional share.", flush=True)
