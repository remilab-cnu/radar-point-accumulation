"""Velocity-spectrogram DESIGN SWEEP: which construction wins gestures ACROSS THE BOARD?

Fixed SmallCNN; only the representation changes. Variants compose per-cell statistics
that address the measured failure modes of plain sum/mean accumulation:
  - sign cancellation (32% of cells)  -> signed split (approach/recede means, peaks)
  - multi-component cells (65%)       -> 4-bin velocity histogram, mean+std
  - spatial marginalization loss      -> add a space-agnostic V-T micro-Doppler channel
  - geometry-strong datasets          -> velocity-primary composite (+occupancy)
  - coordinate frame                  -> spherical (r,az,el) versions of the winners
Datasets: mHomeGes (25 subj, 5-fold) + Infineon (12 users, 4-fold), same folds as gates.
Reference arms (occupancy, intensity-mean) rerun in-config for honest comparison.
"""
import os, io, json, time, zipfile, re, pickle
import numpy as np
import pandas as pd
from spectra_dataset import mhomeges_instances, fit_ranges
from preprocess import SpecConfig
from cnn import train_eval

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DOCS = os.path.join(HERE, "..", "docs")
NB, T = 32, 40
VT_BINS, VT_MAX = 32, 3.0
SEEDS = (0, 1)


def add_sph(df):
    df = df.copy()
    r = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df["r"] = r
    df["az"] = np.arctan2(df.x, df.y)
    df["el"] = np.arcsin(np.clip(df.z / (r + 1e-9), -1, 1))
    return df


def cell_stats(inst, axes, ranges, nb=NB):
    """One pass -> per-axis per-cell stats + global V-T map."""
    f = inst["frame"].values.astype(float)
    f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
    v = inst["doppler"].values.astype(float)
    a = np.abs(inst["intensity"].values.astype(float))
    out = {}
    for ax in axes:
        lo, hi = ranges[ax]
        bi = np.floor((inst[ax].values - lo) / (hi - lo) * nb).astype(int)
        m = (bi >= 0) & (bi < nb)
        b, t, vv, aa = bi[m], ti[m], v[m], a[m]
        S = lambda: np.zeros((nb, T), np.float32)
        sum_v, cnt, sum_a, sum_v2 = S(), S(), S(), S()
        pos_s, pos_c, neg_s, neg_c, pos_mx, neg_mn = S(), S(), S(), S(), S(), S()
        np.add.at(sum_v, (b, t), vv); np.add.at(cnt, (b, t), 1.0)
        np.add.at(sum_a, (b, t), aa); np.add.at(sum_v2, (b, t), vv ** 2)
        p = vv > 0; n = vv < 0
        np.add.at(pos_s, (b[p], t[p]), vv[p]); np.add.at(pos_c, (b[p], t[p]), 1.0)
        np.add.at(neg_s, (b[n], t[n]), vv[n]); np.add.at(neg_c, (b[n], t[n]), 1.0)
        np.maximum.at(pos_mx, (b[p], t[p]), vv[p]); np.maximum.at(neg_mn, (b[n], t[n]), -vv[n])
        hist = np.zeros((4, nb, T), np.float32)
        edges = [-99, -0.6, 0, 0.6, 99]
        for k in range(4):
            sel = (vv > edges[k]) & (vv <= edges[k + 1])
            np.add.at(hist[k], (b[sel], t[sel]), 1.0)
        div = lambda x, c: np.divide(x, c, out=np.zeros_like(x), where=c > 0)
        mean = div(sum_v, cnt)
        out[ax] = {"sum": sum_v, "cnt": cnt, "mean": mean,
                   "int_mean": div(sum_a, cnt),
                   "pos_mean": div(pos_s, pos_c), "neg_mean": div(neg_s, neg_c),
                   "std": np.sqrt(np.clip(div(sum_v2, cnt) - mean ** 2, 0, None)),
                   "pos_max": pos_mx, "neg_max": neg_mn, "hist": hist}
    vt = np.zeros((nb, T), np.float32)                      # V-T micro-Doppler map at nb rows
    vi = np.floor((v + VT_MAX) / (2 * VT_MAX) * (nb - 1e-9)).astype(int)
    mvt = (vi >= 0) & (vi < nb)
    np.add.at(vt, (vi[mvt], ti[mvt]), 1.0)
    out["vt"] = vt
    return out


def norm(c):
    m = np.abs(c).max()
    return (c / m if m > 0 else c).astype(np.float32)


def compose(st, axes, spec):
    """spec: list of stat keys; 'hist' expands to 4; 'vt' is global."""
    ch = []
    for key in spec:
        if key == "vt":
            ch.append(norm(st["vt"])); continue
        for ax in axes:
            if key == "hist":
                ch += [norm(st[ax]["hist"][k]) for k in range(4)]
            else:
                ch.append(norm(st[ax][key]))
    return np.stack(ch)


VARIANTS = {  # name -> (coord, spec, n_bins)
    "v_sum(orig)":        ("cart", ["sum"], 32),
    "v_mean":             ("cart", ["mean"], 32),
    "v_signed":           ("cart", ["pos_mean", "neg_mean"], 32),
    "v_meanstd":          ("cart", ["mean", "std"], 32),
    "v_peak":             ("cart", ["pos_max", "neg_max"], 32),
    "v_hist4":            ("cart", ["hist"], 32),
    "vt_only":            ("cart", ["vt"], 32),
    "v_mean+vt":          ("cart", ["mean", "vt"], 32),
    "v_signed+vt":        ("cart", ["pos_mean", "neg_mean", "vt"], 32),
    "v_signed+occ":       ("cart", ["pos_mean", "neg_mean", "cnt"], 32),
    "v_signed+vt+occ":    ("cart", ["pos_mean", "neg_mean", "vt", "cnt"], 32),
    "v_signed+vt+occ@64": ("cart", ["pos_mean", "neg_mean", "vt", "cnt"], 64),
    "sph_v_signed":       ("sph", ["pos_mean", "neg_mean"], 32),
    "sph_v_signed+vt":    ("sph", ["pos_mean", "neg_mean", "vt"], 32),
    "sph_signed+vt+occ":  ("sph", ["pos_mean", "neg_mean", "vt", "cnt"], 32),
    "sph_signed+vt+occ@64": ("sph", ["pos_mean", "neg_mean", "vt", "cnt"], 64),
    "REF_occupancy":      ("cart", ["cnt"], 32),
    "REF_int_mean":       ("cart", ["int_mean"], 32),
}
CAXES, SAXES = ["x", "y", "z"], ["r", "az", "el"]

# Round 2: top round-1 variants + meanstd-centered combos; 40 epochs, 3 seeds.
# (Round-1 finding: Infineon flips to velocity-centered wins with multi-component
#  preservation; mHomeGes flat at ~67 with high variance -> test underfit + combos.)
VARIANTS2 = {
    "v_sum(orig)":        ("cart", ["sum"], 32),
    "v_meanstd":          ("cart", ["mean", "std"], 32),
    "v_meanstd+vt":       ("cart", ["mean", "std", "vt"], 32),
    "v_meanstd+hist4":    ("cart", ["mean", "std", "hist"], 32),
    "v_meanstd+vt+occ":   ("cart", ["mean", "std", "vt", "cnt"], 32),
    "v_hist4":            ("cart", ["hist"], 32),
    "v_signed+vt+occ":    ("cart", ["pos_mean", "neg_mean", "vt", "cnt"], 32),
    "sph_v_meanstd":      ("sph", ["mean", "std"], 32),
    "REF_occupancy":      ("cart", ["cnt"], 32),
    "REF_int_mean":       ("cart", ["int_mean"], 32),
}


def infineon_recs():
    cache = os.path.join(DATA, "infineon_recs.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import infineon_detection as ifx
    ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
    LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    recs = []
    for m in members:
        user = "u" + re.search(r"user(\d+)", m).group(1)
        with zf.open(m) as f:
            d = np.load(io.BytesIO(f.read())); inputs, targets = d["inputs"], d["targets"]
        by = {}
        for r in np.random.RandomState(0).permutation(len(inputs)):
            g = np.where(targets[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
            if cls not in LM or by.get(cls, 0) >= 40:
                continue
            df = ifx.process_recording(inputs[r, max(0, g.min() - 6):g.max() + 7])
            if len(df) < 8:
                continue
            recs.append((df, LM[cls], user)); by[cls] = by.get(cls, 0) + 1
    pickle.dump(recs, open(cache, "wb"))
    return recs


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


def run_dataset(name, insts, folds, ncls, epochs):
    print(f"\n######## {name}: {len(insts)} inst ########", flush=True)
    t0 = time.time()
    cart = [t[0] for t in insts]
    sph = [add_sph(d) for d in cart]
    r_cart, r_sph = fit_ranges(cart), None
    # spherical ranges via same percentile logic
    r_sph = {}
    for axk in SAXES:
        vals = np.concatenate([d[axk].values for d in sph])
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99); pad = 0.05 * (hi - lo + 1e-9)
        r_sph[axk] = (float(lo - pad), float(hi + pad))
    y = np.array([t[1] for t in insts]); subj = np.array([t[2] for t in insts])
    res = {}
    # group by (coord, nb): build stats once, run its variants, then free (bounds RAM)
    needed = sorted({(c, nb) for c, _, nb in VARIANTS.values()})
    for coord, nb in needed:
        data, axes, rng = (cart, CAXES, r_cart) if coord == "cart" else (sph, SAXES, r_sph)
        stats = [cell_stats(d, axes, rng, nb=nb) for d in data]
        print(f"  stats {coord}@{nb} built ({time.time()-t0:.0f}s)", flush=True)
        for vname, (c2, spec, nb2) in VARIANTS.items():
            if (c2, nb2) != (coord, nb):
                continue
            X = np.stack([compose(st, axes, spec) for st in stats])
            accs = []
            for te_s in folds:
                te = np.isin(subj, list(te_s)); tr = ~te
                accs += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs, seed=s) for s in SEEDS]
            res[vname] = (float(np.mean(accs)) * 100, float(np.std(accs)) * 100)
            print(f"  {vname:22s} ({X.shape[1]:2d}ch@{nb}): {res[vname][0]:6.2f}% (+-{res[vname][1]:.1f})", flush=True)
            del X
        del stats
    return res


if __name__ == "__main__":                      # import-safe (audit hygiene)
    round2 = os.environ.get("REP_ROUND") == "2"
    if round2:                                   # module-scope rebinds are intentional
        VARIANTS = VARIANTS2
        SEEDS = (0, 1, 2)
        ep_mh = ep_inf = 40
        tag = "_round2"
    else:
        ep_mh, ep_inf, tag = 20, 30, ""
    out = {}
    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    out["mHomeGes"] = run_dataset("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10, epochs=ep_mh)
    inf = infineon_recs()
    out["Infineon"] = run_dataset("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5, epochs=ep_inf)

    json.dump(out, open(os.path.join(DOCS, f"repsweep_results{tag}.json"), "w"), indent=1)
    print(f"\nwrote docs/repsweep_results{tag}.json", flush=True)
    print("targets: beat point-input (job1 pointset_results.json) AND in-config REFs", flush=True)
