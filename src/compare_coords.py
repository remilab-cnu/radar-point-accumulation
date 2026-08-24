"""Is the velocity-vs-geometry effect an artifact of the spherical->Cartesian
conversion? Rebuild the SAME points+velocities binned in native spherical (r, az, el)
vs Cartesian (x, y, z) and compare the velocity-minus-geometry gap cross-subject on
both datasets. If the pattern holds in BOTH coordinate systems it is not a Cartesian
artifact; if it changes, the conversion matters.

Note: radial velocity is naturally aligned with r. Binning it along r/az/el keeps it in
the sensor's native frame; binning along x/y/z is what the current method does.
"""
import os, numpy as np
from preprocess import build_spectrum, max_norm, SpecConfig
from spectra_dataset import mhomeges_instances, mmfi_instances
from cnn import train_eval

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
CART = ["x", "y", "z"]; SPH = ["r", "az", "el"]


def add_sph(df):
    df = df.copy()
    r = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df["r"] = r
    df["az"] = np.arctan2(df.x, df.y)                       # azimuth from forward (y)
    df["el"] = np.arcsin(np.clip(df.z / (r + 1e-9), -1, 1)) # elevation
    return df


def fit_ax(insts, axes):
    rng = {}
    for a in axes:
        v = np.concatenate([i[a].values for i in insts])
        lo, hi = np.percentile(v, 1), np.percentile(v, 99); pad = 0.05 * (hi - lo + 1e-9)
        rng[a] = (float(lo - pad), float(hi + pad))
    return rng


def build(insts_labeled, axes, nbins=32, nframes=40):
    insts = [add_sph(t[0]) for t in insts_labeled]
    cfg = SpecConfig(nbins, nframes, fit_ax(insts, axes))
    Xv, Xo, y, subj = [], [], [], []
    for inst, (_, lab, s) in zip(insts, insts_labeled):
        Xv.append(np.stack([max_norm(build_spectrum(inst, a, "doppler", cfg)) for a in axes]))
        Xo.append(np.stack([max_norm(build_spectrum(inst, a, "count", cfg)) for a in axes]))
        y.append(lab); subj.append(s)
    return (np.array(Xv, np.float32), np.array(Xo, np.float32), np.array(y), np.array(subj))


def cross_subject(Xv, Xo, y, subj, folds, ncls, epochs, seeds=(0, 1)):
    def ev(X):
        accs = []
        for te_s in folds:
            te = np.isin(subj, list(te_s)); tr = ~te
            accs += [train_eval(X[tr], y[tr], X[te], y[te], ncls, epochs=epochs, seed=s) for s in seeds]
        return float(np.mean(accs))
    v, o = ev(Xv), ev(Xo)
    return v, o


def kfold(subj, k, seed=0):
    s = sorted(set(subj.tolist())); rng = np.random.RandomState(seed)
    return [list(g) for g in np.array_split(rng.permutation(s), k)]


print("collecting mHomeGes ...", flush=True)
mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
print(f"  {len(mh)} instances", flush=True)
print("collecting MM-Fi ...", flush=True)
mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
print(f"  {len(mf)} instances", flush=True)
S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]

print(f"\n{'dataset':10s} {'coord':10s} {'velocity':>9s} {'geometry':>9s} {'vel-geom':>9s}")
for dname, insts, folds, ncls, ep in [
        ("mHomeGes", mh, None, 10, 15),
        ("MM-Fi", mf, [S2], 27, 30)]:
    for axes, cname in ((CART, "cartesian"), (SPH, "spherical")):
        Xv, Xo, y, subj = build(insts, axes)
        f = folds if folds is not None else kfold(subj, 3)
        v, o = cross_subject(Xv, Xo, y, subj, f, ncls, ep)
        print(f"{dname:10s} {cname:10s} {v*100:9.2f} {o*100:9.2f} {(v-o)*100:+9.2f}", flush=True)
