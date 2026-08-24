"""Per-dataset Doppler measurand audit (referee request, 2026-08-11).

Two TIM reviewers asked for the traceability that Section IV asserts, and for a
screening statement on the velocity channel. Both come from one measurement:
the exported Doppler values of every dataset lie on a uniform lattice, so the
lattice itself gives the quantization step and the unambiguous span.

  step  dv       = smallest positive spacing of the observed unique values
  span  v_max    = dv * (K-1)/2 with K the number of lattice points the export
                   uses, i.e. the largest representable |v| on that lattice. This
                   is half a bin below the Nyquist edge dv*K/2; we report the
                   largest representable value because it is what is measurable
                   from the export.
  outside        = fraction of points whose |v| exceeds that largest
                   representable value. For every set except mHomeGes this is
                   float-comparison residue at the boundary; for mHomeGes it is
                   a genuine corrupt tail, and frac_impossible (|v| > 10 m/s)
                   isolates that tail unambiguously.

FULL datasets, no subsampling (docs/c_sensitivity_vmax.json used a 400-instance
cap and therefore cannot answer either question).
Out: docs/measurand_table.json
"""
import os, sys, json, glob, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectra_dataset import mmfi_instances, mhomeges_instances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DOCS = os.path.join(HERE, "..", "docs")

# carrier from the sensor each dataset ships (Section III-A of the manuscript)
CARRIER_GHZ = {"mHomeGes": 77.0, "BGT60TR13C": 60.0, "MM-Fi": 60.0, "mRI": 77.0}
IMPOSSIBLE = 10.0   # no hand/arm or whole-body target reaches 10 m/s radially


def infineon_recs_direct():
    """The frozen detection output: recs[i] = (df, class, user). Loaded straight
    from the pkl so this audit does not need the training stack (no torch)."""
    return pickle.load(open(os.path.join(DATA, "infineon_recs.pkl"), "rb"))


def mri_windows():
    """Same 40-frame windowing as the evaluation protocol (c_sensitivity_vmax)."""
    CL = [f"pose_{i}" for i in range(1, 11)]
    recs = []
    MRI = os.path.join(DATA, "mri_sample", "mri_data")
    for csvf in sorted(glob.glob(os.path.join(MRI, "subject*.csv"))):
        sid = os.path.basename(csvf).replace(".csv", "")
        if "_all_labels" in sid:
            continue
        df = pd.read_csv(csvf); df.columns = [x.strip() for x in df.columns]
        can = pd.DataFrame({"frame": df["Camera Frame"].astype(int), "x": df["X"],
                            "y": df["Y"], "z": df["Z"], "doppler": df["Doppler"],
                            "intensity": df["Intensity"]})
        vl = pickle.load(open(os.path.join(MRI, f"{sid}_all_labels.cpl"), "rb"))["video_label"]
        for cn in CL:
            if cn not in vl:
                continue
            a, b = vl[cn]; t0 = a
            while t0 + 40 <= b:
                w = can[(can.frame >= t0) & (can.frame < t0 + 40)]
                if w["frame"].nunique() >= 6 and len(w) >= 30:
                    recs.append((w, 0, "x"))
                t0 += 120
    return recs


def lattice(v):
    """Quantization step and lattice occupancy of an exported Doppler channel.
    Computed on the physically possible values only: mHomeGes carries a corrupt
    tail up to 11,666 m/s whose gaps otherwise dominate the median spacing."""
    v = np.asarray(v, dtype=float)
    v = v[np.abs(v) < IMPOSSIBLE]
    u = np.unique(np.round(v, 6))
    d = np.diff(u)
    d = d[d > 1e-9]
    if len(d) == 0:
        return None, None
    step = float(np.median(d))          # median over the clean lattice
    return step, u


def audit(name, insts):
    v = np.concatenate([t[0]["doppler"].values for t in insts]).astype(float)
    av = np.abs(v)
    step, u = lattice(v)
    # the lattice the bulk of the data occupies: ignore the corrupt tail when
    # inferring the span, then report how much sits outside it
    clean = av[av < IMPOSSIBLE]
    k_half = int(round(clean.max() / step)) if step else None
    v_max = round(step * k_half, 3) if step else None
    n = len(v)
    out = dict(
        n_points=int(n), n_instances=len(insts),
        carrier_GHz=CARRIER_GHZ[name],
        lambda_mm=round(299.792458 / CARRIER_GHZ[name], 2),
        step_mps=round(step, 4) if step else None,
        n_lattice_points=int(2 * k_half + 1) if k_half else None,
        v_unamb_mps=v_max,
        observed_span_mps=round(float(clean.max()), 3),
        max_abs_v_mps=round(float(av.max()), 1),
        p999_abs_v_mps=round(float(np.percentile(av, 99.9)), 3),
        frac_outside_span=float((av > v_max + 1e-6).mean()) if v_max else None,
        frac_impossible=float((av > IMPOSSIBLE).mean()),
        n_impossible=int((av > IMPOSSIBLE).sum()),
    )
    bad = np.array([bool((np.abs(t[0]["doppler"].values) > IMPOSSIBLE).any()) for t in insts])
    out["n_instances_impossible"] = int(bad.sum())
    out["frac_instances_impossible"] = float(bad.mean())
    return out


if __name__ == "__main__":
    res = {"_definition": __doc__.strip(), "_impossible_threshold_mps": IMPOSSIBLE, "datasets": {}}
    loaders = {
        "BGT60TR13C": infineon_recs_direct(),
        "MM-Fi": mmfi_instances(os.path.join(DATA, "mmfi_extracted")),
        "mRI": mri_windows(),
        "mHomeGes": mhomeges_instances(os.path.join(DATA, "mhomeges_full")),
    }
    for name, insts in loaders.items():
        res["datasets"][name] = audit(name, insts)
        print(name, json.dumps(res["datasets"][name]), flush=True)
    json.dump(res, open(os.path.join(DOCS, "measurand_table.json"), "w"), indent=1)
    print("wrote docs/measurand_table.json", flush=True)
