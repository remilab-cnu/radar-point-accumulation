"""ORACLE-HEADROOM KILL-GATE (cold-review-prescribed decision experiment, 2026-07-13).

Question: does the CFAR detection step DISCARD gesture-class-relevant information that a
task-relevance selector recovers AT EQUAL POINT BUDGET? If not, the "detection is
recognition-suboptimal" premise is empirically empty and the info-theoretic-detection
line is closed with a number.

Design (Infineon raw BGT60 + our detection chain infineon_detection):
  For each recording-window (frozen infineon_recs set: users u1..u12, 5 classes, cap 40/cls),
  per frame build the RD magnitude map once, then select detections under 3 policies at a
  MATCHED per-frame budget n = the CFAR count for that frame:
    - CFAR    : the CA-CFAR detections (the standard pipeline).
    - ENERGY  : top-n candidate cells by magnitude (a trivial alternative selection).
    - ORACLE  : top-n candidate cells by class-relevance, where relevance is a per-(Doppler,
                range)-bin F-statistic of cell magnitude vs class. Computed on the WHOLE set
                => an OPTIMISTIC UPPER BOUND on any positional selector (mild leakage, on
                purpose: if even a leaky oracle barely beats CFAR, the headroom is truly
                empty). Candidate pool = cells in the range gate above 2x local noise.
  Each policy -> canonical point cloud (x,y,z,doppler,intensity via the same monopulse code)
  -> SmallCNN velocity-sum map -> subject-disjoint 4-fold recognition.

READOUT: acc(ORACLE) - acc(CFAR) at equal budget = the class-MI headroom CFAR leaves.
  gap < ~2 pp  -> premise empirically empty -> CLOSE the line.
  gap > 3-5 pp (reproduced) -> real headroom -> a scoped diagnostic paper may be justified.
Also report ENERGY (does even trivial re-selection beat CFAR's criterion?).
Env SMOKE=1: 3 users, 2 classes.
"""
import os, io, re, zipfile, json
import numpy as np
import pandas as pd
import infineon_detection as ifx
from infineon_detection import _rd_cube, _ca_ring_noise, RANGE_MIN_BIN, RANGE_MAX_BIN, \
    RANGE_RES, V_MAX, N_CHIRP, D_OVER_LAMBDA, AZ_PAIR, EL_PAIR, AZ_SIGN, EL_SIGN
from rep_variants import cell_stats, compose, CAXES, kfold
from spectra_dataset import fit_ranges
from cnn import train_eval_full
from pointset_models import build_point_tensors, DeepSets, train_eval_set_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
SMOKE = os.environ.get("SMOKE") == "1"
LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
DOPP_IDX = np.arange(N_CHIRP) - N_CHIRP // 2
NB = 32; SEEDS = (0,) if SMOKE else (0, 1, 2); EP = 4 if SMOKE else 40; WIDTH = 16


def _xyz_row(RD, d, r, mag):
    rng = r * RANGE_RES
    v = DOPP_IDX[d] * (2 * V_MAX / N_CHIRP)
    c = RD[:, d, r]
    daz = np.angle(c[AZ_PAIR[0]] * np.conj(c[AZ_PAIR[1]]))
    dele = np.angle(c[EL_PAIR[0]] * np.conj(c[EL_PAIR[1]]))
    az = np.arcsin(np.clip(AZ_SIGN * daz / (2 * np.pi * D_OVER_LAMBDA), -1, 1))
    el = np.arcsin(np.clip(EL_SIGN * dele / (2 * np.pi * D_OVER_LAMBDA), -1, 1))
    return (rng * np.cos(el) * np.sin(az), rng * np.cos(el) * np.cos(az),
            rng * np.sin(el), v, float(mag[d, r]))


def frame_maps(cube):
    """Per frame: (mag, noise, cfar_mask, candidate_mask, RD). Reused across policies."""
    out = []
    for f in range(cube.shape[0]):
        RD = _rd_cube(cube[f]); mag = np.abs(RD).sum(0)
        noise = _ca_ring_noise(mag, 9, 3)
        gate = np.zeros_like(mag, dtype=bool); gate[:, RANGE_MIN_BIN:RANGE_MAX_BIN] = True
        cfar = (mag > ifx.CFAR_ALPHA * noise) & gate
        cand = (mag > 2.0 * noise) & gate
        out.append((mag, cfar, cand, RD))
    return out


def build_clouds(cube, relev):
    """Return dict policy-> DataFrame(frame,x,y,z,doppler,intensity) at matched per-frame budget."""
    rows = {"cfar": [], "energy": [], "oracle": []}
    for f, (mag, cfar, cand, RD) in enumerate(frame_maps(cube)):
        n = int(cfar.sum())
        if n == 0:
            continue
        # CFAR: its own detections
        for d, r in zip(*np.where(cfar)):
            rows["cfar"].append((f,) + _xyz_row(RD, d, r, mag))
        cd, cr = np.where(cand)
        if len(cd) == 0:
            continue
        cmag = mag[cd, cr]
        # ENERGY: top-n candidates by magnitude
        e_ord = np.argsort(cmag)[::-1][:n]
        for i in e_ord:
            rows["energy"].append((f,) + _xyz_row(RD, cd[i], cr[i], mag))
        # ORACLE: top-n candidates by class-relevance (x magnitude tiebreak)
        rscore = relev[cd, cr] * cmag
        o_ord = np.argsort(rscore)[::-1][:n]
        for i in o_ord:
            rows["oracle"].append((f,) + _xyz_row(RD, cd[i], cr[i], mag))
    cols = ["frame", "x", "y", "z", "doppler", "intensity"]
    return {k: pd.DataFrame(v, columns=cols) for k, v in rows.items()}


def load_recordings():
    """Frozen infineon_recs iteration -> list of (cube_window, label, user)."""
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    recs = []
    users_keep = None
    if SMOKE:
        users_keep = set(members[:3])
    for m in members:
        if users_keep is not None and m not in users_keep:
            continue
        user = "u" + re.search(r"user(\d+)", m).group(1)
        with zf.open(m) as fh:
            d = np.load(io.BytesIO(fh.read())); inputs, targets = d["inputs"], d["targets"]
        by = {}; cap = 8 if SMOKE else 40
        for r in np.random.RandomState(0).permutation(len(inputs)):
            g = np.where(targets[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
            if cls not in LM or by.get(cls, 0) >= cap:
                continue
            if SMOKE and LM[cls] >= 2:
                continue
            recs.append((inputs[r, max(0, g.min() - 6):g.max() + 7], LM[cls], user))
            by[cls] = by.get(cls, 0) + 1
    return recs


def relevance_map(recs):
    """Per-(Doppler,range)-bin F-stat of summed magnitude vs class (whole-set = optimistic)."""
    agg, ys = [], []
    for cube, lab, _ in recs:
        m = np.zeros((N_CHIRP, 33))
        for f in range(cube.shape[0]):
            m += np.abs(_rd_cube(cube[f])).sum(0)
        agg.append(m.ravel()); ys.append(lab)
    A = np.array(agg); y = np.array(ys)
    from scipy.stats import f_oneway
    F = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        groups = [A[y == c, j] for c in np.unique(y)]
        try:
            F[j] = f_oneway(*groups).statistic
        except Exception:
            F[j] = 0.0
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    return F.reshape(N_CHIRP, 33)


def recognize(insts_by_policy, ncls):
    """Subject-disjoint 4-fold accuracy per policy under TWO recognizers:
    'map' = SmallCNN velocity-sum map (lossy rasterization, may under-detect which
    points were selected); 'pointnet' = DeepSets on the raw selected points (directly
    sensitive to point selection => the honest test of the headroom)."""
    out = {}
    for pol, insts in insts_by_policy.items():
        insts = [t for t in insts if len(t[0]) >= 4]
        y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
        ranges = fit_ranges([t[0] for t in insts])
        Xmap = np.stack([compose(cell_stats(t[0], CAXES, ranges, nb=NB), CAXES, ["sum"])
                         for t in insts]).astype(np.float32)
        Xp, Mp, yp, sp = build_point_tensors(insts, ranges)     # (N,384,6)=[x,y,z,v,A,t]
        sp = np.array([str(s) for s in sp])
        folds = kfold(subj, 2 if SMOKE else 4)
        macc, pacc = [], []
        for te_s in folds:
            te = np.isin(subj, [str(s) for s in te_s]); tr = ~te
            if tr.sum() == 0 or te.sum() == 0:
                continue
            for s in SEEDS:
                a, _, _, _ = train_eval_full(Xmap[tr], y[tr], Xmap[te], y[te], ncls,
                                             epochs=EP, seed=s, width=WIDTH)
                macc.append(a)
                b, _, _, _ = train_eval_set_full(DeepSets, Xp[tr], Mp[tr], y[tr],
                                                 Xp[te], Mp[te], y[te], ncls, 6,
                                                 epochs=EP, seed=s)
                pacc.append(b)
        out[pol] = {"map": (round(float(np.mean(macc)) * 100, 2), round(float(np.std(macc)) * 100, 2)),
                    "pointnet": (round(float(np.mean(pacc)) * 100, 2), round(float(np.std(pacc)) * 100, 2)),
                    "pts": round(float(np.mean([len(t[0]) for t in insts])), 1)}
    return out


if __name__ == "__main__":
    print(f"ORACLE-HEADROOM KILL-GATE  SMOKE={SMOKE}", flush=True)
    recs = load_recordings()
    print(f"recordings: {len(recs)}, users {sorted(set(t[2] for t in recs))}", flush=True)
    relev = relevance_map(recs)
    print(f"relevance map built (F-stat), max={relev.max():.1f}", flush=True)
    by_pol = {"cfar": [], "energy": [], "oracle": []}
    for cube, lab, user in recs:
        clouds = build_clouds(cube, relev)
        for pol, df in clouds.items():
            if len(df):
                by_pol[pol].append((df, lab, user))
    for pol in by_pol:
        pf = np.mean([len(t[0]) for t in by_pol[pol]]) if by_pol[pol] else 0
        print(f"  {pol}: {len(by_pol[pol])} inst, {pf:.0f} pts/inst", flush=True)
    res = recognize(by_pol, 2 if SMOKE else 5)
    headroom = {}
    for rec in ("map", "pointnet"):
        print(f"\n=== MATCHED-BUDGET RECOGNITION [{rec}] (subj-disjoint) ===", flush=True)
        for pol in ("cfar", "energy", "oracle"):
            a, s = res[pol][rec]; print(f"  {pol:7s}: {a:6.2f}% (+-{s:.1f})  [{res[pol]['pts']} pts/inst]", flush=True)
        go = res["oracle"][rec][0] - res["cfar"][rec][0]; ge = res["energy"][rec][0] - res["cfar"][rec][0]
        headroom[rec] = {"oracle_minus_cfar": round(go, 2), "energy_minus_cfar": round(ge, 2)}
        print(f"  HEADROOM[{rec}] oracle-cfar = {go:+.2f}pp ; energy-cfar = {ge:+.2f}pp", flush=True)
    print("\nKILL RULE: point-net |oracle-cfar| < 2pp => detection-misalignment premise EMPTY => close;"
          " >3-5pp reproduced => real headroom.", flush=True)
    out = {"n_recordings": len(recs), "epochs": EP, "seeds": list(SEEDS), "smoke": SMOKE,
           "results": res, "headroom": headroom,
           "note": "oracle uses whole-set relevance = optimistic upper bound (leaky on purpose); "
                   "pointnet is the honest recognizer (map may under-detect point selection)"}
    json.dump(out, open(os.path.join(DOCS, f"kill_gate_headroom2{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"wrote docs/kill_gate_headroom2{'_smoke' if SMOKE else ''}.json", flush=True)
