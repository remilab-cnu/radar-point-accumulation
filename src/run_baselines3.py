"""Baselines round 3 — CPDP/mGesNet, the mHomeGes dataset's NATIVE published method
(Liu et al., IMWUT 2020, DOI 10.1145/3432235), under the frozen equal-HPO protocol
(docs/EQUAL_HPO_PROTOCOL.md; repositioning memo queue #4).

Arms: MGesNetCPDP x CPDP input (32x32 range x Doppler intensity maps, 30-frame
windows, masked score-mean voting) x lr{3e-4,1e-3} x ep{30,40} x seeds{0,1,2},
batch 64, Adam, grad-clip 5.0, no augmentation. BOTH LRs reported (round-1 LR
sensitivity is live). Datasets: mHomeGes 5-fold (native home turf), Infineon frozen
pkl 4-fold, MM-Fi documented S2 split; folds asserted equal to baselines2.json.
Outputs: docs/baselines3.json (+ per-instance preds docs/baselines3_preds.npz).

Input note: the published form is a window-level (range x Doppler) intensity map
classified by a shallow CNN — the whole-instance 384-point tuples and the 32-bin/
40-frame space-time grids are both structurally inapplicable, so only the
published-form input runs (protocol 'published form dictates otherwise' clause).
The paper's training hyperparameters are not public (paywalled): the protocol pair
{3e-4, 1e-3} is the only LR arm set; no published-default arm exists to run.
CPDP construction provenance + every non-public guess: baselines_cpdp.DEVIATIONS.
"""
import os, json, time, hashlib
import numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances
from rep_variants import infineon_recs
from rep_round3 import kfold
from baselines_pointnets import train_eval_set_preds_tr
from baselines_cpdp import DEVIATIONS, MGesNetCPDP, build_cpdp_tensors, fit_cpdp_ranges

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SEEDS = (0, 1, 2); EPOCH_BUDGETS = (30, 40); LRS = (("3e-4", 3e-4), ("1e-3", 1e-3))
NR, ND, WIN, STRIDE, MAX_W = 32, 32, 30, 15, 16

results, preds, datasets = {}, {}, {}


def run_set(tag, insts, folds, ncls, frozen_folds=None):
    if frozen_folds is not None:                 # folds must equal the frozen protocol's
        assert [sorted(map(str, f)) for f in folds] == frozen_folds, \
            f"fold mismatch vs baselines2 ({tag})"
    ranges = fit_cpdp_ranges([t[0] for t in insts])
    X, M, y, s = build_cpdp_tensors(insts, ranges, NR, ND, WIN, STRIDE, MAX_W)
    print(f"  {tag} CPDP tensors {X.shape}, window-mask fill {M.mean():.3f}", flush=True)
    datasets[tag] = {"n_instances": len(insts), "n_cls": ncls, "chance": round(1 / ncls, 4),
                     "folds": [sorted(map(str, f)) for f in folds],
                     "test_N": [int(np.isin(s, list(f)).sum()) for f in folds],
                     "cpdp_ranges": ranges, "n_windows": int(X.shape[1]),
                     "window_mask_fill": round(float(M.mean()), 4)}
    for lrname, lr in LRS:
        for ep in EPOCH_BUDGETS:
            accs, tr_accs = [], []
            for fi, te_s in enumerate(folds):
                te = np.isin(s, list(te_s)); tr = ~te; te_idx = np.where(te)[0]
                for sd in SEEDS:
                    a, yt, yp, ta = train_eval_set_preds_tr(
                        MGesNetCPDP, X[tr], M[tr], y[tr], X[te], M[te], y[te], ncls, 1,
                        epochs=ep, lr=lr, seed=sd)
                    accs.append(a); tr_accs.append(ta)
                    preds[f"{tag}|CPDP|lr{lrname}|ep{ep}|f{fi}|s{sd}"] = np.stack([te_idx, yt, yp])
            key = f"{tag}|CPDP-mGesNet|cpdp(r,d;int-sum)|lr{lrname}|ep{ep}"
            results[key] = {"mean": float(np.mean(accs)) * 100, "std": float(np.std(accs)) * 100,
                            "accs": [round(float(a) * 100, 2) for a in accs],
                            "min_train_acc": round(float(min(tr_accs)) * 100, 2),
                            "underfit": bool(min(tr_accs) < 0.95)}
            print(f"  {key:52s}: {results[key]['mean']:6.2f}% (+-{results[key]['std']:.1f})"
                  f"{'  UNDERFIT' if results[key]['underfit'] else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    b2p = os.path.join(DOCS, "baselines2.json")
    FROZEN = json.load(open(b2p))["datasets"] if os.path.exists(b2p) else {}
    frozen_folds = lambda tag: FROZEN[tag]["folds"] if tag in FROZEN else None

    mh = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
    print(f"mHomeGes {len(mh)} inst", flush=True)
    run_set("mHomeGes", mh, kfold(np.array([t[2] for t in mh]), 5), 10,
            frozen_folds("mHomeGes"))

    inf = infineon_recs()
    print(f"Infineon {len(inf)} inst", flush=True)
    run_set("Infineon", inf, kfold(np.array([t[2] for t in inf]), 4), 5,
            frozen_folds("Infineon"))

    mf = mmfi_instances(os.path.join(DATA, "mmfi_extracted"))
    S2 = [f"S{i:02d}" for i in (5, 10, 15, 20, 25, 30, 35, 40)]
    print(f"MM-Fi {len(mf)} inst (S2 split)", flush=True)
    run_set("MM-Fi", mf, [S2], 27, frozen_folds("MM-Fi"))

    n_params = sum(p.numel() for p in MGesNetCPDP(in_dim=1, n_cls=10).parameters())
    out = {"manifest": {"infineon_pkl_md5": hashlib.md5(
                            open(os.path.join(DATA, "infineon_recs.pkl"), "rb").read()).hexdigest(),
                        "n_instances": {k: v["n_instances"] for k, v in datasets.items()}},
           "datasets": datasets,
           "protocol": {"epochs": list(EPOCH_BUDGETS), "batch": 64, "seeds": list(SEEDS),
                        "optimizer": "adam", "lrs": [lr for _, lr in LRS], "grad_clip": 5.0,
                        "aug": "none",
                        "input": f"published-form CPDP: per 30-frame window (stride 15, "
                                 f"max {MAX_W} windows/instance, adaptive stride beyond), "
                                 f"{NR}x{ND} range x Doppler map of summed |intensity|, "
                                 "per-window max-norm, percentile-fit extents; "
                                 "window-validity mask",
                        "lr_policy": "protocol pair {3e-4,1e-3} both reported; the paper's "
                                     "training hyperparameters are not public (ACM-closed), "
                                     "so no published-default arm exists",
                        "models": {"CPDP-mGesNet":
                                   "shared shallow CNN per window: 3x[conv3x3-BN-ReLU-"
                                   "maxpool2] (16/32/64) -> adaptive-avg-pool 2x2 -> "
                                   "FC128 -> FC n_cls; masked mean of per-window logits "
                                   f"(voting reduction) ({n_params:,} params @ 10 cls)"},
                        "deviations": DEVIATIONS},
           "results": results}
    json.dump(out, open(os.path.join(DOCS, "baselines3.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DOCS, "baselines3_preds.npz"), **preds)
    print(f"\nwrote docs/baselines3.json + baselines3_preds.npz in {time.time()-t0:.0f}s", flush=True)
