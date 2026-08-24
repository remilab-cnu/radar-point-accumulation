"""MOTION-SEGMENTATION DE-RISK (2026-07-14): measure the trimmed->continuous COLLAPSE.

Kill-gate for the motion-segmentation direction. Premise: radar gesture recognition is
trained/scored on PRE-SEGMENTED clips, but deployment sees a continuous stream. If a
clip-trained classifier, run ONLINE on the continuous stream (sliding window + a null
'no-action' class), collapses a lot vs its trimmed-clip accuracy, the deployment gap is
real and worth a benchmark/measurement paper. If it barely drops, the gap is empty -> walk.

Infineon BGT60 recordings ARE continuous streams with PER-FRAME labels (targets[r]:
0 = idle, class-id = active gesture) -> ground truth for free. No external data needed.

Protocol (subject-disjoint 4-fold over users):
  TRAIN: from train users, gesture windows (label LM[cls], 5 classes) + sampled IDLE
         windows (label 5 = null). SmallCNN velocity-sum map, W-frame windows.
  TRIMMED metric: classify the true gesture window of each test recording among the 5
         gestures (the standard paradigm) -> A_trim.
  CONTINUOUS metric: slide the SAME classifier over the FULL test recording (stride S),
         predict among {5 gestures + null}; score vs per-frame GT:
           - frame accuracy (6-way, incl null)
           - gesture DETECTION rate (does >=1 window in the true gesture span fire the
             correct class?) + onset latency
           - false positives in idle regions (non-null fires / minute)
  GAP = A_trim - continuous gesture-detection accuracy (correct-class-and-localized).
Env SMOKE=1: 3 users, 2 classes.
"""
import os, io, re, zipfile, json
import numpy as np
import pandas as pd
import infineon_detection as ifx
from rep_variants import cell_stats, compose, CAXES, kfold
from spectra_dataset import fit_ranges
from cnn import train_eval_full, SmallCNN
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
SMOKE = os.environ.get("SMOKE") == "1"
LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
NULL = 5
W, S = 24, 8                         # window frames, stride
FPS = 33.0
NB = 32; SEEDS = (0,) if SMOKE else (0, 1, 2); EP = 4 if SMOKE else 40; WIDTH = 16


def load_full_recordings():
    """Full continuous recordings: (per_frame_df, per_frame_gt_array, user)."""
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    if SMOKE:
        members = members[:3]
    recs = []
    for m in members:
        user = "u" + re.search(r"user(\d+)", m).group(1)
        with zf.open(m) as fh:
            d = np.load(io.BytesIO(fh.read())); inputs, targets = d["inputs"], d["targets"]
        by = {}; cap = 6 if SMOKE else 30
        for r in np.random.RandomState(0).permutation(len(inputs)):
            tg = targets[r]; g = np.where(tg > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(tg[tg > 0]).argmax())
            if cls not in LM or by.get(cls, 0) >= cap:
                continue
            df = ifx.process_recording(inputs[r])          # FULL recording, per-frame points
            recs.append((df, tg.astype(int), user)); by[cls] = by.get(cls, 0) + 1
    return recs


def window_map(df, t0, ranges):
    w = df[(df.frame >= t0) & (df.frame < t0 + W)]
    if len(w) < 3:
        return None
    return compose(cell_stats(w.reset_index(drop=True), CAXES, ranges, nb=NB), CAXES, ["sum"]).astype(np.float32)


def gt_window_label(gt, t0):
    seg = gt[t0:t0 + W]
    if len(seg) == 0:
        return NULL
    active = seg[seg > 0]
    if len(active) >= max(3, W // 4):                      # enough active frames -> gesture
        return LM.get(int(np.bincount(active).argmax()), NULL)
    return NULL


def build_windows(recs, ranges, idle_frac=1.0):
    """All windows -> (X, y, rec_idx, t0). idle windows subsampled to idle_frac of gestures."""
    Xs, ys, ri, ts = [], [], [], []
    for i, (df, gt, _) in enumerate(recs):
        F = int(df.frame.max()) + 1 if len(df) else 0
        for t0 in range(0, max(1, F - W + 1), S):
            X = window_map(df, t0, ranges)
            if X is None:
                continue
            Xs.append(X); ys.append(gt_window_label(gt, t0)); ri.append(i); ts.append(t0)
    X = np.stack(Xs); y = np.array(ys); ri = np.array(ri); ts = np.array(ts)
    return X, y, ri, ts


if __name__ == "__main__":
    print(f"SEG-GAP DE-RISK  SMOKE={SMOKE}  W={W} S={S}", flush=True)
    recs = load_full_recordings()
    users = sorted(set(t[2] for t in recs))
    print(f"recordings: {len(recs)}, users {users}, "
          f"mean frames {np.mean([t[0].frame.max()+1 for t in recs if len(t[0])]):.0f}", flush=True)
    ranges = fit_ranges([t[0] for t in recs if len(t[0])])
    Xall, yall, rall, tall = build_windows(recs, ranges)
    print(f"windows: {len(yall)} (gesture {(yall!=NULL).sum()}, null {(yall==NULL).sum()})", flush=True)
    rec_user = np.array([recs[i][2] for i in range(len(recs))])

    folds = kfold(np.array(users), 2 if SMOKE else 4)
    trim_acc, cont_frame, det_rate, fp_per_min = [], [], [], []
    ncls = 6                                                # 5 gestures (0-4) + null (5)
    for te_users in folds:
        te_u = set(str(u) for u in te_users)
        te_rec = np.array([rec_user[i] in te_u for i in range(len(recs))])
        tr_w = np.array([rec_user[rall[k]] not in te_u for k in range(len(yall))])
        te_w = ~tr_w
        # balance null in TRAIN (subsample to ~#gesture windows)
        tr_idx = np.where(tr_w)[0]
        g = tr_idx[yall[tr_idx] != NULL]; n = tr_idx[yall[tr_idx] == NULL]
        rng = np.random.RandomState(0)
        n = n[rng.permutation(len(n))[:len(g)]]
        tr_sel = np.concatenate([g, n])
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            model = SmallCNN(in_ch=Xall.shape[1], n_cls=ncls, width=WIDTH).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            # train
            _, _, _, _ = (None, None, None, None)
            from cnn import _loaders
            import torch.nn as nn
            opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
            dev = next(model.parameters()).device
            trl, _ = _loaders(Xall[tr_sel], yall[tr_sel], Xall[tr_sel][:1], yall[tr_sel][:1], seed=s)
            for _ep in range(EP):
                model.train()
                for xb, yb in trl:
                    opt.zero_grad(); lossf(model(xb.to(dev)), yb.to(dev)).backward(); opt.step()
            model.eval()

            def predict(X):
                with torch.no_grad():
                    return model(torch.from_numpy(X).float().to(dev)).argmax(1).cpu().numpy()

            # TRIMMED: the gesture windows of test recordings, scored 5-way (exclude null preds->wrong)
            gmask = te_w & (yall != NULL)
            pg = predict(Xall[gmask]); yg = yall[gmask]
            trim_acc.append(float((pg == yg).mean()))
            # CONTINUOUS: per test recording, slide all windows, score
            correct_frames = tot_frames = detected = n_gest = fp = idle_min = 0
            for i in np.where(te_rec)[0]:
                sel = (rall == i) & te_w
                if sel.sum() == 0:
                    continue
                order = np.argsort(tall[sel]); idx = np.where(sel)[0][order]
                preds = predict(Xall[idx]); gts = yall[idx]
                correct_frames += int((preds == gts).sum()); tot_frames += len(gts)
                # gesture-level detection: true gesture windows in this rec
                true_cls = gts[gts != NULL]
                if len(true_cls):
                    n_gest += 1
                    gc = int(np.bincount(true_cls).argmax())
                    detected += int(np.any(preds[gts != NULL] == gc))
                # false positives in idle windows
                idle = gts == NULL
                fp += int((preds[idle] != NULL).sum())
                idle_min += idle.sum() * S / FPS / 60.0
            cont_frame.append(correct_frames / max(tot_frames, 1))
            det_rate.append(detected / max(n_gest, 1))
            fp_per_min.append(fp / max(idle_min, 1e-6))

    A_trim = np.mean(trim_acc) * 100; A_cont = np.mean(cont_frame) * 100
    D = np.mean(det_rate) * 100; FP = np.mean(fp_per_min)
    print("\n=== TRIMMED -> CONTINUOUS COLLAPSE (Infineon, subj-disjoint) ===", flush=True)
    print(f"  TRIMMED clip acc (5-way on true gesture window): {A_trim:6.2f}%", flush=True)
    print(f"  CONTINUOUS frame acc (6-way incl null, online):  {A_cont:6.2f}%", flush=True)
    print(f"  gesture DETECTION rate (correct class fired in span): {D:6.2f}%", flush=True)
    print(f"  false positives in idle: {FP:.2f} / min", flush=True)
    print(f"\nCOLLAPSE (trimmed - detection) = {A_trim - D:+.2f} pp   (large => gap real => GO)", flush=True)
    out = {"n_recordings": len(recs), "W": W, "S": S, "epochs": EP, "seeds": list(SEEDS), "smoke": SMOKE,
           "trimmed_clip_acc": round(A_trim, 2), "continuous_frame_acc": round(A_cont, 2),
           "gesture_detection_rate": round(D, 2), "false_pos_per_min": round(FP, 2),
           "collapse_trimmed_minus_detection": round(A_trim - D, 2)}
    json.dump(out, open(os.path.join(DOCS, f"seg_gap_derisk{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"wrote docs/seg_gap_derisk{'_smoke' if SMOKE else ''}.json", flush=True)
