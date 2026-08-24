"""MULTI-GESTURE STREAMING recognition (autonomous, 2026-07-14) — the honest test of the
PI's question: in a stream of consecutive DIFFERENT gestures, how well does online
recognition actually work, and how does it degrade under co-articulation?

Sidesteps the single-recording confounds (tight labels, motion-beyond-label, one-gesture-
per-file leakage) that made the windowing result uninterpretable. Built per the HCI
real-time survey: causal sliding-window classifier over {gestures + BACKGROUND} + a
finite-state debouncer (RadarNet/Soli-style) -> discrete events, scored with DETECTION
metrics (not clip accuracy).

Data: Infineon full recordings (per-frame targets: 0=idle, id=gesture). TRAIN a per-window
6-way classifier {5 gestures + background} on train-user recordings (gesture windows +
idle windows). TEST: concatenate held-out-user recordings of DIFFERENT classes into
A->B->C streams, two variants:
  rest      : keep the recordings' idle padding between gestures (separable; Soli regime)
  coartic   : strip inter-gesture idle so gestures abut (co-articulated; realistic)
Slide causal window, classify, debounce (fire class c when mean prob_c over last k windows
> HI and currently not refractory; refractory R windows) -> events (class, onset).

Metrics (detection, not clip acc):
  offline isolated clip acc (baseline, 5-way on the gesture windows)
  event precision/recall/F1 (a GT gesture is HIT if some fired event of the right class
    falls within its span); FP per minute (fired events not matching any GT span);
    order edit-distance (predicted vs GT class sequence); mean onset latency.
Reports rest vs coartic to show the co-articulation degradation. Env SMOKE=1: 3 users.
"""
import os, io, re, zipfile, json
import numpy as np
import pandas as pd
import infineon_detection as ifx
from rep_variants import cell_stats, compose, CAXES, kfold
from spectra_dataset import fit_ranges
from cnn import SmallCNN, _loaders
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "docs")
ZIP = os.path.join(HERE, "..", "data", "infineon", "radar_dataset.zip")
SMOKE = os.environ.get("SMOKE") == "1"
LM = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}; NULL = 5; NC = 6
W, S = 24, 6; NB = 32; FPS = 33.0
SEEDS = (0,) if SMOKE else (0, 1, 2); EP = 4 if SMOKE else 40; WIDTH = 16
HI, K, R = 0.5, 2, 6                    # debouncer: prob>HI over K windows, refractory R


def load_full():
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
        by = {}; cap = 6 if SMOKE else 30
        for r in np.random.RandomState(0).permutation(len(inp)):
            g = np.where(tg[r] > 0)[0]
            if len(g) < 2:
                continue
            cls = int(np.bincount(tg[r][tg[r] > 0]).argmax())
            if cls not in LM or by.get(cls, 0) >= cap:
                continue
            df = ifx.process_recording(inp[r])
            if len(df):
                recs.append((df, int(g.min()), int(g.max()), LM[cls], u)); by[cls] = by.get(cls, 0) + 1
    return recs


def win_map(df, t0, ranges):
    w = df[(df.frame >= t0) & (df.frame < t0 + W)]
    if len(w) < 3:
        return None
    return compose(cell_stats(w.reset_index(drop=True), CAXES, ranges, nb=NB), CAXES, ["sum"]).astype(np.float32)


def train_windows(recs, ranges):
    X, y = [], []
    for df, g0, g1, c, u in recs:
        F = int(df.frame.max()) + 1
        for t0 in range(0, max(1, F - W + 1), S):
            m = win_map(df, t0, ranges)
            if m is None:
                continue
            mid = t0 + W // 2
            X.append(m); y.append(c if g0 <= mid <= g1 else NULL)
    return np.stack(X), np.array(y)


def build_stream(insts, coartic, ranges, pad=10):
    """Concatenate isolated (df restricted to its gesture-ish span) instances into one
    frame-indexed stream. Returns (stream_df, gt_events=[(cls,s,e)], total_frames)."""
    rows = []; ev = []; off = 0
    rng = np.random.RandomState(0)
    for df, g0, g1, c, u in insts:
        span = df[(df.frame >= max(0, g0 - 4)) & (df.frame <= g1 + 4)].copy()
        if len(span) < 3:
            continue
        f0 = int(span.frame.min())
        span["frame"] = span["frame"] - f0 + off
        s, e = off, int(span.frame.max())
        rows.append(span); ev.append((c, s, e))
        gap = (2 if coartic else pad + int(rng.randint(0, 8)))
        off = e + 1 + gap
    stream = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
    return stream, ev, off


def debounce(prob):
    """prob: (T,NC). Fire class c at window t when mean prob_c over last K > HI and not
    refractory. Returns list of (class, window_idx)."""
    T = len(prob); events = []; refr = -999
    for t in range(K - 1, T):
        seg = prob[t - K + 1:t + 1].mean(0)
        c = int(seg.argmax())
        if c != NULL and seg[c] > HI and t - refr > R:
            events.append((c, t)); refr = t
    return events


if __name__ == "__main__":
    print(f"MULTI-GESTURE STREAM  SMOKE={SMOKE}  W={W} S={S} HI={HI} K={K} R={R}", flush=True)
    recs = load_full()
    print(f"recordings: {len(recs)}, users {sorted(set(t[4] for t in recs))}", flush=True)
    ranges = fit_ranges([t[0] for t in recs])
    users = sorted(set(t[4] for t in recs))
    folds = kfold(np.array(users), 2 if SMOKE else 4)
    agg = {"offline_acc": [], "rest": {"P": [], "Rc": [], "F1": [], "fp_min": [], "lat": [], "edit": []},
           "coartic": {"P": [], "Rc": [], "F1": [], "fp_min": [], "lat": [], "edit": []}}
    for te in folds:
        te_u = set(str(x) for x in te)
        tr_recs = [t for t in recs if t[4] not in te_u]; te_recs = [t for t in recs if t[4] in te_u]
        if not tr_recs or not te_recs:
            continue
        Xtr, ytr = train_windows(tr_recs, ranges)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for sd in SEEDS:
            torch.manual_seed(sd); np.random.seed(sd)
            model = SmallCNN(in_ch=Xtr.shape[1], n_cls=NC, width=WIDTH).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
            # class-balance null down to ~mean gesture count
            gi = np.where(ytr != NULL)[0]; ni = np.where(ytr == NULL)[0]
            ni = ni[np.random.RandomState(sd).permutation(len(ni))[:len(gi)]]
            sel = np.concatenate([gi, ni])
            trl, _ = _loaders(Xtr[sel], ytr[sel], Xtr[sel][:1], ytr[sel][:1], seed=sd)
            for _ in range(EP):
                model.train()
                for xb, yb in trl:
                    opt.zero_grad(); lossf(model(xb.to(dev)), yb.to(dev)).backward(); opt.step()
            model.eval()

            def prob_of(X):
                with torch.no_grad():
                    return torch.softmax(model(torch.from_numpy(X).float().to(dev)), 1).cpu().numpy()

            # offline isolated clip acc (5-way on true gesture windows of test recs)
            off_X, off_y = [], []
            for df, g0, g1, c, u in te_recs:
                m = win_map(df, max(0, g0), ranges) if g1 - g0 < W else win_map(df, g0, ranges)
                if m is not None:
                    off_X.append(m); off_y.append(c)
            if off_X:
                pr = prob_of(np.stack(off_X))[:, :5].argmax(1)
                agg["offline_acc"].append(float((pr == np.array(off_y)).mean()))
            # streaming, two variants
            for variant in ("rest", "coartic"):
                order = np.random.RandomState(sd).permutation(len(te_recs))
                insts = [te_recs[i] for i in order]
                stream, ev, F = build_stream(insts, variant == "coartic", ranges)
                if F < W or not ev:
                    continue
                wins, wt = [], []
                for t0 in range(0, F - W + 1, S):
                    m = win_map(stream, t0, ranges)
                    if m is not None:
                        wins.append(m); wt.append(t0 + W // 2)
                if not wins:
                    continue
                prob = prob_of(np.stack(wins)); wt = np.array(wt)
                fired = debounce(prob)                      # (class, window_idx)
                fired = [(c, wt[t]) for c, t in fired]      # -> (class, center-frame)
                # match events to GT spans
                hit = 0; used = [False] * len(fired); lat = []
                for (gc, s, e) in ev:
                    for j, (fc, ft) in enumerate(fired):
                        if not used[j] and fc == gc and s <= ft <= e:
                            hit += 1; used[j] = True; lat.append((ft - s) / FPS); break
                tp = hit; fp = used.count(False); fn = len(ev) - hit
                P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
                F1 = 2 * P * Rc / max(P + Rc, 1e-9)
                mins = F / FPS / 60.0
                # order edit distance (predicted class seq vs GT class seq)
                pred_seq = [c for c, _ in sorted(fired, key=lambda z: z[1])]
                gt_seq = [c for c, _, _ in ev]
                dp = np.zeros((len(pred_seq) + 1, len(gt_seq) + 1))
                for a in range(len(pred_seq) + 1): dp[a][0] = a
                for b in range(len(gt_seq) + 1): dp[0][b] = b
                for a in range(1, len(pred_seq) + 1):
                    for b in range(1, len(gt_seq) + 1):
                        dp[a][b] = min(dp[a-1][b]+1, dp[a][b-1]+1, dp[a-1][b-1]+(pred_seq[a-1] != gt_seq[b-1]))
                edit = dp[-1][-1] / max(len(gt_seq), 1)
                A = agg[variant]
                A["P"].append(P); A["Rc"].append(Rc); A["F1"].append(F1)
                A["fp_min"].append(fp / max(mins, 1e-6)); A["lat"].append(np.mean(lat) if lat else np.nan)
                A["edit"].append(edit)

    def mean(x): return round(float(np.nanmean(x)) * 100, 2) if x else None
    oa = mean(agg["offline_acc"])
    print(f"\n[OFFLINE] isolated clip acc (5-way): {oa}%", flush=True)
    out = {"offline_clip_acc": oa, "W": W, "S": S, "debounce": {"HI": HI, "K": K, "R": R},
           "epochs": EP, "seeds": list(SEEDS), "smoke": SMOKE, "streaming": {}}
    for v in ("rest", "coartic"):
        A = agg[v]
        r = {"event_P": mean(A["P"]), "event_R": mean(A["Rc"]), "event_F1": mean(A["F1"]),
             "fp_per_min": round(float(np.nanmean(A["fp_min"])), 2) if A["fp_min"] else None,
             "onset_latency_s": round(float(np.nanmean(A["lat"])), 3) if A["lat"] else None,
             "order_edit_norm": round(float(np.nanmean(A["edit"])), 3) if A["edit"] else None}
        out["streaming"][v] = r
        print(f"[STREAM:{v:7s}] event F1={r['event_F1']}% (P={r['event_P']} R={r['event_R']}) "
              f"FP/min={r['fp_per_min']} lat={r['onset_latency_s']}s edit={r['order_edit_norm']}", flush=True)
    if oa and out["streaming"].get("coartic", {}).get("event_F1") is not None:
        print(f"\nCOLLAPSE offline clip acc {oa}% -> coartic stream event-F1 {out['streaming']['coartic']['event_F1']}%", flush=True)
    json.dump(out, open(os.path.join(DOCS, f"multi_gesture_stream{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"wrote docs/multi_gesture_stream{'_smoke' if SMOKE else ''}.json", flush=True)
