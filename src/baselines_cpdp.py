"""CPDP / mGesNet dataset-native baseline (queue #4): Liu et al., "Real-time Arm
Gesture Recognition in Smart Home Scenarios via Millimeter Wave Sensing" (the mHomeGes
paper), ACM IMWUT 4(4):140, Dec 2020, DOI 10.1145/3432235 — adapted to this project's
frozen instance protocol (docs/EQUAL_HPO_PROTOCOL.md).

The paper is ACM-closed (no OA copy; unpaywall=closed). The CPDP construction below is
pinned from what the paper's PUBLIC material documents, via three independent sources:
- ACM abstract (doi.org/10.1145/3432235): "distill arm gesture's position and dynamic
  variation, and then custom-design a lightweight convolution neural network"; the
  indexed description names the profile CPDP and the "novel shallow neural network"
  mGesNet.
- Xi et al., "基于毫米波的人体感知研究进展", J. Software 32(7), 2021 (survey §gesture):
  "mHomeGes 首先使用固定长度的滑动窗口捕获一系列点云, 提取集中位置多普勒轮廓(CPDP),
  并将每个点的强度压缩到距离维度和多普勒维度. 随后, 将CPDP 作为卷积神经网络输入以识别
  细粒度手势" -> fixed-length sliding window captures a series of point clouds; each
  point's INTENSITY is compressed onto the (range, Doppler) plane; the resulting 2-D
  CPDP is the CNN input.
- Reading note quoting the paper (huaweicloud.csdn.net/64f5a1866b896f66024c904d.html):
  window length 30 frames; "根据距离r和多普勒d, 将对应相等的ε相加从而构成新的2维矩阵"
  (points with equal quantized (r,d) have intensities ε summed -> the 2-D matrix);
  CPDP = time-collapsed sum of the denoised RD maps; mGesNet models are trained per
  anchor position; an HMM-based voting mechanism (HMM-VM) merges per-window scores on
  the continuous stream.
- GesturePrint (arXiv:2408.05358 §2): mHomeGes/mTransSee "convert point clouds into
  the concentrated position-doppler profile to emphasize the positional relationship
  and speed differences among points".

Implemented form: per instance, 30-frame windows (stride 15) -> per-window (32x32)
range x Doppler intensity-sum map, max-normed -> shared shallow CNN -> per-window
logits -> masked mean over valid windows (voting reduction). mHomeGes instances are
median 17 frames, so the single whole-instance window (== the published window-level
classification) is the common case; long instances (MM-Fi actions) get the paper's
sliding-window treatment. Every non-public choice is in DEVIATIONS below.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pointset_models import DEVICE

# every guess forced by the paywall, + protocol adaptations (goes into run JSONs)
DEVIATIONS = [
    "CPDP rebuilt from the distributed point clouds: r = ||(x,y,z)|| per detected "
    "point; the paper compresses intensity onto the radar's native range/Doppler "
    "axes upstream of point detection (exact axis geometry not public)",
    "profile dims 32x32 (range x Doppler), percentile-fit extents (r: 1-99% +5% pad; "
    "Doppler: symmetric 99% of |v| +5% pad, 0 centered); published bin counts/extents "
    "are in no public material",
    "cell value = sum of |intensity| of the window's points in the (r,d) cell "
    "(documented), then per-window max-norm (project map convention; published "
    "normalization undocumented); no clutter removal beyond the datasets' own "
    "detection front-end (paper: 'denoised' RD upstream + UDAN user discovery, "
    "inapplicable to single-user pre-segmented instances)",
    "window = 30 frames, stride 15, in frame-number space over PRE-SEGMENTED "
    "instances (paper: 30-frame sliding window on the continuous 10-fps stream); "
    "instances shorter than one window -> a single whole-instance window; instances "
    "needing > max_w windows use a proportionally larger stride (full coverage kept); "
    "frame-count windows mean the real-time duration differs where the frame period "
    "differs from mHomeGes's 100 ms (Infineon)",
    "instance decision = masked MEAN of per-window logits; the published HMM-based "
    "voting mechanism (HMM-VM) segments continuous streams and is inapplicable to "
    "pre-segmented instances",
    "one model per fold across all anchor distances/subjects (paper trains a separate "
    "mGesNet per anchor position); required by the frozen subject-disjoint protocol",
    "mGesNet layer spec is not public ('novel shallow neural network' / 'lightweight "
    "CNN' is all the public record states): implemented as 3x[conv3x3-BN-ReLU-"
    "maxpool2] widths 16/32/64 -> adaptive-avg-pool 2x2 -> FC128 -> FC n_cls "
    "(~58k params); BN track_running_stats=False (round-1 corruption fix)",
    "point-less windows are masked invalid; all-windows-empty instance -> zero "
    "logits via has-guard (padded window slots never enter the network)",
    "published training hyperparameters are not public -> frozen equal-HPO budgets "
    "only; no published-default LR arm is possible",
]


# ---------------- CPDP tensor builders (canonical df: frame,x,y,z,doppler,intensity) --
def fit_cpdp_ranges(insts, pad=0.05):
    """Robust CPDP axis extents from a list of instances (fit_ranges convention):
    r = point range (m), 1st-99th pct + pad; d = Doppler, symmetric 99th pct of |v|."""
    r = np.concatenate([np.sqrt(i["x"].values ** 2 + i["y"].values ** 2
                                + i["z"].values ** 2) for i in insts])
    d = np.concatenate([i["doppler"].values.astype(float) for i in insts])
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    span = max(hi - lo, 1e-9)
    dmax = max(float(np.percentile(np.abs(d), 99)) * (1 + pad), 1e-9)
    return {"r": (float(lo - pad * span), float(hi + pad * span)),
            "d": (-dmax, dmax)}


def instance_to_cpdp(inst, ranges, nr=32, nd=32, win=30, stride=15, max_w=16):
    """One instance -> (W,nr,nd) float32 stack of per-window CPDPs + (W,) validity.
    Each window: sum |intensity| of its points into (range-bin, Doppler-bin) cells,
    then max-norm. Out-of-extent points dropped (cell_stats convention). Point-less
    windows stay zero + invalid."""
    f = inst["frame"].values.astype(float)
    if len(f) == 0:                                        # empty-instance guard
        return np.zeros((1, nr, nd), np.float32), np.zeros((1,), bool)
    r = np.sqrt(inst["x"].values ** 2 + inst["y"].values ** 2 + inst["z"].values ** 2)
    d = inst["doppler"].values.astype(float)
    a = np.abs(inst["intensity"].values.astype(float))
    (rlo, rhi), (dlo, dhi) = ranges["r"], ranges["d"]
    ri = np.floor((r - rlo) / (rhi - rlo) * nr).astype(int)
    di = np.floor((d - dlo) / (dhi - dlo) * nd).astype(int)
    ok = (ri >= 0) & (ri < nr) & (di >= 0) & (di < nd)
    f0, span = f.min(), f.max() - f.min() + 1
    nwin = 1 + max(0, int(np.ceil((span - win) / stride)))
    if nwin > max_w:                       # long instance: widen stride, keep coverage
        stride = max(1, int(np.ceil((span - win) / max(max_w - 1, 1))))
        nwin = max_w
    maps = np.zeros((nwin, nr, nd), np.float32)
    valid = np.zeros((nwin,), bool)
    for k in range(nwin):
        s0 = f0 + k * stride
        sel = ok & (f >= s0) & (f < s0 + win)
        if not sel.any():
            continue
        np.add.at(maps[k], (ri[sel], di[sel]), a[sel])
        mx = maps[k].max()
        if mx > 0:                          # has-guard: all-zero-intensity window
            maps[k] /= mx
        valid[k] = True
    return maps, valid


def build_cpdp_tensors(insts_labeled, ranges, nr=32, nd=32, win=30, stride=15, max_w=16):
    """(B,W,nr,nd) + window mask (B,W) + y + subj; W = dataset max window count."""
    per = [instance_to_cpdp(t[0], ranges, nr, nd, win, stride, max_w)
           for t in insts_labeled]
    B, W = len(per), max(p[0].shape[0] for p in per)
    X = np.zeros((B, W, nr, nd), np.float32)
    M = np.zeros((B, W), bool)
    for b, (mp, va) in enumerate(per):
        X[b, :len(va)] = mp
        M[b, :len(va)] = va
    return (X, M, np.array([t[1] for t in insts_labeled], np.int64),
            np.array([t[2] for t in insts_labeled]))


# ---------------- mGesNet-style classifier ----------------
class MGesNetCPDP(nn.Module):
    """Shallow CNN on per-window CPDPs, masked score-mean voting across windows.
    in_dim = profile channels (1 = the documented intensity CPDP); profile dims are
    free (adaptive pooling) — pass nr/nd at build time. Padded/invalid window slots
    are gathered OUT before the network, so padding cannot leak (exact invariance,
    BN batch stats included)."""

    def __init__(self, in_dim=1, n_cls=10, w=16, drop=0.4):
        super().__init__()
        def blk(ci, co):
            return [nn.Conv2d(ci, co, 3, padding=1),
                    nn.BatchNorm2d(co, track_running_stats=False),
                    nn.ReLU(True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blk(in_dim, w), *blk(w, 2 * w),
                                      *blk(2 * w, 4 * w), nn.AdaptiveAvgPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 4, 8 * w),
                                  nn.ReLU(True), nn.Dropout(drop),
                                  nn.Linear(8 * w, n_cls))
        self.in_dim, self.n_cls = in_dim, n_cls

    def forward(self, x, m):                # x (B,W,nr,nd) or (B,W,C,nr,nd), m (B,W)
        if x.dim() == 4:
            x = x.unsqueeze(2)              # single profile channel
        B, W = x.shape[:2]
        fm = m.reshape(B * W)
        out = torch.zeros(B, self.n_cls, device=x.device, dtype=x.dtype)
        if fm.any():                        # has-guard: nothing valid in batch
            xv = x.reshape(B * W, *x.shape[2:])[fm]      # VALID windows only
            lv = self.head(self.features(xv))            # (Nv, n_cls)
            bidx = torch.arange(B, device=x.device).repeat_interleave(W)[fm]
            out = out.index_add(0, bidx, lv) / m.sum(1, keepdim=True).clamp(min=1)
        return out                          # all-empty instance -> zero logits


# ---------------- CPU smoke test (NOT a training experiment; VESSL-only rule) -------
if __name__ == "__main__":
    import os, glob, time
    from preprocess import load_mgesture_csv, segment_instances
    HERE = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(HERE, "..", "data", "mhomeges_full", "longGes_1.2m", "806")
    insts = []
    for ci, cls in enumerate(("circle", "push", "clap", "up")):           # 4-class toy
        segs, _ = segment_instances(load_mgesture_csv(
            glob.glob(os.path.join(root, f"point_*_{cls}.csv"))[0]))
        insts += [(s, ci, "806") for s in segs[:3]]                        # ~12 real inst
    print(f"smoke: {len(insts)} real mHomeGes instances, 4 classes, device={DEVICE}")
    ranges = fit_cpdp_ranges([t[0] for t in insts])
    X, M, y, _ = build_cpdp_tensors(insts, ranges)
    if M.all():                                # guarantee padded slots for the test
        X = np.concatenate([X, np.zeros_like(X[:, :1])], 1)
        M = np.concatenate([M, np.zeros_like(M[:, :1])], 1)
    print(f"CPDP tensors: X{X.shape} window-mask fill {M.mean():.2f} "
          f"r=({ranges['r'][0]:.2f},{ranges['r'][1]:.2f}) "
          f"d=({ranges['d'][0]:.2f},{ranges['d'][1]:.2f})")
    torch.manual_seed(0)
    model = MGesNetCPDP(in_dim=1, n_cls=4)
    print(f"MGesNetCPDP: params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb, mb, yb = torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)
    t0 = time.time(); model.train(); losses = []
    for it in range(150):                                                 # tiny-overfit
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(xb, mb), yb); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        acc = float((model(xb, mb).argmax(1) == yb).float().mean())
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f} | train acc {acc:.2f} "
          f"(chance 0.25, gate 0.80) | {time.time()-t0:.1f}s "
          f"{'PASS' if acc > 0.8 and losses[-1] < losses[0] else 'FAIL'}")
    with torch.no_grad():                                 # padding invariance: garbage
        o1 = model(xb, mb)                                # in masked window slots must
        g = torch.Generator().manual_seed(1)              # change NOTHING (exact 0)
        xg = xb.clone()
        xg[~mb] = 100.0 * torch.randn((int((~mb).sum()), *xb.shape[2:]), generator=g) + 7.0
        o2 = model(xg, mb)
        dmax = (o1 - o2).abs().max().item()
    print(f"  padding invariance: max |diff| = {dmax} {'PASS' if dmax == 0.0 else 'FAIL'}")
    assert dmax == 0.0
    with torch.no_grad():                                 # profile-dim variant + empty
        X16, M16, _, _ = build_cpdp_tensors(insts, ranges, nr=16, nd=16)
        o16 = MGesNetCPDP(in_dim=1, n_cls=4).eval()(torch.from_numpy(X16),
                                                    torch.from_numpy(M16))
        me = mb.clone(); me[0] = False                    # all-empty instance
        oe = model(xb, me)
    assert o16.shape == (len(insts), 4) and torch.isfinite(o16).all()
    assert torch.isfinite(oe).all() and (oe[0] == 0).all()
    print("  16x16 profile forward OK; all-empty instance -> zero logits, finite OK")
