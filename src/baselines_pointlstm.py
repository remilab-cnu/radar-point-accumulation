"""PointLSTM parity-anchor baseline (queue #3): Min et al., "An Efficient PointLSTM for
Point Clouds Based Gesture Recognition", CVPR 2020 (repo Blueprintf/pointlstm-gesture-
recognition-pytorch). Compact masked adaptation for this project's frozen frame tensors
(pointset_models.build_frame_tensors: X (B,T=40,K=24,5)=[x,y,z,v,A] + mask (B,T,K)).

Core mechanism kept faithful (verified against the reference PointLSTMCell/PointLSTM):
per point i in frame t, group k nearest neighbors j (on xyz) from frame t-1; shared
gate weights act on [feat_i ; (p_{t-1,j} - p_{t,i}) ; h_{t-1,j}]; each neighbor yields
a candidate (h~,c~) via the standard LSTM update with the NEIGHBOR's cell state c_j;
both h and c are then max-pooled over neighbors (reference AdaptiveMaxPool2d((None,1)))
to give the point's new state. Frame 0 groups within itself with zero initial states.

(+published-form notes) — deviations from the published form (record in run JSON):
- backbone: the published Motion net is a 4-stage hierarchy (intra-frame kNN-MLP stage,
  two inter-frame MotionBlocks, range-based point downsampling 128->64->32->16, LSTM at
  the middle stage (hidden 256), [1024] MLP + global max-pool). Here: one pointwise
  shared MLP -> one PointLSTM layer (hidden 128) -> masked spatio-temporal max+mean
  pooling -> FC head. No intra-frame kNN grouping, no MotionBlocks, no downsampling.
- grouping: published direct grouping topk=16 of 128 pts/frame; here k=4 of <=24
  pts/bin (similar neighbor ratio). k=1 reduces to pure 1-NN state association.
- offsets: published positions/offsets are 4-D (x,y,z,d); here 3-D xyz offsets, and the
  cell input is the point's MLP feature (abs xyz is in the raw features) rather than
  the raw position+feature concat. offsets=False variant only (no aligned grouping).
- point features: published uses 4-D positions only; here the protocol's common columns
  [x,y,z,v,A] (in_dim param for subsets; first 3 MUST be x,y,z).
- masking / empty bins (published frames are dense 32x128, no padding): masked kNN
  (+1e9 distance fill, padded slots never chosen while a valid one exists), neighbor
  max-pool masked by pair validity with an explicit has-guard -> zeros (NO finfo.min);
  a sample's empty time bin skips its state update (state + positions carry over from
  the last non-empty bin); each sample's FIRST non-empty bin self-groups with zero
  states (the published t=0 behavior, generalized per sample).
- published training (pointlstm.yaml): Adam base_lr 1e-4, batch 8, 200 epochs, wd 5e-3,
  step decay. Here: the frozen equal-HPO budgets (see docs/EQUAL_HPO_PROTOCOL.md).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pointset_models import DEVICE
from baselines_pointnets import index_points


class PointLSTM(nn.Module):
    def __init__(self, in_dim=5, n_cls=10, w=64, k=4, drop=0.3):
        super().__init__()
        self.k, self.hid = k, 2 * w
        self.phi = nn.Sequential(nn.Linear(in_dim, w), nn.ReLU(True), nn.Linear(w, 2 * w), nn.ReLU(True))
        self.gates = nn.Linear(2 * w + 3 + self.hid, 4 * self.hid)   # shared across points+neighbors
        self.head = nn.Sequential(nn.Linear(2 * self.hid, 2 * w), nn.ReLU(True), nn.Dropout(drop),
                                  nn.Linear(2 * w, n_cls))

    def forward(self, x, m):                          # x (B,T,K,F) [x,y,z,...], m (B,T,K) bool
        B, T, K, _ = x.shape
        k, H = min(self.k, K), self.hid
        xyz, feat = x[..., :3], self.phi(x)           # (B,T,K,3), (B,T,K,2w)
        z3 = torch.zeros(B, K, 3, device=x.device, dtype=x.dtype)
        prev_xyz, prev_valid = z3, torch.zeros(B, K, dtype=torch.bool, device=x.device)
        prev_h = torch.zeros(B, K, H, device=x.device, dtype=x.dtype)
        prev_c = torch.zeros_like(prev_h)
        outs = []
        for t in range(T):
            cx, cm, cf = xyz[:, t], m[:, t], feat[:, t]
            # neighbor source: previous non-empty bin; before any -> this bin, zero states
            fresh = ~prev_valid.any(1)                                    # (B,)
            nb_xyz = torch.where(fresh[:, None, None], cx, prev_xyz)
            nb_h = torch.where(fresh[:, None, None], torch.zeros_like(prev_h), prev_h)
            nb_c = torch.where(fresh[:, None, None], torch.zeros_like(prev_c), prev_c)
            nb_valid = torch.where(fresh[:, None], cm, prev_valid)
            # masked kNN on xyz: padded neighbors pushed to +1e9, never chosen while a
            # valid one exists; leftover invalid picks are masked out of the pool below
            d = torch.cdist(cx, nb_xyz).masked_fill(~nb_valid[:, None, :], 1e9)
            idx = d.topk(k, -1, largest=False).indices                    # (B,K,k)
            pv = index_points(nb_valid[..., None], idx).squeeze(-1)       # pair validity (B,K,k)
            rel = index_points(nb_xyz, idx) - cx[:, :, None]              # (B,K,k,3)
            hj, cj = index_points(nb_h, idx), index_points(nb_c, idx)     # (B,K,k,H)
            g = self.gates(torch.cat([cf[:, :, None].expand(-1, -1, k, -1), rel, hj], -1))
            gi, gf, go, gg = g.split(H, -1)
            cc = torch.sigmoid(gf) * cj + torch.sigmoid(gi) * torch.tanh(gg)
            hc = torch.sigmoid(go) * torch.tanh(cc)                       # candidates (B,K,k,H)
            hp = hc.masked_fill(~pv[..., None], -1e9).max(2).values       # max-pool over neighbors
            cp = cc.masked_fill(~pv[..., None], -1e9).max(2).values
            has = pv.any(2, keepdim=True)                                 # has-guard: no valid
            hp = torch.where(has, hp, torch.zeros_like(hp)) * cm[..., None]  # neighbor -> zeros;
            cp = torch.where(has, cp, torch.zeros_like(cp)) * cm[..., None]  # padded slots zeroed
            outs.append(hp)
            upd = cm.any(1)                                               # empty bin: carry state
            prev_xyz = torch.where(upd[:, None, None], cx, prev_xyz)
            prev_h = torch.where(upd[:, None, None], hp, prev_h)
            prev_c = torch.where(upd[:, None, None], cp, prev_c)
            prev_valid = torch.where(upd[:, None], cm, prev_valid)
        h = torch.stack(outs, 1).reshape(B, T * K, H)                     # spatio-temporal pool
        mm = m.reshape(B, T * K)
        hmax = h.masked_fill(~mm[..., None], -1e9).max(1).values
        hall = mm.any(1, keepdim=True)                                    # all-empty guard
        hmax = torch.where(hall, hmax, torch.zeros_like(hmax))
        hmean = (h * mm[..., None]).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([hmax, hmean], -1))


# ---------------- CPU smoke test (NOT a training experiment; VESSL-only rule) -------
if __name__ == "__main__":
    import os, glob, time
    from preprocess import load_mgesture_csv, segment_instances
    from spectra_dataset import fit_ranges
    from pointset_models import build_frame_tensors
    HERE = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(HERE, "..", "data", "mhomeges_full", "longGes_1.2m", "806")
    insts = []
    for ci, cls in enumerate(("circle", "push", "clap", "up")):           # 4-class toy subset
        segs, _ = segment_instances(load_mgesture_csv(
            glob.glob(os.path.join(root, f"point_*_{cls}.csv"))[0]))
        insts += [(s, ci, "806") for s in segs[:3]]                       # ~12 real instances
    print(f"smoke: {len(insts)} real mHomeGes instances, 4 classes, device={DEVICE}")
    X, M, y, _ = build_frame_tensors(insts, fit_ranges([t[0] for t in insts]))
    print(f"frame tensors: X{X.shape} mask fill {M.mean():.2f}")
    torch.manual_seed(0)
    model = PointLSTM(in_dim=5, n_cls=4)
    print(f"PointLSTM: params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb, mb, yb = torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)
    t0 = time.time(); model.train(); losses = []
    for it in range(150):                                                 # tiny-overfit sanity
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(xb, mb), yb); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        acc = float((model(xb, mb).argmax(1) == yb).float().mean())
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f} | train acc {acc:.2f} "
          f"(chance 0.25, gate 0.80) | {time.time()-t0:.1f}s "
          f"{'PASS' if acc > 0.8 and losses[-1] < losses[0] else 'FAIL'}")
    with torch.no_grad():                                                 # padding invariance:
        o1 = model(xb, mb)                                                # garbage in padded
        g = torch.Generator().manual_seed(1)                              # slots must change
        xg = xb.clone()                                                   # NOTHING (exact 0)
        xg[~mb] = 100.0 * torch.randn((~mb).sum(), xb.shape[-1], generator=g) + 7.0
        o2 = model(xg, mb)
        dmax = (o1 - o2).abs().max().item()
    print(f"  padding invariance: max |diff| = {dmax} {'PASS' if dmax == 0.0 else 'FAIL'}")
    assert dmax == 0.0
    with torch.no_grad():                                                 # column subsets +
        o4 = PointLSTM(in_dim=4, n_cls=4).eval()(xb[..., :4], mb)         # empty bins/sample
        o3 = PointLSTM(in_dim=3, n_cls=4).eval()(xb[..., :3], mb)
        me = mb.clone(); me[0] = False                                    # all-empty sample
        me[1, 10:30] = False                                              # long empty-bin gap
        oe = model(xb, me)
    assert o4.shape == o3.shape == (len(insts), 4) and torch.isfinite(oe).all()
    print(f"  in_dim 4/3 forward OK; all-empty sample + empty-bin gap finite OK")
