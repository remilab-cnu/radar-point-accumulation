"""Published point-cloud baselines, round 1 (repositioning memo queue #1-#2):

PointNetPP    : PointNet++ SSG classification (Qi et al., NeurIPS 2017), compact
                2-level variant sized for 384-point radar clouds.
DGCNNTemporal : DGCNN edge-conv classifier (Wang et al., TOG 2019), 2 EdgeConv
                blocks; time enters as the per-point t feature (full-feature edges).

Both consume the masked variable-length tensors of pointset_models.build_point_tensors
(X (B,384,in_dim) with columns a subset of [x,y,z,v,A,t]; first 3 MUST be x,y,z).

Adaptations vs the published forms (record in run JSON under protocol.deviations):
- masked FPS / ball-query / kNN: padded points are never selected as centroids,
  never grouped, never neighbors (distances to padding forced to +1e10/+1e9).
- DGCNN graph is built ONCE on (x,y,z) (static) instead of dynamic feature-space
  re-kNN per layer; edge features still use the full running feature vector.
- compact widths (2 SA levels / 2 EdgeConv blocks) vs the published 3/4.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pointset_models import DEVICE


# ---------------- masked sampling / grouping primitives ----------------
def index_points(x, idx):
    """x (B,N,C), idx (B,...) long -> x gathered along dim 1, shape (B,...,C)."""
    b = torch.arange(x.size(0), device=x.device).view([-1] + [1] * (idx.dim() - 1)).expand_as(idx)
    return x[b, idx]


def masked_fps(xyz, mask, npoint):
    """Farthest-point sampling over VALID points only. xyz (B,N,3), mask (B,N) bool.
    Padded points get distance -1 so they are never argmax while any valid point
    remains; if a sample has fewer valid points than npoint, valid centroids repeat."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.where(mask, torch.full_like(xyz[..., 0], 1e10), torch.full_like(xyz[..., 0], -1.0))
    farthest = mask.float().argmax(1)                       # first valid point (0 if none)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        d = ((xyz - xyz[ar, farthest].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)                        # padded stay at -1 (d >= 0)
        farthest = dist.argmax(1)
    return idx


def ball_group(xyz, feat, mask, cidx, radius, nsample):
    """Group valid points within radius of each centroid -> (B,S,nsample,3+C)
    [rel_xyz | feat_j]. Padding excluded; out-of-radius slots replicate the first
    in-radius point (centroids are valid points, so self dist=0 is always in)."""
    B, N, _ = xyz.shape
    cx = index_points(xyz, cidx)                             # (B,S,3)
    sq = torch.cdist(cx, xyz).pow(2).masked_fill(~mask[:, None, :], 1e10)
    gidx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, cidx.size(1), N).clone()
    gidx[sq > radius * radius] = N
    gidx = gidx.sort(-1).values[:, :, :nsample]
    first = gidx[:, :, :1].expand(-1, -1, nsample)
    gidx = torch.where(gidx == N, first, gidx)
    return torch.cat([index_points(xyz, gidx) - cx[:, :, None], index_points(feat, gidx)], -1)


def masked_knn(xyz, mask, k):
    """k nearest VALID neighbors on (x,y,z). Returns idx (B,N,k), valid (B,N,k)."""
    sq = torch.cdist(xyz, xyz).masked_fill(~mask[:, None, :], 1e9)
    idx = sq.topk(k, dim=-1, largest=False).indices
    return idx, index_points(mask[..., None], idx).squeeze(-1)


# ---------------- PointNet++ (SSG, compact) ----------------
class SetAbstraction(nn.Module):
    def __init__(self, in_ch, mlp, npoint, radius, nsample):
        super().__init__()
        layers, c = [], in_ch + 3
        for o in mlp:
            layers += [nn.Conv2d(c, o, 1), nn.BatchNorm2d(o, track_running_stats=False), nn.ReLU(True)]; c = o
        self.net = nn.Sequential(*layers)
        self.npoint, self.radius, self.nsample = npoint, radius, nsample

    def forward(self, xyz, feat, mask):
        cidx = masked_fps(xyz, mask, self.npoint)
        g = ball_group(xyz, feat, mask, cidx, self.radius, self.nsample)   # (B,S,K,3+C)
        h = self.net(g.permute(0, 3, 1, 2)).max(-1).values                 # (B,C',S) max over K
        return index_points(xyz, cidx), h.transpose(1, 2)                  # centroids all valid


class GlobalAbstraction(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], in_ch + 3
        for o in mlp:
            layers += [nn.Conv1d(c, o, 1), nn.BatchNorm1d(o, track_running_stats=False), nn.ReLU(True)]; c = o
        self.net = nn.Sequential(*layers)

    def forward(self, xyz, feat, mask):
        h = self.net(torch.cat([xyz, feat], -1).transpose(1, 2))            # (B,C',S)
        h = h.masked_fill(~mask[:, None, :], -1e9).max(-1).values
        has = mask.any(1, keepdim=True)                                     # empty-set guard
        return torch.where(has, h, torch.zeros_like(h))


class PointNetPP(nn.Module):
    """PointNet++ SSG: SA(128, r1, 32, [64,64,128]) -> SA(32, r2, 64, [128,128,256])
    -> global SA [256,512] -> FC. Coords normalized to [-1,1] -> r1=0.2, r2=0.4."""
    def __init__(self, in_dim=6, n_cls=10, r1=0.2, r2=0.4, drop=0.4):
        super().__init__()
        self.sa1 = SetAbstraction(in_dim - 3, [64, 64, 128], 128, r1, 32)
        self.sa2 = SetAbstraction(128, [128, 128, 256], 32, r2, 64)
        self.sa3 = GlobalAbstraction(256, [256, 512])
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(drop),
                                  nn.Linear(256, n_cls))

    def forward(self, x, m):
        m = m.clone(); m[~m.any(1), 0] = True        # all-empty sample -> one zero point
        xyz1, f1 = self.sa1(x[..., :3], x[..., 3:], m)
        m1 = torch.ones(xyz1.shape[:2], dtype=torch.bool, device=x.device)
        xyz2, f2 = self.sa2(xyz1, f1, m1)
        return self.head(self.sa3(xyz2, f2, torch.ones_like(m1[:, :32])))


# ---------------- DGCNN (edge-conv, temporal via t feature) ----------------
class EdgeConv(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], 2 * in_ch
        for o in mlp:
            layers += [nn.Conv2d(c, o, 1), nn.BatchNorm2d(o, track_running_stats=False), nn.LeakyReLU(0.2, True)]; c = o
        self.net = nn.Sequential(*layers)

    def forward(self, x, idx, valid):                # x (B,N,C); idx,valid (B,N,k)
        xi = x[:, :, None].expand(-1, -1, idx.size(-1), -1)
        e = torch.cat([xi, index_points(x, idx) - xi], -1)                  # (B,N,k,2C)
        h = self.net(e.permute(0, 3, 1, 2))                                 # (B,C',N,k)
        h = h.masked_fill(~valid[:, None], -1e9).max(-1).values             # max over k
        has = valid.any(-1)                                                 # empty-nbhd guard
        return torch.where(has[:, None], h, torch.zeros_like(h)).transpose(1, 2)


class DGCNNTemporal(nn.Module):
    """DGCNN: masked kNN(k) graph on (x,y,z), EdgeConv [64,64] then [128] on full
    features (incl. t), concat skip, masked global max+mean pool -> FC head."""
    def __init__(self, in_dim=6, n_cls=10, k=16, drop=0.4):
        super().__init__()
        self.k = k
        self.ec1 = EdgeConv(in_dim, [64, 64])
        self.ec2 = EdgeConv(64, [128])
        self.head = nn.Sequential(nn.Linear(2 * 192, 256), nn.LeakyReLU(0.2, True),
                                  nn.Dropout(drop), nn.Linear(256, n_cls))

    def forward(self, x, m):
        idx, valid = masked_knn(x[..., :3], m, min(self.k, x.size(1)))
        h1 = self.ec1(x, idx, valid)
        h = torch.cat([h1, self.ec2(h1, idx, valid)], -1)                   # (B,N,192)
        hmax = h.masked_fill(~m[..., None], -1e9).max(1).values
        has = m.any(1, keepdim=True)                                        # empty-set guard
        hmax = torch.where(has, hmax, torch.zeros_like(hmax))
        hmean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([hmax, hmean], -1))


# ---------------- trainer: pointset_models pattern + train-acc (underfit gate) ------
def train_eval_set_preds_tr(model_cls, Xtr, Mtr, ytr, Xte, Mte, yte, n_cls, in_dim,
                            epochs=30, lr=1e-3, seed=0):
    """train_eval_set_preds + final TRAIN accuracy for the EQUAL_HPO convergence gate."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(in_dim=in_dim, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True, generator=g)
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in tr:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb, mb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    def _ev(X, M, y):
        yt, yp = [], []
        with torch.no_grad():
            for xb, mb, yb in DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(M),
                                                       torch.from_numpy(y)), batch_size=128):
                yp.append(model(xb.to(DEVICE), mb.to(DEVICE)).argmax(1).cpu().numpy())
                yt.append(yb.numpy())
        return np.concatenate(yt), np.concatenate(yp)
    yt, yp = _ev(Xte, Mte, yte)
    ta, tp = _ev(Xtr, Mtr, ytr)
    return float((yt == yp).mean()), yt, yp, float((ta == tp).mean())


# ---------------- CPU smoke test (NOT a training experiment; VESSL-only rule) -------
if __name__ == "__main__":
    import os, glob, time
    from preprocess import load_mgesture_csv, segment_instances
    from spectra_dataset import fit_ranges
    from pointset_models import build_point_tensors
    HERE = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(HERE, "..", "data", "mhomeges_full", "longGes_1.2m", "806")
    insts = []
    for ci, cls in enumerate(("circle", "push", "clap", "up")):          # 4-class toy subset
        segs, _ = segment_instances(load_mgesture_csv(
            glob.glob(os.path.join(root, f"point_*_{cls}.csv"))[0]))
        insts += [(s, ci, "806") for s in segs[:3]]                       # ~12 real instances
    print(f"smoke: {len(insts)} real mHomeGes instances, 4 classes, device={DEVICE}")
    X, M, y, _ = build_point_tensors(insts, fit_ranges([t[0] for t in insts]))
    for name, cls in (("PointNetPP", PointNetPP), ("DGCNNTemporal", DGCNNTemporal)):
        torch.manual_seed(0)
        model = cls(in_dim=6, n_cls=4)
        print(f"{name}: params={sum(p.numel() for p in model.parameters()):,}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xb, mb, yb = torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)
        t0 = time.time(); model.train(); losses = []
        for it in range(60):                                              # tiny overfit sanity
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(xb, mb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            acc = float((model(xb, mb).argmax(1) == yb).float().mean())
        print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f} | train acc {acc:.2f} "
              f"(chance 0.25) | {time.time()-t0:.1f}s "
              f"{'PASS' if acc > 0.25 and losses[-1] < losses[0] else 'FAIL'}")
        with torch.no_grad():                                             # column-subset path
            o5 = cls(in_dim=5, n_cls=4).eval()(xb[..., [0, 1, 2, 3, 5]], mb)
            o3 = cls(in_dim=3, n_cls=4).eval()(xb[..., :3], mb)
            me = mb.clone(); me[0] = False                                # empty-mask sample
            oe = model(xb, me)
        assert o5.shape == o3.shape == (len(insts), 4) and torch.isfinite(oe).all()
        print(f"  in_dim 5/3 forward OK; empty-mask sample finite OK")
