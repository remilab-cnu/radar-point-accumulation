"""Point-set (rasterization-free) baselines: consume the raw point cloud directly.

DeepSets      : whole-instance point set, per-point (x,y,z,v,A,t) -> shared MLP ->
                masked max+mean pooling -> classifier. Permutation-invariant.
FramePointGRU : points bucketed into T time bins -> shared MLP + masked max-pool per
                bin -> GRU over time -> classifier. (PointLSTM-lite temporal variant.)

Purpose (blueprint 'RawSet' control): (1) quantify the information cost of the 2D-map
compression, (2) test whether the map-domain ranking is a rasterization artifact,
(3) point-feature ablations (drop velocity / drop intensity) = the point-domain analog
of the what-to-accumulate question.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(min(16, torch.get_num_threads() or 8))


# ---------------- tensor builders ----------------
def instance_to_points(inst, ranges, n_points=384, seed=0):
    """(N,6) float32 [x,y,z,v,a,t] normalized + mask. Sampling without replacement."""
    n = len(inst)
    rng = np.random.RandomState(seed + n)
    idx = rng.choice(n, size=min(n, n_points), replace=False)
    out = np.zeros((n_points, 6), np.float32)
    mask = np.zeros((n_points,), bool)
    f = inst["frame"].values.astype(float)
    f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
    a = inst["intensity"].values.astype(float)
    amax = max(np.abs(a).max(), 1e-9)
    for j, i in enumerate(idx):
        r = inst.iloc[i]
        out[j, 0] = 2 * (r.x - ranges["x"][0]) / (ranges["x"][1] - ranges["x"][0]) - 1
        out[j, 1] = 2 * (r.y - ranges["y"][0]) / (ranges["y"][1] - ranges["y"][0]) - 1
        out[j, 2] = 2 * (r.z - ranges["z"][0]) / (ranges["z"][1] - ranges["z"][0]) - 1
        out[j, 3] = r.doppler / 2.0
        out[j, 4] = a[i] / amax
        out[j, 5] = (r.frame - f0) / (f1 - f0)
        mask[j] = True
    return out, mask


def build_point_tensors(insts_labeled, ranges, n_points=384):
    X, M, y, subj = [], [], [], []
    for k, (inst, lab, s) in enumerate(insts_labeled):
        p, m = instance_to_points(inst, ranges, n_points, seed=k)
        X.append(p); M.append(m); y.append(lab); subj.append(s)
    return np.stack(X), np.stack(M), np.array(y, np.int64), np.array(subj)


def build_frame_tensors(insts_labeled, ranges, T=40, K=24):
    """(B,T,K,5) [x,y,z,v,a] + mask (B,T,K); time encoded by bin position."""
    B = len(insts_labeled)
    X = np.zeros((B, T, K, 5), np.float32); M = np.zeros((B, T, K), bool)
    y, subj = [], []
    for b, (inst, lab, s) in enumerate(insts_labeled):
        f = inst["frame"].values.astype(float)
        f0, f1 = f.min(), max(f.max(), f.min() + 1e-9)
        ti = np.floor((f - f0) / (f1 - f0) * (T - 1e-9)).astype(int)
        a = inst["intensity"].values.astype(float); amax = max(np.abs(a).max(), 1e-9)
        feats = np.stack([
            2 * (inst["x"].values - ranges["x"][0]) / (ranges["x"][1] - ranges["x"][0]) - 1,
            2 * (inst["y"].values - ranges["y"][0]) / (ranges["y"][1] - ranges["y"][0]) - 1,
            2 * (inst["z"].values - ranges["z"][0]) / (ranges["z"][1] - ranges["z"][0]) - 1,
            inst["doppler"].values / 2.0,
            a / amax], 1).astype(np.float32)
        rng = np.random.RandomState(b)
        for t in range(T):
            sel = np.where(ti == t)[0]
            if len(sel) == 0:
                continue
            sel = rng.choice(sel, size=min(len(sel), K), replace=False)
            X[b, t, :len(sel)] = feats[sel]; M[b, t, :len(sel)] = True
        y.append(lab); subj.append(s)
    return X, M, np.array(y, np.int64), np.array(subj)


# ---------------- models ----------------
class DeepSets(nn.Module):
    def __init__(self, in_dim=6, n_cls=10, w=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, w), nn.ReLU(True), nn.Linear(w, 2 * w), nn.ReLU(True))
        self.rho = nn.Sequential(nn.Linear(4 * w, 2 * w), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(2 * w, n_cls))

    def forward(self, x, m):
        h = self.phi(x)                                   # (B,N,H)
        hmax = h.masked_fill(~m[..., None], -1e9).max(1).values
        has = m.any(1, keepdim=True)                      # guard: all-empty sample -> zeros
        hmax = torch.where(has, hmax, torch.zeros_like(hmax))
        hmean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.rho(torch.cat([hmax, hmean], -1))


class FramePointGRU(nn.Module):
    def __init__(self, in_dim=5, n_cls=10, w=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, w), nn.ReLU(True), nn.Linear(w, 2 * w), nn.ReLU(True))
        self.gru = nn.GRU(2 * w, 2 * w, batch_first=True)
        self.fc = nn.Linear(2 * w, n_cls)

    def forward(self, x, m):                              # x (B,T,K,F), m (B,T,K)
        B, T, K, F = x.shape
        mm = m.reshape(B * T, K)
        h = self.phi(x.reshape(B * T, K, F))
        h = h.masked_fill(~mm[..., None], -1e9).max(1).values
        # BUG FIX (round-4 collapse): empty bins gave finfo.min (finite!) which passed the
        # old isinf check and saturated the GRU -> constant output. Zero them explicitly.
        has = mm.any(1, keepdim=True)
        h = torch.where(has, h, torch.zeros_like(h))
        out, _ = self.gru(h.reshape(B, T, -1))
        return self.fc(out.mean(1))


# ---------------- train/eval ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval_set_full(model_cls, Xtr, Mtr, ytr, Xte, Mte, yte, n_cls, in_dim, epochs=30, lr=1e-3, seed=0):
    """Protocol-instrumented run: returns (test_acc, y_true, y_pred, final_train_acc).
    Same training loop as train_eval_set_preds; adds an eval-mode pass over the
    training set for the symmetric underfit gate (audit S5/R4)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(in_dim=in_dim, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True, generator=g)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Mte), torch.from_numpy(yte)),
                    batch_size=256)
    tr_eval = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr)),
                         batch_size=256)
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in tr:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb, mb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()

    def _acc_preds(loader):
        yt, yp = [], []
        with torch.no_grad():
            for xb, mb, yb in loader:
                xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                yp.append(model(xb, mb).argmax(1).cpu().numpy()); yt.append(yb.numpy())
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        return float((yt == yp).mean()), yt, yp

    tr_acc, _, _ = _acc_preds(tr_eval)
    te_acc, yt, yp = _acc_preds(te)
    return te_acc, yt, yp, tr_acc


def train_eval_set(model_cls, Xtr, Mtr, ytr, Xte, Mte, yte, n_cls, in_dim, epochs=30, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(in_dim=in_dim, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True, generator=g)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Mte), torch.from_numpy(yte)),
                    batch_size=256)
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in tr:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb, mb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, mb, yb in te:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb, mb).argmax(1) == yb).sum().item(); total += yb.numel()
    return correct / max(total, 1)


def train_eval_set_preds(model_cls, Xtr, Mtr, ytr, Xte, Mte, yte, n_cls, in_dim, epochs=30, lr=1e-3, seed=0):
    """Like train_eval_set but returns (acc, y_true, y_pred) for error analysis."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(in_dim=in_dim, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True, generator=g)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Mte), torch.from_numpy(yte)),
                    batch_size=256)
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in tr:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb, mb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for xb, mb, yb in te:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            yp.append(model(xb, mb).argmax(1).cpu().numpy()); yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float((yt == yp).mean()), yt, yp
