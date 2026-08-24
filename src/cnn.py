"""Lightweight CNN + train/eval for the Doppler-vs-amplitude cross-subject study.

CPU-only PyTorch. The architecture is intentionally small and held FIXED across
the Doppler and amplitude conditions so any accuracy gap is attributable to the
input representation, not model capacity.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(min(16, torch.get_num_threads() or 8))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, n_cls=4, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * w, n_cls)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.head(z)


def _loaders(Xtr, ytr, Xte, yte, batch=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    tr = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    te = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long())
    return (DataLoader(tr, batch_size=batch, shuffle=True, generator=g),
            DataLoader(te, batch_size=256, shuffle=False))


def train_eval(Xtr, ytr, Xte, yte, n_cls, epochs=30, lr=1e-3, seed=0, verbose=False, width=16, batch=64):
    """Train SmallCNN on (Xtr,ytr), return test accuracy on (Xte,yte)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN(in_ch=Xtr.shape[1], n_cls=n_cls, width=width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    trl, tel = _loaders(Xtr, ytr, Xte, yte, batch=batch, seed=seed)
    for ep in range(epochs):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in tel:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item(); total += yb.numel()
    return correct / max(total, 1)


def train_eval_full(Xtr, ytr, Xte, yte, n_cls, epochs=30, lr=1e-3, seed=0, width=16, batch=64):
    """Protocol-instrumented run: returns (test_acc, y_true, y_pred, final_train_acc).

    final_train_acc is measured eval-mode on the full training set after the last
    epoch -> feeds the EQUAL_HPO_PROTOCOL underfit gate (min train acc < 0.95)
    symmetrically for OUR arms (audit S5: the gate could never fire on our rows)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN(in_ch=Xtr.shape[1], n_cls=n_cls, width=width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    trl, tel = _loaders(Xtr, ytr, Xte, yte, batch=batch, seed=seed)
    for _ in range(epochs):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()

    def _acc_preds(loader):
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in loader:
                yp.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy()); yt.append(yb.numpy())
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        return float((yt == yp).mean()), yt, yp

    tr_eval = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()),
                         batch_size=256, shuffle=False)
    tr_acc, _, _ = _acc_preds(tr_eval)
    te_acc, yt, yp = _acc_preds(tel)
    return te_acc, yt, yp, tr_acc


def train_eval_preds(Xtr, ytr, Xte, yte, n_cls, epochs=30, lr=1e-3, seed=0):
    """Like train_eval but returns (accuracy, y_true, y_pred) for macro-F1 / McNemar."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN(in_ch=Xtr.shape[1], n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    trl, tel = _loaders(Xtr, ytr, Xte, yte, seed=seed)
    for _ in range(epochs):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in tel:
            xb = xb.to(DEVICE)
            yp.append(model(xb).argmax(1).cpu().numpy()); yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float((yt == yp).mean()), yt, yp
