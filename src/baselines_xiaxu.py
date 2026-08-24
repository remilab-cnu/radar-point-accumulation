"""FAITHFUL Xia & Xu (2022) baseline row — raw-spectrum pipeline on the Infineon raw cubes.

Reference: Z. Xia and F. Xu, "Time-Space Dimension Reduction of Millimeter-Wave Radar
Point-Clouds for Smart-Home Hand-Gesture Recognition," IEEE Sensors J. 22(5):4425-4437,
2022 (full text: docs/references/xia_xu_2022_jsen_timespace.txt).

This row reproduces their RAW-SPECTRUM representation (Sec. III "Multi-Dimensional
Spectrum Representation"), their spatial-position-alignment preprocessing (Sec. V-A),
and their multi-channel CNN "MyNetV2" (Fig. 13), on the Infineon BGT60TR13C raw ADC
cubes — the only dataset in this campaign for which raw spectra exist (mHomeGes ships
detected point clouds only -> cite-and-position there, no runnable row).

Pipeline, as documented in the paper:
  1. Per chirp range-FFT; static background removed per Eq. (7) (mean over the chirp
     axis subtracted — the MTI step); per frame per channel Doppler-FFT -> RD spectrum.
     Reuses infineon_detection._rd_cube (identical FFT/MTI conventions).
  2. Eq. (8): "The RD spectrum representation is obtained by incoherent superposition
     averaging of all channel RD spectrums", E(r,v) = (1/Nc) sum_k |E~_k(r,v)|.
  3. 2D CA-CFAR detects target CELLS in the RD map; the target bounding box
     (i_min,i_max,j_min,j_max) defines the range spectrum Eq. (9) (mean of E(r,v_j)
     over the Doppler box) and the Doppler spectrum Eq. (10) (mean over the range box).
  4. Eq. (11): per detected target point p, the 1D angular spectrum is the magnitude of
     the FFT/DBF over the antenna pair, averaged over points:
     E(w) = (1/NS) sum_p |E~_p(w)|, for azimuth and elevation.
  5. Columns over the frame axis give the range-time (RTA), Doppler-time (DTA),
     azimuth-time (ATA) and elevation-time (ETA) spectra.
  6. Spatial position alignment (Sec. V-A): cluster the target points of a feature
     image, take the largest cluster, translate its bounding-box center to the image
     center (Eqs. 19-20). Vertical centering applies to the position-varying features
     ("The features of RTA, ATA, XTA, and YTA ... appear in different positions in the
     vertical direction"); horizontal (time) centering applies to all features
     ("eliminate the time difference in the horizontal direction and the space
     difference in the vertical direction").
  7. Feature images "are first scaled to the input size of 64 x 64 for the CNN".
  8. MyNetV2: 3x [conv3x3 stride1 -> BN -> ReLU -> maxpool2x2], depths 64/128/256,
     FC1024 -> dropout 0.5 -> FC N_CLASS -> softmax; Adam lr 1e-3, batch 64.

NO detected-point clouds enter the representation: the CA-CFAR output is used only as
the paper's own target bounding box / target-cell set (their raw-spectrum pipeline is
not defined without it). The 3D point-cloud reconstruction, spherical->cartesian
transform and point-cloud spectrums of their Sec. IV are exactly what this row omits —
that contrast is the point of the faithful row. NOTE: their best-performing variant
(XTA+YTA+ZTA, Table IV) is point-cloud-built; this row implements their spectrum-
representation feature set (RTA/DTA/ATA/ETA), which the paper evaluates side by side.

Frozen-protocol replay: instances are re-derived from the RAW cubes by replaying the
exact selection of rep_variants.infineon_recs() (same per-user RandomState(0)
permutation, same LABELMAP, same +-6-frame gesture window, same >=8-detections filter,
same 40/class/user cap) so the 2400 instances, folds and manifest match the frozen
protocol (docs/EQUAL_HPO_PROTOCOL.md). process_recording is executed ONLY to replay
the inclusion filter; its points are discarded.

All deviations from the published method are listed in DEVIATIONS (recorded in the run
JSON by run_baselines4.py).
"""
from __future__ import annotations
import io
import os
import re
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, TensorDataset

import infineon_detection as ifx
from cnn import DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ZIP = os.path.join(DATA, "infineon", "radar_dataset.zip")
CACHE = os.path.join(DATA, "xiaxu_maps.npz")
FROZEN_PKL = os.path.join(DATA, "infineon_recs.pkl")

LABELMAP = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}      # frozen (rep_variants.infineon_recs)
MARGIN = 6                                      # frozen gesture window +-6 frames
PER_CLS_CAP = 40                                # frozen 40/class/user -> 2400 total
MIN_DETS = 8                                    # frozen >=8-detections inclusion filter

IMG = 64            # their CNN input: feature images "scaled to the input size of 64x64"
NANG = 64           # angle-grid points (their N_omega unspecified; maps end at 64x64 anyway)
ANG_MAX_DEG = 45.0  # their ATA measurement range: "-45deg ~ 45deg" (Sec. V-A)
ALIGN_THR = 0.15    # target-cluster binarization (their clustering algorithm unspecified)

# 2-element DBF steering phases, sign conventions identical to infineon_detection
_THETA = np.deg2rad(np.linspace(-ANG_MAX_DEG, ANG_MAX_DEG, NANG))
_PSI = 2.0 * np.pi * ifx.D_OVER_LAMBDA * np.sin(_THETA)          # (NANG,)

DEVIATIONS = [
    "60GHz 3RX hardware vs their sensor: Infineon BGT60TR13C 60.5GHz/4GHz BW, 1TX x 3RX "
    "L-array (azimuth pair rx(0,2), elevation pair rx(1,2), lambda/2), 32 chirps x 64 "
    "samples/frame, ~33fps, range <=1.2m vs their Calterah 60.5-64GHz TDM-MIMO 4TX x 4RX "
    "2D virtual array (8 azimuth / 3 elevation virtual channels), 256 chirps x 256 "
    "samples, 20fps, 6.4m range. ATA/ETA are therefore 2-element interferometric beam "
    "patterns (single coarse main lobe) — a hardware limitation, not a pipeline change",
    "incoherent superposition (Eq. 8) averages Nc=3 physical RX channels (their Nc=16 "
    "virtual channels)",
    "2D CA-CFAR on the 32x32 RD map uses the chain-validated train=9/guard=3/alpha=2.5 "
    "(their train 9 / guard 15 / Pfa 0.01 assume 256x256 maps and do not fit a 32-bin "
    "axis); frames with zero detections fall back to the global-peak RD cell as the "
    "single target point (the paper leaves weak frames 'incorrectly estimated', Sec. III)",
    "spatial-position-alignment clustering: 8-connected components on a >15%-of-max "
    "binarization, largest component = target cluster (their point-clustering algorithm "
    "is unspecified); vertical centering applied to RTA and ATA only (their stated "
    "position-varying features are RTA/ATA/XTA/YTA; XTA/YTA do not exist in the "
    "raw-spectrum feature set), horizontal (time) centering applied to all channels",
    "windows are the frozen-protocol gesture windows (active frames +-6, variable "
    "~14-40 frames) instead of their 30-frame/1.5s sliding window; native-resolution "
    "maps are rescaled to their 64x64 CNN input exactly as in their preprocessing "
    "(their non-gesture-frame filtering is played by the frozen gesture windowing)",
    "channel set RTA+DTA+ATA+ETA (their Sec. III spectrum-representation features); "
    "their best variant XTA+YTA+ZTA is built from detected point clouds and is excluded "
    "by design from this raw-spectrum row",
    "training uses the frozen equal-HPO budget (ep {30,40} x lr {1e-3,3e-4} x seeds "
    "{0,1,2}, batch 64, Adam, no grad clip — matching cnn.train_eval used for our rows); "
    "their published setting (Adam lr 1e-3 constant, batch 64) is covered at lr 1e-3 but "
    "their 100-epoch budget is capped at the protocol 30/40",
    "CNN mirrors MyNetV2 in every documented dimension (3x conv3x3(s1)-BN-ReLU-"
    "maxpool2x2, depths 64/128/256, FC1024, dropout 0.5); unspecified details chosen "
    "as: padding=1 ('same'), ReLU after FC1024, BatchNorm without running stats (repo "
    "baseline convention)",
    "per-map max-normalization before the CNN (their amplitude normalization is "
    "unspecified; only BN inside the network is documented)",
    "frozen-manifest replay executes the detection chain solely to replay the >=8-"
    "detections inclusion filter of the frozen instance set; no detected points enter "
    "the representation",
]


# ---------------- representation: Sec. III spectra + Sec. V-A alignment ----------------
def _angle_spectrum(ca, cb, sign):
    """Eq. (11) with a 2-element pair: E(w) = (1/NS) sum_p |c_a,p + c_b,p e^{j psi(w)}|/2.

    psi(w) = 2*pi*(d/lambda)*sin(w)*sign, matched to the monopulse convention of
    infineon_detection (peak at sin(w) = sign*angle(c_a c_b*)/(2 pi d/lambda))."""
    steer = np.exp(1j * sign * _PSI)[None, :]                    # (1, NANG)
    return np.abs(ca[:, None] + cb[:, None] * steer).mean(0) / 2.0   # (NANG,)


def frame_spectra(RD):
    """One frame's raw-spectrum features from the complex RD cube (3, Nv, Nr).

    Returns (E_r (32,), E_v (32,), E_az (NANG,), E_el (NANG,)) per Eqs. (8)-(11)."""
    E = np.abs(RD).mean(axis=0)                                  # Eq. (8), (Nv, Nr)
    det = ifx._detect(E, method="ca", alpha=ifx.CFAR_ALPHA,
                      train=ifx.CFAR_TRAIN, guard=ifx.CFAR_GUARD)
    det[:, :ifx.RANGE_MIN_BIN] = False
    det[:, ifx.RANGE_MAX_BIN:] = False
    if not det.any():                                            # weak frame fallback
        sub = E.copy()
        sub[:, :ifx.RANGE_MIN_BIN] = 0.0
        sub[:, ifx.RANGE_MAX_BIN:] = 0.0
        det[np.unravel_index(np.argmax(sub), sub.shape)] = True
    dd, rr = np.where(det)                                       # Doppler idx, range idx
    e_r = E[dd.min():dd.max() + 1, :].mean(axis=0)               # Eq. (9): Doppler-box avg
    e_v = E[:, rr.min():rr.max() + 1].mean(axis=1)               # Eq. (10): range-box avg
    c = RD[:, dd, rr]                                            # (3, NS) complex
    e_az = _angle_spectrum(c[ifx.AZ_PAIR[0]], c[ifx.AZ_PAIR[1]], ifx.AZ_SIGN)
    e_el = _angle_spectrum(c[ifx.EL_PAIR[0]], c[ifx.EL_PAIR[1]], ifx.EL_SIGN)
    return e_r[ifx.RANGE_MIN_BIN:ifx.RANGE_MAX_BIN], e_v, e_az, e_el


def _shift(img, dv, dh):
    """Translate by (dv, dh) with zero fill (Eq. 20 applied to the whole image)."""
    out = np.zeros_like(img)
    H, W = img.shape
    y0, y1 = max(0, dv), min(H, H + dv)
    x0, x1 = max(0, dh), min(W, W + dh)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = img[y0 - dv:y1 - dv, x0 - dh:x1 - dh]
    return out


def spatial_align(img, vertical=True, thr=ALIGN_THR):
    """Sec. V-A spatial position alignment: largest target cluster -> bounding box ->
    off-center pixel offsets (Eq. 19, 0-based) -> translate the box center to the image
    center (Eq. 20). Horizontal (time) centering always; vertical only if `vertical`."""
    m = img.max()
    if m <= 0:
        return img
    lab, n = ndimage.label(img > thr * m, structure=np.ones((3, 3)))
    if n == 0:
        return img
    k = 1 + int(np.argmax(ndimage.sum_labels(np.ones_like(img), lab, range(1, n + 1))))
    ys, xs = np.where(lab == k)
    mu_v = (int(ys.min()) + int(ys.max())) // 2 - (img.shape[0] - 1) // 2
    mu_h = (int(xs.min()) + int(xs.max())) // 2 - (img.shape[1] - 1) // 2
    return _shift(img, -mu_v if vertical else 0, -mu_h)


def _to_img(m):
    """Their preprocessing: scale the feature image to the 64x64 CNN input; max-norm."""
    z = ndimage.zoom(m, (IMG / m.shape[0], IMG / m.shape[1]), order=1)
    mx = z.max()
    return (z / mx if mx > 0 else z).astype(np.float32)


def build_maps(cube, align=True):
    """Raw ADC window (Nf,3,32,64) -> (4, 64, 64) float32 [RTA, DTA, ATA, ETA]."""
    cols = [frame_spectra(ifx._rd_cube(cube[f])) for f in range(cube.shape[0])]
    rta, dta, ata, eta = (np.stack([c[i] for c in cols], axis=1) for i in range(4))
    if align:
        rta = spatial_align(rta, vertical=True)
        dta = spatial_align(dta, vertical=False)
        ata = spatial_align(ata, vertical=True)
        eta = spatial_align(eta, vertical=False)
    return np.stack([_to_img(m) for m in (rta, dta, ata, eta)])


# ---------------- frozen-manifest replay from raw cubes ----------------
def xiaxu_dataset(cache=CACHE):
    """Replay the frozen instance selection from the RAW cubes and build the maps.

    Returns (X (N,4,64,64) float32, y (N,) int64, subj (N,) str, rec_idx (N,) int64,
    n_dets (N,) int64). rec_idx / n_dets document the replay for manifest cross-checks."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], z["subj"].astype(str), z["rec_idx"], z["n_dets"]
    zf = zipfile.ZipFile(ZIP)
    members = sorted([m for m in zf.namelist() if re.search(r"user\d+_e1\.npz$", m)
                      and not re.search(r"_(fast|slow|wrist)", m)],
                     key=lambda m: int(re.search(r"user(\d+)", m).group(1)))
    X, y, subj, rec_idx, n_dets = [], [], [], [], []
    for m in members:
        user = "u" + re.search(r"user(\d+)", m).group(1)
        t0 = time.time()
        with zf.open(m) as f:
            d = np.load(io.BytesIO(f.read()))
            inputs, targets = d["inputs"], d["targets"]
        got = _replay_user(inputs, targets)
        for r, win, lab, nd in got:
            X.append(build_maps(win)); y.append(lab); subj.append(user)
            rec_idx.append(r); n_dets.append(nd)
        print(f"  {user}: {len(got)} instances in {time.time() - t0:.0f}s", flush=True)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, np.int64); subj = np.array(subj)
    rec_idx = np.array(rec_idx, np.int64); n_dets = np.array(n_dets, np.int64)
    np.savez_compressed(cache, X=X, y=y, subj=subj, rec_idx=rec_idx, n_dets=n_dets)
    return X, y, subj.astype(str), rec_idx, n_dets


def _replay_user(inputs, targets, cap=PER_CLS_CAP):
    """Exact selection replay of rep_variants.infineon_recs for one user file.

    Yields (recording_index, raw window cube, mapped label, n detections). The
    detection chain is run ONLY for the frozen >=MIN_DETS inclusion filter."""
    got, by = [], {}
    for r in np.random.RandomState(0).permutation(len(inputs)):
        g = np.where(targets[r] > 0)[0]
        if len(g) < 2:
            continue
        cls = int(np.bincount(targets[r][targets[r] > 0]).argmax())
        if cls not in LABELMAP or by.get(cls, 0) >= cap:
            continue
        win = inputs[r, max(0, g.min() - MARGIN):g.max() + MARGIN + 1]
        df = ifx.process_recording(win)          # frozen inclusion filter ONLY
        if len(df) < MIN_DETS:
            continue
        got.append((int(r), win, LABELMAP[cls], len(df)))
        by[cls] = by.get(cls, 0) + 1
    return got


def verify_against_pkl(y, subj, n_dets, pkl=FROZEN_PKL, users=None):
    """Cross-check the replay against the frozen manifest data/infineon_recs.pkl.

    Compares instance count, (label, user) sequence, and per-instance detection count
    (len(df)). `users` restricts the check to a subset (partial/smoke verification).
    Returns a dict suitable for the run-JSON manifest."""
    import pickle
    if not os.path.exists(pkl):
        return {"checked": False, "reason": "frozen pkl not present"}
    recs = pickle.load(open(pkl, "rb"))
    if users is not None:
        recs = [t for t in recs if t[2] in set(users)]
    ok_n = len(recs) == len(y)
    ok_seq = ok_n and all(int(t[1]) == int(a) and t[2] == b
                          for t, a, b in zip(recs, y, subj))
    ok_det = ok_n and all(len(t[0]) == int(n) for t, n in zip(recs, n_dets))
    return {"checked": True, "n_frozen": len(recs), "n_replayed": int(len(y)),
            "labels_users_match": bool(ok_seq), "det_counts_match": bool(ok_det)}


# ---------------- MyNetV2 (Fig. 13) + protocol trainer ----------------
class XiaXuNet(nn.Module):
    """MyNetV2 (Xia & Xu 2022, Fig. 13): 3x [conv 3x3 stride 1 -> BN -> ReLU ->
    maxpool 2x2 stride 2], depths 64/128/256; FC 1024 -> ReLU -> dropout 0.5 ->
    FC n_cls. Input N-channel 64x64. ~17.2M parameters (FC1024 dominates)."""

    def __init__(self, in_ch=4, n_cls=5, depths=(64, 128, 256), fc=1024, drop=0.5):
        super().__init__()
        layers, c = [], in_ch
        for d in depths:
            layers += [nn.Conv2d(c, d, 3, stride=1, padding=1),
                       nn.BatchNorm2d(d, track_running_stats=False),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
            c = d
        self.features = nn.Sequential(*layers)
        side = IMG // 2 ** len(depths)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * side * side, fc),
                                  nn.ReLU(inplace=True), nn.Dropout(drop),
                                  nn.Linear(fc, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def train_eval_xiaxu(Xtr, ytr, Xte, yte, n_cls, epochs=30, lr=1e-3, seed=0, batch=64):
    """Equal-HPO trainer (mirrors cnn.train_eval / baselines_pointnets contract):
    Adam, batch 64, no grad clip, no augmentation. Returns
    (test_acc, y_true, y_pred, train_acc) for the underfit gate + per-instance preds."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = XiaXuNet(in_ch=Xtr.shape[1], n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    trl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                   torch.from_numpy(ytr).long()),
                     batch_size=batch, shuffle=True, generator=g)
    for _ in range(epochs):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()

    def _ev(X, yv):
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                                   torch.from_numpy(yv).long()),
                                     batch_size=128):
                yp.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                yt.append(yb.numpy())
        return np.concatenate(yt), np.concatenate(yp)

    yt, yp = _ev(Xte, yte)
    ta, tp = _ev(Xtr, ytr)
    return float((yt == yp).mean()), yt, yp, float((ta == tp).mean())


# ---------------- CPU smoke (NOT a training experiment; VESSL-only rule) ----------------
if __name__ == "__main__":
    import sys
    n_par = sum(p.numel() for p in XiaXuNet(4, 5).parameters())
    print(f"XiaXuNet (MyNetV2-faithful): {n_par:,} params, device={DEVICE}")

    if "--replay-user1" in sys.argv:
        # exact-selection replay check for user1 against the frozen pkl (needs ~10GB RAM)
        src = os.path.join(DATA, "infineon", "extracted", "user1_e1.npz")
        d = np.load(src)
        t0 = time.time()
        got = _replay_user(d["inputs"], d["targets"])
        print(f"user1 replay: {len(got)} instances in {time.time() - t0:.0f}s")
        y1 = np.array([g[2] for g in got]); n1 = np.array([g[3] for g in got])
        chk = verify_against_pkl(y1, np.array(["u1"] * len(got)), n1, users=["u1"])
        print("frozen-pkl cross-check (u1):", chk)
        sys.exit(0)

    # default smoke: representation + tiny overfit on windows carved from the sample npy
    cube = np.load(os.path.join(DATA, "infineon_raw_sample", "user10_e1_recording0.npy"))
    print(f"sample cube {cube.shape} {cube.dtype}")
    wins = [cube[s:s + 24] for s in range(0, 80, 8)]                 # 10 windows
    t0 = time.time()
    maps = np.stack([build_maps(w) for w in wins])
    print(f"maps {maps.shape} in {time.time() - t0:.1f}s | finite={np.isfinite(maps).all()}"
          f" | min={maps.min():.3f} max={maps.max():.3f}"
          f" | per-ch max==1: {np.allclose(maps.max(axis=(2, 3)), 1.0)}")
    raw = build_maps(wins[0], align=False)
    for i, nm in enumerate(("RTA", "DTA", "ATA", "ETA")):
        cy0 = float(ndimage.center_of_mass(raw[i])[0])
        cy1 = float(ndimage.center_of_mass(maps[0, i])[0])
        print(f"  {nm}: v-center of mass {cy0:5.1f} -> {cy1:5.1f} (target ~31.5)")
    torch.manual_seed(0)
    model = XiaXuNet(4, 5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.from_numpy(maps).float()
    yb = torch.arange(len(wins)) % 5                                  # toy labels
    losses = []
    for it in range(60):                                              # tiny-overfit sanity
        model.train()
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(xb), yb); loss.backward()
        opt.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        acc = float((model(xb).argmax(1) == yb).float().mean())
    print(f"tiny-overfit: loss {losses[0]:.3f} -> {losses[-1]:.4f}, train acc {acc:.2f}")
