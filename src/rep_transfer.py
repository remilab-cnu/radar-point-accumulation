"""P-SWING go/no-go probe (2026-07-13): is the intra-cell velocity DISTRIBUTION a
transferable cross-sensor invariant, and does it beat the CALIBRATED-SCALAR-velocity
baseline under sensor shift?

Design (per the novelty-swing strategist memo):
- Source sensor -> target sensor transfer between the two HAND-GESTURE radars
  (mHomeGes 77 GHz TI <-> Infineon 60 GHz), which have DISJOINT label spaces, so
  transfer is measured by LINEAR-PROBE: train a map-CNN on ALL source subjects, FREEZE
  the conv backbone, extract 4*width-dim features on the target, fit a linear head on
  target TRAIN subjects, evaluate on target TEST subjects (subject-disjoint k-fold).
- Commensurability is controlled: velocity uses the shared m/s scale (doppler/2.0);
  x,y,z use each dataset's own [-1,1] normalization (relative position in the FoV), and
  this same spatial normalization is used for EVERY arm, so the only thing that differs
  between the scalar and distribution arms is the velocity ENCODING.
- Arms (same channel construction as rep_converge): v_sum (calibrated scalar baseline,
  the one to beat), v_hist4 (distribution), amplitude, occupancy, v_hist4_vshuffled
  (count-matched control).

Readout (strategist's three-way gate):
  hist4 > scalar > chance  -> GO   (the distribution is the transferable invariant)
  scalar ~= hist4 (>chance) -> NO-GO, headline gone -> fall to floor paper
  both ~= chance            -> NO-GO, pipeline non-commensurable

Also reports the WITHIN-target (source==target, upper-bound) linear-probe for context.
Env SMOKE=1: tiny subset. No new architecture is introduced (probe only).
"""
import os, json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from cnn import SmallCNN, DEVICE
from torch.utils.data import TensorDataset, DataLoader
from rep_variants import cell_stats, compose, CAXES, infineon_recs, kfold
from spectra_dataset import fit_ranges, mhomeges_instances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
SMOKE = os.environ.get("SMOKE") == "1"
WIDTH = 8 if SMOKE else 32
SRC_EP = 4 if SMOKE else 120
SEEDS = (0,) if SMOKE else (0, 1, 2)
FROZEN = json.load(open(os.path.join(DOCS, "baselines2.json")))["datasets"]
ARMS = {"v_sum": ["sum"], "v_hist4": ["hist"], "amplitude": ["int_mean"],
        "occupancy": ["cnt"], "v_hist4_vshuffled": ["hist"]}


def shuffle_doppler(inst, seed=0):
    df = inst.copy(); df["doppler"] = np.random.RandomState(seed).permutation(df["doppler"].values.astype(float))
    return df


def build(insts, arm):
    ranges = fit_ranges([t[0] for t in insts])
    if arm == "v_hist4_vshuffled":
        stats = [cell_stats(shuffle_doppler(t[0], i), CAXES, ranges, nb=32) for i, t in enumerate(insts)]
        spec = ["hist"]
    else:
        stats = [cell_stats(t[0], CAXES, ranges, nb=32) for t in insts]; spec = ARMS[arm]
    X = np.stack([compose(st, CAXES, spec) for st in stats]).astype(np.float32)
    y = np.array([t[1] for t in insts]); subj = np.array([str(t[2]) for t in insts])
    return X, y, subj


def train_backbone(X, y, ncls, ep, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SmallCNN(in_ch=X.shape[1], n_cls=ncls, width=WIDTH).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                    batch_size=64, shuffle=True, generator=g)
    for _ in range(ep):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); lossf(m(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
    m.eval()
    return m


def feats(m, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).to(DEVICE)
            out.append(m.features(xb).flatten(1).cpu().numpy())
    return np.concatenate(out)


def linear_probe(F, y, subj, folds):
    """Subject-disjoint linear probe on target features F; returns mean test acc over folds."""
    accs = []
    for te_s in folds:
        te = np.isin(subj, te_s); tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(F[tr], y[tr])
        accs.append(float((clf.predict(F[te]) == y[te]).mean()))
    return float(np.mean(accs)) * 100, float(np.std(accs)) * 100


def subset(insts, n=200):
    idx = np.random.RandomState(0).permutation(len(insts))[:n]
    return [insts[i] for i in idx]


def load(name):
    if name == "mHomeGes":
        insts = mhomeges_instances(os.path.join(DATA, "mhomeges_full"))
        return (subset(insts, 400) if SMOKE else insts), 10, 5
    else:
        insts = infineon_recs()
        return (subset(insts) if SMOKE else insts), 5, 4


def folds_of(insts, k):
    return [[str(s) for s in g] for g in kfold(np.array([str(t[2]) for t in insts]), k)]


if __name__ == "__main__":
    print(f"P-SWING transfer probe SMOKE={SMOKE} width={WIDTH} src_ep={SRC_EP}", flush=True)
    out = {"protocol": {"width": WIDTH, "src_ep": SRC_EP, "seeds": list(SEEDS), "smoke": SMOKE,
                        "metric": "target subject-disjoint linear-probe acc %"}, "transfer": {}}
    (mh, mh_ncls, mh_k) = load("mHomeGes")
    (inf, inf_ncls, inf_k) = load("Infineon")
    reg = {"mHomeGes": (mh, mh_ncls, folds_of(mh, mh_k)),
           "Infineon": (inf, inf_ncls, folds_of(inf, inf_k))}
    chance = {"mHomeGes": 100.0 / mh_ncls, "Infineon": 100.0 / inf_ncls}

    for src, tgt in (("mHomeGes", "Infineon"), ("Infineon", "mHomeGes")):
        (S, s_ncls, _), (T, _, t_folds) = reg[src], reg[tgt]
        print(f"\n==== {src} -> {tgt} (chance {chance[tgt]:.1f}%) ====", flush=True)
        for arm in ARMS:
            Xs, ys, _ = build(S, arm)
            Xt, yt, st = build(T, arm)
            xfer, within = [], []
            for seed in SEEDS:
                m = train_backbone(Xs, ys, s_ncls, SRC_EP, seed)          # source-trained
                a, _ = linear_probe(feats(m, Xt), yt, st, t_folds)         # TRANSFER probe
                xfer.append(a)
                # within-target upper bound: backbone trained on target itself
                mt = train_backbone(Xt, yt, reg[tgt][1], SRC_EP, seed)
                b, _ = linear_probe(feats(mt, Xt), yt, st, t_folds)
                within.append(b)
            k = f"{src}->{tgt}|{arm}"
            out["transfer"][k] = {"transfer_acc": round(float(np.mean(xfer)), 2),
                                  "transfer_std": round(float(np.std(xfer)), 2),
                                  "within_target_acc": round(float(np.mean(within)), 2),
                                  "chance": round(chance[tgt], 2)}
            r = out["transfer"][k]
            print(f"  {arm:18s} transfer={r['transfer_acc']:6.2f} (within-tgt {r['within_target_acc']:.1f}, "
                  f"chance {r['chance']:.1f})", flush=True)

    json.dump(out, open(os.path.join(DOCS, f"transfer_probe{'_smoke' if SMOKE else ''}.json"), "w"), indent=1)
    print(f"\nwrote docs/transfer_probe{'_smoke' if SMOKE else ''}.json", flush=True)
    print("GATE: hist4>scalar>chance=GO | scalar~=hist4=NO-GO(headline gone) | both~=chance=NO-GO(non-commensurable)", flush=True)
