"""Multi-source cross-sensor transfer (go/no-go probe #1, escapes the n=1 wound).

The single mHomeGes<->Infineon pair showed the SCALAR velocity map transfers better
than the finer velocity DISTRIBUTION. That was n=1 (the same small-sample weakness that
drew 5 rejections). Now that RadHAR is in hand we have 5 heterogeneous radar sources and
can test whether the reversal REPLICATES across MULTIPLE real cross-source pairs.

Motion-scale groups (transfer only within a group = meaningful same-task-scale):
  HAND = {mHomeGes, Infineon}                       -> 2 directed pairs
  BODY = {MM-Fi, mRI, RadHAR}                        -> 6 directed pairs
Metric: subject-disjoint LINEAR-PROBE transfer (label-space-agnostic): train a map-CNN
backbone on ALL source instances, freeze, extract features on the target, fit a linear
head on target TRAIN subjects, eval on target TEST subjects. Arms: v_sum (calibrated
scalar, the one to beat), v_hist4 (distribution), amplitude, occupancy, v_hist4_vshuffled
(count-matched control). Reuses rep_transfer's train_backbone/feats/linear_probe/build.

Readout: sign & size of (hist4 - v_sum) transfer across all 8 pairs. Consistently
negative across BODY's independent pairs => the reversal is a real cross-sensor property,
not an n=1 artifact => the "richness is a transfer liability" claim is falsifiable and
supported. Mixed => the finding does not generalize; report honestly.
Env SMOKE=1 tiny subset. RERUN_GROUP=hand|body|all.
"""
import os, json
import numpy as np
from rep_transfer import build, train_backbone, feats, linear_probe, ARMS, WIDTH, SRC_EP, SEEDS, SMOKE
from rep_variants import infineon_recs, kfold
from spectra_dataset import mhomeges_instances, mmfi_instances
from rep_converge import mri_records
from radhar_loader import radhar_instances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); DOCS = os.path.join(HERE, "..", "docs")
GROUP = os.environ.get("RERUN_GROUP", "all")


def subset(insts, n=200):
    idx = np.random.RandomState(0).permutation(len(insts))[:n]
    return [insts[i] for i in idx]


def load_source(name):
    if name == "mHomeGes":
        insts = mhomeges_instances(os.path.join(DATA, "mhomeges_full")); ncls = 10; k = 5
    elif name == "Infineon":
        insts = infineon_recs(); ncls = 5; k = 4
    elif name == "MM-Fi":
        insts = mmfi_instances(os.path.join(DATA, "mmfi_extracted")); ncls = 27; k = 5
    elif name == "mRI":
        insts = mri_records(); ncls = 10; k = 5
    elif name == "RadHAR":
        insts = (radhar_instances(os.path.join(DATA, "radhar"), "Train")
                 + radhar_instances(os.path.join(DATA, "radhar"), "Test")); ncls = 5; k = 5
    else:
        raise ValueError(name)
    if SMOKE:
        insts = subset(insts, 200)
    return insts, ncls, k


GROUPS = {"HAND": ["mHomeGes", "Infineon"], "BODY": ["MM-Fi", "mRI", "RadHAR"]}
if GROUP == "hand":
    GROUPS = {"HAND": GROUPS["HAND"]}
elif GROUP == "body":
    GROUPS = {"BODY": GROUPS["BODY"]}

if __name__ == "__main__":
    print(f"MULTI-SOURCE TRANSFER group={GROUP} SMOKE={SMOKE} width={WIDTH} src_ep={SRC_EP} seeds={SEEDS}", flush=True)
    reg = {}
    for g, names in GROUPS.items():
        for nm in names:
            if nm not in reg:
                insts, ncls, k = load_source(nm)
                subj = np.array([str(t[2]) for t in insts])
                folds = [[str(s) for s in gg] for gg in kfold(subj, k)]
                reg[nm] = {"insts": insts, "ncls": ncls, "folds": folds,
                           "chance": round(100.0 / ncls, 2), "n": len(insts)}
                print(f"  loaded {nm}: n={len(insts)} ncls={ncls} folds={len(folds)}", flush=True)

    out = {"protocol": {"width": WIDTH, "src_ep": SRC_EP, "seeds": list(SEEDS), "smoke": SMOKE,
                        "metric": "target subject-disjoint linear-probe acc %"}, "transfer": {}}
    for g, names in GROUPS.items():
        for src in names:
            for tgt in names:
                if src == tgt:
                    continue
                print(f"\n==== {src} -> {tgt}  (tgt chance {reg[tgt]['chance']}%) ====", flush=True)
                for arm in ARMS:
                    Xs, ys, _ = build(reg[src]["insts"], arm)
                    Xt, yt, st = build(reg[tgt]["insts"], arm)
                    xfer = []
                    for seed in SEEDS:
                        m = train_backbone(Xs, ys, reg[src]["ncls"], SRC_EP, seed)
                        a, _ = linear_probe(feats(m, Xt), yt, st, reg[tgt]["folds"])
                        xfer.append(a)
                    k = f"{src}->{tgt}|{arm}"
                    out["transfer"][k] = {"transfer_acc": round(float(np.mean(xfer)), 2),
                                          "transfer_std": round(float(np.std(xfer)), 2),
                                          "chance": reg[tgt]["chance"]}
                    print(f"  {arm:18s} transfer={out['transfer'][k]['transfer_acc']:6.2f} "
                          f"(chance {reg[tgt]['chance']:.1f})", flush=True)

    # summary: (hist4 - v_sum) transfer delta per pair -> does the reversal replicate?
    pairs = sorted({k.split("|")[0] for k in out["transfer"]})
    print("\n=== (hist4 - v_sum) transfer delta per pair [negative = scalar wins = inversion] ===", flush=True)
    deltas = {}
    for p in pairs:
        h = out["transfer"].get(f"{p}|v_hist4", {}).get("transfer_acc")
        s = out["transfer"].get(f"{p}|v_sum", {}).get("transfer_acc")
        if h is not None and s is not None:
            deltas[p] = round(h - s, 2)
            print(f"  {p:22s} hist4-sum = {h-s:+.2f}pp", flush=True)
    out["hist4_minus_sum"] = deltas
    neg = sum(1 for v in deltas.values() if v < 0)
    print(f"\nReversal (scalar>distribution) holds in {neg}/{len(deltas)} pairs", flush=True)
    suff = f"_{GROUP}" + ("_smoke" if SMOKE else "")
    json.dump(out, open(os.path.join(DOCS, f"transfer_multi{suff}.json"), "w"), indent=1)
    print(f"wrote docs/transfer_multi{suff}.json", flush=True)
