"""Referee item W6 (2026-07-19): parameters, MACs, and CPU latency per model.

Params: exact (sum of numel), Infineon config (n_cls=5), constructors as used in the
runs. MACs: thop-counted multiply-accumulates over conv/linear layers (custom masked
ops not counted -> lower bound for point models; exact for the pure-conv SmallCNN).
Latency: batch-1 forward, CPU, median of 100 runs after 10 warmups (this machine;
relative numbers are what matter). Inputs: SmallCNN (1,3,32,40) map; DeepSets/
PointNet++/DGCNN (1,384,6)+(1,384) points; PointLSTM (1,40,24,5)+(1,40,24) frames.
Out: docs/complexity.json
"""
import os, json, time
import numpy as np
import torch
from cnn import SmallCNN
from pointset_models import DeepSets
from baselines_pointlstm import PointLSTM
from baselines_pointnets import PointNetPP, DGCNNTemporal
from baselines_cpdp import MGesNetCPDP
from baselines_xiaxu import XiaXuNet

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
torch.set_num_threads(1)  # single-thread latency: comparable, conservative


def bench(model, inputs, reps=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(*inputs)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); model(*inputs); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000, 3)  # ms


def macs(model, inputs):
    try:
        from thop import profile
        m, _ = profile(model, inputs=inputs, verbose=False)
        return int(m)
    except Exception as e:
        return f"unavailable ({e})"


if __name__ == "__main__":
    xmap = torch.zeros(1, 3, 32, 40)
    xpts, mpts = torch.zeros(1, 384, 6), torch.ones(1, 384, dtype=torch.bool)
    xfrm, mfrm = torch.zeros(1, 40, 24, 5), torch.ones(1, 40, 24, dtype=torch.bool)
    MODELS = {
        "SmallCNN_velocity_map": (SmallCNN(in_ch=3, n_cls=5, width=16), (xmap,)),
        "DeepSets":              (DeepSets(in_dim=6, n_cls=5), (xpts, mpts)),
        "PointLSTM":             (PointLSTM(in_dim=5, n_cls=5), (xfrm, mfrm)),
        "PointNetPP":            (PointNetPP(in_dim=6, n_cls=5), (xpts, mpts)),
        "DGCNNTemporal":         (DGCNNTemporal(in_dim=6, n_cls=5), (xpts, mpts)),
        "MGesNetCPDP":           (MGesNetCPDP(in_dim=1, n_cls=5), None),
        "XiaXuNet":              (XiaXuNet(in_ch=4, n_cls=5), None),
    }
    out = {}
    for name, (model, inputs) in MODELS.items():
        n = sum(p.numel() for p in model.parameters())
        row = {"params": int(n)}
        if inputs is not None:
            try:
                row["latency_ms_cpu_b1"] = bench(model, inputs)
                row["macs_thop"] = macs(model, inputs)
            except Exception as e:
                row["latency_ms_cpu_b1"] = f"failed ({type(e).__name__}: {e})"
        out[name] = row
        print(f"{name:24s} params={n:>10,}  {row.get('latency_ms_cpu_b1','-'):>10} ms  "
              f"MACs={row.get('macs_thop','-')}", flush=True)
    json.dump(out, open(os.path.join(DOCS, "complexity.json"), "w"), indent=1)
    print("wrote docs/complexity.json", flush=True)
