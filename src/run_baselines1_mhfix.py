"""mHomeGes baseline collapse diagnosis + equal-HPO LR pass.
Both PointNetPP and DGCNN collapsed to ~chance on mHomeGes (constant-output signature)
while passing tiny-overfit smoke and working on Infineon/MM-Fi. This run: mHomeGes only,
lr in {1e-3, 3e-4}, with per-epoch loss logging + NaN detection on fold 0 seed 0 first
(fast diagnosis), then the full grid at the surviving lr."""
import os, json, numpy as np, torch
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_round3 import kfold
from pointset_models import build_point_tensors, DEVICE
from baselines_pointnets import PointNetPP, DGCNNTemporal, train_eval_set_preds_tr

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data'); DOCS=os.path.join(HERE,'..','docs')
mh=mhomeges_instances(os.path.join(DATA,'mhomeges_full'))
ranges=fit_ranges([t[0] for t in mh])
X,M,y,subj=build_point_tensors(mh,ranges)
folds=kfold(subj,5)

# ---- diagnosis: fold0 seed0, both lrs, loss trajectory + NaN watch ----
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
def diag(model_cls,lr,epochs=8):
    te=np.isin(subj,list(folds[0])); tr=~te
    torch.manual_seed(0); np.random.seed(0)
    m=model_cls(in_dim=6,n_cls=10).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(M[tr]),torch.from_numpy(y[tr])),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(0))
    for ep in range(epochs):
        m.train(); tot=0.0; nb=0; nan=False
        for xb,mb,yb in dl:
            xb,mb,yb=xb.to(DEVICE),mb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=lf(m(xb,mb),yb)
            if not torch.isfinite(loss): nan=True; break
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
            tot+=loss.item(); nb+=1
        print(f'  {model_cls.__name__} lr={lr} ep{ep}: loss={tot/max(nb,1):.4f}{"  NaN!" if nan else ""}',flush=True)
        if nan: return 'NaN'
    return 'ok'

print('=== DIAGNOSIS (fold0 seed0, 8 epochs) ===',flush=True)
status={}
for mc in (PointNetPP,DGCNNTemporal):
    for lr in (1e-3,3e-4):
        status[(mc.__name__,lr)]=diag(mc,lr)
print('diagnosis:',status,flush=True)

# ---- full grid at 3e-4 (equal-HPO LR budget member) ----
print('=== FULL mHomeGes grid at lr=3e-4 ===',flush=True)
results={}; preds={}
for mc,name in ((PointNetPP,'PointNetPP'),(DGCNNTemporal,'DGCNNTemporal')):
    for ep in (30,40):
        accs=[]; tr_accs=[]
        for fi,te_s in enumerate(folds):
            te=np.isin(subj,list(te_s)); tr=~te; te_idx=np.where(te)[0]
            for sd in (0,1,2):
                a,yt,yp,tra=train_eval_set_preds_tr(mc,X[tr],M[tr],y[tr],X[te],M[te],y[te],10,6,epochs=ep,lr=3e-4,seed=sd)
                accs.append(a); tr_accs.append(tra)
                preds[f'mHomeGes|{name}|lr3e-4|ep{ep}|f{fi}|s{sd}']=np.stack([te_idx,yt,yp])
        results[f'mHomeGes|{name}|lr3e-4|ep{ep}']={'mean':float(np.mean(accs))*100,'std':float(np.std(accs))*100,
            'min_train_acc':float(min(tr_accs)),'underfit':bool(min(tr_accs)<0.95)}
        r=results[f'mHomeGes|{name}|lr3e-4|ep{ep}']
        print(f"  {name} ep{ep}: {r['mean']:.2f}% (+-{r['std']:.1f}){'  UNDERFIT' if r['underfit'] else ''}",flush=True)
json.dump({'diagnosis':{f'{k[0]}|{k[1]}':v for k,v in status.items()},'results':results},
          open(os.path.join(DOCS,'baselines1_mhfix.json'),'w'),indent=1)
np.savez_compressed(os.path.join(DOCS,'baselines1_mhfix_preds.npz'),**preds)
print('wrote docs/baselines1_mhfix.json',flush=True)
