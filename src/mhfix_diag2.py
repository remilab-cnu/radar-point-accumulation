"""Pinpoint the mHomeGes baseline collapse: (H-A) late-epoch divergence after ep8 vs
(H-B) BatchNorm train/eval-mode mismatch (running-stats corruption).
30-epoch run, per-epoch loss; at checkpoints report train-slice & test-slice accuracy
in BOTH modes (train-mode = BN batch stats, eval-mode = running stats)."""
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from spectra_dataset import mhomeges_instances, fit_ranges
from rep_round3 import kfold
from pointset_models import build_point_tensors, DEVICE
from baselines_pointnets import PointNetPP, DGCNNTemporal

DATA=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
mh=mhomeges_instances(os.path.join(DATA,'mhomeges_full'))
X,M,y,subj=build_point_tensors(mh,fit_ranges([t[0] for t in mh]))
folds=kfold(subj,5); te=np.isin(subj,list(folds[0])); tr=~te
Xtr,Mtr,ytr,Xte,Mte,yte=X[tr],M[tr],y[tr],X[te],M[te],y[te]

def acc(m,Xa,Ma,ya,mode):
    m.train(mode=='train')
    if mode=='eval': m.eval()
    good=0; tot=0
    with torch.no_grad():
        for i in range(0,len(Xa),256):
            xb=torch.from_numpy(Xa[i:i+256]).to(DEVICE); mb=torch.from_numpy(Ma[i:i+256]).to(DEVICE)
            p=m(xb,mb).argmax(1).cpu().numpy()
            good+=(p==ya[i:i+256]).sum(); tot+=len(p)
    return good/tot

for mc in (PointNetPP,DGCNNTemporal):
    print(f'==== {mc.__name__} lr=3e-4 ====',flush=True)
    torch.manual_seed(0); np.random.seed(0)
    m=mc(in_dim=6,n_cls=10).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=3e-4); lf=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Mtr),torch.from_numpy(ytr)),
                  batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(0))
    for ep in range(30):
        m.train(); tot=0.0; nb=0; bad=False
        for xb,mb,yb in dl:
            xb,mb,yb=xb.to(DEVICE),mb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=lf(m(xb,mb),yb)
            if not torch.isfinite(loss): bad=True; break
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
            tot+=loss.item(); nb+=1
        msg=f'ep{ep:02d} loss={tot/max(nb,1):.4f}'
        if bad: msg+='  NaN!'
        if ep in (7,14,21,29):
            msg+=(f'  | trainacc T-mode={acc(m,Xtr[:4000],Mtr[:4000],ytr[:4000],"train"):.3f}'
                  f' E-mode={acc(m,Xtr[:4000],Mtr[:4000],ytr[:4000],"eval"):.3f}'
                  f' | testacc E-mode={acc(m,Xte,Mte,yte,"eval"):.3f} T-mode={acc(m,Xte,Mte,yte,"train"):.3f}')
        print('  '+msg,flush=True)
        if bad: break
print('DIAG2 DONE',flush=True)
