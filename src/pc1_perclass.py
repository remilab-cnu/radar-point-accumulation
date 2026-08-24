"""P-C1 fine-grained: within-MM-Fi, per-class cancellation C_c vs per-class
(velocity - occupancy) map recall delta (from P2 artifacts). Prediction: positive
Spearman (higher C -> velocity holds up; low C -> velocity fails)."""
import os, json, numpy as np
from spectra_dataset import mmfi_instances
from scipy.stats import spearmanr

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data'); DOCS=os.path.join(HERE,'..','docs')
mf=mmfi_instances(os.path.join(DATA,'mmfi_extracted'))
nb,T=32,40
Cc={}
for inst,lab,_ in mf:
    f=inst['frame'].values.astype(float); f0,f1=f.min(),max(f.max(),f.min()+1e-9)
    ti=np.floor((f-f0)/(f1-f0)*(T-1e-9)).astype(int)
    v=inst['doppler'].values.astype(float)
    num=den=0.0
    for ax in ('x','y','z'):
        lo,hi=np.percentile(inst[ax],1),np.percentile(inst[ax],99)
        bi=np.floor((inst[ax].values-lo)/max(hi-lo,1e-9)*nb).astype(int)
        m=(bi>=0)&(bi<nb)
        sv=np.zeros((nb,T)); sa=np.zeros((nb,T))
        np.add.at(sv,(bi[m],ti[m]),v[m]); np.add.at(sa,(bi[m],ti[m]),np.abs(v[m]))
        num+=np.abs(sv).sum(); den+=sa.sum()
    if den>0: Cc.setdefault(lab,[]).append(num/den)
Cmean={c:float(np.mean(vs)) for c,vs in Cc.items()}
p2=json.load(open(os.path.join(DOCS,'p2_boundary_predictor.json')))
delta=p2['MM-Fi']['delta']   # per-class velocity - occupancy CNN recall
xs=[Cmean[int(c)] for c in sorted(delta,key=int)]
ys=[delta[c] for c in sorted(delta,key=int)]
r=spearmanr(xs,ys)
rng=np.random.RandomState(0); boots=[]
xs=np.array(xs); ys=np.array(ys); n=len(xs)
for _ in range(4000):
    i=rng.randint(0,n,n); boots.append(spearmanr(xs[i],ys[i]).correlation)
lo,hi=np.nanpercentile(boots,[2.5,97.5])
print(f'within-MM-Fi per-class: Spearman(C_c, vel-occ delta) = {r.correlation:.3f}  95%CI[{lo:.3f},{hi:.3f}]  n=27')
print('per-class C range: %.3f - %.3f' % (min(xs),max(xs)))
json.dump({'C_per_class':Cmean,'spearman':r.correlation,'ci':[float(lo),float(hi)]},
          open(os.path.join(DOCS,'pc1_perclass.json'),'w'),indent=1)
print('wrote docs/pc1_perclass.json')
