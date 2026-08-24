"""P-C3 (pre-registered, subset-level): split MM-Fi's 27 classes by median per-class
cancellation C_c into LOW-C / HIGH-C subsets; retrain per subset (removes joint-
classification contamination that nulled the per-class correlation).
PREDICTIONS: (a) velocity-occupancy gap more negative on LOW-C than HIGH-C;
(b) hist4 gain over v_sum larger on LOW-C than HIGH-C."""
import os, json, numpy as np
from spectra_dataset import mmfi_instances, fit_ranges
from rep_variants import cell_stats, norm, CAXES
from cnn import train_eval_preds

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data'); DOCS=os.path.join(HERE,'..','docs')
SEEDS=(0,1,2); S2=[f'S{i:02d}' for i in (5,10,15,20,25,30,35,40)]
Cc=json.load(open(os.path.join(DOCS,'pc1_perclass.json')))['C_per_class']
med=np.median(list(Cc.values()))
LOW=[int(c) for c,v in Cc.items() if v<=med]; HIGH=[int(c) for c,v in Cc.items() if v>med]
print(f'PRE-REGISTERED: vel-occ more negative & hist4 gain larger on LOW-C. median C={med:.3f}')
print(f'LOW-C ({len(LOW)}): {sorted(LOW)}\nHIGH-C ({len(HIGH)}): {sorted(HIGH)}',flush=True)

mf=mmfi_instances(os.path.join(DATA,'mmfi_extracted'))
ranges=fit_ranges([t[0] for t in mf])
stats=[cell_stats(t[0],CAXES,ranges,nb=32) for t in mf]
def arm(fn): return np.stack([fn(st) for st in stats]).astype(np.float32)
X={'v_sum':arm(lambda st:np.stack([norm(st[ax]['sum']) for ax in CAXES])),
   'v_hist4':arm(lambda st:np.stack(sum([[norm(st[ax]['hist'][k]) for k in range(4)] for ax in CAXES],[]))),
   'occupancy':arm(lambda st:np.stack([norm(st[ax]['cnt']) for ax in CAXES]))}
del stats
y=np.array([t[1] for t in mf]); subj=np.array([t[2] for t in mf])

out={}
for name,classes in (('LOW-C',LOW),('HIGH-C',HIGH)):
    m=np.isin(y,classes); remap={c:i for i,c in enumerate(sorted(classes))}
    yy=np.array([remap[c] for c in y[m]]); ss=subj[m]
    te=np.isin(ss,S2); tr=~te
    res={}
    for ep in (30,40):
        for k,Xa in X.items():
            Xs=Xa[m]
            a=[train_eval_preds(Xs[tr],yy[tr],Xs[te],yy[te],len(classes),epochs=ep,seed=s)[0] for s in SEEDS]
            res[f'{k}|ep{ep}']=(float(np.mean(a))*100,float(np.std(a))*100)
            print(f'  {name} {k:10s} ep{ep}: {res[f"{k}|ep{ep}"][0]:6.2f}% (+-{res[f"{k}|ep{ep}"][1]:.1f})',flush=True)
    out[name]=res
for name in out:
    r=out[name]
    for ep in (30,40):
        print(f'{name} ep{ep}: vel-occ={r[f"v_sum|ep{ep}"][0]-r[f"occupancy|ep{ep}"][0]:+.2f}  '
              f'hist4 gain={r[f"v_hist4|ep{ep}"][0]-r[f"v_sum|ep{ep}"][0]:+.2f}',flush=True)
json.dump(out,open(os.path.join(DOCS,'pc3_csubset.json'),'w'),indent=1)
print('wrote docs/pc3_csubset.json',flush=True)
