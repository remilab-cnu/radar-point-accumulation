"""P-C2 (theory-driven intervention): if sign-cancellation is why velocity MAPS fail on
MM-Fi, then cancellation-PRESERVING maps (signed split / velocity histogram) should
recover the gap THERE — while on datasets with high C (gestures: signed already known
to HURT; mRI predicted high C) the gain should be ~0 or negative.
PRE-REGISTERED: gain(signed vs sum) large positive on MM-Fi; ~0/negative elsewhere.
Frozen protocol: ep30&40, batch64, seeds 0-2.
AUDIT NOTE (2026-07-12): this script NEVER saved per-instance preds (train_eval_preds
return values discarded) despite an earlier docstring claiming so. Table-III-grade
numbers with preds + subject IDs + train acc come from rep_rerun_audit.py instead;
this file is kept for the pre-registered mechanism record only."""
import os, json, glob, pickle
import numpy as np
import pandas as pd
from rep_variants import cell_stats, norm, CAXES
from spectra_dataset import fit_ranges, mmfi_instances
from rep_round3 import kfold
from cnn import train_eval_preds

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data'); DOCS=os.path.join(HERE,'..','docs')
SEEDS=(0,1,2); EPS=(30,40)
S2=[f'S{i:02d}' for i in (5,10,15,20,25,30,35,40)]
print("PRE-REGISTERED: signed/hist gains vs v_sum should be LARGE on MM-Fi, ~0/neg on mRI.",flush=True)

def build_arms(insts):
    ranges=fit_ranges([t[0] for t in insts])
    stats=[cell_stats(t[0],CAXES,ranges,nb=32) for t in insts]
    arms={
      'v_sum':     np.stack([np.stack([norm(st[ax]['sum']) for ax in CAXES]) for st in stats]).astype(np.float32),
      'v_signed':  np.stack([np.stack([norm(st[ax]['pos_mean']) for ax in CAXES]+[norm(st[ax]['neg_mean']) for ax in CAXES]) for st in stats]).astype(np.float32),
      'v_hist4':   np.stack([np.stack(sum([[norm(st[ax]['hist'][k]) for k in range(4)] for ax in CAXES],[])) for st in stats]).astype(np.float32),
      'occupancy': np.stack([np.stack([norm(st[ax]['cnt']) for ax in CAXES]) for st in stats]).astype(np.float32),
    }
    return arms

def run(tag, insts, folds, ncls):
    arms=build_arms(insts)
    y=np.array([t[1] for t in insts]); subj=np.array([t[2] for t in insts])
    res={}
    for ep in EPS:
        for arm,X in arms.items():
            accs=[]
            for te_s in folds:
                te=np.isin(subj,list(te_s)); tr=~te
                for s in SEEDS:
                    a,_,_=train_eval_preds(X[tr],y[tr],X[te],y[te],ncls,epochs=ep,seed=s)
                    accs.append(a)
            res[f'{arm}|ep{ep}']=(float(np.mean(accs))*100,float(np.std(accs))*100)
            print(f'  {tag} {arm:10s} ep{ep}: {res[f"{arm}|ep{ep}"][0]:6.2f}% (+-{res[f"{arm}|ep{ep}"][1]:.1f})',flush=True)
    return res

out={}
mf=mmfi_instances(os.path.join(DATA,'mmfi_extracted'))
out['MM-Fi']=run('MM-Fi',mf,[S2],27)

CL=[f'pose_{i}' for i in range(1,11)]; recs=[]
MRI=os.path.join(DATA,'mri_sample','mri_data')
for csvf in sorted(glob.glob(os.path.join(MRI,'subject*.csv'))):
    sid=os.path.basename(csvf).replace('.csv','')
    if '_all_labels' in sid: continue
    df=pd.read_csv(csvf); df.columns=[x.strip() for x in df.columns]
    can=pd.DataFrame({'frame':df['Camera Frame'].astype(int),'x':df['X'],'y':df['Y'],'z':df['Z'],
                      'doppler':df['Doppler'],'intensity':df['Intensity']})
    vl=pickle.load(open(os.path.join(MRI,f'{sid}_all_labels.cpl'),'rb'))['video_label']
    for ci,cn in enumerate(CL):
        if cn not in vl: continue
        a,b=vl[cn]; t0=a
        while t0+40<=b:
            w=can[(can.frame>=t0)&(can.frame<t0+40)]
            if w['frame'].nunique()>=6 and len(w)>=30: recs.append((w.reset_index(drop=True),ci,sid))
            t0+=20
out['mRI']=run('mRI',recs,kfold(np.array([t[2] for t in recs]),5),10)
json.dump(out,open(os.path.join(DOCS,'pc2_intervention.json'),'w'),indent=1)
print('\nwrote docs/pc2_intervention.json',flush=True)
print('reference gains from r1 (gestures): mH signed-sum = 61.7-66.7 = -5.0; INF = 89.9-91.9 = -2.0',flush=True)
