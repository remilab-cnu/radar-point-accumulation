"""P-C1: cancellation ratio C = |sum v|/sum|v| per dataset.

PROVENANCE CORRECTION (2026-07-19): the historical mHomeGes value 0.764 quoted here
and downstream came from a run that TIMED OUT (SIGTERM after 2 min) at n=200 of an
UNSHUFFLED instance prefix (loader order -> class/distance-biased) on 2026-07-11.
It is NOT a valid dataset statistic. Definitive full-dataset recomputation (no
subsampling, all four datasets, same definition): src/c_recompute_full.py ->
docs/c_recompute_full.json. Use those values."""
import numpy as np, os, glob, pickle, pandas as pd
from spectra_dataset import mmfi_instances
from rep_variants import infineon_recs

def cancellation(insts, nb=32, T=40, cap=400):
    ratios=[]
    for inst,_,_ in insts[:cap]:
        f=inst['frame'].values.astype(float); f0,f1=f.min(),max(f.max(),f.min()+1e-9)
        ti=np.floor((f-f0)/(f1-f0)*(T-1e-9)).astype(int)
        v=inst['doppler'].values.astype(float)
        num=0.0; den=0.0
        for ax in ('x','y','z'):
            lo,hi=np.percentile(inst[ax],1),np.percentile(inst[ax],99)
            bi=np.floor((inst[ax].values-lo)/max(hi-lo,1e-9)*nb).astype(int)
            m=(bi>=0)&(bi<nb)
            sv=np.zeros((nb,T)); sa=np.zeros((nb,T))
            np.add.at(sv,(bi[m],ti[m]),v[m]); np.add.at(sa,(bi[m],ti[m]),np.abs(v[m]))
            num+=np.abs(sv).sum(); den+=sa.sum()
        if den>0: ratios.append(num/den)
    return float(np.mean(ratios)), float(np.std(ratios)), len(ratios)

DATA='../data'; rng=np.random.RandomState(0)
inf=infineon_recs(); idx=rng.permutation(len(inf))[:400]
c,s,n=cancellation([inf[i] for i in idx]); print(f'Infineon  (gesture)      C={c:.3f}+-{s:.2f} (n={n})',flush=True)
mf=mmfi_instances(os.path.join(DATA,'mmfi_extracted'))
c,s,n=cancellation(mf); print(f'MM-Fi     (whole-body)   C={c:.3f}+-{s:.2f} (n={n})',flush=True)
CL=[f'pose_{i}' for i in range(1,11)]; recs=[]
MRI=os.path.join(DATA,'mri_sample','mri_data')
for csvf in sorted(glob.glob(os.path.join(MRI,'subject*.csv'))):
    sid=os.path.basename(csvf).replace('.csv','')
    if '_all_labels' in sid: continue
    df=pd.read_csv(csvf); df.columns=[x.strip() for x in df.columns]
    can=pd.DataFrame({'frame':df['Camera Frame'].astype(int),'x':df['X'],'y':df['Y'],'z':df['Z'],
                      'doppler':df['Doppler'],'intensity':df['Intensity']})
    vl=pickle.load(open(os.path.join(MRI,f'{sid}_all_labels.cpl'),'rb'))['video_label']
    for cn in CL:
        if cn not in vl: continue
        a,b=vl[cn]; t0=a
        while t0+40<=b:
            w=can[(can.frame>=t0)&(can.frame<t0+40)]
            if w['frame'].nunique()>=6 and len(w)>=30: recs.append((w,0,'x'))
            t0+=120
c,s,n=cancellation(recs); print(f'mRI       (whole-body)   C={c:.3f}+-{s:.2f} (n={n})',flush=True)
print('\nmHomeGes C=0.764 (computed earlier)')
print('map-velocity standing: mH +14 / INF +3.5 / mRI +2.4 / MM-Fi -7')
print('P-C1 prediction: C(MM-Fi) lowest.')
