"""Baselines round-1 FIX2: BN track_running_stats=False (batch-stats at eval; fixes the
mHomeGes running-stats corruption collapse — see mhfix_diag2: train-mode test acc was
0.81/0.66 while eval-mode was chance). Re-run ALL datasets under the single fixed
implementation, both LR budget points {1e-3, 3e-4}. Deviation recorded."""
import os, json, hashlib, numpy as np
from spectra_dataset import mhomeges_instances, mmfi_instances, fit_ranges
from rep_variants import infineon_recs
from rep_round3 import kfold
from pointset_models import build_point_tensors
from baselines_pointnets import PointNetPP, DGCNNTemporal, train_eval_set_preds_tr

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data'); DOCS=os.path.join(HERE,'..','docs')
SEEDS=(0,1,2); EPS=(30,40); LRS=(("1e-3",1e-3),("3e-4",3e-4))
results={}; preds={}

def run_set(tag, insts, folds, ncls):
    ranges=fit_ranges([t[0] for t in insts])
    X,M,y,subj=build_point_tensors(insts,ranges)
    print(f"{tag}: {len(y)} inst",flush=True)
    for mc,name in ((PointNetPP,'PointNetPP'),(DGCNNTemporal,'DGCNNTemporal')):
        for lrname,lr in LRS:
            for ep in EPS:
                accs=[]; tr_accs=[]
                for fi,te_s in enumerate(folds):
                    te=np.isin(subj,list(te_s)); tr=~te; te_idx=np.where(te)[0]
                    for sd in SEEDS:
                        a,yt,yp,tra=train_eval_set_preds_tr(mc,X[tr],M[tr],y[tr],X[te],M[te],y[te],ncls,6,epochs=ep,lr=lr,seed=sd)
                        accs.append(a); tr_accs.append(tra)
                        preds[f"{tag}|{name}|lr{lrname}|ep{ep}|f{fi}|s{sd}"]=np.stack([te_idx,yt,yp])
                key=f"{tag}|{name}|lr{lrname}|ep{ep}"
                results[key]={'mean':float(np.mean(accs))*100,'std':float(np.std(accs))*100,
                              'min_train_acc':float(min(tr_accs)),'underfit':bool(min(tr_accs)<0.95)}
                r=results[key]
                print(f"  {key:44s}: {r['mean']:6.2f}% (+-{r['std']:.1f}){'  UNDERFIT' if r['underfit'] else ''}",flush=True)

if __name__=='__main__':
    mh=mhomeges_instances(os.path.join(DATA,'mhomeges_full'))
    run_set("mHomeGes",mh,kfold(np.array([t[2] for t in mh]),5),10)
    inf=infineon_recs()
    run_set("Infineon",inf,kfold(np.array([t[2] for t in inf]),4),5)
    mf=mmfi_instances(os.path.join(DATA,'mmfi_extracted'))
    S2=[f"S{i:02d}" for i in (5,10,15,20,25,30,35,40)]
    run_set("MM-Fi",mf,[S2],27)
    out={"manifest":{"infineon_pkl_md5":hashlib.md5(open(os.path.join(DATA,'infineon_recs.pkl'),'rb').read()).hexdigest()},
         "protocol":{"epochs":list(EPS),"batch":64,"seeds":list(SEEDS),"lrs":[l for l,_ in LRS],"aug":"none",
                     "deviations":["BN track_running_stats=False (batch-stats at eval; fixes running-stats corruption, see mhfix_diag2)",
                                    "masked FPS/ball-query/kNN (padding excluded)","compact depths vs published"]},
         "results":results}
    json.dump(out,open(os.path.join(DOCS,'baselines1_fix2.json'),'w'),indent=1)
    np.savez_compressed(os.path.join(DOCS,'baselines1_fix2_preds.npz'),**preds)
    print("wrote docs/baselines1_fix2.json",flush=True)
