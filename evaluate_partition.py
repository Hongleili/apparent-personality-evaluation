"""Evaluate saved predictions against a licensed test manifest; never loads pickle.
Example: python evaluate_partition.py --manifest labeled_test.csv --predictions results/ --output audit/ --seed 20261123
Supply multiple prediction folders for artifacts stored separately. Existing output folders are refused.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
TRAITS=['openness','conscientiousness','extraversion','agreeableness','neuroticism']
def read(p):return list(csv.DictReader(Path(p).open()))
def write(p,rs):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def evaluate(manifest,prediction_folders,output,seed):
 m=read(manifest);names=[r['filename'] for r in m];assert len(names)==len(set(names));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 y=np.array([[float(r[t]) for t in TRAITS] for r in m],dtype=np.float32).astype(float)
 assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
 summary=[];traits=[];errors={}
 files=sorted(p for root in prediction_folders for p in Path(root).glob('*_test_predictions.csv'))
 assert files,'No test prediction files'
 for p in files:
  model=p.name.removesuffix('_test_predictions.csv');assert model not in errors,'Duplicate model file'
  rr=read(p);assert [r['filename'] for r in rr]==names,('Filename/order mismatch',p)
  x=np.array([[float(r[t]) for t in TRAITS] for r in rr]);assert x.shape==y.shape and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
  err=np.abs(x-y);errors[model]=err.mean(1);pcs=[]
  for j,t in enumerate(TRAITS):
   a=y[:,j];b=x[:,j];ya=a>=.5;pb=b>=.5;tp=int((ya&pb).sum());tn=int((~ya&~pb).sum());fp=int((~ya&pb).sum());fn=int((ya&~pb).sum());pc=float(np.corrcoef(a,b)[0,1]) if np.ptp(a)>0 and np.ptp(b)>0 else None;pcs.append(pc)
   traits.append({'model':model,'trait':t,'n':len(y),'MAE':float(err[:,j].mean()),'PCC':pc,'FPR_percent':100*fp/(fp+tn) if fp+tn else None,'FNR_percent':100*fn/(fn+tp) if fn+tp else None,'TP':tp,'TN':tn,'FP':fp,'FN':fn})
  summary.append({'model':model,'n':len(y),'MAE':float(err.mean()),'MSE':float(((x-y)**2).mean()),'PCC':float(np.mean(pcs)) if all(p is not None for p in pcs) else None})
 write(out/'metrics.csv',summary);write(out/'traits.csv',traits)
 required=['ridge','training_median']+[f'cnn_lstm_seed{s}' for s in range(42,47)]
 if all(k in errors for k in required):
  diffs=np.stack([errors['ridge']-errors['training_median'],np.mean([errors[k] for k in required[2:]],axis=0)-errors['ridge']],axis=1)
  groups,gi=np.unique([r['source_prefix'] for r in m],return_inverse=True);sizes=np.bincount(gi);sums=np.stack([np.bincount(gi,weights=diffs[:,j]) for j in range(2)],axis=1);gen=np.random.default_rng(seed);draws=[]
  for _ in range(20):
   w=gen.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=500);draws.append((w@sums)/(w@sizes)[:,None])
  bs=np.concatenate(draws);results=[]
  for j,name in enumerate(['ridge_minus_training_median','cnn_lstm_mean_error_minus_ridge']):
   ci=np.quantile(bs[:,j],[.025,.975,.0025,.9975]);results.append({'contrast':name,'difference_MAE':float(diffs[:,j].mean()),'CI95_lower':float(ci[0]),'CI95_upper':float(ci[1]),'CI99_5_lower':float(ci[2]),'CI99_5_upper':float(ci[3]),'n_source_groups':len(groups),'bootstrap_replicates':10000})
  write(out/'cluster_contrasts.csv',results)
 (out/'scope.json').write_text(json.dumps({'bootstrap_seed':seed,'label_dtype_before_evaluation':'float32, then float64','prediction_dtype':'float64','interval_family':'99.5% marginal intervals for 10 comparisons across all five source-separated tests; not a global study-wide family','scope':'Exploratory; fixed fitted models; not a causal cross-condition effect; source groups are recording proxies, not verified identities'},indent=2))
 return summary
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--predictions',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,required=True);args=a.parse_args();evaluate(args.manifest,args.predictions,args.output,args.seed)
