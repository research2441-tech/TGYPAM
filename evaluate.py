"""Independent held-out test scores, in original physical target units."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import yaml,torch
from train import run
from tgypam import pss,normalize_support

def evaluate(net,prepared,weights=(1/3,1/3,1/3),seed=42):
    net.eval();x,y,c=prepared.test
    with torch.no_grad():pred=net(torch.from_numpy(x),torch.from_numpy(c)).numpy()
    yt=prepared.target_scaler.inverse_transform(y)
    yp=prepared.target_scaler.inverse_transform(pred)
    results={}
    for i,name in enumerate(['growth','yield_t_ha','profit_usd_ha']):
        observed=yt[:,i];prediction=yp[:,i]
        results[name]={'mae':float(mean_absolute_error(observed,prediction)), 'rmse':float(np.sqrt(mean_squared_error(observed,prediction))), 'r2':float(r2_score(observed,prediction)) if len(observed)>1 else None}
        # Percentile bootstrapping samples the entire held-out observation level; not field-cluster CIs.
        rng=np.random.default_rng(seed+i);errors=np.abs(observed-prediction);boot=[np.mean(rng.choice(errors,len(errors),replace=True)) for _ in range(200)]
        results[name]['mae_bootstrap_95pct_ci']=[float(v) for v in np.percentile(boot,[2.5,97.5])]
    # Crop-wise descriptive composite: fit calibration bounds using training labels only.
    ytrain=prepared.target_scaler.inverse_transform(prepared.train[1]);mins=ytrain.min(axis=0);maxs=ytrain.max(axis=0)
    support=np.column_stack([normalize_support(yp[:,i],mins[i],maxs[i]) for i in range(3)])
    scores=pss(support[:,0],support[:,1],support[:,2],weights=weights)
    results['crop_score_summary']={crop:{'n':int((c==i).sum()),'mean_pss':float(scores[c==i].mean())} for i,crop in enumerate(prepared.crops) if (c==i).any()}
    results['important_notes']=['Scores do not reproduce manuscript 99/97/98 without source data and original metric definitions.', 'PSS ranking across crops is exploratory; crop-counterfactual suitability is not validated.','Bootstrap treats sequences independently, which can underestimate uncertainty for field-correlated samples.']
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());net,data,hist=run(cfg)
    metrics=evaluate(net,data,cfg['pss_weights'],cfg['seed']);out=cfg['output_prefix']+'_metrics.json'
    Path(out).write_text(json.dumps(metrics,indent=2));print(f'Wrote {out}')
