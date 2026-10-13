"""Run available corpus-size experiments; never synthesize missing study data."""
import argparse,csv,json
from pathlib import Path
import pandas as pd,yaml
from train import run
from evaluate import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--allow-smaller',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());original=pd.read_csv(cfg['data_path']);sizes=cfg['benchmark_sizes'];records=[]
    for n in sizes:
        if len(original)<n:
            print(f'SKIP {n}: only {len(original)} source rows available');continue
        # Deterministic prefix in calendar order, not random hold-out mixing
        sample=original.sort_values(cfg['time_column'],kind='mergesort').iloc[:n]
        temp=Path(cfg['output_prefix']+'_subset.csv');sample.to_csv(temp,index=False)
        try:
            local={**cfg,'data_path':str(temp)};net,prepared,history=run(local)
            metrics=evaluate(net,prepared,cfg['pss_weights'],cfg['seed'])
            for label,value in metrics.items():
                if isinstance(value,dict) and 'mae' in value:records.append({'rows':n,'target':label,'mae':value['mae'],'rmse':value['rmse'],'r2':value['r2']})
        except ValueError as exc:print(f'SKIP {n}: {exc}')
        finally:temp.unlink(missing_ok=True)
    if not records:
        print('No benchmark-size experiments were possible. Run train.py/evaluate.py on a smaller sample; never claim manuscript reproduction.')
        return
    dest=cfg['output_prefix']+'_benchmark.csv'
    with open(dest,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Wrote',dest)
if __name__=='__main__':main()
