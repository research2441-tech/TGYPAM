"""Strict chronological evaluation with fit-on-training preprocessing only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

@dataclass
class Prepared:
    train: tuple
    val: tuple
    test: tuple
    features: list
    crops: list
    target_scaler: object
    metadata: dict


def prepare(csv_path, cfg):
    df=pd.read_csv(csv_path)
    cols=[cfg['time_column'],cfg['entity_column'],cfg['crop_column']]+cfg['features']+cfg['targets']
    absent=[c for c in cols if c not in df]
    if absent: raise ValueError(f'Missing required columns: {absent}')
    df=df.copy()
    time_col=cfg['time_column'];ent=cfg['entity_column'];crop=cfg['crop_column']
    df[time_col]=pd.to_datetime(df[time_col],errors='coerce')
    df=df.dropna(subset=[time_col,ent,crop]+cfg['targets'])
    if len(df)==0: raise ValueError('No valid labeled observations')
    if df.duplicated([ent,time_col]).any(): raise ValueError('Duplicate (entity, timestamp) pairs: fix source data first')
    for col in cfg['features']+cfg['targets']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=cfg['targets']).sort_values([ent,time_col]).reset_index(drop=True)
    # Distinct timestamps ensure no calendar time appears in more than one partition.
    dates=np.array(sorted(df[time_col].unique()))
    if len(dates)<8: raise ValueError('At least eight unique timestamps required for chronological validation')
    a=int(len(dates)*cfg['split_train']);b=int(len(dates)*(cfg['split_train']+cfg['split_val']))
    if not (0<a<b<len(dates)): raise ValueError('Invalid temporal split ratios')
    boundaries=[dates[a-1],dates[b-1]]
    split=np.where(df[time_col]<=boundaries[0],0,np.where(df[time_col]<=boundaries[1],1,2))
    train_rows=df.loc[split==0]
    if len(train_rows)==0: raise ValueError('Empty train split')
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    imputer.fit(train_rows[cfg['features']])
    scaler=StandardScaler().fit(imputer.transform(train_rows[cfg['features']]))
    yscaler=StandardScaler().fit(train_rows[cfg['targets']])
    X=scaler.transform(imputer.transform(df[cfg['features']])).astype('float32')
    Y=yscaler.transform(df[cfg['targets']]).astype('float32')
    # No target, harvest outcome or actual future observation appears among inputs.
    if set(cfg['features']) & set(cfg['targets']): raise ValueError('Targets cannot also be features')
    crops=sorted(df[crop].unique().tolist()); crop_ids={v:i for i,v in enumerate(crops)}
    seq_len=int(cfg['sequence_length']);result=[[],[],[]];ys=[[],[],[]];cs=[[],[],[]];ts=[[],[],[]]
    for _,g in df.groupby(ent,sort=False):
        ids=g.index.to_numpy()
        for j in range(seq_len-1,len(ids)):
            window=ids[j-seq_len+1:j+1]
            # Prevent windows straddling partitions; prevents train-period history being injected into validation.
            if len(set(split[window]))!=1: continue
            k=int(split[window[-1]])
            result[k].append(X[window]);ys[k].append(Y[window[-1]])
            cs[k].append(crop_ids[df.at[int(window[-1]),crop]])
            ts[k].append(str(df.at[int(window[-1]),time_col]))
    if any(len(z)==0 for z in result): raise ValueError('No sequences in at least one split; increase observation horizon or reduce sequence_length')
    bundles=tuple((np.stack(result[k]),np.stack(ys[k]),np.array(cs[k],dtype='int64')) for k in range(3))
    meta={'split_dates':[str(pd.Timestamp(x).date()) for x in boundaries], 'sequences':[len(z) for z in result], 'rows':len(df),'last_sample_timestamps':ts,'missing_train':train_rows[cfg['features']].isna().sum().to_dict()}
    return Prepared(*bundles,list(cfg['features']),crops,yscaler,meta)
