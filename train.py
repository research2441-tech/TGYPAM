"""Train using only training observations; select model by validation loss."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import yaml
from pipeline import prepare
from tgypam import TGYPAM

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def run(cfg):
    seed_all(int(cfg['seed']))
    prepared=prepare(cfg['data_path'],cfg)
    net=TGYPAM(len(prepared.features),len(prepared.crops),cfg['hidden_dim'],cfg['layers'],cfg['dropout'])
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    weights=torch.tensor(cfg['loss_weights'],dtype=torch.float32)
    def dl(bundle,shuffle):
        x,y,c=bundle
        return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(c)),batch_size=cfg['batch_size'],shuffle=shuffle)
    tr=dl(prepared.train,True);va=dl(prepared.val,False)
    best=float('inf');wait=0;hist=[];state=None
    for ep in range(cfg['epochs']):
        net.train();train_losses=[]
        for x,y,c in tr:
            opt.zero_grad(set_to_none=True);pred=net(x,c)
            loss=(((pred-y)**2)*weights).mean();loss.backward()
            nn_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1.)
            opt.step();train_losses.append(loss.item())
        net.eval();valid=[]
        with torch.no_grad():
            for x,y,c in va:valid.append(((((net(x,c)-y)**2)*weights).mean()).item())
        val=float(np.mean(valid));hist.append({'epoch':ep+1,'train_loss':float(np.mean(train_losses)),'val_loss':val})
        if val<best-1e-6:
            best=val;wait=0;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:wait+=1
        if wait>=cfg['patience']:break
    net.load_state_dict(state)
    return net,prepared,hist

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());net,prepared,hist=run(cfg)
    prefix=cfg['output_prefix'];torch.save({'model':net.state_dict(),'config':cfg,'crops':prepared.crops},prefix+'.pt')
    Path(prefix+'_history.json').write_text(json.dumps({'training':hist,'metadata':{k:v for k,v in prepared.metadata.items() if k!='last_sample_timestamps'}},indent=2))
    print(f'Saved {prefix}.pt (not manuscript performance proof)')
