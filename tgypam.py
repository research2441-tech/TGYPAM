"""TGYPAM computational operationalization; see README for equation caveats."""
import numpy as np
import torch
from torch import nn

EPS=1e-8

def normalize_support(x, minimum, maximum):
    """Clip a physically meaningful interval to [0,1]; bounds calibrated on training only."""
    if maximum<=minimum: raise ValueError('maximum must exceed minimum')
    return np.clip((np.asarray(x,dtype=float)-minimum)/(maximum-minimum),0,1)

def factor_geomean(soil, fluid, environment, climate):
    """Multiplicative TCGA/TCYA factor: zero on any essential factor means zero."""
    x=np.array([soil,fluid,environment,climate],dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x<0)|(x>1)): raise ValueError('All support factors must be finite in [0,1]')
    return np.prod(x,axis=0)**.25

def tcga(scgsf,fcgsf,ecgsf,ccgsf): return factor_geomean(scgsf,fcgsf,ecgsf,ccgsf)
def tcya(scysf,fcysf,ecysf,ccysf): return factor_geomean(scysf,fcysf,ecysf,ccysf)
def tcpa(yield_t_ha,price_usd_t,cost_usd_ha):
    """Net profit, USD/ha, not a percentage. Inputs must describe the same period."""
    return np.asarray(yield_t_ha)*np.asarray(price_usd_t)-np.asarray(cost_usd_ha)

def pss(growth_support,yield_support,profit_support,weights=(1/3,1/3,1/3)):
    """Weighted multiplicative plant selection score; supports must be in [0,1]."""
    w=np.asarray(weights,dtype=float)
    if w.shape!=(3,) or (w<0).any() or not np.isclose(w.sum(),1): raise ValueError('weights must be three nonnegative values summing to one')
    x=np.stack(np.broadcast_arrays(growth_support,yield_support,profit_support),axis=-1).astype(float)
    if not np.isfinite(x).all() or (x<0).any() or (x>1).any(): raise ValueError('PSS inputs must be [0,1]')
    return np.prod(x**w,axis=-1)

def fluid_factor(historical_water_mm):
    """Historical observed water-use summary in mm; NOT an irrigation prescription."""
    x=np.asarray(historical_water_mm,dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')

def fertilizer_requirement_factor(historical_fertilizer_kg_ha):
    """Historical fertilizer-use summary in kg/ha; NOT nutrient need."""
    x=np.asarray(historical_fertilizer_kg_ha,dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')

class TGYPAM(nn.Module):
    """Shared time series GRU with TCGA, TCYA, TCPA supervised heads.

    This is a specified, testable architecture; original paper does not identify
    a reproducible layer-by-layer deep neural network specification.
    """
    def __init__(self,n_features,n_crops,hidden_dim=48,layers=1,dropout=.1):
        super().__init__()
        self.gru=nn.GRU(n_features,hidden_dim,layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.crop_embed=nn.Embedding(n_crops,min(12,hidden_dim))
        h=hidden_dim+min(12,hidden_dim)
        self.shared=nn.Sequential(nn.Linear(h,hidden_dim),nn.ReLU(),nn.Dropout(dropout))
        self.growth_head=nn.Linear(hidden_dim,1)
        self.yield_head=nn.Linear(hidden_dim,1)
        self.profit_head=nn.Linear(hidden_dim,1)
    def forward(self,x,crop_ids):
        _,last=self.gru(x)
        z=self.shared(torch.cat([last[-1],self.crop_embed(crop_ids)],dim=-1))
        return torch.cat([self.growth_head(z),self.yield_head(z),self.profit_head(z)],dim=-1)
