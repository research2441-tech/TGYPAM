import numpy as np
import pytest
from tgypam import tcga,tcya,tcpa,pss,fluid_factor,fertilizer_requirement_factor,TGYPAM
from generate_demo_data import generate
from pipeline import prepare
import torch,yaml

def test_factors():
    assert np.isclose(tcga(.8,.8,.8,.8),.8)
    assert tcya(1,1,0,1)==0
    assert np.isclose(tcpa(3,200,400),200)
    assert np.isclose(pss(.5,.5,.5),.5)
    with pytest.raises(ValueError):pss(1,1,1,(.6,.6,-.2))
    assert np.isclose(fluid_factor([10,20]),15)
    assert np.isclose(fertilizer_requirement_factor([40,60]),50)

def test_model():
    net=TGYPAM(14,4);res=net(torch.zeros(5,6,14),torch.zeros(5,dtype=torch.long))
    assert res.shape==(5,3)

def test_temporal_pipeline(tmp_path):
    f=tmp_path/'data.csv';generate(10,60).to_csv(f,index=False)
    cfg=yaml.safe_load(open('config.yaml'));data=prepare(f,cfg)
    assert all(len(bundle[0])>0 for bundle in [data.train,data.val,data.test])
    dates=data.metadata['last_sample_timestamps']
    assert max(dates[0])<min(dates[1])<min(dates[2])
