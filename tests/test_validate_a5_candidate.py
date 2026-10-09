import numpy as np
from scripts.validate_a5_candidate import rank01, per_target_auc

def test_rank01_monotonic():
    x=np.array([[3.,1.],[1.,3.],[2.,2.]])
    r=rank01(x)
    assert np.allclose(r[:,0],[1,.0,.5])
    assert np.allclose(r[:,1],[0,1,.5])

def test_auc_per_target():
    y=np.array([[0,1],[1,0],[0,1],[1,0]],float)
    p=np.array([[.1,.9],[.8,.2],[.2,.8],[.9,.1]],float)
    a=per_target_auc(y,p)
    assert np.allclose(a,[1,1])
