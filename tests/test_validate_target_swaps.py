import numpy as np
from scripts.validate_target_swaps import holm_adjust, apply_swaps

def test_holm_monotone():
    p=np.array([.001,.02,.2])
    h=holm_adjust(p)
    assert np.all(np.diff(h[np.argsort(p)])>=-1e-12)
    assert h[0] <= h[1] <= h[2]

def test_apply_swaps():
    a=np.zeros((3,4)); b=np.ones((3,4))
    out=apply_swaps(a,b,np.array([True,False,True,False]))
    assert np.all(out[:,0]==1) and np.all(out[:,2]==1)
    assert np.all(out[:,1]==0) and np.all(out[:,3]==0)
