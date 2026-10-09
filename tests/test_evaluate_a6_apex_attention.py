import numpy as np
from scripts.evaluate_a6_apex_attention import softmax,layernorm

def test_softmax_rows():
    x=np.array([[1.,2.,3.],[3.,2.,1.]])
    s=softmax(x,axis=1)
    assert np.allclose(s.sum(1),1)

def test_layernorm_finite():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(2,3,4))
    y=layernorm(x,np.ones(4),np.zeros(4))
    assert np.isfinite(y).all()
    assert np.allclose(y.mean(-1),0,atol=1e-6)
