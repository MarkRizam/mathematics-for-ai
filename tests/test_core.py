import numpy as np
import pytest
from math_for_ai import Value, gradient, lu, optimize, qr, solve

@pytest.mark.parametrize('seed', [11,22,33])
def test_lu(seed):
    a = np.random.default_rng(seed).normal(size=(5,5))
    p,l,u = lu(a)
    np.testing.assert_allclose(p@a,l@u,atol=1e-12)
    np.testing.assert_allclose(np.diag(l),1)
    np.testing.assert_allclose(np.tril(u,-1),0)
    np.testing.assert_allclose(solve(a,np.ones(5)),np.linalg.solve(a,np.ones(5)),atol=1e-12)

@pytest.mark.parametrize('seed', [11,22,33])
def test_qr(seed):
    a = np.random.default_rng(seed).normal(size=(8,4))
    q,r = qr(a)
    np.testing.assert_allclose(q@r,a,atol=1e-12)
    np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-12)

@pytest.mark.parametrize('x', [[1.,2.],[-3.,0.],[.2,-.5]])
def test_gradient(x):
    x=np.asarray(x)
    np.testing.assert_allclose(gradient(lambda v: np.sum(v**2),x),2*x,atol=1e-8)

@pytest.mark.parametrize('method', ['gd','momentum','adam'])
def test_optimization(method):
    x,h=optimize(lambda x: x@x,lambda x:2*x,[2.,-1.],method=method,rate=.03,steps=400)
    assert h[-1]<1e-6
    assert len(h)==401

@pytest.mark.parametrize('x,y', [(1.,2.),(-.3,.5),(0.,0.)])
def test_autodiff(x,y):
    a,b=Value(x),Value(y)
    z=(a*b+a).tanh()+a*a
    z.backward()
    t=1-np.tanh(x*y+x)**2
    np.testing.assert_allclose([a.grad,b.grad],[t*(y+1)+2*x,t*x],atol=1e-12)
    z.backward()
    np.testing.assert_allclose([a.grad,b.grad],[t*(y+1)+2*x,t*x],atol=1e-12)

@pytest.mark.parametrize('f', [lambda:lu([[1,2],[2,4]]),lambda:qr([[1,2],[2,4]]),lambda:lu([[1,2,3]]),lambda:qr([[1,2,3]]),lambda:gradient(sum,[1],0),lambda:gradient(sum,[1],np.nan),lambda:optimize(sum,lambda x:x,[1],steps=True),lambda:optimize(sum,lambda x:x,[1],rate=np.inf),lambda:Value(np.nan),lambda:solve([[1,0],[0,1]],[1]),lambda:lu([[np.nan]])])
def test_invalid(f):
    with pytest.raises(ValueError): f()
