import numpy as np
import torch
from src.distill import fit_field,PolynomialField,SymbolicSDE
from src.teacher import simulate_sde
from src.prepare import clone_split
import pandas as pd


def test_polynomial_interactions_recovery_roundtrip(tmp_path):
    x=np.random.default_rng(4).normal(size=(400,2))
    y=np.column_stack([1+2*x[:,0]+3*x[:,0]*x[:,1],-x[:,1]**2])
    field=fit_field(x,y,degree=2,threshold=.000001,alpha=.001)
    np.testing.assert_allclose(field.predict(x),y,atol=1e-8)
    assert len(field.powers)==6
    assert np.any(np.all(field.powers==[1,1],axis=1))
    field.save(tmp_path/'model.npz')
    np.testing.assert_allclose(PolynomialField.load(tmp_path/'model.npz').predict(x),y,atol=1e-8)


def test_symbolic_tensor_covariance_and_seed():
    rng=np.random.default_rng(0);x=rng.normal(size=(200,2))
    f=fit_field(x,-.1*x,1,.001,.01)
    g=fit_field(x,np.ones_like(x)*.05,1,.001,.01)
    sde=SymbolicSDE(f,g)
    xx=torch.as_tensor(x[:4],dtype=torch.float32)
    assert sde.f(torch.tensor(0.),xx).shape==(4,2)
    gg=sde.g(torch.tensor(0.),xx)
    assert gg.shape==(4,2,1)
    assert torch.linalg.eigvalsh(gg@gg.transpose(-1,-2)).min()>-1e-6
    a=simulate_sde(sde,x[:4],np.array([0.,.2,.4]),seed=23)
    b=simulate_sde(sde,x[:4],np.array([0.,.2,.4]),seed=23)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(sde.f(torch.tensor(0.),xx).numpy(),f.predict(x[:4]),atol=1e-7)


def test_clone_disjoint_partition():
    obs=pd.DataFrame({'clone_idx':np.repeat(np.arange(50),3)})
    labels=clone_split(obs)
    assert set(labels)=={'train','validation','test'}
    for clone in obs.clone_idx.unique():assert len(set(labels[obs.clone_idx==clone]))==1


def test_teacher_potential_evaluation_inside_no_grad():
    from pathlib import Path
    import pytest
    from src.teacher import TeacherSDE
    checkpoint=Path('outputs/baseline/official/teacher.ckpt')
    if not checkpoint.exists():pytest.skip('Local trained teacher is an external artifact')
    teacher=TeacherSDE(checkpoint)
    x=np.random.default_rng(2).normal(size=(5,teacher.dimension)).astype('float32')
    with torch.no_grad():f,g=teacher.evaluate(x)
    assert f.shape==(5,teacher.dimension)
    assert g.shape==(5,teacher.dimension,teacher.brownian_dim)
    assert np.isfinite(f).all() and np.isfinite(g).all()
    assert f.dtype==x.dtype and g.dtype==x.dtype


def test_failure_accounting_keeps_invalid_and_past_exploded_paths():
    from src.validate import failure_masks
    trajectory=np.zeros((3,4,2),dtype='float32')
    trajectory[1,1,0]=np.nan
    trajectory[1,2,0]=100.
    mask,norms=failure_masks(trajectory,radius=1.)
    assert not mask[0].any()
    assert mask[1].tolist()==[False,True,True,False]
    assert mask[2].tolist()==[False,True,True,False]
    assert norms.shape==(3,4)
