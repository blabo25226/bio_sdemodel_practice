"""Polynomial-only direct function distillation with PySINDy STLSQ."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
import pysindy as ps
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch


def errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Function errors for (n, outputs), normalized by RMS target (not its variance)."""
    residual = prediction-truth
    norm = np.linalg.norm(truth,axis=1)
    cosine = np.sum(truth*prediction,axis=1)/(norm*np.linalg.norm(prediction,axis=1)+1e-15)
    return {'mse':float(np.mean(residual**2)),
            'nrmse':float(np.sqrt(np.mean(residual**2)/max(np.mean(truth**2),1e-15))),
            'cosine_mean':float(np.mean(cosine)),
            'error_norm_q95':float(np.quantile(np.linalg.norm(residual,axis=1),.95)),
            'error_norm_max':float(np.max(np.linalg.norm(residual,axis=1)))}


@dataclass
class PolynomialField:
    """Sparse polynomial in standardized PCs; output remains original PC units."""
    powers: np.ndarray
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Input (n,d) → output (n,k); only constant, powers and monomial interactions."""
        if x.ndim!=2 or x.shape[1]!=len(self.mean):raise ValueError('Invalid state shape')
        z=(x-self.mean)/self.scale
        theta=np.ones((len(x),len(self.powers)),dtype=np.float64)
        for j,power in enumerate(self.powers):
            active=np.flatnonzero(power)
            if len(active):theta[:,j]=np.prod(z[:,active]**power[active],axis=1)
        return theta @ self.coefficients.T

    def save(self, path: Path) -> None:
        """Save transform, polynomial powers and coefficients without pickle."""
        np.savez_compressed(path,powers=self.powers,coefficients=self.coefficients,
                            mean=self.mean,scale=self.scale,metadata=json.dumps(self.metadata))
        path.with_suffix('.json').write_text(json.dumps({'powers':self.powers.tolist(),
            'coefficients':self.coefficients.tolist(),'mean':self.mean.tolist(),
            'scale':self.scale.tolist(),'metadata':self.metadata},indent=2))

    @classmethod
    def load(cls, path: Path) -> 'PolynomialField':
        """Restore a function without touching validation/test data."""
        if not path.exists() and path.with_suffix('.json').exists():
            a=json.loads(path.with_suffix('.json').read_text())
            return cls(np.asarray(a['powers'],dtype=int),np.asarray(a['coefficients']),
                       np.asarray(a['mean']),np.asarray(a['scale']),a['metadata'])
        a=np.load(path,allow_pickle=False)
        return cls(a['powers'],a['coefficients'],a['mean'],a['scale'],json.loads(str(a['metadata'])))


def fit_field(x: np.ndarray, y: np.ndarray, degree: int, threshold: float,
              alpha: float, reference: PolynomialField | None = None, unbias: bool = True) -> PolynomialField:
    """Fit states (n,d) → targets (n,k); transforms use only the supplied training set."""
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training data')
    scaler=StandardScaler().fit(x)
    if reference is not None:
        scaler.mean_=reference.mean
        scaler.scale_=reference.scale
    library=ps.PolynomialLibrary(degree=degree,include_interaction=True,include_bias=True)
    theta=np.asarray(library.fit_transform(scaler.transform(x)),dtype=np.float64)
    feature_rms=np.sqrt(np.mean(theta**2,axis=0));feature_rms[feature_rms<1e-12]=1.
    target_rms=np.sqrt(np.mean(y**2,axis=0));target_rms[target_rms<1e-12]=1.
    if reference is not None:
        feature_rms=np.asarray(reference.metadata['feature_rms'])
        target_rms=np.asarray(reference.metadata['target_rms'])
    optimizer=ps.STLSQ(threshold=threshold,alpha=alpha,max_iter=20,
                       normalize_columns=False,unbias=unbias)
    with threadpool_limits(limits=4):optimizer.fit(theta/feature_rms,y/target_rms)
    coefficients=optimizer.coef_*target_rms[:,None]/feature_rms[None,:]
    meta={'degree':degree,'threshold':threshold,'alpha':alpha,'library_size':len(feature_rms),
          'active_terms':int(np.count_nonzero(coefficients)),
          'threshold_units':'dimensionless coefficient after training RMS scaling of library and targets',
          'coordinates':'z_i=(PC_i-training_mean_i)/training_std_i; output in original teacher-field units (drift or Brownian coefficient)',
          'feature_rms':feature_rms.tolist(),'target_rms':target_rms.tolist(),
          'unbias':unbias,'optimizer':'pysindy.STLSQ','pysindy_version':ps.__version__}
    return PolynomialField(library.powers_,coefficients,scaler.mean_,scaler.scale_,meta)


class SymbolicSDE(torch.nn.Module):
    """General-noise Itô SDE: polynomial f and G; D=GG^T is PSD by construction."""
    noise_type='general'
    sde_type='ito'
    def __init__(self, drift: PolynomialField, diffusion: PolynomialField,
                 device: str='cpu', dtype: torch.dtype=torch.float32) -> None:
        super().__init__()
        if drift.coefficients.shape[0]!=len(drift.mean):
            raise ValueError('Drift output dimension must equal state dimension')
        if diffusion.coefficients.shape[0]%len(drift.mean) or len(diffusion.mean)!=len(drift.mean):
            raise ValueError('Diffusion must have d*m outputs on the same state space')
        self.brownian_dim=diffusion.coefficients.shape[0]//len(drift.mean)
        self.dimension=len(drift.mean)
        for prefix,field in [('f',drift),('g',diffusion)]:
            for name,value in [('powers',field.powers),('coefficients',field.coefficients),
                               ('mean',field.mean),('scale',field.scale)]:
                tensor=torch.as_tensor(value,device=device,dtype=torch.int64 if name=='powers' else dtype)
                self.register_buffer(f'{prefix}_{name}',tensor)
        # parameter exposes device/dtype consistently to the integration adapter.
        self.anchor=torch.nn.Parameter(torch.zeros((),device=device,dtype=dtype),requires_grad=False)

    def _field(self,prefix: str,x: torch.Tensor) -> torch.Tensor:
        z=(x-getattr(self,prefix+'_mean'))/getattr(self,prefix+'_scale')
        powers=getattr(self,prefix+'_powers')
        # Degree <=2 is evaluated without a (batch,library,d) intermediate.
        theta=torch.ones((len(x),len(powers)),device=x.device,dtype=x.dtype)
        for degree in range(1,int(powers.sum(1).max())+1):
            mask=powers.sum(1)==degree
            selected=powers[mask]
            if not len(selected):continue
            indices=torch.repeat_interleave(torch.arange(self.dimension,device=x.device).expand(len(selected),-1).flatten(),selected.flatten()).reshape(len(selected),degree)
            theta[:,mask]=z[:,indices].prod(-1)
        return theta @ getattr(self,prefix+'_coefficients').T

    def f(self,t: torch.Tensor,x: torch.Tensor) -> torch.Tensor:
        """States (batch,d) → drift (batch,d)."""
        return self._field('f',x)

    def g(self,t: torch.Tensor,x: torch.Tensor) -> torch.Tensor:
        """States (batch,d) → G (batch,d,m), preserving teacher Brownian structure."""
        return self._field('g',x).reshape(len(x),self.dimension,self.brownian_dim)
