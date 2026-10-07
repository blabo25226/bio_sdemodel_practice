"""Version-pinned checkpoint adapter and direct, batched SDE function access."""
from __future__ import annotations
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import torchsde
from scdiffeq.core.lightning_models import LightningSDE_FixedPotential_RegularizedVelocityRatio


class TeacherSDE(torch.nn.Module):
    """Restored SDE; f: (batch,d), g: (batch,d,m), with potential autograd preserved."""
    def __init__(self, checkpoint: Path, device: str = 'cpu') -> None:
        super().__init__()
        if (version('scdiffeq'), version('neural-diffeqs')) != ('1.1.4', '0.4.1'):
            raise RuntimeError('Checkpoint adapter requires a new audit after version changes')
        self.lightning_model = LightningSDE_FixedPotential_RegularizedVelocityRatio.load_from_checkpoint(
            checkpoint, map_location=device, weights_only=False)
        self.lightning_model.to(device).eval()
        # Version-specific access is confined to this wrapper.
        self.sde = self.lightning_model.DiffEq
        self.noise_type = self.sde.noise_type
        self.sde_type = self.sde.sde_type
        self.brownian_dim = self.sde._brownian_dim
        self.dimension = self.lightning_model.hparams.latent_dim

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate teacher drift on (batch,d), keeping autograd for potential derivatives."""
        with torch.enable_grad():
            return self.sde.f(t, x.detach().clone()).detach()

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate stochastic coefficient G on (batch,d), returning (batch,d,m)."""
        with torch.no_grad():
            return self.sde.g(t, x).detach()

    def evaluate(self, x: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate states (n,d) in batches; return drift (n,d), G (n,d,m)."""
        parameter = next(self.sde.parameters())
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('Invalid teacher states')
        f, g = [], []
        for batch in range(0, len(x), batch_size):
            states = torch.as_tensor(x[batch:batch+batch_size], dtype=parameter.dtype,
                                     device=parameter.device)
            t = states.new_tensor(2.)
            f.append(self.f(t, states).cpu().numpy())
            g.append(self.g(t, states).cpu().numpy())
        return np.concatenate(f), np.concatenate(g)

    def verify_public_fields(self, x: np.ndarray) -> dict[str, object]:
        """Cross-check direct f/G and public AnnData values for states (n,d), m=1."""
        if self.brownian_dim!=1:
            raise ValueError('Re-audit public scalar characterization for m != 1')
        from types import SimpleNamespace
        import anndata as ad
        import scdiffeq as sdq
        a=ad.AnnData(x.copy());a.obsm['X_pca']=x.copy()
        public_adapter=SimpleNamespace(DiffEq=self.lightning_model)
        device=next(self.sde.parameters()).device
        with torch.enable_grad():
            sdq.tl.drift(a,model=public_adapter,device=device)
            sdq.tl.diffusion(a,model=public_adapter,device=device)
        f,g=self.evaluate(x)
        np.testing.assert_allclose(a.obsm['X_drift'],f,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(a.obsm['X_diffusion'],g.squeeze(-1),rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(a.obs['drift'],np.linalg.norm(f,axis=1),rtol=1e-6)
        np.testing.assert_allclose(a.obs['diffusion'],np.linalg.norm(g.squeeze(-1),axis=1),rtol=1e-6)
        return {'public_direct_fields_verified':True,'scalar_norms_verified':True,
                'scalar_drift_obs':'L2 norm of f per cell',
                'scalar_diffusion_obs':'L2 norm of squeezed G per cell for m=1; not G or D'}


def simulate_sde(sde: torch.nn.Module, initial: np.ndarray, times: np.ndarray,
                 seed: int, dt: float = 0.05) -> np.ndarray:
    """Itô Euler-Maruyama using torchsde; (n,d) → (time,n,d), explicit Brownian seed."""
    parameter = next(sde.parameters())
    x0 = torch.as_tensor(initial, dtype=parameter.dtype, device=parameter.device)
    t = torch.as_tensor(times, dtype=parameter.dtype, device=parameter.device)
    bm = torchsde.BrownianInterval(t0=float(t[0]), t1=float(t[-1]),
                                  size=(len(x0), sde.brownian_dim),
                                  dtype=x0.dtype, device=x0.device, entropy=seed,
                                  levy_area_approximation='none')
    result = torchsde.sdeint(sde, x0, t, bm=bm, method='euler', dt=dt)
    return result.detach().cpu().numpy()
