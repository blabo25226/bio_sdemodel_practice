"""Measured official scDiffEq reproduction, with a smoke stage before full training."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import lightning as pl
import scdiffeq as sdq
from .data import quickstart_subset


def write_status(path: Path, **values: object) -> None:
    """Atomically record run state for monitoring from another process."""
    values['recorded_at'] = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(values, indent=2, default=str))
    temp.replace(path)


def inspect_fields(model: object, output_dir: Path) -> dict:
    """Probe actual SDE on states (batch,d), saving exact f/G/D shapes and source."""
    from importlib.metadata import version
    if (version('scdiffeq'), version('neural-diffeqs')) != ('1.1.4', '0.4.1'):
        raise RuntimeError('Re-audit the teacher adapter after package version changes')
    sde = model.DiffEq.DiffEq  # scdiffeq==1.1.4 / neural-diffeqs==0.4.1 adapter boundary
    parameter = next(sde.parameters())
    states = torch.as_tensor(np.asarray(model.adata.obsm['X_pca'][:8]),
                             dtype=parameter.dtype, device=parameter.device)
    t = states.new_tensor(float(model.adata.obs['Time point'].min()))
    sde.eval()
    # Potential drift can require autograd even for evaluation.
    f, g = sde.f(t, states), sde.g(t, states)
    assert f.shape == states.shape and torch.isfinite(f).all()
    assert torch.isfinite(g).all()
    if sde.noise_type in ('general', 'scalar'):
        assert g.ndim == 3 and g.shape[:2] == states.shape
        covariance = g @ g.transpose(-1, -2)
    elif sde.noise_type == 'diagonal':
        assert g.shape == states.shape
        covariance = torch.diag_embed(g.square())
    else:
        raise ValueError(f'Unsupported noise_type: {sde.noise_type}')
    import inspect
    result = {'lightning_class': str(type(model.DiffEq)), 'sde_class': str(type(sde)),
              'sde_type': sde.sde_type, 'noise_type': sde.noise_type,
              'brownian_dim': getattr(sde, '_brownian_dim', None),
              'state_shape': list(states.shape), 'drift_shape': list(f.shape),
              'diffusion_shape': list(g.shape), 'covariance_shape': list(covariance.shape),
              'dtype': str(states.dtype), 'device': str(states.device),
              'f_source': inspect.getsource(sde.f), 'g_source': inspect.getsource(sde.g),
              'time_probe_drift_max_abs_difference': float((sde.f(t+1, states)-f).abs().max().detach()),
              'time_probe_diffusion_max_abs_difference': float((sde.g(t+1, states)-g).abs().max().detach())}
    model.drift(device=parameter.device)
    model.diffusion(device=parameter.device)
    for key, target in [('X_drift', f), ('X_diffusion', g)]:
        stored = torch.as_tensor(np.asarray(model.adata.obsm[key][:8]), device=parameter.device)
        expected = target.detach().reshape(stored.shape)
        torch.testing.assert_close(stored, expected, rtol=1e-4, atol=1e-5)
    result['public_field_shapes'] = {k: list(model.adata.obsm[k].shape)
                                      for k in ('X_drift', 'X_diffusion')}
    result['public_api_matches_direct'] = True
    (output_dir / 'semantics.json').write_text(json.dumps(result, indent=2))
    return result


def run_stage(adata: object, epochs: int, seed: int, output_dir: Path, name: str) -> dict:
    """Fit one fresh teacher; record split membership, losses, GPU memory and simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed, workers=True)
    model = sdq.scDiffEq(adata.copy(), seed=seed, latent_dim=adata.obsm['X_pca'].shape[1],
                        time_key='Time point', working_dir=str(output_dir), name=name)
    columns = [c for c in ('source_cell_id', 'clone_idx', 'Time point', 'train', 'val',
                           'test', 'fit_train', 'fit_val') if c in model.adata.obs]
    model.adata.obs[columns].to_csv(output_dir / 'split_membership.csv')
    splits = {}
    if 'clone_idx' in model.adata.obs and all(c in model.adata.obs for c in ('fit_train', 'fit_val')):
        obs = model.adata.obs
        train_clones = set(obs.loc[obs.fit_train.astype(bool), 'clone_idx'].dropna())
        val_clones = set(obs.loc[obs.fit_val.astype(bool), 'clone_idx'].dropna())
        splits['train_validation_clone_overlap'] = len(train_clones & val_clones)
    splits['interpretation'] = 'Official cell-level reproduction; not an independent clone-held-out biological evaluation.'
    (output_dir / 'split_audit.json').write_text(json.dumps(splits, indent=2))
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    model.fit(train_epochs=epochs, pretrain_epochs=0, devices=[0], accelerator='gpu',
              deterministic=True, keep_ckpts=1, ckpt_frequency=100, print_every=10)
    elapsed = time.perf_counter() - start
    model.trainer.save_checkpoint(output_dir / 'teacher.ckpt')
    semantics = inspect_fields(model, output_dir)
    parameter = next(model.DiffEq.parameters())
    initial = model.adata.obs.index[model.adata.obs['Time point'].eq(
        model.adata.obs['Time point'].min())][:3]
    if len(initial) == 0:
        raise ValueError('No initial states')
    pl.seed_everything(seed, workers=True)
    sim = sdq.tl.simulate(model.adata, idx=initial, N=16, diffeq=model.DiffEq,
                         time_key='Time point', dt=0.1, device=parameter.device)
    assert np.isfinite(np.asarray(sim.X)).all()
    pl.seed_everything(seed, workers=True)
    repeated = sdq.tl.simulate(model.adata, idx=initial, N=16, diffeq=model.DiffEq,
                              time_key='Time point', dt=0.1, device=parameter.device)
    np.testing.assert_allclose(np.asarray(sim.X), np.asarray(repeated.X), rtol=1e-5, atol=1e-6)
    metrics = {'seed': seed, 'epochs': epochs, 'seconds': elapsed,
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(0),
               'simulation_shape': list(sim.shape), 'same_seed_simulation_verified': True,
               'semantics': semantics,
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (output_dir / 'runtime.json').write_text(json.dumps(metrics, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    loss = model.metrics
    loss.to_csv(output_dir / 'metrics_export.csv', index=False)
    fig, ax = plt.subplots()
    for col in loss.columns:
        if 'loss' in col and ('train' in col or 'val' in col):
            clean = loss[['epoch', col]].dropna()
            if len(clean): ax.plot(clean.epoch, clean[col], label=col)
    ax.set(xlabel='Epoch', ylabel='Loss', title=name)
    ax.legend(fontsize=6)
    fig.savefig(output_dir / 'loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return metrics


def main() -> None:
    """Run smoke then 1500-epoch official reproduction, stopping on any failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/baseline'))
    args = parser.parse_args()
    status = Path('outputs/logs') / ('training_status.json' if args.latent_dim==50 else f'training_status_pc{args.latent_dim}.json')
    try:
        write_status(status, stage='loading_data')
        pl.seed_everything(0, workers=True)
        full = sdq.datasets.larry(data_dir=str(args.data_dir), variant=None)
        adata = quickstart_subset(full)
        del full
        adata.obsm['X_pca']=np.asarray(adata.obsm['X_pca'])[:,:args.latent_dim].copy()
        write_status(status, stage='smoke_training', n_cells=adata.n_obs)
        run_stage(adata, 2, 0, args.output_dir/'smoke', 'larry_smoke')
        write_status(status, stage='full_training', epochs=args.epochs, n_cells=adata.n_obs)
        run_stage(adata, args.epochs, 0, args.output_dir/'official', 'larry_baseline')
        measured = (args.output_dir/'official/semantics.json').read_text()
        Path('outputs/logs/model_semantics.md').write_text(
            '# Measured trained LARRY teacher semantics\n\n'
            'Runtime measurements from scdiffeq==1.1.4 / neural-diffeqs==0.4.1.\n\n'
            '```json\n' + measured + '\n```\n\n'
            'G is the stochastic coefficient; D = GG^T is the infinitesimal covariance.\n'
            'General-noise G is not a diagonal diffusion vector.\n'
            'Time probes supplement the inspected f/g source; they alone do not prove time independence.\n'
            'This is teacher approximation semantics, not identified biological noise.\n')
        write_status(status, stage='training_finished', epochs=args.epochs,
                     caveat='Scientific field/distribution audit still required before distillation')
    except Exception as exc:
        write_status(status, stage='failed', error=repr(exc))
        raise

if __name__ == '__main__':
    main()
