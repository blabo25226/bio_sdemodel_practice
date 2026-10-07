"""Official Quickstart subset and auditable LARRY data inventory."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def quickstart_subset(adata: Any) -> Any:
    """Return Quickstart subset, preserving upstream IDs in obs (n_cells, n_genes)."""
    clone_ids = adata.uns['fate_counts'][['Monocyte', 'Neutrophil']].dropna().index
    mask = (adata.obs['Cell type annotation'].isin(
        ['Monocyte', 'Neutrophil', 'Undifferentiated'])
        & adata.obs['clone_idx'].isin(clone_ids))
    subset = adata[mask].copy()
    subset.obs['source_cell_id'] = subset.obs_names.astype(str)
    subset.obs['nm_clones'] = True
    for key in ('X_clone', 'cell_fate_df'):
        if key in subset.obsm:
            del subset.obsm[key]
    subset.obs_names = [str(i) for i in range(subset.n_obs)]
    return subset


def save_inventory(full: Any, subset: Any, output_dir: Path) -> dict:
    """Save dimensions, keys and class/time counts of two AnnData objects."""
    logs, tables = output_dir / 'logs', output_dir / 'tables'
    logs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, adata in [('full', full), ('quickstart_subset', subset)]:
        x = adata.obsm['X_pca']
        if x.ndim != 2 or x.shape[0] != adata.n_obs:
            raise ValueError('X_pca must have shape (n_cells, latent_dimension)')
        summary[label] = {'n_cells': adata.n_obs, 'n_genes': adata.n_vars,
                          'pca_shape': list(x.shape),
                          'keys': {k: list(getattr(adata, k).keys())
                                   for k in ('obs', 'obsm', 'layers', 'uns')}}
        for column in ('Time point', 'Cell type annotation'):
            adata.obs[column].value_counts(dropna=False).rename('cell_count').to_csv(
                tables / f'{label}_{column.replace(" ", "_")}.csv')
        adata.obs.groupby(['Time point', 'Cell type annotation'], observed=True).size().rename(
            'cell_count').to_csv(tables / f'{label}_time_by_class.csv')
    import pandas as pd
    pd.DataFrame([{ 'dataset': k, **{f: v[f] for f in ('n_cells', 'n_genes')}}
                  for k, v in summary.items()]).to_csv(tables / 'dataset_summary.csv', index=False)
    (logs / 'dataset_inventory.json').write_text(json.dumps(summary, indent=2))
    return summary


def sha256_file(path: Path) -> str:
    """Compute SHA256 in bounded memory for external data provenance."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Load official LARRY, save inventory and external cache checksums."""
    import argparse
    import subprocess
    import sys
    from .preflight import inspect_environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    status = inspect_environment(args.data_dir)
    if not status['ready']:
        raise SystemExit('Run python -m src.preflight; prerequisites are incomplete.')
    import lightning as pl
    import scdiffeq as sdq
    pl.seed_everything(0, workers=True)
    full = sdq.datasets.larry(data_dir=str(args.data_dir), variant=None)
    subset = quickstart_subset(full)
    summary = save_inventory(full, subset, Path('outputs'))
    cache = args.data_dir / 'scdiffeq_data' / 'larry'
    provenance = {
        'seed': 0, 'variant': None, 'loader': 'sdq.datasets.larry',
        'preflight': status,
        'files': [{'path': str(p), 'size': p.stat().st_size, 'sha256': sha256_file(p)}
                  for p in sorted(cache.glob('*')) if p.is_file()],
        'caveat': 'Upstream PCA/scaling may use all cells. This is official reproduction, not independent held-out preprocessing.',
    }
    Path('outputs/logs/data_provenance.json').write_text(json.dumps(provenance, indent=2))
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    Path('requirements-lock.txt').write_text(freeze)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
