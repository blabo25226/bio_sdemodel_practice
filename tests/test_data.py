"""Synthetic data tests: not evidence that LARRY has been loaded."""
import numpy as np
import pandas as pd
import anndata as ad
from src.data import quickstart_subset, save_inventory, sha256_file


def fixture():
    data = ad.AnnData(np.zeros((6, 3)), obs=pd.DataFrame({
        'clone_idx': [1, 1, 2, 3, 1, 4],
        'Cell type annotation': ['Monocyte', 'Undifferentiated', 'Neutrophil',
                                 'Monocyte', 'Other', 'Neutrophil'],
        'Time point': [2, 2, 4, 6, 6, 6],
    }, index=[f'cell{i}' for i in range(6)]))
    data.uns['fate_counts'] = pd.DataFrame({
        'Monocyte': [1., np.nan, 0., 1.], 'Neutrophil': [2., 1., 0., np.nan]},
        index=[1, 2, 3, 4])
    data.obsm['X_pca'] = np.zeros((6, 2))
    data.obsm['X_clone'] = np.zeros((6, 4))
    data.obsm['cell_fate_df'] = np.zeros((6, 2))
    return data


def test_official_mask_and_original_ids():
    full = fixture()
    subset = quickstart_subset(full)
    assert subset.obs['source_cell_id'].tolist() == ['cell0', 'cell1', 'cell3']
    assert subset.obs_names.tolist() == ['0', '1', '2']
    assert 'X_clone' not in subset.obsm
    assert 'X_clone' in full.obsm
    assert 'source_cell_id' not in full.obs
    # dropna excludes missing counts but intentionally keeps zero counts, as Quickstart.
    assert subset.obs['clone_idx'].tolist() == [1, 1, 3]


def test_inventory_counts(tmp_path):
    full = fixture()
    summary = save_inventory(full, quickstart_subset(full), tmp_path)
    assert summary['quickstart_subset']['pca_shape'] == [3, 2]
    counts = pd.read_csv(tmp_path / 'tables/quickstart_subset_time_by_class.csv')
    assert counts.cell_count.sum() == 3
    table = pd.read_csv(tmp_path / 'tables/dataset_summary.csv')
    assert table.n_cells.tolist() == [6, 3]


def test_checksum(tmp_path):
    path = tmp_path / 'sample'
    path.write_bytes(b'abc')
    assert sha256_file(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
