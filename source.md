# Sources / links

最終確認日: **2026-10-07**

このファイルは、`bio_sdemodel_practice` で使う一次情報・公式実装を優先したリンク集。

## 1. scDiffEq 論文

### Version of record

- **Vinyard et al., Learning cell dynamics with neural differential equations**
- Nature Machine Intelligence 7, 1969–1984 (2025)
- DOI: https://doi.org/10.1038/s42256-025-01150-3
- Nature: https://www.nature.com/articles/s42256-025-01150-3

主用途:
- 最終的な方法・結果・Data availability / Code availability の確認
- state-dependent drift / diffusion、LARRY 解析、fate prediction の参照

### Free preprint

- **scDiffEq: drift-diffusion modeling of single-cell dynamics with neural stochastic differential equations**
- bioRxiv DOI: https://doi.org/10.1101/2023.12.06.570508
- v2: https://www.biorxiv.org/content/10.1101/2023.12.06.570508v2

主用途:
- Nature 版にアクセスできない場合の本文理解
- 数式・方法の読み込み

## 2. scDiffEq 公式実装・ドキュメント

### Package

- GitHub: https://github.com/scDiffEq/scDiffEq
- PyPI: https://pypi.org/project/scdiffeq/
- Documentation: https://www.scdiffeq.com/
- Installation: https://www.scdiffeq.com/install.html
- API / model: https://www.scdiffeq.com/_api/model.html

### Quickstart — 最優先

- https://www.scdiffeq.com/_tutorials/quickstart.html

確認済みの重要点:
- LARRY in vitro を使用
- `sdq.datasets.larry(...)` で取得可能
- Quickstart の processed data は約 130,887 cells × 約 2,492 genes
- `Monocyte`, `Neutrophil`, `Undifferentiated` の subset 例あり
- default cell state は `adata.obsm['X_pca']`
- default drift / diffusion はそれぞれ PyTorch neural network
- `model.drift()` で `X_drift` 等を追加
- `model.diffusion()` で `X_diffusion` 等を追加
- trajectory simulation の例あり

### Trained model loading

- https://www.scdiffeq.com/_tutorials/scdiffeq_tutorial.load_trained_model.html

主用途:
- LARRY full dataset 用公開モデル / checkpoint の読み込み手順
- 最初の symbolic distillation proof-of-concept

### Dataset loader

- https://www.scdiffeq.com/_api/datasets.html

現在の docs では scDiffEq dataset は Zenodo から初回 download / cache される。

### Model details

- Lightning SDE fixed potential regularized velocity ratio:
  https://www.scdiffeq.com/_api/_model/lightning_sde_fixed_potential_regularized_velocity_ratio.html

現在の docs で確認できる例:
- default latent dimension: 50
- `sde_type='ito'`
- `noise_type='general'`
- `brownian_dim=1`

**実際に使う checkpoint / model instance の値を必ずコード上で再確認すること。**

## 3. 論文再現コード

- scDiffEq analyses repository:
  https://github.com/scDiffEq/scdiffeq-analyses

Nature 論文の Code availability に記載されている再現 repository。

用途:
- checkpoint
- manuscript analyses
- figure reproduction
- scDiffEq 独自 helper の確認

## 4. LARRY dataset

### scDiffEq 用 LARRY package

- https://github.com/scDiffEq/LARRY-dataset

LARRY は mouse bone marrow hematopoietic progenitor + lentiviral lineage barcode を用いた実データ。

同 repository では以下を扱う:
- in vitro
- in vivo
- cytokine-perturbed in vitro

### Original data location cited by Nature paper

- Allon Klein Lab paper-data:
  https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation

### Original LARRY paper

- Weinreb et al., *Lineage tracing on transcriptional landscapes links state to fate during differentiation*, Science (2020)
- DOI lookup / publisher pageは論文タイトルで確認すること。

## 5. Zenodo archives

Nature 論文 Code/Data availability:

- scdiffeq-analyses archive:
  https://doi.org/10.5281/zenodo.17238611
- scDiffEq code archive:
  https://doi.org/10.5281/zenodo.17238594

scDiffEq current dataset docs:

- dataset cache record:
  https://doi.org/10.5281/zenodo.21947161

## 6. SINDy

### PySINDy

- Documentation: https://pysindy.readthedocs.io/en/stable/
- GitHub: https://github.com/dynamicslab/pysindy

PySINDy は SINDy / sparse system identification の実装候補。

最初の候補:
- PolynomialLibrary
- STLSQ
- SR3

ただし `f_θ(x)` の teacher function を直接 sparse regression する場合、PySINDy の通常の trajectory derivative workflow に無理に合わせず、library matrix と sparse optimizer を明示的に扱う実装も比較検討する。

## 7. 参照優先順位

不一致があった場合は原則として

```text
現在インストールした package の実コード / runtime shape
> 現行公式 docs
> version-of-record paper
> reproducibility repository
> bioRxiv preprint
> 二次解説
```

の順で API の事実を確認する。

一方、生物学的主張や論文の結論については version-of-record paper を優先する。

## 8. 注意

- scDiffEq は更新が続いているため、古い notebook と現行 package API が一致しない可能性がある。
- URL / checkpoint path / dataset shape をコードにハードコードする前に現行 docs と runtime を確認する。
- 論文の文章を大量に repository へ転載しない。要約と引用元リンクで管理する。

## 9. Bootstrap source checks (2026-10-07)

Title: scDiffEq 1.1.4 PyPI distribution metadata and source wheel
Authors: scDiffEq maintainers
Year: 2026 (access year; release date not asserted)
URL: https://pypi.org/project/scdiffeq/1.1.4/
Why relevant: Python >=3.11 requirement, official default LARRY loader and model API source inspection.
Accessed: 2026-10-07. Wheel/module SHA256 recorded in outputs/logs/package_inspection.json.

Title: scDiffEq dataset cache record metadata
Authors: See Zenodo record metadata
Year: 2026 (access year)
URL: https://zenodo.org/records/21947161
Why relevant: Default larry.h5ad download is 5,308,287,542 bytes; per-file checksum and URLs saved in preflight.json.
Accessed: 2026-10-07. No dataset payload downloaded.

## 10. Polynomial STLSQ implementation check

Title: PySINDy 2.1.0 documentation and installed STLSQ source
Authors: PySINDy maintainers
Year: 2026 (access year)
URL: https://pysindy.readthedocs.io/en/stable/
Why relevant: PolynomialLibrary contains monomial interaction terms; STLSQ supports
thresholded ridge regression and an optional unregularized final refit. Installed
source was inspected; initial unbias=True fit failed validation, so retained ridge
regularization is evaluated explicitly. No trajectory derivative API is forced
onto direct teacher-function regression.
Accessed: 2026-10-07.
