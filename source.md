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

## 11. Day 11 paper (daily Neural SDE series)

Title: SDE-Net: Equipping Deep Neural Networks with Uncertainty Estimates
Authors: Lingkai Kong, Jimeng Sun, Chao Zhang
Year: 2020
URL: https://proceedings.mlr.press/v119/kong20b.html
PDF: http://proceedings.mlr.press/v119/kong20b/kong20b.pdf
Why relevant: Neural drift f(x_t, t) trained by a task loss and neural diffusion g(x_0) trained to be small on in-distribution inputs and large on Gaussian-perturbed inputs (ICML 2020, eq. 4 and eq. 7, Algorithm 1). Day 11 distills those two networks with a power-only SINDy library. The supplement's test-time sigma_max = 50 is not part of the distilled SDE; the training value sigma_max = 1 is.
Accessed: 2026-11-04.

## 12. Day 12 paper (daily Neural SDE series)

Title: Efficient Training of Neural Stochastic Differential Equations by Matching Finite Dimensional Distributions
Authors: Jianxin Zhang, Josh Viktorov, Doosan Jung, Emily Pitler
Year: 2025
URL: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5acf5a0ee5c17d372bfe7fdaeffd6e33-Abstract-Conference.html
PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/5acf5a0ee5c17d372bfe7fdaeffd6e33-Paper-Conference.pdf
arXiv: https://arxiv.org/abs/2410.03973
Code: https://github.com/Z-Jianxin/FDM
Why relevant: ICLR 2025. Neural drift and neural diffusion are trained by a kernel scoring rule on two-time joints (Finite Dimensional Matching, Theorem 2, Algorithm 1). The released FDDiscriminator minimizes E[k(X,X')] - 2 E[k(X,Y)] on consecutive grid times with a Gaussian RBF of bandwidth 1. Day 12 distills that teacher on an observable 2D Markov diffusion. The paper's Dukascopy KS tables and the rough Bergomi experiment are outside the cut.
Accessed: 2026-11-04.

## 13. Day 13 paper (daily Neural SDE series)

Title: SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations
Authors: Grigory Bartosh, Dmitry Vetrov, Christian A. Naesseth
Year: 2025
URL: https://proceedings.mlr.press/v267/bartosh25a.html
PDF: https://raw.githubusercontent.com/mlresearch/v267/main/assets/bartosh25a/bartosh25a.pdf
arXiv: https://arxiv.org/abs/2502.02472
Code: https://github.com/GrigoryBartosh/sde_matching
Why relevant: ICML 2025, PMLR 267:3054-3070. The prior Neural SDE has neural drift h_theta and neural diffusion g_theta (eq. 15). Training minimizes the latent-SDE variational bound without integrating the posterior SDE, by sampling z_t = mu_phi(X, t) + sigma_phi(X, t) epsilon (eq. 37) and converting that flow to a posterior drift with the probability-flow identity (eq. 19-20). Day 13 distills the state-dependent prior on an observable 2D stochastic Lotka-Volterra system (Appendix C.3, sigma = 0.15; drift coefficients from Course and Nair, NeurIPS 2023, Appendix G.1). Motion capture, pendulum video, and the 3D Lorenz experiment are outside the cut.
Accessed: 2026-11-06.

## 14. Day 14 paper (daily Neural SDE series)

Title: Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL
Authors: Cevahir Koprulu, Franck Djeumou, Ufuk Topcu
Year: 2025
URL: https://proceedings.iclr.cc/paper_files/paper/2025/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html
PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/0f25eb6e9dc26c933a5d7516abf1eb8c-Paper-Conference.pdf
Partial reimplementation: https://github.com/matthew-lowery/NSDE_ORL
Why relevant: ICLR 2025. The dynamics model is a neural SDE (eq. 5) whose drift can use rigid-body structure (eq. 8: position rate equals velocity, velocity rate is G(v)a + H(s)v plus a penalized residual) and whose diagonal diffusion is aleatoric sigma_phi plus a bounded monotone function of a distance-aware eta (eq. 3-4, Appendix C.4). Day 14 distills that teacher on an observable controlled point mass with cubic drag. D4RL, NeoRL, MuJoCo, reward learning, and SAC policy optimization are outside the cut. Appendix D's aleatoric-only ablation (NUNO-al) is the H6 baseline.
Accessed: 2026-11-07.

## 15. Day 15 paper (daily Neural SDE series)

Title: In-Context Learning of Stochastic Differential Equations with Foundation Inference Models
Authors: Patrick Seifner, Kostadin Cvejoski, David Berghaus, César Ojeda, Ramsés J. Sánchez
Year: 2025
URL: https://proceedings.neurips.cc/paper_files/paper/2025/hash/dff573cecb50646083b5186b565e23a3-Abstract-Conference.html
PDF: https://proceedings.neurips.cc/paper_files/paper/2025/file/dff573cecb50646083b5186b565e23a3-Paper-Conference.pdf
Why relevant: NeurIPS 2025. A transformer recognition model maps a context of transitions (y, Δy, Δy², Δτ) to drift and diagonal diffusion at a query state, with a detached uncertainty head U (eqs. 7-10). Dense finetuning is the short-time Gaussian likelihood (eq. 34). The pretraining class is polynomial drift of degree at most 3 and a degree-at-most-2 polynomial inside the diffusion square root. Day 15 distills a miniature of that teacher on the paper's 2D synthetic system (Appendix G.1, eqs. 38-39). The 20M-parameter pretraining run, signature-kernel MMD, BISDE, SparseGP, Latent SDE, Lorenz, and the wind, oil, and equity series are outside the cut.
Accessed: 2026-11-08.

## 16. Day 16 paper (daily Neural SDE series)

Title: LieStoNet: Learning Lie Symmetries from Spatiotemporal Data for Stochastic Dynamical Systems
Authors: Shida Liu, Abhishek Gupta, Sumit Sinha, L. Mahadevan
Year: 2026
URL: https://proceedings.mlr.press/v306/liu26aq.html
PDF: https://raw.githubusercontent.com/mlresearch/v306/main/assets/liu26aq/liu26aq.pdf
arXiv: https://arxiv.org/abs/2608.01582
Code: https://github.com/sumit-sinha-seas/LieStoNet_Final
Why relevant: ICML 2026, PMLR 306:76384-76408. Stage 1 fits a neural drift and a neural Cholesky diffusion by the Euler-Gauss increment likelihood (Section 3.1, equation 8; Appendix P). Example 4 is the 2D Ito equation dx = (1/x) dt + dW1, dy = dt + dW2 (Table 2). The released notebook uses a2 = 0.5; Day 16 follows Table 2 and sets a2 = 1. Day 16 distills that stage-1 surrogate. Stage-2 generator training, the Lie-algebra sweep, principal angles, Fokker-Planck symmetries, the 10D Brownian motion, and BTC/USDT are outside the cut. The analytic scaling generator is used only as a probe of the learned drift.
Accessed: 2026-11-08.

## 17. Day 17 paper (daily Neural SDE series)

Title: Neural Diffusion Intensity Models for Point Process Data
Authors: Xinlong Du, Harsha Honnappa, Vinayak Rao
Year: 2026
URL: https://proceedings.mlr.press/v337/du26a.html
PDF: https://raw.githubusercontent.com/mlresearch/v337/main/assets/du26a/du26a.pdf
Why relevant: UAI 2026, PMLR 337:1448-1473. A Cox process is driven by a latent diffusion intensity. The prior drift is a neural network (equation 1); the diffusion coefficient in the experiments is the known CIR factor \(\sqrt{z}\), not a network. The variational posterior keeps that diffusion and adds a neural drift correction (equation 3, Theorem 2.1, Theorem 3.1). Day 17 distills the prior drift on Section 4.1, \(dZ=0.3(80-Z)\,dt+\sqrt{Z}\,dW\), and probes \(\sqrt{Z}\) with the power library. The remaining-count encoder is the paper's misspecified option in Section 3.3.1. Deep Sets, the hard constraint, MCMC, the U.S. bank calls, the Allen spike trains, and the multivariate CIR are outside the cut.
Accessed: 2026-11-09.

## 18. Direct stochastic sparse identification assumptions

Title: Sparse learning of stochastic dynamical equations
Authors: Lorenzo Boninsegna, Feliks Nüske, Cecilia Clementi
Year: 2018
URL/DOI: https://doi.org/10.1063/1.5018409
Preprint: https://arxiv.org/abs/1712.02432
Author manuscript: https://fnueske.github.io/pdf/18_Boninsegna_SINDy.pdf
Why relevant: Sparse stochastic identification uses dynamical transition/increment
information for drift/diffusion estimation. LARRY clone-centroid pseudo-transitions
cannot be silently treated as observed individual-cell SDE increments. This motivates
reporting the available drift proxy and the diffusion identifiability limitation.
Accessed: 2026-10-07.
