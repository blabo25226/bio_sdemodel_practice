# Skill: reproduce scDiffEq on LARRY in vitro

## Goal

公式 Quickstart を基準に、LARRY in vitro subset と scDiffEq baseline を再現する。

## Procedure

1. `source.md` の Quickstart / install docs を確認。
2. isolated environment を作る。
3. `scdiffeq`, `torch`, `torchsde`, `anndata` 等の version を記録。
4. `sdq.datasets.larry()` でデータを取得。
5. AnnData の `obs`, `obsm`, `layers`, `uns` keys を保存。
6. Quickstart の Monocyte / Neutrophil fate clones を再現。
7. subset 前後の cell counts を表にする。
8. model を fit または checkpoint を load。
9. `model.drift()` / `model.diffusion()` を実行。
10. official figure と完全一致を要求せず、field / trajectory が妥当に出ることを確認。

## Deliverables

- `notebooks/00_data_check.ipynb`
- `notebooks/01_scdiffeq_baseline.ipynb`
- `outputs/tables/dataset_summary.csv`
- `outputs/logs/environment.txt`
- baseline figures

## Stop condition

baseline が動かなければ SINDy に進まない。API mismatch は version difference として先に解決する。
