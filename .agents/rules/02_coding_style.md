# Coding style

- Python を主言語とする。
- notebook は exploration / plotting 用、コア処理は `src/` に置く。
- public function に type hints と短い docstring を付ける。
- tensor / ndarray を受ける関数は expected shape を docstring に書く。
- device / dtype を暗黙に変更しない。
- batch evaluation を優先し、巨大データを Python loop で回さない。
- hard-coded absolute path を禁止する。
- config / CLI 引数 / project-root relative path を使う。
- 重要変換（PCA, scaling, inverse transform）は fit object を保存する。
- 外部 package の private attribute に依存する場合は wrapper に隔離し、version を pin する。

最低限の tests:

- state batch → drift output shape
- state batch → diffusion output shape
- symbolic model → output shape
- same seed simulation reproducibility
- no NaN / Inf in sampled domain
