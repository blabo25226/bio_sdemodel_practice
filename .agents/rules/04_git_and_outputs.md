# Git and outputs

## Commit discipline

1つの commit で大きく異なる目的を混ぜない。

推奨 prefix:

- `setup:`
- `data:`
- `baseline:`
- `inspect:`
- `sindy:`
- `validate:`
- `docs:`
- `fix:`

## Track

- source code
- configs
- notebooks（出力を必要に応じて整理）
- lightweight figures / tables
- experiment notes

## Do not track by default

- raw LARRY dataset
- giant h5ad files
- large checkpoints duplicated from upstream
- cache
- temporary simulation arrays

大きな外部ファイルは取得元 URL / checksum /生成手順を残す。
