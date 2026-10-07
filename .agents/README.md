# .agents

Codex / Claude 共通の研究運用ディレクトリ。

```text
.agents/
├── rules/      # 常時守るルール
├── skills/     # 特定作業の標準手順
└── commands/   # 短い実行プレイブック
```

## 読み方

- 新規タスク開始: `rules/` を読む
- scDiffEq 再現: `skills/01_reproduce_scdiffeq_larry.md`
- SDE 内部確認: `skills/02_inspect_neural_sde.md`
- SINDy: `skills/03_symbolic_distillation.md`
- 評価: `skills/04_validate_symbolic_sde.md`
- 文献確認: `skills/05_literature_and_sources.md`

`commands/` は「何から実行するか」が不明なときの簡易手順。
