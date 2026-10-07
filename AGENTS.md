# AGENTS.md

この repository で作業する Codex 系エージェントは、実装前に以下を読むこと。

1. `1007_instruction.md`
2. `.agents/rules/00_project_scope.md`
3. `.agents/rules/01_reproducibility.md`
4. `.agents/rules/02_coding_style.md`
5. `.agents/rules/03_science_integrity.md`
6. `.agents/rules/04_git_and_outputs.md`
7. 実行するタスクに対応する `.agents/skills/`
8. `source.md`

プロジェクトの中心は **LARRY → scDiffEq Neural SDE → SINDy symbolic distillation → validation**。

最優先事項は「動くコード」よりも、**数理的意味・tensor shape・データリーク・再現性を壊さないこと**。
