# CLAUDE.md

Claude はこの repository で作業する前に `1007_instruction.md` と `.agents/rules/` をすべて読み、該当する `.agents/skills/` を参照すること。

特に以下を毎回確認する。

- latent space の式を gene regulation と誤解していないか
- drift / diffusion の tensor semantics を runtime で確認したか
- teacher Neural SDE と実データを区別して評価しているか
- SINDy の sparsity と fidelity の両方を評価したか
- validation split / test split の leakage がないか

詳細は `.agents/README.md`。
