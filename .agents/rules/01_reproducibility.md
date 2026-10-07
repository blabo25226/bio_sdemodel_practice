# Reproducibility rules

1. Random seed を明示する。
2. package versions を保存する。
3. Python / PyTorch / CUDA / GPU を log に残す。
4. dataset variant、subset condition、cell count、latent dimension を記録する。
5. checkpoint path と git commit を記録する。
6. generated artifact は `outputs/` 以下に保存する。
7. figure を作るコードと table を作るコードを残す。
8. test / validation data を model selection に使わない。
9. 実験条件を notebook cell の暗黙状態だけに置かない。
10. 再現不能な manual edit をしない。

推奨: `environment.yml`, `requirements-lock.txt` または `uv.lock` を実装開始時に作る。
