# Execution status

- `00_data_check.ipynb`: runnable data inventory after environment/storage setup; unexecuted.
- `01_scdiffeq_baseline.ipynb`: baseline execution plan; runtime audit gate intentionally closed.

Run notebooks from the repository or notebooks directory using the `bio-sde` kernel.
Set `BIO_SDE_DATA_DIR` before launching Jupyter; default is project-relative `data/`.
All generated experiment artifacts go under `outputs/`. External dataset caches stay in
`BIO_SDE_DATA_DIR` and are not committed.

Environment setup after a sufficiently large persistent location is available:

```bash
conda env create --prefix "$BIO_SDE_ENV_DIR" --file environment.yml
conda activate "$BIO_SDE_ENV_DIR"
python -m pip freeze > requirements-lock.txt
python -m ipykernel install --user --name bio-sde --display-name 'Python (bio-sde)'
python -m src.preflight --data-dir "$BIO_SDE_DATA_DIR"
python -m src.data --data-dir "$BIO_SDE_DATA_DIR"
python -m pytest -q
```

`BIO_SDE_ENV_DIR` and `BIO_SDE_DATA_DIR` are user-selected storage paths.
The conda package cache also needs sufficient space (configure `CONDA_PKGS_DIRS`
there before environment creation). CUDA/PyTorch wheel compatibility with RTX 2070
must be checked before training; the host driver listing alone does not establish it.

After installation, save `conda list --explicit` under `outputs/logs/` in addition
to the pip freeze. Verify actual CUDA runtime/device in the isolated environment.

The verified script path for automatic training is `python -m src.baseline`.
It uses explicit `Time point`, GPU 0, fresh seed-0 models for the 2-epoch smoke
and 1500-epoch official stages. The synthetic API probe passed GPU training,
direct/public field correspondence, finite simulation and same-seed simulation.
This does not establish success on real LARRY. Model-specific clone overlap is
reported before each real training stage; official cell-level validation is
not presented as an independent clone-held-out evaluation.

`python -m src.queue_baseline --download-pid PID` waits for an existing loader
process and checks its provenance/inventory before starting this script.
Monitor `outputs/logs/training_status.json`, `data_run.txt` and `baseline_run.txt`.
Large baseline artifacts stay local under `outputs/baseline/`.

For an exact pip replay of the recorded CUDA wheel versions, use:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-lock.txt
```

The 50-PC drift trial failed its declared fidelity criterion and is retained under
`outputs/sindy/`. The independent exploratory five-PC teacher run uses:

```bash
python -m src.baseline --latent-dim 5 --output-dir outputs/baseline_pc5
python -m src.prepare --components 5 --state-dir outputs/distillation_pc5 --checkpoint outputs/baseline_pc5/official/teacher.ckpt --figure-prefix pc5_
python -m src.run_distillation --config configs/experiment_pc5.json
python -m src.validate --state-dir outputs/distillation_pc5 --sindy-dir outputs/sindy_pc5 --checkpoint outputs/baseline_pc5/official/teacher.ckpt --output-dir outputs/validation_pc5 --figure-prefix pc5_
python -m src.direct_baseline --state-dir outputs/distillation_pc5 --sindy-dir outputs/sindy_pc5 --checkpoint outputs/baseline_pc5/official/teacher.ckpt --output-dir outputs/direct_baseline_pc5
```

Stop after a failed drift/diffusion gate. `src.campaign_pc5` queues these operations
behind an existing teacher process and records failures rather than silently advancing.
