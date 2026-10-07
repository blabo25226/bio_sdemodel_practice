Date: 2026-10-07 (Asia/Tokyo)
Git commit: 6760499d0cfff3e115c3ad996bf8affb9c9f4729 + working changes
Environment: project .venv, Python 3.11.17; requirements-lock.txt and conda-explicit.txt saved
Data variant/subset: default LARRY in vitro; download via sdq.datasets.larry started
Random seed: 0
Model/checkpoint: none yet
Hypothesis: reproduce real-data teacher before polynomial-only symbolic distillation
Change from previous run: storage freed (~99 GB); isolated environment installed;
user clarified polynomial interaction terms are allowed
Metrics: CUDA tensor operation on RTX 2070 succeeded; pip check passed;
synthetic data tests 3 passed under dedicated environment
Figures: none
Result: environment ready; LARRY transfer ongoing; M0/M1 not yet achieved
Interpretation: no fitted model, no distillation claims
Failure / caveats: Zenodo download throughput is currently slow; large file must
finish and checksum must pass before inspecting real-data counts.
Next action: monitor outputs/logs/data_run.txt, complete M0, audit runtime time/splits,
then train/inspect baseline. Polynomial library includes powers AND interactions;
no trigonometric/exponential/logarithmic candidates.
