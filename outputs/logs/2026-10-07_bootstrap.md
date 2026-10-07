Date: 2026-10-07 (Asia/Tokyo)
Git commit: 6760499d0cfff3e115c3ad996bf8affb9c9f4729 (before working changes)
Environment: Host Python 3.14.6; see preflight.json and environment.txt.
Data variant/subset: Planned default LARRY in vitro, official fate-count dropna + three cell classes.
Random seed: 0 (planned; no teacher training performed)
Model/checkpoint: None. Inspected official scdiffeq 1.1.4 wheel only.
Hypothesis: First reproduce data and baseline before testing symbolic approximation.
Change from previous run: Initial environment/data inventory implementation.
Metrics: Synthetic data tests: 3 passed. Real-data metrics: not available.
Figures: None.
Result: M0/M1 pending. Download not attempted because local storage is insufficient.
Interpretation: This is setup progress, not evidence of scientific hypothesis or successful reproduction.
Failure / caveats:
- Local available capacity ~2.4 GB; official larry.h5ad = 5,308,287,542 bytes.
- Additional processed caches and Python/GPU dependencies require more space.
- Existing GPU environments use Python 3.10; current scdiffeq requires >=3.11.
- Isolated lightweight data tests used /tmp/bio-sde-data-test (ephemeral Python 3.14),
  not the planned training environment. It contains no LARRY or teacher model.
- environment.yml pins requested scdiffeq, but is not a complete transitive lock.
- Notebooks are unexecuted; training API/logger/time/split checks remain pending.
- Upstream preprocessing may have used all cells; strict held-out claims require separate audit.
Next action: User is preparing persistent storage. Set up isolated Python 3.11 environment there,
freeze resolved dependencies, execute 00_data_check, audit splits/time/logger, smoke baseline,
then official 1500-epoch reproduction and runtime semantics inspection.
