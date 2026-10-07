Date: 2026-10-07 (Asia/Tokyo)
Git commit: 6d2369c + working changes
Environment: requirements-lock.txt, PySINDy 2.1.0
Data variant/subset: 50-PC LARRY; clone-disjoint symbolic partitions
Random seed: 0
Model/checkpoint: outputs/baseline/official/teacher.ckpt
Hypothesis: Degree 1 or 2 approximates direct teacher drift.
Change from previous run: Initial STLSQ with training RMS scaling, alpha=1, unbias=True.
Metrics: Initial degree1 best validation NRMSE ~0.897; degree2 threshold0.01
train NRMSE ~0.401, validation NRMSE ~56.957. Sweep preserved in initial_unbiased_drift/sweep.csv.
Figures: teacher_baseline.png
Result: Failed validation. Run interrupted before bootstrap/test and diffusion.
Interpretation: Training fidelity does not imply usable held-out polynomial behavior.
Failure / caveats: Validation state norm max ~213.6 vs training max ~99.8;
unregularized final refit and polynomial extrapolation can magnify errors.
Next action: Keep ridge regularization, use alpha 100/1000 validation sweep.
Define a local demo domain from the 99.5th percentile of training PC norms only.
Fit and select within this domain (require >=95% validation coverage), while
reporting all-domain validation/test errors and coverage without clipping the polynomial.
Test remains unread and unused for all choices; this is local teacher approximation,
not globally faithful symbolic dynamics. Degree3 is not justified by extrapolation failure.
