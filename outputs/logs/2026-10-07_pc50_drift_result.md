Date: 2026-10-07 (Asia/Tokyo)
Git commit: f757bc3 (distillation implementation)
Environment: requirements-lock.txt; PySINDy 2.1.0
Data variant/subset: 50-PC LARRY; clone-disjoint symbolic partition; 99.5% training norm domain
Random seed: 0
Model/checkpoint: outputs/baseline/official/teacher.ckpt
Hypothesis: Degree <=2 sparse polynomials reproduce teacher drift.
Change from previous run: Retained ridge (unbias=False), alpha100/1000; local training-defined domain.
Metrics: selected degree2/alpha100/threshold0.05, 1326 library terms, 2815 active coefficients;
validation NRMSE 0.5544 (all-domain 9.0655); test NRMSE 0.6585 (in-domain 0.6284).
Mean selected bootstrap frequency 0.7311, support Jaccard ~0.50–0.55 (5 clone-group bootstraps).
Figures: outputs/figures/drift_sparsity.png
Result: M2 insufficient at predeclared NRMSE <=0.5 criterion; diffusion deferred for this teacher.
Interpretation: Polynomial restriction gives a sparse but imperfect approximation;
severe extrapolation sensitivity persists. Do not claim full symbolic-SDE success.
Failure / caveats: Teacher/upstream PCA exposure; local domain still has appreciable error;
no independent biological inference. Test reported after frozen model selection, not tuned upon.
Next action: Separately train a five-PC teacher as a lower-dimensional proof-of-concept.
This decision was made from validation before the 50-PC final test summary was examined.
Keep all 50-PC results; predeclare five-PC settings and permit degree3 only if degree2 fails validation.
