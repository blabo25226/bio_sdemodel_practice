Date: 2026-10-07 (Asia/Tokyo)
Git commit: 3623897 (baseline implementation), later generic path adapter added
Environment: requirements-lock.txt
Data variant/subset: 50-PC Quickstart subset; consecutive 2-day clone-centroid pairs
Random seed: 305 for integrated teacher endpoint sampling; clone split seed0
Model/checkpoint: outputs/baseline/official/teacher.ckpt, outputs/direct_baseline/model.npz
Hypothesis: Compare a raw-observation polynomial proxy with integrated teacher displacement.
Change from previous run: No teacher-function targets used to fit the raw baseline.
Metrics: train/validation/test pairs310/75/90; selected degree1/threshold0.05/alpha1000,
51 library features, 131 active coefficients. Test displacement NRMSE raw proxy0.8428,
Neural SDE teacher0.9673.
Figures: none yet (tables and complete equations saved)
Result: Raw proxy is better on this qualified finite-interval centroid metric.
Interpretation: The teacher-distillation superiority hypothesis is not established;
this proxy is not an identifiable direct stochastic SINDy benchmark.
Failure / caveats: Different-cell clone means include growth/branching/sampling;
Euler proxy vs stochastic integrated mean use different approximation assumptions;
teacher/upstream-PCA exposure; diffusion cannot fairly be inferred from centroid pairs alone.
Next action: Add separate five-PC comparison and final report; do not extrapolate
this metric into a claim about ground-truth dynamics or general method superiority.
