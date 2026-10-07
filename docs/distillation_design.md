# Symbolic distillation design (2026-10-07)

User constraint: use only polynomial powers and their interaction monomials.
With state x in R^d and maximum total degree p, the library is
Theta(x) = {prod_i x_i^a_i : a_i >= 0 integer, sum_i a_i <= p}.
Use one constant. No sin/cos, exp/log, rational, or other function families.
Library size is binomial(d+p, p): d=50 gives 51, 1326, 23426 terms for p=1,2,3.
Do not advance to degree 3 without evidence from validation and memory review.

First finish the data/baseline and runtime SDE audit before implementing fits.
STLSQ targets are direct teacher drift evaluations, not finite differences.
A preprocessing/scaling transform must be fitted only on training states,
saved, and incorporated into evaluation/equation export. Report the coordinate
system and threshold units; scaled-input coefficients are not original-PC coefficients.

Split plan requiring runtime data audit:
- Distinguish teacher training/validation/test from distillation splits.
- Record teacher exposure for every observed state used for distillation evaluation.
- Avoid clone leakage; missing clone IDs must not silently become one artificial clone.
- Reserve a test partition before fitting transforms or selecting degree/threshold/alpha.
- Teacher simulations for training start only from training-origin initial states.
- Bootstrap whole clone/trajectory groups where dependence requires it.
- Upstream PCA may already expose all cells; do not call this independent gene-space validation.

Record function MSE and normalized error with the denominator explicitly defined,
active terms, threshold sweeps and group bootstrap selection frequencies.
If the restricted library is inadequate, report failure rather than silently
adding disallowed functions. Low sparsity alone is not success.

Diffusion fit target is chosen only after runtime noise semantics are verified.
If fitting G, compute D_hat = G_hat G_hat^T for covariance comparisons.
This matrix multiplication is part of SDE semantics, not an added library family.
If fitting D through a factor, explicitly document both factor and covariance
formula; products can increase the resulting covariance polynomial degree.

## Recorded experiment decisions

The original 50-PC teacher was trained for the official baseline. Its degree<=2
polynomial trial failed validation, including strong extrapolation sensitivity.
The result is retained; it was not silently replaced with an easier problem.

A separate five-PC teacher is trained from scratch to test a lower-dimensional
proof-of-concept. It retains ~59.6% of empirical variance of the first50 PC states
in the observed symbolic-training subset; this is not gene-space explained variance.
The five-PC model is not the projected field of the 50-PC teacher.
Degree3 (56 features at d=5) is attempted only after degree2 validation is insufficient.

The local approximation domain is the 99.5th percentile of training-state Euclidean
PC norms. Validation coverage must be >=95%. All-domain errors and test coverage
are also reported. Neither the polynomial nor the simulator clips states to this
domain. Simulation explosion is separately counted at evaluated times when a state
exceeds ten times the training-defined domain radius, independently of NaN/Inf checks.
No simulation-result-driven hyperparameter tuning on test origins is allowed.
