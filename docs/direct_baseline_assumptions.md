# Direct observed-data baseline

LARRY observations are snapshots of different cells, not repeated measurements
of the same cell. A clone barcode connects related cells but does not establish
an individual-cell trajectory or identify an infinitesimal diffusion tensor.
Consequently, finite differences of all cells ordered by time are invalid here.

The implementable proxy is a clone-centroid finite-interval drift baseline:
for clones observed at consecutive times, x is the earlier PC centroid and
y=(later centroid - earlier centroid)/time gap. Fit a polynomial-only STLSQ
function to training-clone pairs, choose degree/threshold/alpha on validation
clones, and report test-clone finite-interval displacement errors.

This assumes earlier/later sampled cells from the clone can stand in for its
population mean, despite branching, proliferation, death and sampling bias.
The target is a finite-interval clone-population displacement, not true
instantaneous single-cell drift. Its limited performance is not evidence
that teacher distillation is superior to valid stochastic SINDy.

A direct stochastic diffusion baseline cannot be identified fairly from these
pseudo-transitions alone: within-clone variance includes biological state
heterogeneity, branching, measurement and sampling effects. Do not turn it
into independent Brownian noise or compare it to teacher G as a ground truth.
Report this non-identifiability and the available drift proxy explicitly.
