Date: 2026-10-07 (Asia/Tokyo), training finished 20:18 JST
Git commit: 5f99714 (training implementation)
Environment: requirements-lock.txt, Python 3.11.17 / torch 2.6.0+cu124
Data variant/subset: default LARRY; full 130887 x 2492; Quickstart subset 9350 x 2492; 50 PCs
Random seed: 0
Model/checkpoint: outputs/baseline/official/teacher.ckpt (local, excluded from Git)
Hypothesis: Official baseline can execute before polynomial-only distillation.
Change from previous run: Completed real-data load, 2-epoch smoke and fresh 1500-epoch training.
Metrics: Training 1784.068 seconds; peak CUDA allocated memory 2422240256 bytes;
final epoch train loss 56.06026077, validation loss 87.79185486.
Figures: outputs/baseline/official/loss.png (local)
Result: M0 complete; M1 numerical evaluation/simulation complete.
Observed f shape (8,50); G shape (8,50,1); Ito general noise.
Public field values agree with direct outputs; simulation finite; same-seed reproducibility verified.
Interpretation: Runtime baseline works. No symbolic or biological fidelity claim yet.
Failure / caveats: 292 clones overlap training and validation; this reproduces the
cell-level official baseline, not independent clone-held-out validation.
Upstream PCA/scaling exposure still prevents strict independent preprocessing claims.
Final epoch losses alone do not establish qualitative match to the official example.
Next action: Audit learned fields and trajectory distributions, update executed
notebooks, then implement polynomial powers/interactions-only drift STLSQ distillation.
