# Command playbook: distill

1. Confirm `outputs/logs/model_semantics.md` exists.
2. Extract states and teacher drift on train/val/test splits.
3. Fit degree-1 and degree-2 SINDy models.
4. Sweep sparsity threshold.
5. Select by validation fidelity + sparsity.
6. Test once on held-out states.
7. Only after drift succeeds, repeat for diffusion with correct constraints.
