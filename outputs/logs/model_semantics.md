# Model semantics — not yet measured

Date: 2026-10-07 (Asia/Tokyo)
Status: No model instance has been constructed or trained; M1 not achieved.

The inspected distribution is the official PyPI wheel scdiffeq==1.1.4.
Its source provenance is recorded in `package_inspection.json`.
This is source inspection, not runtime verification.

Pending runtime observations:
- type(model.DiffEq) and nested SDE class
- latent dimension / input state shape
- sde_type / noise_type / brownian_dim
- actual f(t, X) and g(t, X) shapes, dtype, device and finite checks
- implementation of time input handling; probe at multiple times
- correspondence of model.drift()/diffusion() AnnData fields to direct outputs
- interpretation and construction of D = GG^T appropriate to actual noise_type

No diagonal, scalar, independent-coordinate, or rank-one noise assumption is made.
For a general-noise tensor G shaped (batch, d, m), D is (batch, d, d)
with D[b] = G[b] @ G[b].T. This is conditional mathematical notation,
not an observation of this project's model. For diagonal noise, first interpret
its vector representation through the installed torchsde contract.

SINDy implementation is deferred until baseline execution and this audit succeed,
as required by `.agents/skills/01_reproduce_scdiffeq_larry.md`:
「baseline が動かなければ SINDy に進まない。」
