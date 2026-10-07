# Measured trained LARRY teacher semantics

scdiffeq 1.1.4 / neural-diffeqs 0.4.1, PyTorch 2.6.0+cu124.
Both independently trained teachers use LightningSDE_FixedPotential_RegularizedVelocityRatio
wrapping PotentialSDE: Itô, general noise, one Brownian driver, autonomous f/G.
Source inspection shows the time argument is ignored; time probes agree.

| Quantity | Official 50-PC teacher | Exploratory 5-PC teacher |
|---|---|---|
| State X | (batch,50) | (batch,5) |
| Potential network μ(X) | (batch,1) | (batch,1) |
| Drift f(X)=coef × ∇μ(X) | (batch,50) | (batch,5) |
| Raw σ network | (batch,50) | (batch,5) |
| G(X) | (batch,50,1) | (batch,5,1) |
| D(X)=G(X)G(X)ᵀ | (batch,50,50) | (batch,5,5) |

G is the Brownian coefficient (PC/√day); D is instantaneous covariance (PC²/day).
One Brownian driver does **not** mean diagonal/independent noise per PC.
D is PSD and rank at most one. Drift units are PC/day.
Public X_drift/X_diffusion agree with direct fields. AnnData obs['drift'] and
obs['diffusion'] are L2 norms, not the vector/tensor fields.
Teacher drift requires autograd even during inference because it differentiates μ;
src/teacher.py isolates pinned-version checkpoint access and enables gradients.
Independent polynomial fitting of f does not enforce the teacher potential structure.
These are teacher semantics, not identified intrinsic biological noise.

## Runtime probes (batch 8)

### 50 PCs

```json
{
  "lightning_class": "<class 'scdiffeq.core.lightning_models._lightning_sde_fixed_potential_regularized_velocity_ratio.LightningSDE_FixedPotential_RegularizedVelocityRatio'>",
  "sde_class": "<class 'neural_diffeqs._potential_sde.PotentialSDE'>",
  "sde_type": "ito",
  "noise_type": "general",
  "brownian_dim": 1,
  "state_shape": [
    8,
    50
  ],
  "drift_shape": [
    8,
    50
  ],
  "diffusion_shape": [
    8,
    50,
    1
  ],
  "covariance_shape": [
    8,
    50,
    50
  ],
  "dtype": "torch.float32",
  "device": "cpu",
  "time_probe_drift_max_abs_difference": 0.0,
  "time_probe_diffusion_max_abs_difference": 0.0,
  "public_field_shapes": {
    "X_drift": [
      9350,
      50
    ],
    "X_diffusion": [
      9350,
      50
    ]
  },
  "public_api_matches_direct": true,
  "potential_network_output_shape": [
    8,
    1
  ],
  "diffusion_network_raw_output_shape": [
    8,
    50
  ],
  "drift_definition": "coef_drift * gradient_x potential(x)",
  "drift_parameters": 289280,
  "diffusion_parameters": 4338,
  "scalar_drift_obs": "L2 norm of f per cell",
  "scalar_diffusion_obs": "L2 norm of squeezed G per cell for brownian_dim=1; not G or D",
  "scalar_norms_verified": true
}
```

### 5 PCs

```json
{
  "lightning_class": "<class 'scdiffeq.core.lightning_models._lightning_sde_fixed_potential_regularized_velocity_ratio.LightningSDE_FixedPotential_RegularizedVelocityRatio'>",
  "sde_class": "<class 'neural_diffeqs._potential_sde.PotentialSDE'>",
  "sde_type": "ito",
  "noise_type": "general",
  "brownian_dim": 1,
  "state_shape": [
    8,
    5
  ],
  "drift_shape": [
    8,
    5
  ],
  "diffusion_shape": [
    8,
    5,
    1
  ],
  "covariance_shape": [
    8,
    5,
    5
  ],
  "dtype": "torch.float32",
  "device": "cpu",
  "time_probe_drift_max_abs_difference": 0.0,
  "time_probe_diffusion_max_abs_difference": 0.0,
  "public_field_shapes": {
    "X_drift": [
      9350,
      5
    ],
    "X_diffusion": [
      9350,
      5
    ]
  },
  "public_api_matches_direct": true,
  "potential_network_output_shape": [
    8,
    1
  ],
  "diffusion_network_raw_output_shape": [
    8,
    5
  ],
  "drift_definition": "coef_drift * gradient_x potential(x)",
  "scalar_drift_obs": "L2 norm of f per cell",
  "scalar_diffusion_obs": "L2 norm of squeezed G per cell for brownian_dim=1; not G or D",
  "scalar_norms_verified": true
}
```
