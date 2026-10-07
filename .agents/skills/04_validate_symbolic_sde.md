# Skill: validate symbolic SDE

## Goal

symbolic SDE が teacher functionだけでなく stochastic dynamics を保持しているか評価する。

## Level 1 — function fidelity

Held-out states で:

- drift MSE / NRMSE / cosine agreement
- diffusion error
- max / quantile error

## Level 2 — simulation fidelity

同一 initial-state distribution から複数 seeds で simulate。

比較:

- marginal distributions
- joint low-dimensional projections
- mean / covariance over time
- trajectory endpoints
- stability / NaN / explosion rate

## Level 3 — biological behavior

可能なら:

- Monocyte / Neutrophil fate proportions
- time-point occupancy
- fate prediction / endpoint classification

## Level 4 — original data

Teacher vs symbolic の比較だけで完結しない。観測 LARRY time points とも比較する。

## Reporting

必ず次の3列で報告する。

```text
Observed data | Neural SDE teacher | Symbolic SDE
```

## Failure criteria

次の場合は「symbolic distillation failed / insufficient」と報告する。

- sparse model にすると distribution が崩れる
- coefficients が bootstrap で極端に不安定
- diffusion constraint が満たせない
- held-out region で error が急増
