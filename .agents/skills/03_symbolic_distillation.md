# Skill: symbolic distillation with SINDy

## Goal

Neural SDE teacher の drift / diffusion functions を sparse symbolic functions に近似する。

## Stage 1 — sample states

最低3種類を区別する。

- observed training states
- held-out observed states
- teacher-simulated states

training / validation / test の境界を保つ。

## Stage 2 — drift targets

各 state `x_k` に対して teacher drift `f_theta(x_k)` を取得する。

trajectory finite difference を最初の方法にはしない。teacher function が直接取れるなら direct target を優先する。

## Stage 3 — library

順番:

1. degree 1
2. degree 2
3. degree 3 only if justified

高次元で全 polynomial interaction が膨張するので、feature count を必ず log する。

## Stage 4 — sparse fit

Baseline: STLSQ.

Hyperparameters は validation set で選択する。

保存:

- coefficients
- feature names
- threshold
- alpha
- active terms
- train/val/test error

## Stage 5 — diffusion

noise structureを確認してから対象を決める。

### If fitting G

symbolic `G_hat(x)` から simulation 可能か確認。

### If fitting D

PSD constraint を確認する。独立 element fit で PSD が壊れる場合は constrained factorization を使う。

## Stability analysis

最低限:

- threshold sweep
- seed/bootstrap sweep
- selected-term frequency

## Deliverables

- `src/distill.py`
- coefficient CSV/JSON
- human-readable equations `.md`
- error-vs-sparsity plot
