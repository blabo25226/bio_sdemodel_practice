# Skill: inspect Neural SDE semantics

## Goal

scDiffEq の drift / diffusion を symbolic distillation できる形で正確に定義する。

## Must inspect at runtime

```python
print(type(model.DiffEq))
print(model.DiffEq)
```

加えて可能な範囲で:

- latent dimension
- `sde_type`
- `noise_type`
- `brownian_dim`
- drift module
- diffusion module
- input shape
- output shape
- time argument handling

## Numerical probes

小さい batch `X` を作り、teacher functions を直接評価する。public API がある場合は public API を優先する。

記録例:

```text
X:             [batch, d]
f(X):          [...]
G(X):          [...]
D(X)=GG^T:     [...]
```

shape を推測で書かない。

## Cross-check

`model.drift()` / `model.diffusion()` が AnnData に追加する値と direct module output の関係を確認する。

特に scalar `adata.obs['diffusion']` が diffusion tensor 本体とは限らず、norm / magnitude の可能性があるため区別する。

## Deliverable

`outputs/logs/model_semantics.md`

内容:

- exact class/version
- mathematical interpretation
- tensor shapes
- extraction API
- caveats
