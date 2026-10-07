# Scientific integrity

## 用語

- `Neural SDE`: scDiffEq teacher model
- `symbolic SDE`: SINDy / sparse symbolic approximation
- `observed data`: LARRY measurements
- `simulated data`: teacher / symbolic model から生成

これらを混同しない。

## 禁止する主張

次を証拠なしに書かない。

- 「真の SDE を発見した」
- 「diffusion が intrinsic biological noise を表す」
- 「この項が causal gene regulation を表す」
- 「trajectory を生成したのでデータ量が増えた」

## Identifiability

同じ観測分布を異なる drift / diffusion が説明し得る。symbolic distillation が成功しても、それはまず **teacher の近似**である。

## Latent coordinates

PC / latent variablesの式は gene-level mechanism ではない。

## Negative results

symbolic approximation が不安定・非疎・distribution を再現しない場合も正式な結果として保存する。
