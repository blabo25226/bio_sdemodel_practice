# Project scope

## Primary objective

LARRY in vitro 実データ上で学習した scDiffEq Neural SDE を、SINDy により sparse symbolic SDE へ蒸留できるかを検証する。

## In scope

- official scDiffEq baseline reproduction
- learned drift / diffusion inspection
- teacher-function sampling
- SINDy / sparse regression
- diffusion constraints
- Neural vs symbolic SDE simulation comparison
- real-data validation

## Out of scope for the first demo

- 全遺伝子 gene-space GRN の発見
- 新規 biological mechanism の断定
- genome-wide symbolic SDE
- causal regulatory edge の主張
- state-of-the-art benchmark の網羅

まず proof-of-concept を完成させる。
