# 2026-10-07 指示書 — scDiffEq → SINDy demo research

## 0. プロジェクトのゴール

LARRY in vitro の実 single-cell lineage-tracing データを使い、scDiffEq で Neural SDE を学習または公開 checkpoint から復元し、その drift / diffusion を SINDy で解釈可能な SDE に蒸留する proof-of-concept を構築する。

中心仮説:

> 柔軟な Neural SDE を teacher として用いることで、raw noisy single-cell snapshots から直接 stochastic SINDy を行うよりも、滑らかな drift / diffusion field を介して安定した symbolic SDE を得られる可能性がある。

ただし、この仮説を最初から正しいものとして扱わない。必ず baseline と比較する。

---

## 1. 絶対に守る順序

### Step 1 — 環境・データ再現

1. Python / PyTorch / CUDA / scdiffeq / torchsde / anndata / scanpy のバージョンを記録。
2. `sdq.datasets.larry()` で LARRY in vitro を取得。
3. データ shape、time points、cell types、PCA dimension を確認。
4. 公式 Quickstart と同じ Monocyte / Neutrophil / Undifferentiated subset を再現。
5. ここまでを notebook と log に残す。

### Step 2 — scDiffEq baseline

1. まず公式 Quickstart を可能な限りそのまま再現。
2. train 前に seed を固定。
3. GPU メモリ、学習時間、train/validation loss を記録。
4. `model.drift()` と `model.diffusion()` を実行。
5. learned field と trajectory simulation が公式例と定性的に一致することを確認。

公開 checkpoint を使える場合は、**最初の symbolic distillation は checkpoint で先に試してよい**。ただし最終的には自前 training の再現も行う。

### Step 3 — learned SDE の構造確認

SINDy を書く前に必ず確認する。

- `type(model.DiffEq)`
- latent dimension
- `sde_type`
- `noise_type`
- `brownian_dim`
- drift network の出力 shape
- diffusion network の出力 shape
- `G(x)` の数学的意味
- `D(x)=G(x)G(x)^T` を構成すべきか
- time dependence が network input に明示的に入っているか

**「diffusion は対角」「各遺伝子に独立 Brownian motion」などを推測で置かない。**

結果は `outputs/logs/model_semantics.md` に記録する。

### Step 4 — drift symbolic distillation

最初の成功条件は drift のみ。

状態点 `x_k` を training data / held-out data / teacher simulation の複数ソースから集め、teacher の

\[
y_k=f_\theta(x_k)
\]

を評価する。

候補 library:

1. constant + linear
2. polynomial degree 2
3. polynomial degree 3 は必要な場合のみ

まず PySINDy の STLSQ を baseline とする。

記録するもの:

- library size
- threshold
- alpha / regularization
- active term count
- train / validation MSE
- normalized error
- bootstrap selection frequency

### Step 5 — diffusion symbolic distillation

Drift が成功してから進む。

scDiffEq の noise structure を確認した上で、次のどれを蒸留するか決定する。

```text
A. G_θ(x) を直接蒸留
B. D_θ(x)=G_θ(x)G_θ(x)^T を蒸留
C. scalar diffusion magnitude のみ（可視化デモ）
```

研究として優先するのは A または B。C は proof-of-concept 可視化に限定。

正定値性が必要な `D` を各成分独立に SINDy fit すると PSD が壊れる可能性がある。必要なら

\[
D(x)=L(x)L(x)^T
\]

のような constrained parameterization を検討する。

### Step 6 — symbolic SDE simulation

得られた symbolic model を数値積分可能な関数に変換し、Neural SDE と同じ初期状態集合から simulation する。

比較:

- drift field
- diffusion field
- trajectory distribution
- 各観測時点の分布
- fate composition
- distribution distance

### Step 7 — direct stochastic-SINDy baseline

可能なら raw / observed transition information から直接 SINDy 系を適用する baseline を作る。

目的は

```text
raw → SINDy
vs
raw → Neural SDE → SINDy
```

の比較。

LARRY は同一細胞の連続 trajectory ではないため、direct baseline の意味・仮定を明記すること。無理に公平でない比較を作らない。

---

## 2. 最初に作る成果物

優先順位順:

1. `notebooks/00_data_check.ipynb`
2. `notebooks/01_scdiffeq_baseline.ipynb`
3. `outputs/logs/model_semantics.md`
4. `notebooks/02_inspect_sde.ipynb`
5. `src/distill.py`
6. `notebooks/03_sindy_drift.ipynb`
7. `notebooks/04_sindy_diffusion.ipynb`
8. `notebooks/05_validation.ipynb`

notebook は説明・可視化用。再利用処理は `src/` に置く。

---

## 3. 最初の成功判定

### Milestone M0 — Data

- LARRY が再現可能に取得できる
- subset の cell 数と class / time point 分布を保存できる

### M1 — Neural SDE

- checkpoint load または training が成功
- 任意 batch に対して drift / diffusion の数値評価ができる
- trajectory simulation が動く

### M2 — Drift SINDy

- held-out points で teacher drift を再現
- dense NN に比べて active term 数が十分少ない
- threshold を少し変えても主要項が極端に崩れない

### M3 — Symbolic SDE

- diffusion まで symbolic / constrained symbolic representation を得る
- simulation が数値的に破綻しない

### M4 — Biological demo

- Neural SDE と symbolic SDE が Monocyte / Neutrophil 分岐に関する主要な distributional behavior をある程度保持する

---

## 4. やってはいけないこと

- UMAP 2D 上の見た目だけで SDE が正しいと結論する。
- PCA/latent coordinate の式を直接 gene regulation mechanism と呼ぶ。
- Neural SDE から大量 trajectory を生成しただけで「データ量が増えた」と主張する。
- test data を symbolic library / threshold tuning に漏らす。
- diffusion tensor の意味を確認せず scalar diffusion と混同する。
- SINDy が綺麗な式を返したこと自体を biological discovery とする。
- 先行研究のコードをコピーした場合に出典・license を消す。
- package version を固定せず、再現性のない notebook だけ残す。

---

## 5. Codex / Claude の役割

どちらのエージェントも同じ scientific rules に従う。役割は固定しないが、並列に使う場合は次の分担が望ましい。

### Agent A: implementation / reproducibility

- package setup
- official Quickstart reproduction
- checkpoint / model API inspection
- tests
- clean reusable code

### Agent B: methodology / audit

- SINDy formulation
- tensor semantics audit
- identifiability / leakage checks
- validation design
- literature/source verification

Agent B は Agent A の結果を無条件で信用せず、shape・式・評価方法を独立に確認する。

---

## 6. 現時点での研究上の問い

1. scDiffEq の learned drift は低次数 sparse library でどこまで近似可能か。
2. diffusion NN は drift より symbolic distillation が難しいか。
3. latent dimension を増やしたとき sparsity / fidelity はどう変化するか。
4. teacher query points の選び方で symbolic model は変化するか。
5. symbolic SDE は teacher の fate distribution をどこまで保存するか。
6. symbolic model が teacher の extrapolation error まで蒸留していないか。
7. 低次元 latent symbolic dynamics を gene space へどう解釈すべきか。

---

## 7. 研究ノートの書式

各実験で必ず残す。

```text
Date:
Git commit:
Environment:
Data variant/subset:
Random seed:
Model/checkpoint:
Hypothesis:
Change from previous run:
Metrics:
Figures:
Result:
Interpretation:
Failure / caveats:
Next action:
```

「成功した run」だけでなく失敗 run も残す。

## 8. ユーザー指定の更新 (2026-10-07)

SINDy の候補関数は理解しやすさのため **べき関数とその交差項からなる多項式のみ** とする。
多変量状態では `prod_i x_i^n_i` (各 n_i は非負整数、総次数 <= p) を用いる。
定数項は一つだけ。`x_i x_j`, `x_i^2 x_j` 等の交差項を含める。
三角関数、指数関数、対数関数は含めない。
次数は degree 1 → 2 → 必要な場合のみ 3 の順序を維持する。
モデル選択には validation を使い、test は使わない。
この制限は drift と diffusion の sparse fit の両方に適用する。
ユーザーの補足により、交差項を除外した初期解釈は撤回した。
