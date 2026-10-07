# bio_sdemodel_practice

2026-10-07 の指示書に沿った取得・学習・蒸留・検証を実行済み。
**5-PC の多項式 drift/G は局所近似できたが、symbolic SDE は4日間の積分で
58.6%の軌道が数値的不安定となり、M3/M4 は未達。中心仮説は支持されなかった。**
結果・制約・再現手順は [実験報告](docs/1007_results.md)、
実行済みの説明用 Notebook は [notebooks](notebooks/README.md) にある。

## 目的

実データ **LARRY in vitro hematopoietic differentiation** と **scDiffEq** を使って Neural SDE を再現し、学習済み SDE の drift / diffusion を **SINDy によって解釈可能な数式へ蒸留する**デモ研究を行う。

最終的に検証したい流れは次の通り。

```text
real single-cell data (LARRY)
        ↓
scDiffEq / Neural SDE
        ↓
f_θ(x), G_θ(x) あるいは D_θ(x)=G_θ(x)G_θ(x)^T
        ↓
SINDy / sparse symbolic distillation
        ↓
interpretable symbolic SDE
        ↓
Neural SDE・実データとの再検証
```

SDE は概念的に

\[
dX_t=f_\theta(X_t,t)\,dt+G_\theta(X_t,t)\,dW_t
\]

と書く。SINDy では、まず drift を

\[
f_\theta(x)\approx \Theta(x)\Xi_f
\]

と疎な関数基底で近似する。diffusion については scDiffEq の実装上の `G` の形状・noise type を確認してから、`G` 自体または

\[
D(x)=G(x)G(x)^\top
\]

を対象にする。**対角 diffusion だと決めつけないこと。**

## 先行研究

中心となる先行研究は以下。

- Vinyard et al., *Learning cell dynamics with neural differential equations*, Nature Machine Intelligence (2025)
- preprint: *scDiffEq: drift-diffusion modeling of single-cell dynamics with neural stochastic differential equations*
- LARRY: Weinreb et al., *Lineage tracing on transcriptional landscapes links state to fate during differentiation*, Science (2020)

アクセス先は [`source.md`](source.md) を参照。

## 最初のデモ範囲

最初から全細胞・全遺伝子・高次元 symbolic model を狙わない。

1. 公式 Quickstart と同じ LARRY in vitro データを取得する。
2. `Undifferentiated / Monocyte / Neutrophil` と、Monocyte/Neutrophil fate を持つ clone に絞る。
3. 公式 scDiffEq baseline を再現する。
4. 学習済み Neural SDE の drift / diffusion を数値評価できることを確認する。
5. まず drift の symbolic distillation を成功させる。
6. その後 diffusion に進む。
7. 最後に symbolic SDE をシミュレーションし、Neural SDE と実データに対して検証する。

## 重要な研究上の方針

### 1. まず再現、次に拡張

scDiffEq の再現と SINDy 拡張を同時にデバッグしない。必ず以下を分離する。

```text
Phase A: official baseline reproduction
Phase B: inspect learned SDE
Phase C: symbolic distillation
Phase D: symbolic SDE validation
```

### 2. trajectory を大量生成するだけでは情報は増えない

Neural SDE から生成した trajectory は teacher model の情報を高密度に問い合わせたものに過ぎない。したがって最初の SINDy は、可能なら

```text
state x → neural drift f_θ(x)
state x → neural diffusion G_θ(x) / D_θ(x)
```

を直接教師として使う。

trajectory-based stochastic SINDy は比較実験として後から追加する。

### 3. latent space と gene space を混同しない

scDiffEq の標準状態は PCA latent space。最初に得られる symbolic SDE が PC 座標上の式なら、それを「遺伝子間制御式」と解釈してはいけない。

### 4. symbolic model は元データでも再検証する

SINDy が Neural SDE に近いだけでは不十分。

- drift / diffusion function error
- generated distribution
- fate proportion / fate probability
- time-point distribution
- transition behavior

などを、可能な範囲で元の LARRY データと比較する。

## 推奨フェーズ

### Phase A — scDiffEq baseline

- `scdiffeq` のバージョンを記録
- LARRY を公式 loader で取得
- Quickstart subset を再現
- drift / diffusion field を可視化
- trajectory simulation を再現
- seed・依存関係・GPU/CPU 情報を保存

### Phase B — SDE inspection

- `model.DiffEq` の class を記録
- `sde_type`, `noise_type`, `brownian_dim`, latent dimension を記録
- drift NN と diffusion NN の入力・出力 tensor shape をテスト
- `model.drift()` / `model.diffusion()` の出力意味を確認
- `X_drift`, `X_diffusion`, `drift`, `diffusion` の shape と定義を確認

### Phase C — SINDy distillation

最初は低次数の polynomial library から始める。

```text
degree 1 → degree 2 → 必要なら degree 3
```

比較する optimizer 候補:

- STLSQ
- SR3
- Lasso 系

最低限、threshold / regularization strength に対する安定性を見る。

高次元 50-PC のまま可否を確認しつつ、解釈可能なデモ用には 3–10 次元程度の低次元 scDiffEq model を別途学習する案も検討する。**低次元化してから teacher を変える場合は「50D model の蒸留」ではなく「低次元 Neural SDE の蒸留」であることを明記する。**

### Phase D — validation

Neural SDE と symbolic SDE を同一初期条件・同一時間区間で多数回 simulate して比較する。

最低限:

- pointwise drift error
- pointwise diffusion error
- sparsity / number of active terms
- time-point distribution distance
- qualitative trajectory / vector-field agreement

可能なら:

- fate distribution
- Wasserstein / Sinkhorn distance
- bootstrap stability
- held-out state region での一般化

## 想定する将来のディレクトリ

実装を開始したら、以下を目安にする。

```text
bio_sdemodel_practice/
├── .agents/
├── README.md
├── 1007_instruction.md
├── source.md
├── AGENTS.md
├── CLAUDE.md
├── notebooks/
│   ├── 00_data_check.ipynb
│   ├── 01_scdiffeq_baseline.ipynb
│   ├── 02_inspect_sde.ipynb
│   ├── 03_sindy_drift.ipynb
│   ├── 04_sindy_diffusion.ipynb
│   └── 05_validation.ipynb
├── src/
│   ├── data.py
│   ├── neural_sde.py
│   ├── distill.py
│   ├── simulate.py
│   └── metrics.py
├── configs/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── logs/
└── tests/
```

notebook に重要ロジックを閉じ込めず、再利用する処理は `src/` に移す。

## エージェント向け

Codex / Claude は、作業開始前に以下を読むこと。

1. `1007_instruction.md`
2. `.agents/rules/` 以下すべて
3. 該当タスクの `.agents/skills/`
4. `source.md`

`AGENTS.md` と `CLAUDE.md` は入口ファイルであり、詳細ルールは `.agents/` に集約する。
