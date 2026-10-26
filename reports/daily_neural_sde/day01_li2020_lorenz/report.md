# Day 1 — Li et al. (2020) Scalable Gradients for SDEs: stochastic Lorenz attractor

- Date: 2026-10-26 (Day 1)
- Paper: Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, David Duvenaud,
  *Scalable Gradients for Stochastic Differential Equations*, AISTATS 2020 (PMLR 108),
  arXiv:2001.01328. 公式実装: torchsde (`examples/latent_sde.py`)。
- 成果物: config `configs/daily_neural_sde_day01.json` / コード `src/daily_neural_sde/day01_lorenz.py`
  (共通部品 `src/daily_neural_sde/common.py`) / 数値 `outputs/daily_neural_sde/day01/results.json`、
  teacher 重み `teacher_weights.json`、symbolic モデル `symbolic_drift.json` / `symbolic_diffusion.json`、
  TeX `symbolic_sde.tex` / tests `tests/test_daily_neural_sde.py` (5 tests, 全て通過)。
- 実行: `python -m src.daily_neural_sde.day01_lorenz` (seed 0、PyTorch 2.14.1+cpu、約 6 分)。

## 1. 論文要約

### 1.1 問題と提案

Neural ODE の adjoint 感度法を SDE に一般化する。後方 Stratonovich SDE を解くことで、
ソルバ経由の backpropagation が O(L) メモリのところを、ほぼ定数メモリ・O(L log L) 時間で
勾配が得られる (Table 1)。加えて Brownian motion を seed のみから任意精度で再構成する
キャッシュ方式と、latent SDE に対する stochastic variational inference (SVI) への統合を示す。

### 1.2 実験 (§7.2, Appendix 9.10) と主張の範囲

合成データで latent SDE を学習し、(1) posterior が訓練データを再構成できること、
(2) prior が決定的でないこと、の確認を主目的とする。対象は 1 次元 geometric Brownian motion と
3 次元 stochastic Lorenz attractor。実データは 50 次元 motion capture (test MSE で ODE2VAE 等に優位)。
**drift / diffusion の関数としての回復精度は論文の主張ではない**点に注意する。

Lorenz 実験の真値 SDE (Appendix 9.10.2、対角・定数 diffusion):

$$
\begin{aligned}
\mathrm{d}X_t &= \sigma (Y_t - X_t) \,\mathrm{d}t + \alpha_x \,\mathrm{d}W_t,\\
\mathrm{d}Y_t &= (X_t (\rho - Z_t) - Y_t) \,\mathrm{d}t + \alpha_y \,\mathrm{d}W_t,\\
\mathrm{d}Z_t &= (X_t Y_t - \beta Z_t) \,\mathrm{d}t + \alpha_z \,\mathrm{d}W_t,
\end{aligned}
\qquad
\sigma = 10,\ \rho = 28,\ \beta = \tfrac{8}{3},\ (\alpha_x,\alpha_y,\alpha_z) = (0.15, 0.15, 0.15).
$$

データは $(x_0,y_0,z_0) \sim \mathcal{N}(0, I)$ から 1024 系列、観測間隔 0.025 で $t \in [0,1]$
(41 点)、次元ごとに平均・標準偏差で正規化し、標準偏差 0.01 の Gaussian ノイズを加えたもの。
モデルは GBM 実験と同架構 (GRU encoder 100 hidden、prior/posterior drift は MLP 1 hidden 100、
diffusion は latent 次元ごとに小ネット ×4 → **latent 次元 4**、decoder は latent 4 次元から
観測への線形写像)、Adam (lr 0.01、0.999 減衰/iter)、KL 線形 annealing (50 iter)、固定刻み 0.01。
結果として posterior の再構成が良好で、prior サンプルが bimodal になることが Fig. 6 で示される。

## 2. データと Neural SDE の入手・再構成

### 2.1 データ

論文は生成データを公開していないため、Appendix 9.10.2 の設定どおり Euler–Maruyama
(dt = 0.001 の内部刻み) で再生成した (1024 系列 × 41 点 × 3 次元、観測ノイズ std 0.01 を付与)。
状態統計: 平均 (−0.056, −0.006, 20.85)、標準偏差 (8.89, 11.55, 15.31)、最大ノルム 54.9。
分割は trajectory-disjoint に train 640 / validation 192 / test 192 (seed 0)。

### 2.2 逸脱の明示 (teacher の再構成方法)

論文の latent SDE は 4 次元 latent・GRU encoder・線形 decoder・ELBO・確率 adjoint で学習されるが、
**latent 座標は識別不能**であり、その座標の symbolic 式は真値 Lorenz 式と比較できない。
本シリーズの目的 (多項式 SINDy で近似し式として評価する) のため、本日は範囲を切り取り、
**観測 3 次元座標で直接 Neural SDE を推定**した: drift / diffusion は別々の MLP
(Linear(3,64)–Softplus–Linear(64,64)–Softplus–Linear(64,3)、diffusion 出力は Softplus + 0.01、
入力は train 状態の平均・標準偏差で標準化)、対角 diffusion。これは論文のモデルの再現ではなく、
論文のデータ生成とモデル族 (drift・diffusion が NN の SDE) を引き継いだ再構成である。

### 2.3 推定器の比較 — 観測間隔の離散化バイアス

遷移尤度は 3 種を比較した (いずれも Gaussian 1-step 尤度、分散に観測ノイズ補正
$2 \times 0.01^2$ を含む):

| 推定器 | 内容 | drift NRMSE vs 真値 (test) | diffusion NRMSE vs 真値 (test) | $\hat g$ 平均 (validation 状態) |
|---|---|---|---|---|
| v1 `em` | 素朴な Euler–Maruyama 尤度 | 0.2147 | 2.0586 | (0.155, 0.544, 0.507) |
| v2 `local_linearization` | 平均に $\frac{1}{2}\Delta t^2 (J f)$、分散に $\frac{\Delta t^3}{3}\sum_j (\partial_j f_i)^2 g_j^2$ の補正 | 0.0438 | 1.1428 | (0.152, 0.388, 0.323) |
| v3 `local_linearization_exp` | 局所線形化を行列指数で再和約 (平均 $\Psi f$、$\Psi = \int_0^{\Delta t} e^{As}ds$)、分散は対角 Van Loan 因子 $\kappa_i g_i^2$ + 非対角 $\frac{\Delta t^3}{3}$ 項 | **0.0283** | 1.5439 | (0.166, 0.500, 0.337) |

読み取り:

- v1 は決定的にバイアスする。真値 diffusion は全成分 0.15 の定数だが、線形な第 1 成分だけが
  0.155 と正しく、双線形項を持つ第 2・3 成分が 0.54 / 0.51 に膨らむ。$\Delta t = 0.025$ の区間内で
  drift が曲がる分を、1-step 尤度が diffusion に誤帰属する構造的バイアスである
  (validation の標準化残差分散は 0.89 と ≈1 で、尤度内部では検出できない)。
- 補正で drift は 0.215 → 0.044 → 0.028 と大きく改善する。一方 diffusion は v2 が最も真値に近く
  (それでも NRMSE 1.14)、v3 は平均の再和約で分散モデルが小さくなった分だけ $g$ をさらに膨らませた。
  **この観測間隔・1-step Gaussian 尤度の範囲では、3 つの推定器いずれでも定数 diffusion 0.15 を
  第 2・3 成分について回復できなかった** (本日の主要な負の結果の一つ)。
- 蒸留対象の teacher は、事前に config で宣言したとおり v3 とした (drift が最良で、理論的にも
  再和約が最も忠実)。真値比較は合成ベンチマークとしての診断であり、symbolic 側の選択には
  一切使っていない。

## 3. 仮説 (実行前に config へ凍結)

- **H1 (drift 蒸留)**: symbolic drift の test NRMSE (対 teacher) ≤ 0.10、かつ次数 2 の最良
  validation NRMSE が次数 3 の 10% 以内 (真の drift が次数 2 なので次数 2 で足りるはず)。
- **H2 (diffusion 蒸留)**: symbolic diffusion の test NRMSE (対 teacher) ≤ 0.20、かつ
  非定数項の寄与 (test query での RMS 比) ≤ 0.25 (teacher の diffusion はほぼ定数のはず)。
- **H3 (真値回復)**: 元座標展開した symbolic drift が真値の 7 単項式全ての符号を一致させ、
  各係数の相対誤差 ≤ 0.35、真値 support 外項の寄与 ≤ 0.15。
- **H4 (シミュレーション)**: symbolic SDE の失敗率 ≤ 0.05 かつ neural との差 ≤ 0.02、
  paired RMSE (対 neural) の中央値 ≤ 2.0。

## 4. 方法 (main と同じ流儀)

- teacher への直接 query で function distillation (有限差分は使わない)。query 点は各 split の
  観測状態 (trajectory-disjoint)。変換 (mean/scale) と RMS 正規化は train query のみから。
- 候補: drift は次数 {2, 3}、diffusion は次数 {0, 1, 2}、STLSQ しきい値 {0.02, 0.05, 0.1, 0.2, 0.35}。
  field ごとに validation NRMSE で選択し、test は確定後に 1 回だけ評価。
- oracle 診断: 同じ SINDy 手順を真値 field に fit (候補ではなく天井の診断)。
- シミュレーション検証: test 系列の $t=0$ 観測状態 192 点から、neural / symbolic / 真値の
  3 モデルを同一 Brownian 増分 (paired) で Euler–Maruyama (dt = 0.0025, $t \in [0,1]$)。
  失敗は non-finite / ノルム > 200 で分類し、質量を保持する。

## 5. 結果

### 5.1 蒸留の精度

| field | 選択 (次数, しきい値, 活性項) | validation NRMSE | test NRMSE 対 teacher | test NRMSE 対真値 |
|---|---|---|---|---|
| drift | 次数 3, 0.02, 29 項 (延べ) | 0.0170 | **0.0174** | 0.0276 |
| diffusion | 次数 2, 0.02, 11 項 (延べ) | 0.0211 | **0.0217** | 1.5439 |

次数別の test NRMSE (対 teacher、各次数の validation 最良モデル): drift は次数 2 で 0.0244、
次数 3 で 0.0174。diffusion は次数 0 で 0.0856、次数 1 で 0.0252、次数 2 で 0.0217。
oracle (真値への fit) の天井は drift test NRMSE 0.0126 (次数 3 が選択された。真値は次数 2 だが、
後述の共線性としきい値のため次数 2 の oracle でも厳密な 0 にはならない)、diffusion は次数 0 で厳密に 0。
つまり **SINDy 蒸留そのものの誤差 (対 teacher 0.017) は oracle 天井 (0.013) に近く、
symbolic 側はほぼ最適**で、対真値の誤差は teacher 側の誤差が支配的である。

### 5.2 求めた symbolic SDE (TeX、元座標)

蒸留で得た SDE は次である ($W^{(i)}$ は独立な Wiener 過程、$\hat f$ / $\hat g$ は
`outputs/daily_neural_sde/day01/symbolic_sde.tex` の生成 TeX と同一):

$$
\mathrm{d}X_t = \hat f_1(X_t, Y_t, Z_t) \,\mathrm{d}t + \hat g_1(X_t, Y_t, Z_t) \,\mathrm{d}W^{(1)}_t, \quad
\mathrm{d}Y_t = \hat f_2 \,\mathrm{d}t + \hat g_2 \,\mathrm{d}W^{(2)}_t, \quad
\mathrm{d}Z_t = \hat f_3 \,\mathrm{d}t + \hat g_3 \,\mathrm{d}W^{(3)}_t
$$

```tex
\begin{aligned}
\hat f_1 &= - 0.5455 - 0.01369\,z + 8.88\,y - 7.612\,x + 0.000173\,z^{2} + 0.08777\,y z + 0.0002335\,y^{2} - 0.1548\,x z - 0.000253\,x y + 0.0004945\,x^{2} - 0.001431\,y z^{2} - 0.001871\,y^{3} + 0.002044\,x z^{2} + 0.003744\,x y^{2} - 0.002217\,x^{2} y + 0.001779\,x^{3} \\
\hat f_2 &= 1.846 - 0.05741\,z + 4.218\,y + 16.57\,x - 0.0007048\,z^{2} + 0.02169\,y z - 0.001008\,y^{2} - 0.5565\,x z + 0.002765\,x y + 0.00388\,x^{2} - 0.004952\,y z^{2} - 0.005234\,y^{3} - 0.005433\,x z^{2} - 0.005461\,x y^{2} + 0.01473\,x^{2} y + 0.01109\,x^{3} \\
\hat f_3 &= 0.04695 - 2.629\,z + 0.1045\,y + 0.05225\,x - 0.01891\,z^{2} + 3.858e-06\,y z - 0.02085\,y^{2} - 0.0007237\,x z + 1.013\,x y + 0.1148\,x^{2} + 0.0005978\,z^{3} + 6.903e-05\,y^{2} z - 0.003471\,x^{2} z \\
\end{aligned}
\begin{aligned}
\hat g_1 &= 0.1762 - 3.029e-06\,z + 0.0003343\,y + 0.002506\,x - 4.248e-05\,y^{2} - 2.905e-05\,x z + 8.135e-05\,x y - 0.0001387\,x^{2} \\
\hat g_2 &= 0.4998 + 0.003876\,y \\
\hat g_3 &= 0.3368 + 0.001754\,x \\
\end{aligned}
```

支配項だけ抜き出すと $\hat f_1 \approx -7.61 x + 8.88 y$、$\hat f_2 \approx 16.57 x + 4.22 y - 0.557 xz$、
$\hat f_3 \approx 1.013 xy - 2.629 z$ で、$\hat g \approx (0.176,\ 0.500,\ 0.337)$ とほぼ定数である。
3 次項は係数が小さいが、状態の 3 乗と掛かるため寄与は無視できない (5.3 節)。

### 5.3 真値との係数比較 (H3 の内訳)

| 成分 | 単項式 | 真値 | fit | 相対誤差 | 符号 |
|---|---|---|---|---|---|
| $f_1$ | $x$ | −10 | −7.612 | 0.239 | ✓ |
| $f_1$ | $y$ | +10 | +8.880 | 0.112 | ✓ |
| $f_2$ | $x$ | +28 | +16.573 | 0.408 | ✓ |
| $f_2$ | $y$ | −1 | **+4.218** | 5.218 | ✗ |
| $f_2$ | $xz$ | −1 | −0.557 | 0.444 | ✓ |
| $f_3$ | $xy$ | +1 | +1.013 | 0.013 | ✓ |
| $f_3$ | $z$ | −8/3 | −2.629 | 0.014 | ✓ |

真値 support 外項の寄与は RMS 比 0.4516 (しきい値 0.15 超過)。一方 $f_3$ はほぼ完璧に回復した。
失敗の主因は共線性である: test 状態で $\mathrm{corr}(x, xz) = 0.931$ ($z$ が平均 20.8、
標準偏差 15.3 と $\rho = 28$ の近傍に分布し、$x(\rho - z)$ の 2 項が強く相関する)、
$f_2$ の $x$ / $xz$ / $y$ への配分はデータ多様体上ではほぼ同値な表現が多数ある。
field としては正しい (対真値 drift NRMSE 0.028) が、係数の同定はこの query 分布ではできない。

### 5.4 シミュレーション検証 (paired、192 系列、$t \in [0,1]$)

| モデル | 失敗率 | 終端平均 (finisher) | 終端 std (finisher) |
|---|---|---|---|
| neural (teacher v3) | 0.000 | (−0.07, −0.28, 26.75) | (8.60, 9.13, 1.14) |
| symbolic | 0.000 | (−0.88, −0.85, 26.35) | (8.36, 8.87, 0.64) |
| 真値 | 0.000 | (−0.57, −0.61, 26.82) | (8.31, 8.59, 0.60) |

paired RMSE 中央値: symbolic 対 neural **1.495**、neural 対真値 2.248、symbolic 対真値 2.973。
symbolic 化による追加の乖離は teacher 自身の真値からの乖離より小さい。

### 5.5 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1 drift 蒸留 | **否** | test 条項は満たす (0.0174 ≤ 0.10) が、簡潔性条項が失敗: 次数 2 の validation 0.0242 は次数 3 の 0.0170 の 10% 以内ではない (42% 悪化)。teacher は滑らかな非多項式で、その微細構造は次数 3 で拾われる |
| H2 diffusion 蒸留 | **成** | test 0.0217 ≤ 0.20、非定数寄与 0.0837 ≤ 0.25。ただしこれは teacher の写像としての合格で、teacher 自体の diffusion が真値からずれていることは別問題 (§2.3) |
| H3 真値回復 | **否** | 符号不一致 1 件 ($f_2$ の $y$)、相対誤差 > 0.35 が 2 件、support 外寄与 0.4516 > 0.15。field 水準の回復と係数水準の回復は別物であることが明確になった |
| H4 シミュレーション | **成** | 失敗率 0 (差 0)、paired RMSE 1.495 ≤ 2.0 |

## 6. 考察

1. **蒸留 ≠ 真値回復**が本日最も明瞭に定量化された。SINDy 側は oracle 天井近くまで teacher を
   写し取る (0.017 vs 0.013) が、teacher のバイアス (特に diffusion) と同定可能性の限界
   (共線性) はそのまま式に継承される。main の LARRY 解析で「symbolic 化の成功はまず teacher の
   近似である」とされた原則が、真値既知の合成系でも係数レベルでは強く現れる。
2. **推定器の離散化バイアスは論文の観測間隔で決まる**。Li et al. の設定 ($\Delta t = 0.025$) では
   素朴な EM 尤度は使い物にならない水準で diffusion を膨らませる。論文自身は latent SDE を
   連続時間のまま SVI + adjoint で学習しており、この種の 1-step 尤度の罠を回避している —
   再構成で学習法を置換する場合、その置換が推定対象を変えることの実例になった。
3. **次数 3 が選ばれたことは「真の系が次数 3」という意味ではない**。teacher が MLP である以上、
   蒸留の最適次数は teacher の滑らかさで決まる。次数 2 でも test NRMSE 0.0244 と実用水準で、
   式の解釈可能性を優先するなら次数 2 を採る判断もあり得る (ただし本シリーズでは
   validation 選択の規律を優先し、事後に入れ替えない)。
4. diffusion の絶対水準の回復は未解決のまま残った。1-step 尤度の補正次数を上げる方向は
   v2 → v3 で既に非単調であり、正攻法は内部 substep を持つシミュレーションに基づく推定
   (shooting / moment matching) か、より細かい観測間隔のデータを前提とする方法である。

## 7. 反省 (Day 2 以降に適用)

- 推定器を選ぶ前に離散化バイアスの診断を組み込む: 線形に近い成分と非線形な成分で
  $\hat g$ を比較する、補正次数を変えて安定性を見る、を標準手順にする。
- 係数回復を仮説に含める日は、query 特徴の共線性診断 (相関・VIF) を仮説とセットで事前に置く。
  共線性が高い場合は「field 水準」と「係数水準」の仮説を分けて立てる。
- diffusion の絶対水準が主題の日は、1-step 尤度ではなく内部 substep のシミュレーション
  に基づく推定を最初から使う。

## 8. 限界

- 本日の teacher は論文の latent SDE そのものではない (§2.2 の逸脱)。論文の prior が作る
  bimodal 分布の再現性など、論文本来の評価軸は本日は扱っていない。
- データは再生成であり、論文の乱数系列そのものではない。統計的設定は同一。
- シミュレーション horizon は論文と同じ $t \in [0,1]$ のみ。長期の attractor 統計
  (滞在時間、Lyapunov 指数など) は未検証。
- 既存 repo tests (`tests/test_distill.py` 等) は本環境 (environment.yml の完全環境ではない)
  では未実行。本日追加した `tests/test_daily_neural_sde.py` の 5 tests は全て通過。
