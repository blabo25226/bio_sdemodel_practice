# Day 4 — Gierjatowicz et al. (2020) Robust pricing and hedging via neural SDEs: LSV benchmark

- Date: 2026-10-29 (Day 4)
- Paper: Patryk Gierjatowicz, Marc Sabate-Vidales, David Šiška, Lukasz Szpruch, Žan Žurič,
  *Robust pricing and hedging via neural SDEs*,
  arXiv:2007.04154 (submitted 8 Jul 2020).
- 成果物: config `configs/daily_neural_sde_day04.json` / コード `src/daily_neural_sde/day04_gierjatowicz_lsv.py`
  (共通部品 `src/daily_neural_sde/common.py` を再利用、ただし全行列 diffusion の paired シミュレーションは本日モジュール内の
  `simulate_matrix` で実装) / 数値 `outputs/daily_neural_sde/day04/results.json`、
  teacher 重み `teacher_weights.json`、symbolic モデル `symbolic_drift.json` / `symbolic_diffusion.json`、
  TeX `symbolic_sde.tex` / tests `tests/test_daily_neural_sde_day04.py` (4 tests 追加、Day 1–3 分と合わせて 17 tests 全て通過)。
- 実行: `python -m src.daily_neural_sde.day04_gierjatowicz_lsv` (seed 0、PyTorch 2.15.0+cpu、約 6 分)。
  注意: 本環境のシステム Python には PyTorch が無く `/tmp` (tmpfs 512MB) には CPU wheel が入らないため、
  `~/.venv-day04` に venv を作って実行した (repo 外の作業環境であり、成果物には含めない)。

## 0. 前日までの反省の適用 (本日の分析に組み込んだもの)

1. **推定器の離散化バイアス診断 (Day 1)**: EM (v1) と内部 substep (v4、K = 8) の 2 推定器で
   drift / Sigma の真値との NRMSE を比較し、さらに分散レジーム (low v < 0.02 / mid / high v > 0.06) ごとに分けて報告した。
   本日のバイアス経路は状態依存 diffusion の凍結 (区間内で sqrt(v) が動くこと) であり、
   Day 1 の「線形 vs 非線形」や Day 3 の「sigmoid 飽和レジーム」の対応物は low-v レジームである。
   蒸留する teacher は事前に v4 と宣言した (diffusion の絶対水準と状態依存性が主題のため)。
   ただし v4 の共分散伝播は drift-Jacobian 項を省略している (config に明記の限定)。
2. **off-manifold 格子診断の標準化 (Day 2)**: x ∈ {−0.6, −0.3, 0, 0.3, 0.6}、v ∈ {0.005, 0.02, 0.04, 0.08, 0.16} の 25 点格子で
   symbolic / teacher / 真値を比較し、field 水準・係数水準・off-manifold 水準の 3 軸で分けて報告した。
   v = 0.16 列は別途「tail 列」として分離した (§5.6)。
3. **NLL の順序を過大評価しない (Day 2)**: validation NLL は v4 (−5.18374) が v1 (−5.17615) より良いが、
   drift の順序は field NRMSE で判断し、NLL の順序は系統的な優劣と読まない。実際、本日は NLL と drift の順序が逆転した (§5.1)。
4. **共線性ゲート (Day 1・2)**: fit の前に train query 状態で corr(x, v)、VIF、次数 2 特徴量間の最大相関を計算した。
   VIF = 1.76 で 10 以下だが、corr(x, v) = −0.658 と次数 2 特徴量間の最大相関 0.924 は高く、
   本日の共線性は「多様体が斜めに潰れている」形である (§5.4)。係数水準の解釈は行わない (teacher は汎用 MLP)。
5. **diffusion 比較の同点域 (Day 3)**: H5 の diffusion 条項には Sigma NRMSE の差 < 0.01 を同点域として事前に設けた。
   実際には差は 0.018 で同点域を上回り、v4 の Sigma 優位は同点ではない (§5.1)。H5 の判定は drift 条項に置いた。
6. **共通乱数 (Day 3 のメッシュ研究の教訓)**: paired シミュレーションと価格計算 (H6) は全モデルで同一の Brownian 増分を使い、
   モデル差が独立なモンテカルロ誤差に埋もれないようにした。価格差には paired 標準誤差を併記した。
7. **off-manifold 格子とデータ範囲の重なり (Day 3)**: 本日の格子は周辺範囲ではデータ範囲と重なる
   (x 範囲 [−1.115, 0.554]、v 範囲 [−0.00002, 0.277] に対し格子はその内側) ため、強い外挿テストとは主張しない。
   格子の診断力は周辺範囲ではなく (x, v) の**同時分布からの乖離** (斜めの多様体を外れた組み合わせ) にある (§5.6)。

## 1. 論文要約

### 1.1 問題と提案

市場モデルは本質的に不確実であり、1 つの手作りモデル (Black–Scholes、local volatility、Heston など) を
市場価格に較正しても、同じ市場価格に整合するモデルは無数に存在する (Knightian uncertainty)。
論文はこのモデル選択・モデル不確実性の問題に対し、drift / diffusion をニューラルネットで与えた
neural SDE を提案する。neural SDE は、市場価格への較正とモデル選択を同時に行い、
さらに因果的最適輸送 (causal optimal transport) の双対を通じて、流動性の低いデリバティブ
(例: lookback オプション) の価格の頑健な上下界と対応するヘッジ戦略を与える枠組みである。
リスク中立測度 Q での較正と実測度 P での較正を、Girsanov によるドリフトシフト (market price of risk ζ) で
整合的に接続できる点も主張される。

### 1.2 モデル (§1.2, §5)

取引資産 S と非取引成分 V からなる Itô 過程

$$
\mathrm{d}S_t = r S_t \,\mathrm{d}t + \sigma^S(t, X_t, \theta) \,\mathrm{d}W_t, \qquad
\mathrm{d}V_t = b^V(t, X_t, \theta) \,\mathrm{d}t + \sigma^V(t, X_t, \theta) \,\mathrm{d}W_t
$$

を考える。$e^{-rt} S_t$ が (local) マルチンゲールになるよう $S$ の drift は $rS_t$ に固定される。
数値実験 (§5) は 2 系統である:

- **LV (local volatility) モデル (5.2)**: $\mathrm{d}S_t = r S_t \mathrm{d}t + \sigma(t, S_t, \theta) S_t \mathrm{d}W_t$、$S_0 = 1$。
  Dupire の公式により、連続体の call/put 価格があれば一意の $\sigma$ が存在する。
- **LSV (local stochastic volatility) モデル (5.3)**:
  $\mathrm{d}S_t = r S_t \mathrm{d}t + \sigma^S(t, S_t, V_t, \nu) S_t \mathrm{d}B^S_t$、
  $\mathrm{d}V_t = b^V(V_t, \phi) \mathrm{d}t + \sigma^V(V_t, \varphi) \mathrm{d}B^V_t$、
  $\mathrm{d}\langle B^S, B^V \rangle_t = \rho \,\mathrm{d}t$、$S_0 = 1$、$V_0 = v_0$。
  $\sigma^S, b^V, \sigma^V$ は maturity ごとに 4 隠れ層 × 50 ニューロンの ReLU ネット (diffusion 出力には softplus)、
  ヘッジ戦略も別ネットで学習し、tamed Euler (T = 1 年で 96 ステップ)、SGD 1 反復あたり $4 \times 10^4$ path、
  評価は $4 \times 10^5$ path (antithetic 込み) である。

較正対象は満期 2, 4, …, 12 か月、ストライク 21 点 ([0.8, 1.2] に均等配置) のヨーロピアン・コール価格で、
非流動性デリバティブの例として lookback ペイオフ $\Psi = \max_{t} S_t - S_T$ の価格区間を求める。
報告される主張は、市場価格への較正 MSE が 6 か月満期で約 $10^{-9}$、12 か月で約 $10^{-8}$ と高いこと、
ストライク数を増やすと lookback 価格の区間が狭まること、市場データが限られると区間が目に見えて広がることである。
**本日はこの較正主張をテストしない** (§2.2 の逸脱)。テストするのは LSV の neural SDE 構造
(状態依存 diffusion + 相関 Brownian + 正の分散というドメイン制約) の SINDy 蒸留可能性である。

## 2. データと Neural SDE の入手・再構成

### 2.1 データ

論文は市場オプション価格を較正対象としており、軌道データセットを公開していない。
本日は LSV 形式 (5.3) のベンチマーク実例として、次の Heston 型システムを宣言して再生成した
(パラメータは標準的な Heston 較正値であり、論文の市場較正からの推定値ではない):

$$
\mathrm{d}x_t = (r - \tfrac{1}{2} v_t)\,\mathrm{d}t + \sqrt{v_t}\,\mathrm{d}B^S_t, \qquad
\mathrm{d}v_t = \kappa(\theta - v_t)\,\mathrm{d}t + \xi\sqrt{v_t}\,\mathrm{d}B^V_t, \qquad
\mathrm{d}\langle B^S, B^V \rangle_t = \rho\,\mathrm{d}t
$$

ここで $x_t = \log S_t$、$(r, \kappa, \theta, \xi, \rho) = (0.02, 2.0, 0.04, 0.30, -0.70)$、
$x_0 = 0$ ($S_0 = 1$)、$v_0 = 0.04$ である。Feller 条件 $2\kappa\theta = 0.16 > \xi^2 = 0.09$ を満たす。
独立 Brownian $(W^1, W^2)$ による Cholesky 表示では diffusion 行列は

$$
L(x, v) = \begin{pmatrix} \sqrt{v} & 0 \\ \xi\rho\sqrt{v} & \xi\sqrt{1-\rho^2}\sqrt{v} \end{pmatrix},
\qquad \Sigma = L L^\top = \begin{pmatrix} v & \rho\xi v \\ \rho\xi v & \xi^2 v \end{pmatrix}
$$

である。生成は full-truncation Euler ($dt = 0.001$、各ステップで $v_+ = \max(v, 0)$ を drift と sqrt に使う)、
$0.05$ ごとに観測 (4096 系列 × 21 点、$t \in [0, 1]$)。
状態統計: x の平均 0.0001、標準偏差 0.143、範囲 [−1.115, 0.554]。
v の平均 0.0399、標準偏差 0.0259、範囲 [−0.00002, 0.277]、観測点で v < 0 の割合は $1.2 \times 10^{-5}$ とごく小さい。
分割は trajectory-disjoint に train 2560 / validation 768 / test 768 (seed 0)。
x の範囲の非対称性 ([−1.115, 0.554]) は、レバレッジ ($\rho = -0.7$) と $-0.5v$ の drift drag の帰結である。

### 2.2 逸脱の明示

1. **市場較正をしない**: 論文の主張の中心 (市場価格への較正 MSE、lookback 価格区間、ヘッジ戦略) は本日テストしない。
   市場データへのアクセス、ヘッジネット、頑健区間の最適化は 1 日の CPU 規模を超えるためである。
   代わりに LSV 形式 (5.3) をベンチマーク真値として固定し、軌道尤度で neural SDE を推定して蒸留する。
   したがって本日の H6 は論文の主張の検証ではなく、論文の lookback ペイオフを借りた「蒸留の downstream 代理指標」である。
2. **ネットワーク規模**: 論文は maturity ごとに 4 × 50 ReLU ネットだが、本日は field ごとに 1 つの小さい MLP
   (隠れ層 [32, 32]、tanh) とし、plain Euler 遷移尤度で学習した。tamed Euler、ヘッジ control variate は使っていない。
3. **時間依存性の切り取り**: 論文の $\sigma^S(t, S, V)$ は $t$ にも依存しうるが、ベンチマークは時間斉次とし、
   対数座標では diffusion は $v$ のみに依存する ($\sqrt{v}$) 形に限定した。
4. **teacher のモデル族**: 論文の NN 族に沿った汎用 MLP teacher (drift MLP + Cholesky diffusion MLP) を使った。
   diffusion は $L_{11} = \mathrm{softplus} + 0.01$、$L_{22} = \mathrm{softplus} + 0.01$、$L_{21}$ は符号自由、$L_{12} = 0$ 固定である。
   この三角形式は識別可能性のための固定である: 尤度が識別するのは $\Sigma = L L^\top$ であって $L$ 自体ではなく、
   直交回転の自由度を $L_{11} \ge 0$、$L_{22} \ge 0$、$L_{12} = 0$ で固定して初めて $L$ が識別される。
   したがって diffusion の評価は $L$ 条項と $\Sigma$ 条項の両方で報告する。
5. **substep 推定器の限定**: v4 の共分散伝播は $V \leftarrow V + \Sigma(\bar x) h$ のみで、
   drift-Jacobian 項 $J V + V J^\top$ を省略している。これは本日の診断の問いが「$\Sigma$ の再評価経路」
   (状態依存 diffusion の凍結バイアス) であるための宣言された限定であり、厳密なモーメント伝播ではない。

## 3. 仮説 (実行前に config へ凍結)

- **H1 (drift 蒸留)**: symbolic drift の test NRMSE (対 teacher) ≤ 0.10。
- **H2 (diffusion 蒸留)**: symbolic diffusion (Cholesky 4 成分の joint) の test NRMSE (対 teacher) ≤ 0.20、かつ
  symbolic の implied $\Sigma$ の test NRMSE (対 teacher の $\Sigma$) ≤ 0.20。
- **H3 (真値回復)**: symbolic drift の test NRMSE (対真値) ≤ 0.15、かつ
  teacher (v4) の $\Sigma$ の test NRMSE (対真値) ≤ 0.20。
  teacher は汎用 MLP のため係数条項はない (Day 3 の H3 とは異なり、$(\kappa, \theta, \xi, \rho)$ 対応物は存在しない)。
- **H4 (シミュレーション)**: symbolic SDE の失敗率 ≤ 0.05 かつ neural との差 ≤ 0.02、
  paired RMSE (対 neural) の中央値 ≤ 0.15 (Day 1–3 の 0.5 より厳しい。終端スケールが x で ~0.2、v で ~0.03 のため)。
- **H5 (推定器バイアス)**: (a) v4 の drift NRMSE (対真値、test) が v1 より小さい、かつ
  (b) low-v レジームで v4 の drift NRMSE ≤ 0.9 × v1 の low-v drift NRMSE。
  diffusion 比較には Sigma NRMSE の差 < 0.01 の同点域を事前に設け、同点域内なら diffusion は判定を決めない (Day 3 の反省)。
  NLL の順序は判定条項に含めない (Day 2 の反省)。
- **H6 (論文軸の代理: 価格)**: 共通乱数によるモンテカルロ価格 (8192 path、$dt = 0.01$) で、
  |call 価格 (symbolic − teacher)| ≤ 0.01 かつ |lookback 価格 (symbolic − teacher)| ≤ 0.02。
  call は $K = 1$、$T = 1$、割引 $e^{-rT}$、lookback は論文 §5 の $\Psi = \max_t S_t - S_T$ (割引つき)。

## 4. 方法 (main と同じ流儀)

- teacher への直接 query で function distillation。query 点は各 split の観測状態
  (trajectory-disjoint)。変換 (mean/scale) と RMS 正規化は train query のみから。
- 候補: drift は次数 {1, 2, 3}、diffusion は次数 {0, 1, 2, 3}、STLSQ しきい値 {0.02, 0.05, 0.1, 0.2, 0.35}。
  field ごとに validation NRMSE で選択し、test は確定後に 1 回だけ評価。
- oracle 診断: 同じ SINDy 手順を真値 field に fit (候補ではなく天井の診断)。
- 共線性診断: fit の前に train query 状態で corr(x, v)、VIF、次数 2 特徴量間の最大相関を計算。
- off-manifold 診断 (Day 2 の反省により標準手順): x ∈ {−0.6, −0.3, 0, 0.3, 0.6}、v ∈ {0.005, 0.02, 0.04, 0.08, 0.16} の 25 点格子。
  v = 0.16 列は tail 列として分離して報告する。
- シミュレーション検証: test 系列の $t = 0$ 状態 768 点から、neural / symbolic / 真値の
  3 モデルを同一 Brownian 増分 (paired) で Euler–Maruyama (dt = 0.01、$t \in [0, 1]$、全行列 diffusion)。
  失敗は non-finite / $\|x\| > 20$ で分類し、質量を保持する。v < 0 の経験は失敗とは別に「ドメイン逸脱」として数える。
- 価格計算: 初期点 (x, v) = (0, 0.04) から 8192 path、同一増分で 3 モデルをシミュレーションし、
  call / lookback の paired 差と paired 標準誤差を報告する。

## 5. 結果

### 5.1 推定器の比較 — 離散化バイアス診断 (Day 1 の反省)

観測間隔は $\Delta t = 0.05$。test 遷移数は 15,360 (low 3,337 / mid 9,314 / high 2,709)。

| 推定器 | validation NLL | drift NRMSE vs 真値 (test) | $\Sigma$ NRMSE vs 真値 (test) |
|---|---|---|---|
| v1 `em` | −5.17615 | 0.2209 | 0.1309 |
| v4 `substep` (teacher) | −5.18374 | 0.2405 | **0.1129** |

レジーム別 (test 遷移開始点の v):

| 推定器 | low (v < 0.02) drift / $\Sigma$ | mid drift / $\Sigma$ | high (v > 0.06) drift / $\Sigma$ |
|---|---|---|---|
| v1 em | 0.1231 / 0.3001 | 0.3518 / 0.0740 | 0.2247 / 0.1526 |
| v4 substep | 0.1624 / 0.2763 | 0.4837 / 0.0646 | 0.2065 / 0.1304 |

読み取り:

- **$\Sigma$ については substep が全レジームで改善する** (全体 0.1309 → 0.1129、差 0.0179 > 同点域 0.01)。
  これが本日の診断が狙った経路である: 区間内で $\sqrt{v}$ を再評価することで、凍結バイアスが減る。
  low-v でも $\Sigma$ は改善する (0.3001 → 0.2763) が、low-v の $\Sigma$ 誤差自体は依然として最大である
  ($\sqrt{v}$ の曲率が最も大きく、相対的な v の動きも大きいため)。
- **drift については substep が悪化する** (0.2209 → 0.2405、low-v でも 0.1231 → 0.1624)。
  したがって H5 は否である。ただしこの drift 比較は、系統的な離散化バイアスというより
  **drift の弱同定**を映している。$\Delta t = 0.05$ では drift の寄与 ($f \Delta t \sim 0.001$–$0.003$) が
  diffusion の 1 ステップ標準偏差 ($\sqrt{v \Delta t} \sim 0.045$) の 20 分の 1 程度しかなく、
  drift の推定はノイズに支配される。2 つの MLP が異なる最適解 (異なる初期値・最適化経路) に落ちた差が、
  推定器の系統差を上回っている可能性が高い。
- **NLL と drift の順序が逆転した**。v4 の validation NLL は v1 より良い (−5.18374 < −5.17615) のに、
  drift NRMSE は v4 の方が悪い。Day 2 の反省どおり、NLL の順序は field の順序ではない。
  なお mid レジームの drift NRMSE が両推定器で最大 (0.35 / 0.48) なのは、
  mid レジーム ($v \approx \theta = 0.04$ 付近) では真値 drift $f_v = \kappa(\theta - v)$ がゼロ付近を通り、
  NRMSE の分母 (真値 field の RMS) が小さくなる分母効果を含む。レジーム別 NRMSE の絶対値比較にはこの注意が必要である。
- **drift の絶対水準自体が悪い**。oracle (真値への SINDy) の drift test NRMSE は次数 1 で 0.0073 であり、
  真値 drift はアフィンなので多項式でほぼ厳密に表せる。teacher の drift 誤差 0.24 は、
  したがって蒸留の失敗ではなく teacher 推定の失敗 (弱同定) である。Day 1・3 と同じ構造
  (symbolic の対真値誤差は teacher 側の誤差が支配的) が、drift 側で最も強く現れた日である。

### 5.2 蒸留の精度

| field | 選択 (次数, しきい値, 活性項) | validation NRMSE | test NRMSE 対 teacher | test NRMSE 対真値 |
|---|---|---|---|---|
| drift | 次数 3, 0.02, 20 項 | 0.07780 | **0.07992** | 0.2249 |
| diffusion (L 4 成分 joint) | 次数 3, 0.02, 19 項 | 0.02853 | **0.02834** | — |
| diffusion の implied $\Sigma$ | — | — | **0.04074** | 0.1103 |

diffusion の成分別 test RMS 誤差 (対 teacher): $L_{11}$ 0.00560 (RMS 0.2018)、$L_{12}$ 0 (恒等的に 0)、
$L_{21}$ 0.00153 (RMS 0.0414)、$L_{22}$ 0.00134 (RMS 0.0417)。
oracle (真値への fit) の天井は drift 0.0073 (次数 1)、diffusion 0.0208 (次数 3)。
symbolic の対 teacher diffusion 誤差 (0.0283) は oracle 天井に近く、**on-manifold では $\sqrt{v}$ の多項式近似は成功している**。
これは Day 3 の「diffusion は構成上自明」な成功とは異なり、非自明な関数近似の成功である。
ただし §5.6 のとおり、この成功は多様体上に限定される。

### 5.3 求めた symbolic SDE (TeX、元座標)

蒸留で得た SDE は次である ($W$ は 2 次元独立 Wiener 過程、$\hat f$ / $\hat L$ は
`outputs/daily_neural_sde/day04/symbolic_sde.tex` の生成 TeX と同一。
状態は $x = \log S$、$v$ = 分散):

$$
\mathrm{d}\begin{pmatrix} x_t \\ v_t \end{pmatrix}
= \hat f(x_t, v_t) \,\mathrm{d}t + \hat L(x_t, v_t) \,\mathrm{d}W_t
$$

```tex
\begin{aligned}
\hat f_x &= 0.03799 - 1.198\,v - 0.07401\,x + 0.5653\,v^{2} - 0.3986\,x v + 0.04464\,x^{2} + 19.28\,v^{3} + 11.01\,x v^{2} + 2.31\,x^{2} v + 0.2308\,x^{3} \\
\hat f_v &= 0.06919 - 1.366\,v + 0.06342\,x - 10.32\,v^{2} - 1.358\,x v - 0.1329\,x^{2} + 57.39\,v^{3} + 13.95\,x v^{2} + 1.011\,x^{2} v - 0.3183\,x^{3} \\
\end{aligned}
\begin{aligned}
\hat L_{11} &= 0.08947 + 2.63\,v - 0.1295\,x + 3.236\,x v + 0.2066\,x^{2} \\
\hat L_{12} &= 0 \\
\hat L_{21} &= - 0.0162 - 0.6388\,v + 0.0152\,x + 1.329\,v^{2} - 0.249\,x v - 0.02666\,x^{2} - 4.858\,v^{3} - 3.312\,x v^{2} - 0.725\,x^{2} v - 0.09543\,x^{3} \\
\hat L_{22} &= 0.02175 + 0.3286\,v - 0.03216\,x + 2.206\,v^{2} + 0.7157\,x v + 0.09142\,x^{2} + 2.263\,x v^{2} \\
\end{aligned}
```

読み取り: 真値 drift はアフィン ($f_x = 0.02 - 0.5v$、$f_v = 0.08 - 2v$) なのに、
symbolic drift は $v^3$ の係数が 19.28 / 57.39 という大きな高次項を持つ。
$v \approx 0.04$ では $v^3 \approx 6.4 \times 10^{-5}$ なので寄与は 0.001–0.004 程度に収まり、
on-manifold では線形項・2 次項と打ち消し合って正しい field を再現する。
これは**多様体上の打ち消し (manifold cancellation)** であり、物理的な意味を持つ係数ではない。
$v = 0.16$ では $v^3 = 0.0041$ となり、$57.39 \times 0.0041 \approx 0.23$ と発散的に効き、
off-manifold 崩壊 (§5.6) の直接の原因になる。diffusion 側も同様に、真値は $v$ のみに依存するのに
symbolic には $x$ 依存項が 13 項も現れる。これは teacher の MLP が $x$ を $v$ の代理変数として使った漏れを、
SINDy が忠実にコピーしたものである (§5.4)。

### 5.4 真値との比較と共線性 (H3 の内訳)

train query 状態での共線性診断: $\mathrm{corr}(x, v) = -0.6579$、VIF $= 1.76$、
次数 2 特徴量間の最大絶対相関 $= 0.924$。VIF は 10 以下だが、これは「2 変数間の VIF」という粗い指標であり、
次数 2 特徴量間の 0.924 という相関が、次数 3 までの多項式ライブラリ内での実質的な共線性の強さを示している。
負の相関の機構は論文の LSV 構造そのものである: レバレッジ $\rho = -0.7$ により $v$ が跳ねたときに $x$ が下がりやすく、
かつ drift の $-0.5v$ 項が高 $v$ 状態の $x$ を押し下げる。したがって観測状態は $(x, v)$ 平面の斜めの帯に集中し、
$x$ と $v$ は単独では分離できない。teacher の MLP はこの帯上で $x$ から $v$ を (部分的に) 推測できてしまうため、
diffusion が $x$ に漏れる。**共線性は本日、Day 2 のような $t$ との構造的共線性ではなく、レバレッジが生む同時分布の傾きである。**

H3 の判定: symbolic drift の対真値 NRMSE 0.2249 > 0.15 で失敗するが、
teacher の $\Sigma$ 対真値 NRMSE 0.1103 ≤ 0.20 は成る。
失敗の主因は §5.1 の teacher drift の弱同定であり、蒸留手順の失敗ではない
(symbolic の対 teacher 誤差は 0.0799 であり、teacher の誤差 0.2405 の大半はそのまま引き継がれる)。

### 5.5 シミュレーション検証 (paired、768 系列、$t \in [0, 1]$)

| モデル | 失敗率 | v < 0 逸脱率 | 終端平均 (x, v) (finisher) | 終端 std (x, v) (finisher) |
|---|---|---|---|---|
| neural (teacher v4) | 0.000 | 0.191 | (0.0098, 0.0384) | (0.1967, 0.0295) |
| symbolic | 0.000 | 0.188 | (0.0101, 0.0384) | (0.1962, 0.0294) |
| 真値 | 0.000 | 0.035 | (0.0148, 0.0389) | (0.1960, 0.0293) |

paired RMSE 中央値: symbolic 対 neural **0.00256**、768 系列すべて paired。
H4 の数値条項 (失敗率・paired RMSE) はすべて満たすが、**ドメイン妥当性では teacher / symbolic が真値と大きく異なる**:
v < 0 を経験する系列の割合が teacher 19.1%、symbolic 18.8% に対し真値は 3.5% である。
真値の 3.5% は Euler 離散化の人工物だが、teacher / symbolic の 19% はモデル自体の問題である
(low-v での $f_v$ / $L$ の誤差が、分散を負の領域へ押し込みやすい)。
H4 の凍結条項には v < 0 逸脱率が含まれていなかったため H4 は「成」となるが、
これは判定条項の欠落であり、価格計算 (§5.7) でも lookback の不確かさとして跳ね返る。
**失敗率 0 とドメイン妥当性は別の軸である** (Day 1 の「失敗質量の保持」の教訓の発展形)。

### 5.6 off-manifold 格子診断 (3 軸)

| 比較 | drift NRMSE | diffusion (L) NRMSE |
|---|---|---|
| symbolic vs teacher | 0.7878 | 0.1501 |
| symbolic vs 真値 | 0.6767 | 0.5138 |
| teacher vs 真値 | 0.3693 | 0.3880 |

v = 0.16 の tail 列のみでは、drift symbolic vs teacher 0.7504、diffusion symbolic vs teacher 0.1730、
diffusion symbolic vs 真値 0.6030 であり、全体と同程度に悪い。つまり崩壊は tail 列に限らず、格子全体に及ぶ。
Day 3 (格子上で安定) とは対照的に、本日は **teacher 自体が格子上で崩壊している** (teacher vs 真値で drift 0.3693、diffusion 0.3880)。
理由は 2 つある: (a) 斜めの多様体 (§5.4) のため、格子点の多くは学習状態の同時分布から外れており、
MLP の外挿は保証されないこと、(b) symbolic の高次項 (§5.3) が格子上で発散すること。
格子は周辺範囲ではデータ範囲の内側にあるため、これは「遠方外挿の失敗」ではなく
**「同時分布からの乖離に対する失敗」**として報告するのが正確である。
on-manifold の成功 (diffusion 0.0283) と off-manifold の崩壊 (0.5138) の落差が、本日の最大の対比である。

### 5.7 論文軸の代理: 価格 (H6、共通乱数、8192 path)

| モデル | call 価格 (K = 1) | lookback 価格 ($\max_t S_t - S_T$、割引つき) | 失敗率 |
|---|---|---|---|
| neural (teacher) | 0.08540 | 0.12668 | 0.000 |
| symbolic | 0.08534 | 0.14018 | 0.00012 (1 path) |
| 真値 | 0.08801 | 0.12345 | 0.000 |

paired 差 (symbolic − teacher): call **−0.000058 ± 0.000101** (paired 標準誤差)、
lookback **+0.01350 ± 0.01423**。凍結しきい値 (call ≤ 0.01、lookback ≤ 0.02) に対して点推定では両方とも成る。
ただし lookback の paired 標準誤差 (0.0142) は差自体と同程度であり、
点推定の合格は統計的に脆い: 差の 95% 区間はおよそ [−0.014, +0.041] で、しきい値 0.02 を含む。
call については symbolic が teacher をほぼ完全に再現する (差は paired SE の 0.6 倍) が、
teacher 自体が真値から −0.00261 ずれており、symbolic はそのバイアスも引き継ぐ。
lookback が call より脆いのは、lookback が経路全体の最大値に依存し、§5.5 の v < 0 逸脱を含む
経路レベルの歪みに敏感なためと考えられる。H6 は「成」と判定するが、lookback 条項は
「点推定では成るが、paired SE では確定しない」という限定つきである。
なお本日の H6 は論文の較正主張の検証ではない (§2.2)。論文の MSE $10^{-9}$ や価格区間とは無関係の数値である。

### 5.8 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1 drift 蒸留 | **成** | test 0.07992 ≤ 0.10 (次数 3、20 項)。ただし teacher 自体が真値から 0.24 ずれているため、蒸留の成功は teacher 近似の成功である |
| H2 diffusion 蒸留 | **成** | L joint test 0.02834 ≤ 0.20、かつ implied $\Sigma$ 対 teacher 0.04074 ≤ 0.20。on-manifold では $\sqrt{v}$ の近似として非自明な成功 |
| H3 真値回復 | **否** | symbolic drift の対真値 0.2249 > 0.15。teacher の $\Sigma$ 条項 (0.1103 ≤ 0.20) は成るが、drift 条項が失敗する。主因は teacher drift の弱同定 (§5.1) |
| H4 シミュレーション | **成** (ドメイン妥当性の限定つき) | 失敗率 0 (差 0)、paired RMSE 0.00256 ≤ 0.15。ただし v < 0 逸脱率は symbolic 18.8% vs 真値 3.5% であり、失敗率の合格はドメイン妥当性を保証しない (§5.5) |
| H5 推定器バイアス | **否** | drift 条項が失敗: v4 0.2405 > v1 0.2209。low-v 条項も失敗: v4 0.1624 > 0.9 × v1 0.1231。$\Sigma$ 経路では v4 が改善する (0.1309 → 0.1129、差 0.0179 > 同点域 0.01) が、H5 の判定は drift 条項に置いたため否である。NLL では v4 が良いのに drift では悪い、という逆転も併せて記録する |
| H6 論文軸の代理 (価格) | **成** (lookback の統計的限定つき) | call 差 −0.000058 ≤ 0.01、lookback 差 +0.01350 ≤ 0.02 (点推定)。ただし lookback の paired SE は 0.01423 であり、合格は統計的に確定しない (§5.7) |

## 6. 考察

1. **状態依存 diffusion の蒸留は on-manifold では成功し、drift の蒸留は弱同定に阻まれる**。
   Day 3 と軸が反転した: Day 3 では drift (sigmoid) が主題で diffusion は定数だったが、
   本日は drift がアフィン (oracle 0.0073) にもかかわらず teacher の drift 推定が 0.24 と悪く、
   diffusion ($\sqrt{v}$、相関つき) の蒸留が 0.028 と良い。SINDy の難しさは field の関数形の複雑さだけでは決まらず、
   **その field がデータからどれだけ同定されるか**で決まる。短い観測間隔では drift は diffusion ノイズに埋もれる。
2. **レバレッジが生む共線性は、金融系 neural SDE に固有の罠である**。
   $\rho = -0.7$ の相関と $-0.5v$ の drift が、観測状態を斜めの帯に潰す。
   その帯上では $x$ が $v$ の代理変数として機能するため、MLP teacher は diffusion に $x$ 依存性を漏らし、
   SINDy はそれを忠実にコピーする (diffusion の $x$ 依存項 13 項)。
   on-manifold の指標だけを見るとこの漏れは見えない。off-manifold 格子がそれを暴く、という Day 2 の教訓の金融版である。
3. **失敗率とドメイン妥当性は別の軸**である。
   teacher / symbolic は 1 本も「失敗」しないが、約 19% の系列で分散が負になる。
   負の分散は SDE として無意味な領域であり、特に lookback のような経路依存ペイオフの価格に跳ね返る。
   シミュレーション検証の判定条項にドメイン逸脱率 (本日なら v < 0 率) を含めるべきだった。
4. **NLL の逆転は再現性のあるパターンになった**。
   Day 2・Day 3 に続き、本日も validation NLL の順序と field 誤差の順序が一致しなかった
   (v4 の NLL が良いのに drift が悪い)。NLL は尤度の総合指標であり、drift の field 精度を代理しない。
   推定器比較の主軸は field NRMSE とレジーム別誤差に置くべきである、という Day 2 の反省が 3 回目の裏づけを得た。
5. **蒸留の成功と論文の主張は別の軸**である (Day 2・3 と同じ教訓の再確認)。
   本日の field 蒸留の成功 (diffusion 0.028) は Gierjatowicz et al. の主張 (市場較正 MSE、lookback 価格区間) の検証ではない。
   逆に H3・H5 の否も、論文の主張への反証ではない。論文が主張していない軸での「失敗」を数えない、という姿勢を維持する。

## 7. 反省 (Day 5 以降に適用)

- シミュレーション検証の判定条項には、失敗率とは別にドメイン妥当性 (本日なら v < 0 逸脱率、一般には状態の許容領域からの逸脱率) を
  事前に含めること。失敗率 0 でもドメイン逸脱率が高いモデルは「成」と読まないこと。
- drift の弱同定が疑われる系 (短い観測間隔で diffusion が支配的) では、drift の NRMSE 比較だけで推定器の優劣を判定しないこと。
  drift の絶対誤差 (RMS 誤差) と $\Sigma$ の誤差を併記し、drift 条項には分母効果 (真値 field がゼロ付近を通るレジーム) の注記を添えること。
- レジーム別 NRMSE は、レジーム内の真値 field の RMS が小さい場合に膨らむため、レジーム別の分母 (真値 RMS) も併記すること。
- 価格のような downstream 指標では、点推定のしきい値合格だけでなく paired 標準誤差も判定文に含めること。
  paired SE がしきい値と同程度の場合、その条項は「統計的に確定しない」と明記すること。
- 多項式の高次係数が on-manifold の打ち消しで大きくなっている場合 (本日の $v^3$ 係数 19 / 57)、
  その係数を物理的意味として読まず、off-manifold 格子での発散リスクとして明記すること。

## 8. 限界

- 本日の teacher は論文の市場較正モデルではない (§2.2 の逸脱)。
  市場データ、ヘッジネット、頑健価格区間、per-maturity ネット、tamed Euler は本日は扱っていない。
  論文の較正 MSE や価格区間についての主張は、本日の結果では何も言えない。
- パラメータ $(r, \kappa, \theta, \xi, \rho)$ は宣言したベンチマーク値であり、論文の推定値ではない。
- substep 推定器は drift-Jacobian 項を省略した限定版のモーメント伝播である (§2.2-5)。
  厳密な substep モーメント伝播との差は未測定である。
- 生成の full-truncation と field 定義 (drift は $v$ そのもの、diffusion は $\sqrt{\max(v, 0)}$) の間には、
  v < 0 のステップでの小さな不一致がある。観測点での v < 0 割合は $1.2 \times 10^{-5}$ と小さいが、厳密には同一モデルではない。
- シミュレーション horizon は $t \in [0, 1]$ のみ。長期の定常分布への収束は未検証。
- off-manifold 格子は周辺範囲ではデータ範囲の内側にあり、強い外挿テストではない (§5.6)。
  その診断力は同時分布からの乖離にある。
- 既存 repo tests (`tests/test_distill.py` 等) は本環境 (environment.yml の完全環境ではない) では未実行。
  本日追加した `tests/test_daily_neural_sde_day04.py` の 4 tests と Day 1–3 の tests は全て通過 (計 17 tests)。
