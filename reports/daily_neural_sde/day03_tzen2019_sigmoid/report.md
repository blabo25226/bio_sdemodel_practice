# Day 3 — Tzen & Raginsky (2019) Neural SDEs: sigmoid-drift synthetic system

- Date: 2026-10-28 (Day 3)
- Paper: Belinda Tzen, Maxim Raginsky,
  *Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit*,
  arXiv:1905.09883 (v2, 28 Oct 2019).
- 成果物: config `configs/daily_neural_sde_day03.json` / コード `src/daily_neural_sde/day03_tzen_sigmoid.py`
  (共通部品 `src/daily_neural_sde/common.py` を再利用) / 数値 `outputs/daily_neural_sde/day03/results.json`、
  teacher 重み `teacher_weights.json`、symbolic モデル `symbolic_drift.json` / `symbolic_diffusion.json`、
  TeX `symbolic_sde.tex` / tests `tests/test_daily_neural_sde_day03.py` (4 tests 追加、Day 1・2 分と合わせて 13 tests 全て通過)。
- 実行: `python -m src.daily_neural_sde.day03_tzen_sigmoid` (seed 0、PyTorch 2.14.1+cpu、約 3 分)。

## 0. 前日までの反省の適用 (本日の分析に組み込んだもの)

Day 1・Day 2 の反省をすべて適用した:

1. **推定器の離散化バイアス診断 (Day 1)**: 4 種の推定器 (v1 EM / v2 局所線形化 / v3 行列指数再和約 / v4 内部 substep モーメント伝播)
   で $\hat g$ と drift の真値との NRMSE を比較し、さらに遷移開始点のレジーム
   (near-linear: $\max|Ax| < 0.5$ / intermediate / saturated: $\max|Ax| > 1.5$) ごとに
   drift NRMSE を分けて報告した。sigmoid drift における Day 1 の「線形成分と非線形成分での比較」の対応物である。
   蒸留する teacher は事前に v4 と宣言した (diffusion の絶対水準が主題のため)。
2. **off-manifold 格子診断の標準化 (Day 2)**: $x_1, x_2 \in \{-3, -1.5, 0, 1.5, 3\}$ の 25 点格子で
   symbolic / teacher / 真値を比較し、field 水準・係数水準・off-manifold 水準の 3 軸で分けて報告した。
3. **NLL の順序を過大評価しない (Day 2)**: 4 推定器の validation NLL は $-0.14758 \sim -0.14732$ と
   $3 \times 10^{-4}$ の幅に収まるため、推定器の drift 順序は field NRMSE で判断し、NLL の順序は系統的な優劣と読まない。
4. **共線性ゲート (Day 1・2)**: fit の前に train query 状態で $\mathrm{corr}(x_1, x_2)$、VIF、
   次数 2 特徴量間の最大相関を計算した。VIF $= 1.00$ と低く、Day 2 のような構造的共線性は本日は存在しない。
   ただし多項式係数を $A$ と読み替えることはしない (多項式は sigmoid ではないため、係数に $A$ 対応物が存在しない)。

## 1. 論文要約

### 1.1 問題と提案

深層潜在ガウスモデル (DLGM, Rezende et al. 2014) では、潜在変数が層ごとに非線形写像と
小さなガウス摂動で生成される時間非斉次マルコフ連鎖である。
層数を無限大、ステップ幅とノイズ分散をゼロに持っていく拡散極限では、潜在対象は
drift / diffusion がニューラルネットで実装された Itô 拡散過程、すなわち neural SDE になる。
本論文はこの neural SDE に対する変分推論の枠組みを与える:

- 潜在ランダムネスはすべて標準 Wiener 過程のサンプルで生成できる (Wiener 空間が潜在空間)。
- 周辺対数尤度への変分上限は、path 空間上の Gibbs 変分原理から導かれる。
  Girsanov の定理により、近似事後分布は標準 Wiener 過程への平均シフト (mean-shift) として得られる:
  観測依存のニューラルネット drift を標準 Wiener 過程に加えたものが、mean-field 近似事後分布の neural SDE 対応物である。
- 勾配計算は Wiener 空間での確率微分 (stochastic flows) による pathwise 微分で行う。
  SDE の adjoint 法は時間適合性 (time-adaptedness) の問題から black-box ソルバには使えないため、
  discretize-then-differentiate (Euler 離散化に対する確率的 backpropagation と等価) か
  differentiate-then-discretize (forward-mode の pathwise 微分) のいずれかが必要になる。

### 1.2 実験 (§6) と主張の範囲

Julia + DiffEqFlux (CPU) による合成データ実験のみである。真の過程は

$$
\mathrm{d}X_t = \mathrm{sigmoid}(A X_t)\,\mathrm{d}t + \mathrm{d}W_t, \qquad t \in [0, 1]
$$

で、$A \in \mathbb{R}^{d \times d}$ の要素は i.i.d. $\mathcal{N}(0, 1)$ からランダムに選ばれる。
観測は $n$ 個の独立コピーについて $X_1$ に独立なガウス摂動 (共分散 $I_d$) を加えたもの
($p(y|x)$ は平均 $x$、共分散 $I_d$ のガウス分布)。
変分近似は定数 drift シフト $\beta$ を加えたもの
$\mathrm{d}X^{A,\beta}_t = (\mathrm{sigmoid}(AX^{A,\beta}_t) + \beta)\,\mathrm{d}t + \mathrm{d}W_t$ で、
実験は $d = 10$ ($A$ は $10 \times 10$、$\beta \in \mathbb{R}^{10}$)、$n = 1000$ で
自由エネルギー (free energy) の離散化メッシュ依存性とサンプルサイズ依存性を調べる。
報告される主張は「$h = 1/32$ より細かいメッシュの改善は漸進的 (incremental) である」
「$n \approx d$ では頑健だが $n \to \sqrt{d}$ の低データ領域では悪化する」という定性的なものであり、
**$A$ の回復精度や drift / diffusion の関数としての回復精度は論文の主張ではない**点に注意する。
本日の変分フィットの $A$ に関する負の結果 (§5.4) は、したがって論文の主張への反証ではない。

## 2. データと Neural SDE の入手・再構成

### 2.1 データ

論文は生成データを公開していないため、§6 の設定どおり再生成した (次元とサンプル数の切り取りは §2.2)。
真値行列は `np.random.default_rng(0).standard_normal((2, 2))` で

$$
A = \begin{pmatrix} 0.1257 & -0.1321 \\ 0.6404 & 0.1049 \end{pmatrix}
$$

である。非線形 drift には厳密遷移が存在しないため、データ生成は $dt = 0.001$ の
Euler–Maruyama (1000 ステップ) で行い、$0.05$ ごとに観測した (4096 系列 × 21 点、$t \in [0, 1]$)。
状態統計: 平均 $(0.251, 0.264)$、標準偏差 $(1.256, 1.247)$、範囲 $[-4.45, 5.72]$。
終端 $X_1$ の平均 / 標準偏差は $(0.481, 0.539)$ / $(1.458, 1.433)$。
分割は trajectory-disjoint に train 2560 / validation 768 / test 768 (seed 0)。

### 2.2 逸脱の明示

1. **次元とサンプル数**: 論文は $d = 10$、$n = 1000$ だが、本日は CPU 切り取りとして $d = 2$、$n = 4096$ とした。
   次元を下げた分だけ系列数を増やし、遷移数 (train 51,200) を確保している。
2. **初期分布**: 論文 §6 は初期分布を明示しない。§2 の DLGM 定義 ($X_0 = Z_0 \sim \mathcal{N}(0, I)$) に従い
   $X_0 \sim \mathcal{N}(0, I_2)$ と宣言して採用した。
3. **活性化関数の読み**: 論文の式 (21) は $\mathrm{sigmoid}(AX_t)$ と書く。本日はこれを通常の読みである
   coordinatewise logistic sigmoid として採用した。ただし ar5iv 版の本文は
   「sigmoid softplus activation function」と描出されており、logistic か softplus かの読みに曖昧さが残る。
   この曖昧さは解消せず、読みのリスクとしてここに記録する。softplus であった場合、本日の真値 drift は
   論文のものと異なる (定数オフセットとスケールの違い) ことになる。
4. **teacher のモデル族**: 論文のモデル族そのもの、すなわち drift $= \mathrm{sigmoid}(Ax)$ (1 層の線形層 + sigmoid)、
   diffusion $= I$ を teacher の族とした。Day 1・2 の汎用 MLP teacher は意図的に使っていない。
   これにより teacher の係数 $A$ が真値 $A$ と直接比較可能になる。
   尤度 teacher (v1–v4) はシリーズ流儀の診断用比較器であり、論文の推定法ではない。
5. **変分推論の切り取り**: 論文の変分近似を、観測ごとに異なる $\beta(y)$ ではなく
   **全観測で共有する単一の定数シフト $\beta \in \mathbb{R}^2$** として実装した。
   これは論文 §6 の実験記述 ($\beta \in \mathbb{R}^{10}$ が単一のベクトルとして扱われる) に沿うが、
   観測依存の amortized エンコーダは本日の範囲外である。
   また論文の forward-mode pathwise 微分ではなく、discretize-then-differentiate (Euler シミュレーション + Adam) を使った。
   論文 §5.2 自身が、Euler 離散化に対する discretize-then-differentiate は確率的 backpropagation と等価であることを示している。
   変分フィットの KL 項は、単位 diffusion に対する定数シフトの Girsanov 形 $0.5 \|\beta\|^2$ (区間 $[0,1]$ 上の積分) である。
6. **メッシュ / サンプルサイズ軸**: 論文の 2 軸のうちメッシュ軸のみを再テストした (H6)。
   サンプルサイズ軸 (異なる $n$ での再フィット) は本日は実施していない。

## 3. 仮説 (実行前に config へ凍結)

- **H1 (drift 蒸留)**: symbolic drift の test NRMSE (対 teacher) ≤ 0.10。
- **H2 (diffusion 蒸留)**: symbolic diffusion の test NRMSE (対 teacher) ≤ 0.20、かつ
  非定数項の寄与 (test query での RMS 比) ≤ 0.25。
- **H3 (真値回復)**: symbolic drift の test NRMSE (対真値) ≤ 0.15、かつ
  尤度 teacher (v4) の $\hat A$ の相対 Frobenius 誤差 ≤ 0.35。
  symbolic 多項式の係数は $A$ と読まない (多項式 ≠ sigmoid)。
- **H4 (シミュレーション)**: symbolic SDE の失敗率 ≤ 0.05 かつ neural との差 ≤ 0.02、
  paired RMSE (対 neural) の中央値 ≤ 0.5。
- **H5 (推定器バイアス)**: v4 の $|\hat g - 1|$ が v1 より小さく、かつ
  v4 の drift NRMSE (対真値、test) が v1 より小さい。
  診断としてレジーム別 (near-linear / intermediate / saturated) の drift NRMSE も報告する。
- **H6 (変分メッシュ)**: 適合済み共有-$\beta$ 変分モデルを validation 終端観測で再シミュレーションしたとき、
  $|F(h = 1/64) - F(h = 1/32)| \le 0.05$ (定数 $(d/2)\log 2\pi$ を除く自由エネルギー、観測あたり)。

## 4. 方法 (main と同じ流儀)

- teacher への直接 query で function distillation。query 点は各 split の観測状態
  (trajectory-disjoint)。変換 (mean/scale) と RMS 正規化は train query のみから。
- 候補: drift は次数 {1, 2, 3}、diffusion は次数 {0, 1, 2}、STLSQ しきい値 {0.02, 0.05, 0.1, 0.2, 0.35}。
  field ごとに validation NRMSE で選択し、test は確定後に 1 回だけ評価。
- oracle 診断: 同じ SINDy 手順を真値 field に fit (候補ではなく天井の診断)。
- 共線性診断: fit の前に train query 状態で $\mathrm{corr}(x_1, x_2)$、VIF、次数 2 特徴量間の最大相関を計算。
- off-manifold 診断 (Day 2 の反省により標準手順): $x_1, x_2 \in \{-3, -1.5, 0, 1.5, 3\}$ の 25 点格子。
  ただし本日の格子はデータ範囲 $[-4.45, 5.72]$ と大きく重なるため、強い外挿テストではなく
  格子上の一致度診断である点は §7 で明記する。
- シミュレーション検証: test 系列の $t = 0$ 状態 768 点から、neural / symbolic / 真値の
  3 モデルを同一 Brownian 増分 (paired) で Euler–Maruyama (dt = 0.01、$t \in [0, 1]$)。
  失敗は non-finite / $\|x\| > 20$ で分類し、質量を保持する。
- 変分フィット: train 終端観測 $Y = X_1 + \mathcal{N}(0, I_2)$ (ノイズ seed 5) に対し、
  バッチ 1024 path、Adam lr 0.01、300 epoch。チェックポイントは validation 自由エネルギー (固定 seed シミュレーション) の最小値。
  メッシュ研究は適合後のモデルを $h \in \{1/8, 1/16, 1/32, 1/64\}$ で validation 観測・4096 path・seed 7 で再シミュレーションし、
  モンテカルロ標準誤差つきで報告する。

## 5. 結果

### 5.1 推定器の比較 — 離散化バイアス診断 (Day 1 の反省)

観測間隔は $\Delta t = 0.05$。test 遷移数は 15,360 (near-linear 7,259 / intermediate 7,115 / saturated 986)。

| 推定器 | 内容 | validation NLL | $\hat g$ | $\|\hat g - 1\|$ | drift NRMSE vs 真値 (test) |
|---|---|---|---|---|---|
| v1 `em` | 素朴な Euler–Maruyama 尤度 | −0.14752 | 0.9931 | 0.0069 | 0.0810 |
| v2 `local_linearization` | 平均に $\frac{1}{2}\Delta t^2 (J f)$、分散に $J$ の対称化補正 | −0.14755 | 0.9929 | 0.0071 | 0.1254 |
| v3 `local_linearization_exp` | 局所線形化を行列指数で再和約 (8 点 Gauss–Legendre 求積で共分散積分を近似) | −0.14732 | 0.9923 | 0.0077 | 0.1111 |
| v4 `substep_moments` (teacher) | 内部 8 substep のモーメント伝播 (平均は $f$ で、共分散は $(JV + VJ^\top + g^2 I)h$ で更新) | −0.14758 | 0.9928 | 0.0072 | **0.0513** |

読み取り:

- **diffusion の絶対水準は全推定器が 1% 以内で回復する**。Day 1 の Lorenz 系 (定数 diffusion が非線形成分で回復しなかった)
  とは異なり、観測間隔 $0.05$ と有界な sigmoid drift の組み合わせでは、EM のバイアスは小さい。
  $|\hat g - 1|$ は v1 $0.0069$ < v2 $0.0071$ < v4 $0.0072$ < v3 $0.0077$ であり、
  v1 と v4 の差は $0.0003$ で実質的に同点である。
  これが H5 の $g$ 条項を失敗させる (後述) が、$0.0003$ の差に系統的な意味はない。
- **drift の順序は NLL の順序と一致しない**。validation NLL は 4 推定器で $3 \times 10^{-4}$ の幅に収まるのに、
  drift NRMSE は 0.0513 (v4) から 0.1254 (v2) まで 2.4 倍違う。Day 2 の反省どおり、NLL の順序は読まない。
- **バイアスは saturated レジームに集中する** (レジーム別 drift NRMSE、test 遷移開始点):

| 推定器 | near-linear | intermediate | saturated |
|---|---|---|---|
| v1 em | 0.0471 | 0.0950 | 0.1150 |
| v2 local_linearization | 0.1210 | 0.1280 | 0.1305 |
| v3 local_linearization_exp | 0.1071 | 0.1148 | 0.1087 |
| v4 substep_moments | 0.0479 | 0.0543 | 0.0496 |

  EM の誤差は near-linear では v4 と同等 (0.047) だが、saturated では 2.3 倍 (0.115) に悪化する。
  sigmoid が飽和する領域では区間内の Jacobian 変化が大きく、1-step 近似が破綻する。
  substep 伝播は全レジームで 0.048–0.054 と平坦であり、バイアスのレジーム集中を解消する。
  これが Day 1 の「非線形成分だけが膨らむ」パターンの sigmoid 版である。
  なお v2/v3 が v1 より drift で悪いのは、Jacobian をステップ開始点で固定する局所線形化が
  saturated 領域でむしろ悪化方向に働くためと考えられる (補正が常に改善するとは限らない)。

### 5.2 蒸留の精度

| field | 選択 (次数, しきい値, 活性項) | validation NRMSE | test NRMSE 対 teacher | test NRMSE 対真値 |
|---|---|---|---|---|
| drift | 次数 3, 0.02, 6 項 | 0.00646 | **0.00635** | 0.0537 |
| diffusion | 次数 0, 0.02, 2 項 (両座標とも定数) | $1.4 \times 10^{-12}$ | **$1.4 \times 10^{-12}$** | — |

oracle (真値への fit) の天井は drift test NRMSE $0.01245$ (次数 3)、diffusion は次数 0 で厳密に 0。
symbolic の対 teacher 誤差 (0.00635) は oracle 天井 (0.01245) より小さい:
teacher 自体が真値からずれている方向に symbolic が追従しているためであり、
対真値の誤差 (0.0537) は teacher 側の誤差 (drift NRMSE 0.0513) がほぼ支配的である (Day 1・2 と同じ構造)。
diffusion の蒸留が厳密なのは、teacher の diffusion が族の定義により定数であるためであり、
発見ではなく構成上の自明さである (H2 の判定はその旨を併記する)。

### 5.3 求めた symbolic SDE (TeX、元座標)

蒸留で得た SDE は次である ($W$ は 2 次元 Wiener 過程、$\hat f$ / $\hat g$ は
`outputs/daily_neural_sde/day03/symbolic_sde.tex` の生成 TeX と同一):

$$
\mathrm{d}X_t = \hat f(X_t) \,\mathrm{d}t + \hat g(X_t) \,\mathrm{d}W_t
$$

```tex
\begin{aligned}
\hat f_1 &= 0.5 - 0.01667\,x2 + 0.03179\,x1 \\
\hat f_2 &= 0.4974 + 0.1387\,x1 + 0.001871\,x1^{2} - 0.002519\,x1^{3} \\
\end{aligned}
\begin{aligned}
\hat g_1 &= 0.9928 \\
\hat g_2 &= 0.9928 \\
\end{aligned}
```

読み取り: これは teacher の sigmoid の Taylor 展開と整合する。
$\sigma(u) = 0.5 + u/4 - u^3/48 + \cdots$ より、teacher $\hat A$ に対する線形係数の期待値は
$0.25\hat A$ であり、$\hat f_1$ の $x_1$ 係数 $0.03179$ は $0.25 \times 0.1282 = 0.03205$、
$x_2$ 係数 $-0.01667$ は $0.25 \times (-0.0672) = -0.01680$、
$\hat f_2$ の $x_1$ 係数 $0.1387$ は $0.25 \times 0.5701 = 0.1425$ と近い。
$\hat A$ の $(2,2)$ 成分 $0.0045$ に対応する $x_2$ 係数 ($0.0011$) は STLSQ しきい値で落とされ、
$\hat f_2$ には $x_2$ 項が現れない。これが 6 項の内訳である。
3 次項が $x_1$ のみに現れるのは、$A$ の第 2 行が $x_1$ 支配的であるためである。

### 5.4 真値との係数比較と共線性 (H3 の内訳)

train query 状態での共線性診断: $\mathrm{corr}(x_1, x_2) = 0.0484$、VIF $= 1.00$、
次数 2 特徴量間の最大絶対相関 $= 0.367$。Day 2 の VIF $13.11$ とは対照的に、
時間非依存の自律系では状態成分間の構造的共線性は生じない。

尤度 teacher (v4) の $\hat A$ と真値 $A$ の比較:

| 成分 | 真値 | $\hat A$ (v4) | 絶対誤差 |
|---|---|---|---|
| $A_{11}$ | 0.1257 | 0.1282 | 0.0025 |
| $A_{12}$ | −0.1321 | −0.0672 | 0.0649 |
| $A_{21}$ | 0.6404 | 0.5701 | 0.0703 |
| $A_{22}$ | 0.1049 | 0.0045 | 0.1004 |

相対 Frobenius 誤差 $= 0.206 \le 0.35$。絶対値の小さい成分 ($A_{12}$、$A_{22}$) の相対誤差は大きいが、
Frobenius 意味では支配的な $A_{21}$ が 11% の精度で回復されており、field 水準の誤差は 0.0513 に収まる。
field 水準と係数水準は別々の軸である (Day 1 の教訓どおり)。

### 5.5 変分フィットとメッシュ研究 (論文 §6 の軸、負の結果を含む)

共有-$\beta$ 変分フィットの結果: $\hat\beta = (0.0286, 0.0523)$、$\|\hat\beta\| = 0.0596$ と小さく、
事後 drift は事前 drift からほとんどシフトしていない。
一方で適合された $\hat A_{\mathrm{var}}$ は

$$
\hat A_{\mathrm{var}} = \begin{pmatrix} -1.605 & 0.082 \\ 0.178 & -3.149 \end{pmatrix}
$$

と真値から大きくずれ、相対 Frobenius 誤差は $5.52$ である。
自由エネルギー (定数除く、観測あたり) は train $4.333$ / validation $4.707$ / test $4.317$
(定数 $(d/2)\log 2\pi = 1.838$ を加えるとそれぞれ $6.171$ / $6.545$ / $6.155$)。

読み取り: **終端観測のみからは $A$ は識別できない**。
$X_1$ の周辺分布は、区間 $[0, 1]$ 上の拡散 (分散 $1$) と有界な drift ($\|\mathrm{sigmoid}\| \le 1$ 成分ごと) の
組み合わせでは、大きく異なる $A$ に対しても似た分布になる。$\beta$ が小さいことは、
変分モデルが事前分布のまま終端分布をほぼ説明できていることを意味し、$A$ の方向への勾配信号が弱い。
これは論文の主張への反証ではない。論文は $A$ の回復を主張しておらず、自由エネルギーの減少と
メッシュ・サンプルサイズ依存性のみを報告しているからである。
むしろ本日の結果は、論文の実験設定 (終端観測のみ、共有 $\beta$) では $A$ の同定はそもそも設計されていないことを示す。
中間時刻の観測や軌道全体の尤度があれば話は変わるが、それは論文 §6 の設定外である。

メッシュ研究 (適合済みモデル、validation 終端観測、4096 path、平均 ± モンテカルロ標準誤差):

| $h$ | 自由エネルギー (定数除く) |
|---|---|
| 1/8 | $4.494 \pm 0.071$ |
| 1/16 | $4.577 \pm 0.072$ |
| 1/32 | $4.529 \pm 0.070$ |
| 1/64 | $4.502 \pm 0.071$ |

$|F(1/64) - F(1/32)| = 0.0277 \le 0.05$ であり H6 は成り立つが、
メッシュ間の差はモンテカルロ標準誤差 ($0.07$) より小さく、曲線は単調でもない。
したがって「$1/32$ より細かいメッシュの改善は漸進的」という論文の定性的主張とは整合するものの、
本日のモンテカルロ精度では改善量そのものを精密に測定できてはいない。
共通乱数 (Brownian bridge によるメッシュ間結合) を使えば分解能は上がるが、本日は未実施である。

### 5.6 off-manifold 格子診断 (3 軸)

| 比較 | drift NRMSE |
|---|---|
| symbolic vs teacher | 0.0126 (on-manifold では 0.00635) |
| symbolic vs 真値 | 0.0739 (on-manifold では 0.0537) |
| teacher vs 真値 | 0.0714 (on-manifold では 0.0513) |

Day 2 とは異なり、格子上で symbolic・teacher ともに崩壊しない。
理由は 2 つある: (a) sigmoid drift は有界で、次数 3 多項式近似が $[-3, 3]^2$ の範囲では安定であること、
(b) 状態成分間の共線性がなく (VIF $1.00$)、多様体が格子方向に潰れていないこと。
ただし格子はデータ範囲と大きく重なるため、これは強い外挿テストではない。
$|x| \sim 6$ 以上の遠方では多項式の 3 次項が発散し、symbolic は必ず崩壊する (未検証の既知の限界)。

### 5.7 シミュレーション検証 (paired、768 系列、$t \in [0, 1]$)

| モデル | 失敗率 | 終端平均 (finisher) | 終端 std (finisher) |
|---|---|---|---|
| neural (teacher v4) | 0.000 | (0.573, 0.483) | (1.455, 1.481) |
| symbolic | 0.000 | (0.573, 0.483) | (1.455, 1.480) |
| 真値 | 0.000 | (0.570, 0.491) | (1.458, 1.510) |

paired RMSE 中央値: symbolic 対 neural **0.00186**、768 系列すべて paired。
有界 drift と短い horizon ($t \le 1$) のため、on-manifold の微小な field 誤差は増幅しない。

### 5.8 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1 drift 蒸留 | **成** | test 0.00635 ≤ 0.10 (次数 3、6 項) |
| H2 diffusion 蒸留 | **成** (構成上の自明さつき) | test $1.4 \times 10^{-12}$ ≤ 0.20、非定数寄与 $1.4 \times 10^{-12}$ ≤ 0.25。ただし teacher の diffusion が族の定義により定数であるため、この合格は蒸留の成功というより構成の帰結である |
| H3 真値回復 | **成** | field 条項: 対真値 drift NRMSE 0.0537 ≤ 0.15、かつ teacher $\hat A$ の相対 Frobenius 誤差 0.206 ≤ 0.35。symbolic 係数の $A$ 解釈は行わない |
| H4 シミュレーション | **成** | 失敗率 0 (差 0)、paired RMSE 0.00186 ≤ 0.5 |
| H5 推定器バイアス | **否** | drift 条項は成る (v4 0.0513 < v1 0.0810) が、$g$ 条項が失敗: $\|\hat g_{v4} - 1\| = 0.0072 > \|\hat g_{v1} - 1\| = 0.0069$。差は $0.0003$ で実質同点であり、系統的な優劣ではない。レジーム別診断 (§5.1) では v4 の優位は明確 (saturated で v1 の 2.3 倍の誤差を解消) であるため、H5 の否は「$g$ の一点比較では優位が検出できない」という限定された意味である |
| H6 変分メッシュ | **成** (分解能の限定つき) | $\|F(1/64) - F(1/32)\| = 0.0277 \le 0.05$。ただしメッシュ間差はモンテカルロ標準誤差 ($0.07$) より小さく、改善量の精密測定にはなっていない (§5.5) |

## 6. 考察

1. **離散化バイアスのレジーム集中は Day 1 の一般化である**。
   Day 1 では非線形な Lorenz 成分で $\hat g$ が膨らみ、Day 2 では線形系で EM バイアスが理論値どおりに現れた。
   本日は sigmoid という有界非線形性において、バイアスが saturated レジームに集中し、
   near-linear レジームでは EM と substep が同等であることを示した。
   「Jacobian の符号と非線形性がバイアスの向きと大きさを決める」という Day 2 の枠組みに、
   「バイアスのレジーム集中」という軸が加わった。補正 (v2/v3) が常に改善するとは限らないことも、
   局所線形化の固定 Jacobian 近似が saturated 領域で悪化方向に働く例として記録する価値がある。
2. **蒸留の成功と論文の主張は別の軸**である (Day 2 と同じ教訓の再確認)。
   本日の field 蒸留の成功 (対 teacher 0.00635) は Tzen & Raginsky の主張 (変分自由エネルギーの挙動) の検証ではない。
   逆に、変分フィットの $A$ に関する負の結果も論文の主張への反証ではない。
   論文が主張していない軸で「失敗」を数えることを避け、主張の軸 (メッシュ) では H6 として分離して判定した。
3. **終端観測のみの変分推論は同定が弱い**。
   共有 $\beta$ の変分モデルは終端分布を説明できても $A$ を同定できない。
   これは DLGM / neural SDE の変分推論一般への注意点である:
   観測が軌道の終端のみの場合、drift の同定には中間観測か、事前分布に対する強い構造制約が必要になる。
   本日の $\hat A_{\mathrm{var}}$ の誤差 5.52 は、その弱同定の定量化である。
4. **共線性の有無は系の構造で決まる**。
   Day 2 の時間依存系では $t$ と $z$ の共線性が構造的に生じたが、本日の自律 2 次元系では VIF $1.00$ と皆無であった。
   off-manifold 診断の結果 (Day 2 では崩壊、本日は安定) もこの構造差と対応する。
   格子診断は「常に崩壊する」ものでも「常に安定する」ものでもなく、系の構造を映す診断である。

## 7. 反省 (Day 4 以降に適用)

- 推定器の $g$ 一点比較で優劣を判定する仮説は、差が $10^{-3}$ 程度の同点域では失敗しやすい。
  $g$ 条項には「同点域」 (例: 差 $< 0.002$ なら同点とみなす) を事前に設けるか、drift のレジーム別誤差を主条項にすること。
- 変分フィットで終端観測のみを使う日は、$A$ の同定を仮説に含めないこと。
  同定の弱さを測ること自体は結果になるが、論文が主張していない軸での「失敗」と混同しないこと。
- メッシュ研究ではモンテカルロ誤差が効果量より大きい場合、共通乱数 (Brownian bridge) によるメッシュ間結合を標準とすること。
  さもなければ「漸進的」という定性的主張の検証が分解能不足になる。
- off-manifold 格子の範囲がデータ範囲と重なる場合は、その旨を明記し、強い外挿主張をしないこと。
  遠方外挿のテストは別途設計する (多項式の発散は既知の限界として記録する)。

## 8. 限界

- 本日の teacher は論文の変分推論そのものではない (§2.2 の逸脱)。
  共有 $\beta$、discretize-then-differentiate、$d = 2$ への切り取りがある。
  論文の forward-mode pathwise 微分、$d = 10$ での実験、サンプルサイズ軸は本日は扱っていない。
- データは再生成であり、論文の乱数系列そのものではない。初期分布は §2 の定義からの宣言的採用である。
- 活性化関数の logistic / softplus の読みの曖昧さは未解消 (§2.2-3)。softplus であった場合、真値 drift の定義が変わる。
- 尤度 teacher の v3 共分散積分は 8 点 Gauss–Legendre 求積による近似であり、厳密な行列指数積分ではない。
- シミュレーション horizon は論文と同じ $t \in [0, 1]$ のみ。長期の定常分布への収束は未検証。
- off-manifold 格子はデータ範囲と重なり、強い外挿テストではない (§5.6)。
- 既存 repo tests (`tests/test_distill.py` 等) は本環境 (environment.yml の完全環境ではない) では未実行。
  本日追加した `tests/test_daily_neural_sde_day03.py` の 4 tests と Day 1・2 の tests は全て通過 (計 13 tests)。
