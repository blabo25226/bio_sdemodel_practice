# Day 16 — Liu et al. (2026) LieStoNet

- Date: 2026-11-08（Asia/Tokyo）
- Paper: Shida Liu, Abhishek Gupta, Sumit Sinha, L. Mahadevan,
  *LieStoNet: Learning Lie Symmetries from Spatiotemporal Data for Stochastic Dynamical Systems*.
- 採録先と年: **ICML 2026**（第 43 回、査読付き本会議、PMLR 306, pp. 76384–76408）。preprint ではない。
  https://proceedings.mlr.press/v306/liu26aq.html
  PDF: https://raw.githubusercontent.com/mlresearch/v306/main/assets/liu26aq/liu26aq.pdf
  arXiv: https://arxiv.org/abs/2608.01582
  Code: https://github.com/sumit-sinha-seas/LieStoNet_Final
- 成果物: config `configs/daily_neural_sde_day16.json` /
  コード `src/daily_neural_sde/day16_liu_liestonet.py` /
  数値 `outputs/daily_neural_sde/day16/results.json`、
  teacher 重み `teacher_liestonet_weights.json`、
  symbolic モデル `symbolic_f.json` / `symbolic_L.json`、
  TeX `symbolic_sde.tex`、展開係数 `symbolic_expanded.json`、
  tests `tests/test_daily_neural_sde_day16.py`（20 tests。日次スイート全体は 147 tests が通過）。
- 実行: `python3 -m src.daily_neural_sde.day16_liu_liestonet`
  （seed 16、Python 3.12.3、PyTorch 2.15.0+cpu、NumPy 1.26.4、CUDA なし、19.7 秒）。
  実行時の shape は query \((3,2)\)、drift \((3,2)\)、Cholesky 因子 \((3,2,2)\)。
  NumPy への書き出しと torch の差の最大絶対値は drift \(2.2\times 10^{-16}\)、因子 \(1.1\times 10^{-16}\)。
  `results.json` の `git_head` は実行時点の親 `71ad851` である。

この 2 次元座標は論文 Table 2 の Example 4 の観測座標であり、遺伝子規制の式でも潜在遺伝子空間の式でもない。
推定式は teacher の近似であって、真の SDE の発見ではない。

## 0. 前日までの反省の適用

実行前に config へ入れた。test のあとでしきい値は動かしていない。

1. **対 teacher と対真値を同じ表に置く (Day 5)**。残差 RMS を NRMSE の横に置く (Day 11、Day 12)。
2. **失敗の帰属に oracle 天井を置く (Day 10、Day 12、Day 15)**。同じ train query への次数 3 の通常最小二乗を、teacher の場と真値の場の両方で test に 1 回だけ当てた。選択には使わない。ドリフトの真値 oracle が 0.05 を超えたら、数値の pass を \(1/x\) の閉じた形とは読まない。拡散の定数はライブラリに入るので、その oracle が機械精度なら拡散の残差はライブラリの欠落ではない。
3. **経路の pass を場の回復と読まない (Day 13、Day 14、Day 15)**。H4 は失敗率ゲートの内側でのみ経路と終端共分散を採点し、ドリフトの teacher 対真値が両成分で 0.30 以下であることも要求する。失敗率がゲートの外なら H4 は不定であり、経路の数値は pass にしない。
4. **VIF ゲートは閉じた形の許可ではない (Day 13、Day 14、Day 15)**。次数 3、定数列を除く。ゲートの内外にかかわらず係数を \(1/x\) として読まない。格子の RMS を軌道上の残差 RMS の横に置く。\(f_x\) の 3 次項の RMS をその成分の RMS で割り、0.15 未満なら「小さい項」と記録する。
5. **データ NLL が正規フロアに近いときの一致はドリフト回復ではない (Day 14、Day 15)**。\(|f|\Delta t / \|\sigma\|\sqrt{\Delta t}\) が 0.25 未満なら `drift_step_ratio_blind` と記録する。検証 NLL と真値フィールドの NLL の差が 0.05 未満なら `nll_at_floor` とし、そのフロアをドリフト回復の証拠にしない。
6. **法則指標にはモンテカルロフロアを置く (Day 10、Day 14、Day 15)**。フロアが teacher のしきい値を超えたら H5 は不定である。終端共分散を平均経路の横に残す。
7. **真値が 0 の成分は NRMSE を使わない (Day 11)**。Cholesky の非対角 \(\ell_{21}\) の真値は 0 なので、ゲートは RMS である。選択スコアも対角の NRMSE と、teacher の対角 RMS で割った非対角 RMS の和であり、非対角の NRMSE ではない。
8. **ドメイン妥当性を失敗クラスに分ける (Day 4)**。\(x < 0.2\) は `below_floor` であり、半径外や非有限とは別である。

## 1. 論文要約

### 1.1 問題と提案

未知の Ito SDE の Lie 点対称性を、軌道の離散サンプルだけから見つける問題である。
LieStoNet は二段である。Stage 1 は増分の局所ガウス尤度で、微分可能な surrogate
\(dx_t = f_\theta(t,x_t)\,dt + \sigma_\theta(t,x_t)\,dW_t\) を学ぶ（式 (7)、(8)）。
Stage 2 はその surrogate を凍らせ、射影可能な生成子 \(X_i = \tau_i(t)\partial_t + \xi_i(t,x)\partial_x\) を、
SDE の決定方程式と Lie 括弧の閉包で学ぶ。
スカラー Ito SDE の対称性代数の次元は高々 3、フルランク拡散の \(n\) 次元系は高々 \(n+2\) なので、
生成子の本数 \(m\) は有限回のスイープで閉包損失の最小を選ぶ。
Example 1–4 と 10 次元ブラウン運動、BTC/USDT で、解析的な代数との主角度を報告している。

### 1.2 選定理由

Day 12 は二時刻カーネル、Day 13 はシミュレーションフリーの変分境界、Day 14 はオフライン RL、Day 15 は文脈からの認識モデルである。
本日は、増分尤度で学んだニューラル drift とニューラル拡散を、Lie 対称性の surrogate として使う論文を選んだ。
採録は ICML 2026 で、drift と diffusion の両方がニューラルネットである査読付き論文のうち新しいものである。
系は Table 2 の Example 4 である。ドリフトは \((1/x,\,1)\) で、\(1/x\) は冪のライブラリに入らない。
拡散は定数の単位行列で、Cholesky ネットとしてパラメトライズされる。
Lorenz、OU、van der Pol、Lotka–Volterra、スコア、変分境界、NUNO、FIM の Stuart–Landau は避けた。
Example 1 のブラウン運動、Example 2 の線形不安定、Example 3 の 2 次元線形 Langevin は使っていない。

## 2. データと Neural SDE の入手・再構成

### 2.1 データと teacher

真の Ito SDE は観測座標 \((x,y)\) の上で、Table 2 の通り

$$
\begin{aligned}
dx &= \frac{1}{x}\,dt + dW_1,\\
dy &= 1\,dt + dW_2.
\end{aligned}
$$

\(x>0\) から始めると \(x^{2}\) の Ito 補正は \(d(x^{2}) = 3\,dt + 2x\,dW_1\) であり、連続過程は 3 次元 Bessel 過程と同じく 0 に届かない。
Euler–Maruyama は床を越えうる。生成は床も反射も置かず、\(x < 0.2\)、半径 12 の外、非有限のいずれかに触れた軌道は捨てて引き直す。
捨てた本数は train 4/68、validation 1/33、test 0/48 である。
残った軌道は \(T=2\)、\(\Delta t=0.01\)、200 步。初期値は \(x\sim U[0.5,3]\)、\(y\sim U[-2,2]\)。
分割は軌道単位で train 64、validation 32、test 48。seed 16 の単一ストリームでこの順に引く。
平均と標準偏差は train の非終端状態だけから取る。落とした遷移は train も validation も 0 である。
query 数は train 12800、validation 6400、test 9600。
train の \(x\) は 0.217 から 6.86、\(y\) は \(-2.94\) から 6.19 である。

teacher は Appendix P の Example 4 の縮小である。
入力は標準化した \((x,y)\) だけで、時刻は入らない。
ドリフトは swish MLP、拡散は別の swish MLP が \((\ell_{11}^{\mathrm{raw}},\ell_{21},\ell_{22}^{\mathrm{raw}})\) を出し、

$$
\ell_{11}=\mathrm{softplus}(\ell_{11}^{\mathrm{raw}})+10^{-3},\qquad
\ell_{22}=\mathrm{softplus}(\ell_{22}^{\mathrm{raw}})+10^{-3},\qquad
L=\begin{pmatrix}\ell_{11}&0\\ \ell_{21}&\ell_{22}\end{pmatrix},\qquad
\Sigma=LL^{\top}.
$$

幅 64、隠れ層 2 枚、Glorot 一様、バイアス 0、float64。
損失はノートブックの多変量ガウス NLL（\(2\pi\) を落とす）に、拡散ステップでの JVP 平滑 \(10^{-3}\)、対角の分散 \(5\times 10^{-4}\)、白色化 \(5\times 10^{-3}\) を足す。
AdamW、weight decay 0、勾配クリップ 1。外側 1 步につきドリフトを学習率 \(2\times 10^{-3}\) で 3 回、拡散を \(10^{-3}\) で 1 回。バッチ 512、外側 1500 步。
検証 NLL は 25 步ごとに測り、最小の重みを戻す。場の NRMSE は記録するが選択に使わない。

選択は step 375、検証 NLL \(-3.6228\)。
真の \(f\) と \(L=I\) で同じ検証遷移を評価したフロアは \(-3.6233\) である。
差は \(4.65\times 10^{-4}\) nats で、凍結した 0.05 より小さく、`nll_at_floor` である。

### 2.2 逸脱の明示

1. 幅 128、深さ 3、軌道 2048、外側 20000 步、バッチ 8192 は範囲外である。幅 64、深さ 2、軌道 64/32/48、外側 1500、バッチ 512、CPU、19.7 秒である。
2. ノートブックは全軌道を 1 つの学習集合にする。本日は軌道単位の train / validation / test であり、標準化は train だけから取る。
3. チェックポイントは検証 NLL の最小である。ノートブックは固定步数の最後の重みを使う。場の誤差では選んでいない。
4. Table 2 は \(dy=dt+dW_2\) なので \(a_2=1\)。公開ノートブックの `truth` は \(a_2=0.5\) である。本日は Table 2 に合わせた。Appendix M の生成子は一般の \(a_2\) で書かれている。
5. ノートブックは生成中に \(|x|\) を 0.2 へ押し戻し、床の上では \(1/x_{\mathrm{safe}}\) を使う。本日の生成はその射影を持たない。床に触れた軌道は捨てて引き直す。学習集合は生存で条件付けられている。
6. JVP 平滑は拡散の更新步だけで評価する。ドリフト步ではこの罰則は \(L\) のパラメータにしか依存しないので、ドリフトの勾配は変わらない。
7. Stage 2 の生成子学習、\(m\) のスイープ、主角度、Fokker–Planck 対称性、10 次元ブラウン運動、BTC/USDT、Example 1–3 は範囲外である。解析的な生成子 \(v_2=\partial_y\) と \(v_3=2t\partial_t+x\partial_x+(y+t)\partial_y\) は、学んだドリフトの探針である。論文の表の主角度ではない。
8. 訓練の NLL はノートブックと同じく \(\log(2\pi)\) を落とす。\(\Delta t\) が一定なので argmin は完全なガウス NLL と一致する。

## 3. 仮説（config に事前凍結）

実行前に `configs/daily_neural_sde_day16.json` へ書いた。しきい値の位置は、学習前の解析的 SDE だけで見積もった（箱の上の次数 3 は \(f_x\) の NRMSE 約 0.04、軌道雲では約 0.18、その多項式の \(v_3\) 残差は約 0.33、真値 EM の床ヒットは約 0.03、独立アンサンブルの平均経路フロアは約 0.19）。test のあとでしきい値は動かしていない。

- **H1** (ドリフト、成分ごと): symbolic 対 teacher \(\le 0.25\)、teacher 対真値 \(\le 0.30\)、symbolic 対真値 \(\le 0.40\)。両成分とも必要。残差 RMS は併記し、ゲートにはしない。
- **H2** (Cholesky): 対角 \(\ell_{11},\ell_{22}\) は同じ 3 比較を \(\le 0.20\)、\(\le 0.15\)、\(\le 0.25\)。\(\ell_{21}\) は RMS で、teacher \(\le 0.10\)、symbolic 対 teacher \(\le 0.12\)、symbolic 対真値 \(\le 0.12\)。
- **H2c**: test query で symbolic の対角が 0 以下の行の割合 \(\le 0.02\)。
- **H3**: 係数回復の仮説は置かない。次数 3 の VIF 最大（定数列を除く）が 10 を超えたら不定。10 以下でも \(1/x\) としては読まない。
- **H4**: test の 48 本の初期条件、同じ Wiener 増分。失敗率は各モデル \(\le 0.05\) かつ symbolic と teacher の差 \(\le 0.02\)。失敗クラスは `non_finite`、`nonpositive_chol`、`below_floor`、`outside_radius`。失敗率の最大が 0.05 を超えるか H7 が外れるか共分散が定義できなければ H4 は不定。内側のときだけ、対の経路 RMSE（symbolic 対 teacher \(\le 0.40\)、teacher 対真値 \(\le 0.50\)）、終端共分散の相対 Frobenius（\(\le 0.50\)、\(\le 0.65\)）、ドリフトの teacher 対真値が両成分で \(\le 0.30\)、をすべて要求する。
- **H5**: 初期値をすべて \((1.75, 0)\) にした 64 本。平均経路 RMSE は teacher \(\le 0.45\)、symbolic \(\le 0.60\)。終端共分散の相対 Frobenius は teacher \(\le 0.55\)、symbolic \(\le 0.75\)。失敗率 \(>0.05\)、または真値対真値のフロアが teacher のしきい値を超えたら不定。
- **H6**: 探針格子（\(x\in[0.6,3.5]\)、\(y\in[-1,3]\)、\(t\in\{0.5,1.0,1.5\}\)、選択には未使用）で、\(v_2\) の teacher 平均絶対残差 \(\le 0.08\)、\(v_3\) の teacher \(\le 0.15\)、\(v_3\) の symbolic \(\le 0.15\)。symbolic が外れて、真値への次数 3 oracle との相対差が 0.25 以内なら `library_breaks_scaling_identity`。teacher が先に外れたら `teacher_breaks_scaling`。
- **H7**: teacher の \(\Delta t=0.01\) と \(\Delta t=0.005\)（細かい標準正規 2 本の和を \(\sqrt{2}\) で割った粗い増分）の終端 RMSE \(\le 0.35\)、かつ両方が有限な割合 \(\ge 0.95\)。

## 4. 方法

teacher への直接 query。特徴は非終端状態 \((x,y)\)。時刻は teacher の引数ではないのでライブラリに入れない。
SINDy の標準化は train query だけから取る。ライブラリは \((x,y)\) の冪だけである。
次数 \(\{1,2,3\}\)、しきい値 \(\{0.02,0.05,0.1,0.2,0.35\}\)、ridge \(10^{-4}\)。
drift は validation の pooled NRMSE 最小。拡散は対角 2 成分の NRMSE の平均に、対角 RMS で割った \(\ell_{21}\) の RMS を足したスコアの最小。
同点はより小さい次数、次に小さいしきい値。test はこの選択のあと 1 回。

選択結果: drift は次数 3・しきい値 0.02（活性 11、成分あたりライブラリ 10 項、検証スコア 0.0145）。
拡散は次数 3・しきい値 0.02（活性 10、検証スコア 0.0162）。

診断として、次数 3・通常最小二乗を teacher と真値の両方に train query で当て、test で 1 回評価した。選択には使わない。
格子は train の 5–95 パーセンタイルを半分だけ広げ、\(x\) の下限を 0.25 で止めた \(7^{2}\) 点である。
床の近くの帯は \(x\in[0.25,0.45]\) と train の \(y\) の 10–90 パーセンタイルである。どちらも選択には使わない。
シミュレーションは失敗をラッチし、上の 4 クラスを分ける。

## 5. 結果

### 5.1 仮説の成否一覧

| 仮説 | 指標 | 値 | しきい値 | 判定 |
|---|---|---|---|---|
| H1 drift | symbolic 対 teacher（\(f_x,f_y\)） | 0.0154 / 0.0101 | \(\le 0.25\) | この項は内側 |
| H1 drift | teacher 対真値 | 0.365 / 0.164 | \(\le 0.30\) | **fail**（\(f_x\)） |
| H1 drift | symbolic 対真値 | 0.364 / 0.162 | \(\le 0.40\) | この項は内側 |
| H1 残差 RMS | teacher 対真値 | 0.223 / 0.164 | （併記） | teacher RMS 0.742 / 1.027、真値 RMS 0.610 / 1 |
| H1 天井 | 次数 3 OLS 対真値 / 対 teacher | 0.140 / \(2.6\times 10^{-3}\) | （診断） | 真値 oracle は 0.05 を超える。teacher の場は次数 3 でほぼ閉じる |
| H2 対角 | symbolic 対 teacher | 0.0181 / 0.00829 | \(\le 0.20\) | **pass** |
| H2 対角 | teacher 対真値 | 0.0254 / 0.0167 | \(\le 0.15\) | pass |
| H2 対角 | symbolic 対真値 | 0.0170 / 0.0148 | \(\le 0.25\) | pass。単位行列そのものではない（5.3） |
| H2 \(\ell_{21}\) RMS | teacher / symbolic 対 teacher / symbolic 対真値 | 0.0172 / 0.00035 / 0.0172 | \(\le 0.10\) / \(0.12\) / \(0.12\) | pass |
| H2 天井 | 次数 3 OLS 対真値（3 成分） | \(10^{-13}\) 以下 | （診断） | 定数はライブラリに入る |
| H2c | 対角 \(\le 0\) の行 | 0 | \(\le 0.02\) | pass |
| H3 VIF | 次数 3、定数列を除く | 4.20 | \(\le 10\) | pass（ゲート。係数回復ではない） |
| H4 失敗 | truth / teacher / symbolic | 0.125 / 0.104 / 0.104 | \(\le 0.05\)、差 \(\le 0.02\) | **fail**。H4 本体は不定 |
| H4 経路 | symbolic 対 teacher / teacher 対真値 | 0.0118 / 0.150 | \(\le 0.40\) / \(\le 0.50\) | 失敗率ゲートの外。採点しない |
| H4 共分散 | symbolic 対 teacher / teacher 対真値 | 0.00776 / 0.243 | \(\le 0.50\) / \(\le 0.65\) | 同上 |
| H4 ドリフト | teacher 対真値 | 0.365 / 0.164 | \(\le 0.30\) | 診断フラグ `path_match_without_field_recovery` は true |
| H5 平均経路 | teacher / symbolic / 真値フロア | 0.0779 / 0.0801 / 0.166 | \(\le 0.45\) / \(\le 0.60\) | **pass**。対の誤差は独立フロアより小さい |
| H5 共分散 | teacher / symbolic / 真値フロア | 0.163 / 0.164 / 0.417 | \(\le 0.55\) / \(\le 0.75\) | pass。フロア 0.417 は teacher のしきい値 0.55 の内側 |
| H5 失敗 | truth / teacher / symbolic | 0.0313 / 0.0313 / 0.0313 | \(\le 0.05\) | pass |
| H6 \(v_2\) | teacher の平均絶対残差 | 0.0470 | \(\le 0.08\) | この項は内側。判定は次の項で止まった |
| H6 \(v_3\) | teacher / symbolic / 真値 oracle | 0.480 / 0.474 / 0.161 | teacher \(\le 0.15\) | **fail**（`teacher_breaks_scaling`） |
| H6 真値 | \(v_3\) の中心差分 | \(4.8\times 10^{-11}\) | （健全性） | 解析的恒等式は数値的に 0 |
| H7 | 終端 RMSE / 有限な割合 | 0.00192 / 1 | \(\le 0.35\) / \(\ge 0.95\) | pass |

1 步の尺度は、train 上の真値で平均 \(|f|\Delta t = 0.0116\)、平均 \(\|\sigma\|\sqrt{\Delta t} = 0.141\)、比 0.0821 である。
test では 0.0116、0.141、比 0.0818 である。`drift_step_ratio_blind` である。

検証の場（選択には未使用）は、ドリフト NRMSE 0.342 / 0.130、対角 NRMSE 0.0300 / 0.0167、\(\ell_{21}\) の RMS 0.0144 である。
選択は検証 NLL のままである。

### 5.2 推定した symbolic SDE

標準化座標の活性はドリフト 11、拡散 10 である。
train の平均は \((2.401,\,0.886)\) で 0 ではない。
係数は `symbolic_sde.tex` と一致する。これは teacher の多項式近似であり、上の真の SDE ではない。
拡散は下三角の因子で、増分は \(L\sqrt{\Delta t}\,\varepsilon\) である。

$$
\begin{aligned}
dx &= f_x\,dt + \ell_{11}\,dW_1,\\
dy &= f_y\,dt + \ell_{21}\,dW_1 + \ell_{22}\,dW_2,
\end{aligned}
$$

$$
\begin{aligned}
f_{x} &= 1.884 - 0.03737\,y - 0.6884\,x + 0.005148\,y^{2} + 0.05477\,x^{2},\\
f_{y} &= 1.241 + 0.1601\,y - 0.1261\,x - 0.04506\,x y - 0.01323\,x^{2} + 0.005092\,x^{3},\\
\ell_{11} &= 0.983,\\
\ell_{21} &= 0.04833 + 0.01001\,y - 0.03722\,x - 0.0008752\,y^{2} - 0.001703\,x y + 0.005444\,x^{2} \\
&\quad + 0.000165\,y^{3} - 0.0005099\,x^{2} y,\\
\ell_{22} &= 0.9852.
\end{aligned}
$$

\(f_x\) の主な単項式を、teacher への次数 3 通常最小二乗と並べる。真値 \(1/x\) には対応する単項式がない。

| 項 | teacher OLS の \(f_x\) | symbolic の \(f_x\) |
|---|---|---|
| \(1\) | 1.876 | 1.884 |
| \(x\) | \(-0.693\) | \(-0.688\) |
| \(y\) | \(-0.0223\) | \(-0.0374\) |
| \(x^{2}\) | 0.0588 | 0.0548 |
| \(y^{2}\) | 0.00547 | 0.00515 |
| \(x^{3}\) | \(-0.000335\) | 0 |

\(f_y\) の定数項は teacher OLS で 1.240、symbolic で 1.241 である。真値は 1 である。
VIF は 4.20 でゲート 10 の内側である。
そのうえで \(f_x\) の \(x\) は \(-0.688\) であり、train 平均 \(x=2.40\) での \(d(1/x)/dx=-1/x^{2}=-0.174\) とも、関数 \(1/x\) とも一致しない。
点ごとの値は、\(x=1\) で真値 1 に対し teacher 1.243、symbolic 1.250、\(x=0.3\) で真値 3.333 に対し teacher 1.666、symbolic 1.682 である。
床の近くで teacher は \(1/x\) のおよそ半分である。symbolic はその teacher に沿う。

test 上の \(f_x\) の 3 次部分の RMS 比は 0 である。
teacher OLS の \(x^{3}\) は \(-3.4\times 10^{-4}\) なので、この 0 は大きな 3 次項がしきい値で消えた結果ではない。
真値への次数 3 oracle は NRMSE 0.140 で、teacher の 0.365 より真値に近い。
多項式の表現力は teacher の尤度学習より先に \(1/x\) へ届いている。届いていないのは teacher の場である。

格子（選択に未使用）のドリフト RMS は、symbolic 対 teacher 0.0602、teacher 対真値 0.685、symbolic 対真値 0.685 である。
軌道上の teacher 対真値の残差 RMS 0.195 の横に置くと、比は 3.51 である。
床の帯では teacher 対真値 1.053、symbolic 対 teacher 0.0177 である。

### 5.3 拡散の pass は単位行列の係数 1 ではない

\(\Sigma=I\) の次数 3 通常最小二乗は test で \(10^{-13}\) 以下である。
選択した \(\ell_{11}\) は 0.983、\(\ell_{22}\) は 0.9852 で、状態には依存しない。
\(\ell_{21}\) の RMS は 0.0172 で、ゲート 0.10 の内側だが 0 ではない。
H2 の数値はすべてゲートの内側である。推定した因子は teacher を通したほぼ定数の拡散であり、論文の \(\sigma=I\) を係数まで回復した式ではない。

### 5.4 フロア上の尤度、床を越える軌道、スケーリング恒等式

検証 NLL のフロアとの差 \(4.65\times 10^{-4}\) nats は、残差 RMS 0.223 と 0.164 が \(\Delta t=0.01\) で尤度に入れる大きさと同じ桁である。
\(\tfrac12(\mathrm{RMS}_{x}^{2}+\mathrm{RMS}_{y}^{2})\,\Delta t / \sigma^{2}\) は約 \(3.8\times 10^{-4}\) nats である。
フロアに近いことは、ドリフト NRMSE が 0.30 の内側にあることを意味しない。

H4 の失敗は truth 6/48、teacher 5/48、symbolic 5/48 で、クラスはすべて `below_floor` である。
非有限、非正の対角、半径外は 0 である。失敗した質量は分母に残してある。
失敗率 0.125 は 0.05 の外なので、H4 は不定である。
対の経路 RMSE 0.150 と終端共分散 0.243 はそれぞれの絶対しきい値の内側にあるが、失敗率ゲートの外なので pass にしない。
同じ診断で、ドリフトは 0.30 の外であり、`path_match_without_field_recovery` は true である。

H5 の平均経路は teacher 0.0779、symbolic 0.0801、独立な真値対真値のフロア 0.166 である。
比較は同じ Wiener 増分なので、対の誤差が独立フロアより小さいことは分解能の中にいることを示す。
終端共分散は teacher 0.163、symbolic 0.164、フロア 0.417 である。
フロア 0.417 は teacher のしきい値 0.55 より小さいので、この pass は絶対しきい値がフロアを上回っていることによる。
失敗率は 2/64 = 0.0313 でゲートの内側である。H1 は fail のままである。

\(v_3\) のドリフト決定方程式は、生成子が状態について線形なので拡散項を持たない。
真値の中心差分残差は \(4.8\times 10^{-11}\) である。
真値への次数 3 oracle の平均絶対残差は 0.161 で、凍結した 0.15 のすぐ上にある。
teacher は 0.480、symbolic は 0.474 である。
symbolic は teacher に沿っており、真値の多項式天井 0.161 には届いていない。
判定は teacher が先に外れたので `teacher_breaks_scaling` である。
\(v_2=\partial_y\) の残差 0.047 は 0.08 の内側で、\(y\) への漏れはこのゲートだけでは失敗にならない。
時刻並進 \(v_1=\partial_t\) は、teacher が時刻を引数に持たないので残差が構造的に 0 である。これはデータの対称性ではなくアーキテクチャの対称性である。

## 6. 考察

増分尤度は拡散を定数の近くまで戻し、検証 NLL をフロアの \(5\times 10^{-4}\) nats 以内に置いた。
同じ尤度は \(1/x\) を残している。1 步でドリフトが進む量は拡散の増分の 0.082 倍であり、観測された NLL の差はドリフト残差の二次の寄与と桁が合う。
次数 3 の多項式を真値に直接当てると \(f_x\) の NRMSE は 0.140 まで下がる。teacher は 0.365 で、その天井より悪い。
SINDy はこの teacher を軌道上で NRMSE 0.015 まで写す。teacher 自身の次数 3 oracle は 0.0026 なので、symbolic と真値の 0.364 はライブラリの欠落ではなく teacher の場である。
VIF 4.20 はゲートの内側で、\(x\) の係数 \(-0.688\) は \(1/x\) の接線 \(-0.174\) ではない。
\(f_x\) の 3 次項は 0 だが、teacher OLS の \(x^{3}\) が \(10^{-4}\) なので、Day 14 の「小さい項がしきい値で消えた」機構そのものでもない。
機構が違っても、読む手順は同じである。軌道上の NRMSE、格子の RMS、単項式の係数、スケーリング残差を同じ表に置き、VIF の pass を閉じた形と呼ばない。

スケーリング恒等式は値の誤差より先に壊れる。
\(v_3\) の残差は \(f + x f_x + (y+t)f_y\) の形で、\(1/x\) を特徴づける。
真値の多項式天井が 0.161 なのに teacher は 0.480 なので、この縮小 surrogate は決定方程式に差し込む段階に達していない。
論文の Stage 2 が同じ幅と步数で主角度を戻すかは、この縮小からは言えない。

拡散は逆である。定数はライブラリに入り、oracle は機械精度、選択式の対角は 0.983 と 0.985 である。
H2 の pass は teacher の近似としての pass である。

床を越える軌道は真値でも 48 本中 6 本ある。Euler–Maruyama は 3 次元 Bessel 過程の非到達を守らない。
失敗率を落としてから経路を読む、という凍結した順序では H4 は不定である。

## 7. 反省

1 步 NLL がフロアの \(4.6\times 10^{-4}\) nats 以内でも、\(|f|\Delta t/\|\sigma\|\sqrt{\Delta t}=0.082\) のときドリフト NRMSE は 0.365 と 0.164 のまま残った。
その残差の尤度寄与は約 \(3.8\times 10^{-4}\) nats で、フロア差と同じ桁である。
スケーリング生成子の残差は値の NRMSE より厳しい。teacher の \(v_3\) は 0.480 で、真値への次数 3 oracle の 0.161 より大きく、失敗はライブラリの欠落ではない。
teacher の次数 3 oracle が 0.0026 なら、symbolic の対 teacher 0.015 は teacher の場であり、\(f_x\) の 3 次項が 0 でも小さい項が落ちたとは読まない。
VIF 4.20 の内側で \(x\) の係数 \(-0.688\) を \(1/x\) として読まない。
格子の drift RMS 0.685 を、軌道上の残差 RMS 0.195 の横に置く。
失敗率 0.125 の経路 RMSE 0.150 は不定のまま残す。
平均経路 0.078 はフロア 0.166 の横に置く。
