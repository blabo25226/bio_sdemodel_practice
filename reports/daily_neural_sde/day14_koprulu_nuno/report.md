# Day 14 — Koprulu, Djeumou & Topcu (2025) NUNO

- Date: 2026-11-07（Asia/Tokyo）
- Paper: Cevahir Koprulu, Franck Djeumou, Ufuk Topcu,
  *Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL* (NUNO).
- 採録先と年: **ICLR 2025**（査読付き本会議）。preprint ではない。
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html
  PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/0f25eb6e9dc26c933a5d7516abf1eb8c-Paper-Conference.pdf
  部分再実装: https://github.com/matthew-lowery/NSDE_ORL
  先行研究: Djeumou, Neary, Topcu, CoRL 2023（physics-constrained neural SDE）。
- 成果物: config `configs/daily_neural_sde_day14.json` /
  コード `src/daily_neural_sde/day14_koprulu_nuno.py` /
  数値 `outputs/daily_neural_sde/day14/results.json`、
  teacher 重み `teacher_nuno_weights.json` と `teacher_aleatoric_weights.json`、
  symbolic モデル `symbolic_f.json` / `symbolic_g.json`、
  TeX `symbolic_sde.tex`、展開係数 `symbolic_expanded.json`、
  tests `tests/test_daily_neural_sde_day14.py`（21 tests。Day 1–14 を合わせて 101 tests が通過）。
- 実行: `python3 -m src.daily_neural_sde.day14_koprulu_nuno`
  （seed 14、Python 3.12.3、PyTorch 2.15.0+cpu、NumPy 1.26.4、CUDA なし、6.9 秒）。
  実行時の shape は \(q:(8,)\)、\(v:(8,)\)、\(u:(8,)\)、drift \((8,2)\)、velocity diffusion \((8,)\)、\(\eta:(8,1)\)。
  NumPy への書き出しと torch の差の最大絶対値は \(4.4\times 10^{-16}\)。
  `results.json` の `git_head` は実行時点の親 `cd95ed6` である。

この 2 次元座標は位置と速度の観測座標であり、遺伝子規制の式でも潜在遺伝子空間の式でもない。

## 0. 前日までの反省の適用

実行前に config へ入れた。test のあとでしきい値は動かしていない。

1. **対 teacher と対真値を同じ表に置く (Day 5)**。残差 RMS を NRMSE の横に置く (Day 11、Day 12)。
2. **失敗の帰属に oracle 天井を置く (Day 10、Day 12)**。同じ train query への次数 3 の通常最小二乗を、teacher の場と真値の場の両方で test に 1 回だけ当てた。選択には使わない。oracle の test NRMSE が \(10^{-6}\) 以下なら、対真値の残差は teacher の場に帰属する。
3. **経路の pass を場の回復と読まない (Day 13)**。H4 は平均経路に加えて、終端共分散（symbolic 対 teacher と teacher 対真値）と、速度ドリフトの teacher 対真値ゲートをすべて要求する。平均経路だけが通ってドリフトゲートが外れた場合は `path_match_without_field_recovery` として H4 を fail にする。1 步の \(|f_v|\Delta t\) と \(g\sqrt{\Delta t}\) は平均の比で併記する。
4. **VIF ゲートは閉じた形の許可ではない (Day 13)**。次数 3、定数列を除く。ゲートの内外にかかわらず係数を閉じた形として読まない。格子の RMS を軌道上の残差 RMS の横に置く。
5. **同じ高い床の上の小さな差は優劣ではない (Day 12、Day 13)**。H6 は全モデルの ID の \(g\) NRMSE、\(\eta\) から作る \(h\) の ID から OOD への上昇、全モデルと aleatoric-only の \(g\) 上昇の差、の三つを要求する。両方の ID NRMSE が 0.80 以上なら共倒れと記録する。
6. **法則指標にはモンテカルロフロアを置く (Day 10)**。フロアが teacher のしきい値を超えたら H5 は不定である。終端共分散を平均経路の横に残す。

## 1. 論文要約

### 1.1 問題と提案

オフラインのモデルベース RL は、静的な操作ログからダイナミクスを学び、そのモデルで方策を改善する。
データが覆っていない領域でモデルが報酬を過大にすると、方策がその領域を使う（model exploitation）。
NUNO はダイナミクスを次のニューラル SDE として学ぶ（式 (5)）。

$$
ds = f_\theta(s, a)\,dt + \bigl(\sigma_\phi(s, a) + h_\phi(\eta_\phi(s, a))\bigr)\,dW.
$$

\(\sigma_\phi\) はデータ近傍の確率的ゆらぎ（aleatoric）で、\(h_\phi\circ\eta_\phi\) はデータから離れると大きく、かつ有界な拡散（epistemic）である。
\(h\) はスケール付きシグモイド \(W_{\max}\operatorname{sigmoid}(W\eta+b)\) で、\(W>1\) とする。
\(\eta\) はデータ点でほぼ 0、勾配もほぼ 0 で、データ点の近傍では強凸になるように罰する（式 (3)–(4)）。
ドリフトには剛体の構造を入れる（式 (8)）。位置の微分は速度、速度の微分は \(G(v)a + H(s)v\) に残差を足したものである。
学習はホライズン 2 の列に対する Euler–Maruyama の負の対数尤度と、上の距離罰則の和である（式 (6)–(7)、付録 C.4）。
方策側では \(\eta\) で報酬を罰し、蓄積した \(\eta\) が CVaR しきい値を超えたらロールアウトを切る（式のあとの 4.3 節）。
実験は D4RL と NeoRL の MuJoCo（halfcheetah、hopper、walker2d）である。
低品質データで既存法を最大 93% 上回り、高品質データでも同程度か最大 55% 上回る、と本文は述べている。
付録 D は、罰則に \(\eta\) を使う NUNO と、\(\sigma\) だけを使う NUNO-al を比べている。

### 1.2 選定理由

Day 10 は逆時間のスコア SDE、Day 11 は初期状態に乗せた認識的拡散、Day 12 は二時刻カーネルスコア、Day 13 はシミュレーションフリーの変分境界である。
本日は、既知の運動学と未知のニューラル項を合成したドリフトと、データからの距離で頭打ちになる拡散を、オフラインの操作列から同時に学ぶ、という未使用の学習を選んだ。
採録は ICLR 2025 で、drift と diffusion がどちらもニューラルネットである査読付き論文のうち新しいものである。
同じ年の Finite Dimensional Matching と SDE Matching はすでに Day 12 と Day 13 で蒸留した。
系は MuJoCo を使わず、式 (8) の位置・速度分解がそのまま入る 2 次元の制御付き質点にした。
Lotka–Volterra、Lorenz、van der Pol、OU、逆時間 VP、変分境界、二時刻カーネルスコアは避けた。

## 2. データと Neural SDE の入手・再構成

### 2.1 データと teacher

真の Ito SDE は、観測座標 \((q, v)\) とスカラー入力 \(u\) の上で

$$
dq = v\,dt, \qquad
dv = \bigl(u - 0.3\,v - 0.8\,v^{3}\bigr)\,dt + 0.15\,dW.
$$

位置の拡散は 0 である。
行動方策は \(u = \mathrm{clip}(-0.2 q - 0.2 v + \varepsilon, -1.2, 1.2)\)、\(\varepsilon\sim N(0, 0.8^{2})\)。
初期値は \(q\sim U(-0.7, 0.7)\)、\(v\sim U(-0.8, 0.8)\)。
\(T=1.5\)、\(\Delta t=0.05\)、30 ステップ。生成は保存刻みの Euler–Maruyama そのもので、サブステップは置かない。
分割は軌道単位で train 64、validation 32、test 64。seed 14 の単一ストリームでこの順に引く。
\((q, v, u)\) の平均と標準偏差は train だけから取る。

teacher は式 (8) の構造を持つ。位置ドリフトは \(v\) に固定し、位置の残差は 0 に固定する。
速度ドリフトは \(G(v)\,u + H(q, v)\,v + r(q, v, u)\)。
\(G, H, r\) は幅 32、tanh、隠れ 2 層。速度拡散は \(\mathrm{softplus}(\sigma_\phi)+0.02 + h(\eta)\)。
\(\eta\) は幅 32 の SiLU ネット、\(h = 0.5\,\mathrm{sigmoid}((1+\mathrm{softplus}(w))\eta + b)\)、\(b\) の初期値は \(-4\)。
強凸係数は \(\mu = 4\,\mathrm{sigmoid}(\mathrm{NN})\)。
データ項は teacher forcing したホライズン 2 の速度増分の Gauss NLL で、分散は \(g^{2}\Delta t\) である。
罰則の係数は \(\lambda_{\mathrm{data}}=1\)、\(\lambda_{\mathrm{sc}}=0.05\)、\(\lambda_{\mathrm{grad}}=1\)、\(\lambda_{\mu}=0.01\)、\(\lambda_{\mathrm{res}}=0.02\)。
局所サンプルは各アンカー 8 組、半径 0.25。
Adam、学習率 \(10^{-3}\)、800 step、バッチ 64、勾配クリップ 1。
検証の速度 NLL が最小のチェックポイントを戻す。選択スコアに罰則は入れない。
全モデルの最良は step 440、検証 NLL \(-2.8906\)。
正規増分の理論フロアは \(0.5(1+\log(0.15^{2}\cdot 0.05)) = -2.895\) である。

H6 のベースラインは付録 D の NUNO-al に対応する aleatoric-only で、同じドリフト構造、同じ予算、\(h=0\)。
最良は同じ step 440、検証 NLL \(-2.8905\)。

### 2.2 逸脱の明示

1. D4RL、NeoRL、MuJoCo、報酬学習、SAC による方策最適化は範囲外である。式 (5)–(8) と付録 C.4 のダイナミクス学習だけを、上の立方摩擦の質点で行う。軌道を分けられるように、短い列を多数使う。
2. 幅は 32（\(\mu\) は 16）、隠れ層は 2 である。論文は 256 または 64、ドリフトは隠れ 3 層、学習率は 0.01 から 0.001 へ 5000 step で減衰する。本日は 800 step、一定の学習率 \(10^{-3}\)、CPU。
3. 位置ドリフトは \(v\)、位置拡散は 0 に固定した。速度の残差だけを \(\lambda_{\mathrm{res}}=0.02\) で罰する。論文はこの残差罰則の数値を書いていない。
4. データ項は速度増分の teacher-forced な Euler–Gauss NLL である。論文の 1 粒子残差に \((\Sigma^{-1})\) を掛ける式は、拡散係数を共分散のように使っており、\(g\) を偏らせる。実装したのは、同じ段落が述べている Gauss 遷移の尤度で、分散は \(g^{2}\Delta t\) である。
5. aleatoric の \(\sigma\) は softplus に床 0.02 を足す。論文の \(\sigma\) ネットは制約なしである。
6. 強凸の不足は \(\nabla\eta(x)\cdot(y-x)\) で測る。印刷された内積は \((x-y)\) で、本文が名前を挙げている不等式と向きが逆である。\(\mu\) は上限 4 のシグモイドで、論文の非有界な指数ではない。
7. 局所サンプルは 8 組、半径は元座標で 0.25 である。論文は 20 組、MuJoCo 状態で半径 0.1。
8. \(\lambda_{\mathrm{grad}}=1\)。付録 C は高次元の MuJoCo NLL に対して \(10^{-4}\) を置く。1 チャネルの速度 NLL ではその値は \(\eta^{2}\) を次数 1 で押さない。1 は学習の前に凍結した。
9. H6 のベースラインは時刻だけの拡散ではなく、付録 D の aleatoric-only である。
10. \((q, v, u)\) の平均と標準偏差は train 軌道だけから取る。ネットの入力はその標準化座標で、報告する場は元座標である。

## 3. 仮説（config に事前凍結）

実行前に `configs/daily_neural_sde_day14.json` へ書いた。test のあとでしきい値は動かしていない。

- **H1** (速度ドリフトと位置の運動学): 速度で symbolic 対 teacher の NRMSE \(\le 0.30\)、teacher 対真値 \(\le 0.40\)、symbolic 対真値 \(\le 0.50\)。位置の symbolic 対 teacher \(\le 0.20\)。四つとも必要。位置の teacher 対真値は運動学的恒等式なので併記のみ。残差 RMS は併記し、ゲートにはしない。
- **H2** (速度拡散 \(g_v\)): symbolic 対 teacher \(\le 0.35\)、teacher 対真値 \(\le 0.45\)、symbolic 対真値 \(\le 0.55\)、\(g_v^{2}\) の symbolic 対 teacher \(\le 0.50\)。位置拡散は構造的な 0 で、フィットしない。
- **H2c**: test query で symbolic \(g_v\le 0\) の行の割合 \(\le 0.02\)。
- **H3**: 係数回復の仮説は置かない。次数 3 の VIF 最大（定数列を除く）が 10 を超えたら不定。10 以下でも閉じた形としては読まない。
- **H4**: test の 64 本、同じ初期条件、同じ行動テープ、同じ Wiener 増分。失敗率は各モデル \(\le 0.05\) かつ symbolic と teacher の差 \(\le 0.02\)。失敗クラスは `non_finite` と `outside_radius`。失敗率ゲートと H7 を通り、共分散が定義できるときだけ、平均経路（symbolic 対 teacher \(\le 0.30\)、teacher 対真値 \(\le 0.40\)）、終端共分散の相対 Frobenius（symbolic 対 teacher \(\le 0.55\)、teacher 対真値 \(\le 0.70\)）、速度ドリフトの teacher 対真値 \(\le 0.40\) をすべて要求する。平均経路だけが通りドリフトが外れたら `path_match_without_field_recovery` で H4 は fail。失敗率ゲートか H7 が外れたら不定。
- **H5**: 閉じた行動ループの独立な集団。平均経路 RMSE は teacher \(\le 0.35\)、symbolic \(\le 0.45\)。終端共分散の相対 Frobenius は teacher \(\le 0.60\)、symbolic \(\le 0.75\)。失敗率 \(>0.05\)、または真値対真値のフロアが teacher のしきい値を超えたら不定。
- **H6**: 全モデルの ID の \(g\) NRMSE \(\le 0.45\)、平均 \(h\) の OOD マイナス ID \(\ge 0.05\)、全モデルの \(g\) 上昇が aleatoric-only の上昇を 0.05 以上上回る。両方の ID NRMSE が 0.80 以上なら共倒れ。その床の上の差は回復ではない。OOD は train の \((v, u)\) を保ち、\(q\) を train の \(q_{95}+(q_{95}-q_{5})\) へ移した点である。選択には使わない。
- **H7**: teacher の \(\Delta t=0.05\) と \(\Delta t=0.01\)（細増分 5 本の和を \(\sqrt{5}\) で割った粗増分、行動は粗い区間で固定）の終端 RMSE \(\le 0.35\)、かつ両方が有限な割合 \(\ge 0.95\)。

## 4. 方法

teacher への直接 query。特徴は非終端状態とその区間の行動 \((q, v, u)\)。
train 1920、validation 960、test 1920。落とした行は 0。
SINDy の標準化は train query だけから取る。ライブラリは \((q, v, u)\) の冪だけである。
次数 \(\{1,2,3\}\)、しきい値 \(\{0.02, 0.05, 0.1, 0.2, 0.35\}\)、ridge \(10^{-4}\)。
drift ベクトルと速度拡散を別々に、validation の pooled NRMSE 最小で選ぶ。同点はより小さい次数、次に小さいしきい値。
test はこの選択のあと 1 回。

選択結果: drift は次数 2・しきい値 0.02（標準化座標で活性 9、ライブラリ 10 項、検証 NRMSE 0.001840）。
次数 3・しきい値 0.02 は同じ検証 NRMSE で、同点規則が次数 2 を残した。活性項の集合も同じで、3 次の係数はしきい値 0.02 で落ちている。
diffusion は次数 1・しきい値 0.02（活性 2、ライブラリ 4 項、検証 NRMSE 0.006613）。次数 2 と 3 のしきい値 0.02 は同じ検証値である。

診断として、次数 3・しきい値 0 の通常最小二乗を teacher と真値の両方に train query で当て、test で 1 回評価した。
格子は train の 5–95 パーセンタイルを半分だけ広げた \(5^{3}\) 点で、選択には使わない。
シミュレーションは失敗をラッチし、`non_finite` と `outside_radius` を分ける。半径は 8。

## 5. 結果

### 5.1 仮説の成否一覧

| 仮説 | 指標 | 値 | しきい値 | 判定 |
|---|---|---|---|---|
| H1 速度 drift | symbolic 対 teacher / teacher 対真値 / symbolic 対真値 | 0.00204 / 0.0769 / 0.0764 | \(\le 0.30\) / \(\le 0.40\) / \(\le 0.50\) | **pass** |
| H1 位置 | symbolic 対 teacher | \(1.7\times 10^{-15}\) | \(\le 0.20\) | pass。teacher 対真値は 0 |
| H1 残差 RMS | 速度の三つ | 0.00152 / 0.0577 / 0.0573 | （併記） | teacher RMS 0.745、真値 RMS 0.751 |
| H1 天井 | 次数 3 OLS 対 teacher / oracle 対真値 | \(1.01\times 10^{-4}\) / \(3.4\times 10^{-15}\) | （診断） | oracle は機械精度。対真値の残差は teacher の場 |
| H2 \(g_v\) | symbolic 対 teacher / teacher 対真値 / symbolic 対真値 | 0.00618 / 0.0357 / 0.0355 | \(\le 0.35\) / \(\le 0.45\) / \(\le 0.55\) | **pass** |
| H2 残差 RMS | 上の三つ | 0.000923 / 0.00536 / 0.00533 | （併記） | teacher RMS 0.149、真値 RMS 0.150 |
| H2 \(g^{2}\) | symbolic 対 teacher | 0.0124 | \(\le 0.50\) | pass |
| H2 天井 | OLS 対 teacher / oracle 対真値 | \(3.9\times 10^{-6}\) / \(7.1\times 10^{-15}\) | （診断） | |
| H2c | symbolic \(g_v\le 0\) の行 | 0 | \(\le 0.02\) | pass |
| H3 VIF | 次数 3、定数列を除く | 7.55 | \(\le 10\) | pass（ゲート。係数回復ではない） |
| H4 失敗 | truth / teacher / symbolic | 0 / 0 / 0 | \(\le 0.05\)、差 \(\le 0.02\) | pass。クラスはどちらも 0 |
| H4 平均 | symbolic 対 teacher / teacher 対真値 | \(1.05\times 10^{-4}\) / 0.00500 | \(\le 0.30\) / \(\le 0.40\) | pass |
| H4 共分散 | symbolic 対 teacher / teacher 対真値 | 0.00121 / 0.0312 | \(\le 0.55\) / \(\le 0.70\) | pass |
| H4 ドリフト | 速度の teacher 対真値 | 0.0769 | \(\le 0.40\) | pass。`path_match_without_field_recovery` は false |
| H5 平均経路 | teacher / symbolic / 真値フロア | 0.0461 / 0.126 / 0.0490 | \(\le 0.35\) / \(\le 0.45\) | **pass**。teacher はフロア上。symbolic はフロアの上 |
| H5 共分散 | teacher / symbolic / 真値フロア | 0.317 / 0.293 / 0.340 | \(\le 0.60\) / \(\le 0.75\) | pass。三つともフロアの近傍 |
| H5 失敗 | truth / teacher / symbolic | 0 / 0 / 0 | \(\le 0.05\) | pass |
| H6 \(h\) | OOD の平均 \(h\) マイナス ID | \(3.6\times 10^{-7}\) | \(\ge 0.05\) | **fail** |
| H6 \(g\) 上昇 | 全モデル / aleatoric-only / 差 | 0.00483 / 0.00482 / \(6.8\times 10^{-6}\) | 差 \(\ge 0.05\) | fail |
| H6 ID | 全モデルの \(g\) NRMSE / aleatoric-only | 0.0357 / 0.0356 | 全モデル \(\le 0.45\) | この項は内側。共倒れではない（両方 0.80 未満） |
| H7 | 終端 RMSE / 有限な割合 | 0.00508 / 1 | \(\le 0.35\) / \(\ge 0.95\) | pass |

1 步の尺度は、test 上の teacher で平均 \(|f_v|\Delta t = 0.0313\)、平均 \(g\sqrt{\Delta t} = 0.0334\)、比 1.07 である。
選択後の train 窓 128 本では、データ NLL \(-2.920\)、重み付きの残差罰則 \(6.2\times 10^{-4}\)、強凸 \(6.9\times 10^{-3}\)、勾配罰則 \(6\times 10^{-8}\)、\(\mu\) 罰則 \(0.013\) である。
不確実性の罰則を足すと約 0.020 で、データ項の絶対値 2.92 の約 0.007 倍である。

### 5.2 軌道上の pass は閉じた形ではない

真値の速度ドリフトを成分の RMS に分けると、制御項 \(u\) が 0.714、線形減衰 \(-0.3 v\) が 0.094、立方項 \(-0.8 v^{3}\) が 0.070 である。
立方項はドリフト RMS 0.751 の 0.094 倍である。
teacher 対真値の残差 RMS 0.0577 は、この立方項と同じ大きさである。
分母が制御項で大きいので NRMSE は 0.0769 になり、ゲート 0.40 の内側に入る。

次数 3・しきい値 0 の oracle は真値を機械精度で回復する（\(v\) の係数 \(-0.3\)、\(u\) の係数 \(1\)、\(v^{3}\) の係数 \(-0.8\)、他は 0）。
同じ手続きで teacher を射影すると、\(v\) は \(-0.533\)、\(u\) は \(0.994\)、\(v^{2}\) は \(0.203\)、\(v^{3}\) は \(0.00692\)、ほかに切片 \(-0.0369\)、\(q\) の \(0.0533\)、\(qv\) の \(0.103\)、\(vu\) の \(-0.0905\) が出る。
teacher の \(v^{3}\) は真値 \(-0.8\) の約 100 分の 1 で、STLSQ のしきい値 0.02 はこれを落とす。
だから次数 2 と次数 3 の疎な候補は同じ検証誤差になる。

VIF は 7.55 でゲート 10 の内側である。
そのうえで、選ばれた式の \(v\) の係数は \(-0.528\) であり、真値の \(-0.3\) ではない。
\(v^{3}\) の項は推定式に無い。
格子（選択に未使用）の速度ドリフト RMS は、symbolic 対 teacher 0.0294、teacher 対真値 0.552、symbolic 対真値 0.539 である。
軌道上の残差 RMS 0.0577 の横に置くと、データ箱の外では立方項が残り、teacher も symbolic もそれを持たない。

### 5.3 距離依存の拡散は初期値のままである

ID の平均 \(h\) は 0.00842、OOD も 0.00842 で、差は \(3.6\times 10^{-7}\) である。
平均 \(\eta\) は ID で \(-2.5\times 10^{-4}\)、OOD で \(-2.2\times 10^{-4}\) である。
勾配罰則 \(L_{\mathrm{grad}}\) は選択後のスナップショットで \(6\times 10^{-8}\) なので、\(\eta\) とその勾配はデータ点で 0 に貼り付いている。
初期の \(b=-4\) のままなら \(h\approx 0.5\,\mathrm{sigmoid}(-4)\approx 0.0090\) で、測った 0.00842 はその近傍である。
強凸の生損失は 0.139 残っており、平坦な \(\eta\) は強凸の制約を満たしていない。
重み \(\lambda_{\mathrm{sc}}=0.05\) を掛けると 0.0069 で、データ NLL のスケールより二桁小さい。
最適化は \(\eta\) を曲げずに、この罰則を払っている。

OOD で平均 \(g\) が上がる量は、全モデル 0.00483、aleatoric-only 0.00482 である。
差 \(6.8\times 10^{-6}\) は \(h\) の移動ではない。
ID の対真値 NRMSE は全モデル 0.0357、aleatoric-only 0.0356 で、どちらも定数 \(0.15\) の近傍（teacher の \(g\) RMS 0.149）にいる。
共倒れの床 0.80 には乗っていない。H6 が落ちた理由は、二つのモデルが真値を外したことではなく、\(h\) が ID でも OOD でも動いていないことである。
この陰性は、凍結した \(\lambda_{\mathrm{grad}}=1\) と \(\lambda_{\mathrm{sc}}=0.05\) の切断での結果であり、MuJoCo 上の NUNO の不確実性図の再現ではない。

### 5.4 symbolic SDE（TeX、元座標）

推定した式は `outputs/daily_neural_sde/day14/symbolic_sde.tex` と一致する。
\(g_q=0\) はフィットではなく、位置拡散を固定した構造である。
係数は VIF ゲートの内側でも閉じた形ではない。比較用の真値を下に分ける。

推定:

$$
\begin{aligned}
\dot q &= v \\
\dot v &= -0.03634 + 0.9909\,u - 0.5276\,v + 0.05264\,q - 0.08952\,v u + 0.2038\,v^{2} + 0.1026\,q v \\
g_v &= 0.1491 + 0.00735\,u \\
g_q &= 0
\end{aligned}
$$

データの法則（推定ではない）:

$$
\begin{aligned}
dq &= v\,dt \\
dv &= \bigl(u - 0.3\,v - 0.8\,v^{3}\bigr)\,dt + 0.15\,dW \\
g_q &= 0
\end{aligned}
$$

\(u\) の係数 0.9909 と \(g_v\) の切片 0.1491 は、真値の 1 と 0.15 に近い。
\(v\) の \(-0.5276\)、\(v^{2}\) の \(0.2038\)、および \(q\)、\(vu\)、\(qv\)、切片、\(g_v\) の \(u\) は、真値に無い項か、真値と違う係数である。
\(v^{3}\) は推定式に現れない。

### 5.5 シミュレーションと法則指標

開ループ（保存した行動テープ、共通の Wiener 増分）では、失敗は 0 本である。
平均経路の RMSE は symbolic 対 teacher が \(1.05\times 10^{-4}\)、teacher 対真値が 0.00500。
終端共分散の相対 Frobenius は 0.00121 と 0.0312。
速度ドリフトの対真値 0.0769 もゲート内なので、H4 の pass は平均経路だけの pass ではない。
粗視化の終端 RMSE は 0.00508、有限な割合は 1 である。

閉じたループの独立な集団（各 96 本）では、失敗は 0 である。
平均経路 RMSE は teacher 0.0461、フロア 0.0490 で、teacher はフロア上にある。
symbolic は 0.126 で、絶対しきい値 0.45 の内側だがフロア 0.049 の上にある。
終端共分散は teacher 0.317、symbolic 0.293、フロア 0.340 で、三つとも同じ近傍にある。
H5 の pass は、symbolic の平均経路がフロアまで届いたことを意味しない。

## 6. 考察

この切断では、物理構造つきの速度ドリフトと aleatoric な拡散は、訪れた \((q, v, u)\) の上でデータ NLL を正規フロアまで下げた。
symbolic はその teacher を速度 NRMSE 0.002、拡散 NRMSE 0.006 で写した。
写した多項式は、制御ゲインと拡散の水準を保ったまま、立方摩擦をデータの箱の中の 2 次式に置き換え、線形減衰を \(-0.3\) から \(-0.53\) へ動かしている。
軌道上の NRMSE が 0.08 を切ることと、VIF が 7.55 であることは、この係数を法則の係数として読む根拠にならない。
格子の RMS 0.55 が、箱の外でその差が見える場所である。

距離依存の頭は、データ項がすでにフロアにいると動く必要がない。
凍結した勾配罰則は \(\eta\) を 0 に保ち、強凸の罰則はデータ項より二桁小さいので、\(\eta\) を曲げない。
その結果、位置を学習範囲の外へずらしても \(h\) は \(10^{-6}\) 以下しか変わらず、\(g\) のわずかな上昇は aleatoric ネットだけで説明できる。
NUNO の主張のうち、本日の切断が支持するのは「既知の運動学とニューラルな速度場で、覆われた領域の Euler 尤度を合わせられる」ことまでである。
「データから離れると拡散が頭打ちまで増える」ことは、この \(\lambda\) とこの定数拡散の法則の上では起きていない。

1 步の拡散増分とドリフト増分の比は 1.07 で、Day 13 の事後幅が 1 步拡散の約 4 倍だった状況とは別である。
本日の平均経路が真値に近いのは、ドリフトの対真値ゲートも同時に通っているからで、比が 1 付近であること自体が場の回復を意味するわけではない。
閉じたループでは、symbolic の平均経路がフロアを超える。開ループのテープ再生が 0.005 であることと、方策が状態に依存する集団平均が 0.126 であることは、別の指標である。

## 7. 反省（Day 15 以降に適用）

軌道上の NRMSE がゲート内でも、支配的な項（本日は制御項がドリフト RMS の大部分）が分母を持ち上げ、RMS が 1 割に満たない項（本日は \(v^{3}\) が 0.094）は係数として消える。
次数 3 の密な射影の係数を、疎な選択式の横に 1 回だけ出し、VIF がゲート内でもその係数を閉じた形として読まない。
格子の RMS を軌道上の残差 RMS の横に置く。
データ NLL が正規フロアに到達し、不確実性罰則の重み付き和がデータ項より二桁小さいときは、\(\eta\) の ID と OOD を同じ表に残し、\(g\) の OOD 上昇が aleatoric-only と揃っているなら距離依存の回復と読まない。
閉ループの平均経路が絶対しきい値を通っても、モンテカルロフロアをその横に残す。
