# Day 2 — Kidger et al. (2021) Neural SDEs as Infinite-Dimensional GANs: time-dependent Ornstein–Uhlenbeck process

- Date: 2026-10-27 (Day 2)
- Paper: Patrick Kidger, James Foster, Xuechen Li, Terry J. Lyons,
  *Neural SDEs as Infinite-Dimensional GANs*, ICML 2021 (PMLR 139:5453–5463),
  arXiv:2102.03657. 公式実装への言及: torchsde の neural SDE サンプルコード。
- 成果物: config `configs/daily_neural_sde_day02.json` / コード `src/daily_neural_sde/day02_kidger_ou.py`
  (共通部品 `src/daily_neural_sde/common.py` を再利用) / 数値 `outputs/daily_neural_sde/day02/results.json`、
  teacher 重み `teacher_weights.json`、symbolic モデル `symbolic_drift.json` / `symbolic_diffusion.json`、
  TeX `symbolic_sde.tex` / tests `tests/test_daily_neural_sde_day02.py` (4 tests 追加、Day 1 分と合わせて 9 tests 全て通過)。
- 実行: `python -m src.daily_neural_sde.day02_kidger_ou` (seed 0、PyTorch 2.14.1+cpu、約 11 分)。

## 0. Day 1 の反省の適用 (本日の分析に組み込んだもの)

Day 1 の反省 3 点をすべて適用した:

1. **推定器の離散化バイアス診断**: 4 種の推定器 (v1 EM / v2 局所線形化 / v3 行列指数再和約 / v4 内部 substep モーメント伝播)
   で $\hat g$ を比較し、さらに $t$ の三分位 (前期・中期・後期) ごとに $\hat g$ の平均を分けて報告した。
   Day 1 では非線形な成分で $\hat g$ が膨らんだが、本日は線形系なのでバイアスが $t$ によらず一様かを確かめる設計である。
2. **共線性診断の事前設定**: 係数回復の仮説 (H3) に共線性ゲートを事前に組み込んだ。
   train query 状態で $\mathrm{corr}(t, z) = 0.9611$、VIF $= 13.11 > 10$ だったため、
   config の事前ルールどおり H3 は field 水準の条項のみで判定し、係数は記述的に報告した。
3. **diffusion の絶対水準が主題の日は 1-step 尤度でなく内部 substep の推定を使う**:
   蒸留する teacher を事前に v4 (substep モーメント伝播) と宣言し、v1–v3 は比較用ベースラインとした
   (事後的な teacher の入れ替えはしていない)。

## 1. 論文要約

### 1.1 問題と提案

古典的な SDE モデリングは、事前に決めた統計量 (payoff 関数など) の期待値をデータに合わせることで
パラメータを較正する。本論文はこの手続きを Wasserstein GAN へ一般化する:
Brownian motion を入力ノイズとし、neural SDE を generator (出力は数値ソルバで生成される path)、
neural CDE (controlled differential equation) を discriminator として、
path 空間上の分布そのものを学習する。Wasserstein 距離は唯一の大域的最小値を持ち、
無限データ極限では任意の SDE が学習可能であることが主張される。
先行する neural SDE 研究 (Li et al. 2020 を含む) と異なり、事前指定の統計量も密度関数も使わない点が
本論文の新規性である。実装上の要点として、gradient penalty による Lipschitz 正則化、
最終段の tanh 非線形性、学習率・最適化の選択 (Adadelta が最良だったこと) が §4.5 で報告されている。

### 1.2 実験 (§4.1) と主張の範囲

4 つのデータセットのうち最初は、比較のために真値が既知の 1 次元合成例である。
時間依存 Ornstein–Uhlenbeck 過程

$$
\mathrm{d}z_t = (\mu t - \theta z_t) \,\mathrm{d}t + \sigma \circ \mathrm{d}W_t,
\qquad \mu = 0.02,\ \theta = 0.1,\ \sigma = 0.4
$$

から 8192 サンプルを $t = 0$ から $t = 63$ まで整数時刻で生成し、
$t = 6, 19, 32, 44, 57$ における周辺分布と 50 本のサンプル path を比較する (Fig. 3, 4)。
主張は「neural SDE は古典的な非 neural SDE を回復するに十分である」ことであり、
**drift / diffusion の関数としての回復精度は論文の主張ではない**点に注意する
(Day 1 の Li et al. と同じく、主張の軸は分布の一致である)。

## 2. データと Neural SDE の入手・再構成

### 2.1 データ

論文は生成データを公開していないため、§4.1 の設定どおり再生成した。
真値が affine なので、観測ステップごとに厳密遷移 (平均・分散はコードの `exact_transition_mean_var`、
テストで細かい Euler–Maruyama のモーメントと一致することを検証済み) でサンプリングした
(8192 系列 × 64 点、観測ノイズなし。論文は SDE 自体をサンプルしており、観測ノイズの記述はない)。
状態統計: 平均 4.63、標準偏差 3.44、範囲 [−3.9, 14.3]。
分割は trajectory-disjoint に train 5120 / validation 1536 / test 1536 (seed 0)。

### 2.2 逸脱の明示 (teacher の再構成方法)

論文の neural SDE は generator であり、neural CDE discriminator との Wasserstein GAN で学習されるが、
**その GAN (CDE discriminator、gradient penalty、double backward) の CPU 再現は本日の範囲外**であり、
また generator の hidden 座標は識別不能である。本シリーズの目的 (多項式 SINDy で近似し式として評価する)
のため、範囲を切り取り、**観測 1 次元座標で直接 Neural SDE を推定**した:
drift / diffusion は別々の MLP (Linear(2,32)–Softplus–Linear(32,32)–Softplus–Linear(32,1)、
入力は標準化した $(t, z)$、diffusion 出力は Softplus + 0.01)。
これは論文のモデルの再現ではなく、論文のデータ生成とモデル族 (drift・diffusion が $(t, x)$ の NN の SDE)
を引き継いだ再構成である。論文の主張 (周辺分布・path 分布の回復) は本日は再テストしない;
本日テストするのは、そのような neural SDE の field 水準の内容と多項式蒸留可能性という別の軸である。
その他の切り取り: 初期分布は論文本文に記述がないため $z_0 \sim \mathcal{N}(0, 1)$ と宣言して採用した;
論文の幅 64 に対し幅 32 に縮小した (CPU 切り取り); 論文の最終段 tanh は drift の値域を論文自身の
drift 値 (t が大きいとき $\mu t \approx 1.26$) より小さく cap してしまうため使わなかった;
論文の SDE は Stratonovich だが、尤度は Itô 規約で組んだ (真値の diffusion は定数なので一致する。
fit した teacher については Stratonovich 補正 $0.5 \hat g \hat g'$ の大きさを診断し、
平均絶対値 / $\hat g$ 平均 $= 0.00032$ と無視できることを確認した)。

### 2.3 推定器の比較 — 離散化バイアス診断 (Day 1 の反省 1・3)

観測間隔は $\Delta t = 1$ と大きい。4 種の Gaussian 遷移尤度を比較した (いずれも観測ノイズなし):

| 推定器 | 内容 | drift NRMSE vs 真値 (test) | diffusion NRMSE vs 真値 (test) | $\hat g$ 平均 (validation 状態) | $\hat g$ 平均 (t 三分位: 前期 / 中期 / 後期) |
|---|---|---|---|---|---|
| v1 `em` | 素朴な Euler–Maruyama 尤度 | 0.0294 | 0.0504 | 0.3800 | 0.3775 / 0.3800 / 0.3824 |
| v2 `local_linearization` | 平均に $\frac{1}{2}\Delta t^2 (J f)$、分散に $J g^2 \Delta t^2$ の補正 | 0.0636 | 0.0076 | 0.4020 | 0.3993 / 0.4023 / 0.4044 |
| v3 `local_linearization_exp` | 局所線形化を行列指数で再和約 (平均 $\Psi f$、分散 $\kappa g^2$) | 0.0615 | 0.0058 | 0.4005 | 0.3978 / 0.4007 / 0.4029 |
| v4 `substep_moments` (teacher) | 内部 8 substep のモーメント伝播 (平均は $f$ で、分散は $(2Jv + g^2)h$ で更新) | **0.0226** | **0.0036** | **0.3989** | 0.3983 / 0.3993 / 0.3991 |

読み取り:

- **EM のバイアスは理論値とぴったり一致する**。平均回帰系では区間内のノイズが減衰するため、
  真の遷移分散は $\sigma^2 (1 - e^{-2\theta\Delta t})/(2\theta) = 0.1450$ であり、
  EM 分散 $\sigma^2 \Delta t = 0.16$ より小さい。EM 尤度で fit した $\hat g$ は
  $\sqrt{0.1450} = 0.381$ になるはずで、実測 0.3800 と一致する。
  これは Day 1 の Lorenz 系 (非線形成分で $\hat g$ が膨らむ) とは逆向きのバイアスであり、
  **バイアスの向きは drift の Jacobian の符号構造で決まる**ことを示している。
  補正系 (v2/v3/v4) はすべて $\sigma = 0.4$ を 0.399–0.402 の精度で回復し、
  Day 1 で未解決だった「定数 diffusion の絶対水準の回復」は、**線形系では解決する**。
  すなわち Day 1 の失敗は 1-step 尤度一般の失敗ではなく、非線形性に起因するものであった
  (H5 の根拠)。
- **バイアスは $t$ によらずほぼ一様**である。$\theta$ が定数なので減衰バイアスは全期間で同じであり、
  三分位ごとの $\hat g$ は EM で 0.378–0.382、v4 で 0.398–0.399 と平坦である。
  Day 1 の「非線形成分だけが膨らむ」パターンとは対照的で、診断の設計どおりの対比が得られた。
- **drift の順序は読みすぎてはいけない**。v2/v3 の drift NRMSE (0.064/0.062) は EM (0.029) より悪いが、
  4 推定器の validation NLL は 0.45410–0.45415 と $10^{-5}$ の桁でしか違わず、
  このノイズ水準 ($\sigma = 0.4$ / step に対し drift 誤差は $\sim 0.02$) では尤度は drift の良し悪しに
  ほとんど感度がない。drift の順序は最適化のばらつきの範囲内とみるべきであり、
  系統的なのは diffusion の順序 (EM ≪ v2 ≈ v3 < v4 の精度) の方である。
- 蒸留対象の teacher は、事前に config で宣言したとおり v4 とした。

## 3. 仮説 (実行前に config へ凍結)

- **H1 (drift 蒸留)**: symbolic drift の test NRMSE (対 teacher) ≤ 0.10、かつ次数 1 の最良
  validation NRMSE が最良次数の 10% 以内 (真の drift が $(t, z)$ の次数 1 なので次数 1 で足りるはず)。
- **H2 (diffusion 蒸留)**: symbolic diffusion の test NRMSE (対 teacher) ≤ 0.20、かつ
  非定数項の寄与 (test query での RMS 比) ≤ 0.25。
- **H3 (真値回復)**: 係数条項は事前ルールにより VIF$(t, z) \le 10$ の場合にのみ適用する
  (符号一致、$\mu$・$\theta$ の相対誤差 ≤ 0.35、support 外寄与 ≤ 0.15)。
  VIF $> 10$ の場合は field 条項 (symbolic drift の test NRMSE 対真値 ≤ 0.10、support 外寄与 ≤ 0.15)
  のみで判定し、係数は記述的に報告する。
- **H4 (シミュレーション)**: symbolic SDE の失敗率 ≤ 0.05 かつ neural との差 ≤ 0.02、
  paired RMSE (対 neural) の中央値 ≤ 1.0。
- **H5 (推定器バイアス)**: v4 の $\hat g$ 平均の真値からの絶対誤差が v1 より小さく、
  かつ v4 の drift NRMSE (対真値) が v1 より小さい。

## 4. 方法 (main と同じ流儀)

- teacher への直接 query で function distillation。query 点は各 split の観測 $(t, z)$ 対
  (trajectory-disjoint)。変換 (mean/scale) と RMS 正規化は train query のみから。
- 候補: drift は次数 {1, 2, 3}、diffusion は次数 {0, 1, 2}、STLSQ しきい値 {0.02, 0.05, 0.1, 0.2, 0.35}。
  field ごとに validation NRMSE で選択し、test は確定後に 1 回だけ評価。
- oracle 診断: 同じ SINDy 手順を真値 field に fit (候補ではなく天井の診断)。
- 共線性診断: fit の前に train query 状態で $\mathrm{corr}(t, z)$、VIF、次数 2 特徴量間の最大相関を計算。
- off-manifold 診断 (補足のみ、選択には使わない): $t \in \{0, 16, 32, 48, 63\} \times z \in [-2, 12]$ の
  独立格子 75 点で symbolic / teacher / 真値を比較。
- シミュレーション検証: test 系列の $t=0$ 状態 1536 点から、neural / symbolic / 真値の
  3 モデルを同一 Brownian 増分 (paired) で Euler–Maruyama (dt = 0.05, $t \in [0,63]$)。
  失敗は non-finite / $|z| > 50$ で分類し、質量を保持する。論文の Fig. 3 に対応する
  $t = 6, 19, 32, 44, 57$ の周辺平均・標準偏差も報告する。

## 5. 結果

### 5.1 蒸留の精度

| field | 選択 (次数, しきい値, 活性項) | validation NRMSE | test NRMSE 対 teacher | test NRMSE 対真値 |
|---|---|---|---|---|
| drift | 次数 2, 0.02, 6 項 | 0.00167 | **0.00165** | 0.0226 |
| diffusion | 次数 0, 0.02, 1 項 | 0.00228 | **0.00231** | 0.00279 |

次数別の test NRMSE (対 teacher、各次数の validation 最良モデル): drift は次数 1 で 0.0223、
次数 2 で 0.00165、次数 3 で 0.00165。diffusion は次数 0・1・2 すべて 0.00231
(次数 1・2 でも定数項のみが選択された)。
oracle (真値への fit) の天井は drift test NRMSE $\approx 10^{-15}$ (真値が次数 1 の多項式なので厳密に回復)、
diffusion は次数 0 で厳密に 0。つまり **SINDy 蒸留そのものの誤差 (対 teacher 0.0017) は
oracle 天井に近く、symbolic 側はほぼ最適**で、対真値の誤差は teacher 側の誤差が支配的である
(Day 1 と同じ構造だが、誤差水準は 1 桁以上小さい)。

### 5.2 求めた symbolic SDE (TeX、元座標)

蒸留で得た SDE は次である ($W$ は Wiener 過程、$\hat f$ / $\hat g$ は
`outputs/daily_neural_sde/day02/symbolic_sde.tex` の生成 TeX と同一):

$$
\mathrm{d}z_t = \hat f(t, z_t) \,\mathrm{d}t + \hat g(t, z_t) \,\mathrm{d}W_t
$$

```tex
\begin{aligned}
\hat f &= 0.0009766 - 0.09183\,z + 0.01855\,t + 0.003316\,z^{2} - 0.001227\,t z + 0.0001125\,t^{2} \\
\end{aligned}
\begin{aligned}
\hat g &= 0.3989 \\
\end{aligned}
```

支配項だけ抜き出すと $\hat f \approx 0.01855\,t - 0.09183\,z$、$\hat g \approx 0.3989$ とほぼ定数である。
2 次項は係数が小さいが、$t \sim 30$、$z \sim 5$ では $0.003316\,z^2 \approx 0.083$、
$-0.001227\,tz \approx -0.184$、$0.0001125\,t^2 \approx 0.101$ と互いにほぼ相殺し、
多様体上では小さな補正にしかならない。これが次数 2 が次数 1 より 13 倍良く teacher を写す理由であり、
同時に off-manifold では崩れる理由でもある (§5.3)。

### 5.3 真値との係数比較と共線性 (H3 の内訳)

train query 状態での共線性診断: $\mathrm{corr}(t, z) = 0.9611$、VIF $= 13.11$、
次数 2 特徴量間の最大絶対相関 $= 0.9897$。$t$ と $z$ が強く相関するのは構造的である:
ramp 強制 $\mu t$ が OU 平均を時間とともに押し上げるため、trajectory 上では $z$ は $t$ のほぼ関数になる。
事前ルールにより、H3 の係数条項は適用せず、係数は記述的に報告する:

| 単項式 | 真値 | fit | 相対誤差 | 符号 |
|---|---|---|---|---|
| $t$ ($\mu$) | +0.02 | +0.01855 | 0.073 | ✓ |
| $z$ ($-\theta$) | −0.10 | −0.09183 | 0.082 | ✓ |

support 外項の寄与は RMS 比 0.0582 (しきい値 0.15 以内)。
VIF $> 10$ にもかかわらず係数が 7–8% の精度で回復したのは、query 数が膨大
(train query 約 327,680 点) で共線性による分散増大を相殺したためと考えられる。
ただしこれは **on-manifold での話**である。off-manifold 格子 (75 点) では:

| 比較 | drift NRMSE |
|---|---|
| symbolic vs teacher | 0.0655 (on-manifold では 0.00165) |
| symbolic vs 真値 | 0.2843 (on-manifold では 0.0226) |
| teacher vs 真値 | 0.2201 (on-manifold では 0.0226) |

teacher 自体が trajectory 多様体の外では真値から大きくずれており (0.220)、
symbolic はさらにずれる (0.284)。つまり本日の蒸留は **多様体上の近似**であり、
係数の on-manifold での見かけの良さは off-manifold での妥当性を保証しない。
field 水準・係数水準・off-manifold 水準は 3 つの別々の評価軸である。

### 5.4 シミュレーション検証 (paired、1536 系列、$t \in [0,63]$)

| モデル | 失敗率 | 終端平均 (finisher) | 終端 std (finisher) |
|---|---|---|---|
| neural (teacher v4) | 0.000 | 10.625 | 0.885 |
| symbolic | 0.000 | 10.624 | 0.885 |
| 真値 | 0.000 | 10.635 | 0.882 |

paired RMSE 中央値: symbolic 対 neural **0.00194**、neural 対真値 0.0177、symbolic 対真値 0.0180。
平均回帰系は縮約的なので、on-manifold の微小な field 誤差は horizon 全体で増幅せず、
むしろ減衰する。論文の Fig. 3 に対応する周辺分布も 3 モデルでほぼ一致する:

| $t$ | neural 平均 / std | symbolic 平均 / std | 真値 平均 / std |
|---|---|---|---|
| 6 | 0.311 / 0.961 | 0.313 / 0.963 | 0.307 / 0.947 |
| 19 | 2.112 / 0.912 | 2.111 / 0.912 | 2.115 / 0.915 |
| 32 | 4.491 / 0.889 | 4.490 / 0.889 | 4.493 / 0.896 |
| 44 | 6.831 / 0.908 | 6.832 / 0.908 | 6.836 / 0.913 |
| 57 | 9.397 / 0.885 | 9.399 / 0.885 | 9.408 / 0.886 |

### 5.5 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1 drift 蒸留 | **否** | test 条項は満たす (0.00165 ≤ 0.10) が、簡潔性条項が失敗: 次数 1 の validation 0.0224 は次数 2 の 0.00167 の 10% 以内ではない (13 倍悪化)。teacher は滑らかな非多項式で、その微細構造は次数 2 で拾われる — Day 1 の H1 と同じ失敗パターン |
| H2 diffusion 蒸留 | **成** | test 0.00231 ≤ 0.20、非定数寄与 0.0 ≤ 0.25。ただしこれは teacher の写像としての合格で、teacher 自体の diffusion が真値に近いことは §2.3 の別問題である |
| H3 真値回復 | **成** (field 条項のみ) | field 条項: 対真値 drift NRMSE 0.0226 ≤ 0.10、support 外寄与 0.0582 ≤ 0.15。係数条項は VIF $> 10$ のため事前ルールにより不適用 (記述値: $\mu$ 誤差 7.3%、$\theta$ 誤差 8.2%、符号一致)。off-manifold での劣化 (§5.3) はこの判定の範囲外であり、別途明記した |
| H4 シミュレーション | **成** | 失敗率 0 (差 0)、paired RMSE 0.00194 ≤ 1.0 |
| H5 推定器バイアス | **成** | $\|\hat g_{v4} - 0.4\| = 0.0011 < \|\hat g_{v1} - 0.4\| = 0.0200$、かつ drift NRMSE v4 0.0226 < v1 0.0294 |

## 6. 考察

1. **Day 1 の diffusion 失敗は非線形性に起因する**ことが、線形系での対照実験により強まった。
   EM のバイアスは減衰因子 $\sqrt{(1 - e^{-2\theta\Delta t})/(2\theta\Delta t)} = 0.952$ で
   ぴったり説明でき、補正系はすべて $\sigma$ を 1% 以内で回復した。
   1-step 尤度を一律に疑うのではなく、「Jacobian の符号と非線形性がバイアスの向きと大きさを決める」
   という診断の枠組みが得られた。
2. **共線性は on-manifold では隠れ、off-manifold で顕在化する**。
   VIF 13.1 という警告にもかかわらず on-manifold の係数は良く見えたが、
   格子診断では teacher・symbolic ともに真値から大きくずれた。
   Day 1 の教訓 (field 水準と係数水準を分ける) に「off-manifold 水準」を加えるべきであり、
   時間依存系では $t$ と $z$ の共線性が構造的に生じるため、格子診断は標準装備に値する。
3. **蒸留の成功と論文の主張は別の軸**である。Kidger et al. の主張は path 空間上の分布の一致であり、
   本日のシミュレーション周辺分布の一致 (§5.4) はその方向と整合するが、
   本日の field 蒸留の成功 (対 teacher 0.0017) は論文の主張の検証ではない。
   逆に、論文の GAN が作る generator の field を本日は測っていない (尤度で置換した) ことも明記する。
4. **NLL の感度不足**は推定器比較の罠である。validation NLL が $10^{-5}$ の桁でしか違わないのに
   drift NRMSE が 2 倍違うことがありうる。推定器の drift 比較は field NRMSE で行い、
   NLL の順序を過大評価しないこと。

## 7. 反省 (Day 3 以降に適用)

- 時間依存系では $t$ と状態の共線性が構造的に生じるため、off-manifold 格子診断を標準手順にする
  (on-manifold の指標だけでは多様体局所性を検出できない)。
- 尤度 (NLL) の差が $10^{-5}$ 程度と小さいとき、推定器の drift 順序を系統的な優劣と読まない。
  diffusion の順序と field NRMSE を主軸にし、drift の順序は最適化のばらつきの可能性を併記する。
- 係数が on-manifold で良く見えても、VIF が高い日は off-manifold での係数解釈をしない。
  「field 水準・係数水準・off-manifold 水準」の 3 軸で分けて報告する。

## 8. 限界

- 本日の teacher は論文の GAN generator そのものではない (§2.2 の逸脱)。論文の CDE discriminator、
  Wasserstein 学習、gradient penalty、実データ (株価・大気質) の実験は本日は扱っていない。
- データは再生成であり、論文の乱数系列そのものではない。初期分布は論文本文に記述がなく $z_0 \sim \mathcal{N}(0, 1)$ と宣言して採用した。
- teacher の Itô 規約は Stratonovich との差が無視できることを診断で確認したが、状態依存 diffusion が
  大きい系ではこの置換は正当化されない。
- シミュレーション horizon は論文と同じ $t \in [0,63]$ のみ。長期の定常分布への収束は未検証
  (ramp 強制のため定常分布そのものが存在しない系である)。
- 既存 repo tests (`tests/test_distill.py` 等) は本環境 (environment.yml の完全環境ではない)
  では未実行。本日追加した `tests/test_daily_neural_sde_day02.py` の 4 tests と Day 1 の 5 tests は全て通過。
