# Day 10 — Song et al. (2021) Score-Based Generative Modeling through SDEs: Reverse-Time SDE

- Date: 2026-11-04 (Day 10)
- Paper: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
  Stefano Ermon, Ben Poole,
  *Score-Based Generative Modeling through Stochastic Differential Equations*,
  ICLR 2021 (Oral), arXiv:2011.13456.
- 成果物: config `configs/daily_neural_sde_day10.json` / コード `src/daily_neural_sde/day10_song_score.py`
  (共通部品 `src/daily_neural_sde/common.py` を再利用) / 数値 `outputs/daily_neural_sde/day10/results.json`、
  teacher 重み `teacher_weights.json`、symbolic モデル `symbolic_f.json` / `symbolic_g.json`、
  TeX `symbolic_sde.tex`、展開係数 `symbolic_expanded.json`、
  tests `tests/test_daily_neural_sde_day10.py` (7 tests 追加、Day 1–10 分と合わせて 49 tests 全て通過;
  `tests/test_data.py` / `tests/test_distill.py` は本環境の venv に pysindy が無いための既存の収集エラーで、
  Day 10 とは無関係)。
- 実行: `~/.venv-day04/bin/python -m src.daily_neural_sde.day10_song_score`
  (seed 0、PyTorch 2.15.0+cpu、約 29 秒 + テスト)。

## 0. 前日までの反省の適用 (本日の分析に組み込んだもの)

1. **推定器の離散化バイアス診断 (Day 1・Day 6・Day 7)**: 本日の推定器は DSM (EM ではない) のため、
   離散化軸は teacher 訓練の刻みではなく reverse ソルバーの刻み (dt = 0.002 vs 0.05) とした (H6)。
   粗刻みの増分は細刻み増分の 25 ステップブロック和とすることで共通乱数による完全な対比較とし、
   レジーム分割 (近線形 τ < 0.3 vs 非線形 τ > 0.7) とレジーム別 per-step SNR を併記した。
   線形/非線形比較のアナロジーとして、teacher drift 誤差のレジーム分割も報告した。
2. **共線性ゲートを最大次数で (Day 1・Day 7・Day 8)**: fit の前に train query 状態で次数 3 フルライブラリ
   (選択手順が選びうる最大次数) の VIF を計算し、定数列を除外した。結果は最大 7.14 でゲート (< 10) を通過した。
   ただし真の reverse drift は有理型の mixture-score 関数であり多項式ではないため、
   係数回復の仮説は凍結せず、ゲートは「係数読みを阻む共線性が無いこと」の確認に留めた (Day 1 の field/係数分離)。
3. **off-manifold 格子診断の標準化 (Day 2・Day 9)**: (x, τ) 平面の 85 点格子 (x ∈ [−4, 4]、τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95})
   で f・g を比較し、選択には使わず (Day 5)、symbolic vs teacher・teacher vs 真値・symbolic vs 真値の
   3 者比較として報告した (§5.6)。格子の不一致の帰属を teacher の野生さと symbolic 近似に分離した (Day 9)。
4. **ドメイン妥当性を失敗率と分離 (Day 4)**: H4 には失敗率とは別に全ステップでの |X| > 8 逸脱率を事前に含めた。
5. **分布損失と field の分離 (Day 5)**: 分布指標 (W1・モード質量) と field 指標 (drift/g NRMSE) を同じ表に並べ、
   どちらか一方で他方を代用しない。teacher の DSM 損失 (0.244) も field 誤差と併記し、
   小さい訓練損失を field 回復と読まない。初期条件は 2000 個の prior  draws (単一初期条件ではない)。
6. **downstream 指標の paired SE (Day 4)**: H7 には paired SE と事前の「統計的に確定しない」条項を置いた。
7. **シフト条件での絶対水準を主条項に (Day 7)**: H5 は劣化の差分ではなく teacher・symbolic・no-score の
   W1 とモード質量の絶対水準として凍結した。
8. **diffusion の入力設計の分離検証 (Day 6)**: g は独自のフル (x, τ) ライブラリと verdict を持ち、
   さらに論文の diffusion が τ のみの既知スケジュールであることを利用して、
   symbolic g の x 依存性を H2c として別に検証した。
9. **grid NRMSE の分母効果 (Day 5)**: 格子 NRMSE > 1 の値には絶対 RMS 誤差を併記した。

## 1. 論文要約

### 1.1 問題と提案

データ分布を既知の prior へ滑らかに変換する forward SDE
dx = f(x, t)dt + g(t)dW と、摂動分布の時間依存スコア ∇log p_t(x) のみで決まる
reverse-time SDE (Anderson 1982)

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{W}_t
$$

により、スコアをニューラルネットで推定して prior からデータを生成する枠組みが提案された
(論文 §3)。denoising score matching によるスコア学習、VE-SDE / VP-SDE の具体化、
predictor-corrector サンプラー、同じ周辺分布を持つ等価な probability-flow ODE
(§4、尤度の厳密計算が可能) が導出される。主張は、この枠組みが NCSN・DDPM を包含し、
CIFAR-10 で当時の最高性能 (IS 9.89、FID 2.20) を達成したことである。
本日扱うのは VP-SDE と reverse SDE の構造そのものであり、画像ベンチマークではない。

### 1.2 本日の切り取りと選定理由 (多様性の観点)

前日までとの重なり回避: Day 1 は勾配ベース学習 (Li)、Day 2 は GAN (Kidger)、Day 3 は変分潜在モデル (Tzen)、
Day 4 は市場較正 (Gierjatowicz)、Day 5 は W₂ 分布一致 (Zhang)、Day 6 は遅延 SDDE (Eggen & Midtfjord)、
Day 7 は構造制約による安定化 (Oh et al.)、Day 8 はジャンプ拡散 (Jia & Benson)、
Day 9 は平均場 (Yang et al.) であった。いずれも drift・diffusion が「現在の状態 (と時刻・履歴・ジャンプ・平均)」
の関数として直接パラメトライズされるモデルである。
本日は **drift が学習されたスコア場そのものを含む reverse-time SDE** という未使用の手法系、
**スコアマッチングによる生成モデル**という未使用のドメイン、
**時間依存の飽和型 (有理型) drift と時間のみの diffusion**という未使用の field 構造を選んだ。
次元は 1 次元で Day 2・Day 8・Day 9 と重なるが、問いは「飽和型スコア drift を多項式が近似できるか」と
「スコア項を落とすと何が壊れるか」であり、構造はすべて新しい。

## 2. データと Neural SDE の入手・再構成

### 2.1 データと真値

目標分布は 1 次元対称 2 峰混合 p0 = 0.5 N(−2, 0.3²) + 0.5 N(2, 0.3²) である。
論文の VP-SDE (β(t) = 0.1 + 19.9t、t ∈ [0, 1]) のもとで、摂動核は

$$
p_t(x \mid x_0) = \mathcal{N}(\alpha(t) x_0,\, 1 - \alpha(t)^2), \qquad
\alpha(t) = \exp\!\big(-\tfrac12(0.1t + 9.95t^2)\big)
$$

と解析的であり、摂動混合分布のスコアも解析的である
(平均 ±2α(t)、分散 1 − 0.91α(t)²)。reverse time τ = 1 − t での真の reverse SDE は

$$
dX_\tau = \big[\tfrac12\beta(s) X_\tau + \beta(s)\,\nabla\log p_s(X_\tau)\big]\,d\tau
+ \sqrt{\beta(s)}\,d\bar{W}_\tau, \qquad s = 1 - \tau.
$$

データ標本は 9000 個 (train 5000 / validation 1500 / test 2500、標本-disjoint)。
蒸留 query は各標本から 8 回の (t, z) 抽出による forward 核標本 (x_t, τ = 1 − t)、
t ∈ [0.02, 1] 一様である。train query の x 範囲は [−4.47, 4.46]、τ 範囲は [0, 0.98] である。
teacher は ε-ネットワーク (MLP [64, 64] tanh、入力 (x/2.5, t)) を denoising score matching
(損失 ||ε_θ(x_t, t) − z||²、Adam lr 1e-3、4000 ステップ、バッチ 512、t ∈ [0.01, 1]) で訓練し、
スコアを s_θ(x, t) = −ε_θ(x, t)/√(1 − α(t)²) として reverse drift に代入した。
teacher の diffusion は論文の既知スケジュール √(β(s)) そのものである (学習しない)。
DSM 損失 (最終 200 ステップ平均) は 0.244 であった。

### 2.2 逸脱の明示

1. **画像から 1 次元混合への置換**: 論文の CIFAR-10 / 1024×1024 画像ベンチマークは扱わない。
   1 次元混合では摂動スコアが解析的になるため、teacher-vs-真値の field 比較が厳密である。
   本日の結果は論文の画像性能について何も主張しない。
2. **U-Net から小 MLP への置換**: 論文の画像向けスコアアーキテクチャではなく (x, t) 入力の小 MLP を用いた。
   predictor-corrector サンプラーは使わず、素の Euler–Maruyama reverse サンプリングとしたため、
   離散化効果は補正されずに可視化される (H6 の意図)。
3. **VP-SDE のみ**: VE-SDE・sub-VP-SDE・条件付き生成は対象外。probability-flow ODE は診断のみで、
   フィット対象ではない。
4. **diffusion は既知**: 論文と同様、diffusion スケジュールは学習対象ではなく既知関数である。
   したがって g の対真値指標は対 teacher 指標と同一であり、g の蒸留の問いは「τ のみ関数の回復」と
   「x 依存の混入の有無」に帰着する。

## 3. 仮説 (config に事前凍結)

- **H1** (drift 蒸留): symbolic drift の test NRMSE vs teacher ≤ 0.30。
- **H2** (diffusion 蒸留): symbolic g の test NRMSE vs teacher ≤ 0.30、Σ = g² も ≤ 0.30。
  **H2c** (構造): τ ∈ {0.25, 0.5, 0.75} での symbolic g の x 方向相対散らばり (std/mean、x ∈ [−4, 4]) ≤ 0.10。
- **H3** (真値回復、field 水準のみ): symbolic drift の test NRMSE vs 真値 ≤ 0.60、g は ≤ 0.30。
  係数仮説は凍結しない (真の drift は多項式ではないため)。共線性ゲートは次数 3 フルライブラリで
  VIF 最大 ≤ 10 を確認する (定数列を除外)。
- **H4** (シミュレーション): symbolic 失敗率 ≤ 0.05、teacher との差 ≤ 0.02、|X| > 8 逸脱率の差 ≤ 0.10、
  paired 最終状態 RMSE ≤ 1.0 (副次)、W1: teacher ≤ 0.35、symbolic ≤ 0.50、
  モード質量 (|X| > 1 の割合) の真値との差: teacher・symbolic とも ≤ 0.15。
- **H5** (論文軸): no-score ベースライン (スコア項を落とした OU 反転) の W1 ≥ 0.80 かつ
  モード質量 ≤ 0.60。teacher・symbolic の W1・モード質量は H4 と同じ絶対水準で判定する。
  probability-flow ODE (teacher スコア、決定論的) は診断のみ。
- **H6** (離散化バイアス): teacher reverse SDE の粗刻み (dt = 0.05) の W1 が細刻み (dt = 0.002) より
  0.03 以上悪化すること。レジームギャップ (非線形 τ > 0.7 の teacher drift 誤差 − 近線形 τ < 0.3 の誤差) ≥ 0.05。
- **H7** (downstream one-step): test query からの one-step reverse 予測で |MSE_symbolic − MSE_teacher| ≤ 0.002。
  paired SE ≥ 0.001 なら「統計的に確定しない」とする。

## 4. 方法

teacher への直接 query、標本-disjoint な train/validation/test 分割、変換は train のみ、
選択は validation のみ、test は確定後 1 回。drift・diffusion ともに次数 1–3 のフル (x, τ) ライブラリと
しきい値 {0.02, 0.05, 0.1, 0.2, 0.35} から validation NRMSE 最小で選択した (ridge 1e-4)。
選択結果: drift は次数 3・しきい値 0.02 (標準化座標で 4 活性項、展開後 8 項)、
g は次数 3・しきい値 0.02 (標準化座標で 4 活性項、展開後は τ のみの 4 項)。
oracle 診断として、同じ手順を真値 field 値にフィットしたモデルも作った (選択には使わない)。
reverse サンプリングは prior N(0, 1) からの 2000 標本、dt = 0.005 (200 ステップ)、
モデル間で Brownian 増分を共有し、失敗質量を保持した。
W1 は 1 次元ソート標本の平均絶対差、真値標本は混合分布からの直接抽出 (2000 標本) である。
真値 SDE 自身の reverse サンプリングの W1 (0.028) がモンテカルロフロアの目安になる。

## 5. 結果

### 5.1 仮説の成否一覧

| 仮説 | 指標 | 値 | しきい値 | 判定 |
|---|---|---|---|---|
| H1 drift 蒸留 | symbolic drift test NRMSE vs teacher | 0.396 | ≤ 0.30 | **FAIL** |
| H2 diffusion 蒸留 | symbolic g test NRMSE vs teacher | 0.0084 | ≤ 0.30 | pass |
| H2 Σ | symbolic Σ test NRMSE vs teacher | 0.0104 | ≤ 0.30 | pass |
| H2c 構造 | symbolic g の x 方向相対散らばり最大 | ~0 (1.4e-16) | ≤ 0.10 | pass |
| H3 真値回復 | symbolic drift test NRMSE vs 真値 | 0.390 | ≤ 0.60 | pass |
| H3 真値回復 | symbolic g test NRMSE vs 真値 | 0.0084 | ≤ 0.30 | pass |
| H3 ゲート | 次数 3 フルライブラリ VIF 最大 | 7.14 | ≤ 10 | pass |
| H4 失敗 | symbolic 失敗率 / teacher との差 | 0.0 / 0.0 | ≤ 0.05 / ≤ 0.02 | pass |
| H4 逸脱 | symbolic \|X\| > 8 逸脱率 / teacher との差 | 0.0 / 0.0 | 差 ≤ 0.10 | pass |
| H4 paired | paired 最終状態 RMSE vs teacher | 0.991 | ≤ 1.0 | pass (僅差) |
| H4 W1 | teacher / symbolic の W1 vs 真値標本 | 0.087 / 0.467 | ≤ 0.35 / ≤ 0.50 | pass (symbolic は僅差) |
| H4 モード質量 | teacher / symbolic の \|X\| > 1 割合 (真値 1.000) | 0.9975 / 0.770 | 差 ≤ 0.15 | teacher pass / **symbolic FAIL (差 0.230)** |
| H5 no-score | no-score W1 (生存標本のみ) / モード質量 | 15.22 / 0.966 | ≥ 0.80 / ≤ 0.60 | W1 pass / **モード質量 FAIL (生存者バイアス、§5.5 参照)** |
| H5 PF-ODE | teacher スコア PF-ODE の W1 / モード質量 | 0.181 / 0.9895 | (診断のみ) | — |
| H6 離散化 | W1 悪化 (粗刻み − 細刻み) | 0.0274 | ≥ 0.03 | **FAIL (僅差)** |
| H6 レジーム | teacher drift 誤差: 近線形 0.0687 / 非線形 0.1346、ギャップ | 0.0659 | ≥ 0.05 | pass |
| H7 one-step | \|MSE_symbolic − MSE_teacher\| (paired SE) | 0.000091 (0.000216) | ≤ 0.002、SE < 0.001 | pass (統計的に確定) |

### 5.2 teacher の精度

teacher drift の test NRMSE vs 真値は 0.0944 と良好である。DSM 損失 0.244 と field 誤差 0.094 は
別物であり、損失の小ささだけでは field 回復は保証されないが、本日は両者とも良好であった
(Day 5 の分離の教訓に照らして併記する)。
レジーム別では近線形 (τ < 0.3、prior 的でスコア ≈ −x) で 0.0687、非線形 (τ > 0.7、鋭い 2 峰スコア) で
0.1346 と、Day 1 の線形/非線形アナロジー通り非線形レジームで誤差が大きい。
per-step SNR (|f|dt/(g√dt)、dt = 0.01) は近線形 0.166、非線形 0.228 で、レジーム間の SNR 差は小さい。
したがってレジームギャップは SNR ではなくスコアの非線形性そのものによる。

### 5.3 H1 の失敗はライブラリの限界である (oracle 天井)

H1 は 0.396 で失敗したが、真値 field 値に同じ手順をフィットした oracle の test NRMSE vs 真値も
0.389 であり、symbolic の対真値誤差 0.390 とほぼ一致する。すなわち次数 3 多項式は、
teacher の野生さのせいではなく、**飽和型スコア drift をそもそも近似できない**。
on-manifold でも 0.39 という天井があることが、H1 失敗の主因である。
これは Day 1 の「field 水準と係数水準を分ける」教訓の拡張である:
H1 の失敗を teacher の失敗と読んではならず、oracle 天井を併記して初めて帰属が決まる。

### 5.4 symbolic SDE (TeX、元座標)

$$
\begin{aligned}
f_{\mathrm{rev}}(x, \tau) &= 0.05571 - 0.1287\,\tau - 10.6\,x + 0.05531\,\tau^{2} + 24.49\,x \tau + 0.007372\,x^{2} - 10.53\,x \tau^{2} - 0.4677\,x^{3} \\
\end{aligned}
$$

$$
\begin{aligned}
g_{\mathrm{rev}}(x, \tau) &= 4.517 - 2.99\,\tau + 2.162\,\tau^{2} - 2.969\,\tau^{3} \\
\end{aligned}
$$

with $dX_\tau = f_{\mathrm{rev}}(x, \tau)\,d\tau + g_{\mathrm{rev}}(x, \tau)\,d\bar{W}_\tau$.

読み: drift の主構造は x の線形項と xτ・xτ² の交差項であり、τ = 0 では −10.6x (真値の prior 付近 −10x と一致)、
τ = 1 では 3.36x − 0.468x³ + 定数項となり、±2.68 付近に安定点を持つ形になる。
しかし真値のスコアはモード付近で飽和する有理型であり、3 次多項式の谷間の反発 (3.36x) は弱く、
diffusion も τ = 1 で 0.316 と小さいため、谷間に残る標本が 23% 生じた (モード質量 0.77)。
g の symbolic は展開後に x を一切含まず、τ のみの 4 項多項式として真のスケジュール
(√β(s): τ = 0 で 4.472、τ = 1 で 0.316) を NRMSE 0.0084 で再現した。
τ = 1 端での symbolic g は 0.72 と真値 0.316 を上回るが、test NRMSE は大きな g の領域に支配されるため
全体としては小さい。完全な展開係数は `symbolic_expanded.json` にある。

### 5.5 H5: 爆発するベースラインの生存者バイアス

no-score ベースライン (スコア項なしの OU 反転) は失敗率 0.8965 で爆発した:
drift 0.5β(s)x は τ が小さい領域で +10x という反発 drift となり、スコアの修正なしでは
prior からの reverse 軌道が発散する。したがって W1 = 15.22 とモード質量 0.966 は
**生存した 10.35% の標本のみ**で計算された値であり、凍結したモード質量条項 (≤ 0.60) の
「失敗」は、ベースラインが unimodal であることを示すのではなく、生存者が外側に吹き飛んだ
標本ばかりであることを示している。実質的な分離の証拠は失敗率そのものである:
スコア項なしでは reverse 生成は成立しない。事後的にしきい値を変えることはしない
(憲章: test を見た後の事後チューニングをしない) が、爆発するベースラインに対する分布条項は
失敗率でゲートすべきであったことが本日の反省である。
teacher の W1 0.087・モード質量 0.9975 は良好で、PF-ODE (診断) も W1 0.181・モード質量 0.9895 と良好であった
(ただし PF-ODE の標準偏差 2.52 は真値 2.03 を上回り、決定論的写像の裾の厚さを示す)。

### 5.6 格子診断 (報告専用): 不一致は symbolic 側

(x, τ) 格子 (85 点) の 3 者比較: symbolic vs teacher drift NRMSE 0.803 (絶対 RMS 10.55)、
teacher vs 真値 0.180 (RMS 2.69)、symbolic vs 真値 0.555。
τ スライス別では symbolic vs teacher は τ = 0.5 で 0.919 と最悪、τ = 0.95 で 0.548 と最良であり、
teacher vs 真値は全スライスで 0.14–0.20 と安定している。
Day 9 の教訓通り、格子の不一致は主に teacher の off-manifold での野生さではなく
**symbolic 多項式の off-manifold での増幅**によるものである
(格子は x ∈ [−4, 4] で query 範囲と重なるが、τ 方向の多項式が teacher の飽和を追えない)。
g の格子は symbolic vs teacher 0.007、teacher vs 真値 0.0 (既知スケジュール) と良好である。

### 5.7 H6・H7

H6 の W1 悪化は 0.0274 で凍結しきい値 0.03 に僅差で未達 (FAIL) であった。
真値 SDE 自身の W1 フロアが 0.028 であることと比べると、0.0274 という差はサンプリングノイズの
フロアと同程度であり、W1 の差に不確かさの定量化なしで 0.03 のしきい値を置いた設計自体が
分解能の限界に近かった。レジームギャップ条項は通過した (§5.2)。
H7 は |MSE 差| 0.000091、paired SE 0.000216 で、しきい値の半分未満の SE のもとで統計的に確定した pass である。
one-step 予測では teacher と symbolic の差はほとんど消える: one-step の drift 寄与は dt = 0.01 と小さく、
実現値の diffusion ノイズが支配するためである。これは one-step 指標が field 誤差に対して鈍いことの再確認である。

## 6. 考察

1. **スコア drift は多項式蒸留の苦手な対象である**。真値が有理型の飽和関数である場合、
   次数 3 多項式の on-manifold 天井は 0.39 であり、teacher の精度 (0.094) とは無関係に H1 は失敗する。
   蒸留の成功を論じる際は、まず oracle 天井を測るべきである (本日はそれが H1 失敗の帰属を決めた)。
2. **分布指標と field 指標は別々に失敗する**。symbolic は W1 では僅差で pass (0.467 ≤ 0.50) したが
   モード質量では fail (差 0.230) した。W1 だけでは 2 峰性の谷間質量を見逃す。
   生成モデルの蒸留では W1 に加えてモード質量のような構造的分布指標が必須である。
3. **爆発するベースラインの分布指標は生存者条件付きである**。no-score の失敗率 0.8965 は、
   その W1・モード質量を「分離の証拠」として読むことを禁じる。失敗率を先に報告し、
   分布条項は失敗率でゲートする設計が必要である。
4. **teacher は良好、蒸留は部分的**。teacher の reverse サンプリングは W1 0.087・モード質量 0.9975 と
   ほぼ真値通りであり、DSM によるスコア学習そのものは 1 次元混合では十分に機能した。
   問題は蒸留側の表現力にあり、論文の枠組み自体の否定ではない。
5. **PF-ODE は診断として良好**。teacher スコアによる決定論的 PF-ODE は W1 0.181・モード質量 0.9895 であり、
   論文の SDE/ODE 等価性の 1 次元アナロジーとして整合的である (ただし裾は厚い)。

## 7. 反省 (翌日へ)

- 本日の反省は `reflections_log.md` の Day 10 行に集約した: (a) H1 失敗の帰属には oracle 天井を必ず併記すること、
  (b) 爆発するベースラインの分布条項は失敗率でゲートすること、(c) W1 pass とモード質量 fail が両立しうるため
  生成モデルの蒸留では構造的分布指標を W1 と並置すること、(d) W1 の差のしきい値は真値 SDE 自身の
  モンテカルロフロアと比べて分解能を事前に見積もること。
- Day 11 への申し送り: 本日の (a)–(d) を最低 1 つ、できれば (a) と (b) を適用すること。
  また Day 8–Day 10 で push 承認のタイムアウト / 未ログインによる push 停滞が続いているため、
  Day 10 の push・PR コメントの着地を別途確認すること (本実行の末尾を参照)。
