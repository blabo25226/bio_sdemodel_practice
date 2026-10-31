# Daily Neural SDE → Polynomial SINDy シリーズ

2026-10-26 開始のデイリー連載。**このシリーズ専用の固定ブランチ
`analysis/daily-neural-sde-sindy` に毎日 push し、単一の living Draft PR で集約する**
(main へはマージしない。マージはユーザーの別途指示がある場合のみ)。
既存の issues #1–#4 対応 PR (branch `muse/followup-issues-1-4`) とは独立のシリーズである。

## 毎日の手順 (Day N = 新しい論文 1 本)

1. **論文選定**: Neural SDE (ニューラル確率微分方程式) の研究論文を web 検索で 1 本選ぶ。
   選定条件: (a) データ生成過程または実データの入手先が特定できる、(b) 論文中の
   Neural SDE の構造 (drift / diffusion のパラメトライゼーション、学習法) を読み取れる、
   (c) この環境 (CPU のみ) で再現可能な規模に切り取れる。選定理由をレポートに残す。
   多様性のため、前日までの reflections を読み、手法系・次元・ドメインがなるべく
   重ならない論文を選ぶ。
2. **論文要約**: 問題設定、提案手法、実験設定、主張とその証拠範囲を要約する。
3. **データと Neural SDE の入手**: 論文の設定に従いデータを生成または取得し、
   論文が推定した Neural SDE を読み取る。再現範囲の切り取り (次元削減、latent の
   観測座標への置換、学習法の置換など) が必要な場合は**逸脱として明示**し、
   理由 (識別可能性、計算量など) を書く。切り取った範囲外の結果を論文の結果と
   混同しない。
4. **仮説**: 多項式 SINDy で teacher (Neural SDE) を近似する仮説を、
   **数値しきい値つきで事前に** config に凍結してから実行する。
5. **分析 (main と同じ流儀)**: teacher への query による function distillation。
   trajectory-disjoint な train / validation / test 分割、変換 (mean/scale) は
   train のみから、ハイパーパラメータ選択は validation のみ、test は確定後に 1 回。
   シミュレーション検証では失敗した trajectory の質量を保持し、失敗率と分類を報告する。
   負の結果も正式な結果として残す。
6. **レポート作成**: `reports/daily_neural_sde/dayNN_<slug>/report.md` に
   論文要約、仮説、方法、結果、考察、反省をまとめる。
   **求めた symbolic SDE は必ず TeX で記述する** (drift / diffusion の全成分、
   元座標の式)。数値の出所 (results.json 等) を対応づける。
7. **反省の蓄積**: `reflections_log.md` に 1 行以上追記し、翌日はそれを最低 1 つ適用する。
   この README の索引表にも 1 行追加する。
8. **PR への報告**: その日の要約をシリーズ PR にコメントし、チャット報告に
   コメント URL を含める。

## スコープ境界

- 1 日 1 論文。論文の全実験の完全再現は狙わない。再現範囲は各レポートで宣言する。
- symbolic distillation の成功はまず **teacher の近似**であって、真の SDE の発見や
  生物学的・因果的主張を意味しない (`.agents/rules/03_science_integrity.md` に従う)。
  真の SDE が既知の合成データの場合は「真値との比較」として分離して報告する。
- latent 座標の式は観測座標の機構ではない。latent が識別不能な論文では、
  識別可能な座標で teacher を再構成するか、その旨を明示して近似対象を限定する。
- 大きな checkpoint や raw data は tracked しない。取得元 URL / 生成手順 / seed /
  checksum を残す。小さい teacher 重み (JSON 化できる規模) は再現性のため保存してよい。
- test を見た後の事後チューニングをしない。

## 環境

- Python 3.12 / NumPy / PyTorch (CPU)。詳細は `requirements.txt` (このディレクトリ)。
- 実行は repo ルートから `python -m src.daily_neural_sde.dayNN_<slug>` 形式を基本とする。
- seed は全て config に明示する。

## 索引

| Day | Date | Paper | System | Report |
|---|---|---|---|---|
| 1 | 2026-10-26 | Li et al. 2020, Scalable Gradients for SDEs (AISTATS) | stochastic Lorenz attractor (3D) | `day01_li2020_lorenz/report.md` |
| 2 | 2026-10-27 | Kidger et al. 2021, Neural SDEs as Infinite-Dimensional GANs (ICML) | time-dependent Ornstein–Uhlenbeck process (1D, time-dependent) | `day02_kidger2021_ou/report.md` |
| 3 | 2026-10-28 | Tzen & Raginsky 2019, Neural SDEs: Deep Latent Gaussian Models in the Diffusion Limit (arXiv:1905.09883) | sigmoid-drift synthetic SDE (2D, time-homogeneous) | `day03_tzen2019_sigmoid/report.md` |
| 4 | 2026-10-29 | Gierjatowicz et al. 2020, Robust pricing and hedging via neural SDEs (arXiv:2007.04154) | Heston-type local stochastic volatility SDE (2D, log-price + variance, correlated Brownian, state-dependent diffusion) | `day04_gierjatowicz_lsv/report.md` |
| 5 | 2026-10-30 | Zhang et al. 2025, END-nSDE (PLOS Comput Biol) | circadian clock (2D, linear drift + constant diffusion, W2-trained teacher) | `day05_zhang_circadian/report.md` |
| 6 | 2026-10-31 | Eggen & Midtfjord 2023, Delay-SDE-Net (arXiv:2303.08587) | stochastic delay equation (2D + 4 lags, tanh drift, near-constant diffusion) | `day06_eggen_delay/report.md` |
