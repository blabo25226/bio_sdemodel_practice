# Follow-up 報告 — Issues #1–#4 への対応

2026-10-26。対象は凍結済みの 5-PC symbolic SDE（`outputs/sindy_pc5/`）と
既存の監査ログのみ。**teacher checkpoint（`outputs/baseline_pc5/`）と蒸留用
query データ（`outputs/distillation_pc5/`）は git 管理外の外部 artifact で
この repository には存在しない**ため、実データでの再学習・再蒸留は実行
できなかった。本報告は (a) 凍結モデルだけで完結する定量診断、(b) 再実行を
可能にする実装と設計の凍結、を分けて記録する。できなかったことを
「やった」とは書かない。

| Issue | 状態 | 内容 |
|---|---|---|
| #1 3次多項式への拡張 | 設計凍結・未実行 | 既存 `fit_field` は degree 一般であり、比較条件を `configs/experiment_pc5_degree3.json` に凍結。実 run は外部 artifact の復元が前提 |
| #2 potential 保存型 drift | 実装・合成検証済み / 実 teacher 未適用 | `src/potential_distill.py`。合成勾配場で support を完全回復し、Jacobian 対称性が構成上厳密に 0 |
| #3 領域外挙動・安定性診断 | **実行済み** | `src/diagnostics.py`、結果は §1。凍結モデルのみで完結 |
| #4 公正な評価プロトコル | 策定済み | `docs/evaluation_protocol.md`。現状監査の転記と Tier A/B/C の主張範囲を固定 |

## 1. 診断結果（Issue #3、実測）

実行: `python -m src.diagnostics`（seed 0、NumPy のみ、GPU・checkpoint 不要）。
出力: `outputs/diagnostics_pc5/{shell_statistics.csv, jacobian_asymmetry.json,
simulation_probes.csv, summary.json}`。領域半径は既存定義どおり
`R = 25.4603`（PC ノルム、training 99.5% 分位点）、失敗閾値は `10R`。

### 1.1 Jacobian 非対称性（Issue #2 の前提測定を含む）

領域内（0.25R–1.0R の球面サンプル 8,192 点）で、自由な成分別 SINDy drift の
Jacobian 非対称性 `||J − J^T||_F / ||J||_F` を測った。teacher の drift は
PC 座標での勾配 `∇_x μ` なので、構造を保存した蒸留では **PC 座標** の
Jacobian が対称になるはずである（標準化座標 `z` では軸ごとの scale のため
完全な蒸留でも対称にならない。混同を避けるため両方を記録した）。

| 座標 | 中央値 | 95%点 | 最大 | 0.05 未満の点の割合 |
|---|---:|---:|---:|---:|
| PC 座標 `J_x` | **0.897** | 1.165 | 1.478 | **0 / 8,192** |
| 標準化座標 `J_z`（参考） | 0.996 | 1.215 | 1.481 | 0 / 8,192 |

中央値 0.897 は、Jacobian の歪対称成分が全 Frobenius ノルムの約 45%
（エネルギー比では約 20%）を占めることに相当する。領域内のどの点でも
勾配場に近い場所はなく、**現行 drift は teacher の potential 構造を
定量的に大きく失っている**。これは README §4.2 の 1 点での指摘を
領域全体の分布として確認したものである。

### 1.2 球面上での成長（各球面 4,096 方向、平均値）

| 球面 | ‖f‖ 平均 | 径方向外向きの方向割合 | ‖G‖²=tr D 平均 | `L‖x‖²` 平均 | `L‖x‖² > 0` の割合 |
|---|---:|---:|---:|---:|---:|
| 0.25R | 8.09 | 0.446 | 0.098 | −11.9 | 0.447 |
| 0.5R | 14.6 | 0.508 | 0.337 | −43.5 | 0.509 |
| 1.0R | 41.5 | 0.516 | 2.48 | −183 | 0.517 |
| 2.0R | 150 | 0.520 | 28.6 | −653 | 0.520 |
| 3.0R | 331 | 0.514 | 138 | −1,733 | 0.514 |
| 5.0R | 906 | 0.494 | 1,020 | −5,547 | 0.496 |
| 10R | 3,555 | 0.496 | 16,108 | −3,044 | 0.500 |

読み方（断定を避ける）:

- ‖f‖ は 2 次多項式らしく半径の 2 乗規模で成長し、1R でも平均 41.5 PC/day
  と半径そのものより大きい。領域境界付近では 1 日未満で領域を横切る速度場である。
- 平均の径方向成分と `L‖x‖²` は負だが、**方向別では全ての球面で約半数が
  外向き（`L‖x‖² > 0`）** であり、内向きが支配的になる半径は 10R まで存在しない。
  つまり「平均的には散逸的だが一様には散逸的でない」異方的な場であり、
  Lyapunov 型の一様条件はこの形では満たされない。
- `tr D = ‖G‖²` は G が 2 次式のため半径の 4 乗規模で成長し、10R では
  平均 16,108 PC²/day に達する。拡散の Itô 補正は遠方では drift の
  径方向項と同規模以上になり、**drift の内向き性だけでは SDE 全体の
  安定性は議論できない**（Issue #3 の指摘どおり）。

なお球面サンプルは一様方向であり、実データ分布ではない。低密度方向を
含むため、本節は「fit された場の性質」の診断であって、観測細胞の
挙動の記述ではない。

### 1.3 プローブ simulation（合成初期点、検証プロトコルの複製ではない）

球面上の合成初期点 128 点から symbolic SDE のみを Euler–Maruyama で
4 日間積分した（失敗した軌道は質量を保持し、分類はラッチする）。

| 初期球面 | dt | 失敗（非有限 or >10R） | 有限・領域外 | 有限・領域内 |
|---|---|---:|---:|---:|
| 0.5R | 0.05 | 0.570–0.578 | 0.375–0.391 | 0.031–0.055 |
| 0.5R | 0.025 | 0.578–0.586 | 0.367–0.391 | 0.023–0.055 |
| 1.0R | 0.05 | 0.797–0.813 | 0.164–0.188 | 0.016–0.031 |
| 1.0R | 0.025 | 0.813–0.828 | 0.141–0.172 | 0.016–0.031 |

（範囲は Brownian seed 202/203/204 の最小–最大。）

- 刻みを半分にしても失敗率は下がらない（全条件平均 0.690 → 0.701）。
  少なくともこの範囲では、**失敗は刻み幅の数値誤差では説明できない**。
  ただし Euler–Maruyama の範囲内の話であり、連続時間 SDE の数学的爆発の
  証明ではない。
- 0.5R からでも大半が領域を離れる。これは §1.2 の異方性（どの半径でも
  約半方向が外向き）と整合する。初期分布が一様球面であるため、検証結果
  （58.6%）と数値を直接比較はしない。

## 2. potential 保存型蒸留（Issue #2、実装）

`src/potential_distill.py`:

- potential `μ̃(z)` を多項式で表し、`f_i = (1/scale_i) ∂μ̃/∂z_i` として
  drift を導く。全成分が 1 本の係数ベクトルを共有する結合回帰であり、
  PC 座標 Jacobian は構成上厳密に対称（§1.1 の指標が 0 になる）。
- 次数対応: drift 次数 `p` には potential 次数 `p+1` が必要。5 次元では
  drift 2 次 → potential 55 項、drift 3 次 → potential 125 項（定数項は
  同定不能のため除外）。
- 推定は既存 `src/distill.py` の慣例に合わせ、RMS 正規化した
  STLSQ 型の反復閾値 + ridge（ridge は最終 refit でも保持）。

合成検証（`python -m src.potential_synthetic_demo`、seed 0）:
既知の疎な 3 次 potential（active 10 項、5 次元）から作った厳密な勾配場に
対し、train 6,000 点から fit した結果、**support を 10/10 完全回復、
test NRMSE 0.0087、Jacobian 非対称性は厳密に 0**。これは手法とコードの
検証であり、LARRY teacher についての証拠ではない。実 teacher への適用は
query データの復元後に Tier A 比較（`docs/evaluation_protocol.md`）として
行う。

## 3. 3次拡張（Issue #1、設計凍結）

- 既存 `fit_field` は degree 一般であり、基底数は 5 次元で 21 → 56 になる
  ことをテストで固定した（`tests/test_followup.py`）。新規コードは不要だった。
- 比較の凍結条件（teacher・split・query・選択規則を 2 次と同一にし、
  potential arm と診断の併記を必須化）を
  `configs/experiment_pc5_degree3.json` に記録した。状態は `not_run` であり、
  理由（外部 artifact 不在）も config 内に明記した。

## 4. 評価プロトコル（Issue #4）

`docs/evaluation_protocol.md` に、既存監査の転記（test 1,762 cells の
99.8% が teacher の fit に使用済み、PCA は全細胞 fit、等）と、主張範囲を
Tier A（固定 teacher の探索的比較）/ Tier B（clone-disjoint end-to-end）/
Tier C（独立データ）に分けたプロトコルを策定した。今後のすべての比較は
実行前に Tier を宣言し、結果後に格上げしない。

## 5. テストと再現

- `tests/test_followup.py`（新規 7 件、NumPy のみ）: 基底数と順序、
  解析 Jacobian と有限差分の一致、凍結 drift の非勾配性の回帰ガード、
  potential fit の support 回復と対称性、JSON 往復。**全件 pass**
  （Python 3.11 系 / NumPy 1.26 で確認）。
- 既存テスト（`tests/test_distill.py` 等）は PySINDy / PyTorch を含む
  repo の `environment.yml` 環境でのみ実行可能であり、本作業環境には
  その環境がないため未実行。既存ファイルは変更していない。
- 診断・合成デモはいずれも seed を固定し、再実行で同一の数値になることを
  確認した。

## 6. 残る作業（外部 artifact が必要なもの）

1. checkpoint と `outputs/distillation_pc5/` を復元し、Tier A として
   3次 free / potential arm を実行する（config 凍結済み）。
2. その際、potential 法については teacher の drift だけでなく potential
   `μ` 自体の直接評価も記録し、勾配経由との整合を runtime で確認する。
3. 生物学的汎化を主張する場合は Tier B の再設計（PCA から clone-disjoint）
   が必要であり、これは新規実験として別途起票する。
