# 評価プロトコル — 3次・構造保存モデルの公正な比較のために（Issue #4）

2026-10-26 策定。本書は「何をすれば何を主張できるか」を事前に固定する文書であり、
既存結果の解釈を変えるものではない。実験の実行前に本書と config を凍結し、
結果を見た後に主張の等級（Tier）を格上げしてはならない。

## 1. 現状の監査（既存 artifact からの転記）

出典: `outputs/logs/pc5_distillation_split_audit.json`（50-PC 版
`outputs/logs/distillation_split_audit.json` も分割・exposure は同数）。

| 項目 | 記録 |
|---|---|
| symbolic split | clone-disjoint、seed 0。train 6,137 / validation 1,451 / test 1,762 cells |
| teacher | 5-PC、`LightningSDE_FixedPotential_RegularizedVelocityRatio`、checkpoint SHA256 は監査ログに記録 |
| teacher exposure（test 分割） | test 1,762 cells のうち teacher train に 1,591、teacher validation に 167、計 **1,758 (99.8%)** が teacher の fit に既に使われている |
| teacher exposure（validation 分割） | 1,451 cells のうち 1,285 + 156 = 1,441 (99.3%) |
| upstream PCA | 全細胞で fit されており、分割に対して独立ではない |
| teacher 自身の分割 | 公式の cell-level 分割には 292 clone の重複がある（README §3.2） |
| test の使用履歴 | 2次モデルについて評価済み。次数・制約の追加選択の根拠には使えない |
| query 点 | 観測状態 + teacher simulation 状態。後者は実データ量を増やすものではない |

結論として、現行の held-out は **symbolic fit に対してのみ held-out** であり、
「test で良くなった＝生物学的に汎化した」とは主張できない。これは既存 README の
自己監査と一致する。

## 2. Tier 定義と主張可能な範囲

### Tier A — 固定 teacher の探索的比較（現在可能な最大）

- 既存 teacher・PCA・split・query 点をすべて凍結し、蒸留法だけを変える
  （例: 2次 free / 3次 free / potential 保存型、`configs/experiment_pc5_degree3.json`）。
- モデル選択は training / validation のみ。既存 test は「explored」と明記した上で
  参考値としてのみ併記できる。
- 主張できること: 「この teacher 関数に対する近似として、次数・構造制約が
  局所誤差と simulation 安定性にどう影響するか」。
- 主張できないこと: 独立な汎化性能、生物学的妥当性、実データへの適合改善。

### Tier B — clone-disjoint な end-to-end 再設計

- 細胞を触る前に **clone 単位で分割を確定** する。その後の全段階を train clone
  の内側で完結させる: PCA の fit、teacher 学習（train/val は train clone 内で再分割）、
  蒸留の query 点生成と split、fate classifier の fit。
- evaluation clone はどの段階の fit にも使わない。teacher simulation を query 点に
  使う場合も、初期状態は train clone 由来に限る。
- 主張できること: 未見 clone に対する teacher + 蒸留パイプライン全体の汎化
  （ただし同一実験・同一データセット内）。

### Tier C — 独立データ

- 別実験・別個体・別バッチの LARRY 型データを確保し、一切触らずに最終評価に使う。
- 生物学的な汎化の主張は Tier C でのみ可能。

## 3. 全 Tier 共通の事前固定事項

1. **失敗質量を除外しない。** simulation は非有限 / ノルム `10R` 超過 /
   有限だが領域外 を分けて数え、fate 組成には「失敗」カテゴリを含める。
   survivor のみへの条件付けは診断としてのみ許し、その旨を明記する。
2. **dt・seed 感度を必須とする。** `dt ∈ {0.05, 0.025}`、Brownian seed は
   202/203/204 を最低限とし、失敗率が dt で大きく変わる場合は「数値的問題」、
   変わらない場合は「ベクトル場の問題」と切り分けて報告する（断定はしない）。
3. **関数近似と力学安定性を別判定にする。** validation NRMSE の基準通過と
   simulation 成功は独立の関門であり、一方の成功で他方を代替しない。
4. **構造指標を併記する。** potential 保存を論じる場合は PC 座標 Jacobian の
   非対称性 `||J − J^T||_F / ||J||_F` を `src/diagnostics.py` の定義で報告する
   （標準化座標での非対称性は scale の影響を含むため根拠に使わない）。
5. **選択の凍結。** 次数・閾値・ridge・制約の有無は validation で決めてから
   evaluation を 1 回だけ行う。evaluation 結果を見た再調整は、新しい Tier の
   実験として別途記録する。
6. **artifact の固定。** checkpoint の SHA256、config、seed、パッケージ
   バージョン、git commit を結果と同じ場所に保存する（`01_reproducibility.md`）。
7. **負の結果も結果である。** 改善しない・悪化する・変化なしのいずれも、
   条件と数値とともに保存する（`03_science_integrity.md`）。

## 4. 次に実行する際のチェックリスト

- [ ] Tier を宣言し、本書と config の commit を記録した
- [ ] teacher checkpoint の SHA256 が監査ログと一致する
- [ ] query 点の生成条件（観測 / teacher simulation の内訳）が前回と同一である
- [ ] 選択は validation までで凍結した
- [ ] §3 の報告項目をすべて含む結果表を作った
- [ ] 主張が Tier の範囲内に収まっているか、文書を再読した
